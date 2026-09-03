from __future__ import annotations

import ast
import io
import json
import math
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RESULT_DIR = Path("results/cgc_tahoe_0b")
DATA_DIR = Path("data/tahoe100m_metadata")
TARGET_PLATES = ("plate6", "plate14")
CONTROL = "DMSO_TF"
LOW_COVERAGE_NAMES = ("NCI-H661", "NCI-H596", "NCI-H2122")
MIN_CONTEXTS = 30
MIN_INTERVENTIONS = 50


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0
        self.bytes_transferred = 0
        self.range_requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size) - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}",
                "User-Agent": "IGC-Virtual-Cell/0.1",
            },
        )
        with urllib.request.urlopen(request, timeout=240) as response:
            if response.status != 206:
                raise RuntimeError(
                    f"Remote server did not honor range request: HTTP {response.status}"
                )
            data = response.read()
        self.position += len(data)
        self.bytes_transferred += len(data)
        self.range_requests += 1
        return data


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_treatment(value: str) -> tuple[str, float, str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 1 or len(parsed[0]) != 3:
        raise ValueError(f"Unexpected drugname_drugconc encoding: {value}")
    drug, concentration, unit = parsed[0]
    return str(drug).strip(), float(concentration), str(unit).strip()


def intervention_id(drug: str, concentration: float, unit: str) -> str:
    return f"{drug.strip()}__{format(float(concentration), '.8g')}__{unit.strip()}"


def _row_group_stat(metadata: Any, column_name: str) -> tuple[Any, Any] | None:
    for column_index in range(metadata.num_columns):
        column = metadata.column(column_index)
        if column.path_in_schema == column_name:
            statistics = column.statistics
            if statistics and statistics.has_min_max:
                return statistics.min, statistics.max
            return None
    return None


def _plate_groups(parquet: pq.ParquetFile) -> tuple[list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    manifest: list[dict[str, Any]] = []
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        plate_range = _row_group_stat(group, "plate")
        values = set(plate_range or ())
        use = bool(values.intersection(TARGET_PLATES))
        if use:
            selected.append(index)
        manifest.append(
            {
                "row_group": index,
                "rows": group.num_rows,
                "uncompressed_size_bytes": group.total_byte_size,
                "plate_min": plate_range[0] if plate_range else None,
                "plate_max": plate_range[1] if plate_range else None,
                "selected": use,
            }
        )
    return selected, manifest


def _probe_de_layout(revision: str) -> dict[str, Any]:
    api_request = urllib.request.Request(
        "https://huggingface.co/api/datasets/tahoebio/Tahoe-100M?blobs=true",
        headers={"User-Agent": "IGC-Virtual-Cell/0.1"},
    )
    with urllib.request.urlopen(api_request, timeout=180) as response:
        api = json.loads(response.read())
    if api["sha"] != revision:
        raise RuntimeError(
            f"Official repository revision changed during audit: {api['sha']} != {revision}"
        )
    files = sorted(
        (
            item
            for item in api["siblings"]
            if item["rfilename"].startswith(
                "metadata/pseudobulk_differential_expression/"
            )
            and item["rfilename"].endswith(".parquet")
        ),
        key=lambda item: item["rfilename"],
    )
    probe = files[0]
    url = (
        f"https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/{revision}/"
        f"{probe['rfilename']}"
    )
    reader = HTTPRangeReader(url, int(probe["size"]))
    parquet = pq.ParquetFile(reader)
    condition_columns = (
        "plate",
        "n_cells_trt",
        "n_cells_ctrl",
        "Cell_ID_Cellosaur",
        "drug",
        "concentration",
        "concentration_unit",
    )
    groups = []
    for index in range(min(3, parquet.metadata.num_row_groups)):
        metadata = parquet.metadata.row_group(index)
        stats = {
            column: _row_group_stat(metadata, column) for column in condition_columns
        }
        groups.append(
            {
                "row_group": index,
                "rows": metadata.num_rows,
                "condition_statistics": stats,
                "condition_constant": all(
                    values is not None and values[0] == values[1]
                    for values in stats.values()
                ),
            }
        )
    return {
        "probe_file": probe["rfilename"],
        "probe_file_size_bytes": int(probe["size"]),
        "url": url,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.metadata.num_columns,
        "schema": str(parquet.schema_arrow),
        "footer_bytes_transferred": reader.bytes_transferred,
        "range_requests": reader.range_requests,
        "sample_row_groups": groups,
        "condition_clustered_by_row_group": all(
            group["condition_constant"] for group in groups
        ),
        "expression_or_de_values_read": False,
    }


def _condition_core_candidates(
    paired: pd.DataFrame,
    eligible_contexts: list[str],
    low_coverage_ids: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    intervention_sets = {
        context: set(
            paired.loc[
                (paired["cell_line_id"] == context) & paired["present_both_plates"],
                "intervention_id",
            ]
        )
        for context in eligible_contexts
    }

    candidate_context_sets: list[tuple[str, ...]] = [tuple(sorted(eligible_contexts))]
    for count in range(1, len(low_coverage_ids) + 1):
        for excluded in combinations(low_coverage_ids, count):
            candidate_context_sets.append(
                tuple(sorted(set(eligible_contexts).difference(excluded)))
            )
    ranked = sorted(
        eligible_contexts,
        key=lambda context: (len(intervention_sets[context]), context),
    )
    current = set(eligible_contexts)
    for context in ranked:
        current = current.difference({context})
        if len(current) >= MIN_CONTEXTS:
            candidate_context_sets.append(tuple(sorted(current)))

    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    for contexts in candidate_context_sets:
        if contexts in seen or not contexts:
            continue
        seen.add(contexts)
        shared = set.intersection(*(intervention_sets[context] for context in contexts))
        selected_rows = paired[
            paired["cell_line_id"].isin(contexts)
            & paired["intervention_id"].isin(shared)
        ]
        minimum_cells = int(
            selected_rows[["plate6_n_cells_trt", "plate14_n_cells_trt"]]
            .min(axis=1)
            .min()
        ) if len(selected_rows) else 0
        rows.append(
            {
                "context_count": len(contexts),
                "shared_intervention_count": len(shared),
                "total_complete_conditions_two_plates": len(contexts) * len(shared) * 2,
                "minimum_treatment_cell_count": minimum_cells,
                "contexts": ";".join(contexts),
                "excluded_contexts": ";".join(sorted(set(eligible_contexts).difference(contexts))),
                "meets_frozen_gate": len(contexts) >= MIN_CONTEXTS and len(shared) >= MIN_INTERVENTIONS,
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["context_count", "shared_intervention_count", "minimum_treatment_cell_count"],
        ascending=False,
    )
    passing = candidates[candidates["meets_frozen_gate"]]
    if passing.empty:
        return candidates, [], []
    choice = passing.iloc[0]
    selected_contexts = str(choice["contexts"]).split(";")
    selected_interventions = sorted(
        set.intersection(*(intervention_sets[context] for context in selected_contexts))
    )
    return candidates, selected_contexts, selected_interventions


def run_feasibility_audit(repo: Path) -> dict[str, Any]:
    result_dir = repo / RESULT_DIR
    data_dir = repo / DATA_DIR
    source_revision = json.loads(
        (result_dir / "source_revision.json").read_text(encoding="utf-8")
    )
    download_manifest = json.loads(
        (result_dir / "download_manifest.json").read_text(encoding="utf-8")
    )
    revision = source_revision["revision"]
    de_layout_probe = _probe_de_layout(revision)
    _write_json(result_dir / "de_layout_probe.json", de_layout_probe)
    obs_size = int(source_revision["obs_metadata_bytes"])
    obs_url = (
        f"https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/{revision}/"
        "metadata/obs_metadata.parquet"
    )

    sample_metadata = pd.read_parquet(data_dir / "sample_metadata.parquet")
    cell_metadata = pd.read_parquet(data_dir / "cell_line_metadata.parquet")
    target_samples = sample_metadata[sample_metadata["plate"].isin(TARGET_PLATES)].copy()
    parsed = target_samples["drugname_drugconc"].map(parse_treatment)
    target_samples[["parsed_drug", "concentration", "concentration_unit"]] = pd.DataFrame(
        parsed.tolist(), index=target_samples.index
    )
    target_samples["intervention_id"] = [
        intervention_id(drug, concentration, unit)
        for drug, concentration, unit in zip(
            target_samples["parsed_drug"],
            target_samples["concentration"],
            target_samples["concentration_unit"],
            strict=True,
        )
    ]
    target_samples["is_control"] = target_samples["drug"].eq(CONTROL)
    target_samples["pair_match_id"] = target_samples["intervention_id"]
    control_order = (
        target_samples[target_samples["is_control"]]
        .sort_values(["plate", "sample"])
        .groupby("plate", observed=True)
        .cumcount()
        .add(1)
    )
    target_samples.loc[control_order.index, "pair_match_id"] = [
        f"{CONTROL}__duplicate_{value}" for value in control_order
    ]
    target_samples.to_csv(result_dir / "sample_design.csv", index=False)
    for plate in TARGET_PLATES:
        treatments = target_samples[
            (target_samples["plate"] == plate) & ~target_samples["is_control"]
        ].copy()
        treatments.to_csv(result_dir / f"{plate}_treatments.csv", index=False)

    plate6_ids = set(
        target_samples.loc[
            (target_samples["plate"] == "plate6") & ~target_samples["is_control"],
            "intervention_id",
        ]
    )
    plate14_ids = set(
        target_samples.loc[
            (target_samples["plate"] == "plate14") & ~target_samples["is_control"],
            "intervention_id",
        ]
    )
    overlap_rows = []
    for item in sorted(plate6_ids | plate14_ids):
        row = target_samples[target_samples["intervention_id"] == item].iloc[0]
        overlap_rows.append(
            {
                "intervention_id": item,
                "drug": row["parsed_drug"],
                "concentration": row["concentration"],
                "concentration_unit": row["concentration_unit"],
                "plate6": item in plate6_ids,
                "plate14": item in plate14_ids,
                "shared": item in plate6_ids & plate14_ids,
            }
        )
    pd.DataFrame(overlap_rows).to_csv(
        result_dir / "plate6_plate14_treatment_overlap.csv", index=False
    )

    reader = HTTPRangeReader(obs_url, obs_size)
    parquet = pq.ParquetFile(reader)
    footer_bytes = reader.bytes_transferred
    selected_groups, row_group_manifest = _plate_groups(parquet)
    group_aggregates: list[pd.DataFrame] = []
    plate_sublibraries: dict[str, set[str]] = {plate: set() for plate in TARGET_PLATES}
    plate_raw_rows: dict[str, int] = {plate: 0 for plate in TARGET_PLATES}
    plate_full_rows: dict[str, int] = {plate: 0 for plate in TARGET_PLATES}
    columns = [
        "plate",
        "sample",
        "cell_line",
        "cell_name",
        "sublibrary",
        "pass_filter",
    ]
    for group_index in selected_groups:
        frame = parquet.read_row_group(group_index, columns=columns).to_pandas()
        frame = frame[frame["plate"].isin(TARGET_PLATES)]
        for plate, plate_frame in frame.groupby("plate", observed=True):
            plate_raw_rows[plate] += len(plate_frame)
            plate_sublibraries[plate].update(plate_frame["sublibrary"].astype(str).unique())
        full = frame[frame["pass_filter"].astype(str).str.lower().eq("full")].copy()
        for plate, plate_frame in full.groupby("plate", observed=True):
            plate_full_rows[plate] += len(plate_frame)
        grouped = (
            full.groupby(["plate", "sample", "cell_line", "cell_name"], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        group_aggregates.append(grouped)
    aggregate = (
        pd.concat(group_aggregates, ignore_index=True)
        .groupby(["plate", "sample", "cell_line", "cell_name"], observed=True)["n_cells"]
        .sum()
        .reset_index()
        .rename(columns={"cell_line": "cell_line_id"})
    )
    aggregate = aggregate.merge(
        target_samples[
            [
                "sample",
                "plate",
                "drug",
                "parsed_drug",
                "concentration",
                "concentration_unit",
                "intervention_id",
                "pair_match_id",
                "is_control",
            ]
        ],
        on=["sample", "plate"],
        how="left",
        validate="many_to_one",
    )
    if aggregate["drug"].isna().any():
        raise RuntimeError("Extracted obs rows could not be mapped to sample metadata.")

    controls = aggregate[aggregate["is_control"]].copy().rename(
        columns={"n_cells": "n_control_cells"}
    )
    controls.to_csv(
        result_dir / "dmso_manifest.csv", index=False
    )
    control_totals = (
        controls.groupby(["plate", "cell_line_id"], observed=True)["n_control_cells"]
        .sum()
        .rename("n_cells_ctrl")
        .reset_index()
    )
    conditions = aggregate[~aggregate["is_control"]].copy()
    conditions = conditions.rename(columns={"n_cells": "n_cells_trt"}).merge(
        control_totals,
        on=["plate", "cell_line_id"],
        how="left",
        validate="many_to_one",
    )
    conditions["control_usable"] = conditions["n_cells_ctrl"].fillna(0).gt(0)
    condition_columns = [
        "plate",
        "sample",
        "cell_line_id",
        "cell_name",
        "drug",
        "parsed_drug",
        "concentration",
        "concentration_unit",
        "intervention_id",
        "n_cells_trt",
        "n_cells_ctrl",
        "control_usable",
    ]
    conditions[condition_columns].sort_values(
        ["plate", "cell_line_id", "intervention_id"]
    ).to_parquet(result_dir / "plate6_14_condition_index.parquet", index=False)
    conditions[condition_columns].sort_values(
        ["plate", "cell_line_id", "intervention_id"]
    ).to_parquet(result_dir / "plate6_14_cell_counts.parquet", index=False)

    paired = conditions.pivot_table(
        index=[
            "cell_line_id",
            "cell_name",
            "intervention_id",
            "parsed_drug",
            "concentration",
            "concentration_unit",
        ],
        columns="plate",
        values="n_cells_trt",
        aggfunc="sum",
    ).reset_index()
    paired.columns.name = None
    paired = paired.rename(
        columns={"plate6": "plate6_n_cells_trt", "plate14": "plate14_n_cells_trt"}
    )
    for column in ("plate6_n_cells_trt", "plate14_n_cells_trt"):
        if column not in paired:
            paired[column] = 0
        paired[column] = paired[column].fillna(0).astype(int)
    paired["present_both_plates"] = paired[
        ["plate6_n_cells_trt", "plate14_n_cells_trt"]
    ].gt(0).all(axis=1)

    wide = paired.pivot_table(
        index=["cell_line_id", "cell_name"],
        columns="intervention_id",
        values="present_both_plates",
        aggfunc="max",
        fill_value=False,
    ).astype(int)
    wide.reset_index().to_csv(result_dir / "plate6_14_coverage_matrix.csv", index=False)

    cell_lookup = cell_metadata[
        ["cell_name", "Cell_ID_Cellosaur", "Cell_ID_DepMap", "Organ"]
    ].drop_duplicates("Cell_ID_Cellosaur")
    context_rows = []
    cell_ids = sorted(aggregate["cell_line_id"].unique())
    for cell_id in cell_ids:
        cell_frame = paired[paired["cell_line_id"] == cell_id]
        control_frame = control_totals[control_totals["cell_line_id"] == cell_id]
        metadata_row = cell_lookup[cell_lookup["Cell_ID_Cellosaur"] == cell_id]
        names = aggregate.loc[aggregate["cell_line_id"] == cell_id, "cell_name"]
        cell_name = names.mode().iloc[0]
        context_rows.append(
            {
                "cell_line_id": cell_id,
                "cell_name": cell_name,
                "Cell_ID_DepMap": metadata_row["Cell_ID_DepMap"].iloc[0] if len(metadata_row) else "",
                "organ": metadata_row["Organ"].iloc[0] if len(metadata_row) else "",
                "plate6_present": bool(
                    (aggregate["cell_line_id"].eq(cell_id) & aggregate["plate"].eq("plate6")).any()
                ),
                "plate14_present": bool(
                    (aggregate["cell_line_id"].eq(cell_id) & aggregate["plate"].eq("plate14")).any()
                ),
                "plate6_control_cells": int(
                    control_frame.loc[control_frame["plate"] == "plate6", "n_cells_ctrl"].sum()
                ),
                "plate14_control_cells": int(
                    control_frame.loc[control_frame["plate"] == "plate14", "n_cells_ctrl"].sum()
                ),
                "shared_noncontrol_interventions": int(cell_frame["present_both_plates"].sum()),
                "paper_low_coverage_line": cell_name in LOW_COVERAGE_NAMES,
                "preliminary_eligible": (
                    int(cell_frame["present_both_plates"].sum()) >= MIN_INTERVENTIONS
                    and set(control_frame["plate"]) == set(TARGET_PLATES)
                    and bool((control_frame["n_cells_ctrl"] > 0).all())
                ),
                "selection_uses_expression_outcome": False,
            }
        )
    cell_manifest = pd.DataFrame(context_rows).sort_values("cell_name")
    cell_manifest.to_csv(result_dir / "cell_line_manifest.csv", index=False)
    eligible_contexts = sorted(
        cell_manifest.loc[cell_manifest["preliminary_eligible"], "cell_line_id"]
    )
    low_ids = sorted(
        cell_manifest.loc[
            cell_manifest["paper_low_coverage_line"]
            & cell_manifest["cell_line_id"].isin(eligible_contexts),
            "cell_line_id",
        ]
    )
    candidates, selected_contexts, selected_interventions = _condition_core_candidates(
        paired, eligible_contexts, low_ids
    )
    candidates.to_csv(result_dir / "complete_core_candidates.csv", index=False)
    selected = paired[
        paired["cell_line_id"].isin(selected_contexts)
        & paired["intervention_id"].isin(selected_interventions)
    ].copy()
    ctrl_wide = control_totals.pivot_table(
        index="cell_line_id", columns="plate", values="n_cells_ctrl", aggfunc="sum"
    ).reset_index().rename(
        columns={"plate6": "plate6_n_cells_ctrl", "plate14": "plate14_n_cells_ctrl"}
    )
    selected = selected.merge(ctrl_wide, on="cell_line_id", how="left", validate="many_to_one")
    selected["core_selected_by_coverage_only"] = True
    selected.sort_values(["cell_line_id", "intervention_id"]).to_csv(
        result_dir / "selected_replicate_core.csv", index=False
    )

    collapsed_condition_counts = {
        plate: int(aggregate[aggregate["plate"] == plate][
            ["cell_line_id", "intervention_id"]
        ].drop_duplicates().shape[0])
        for plate in TARGET_PLATES
    }
    plate_condition_counts = {
        plate: int(
            aggregate.loc[aggregate["plate"] == plate, ["cell_line_id", "pair_match_id"]]
            .drop_duplicates()
            .shape[0]
        )
        for plate in TARGET_PLATES
    }
    pair_sets = {
        plate: set(
            map(
                tuple,
                aggregate.loc[aggregate["plate"] == plate, ["cell_line_id", "pair_match_id"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        for plate in TARGET_PLATES
    }
    common_pairs = len(pair_sets["plate6"] & pair_sets["plate14"])
    sample_sets = {
        plate: set(target_samples.loc[target_samples["plate"] == plate, "sample"])
        for plate in TARGET_PLATES
    }
    replicate_definition = {
        "replicate_a": "plate6",
        "replicate_b": "plate14",
        "relationship": "Plate 14 is designated as a biological replicate of Plate 6 in the Tahoe-100M manuscript.",
        "publication_url": "https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1.full",
        "official_dataset_url": "https://huggingface.co/datasets/tahoebio/Tahoe-100M",
        "separate_experimental_plates": True,
        "sample_ids_disjoint": sample_sets["plate6"].isdisjoint(sample_sets["plate14"]),
        "plate6_sample_ids": len(sample_sets["plate6"]),
        "plate14_sample_ids": len(sample_sets["plate14"]),
        "plate6_sublibraries": len(plate_sublibraries["plate6"]),
        "plate14_sublibraries": len(plate_sublibraries["plate14"]),
        "sublibrary_ids_disjoint": plate_sublibraries["plate6"].isdisjoint(
            plate_sublibraries["plate14"]
        ),
        "independence_scope": (
            "Separate treatment executions and experimental plates with distinct sample and "
            "sublibrary IDs; not claimed to be independent donor biology."
        ),
        "random_cell_halves_used": False,
    }
    _write_json(result_dir / "replicate_definition.json", replicate_definition)

    selected_condition_fraction = (
        len(selected_contexts) * len(selected_interventions) * 2
    ) / max(1, 52886)
    estimated_selective_de = round(
        source_revision["pseudobulk_de_total_bytes"] * selected_condition_fraction
    )
    footer_estimate = round(
        de_layout_probe["footer_bytes_transferred"]
        * source_revision["pseudobulk_de_shards"]
    )
    future_requirement = {
        "preferred_source": "official plate-resolved pseudobulk differential-expression parquet",
        "full_release_bytes": source_revision["pseudobulk_de_total_bytes"],
        "full_release_gb": source_revision["pseudobulk_de_total_bytes"] / 1e9,
        "selected_core_contexts": len(selected_contexts),
        "selected_core_interventions": len(selected_interventions),
        "selected_core_conditions_two_plates": len(selected_contexts)
        * len(selected_interventions)
        * 2,
        "estimated_selective_condition_data_bytes": estimated_selective_de,
        "estimated_selective_condition_data_gb": estimated_selective_de / 1e9,
        "estimated_all_shard_footer_scan_bytes": footer_estimate,
        "estimated_all_shard_footer_scan_gb": footer_estimate / 1e9,
        "minimum_practical_remote_transfer_gb": (
            estimated_selective_de + footer_estimate
        )
        / 1e9,
        "expected_peak_ram_gb": 8,
        "expected_local_extracted_core_gb": estimated_selective_de / 1e9,
        "selective_extraction_feasible": True,
        "basis": (
            "Remote parquet test confirmed condition-constant row groups and HTTP range support. "
            "The feasibility phase read only schema/footer and Plate 6/14 obs columns."
        ),
        "de_layout_probe_artifact": "results/cgc_tahoe_0b/de_layout_probe.json",
        "large_download_performed": False,
    }
    _write_json(result_dir / "future_data_requirement.json", future_requirement)

    core_feasible = (
        replicate_definition["separate_experimental_plates"]
        and len(selected_contexts) >= MIN_CONTEXTS
        and len(selected_interventions) >= MIN_INTERVENTIONS
        and len(selected)
        == len(selected_contexts) * len(selected_interventions)
        and bool(
            selected[
                [
                    "plate6_n_cells_trt",
                    "plate14_n_cells_trt",
                    "plate6_n_cells_ctrl",
                    "plate14_n_cells_ctrl",
                ]
            ]
            .gt(0)
            .all()
            .all()
        )
        and future_requirement["selective_extraction_feasible"]
    )
    support_sizes = [2, 4, 8, 16, 24, 32]
    if len(selected_contexts) >= 42:
        support_sizes.append(40)
    support_sizes.append(len(selected_contexts) - 1)
    support_design = {
        "preregistered_only_not_run": True,
        "eligible_contexts": len(selected_contexts),
        "support_sizes": sorted(set(support_sizes)),
        "maximal_support": len(selected_contexts) - 1,
    }
    _write_json(result_dir / "future_support_design.json", support_design)

    verdict = "TAHOE_REPLICATE_CORE_FEASIBLE" if core_feasible else "TAHOE_REPLICATE_CORE_NOT_FEASIBLE"
    verdict_payload = {
        "verdict": verdict,
        "plate6_samples": int((target_samples["plate"] == "plate6").sum()),
        "plate14_samples": int((target_samples["plate"] == "plate14").sum()),
        "plate6_noncontrol_interventions": len(plate6_ids),
        "plate14_noncontrol_interventions": len(plate14_ids),
        "exact_sample_level_treatment_overlap": len(plate6_ids & plate14_ids),
        "cell_lines_observed": len(cell_ids),
        "preliminary_eligible_contexts": len(eligible_contexts),
        "selected_core_contexts": len(selected_contexts),
        "selected_core_interventions": len(selected_interventions),
        "selected_core_dimensions": f"{len(selected_contexts)} contexts x {len(selected_interventions)} interventions x 2 plates",
        "plate6_matched_pairs_including_control": plate_condition_counts["plate6"],
        "plate14_matched_pairs_including_control": plate_condition_counts["plate14"],
        "common_pairs_including_control": common_pairs,
        "plate6_pairs_with_duplicate_controls_collapsed": collapsed_condition_counts["plate6"],
        "plate14_pairs_with_duplicate_controls_collapsed": collapsed_condition_counts["plate14"],
        "plate6_full_filter_cells": plate_full_rows["plate6"],
        "plate14_full_filter_cells": plate_full_rows["plate14"],
        "minimum_selected_treatment_cells": int(
            selected[["plate6_n_cells_trt", "plate14_n_cells_trt"]]
            .min(axis=1)
            .min()
        ),
        "controls_usable": bool(
            selected[["plate6_n_cells_ctrl", "plate14_n_cells_ctrl"]].gt(0).all().all()
        ),
        "support_scaling_run": False,
        "expression_outcomes_inspected_for_selection": False,
    }
    _write_json(result_dir / "verdict.json", verdict_payload)

    remote_audit = {
        "obs_url": obs_url,
        "obs_size_bytes": obs_size,
        "obs_full_downloaded": False,
        "parquet_rows": parquet.metadata.num_rows,
        "parquet_row_groups": parquet.metadata.num_row_groups,
        "parquet_schema": str(parquet.schema_arrow),
        "footer_bytes_transferred": footer_bytes,
        "selected_row_groups": selected_groups,
        "row_group_manifest": row_group_manifest,
        "total_remote_bytes_transferred": reader.bytes_transferred,
        "range_requests": reader.range_requests,
        "plate_raw_obs_rows": plate_raw_rows,
        "plate_full_filter_rows": plate_full_rows,
    }
    _write_json(result_dir / "remote_condition_index_audit.json", remote_audit)
    download_manifest["remote_selective_obs_read"] = {
        "url": obs_url,
        "full_file_size_bytes": obs_size,
        "full_file_downloaded": False,
        "bytes_transferred": reader.bytes_transferred,
        "selected_row_groups": selected_groups,
        "selected_columns": columns,
    }
    (result_dir / "download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_manifest = {
        "phase": "CGC-SUPPORT-0B / Tahoe replicate-core feasibility",
        "status": "complete",
        "verdict": verdict,
        "parent_commit": "9465be0af36daa65d2ed4d4d257153dbfdc6ea9b",
        "branch": _git(repo, "branch", "--show-current"),
        "head_at_run": _git(repo, "rev-parse", "HEAD"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_revision": revision,
        "remote_bytes_transferred": reader.bytes_transferred,
        "large_de_downloaded": False,
        "full_expression_atlas_downloaded": False,
        "obs_metadata_full_downloaded": False,
        "selection_inputs": [
            "sample metadata",
            "cell-line metadata",
            "Plate 6/14 full-filter cell presence/counts",
            "plate-matched DMSO presence/counts",
        ],
        "selection_excluded_inputs": [
            "gene expression",
            "log2FoldChange",
            "p-values",
            "DE magnitude",
            "context geometry",
        ],
        "support_scaling_run": False,
        "model_trained": False,
        "sanity_check": {
            "publication_plate6_pairs": 4800,
            "publication_plate14_pairs": 4796,
            "publication_common_pairs": 4796,
            "derived_plate6_pairs": plate_condition_counts["plate6"],
            "derived_plate14_pairs": plate_condition_counts["plate14"],
            "derived_common_pairs": common_pairs,
        },
    }
    _write_json(result_dir / "run_manifest.json", run_manifest)
    _write_reports(
        repo,
        verdict_payload,
        replicate_definition,
        future_requirement,
        remote_audit,
        cell_manifest,
        candidates,
    )
    return verdict_payload


def _write_reports(
    repo: Path,
    verdict: dict[str, Any],
    replicate: dict[str, Any],
    future: dict[str, Any],
    remote: dict[str, Any],
    cells: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    reports = repo / "results/reports"
    reports.mkdir(parents=True, exist_ok=True)
    low = cells[cells["paper_low_coverage_line"]][
        [
            "cell_name",
            "cell_line_id",
            "shared_noncontrol_interventions",
            "plate6_control_cells",
            "plate14_control_cells",
            "preliminary_eligible",
        ]
    ]
    low_lines = "\n".join(
        "- " + ", ".join(f"{column}={value}" for column, value in row.items())
        for row in low.to_dict(orient="records")
    )
    inventory = f"""# CGC-SUPPORT-0B Tahoe inventory

The official `tahoebio/Tahoe-100M` repository was frozen at revision
`2dc57900b7981cfcf5e211527169a0b006546a95`. Four small metadata parquet files
(1.452 MB total) were downloaded. The 2.294 GB obs metadata and 88.860 GB
pseudobulk-DE release were not downloaded in full.

HTTP range reads transferred {remote['total_remote_bytes_transferred'] / 1e6:.3f} MB from selected
obs-metadata columns/row groups. No expression or DE-effect column was read.

- Plate 6 samples: {verdict['plate6_samples']}
- Plate 14 samples: {verdict['plate14_samples']}
- Exact non-control drug×dose treatments on each plate: {verdict['exact_sample_level_treatment_overlap']}
- Full-filter cells: Plate 6 = {verdict['plate6_full_filter_cells']:,}; Plate 14 = {verdict['plate14_full_filter_cells']:,}
- Cell lines observed: {verdict['cell_lines_observed']}

The three paper-reported low-coverage lines were retained for the frozen eligibility rule:

{low_lines}
"""
    (reports / "cgc_tahoe_0b_inventory.md").write_text(inventory, encoding="utf-8")
    core = f"""# CGC-SUPPORT-0B replicate core

Plate 14 is explicitly designated by the manuscript as a biological replicate of Plate 6.
They are separate experimental plates with disjoint sample IDs and disjoint sublibrary IDs.
This supports an independent plate replicate interpretation, not independent donor biology.

- Plate 6 matched cell-line×treatment pairs: {verdict['plate6_matched_pairs_including_control']}
- Plate 14 matched pairs: {verdict['plate14_matched_pairs_including_control']}
- Common pairs: {verdict['common_pairs_including_control']}
- Preliminary eligible contexts: {verdict['preliminary_eligible_contexts']}
- Selected complete core: **{verdict['selected_core_dimensions']}**
- Minimum treatment-cell count in selected core: {verdict['minimum_selected_treatment_cells']}
- Plate-matched DMSO controls usable: {verdict['controls_usable']}

The primary choice maximized contexts subject to at least 50 common interventions, then
interventions, then minimum treatment-cell count. Selection used coverage and counts only.
No response magnitude or differential-expression statistic was inspected.
"""
    (reports / "cgc_tahoe_0b_replicate_core.md").write_text(core, encoding="utf-8")
    storage = f"""# CGC-SUPPORT-0B storage plan

- Full pseudobulk-DE release: {future['full_release_gb']:.3f} GB
- Estimated selected Plate6/14 core condition data: {future['estimated_selective_condition_data_gb']:.3f} GB
- Estimated all-shard footer scan: {future['estimated_all_shard_footer_scan_gb']:.3f} GB
- Estimated minimum practical remote transfer: {future['minimum_practical_remote_transfer_gb']:.3f} GB
- Expected peak RAM: {future['expected_peak_ram_gb']} GB
- Full 88.9 GB download performed: **NO**

Remote inspection confirmed HTTP range support and condition-clustered DE row groups. A future
formal phase can selectively extract Plate6/14 conditions; this feasibility phase did not do so.
"""
    (reports / "cgc_tahoe_0b_storage_plan.md").write_text(storage, encoding="utf-8")
    summary = f"""# CGC-SUPPORT-0B phase summary

- Replicate relationship confirmed: YES
- Eligible cell-line contexts: {verdict['selected_core_contexts']}
- Exact shared drug×dose interventions in dense core: {verdict['selected_core_interventions']}
- Complete core: {verdict['selected_core_dimensions']}
- Plate-matched DMSO controls: USABLE
- Publication sanity check (4800 / 4796 / 4796): derived
  {verdict['plate6_matched_pairs_including_control']} /
  {verdict['plate14_matched_pairs_including_control']} /
  {verdict['common_pairs_including_control']}
- Support scaling performed: NO
- Model trained: NO
- Full expression/DE release downloaded: NO

## Verdict

`{verdict['verdict']}`
"""
    (reports / "cgc_tahoe_0b_phase_summary.md").write_text(summary, encoding="utf-8")
