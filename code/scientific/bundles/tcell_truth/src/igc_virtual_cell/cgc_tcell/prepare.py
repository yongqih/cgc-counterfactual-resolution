"""Freeze CGC-TCELL cohorts and materialize strict-trans response tensors.

The implementation is deliberately pseudobulk-only. Expression is accessed in
gene chunks from a backed H5AD; complete-matrix densification is prohibited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from igc_virtual_cell.cgc_tcell.core import hash_block
from igc_virtual_cell.cgc_tcell.inventory import _ntc_mask, _targeting_mask
from igc_virtual_cell.phase1.targeted_tensorized import _windows_memory_bytes


def _git(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes"})


def _sha256_lines(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(name).lower(): str(name) for name in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _valid_text(series: pd.Series) -> pd.Series:
    value = series.astype(str).str.strip()
    return series.notna() & ~value.str.lower().isin({"", "nan", "none", "na"})


def _problem_flag(series: pd.Series) -> pd.Series:
    value = series.astype(str).str.strip().str.lower()
    return series.notna() & ~value.isin({"", "nan", "none", "na", "false", "0", "ok", "pass"})


def _normalized_chunk(
    adata: ad.AnnData,
    rows: np.ndarray,
    columns: np.ndarray,
    library_sizes: np.ndarray,
    scale: float,
) -> np.ndarray | sparse.csr_matrix:
    matrix = adata.X[rows, columns]
    factors = (scale / library_sizes).astype(np.float32)
    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.float32, copy=True)
        result = sparse.diags(factors, format="csr") @ result
        np.log1p(result.data, out=result.data)
        return result
    result = np.asarray(matrix, dtype=np.float32)
    result *= factors[:, None]
    np.log1p(result, out=result)
    return result


def _guide_metadata(obs: pd.DataFrame, library: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    library_guide = _column(library, ("sgRNA", "guide_id", "guide"))
    if library_guide is None:
        raise ValueError("Cannot identify guide ID in library metadata")
    metadata = library.copy()
    metadata[library_guide] = metadata[library_guide].astype(str)
    merged = obs.merge(
        metadata,
        left_on="guide_id",
        right_on=library_guide,
        how="left",
        suffixes=("", "_library"),
    )
    return merged, library_guide


def _freeze_cohorts(
    obs: pd.DataFrame,
    library: pd.DataFrame,
    kd: pd.DataFrame,
    *,
    states: list[str],
    primary_flags: list[str],
    output: Path,
    overlap_quantiles: tuple[float, float],
) -> dict[str, Any]:
    merged, library_guide = _guide_metadata(obs, library)
    targeting = _targeting_mask(merged)
    ntc = _ntc_mask(merged)
    primary_flag_mask = pd.Series(True, index=merged.index)
    for flag in primary_flags:
        primary_flag_mask &= _bool(merged[flag])
    ntc_flag_mask = _bool(merged["keep_min_cells"]) & _bool(merged["keep_total_counts"])
    valid_target = _valid_text(merged["perturbed_gene_id"]) & _valid_text(
        merged["perturbed_gene_name"]
    )
    flag_column = _column(library, ("flag",))
    problematic = (
        _problem_flag(merged[flag_column])
        if flag_column is not None
        else pd.Series(False, index=merged.index)
    )
    merged["primary_observation_eligible"] = (
        targeting & primary_flag_mask & valid_target & ~problematic
    )
    merged["ntc_eligible"] = ntc & ntc_flag_mask & ~problematic

    ntc_groups = set(
        map(
            tuple,
            merged.loc[
                merged["ntc_eligible"], ["donor_id", "culture_condition", "10xrun_id"]
            ].astype(str).to_numpy(),
        )
    )
    matched = merged[["donor_id", "culture_condition", "10xrun_id"]].astype(str).apply(
        tuple, axis=1
    ).isin(ntc_groups)
    merged["matched_ntc_available"] = matched
    merged["primary_observation_eligible"] &= matched

    donors = sorted(merged["donor_id"].astype(str).unique())
    expected = {(donor, state) for donor in donors for state in states}
    primary_rows = merged.loc[merged["primary_observation_eligible"]].copy()
    contexts = primary_rows.groupby("perturbed_gene_id", dropna=False).apply(
        lambda frame: set(
            map(tuple, frame[["donor_id", "culture_condition"]].astype(str).to_numpy())
        ),
        include_groups=False,
    )
    complete_targets = set(contexts.index[contexts.map(lambda value: value == expected)].astype(str))
    merged["primary_target_complete"] = merged["perturbed_gene_id"].astype(str).isin(
        complete_targets
    )
    merged["primary_response_row"] = (
        merged["primary_observation_eligible"] & merged["primary_target_complete"]
    )

    distributions = (
        merged.loc[merged["primary_response_row"]]
        .groupby(["donor_id", "culture_condition"], observed=True)["n_cells"]
        .quantile(list(overlap_quantiles))
        .unstack()
    )
    overlap_low = float(distributions.iloc[:, 0].max())
    overlap_high = float(distributions.iloc[:, 1].min())
    in_overlap = merged["n_cells"].between(overlap_low, overlap_high, inclusive="both")
    matched_rows = merged.loc[merged["primary_response_row"] & in_overlap]
    matched_contexts = matched_rows.groupby("perturbed_gene_id", dropna=False).apply(
        lambda frame: set(
            map(tuple, frame[["donor_id", "culture_condition"]].astype(str).to_numpy())
        ),
        include_groups=False,
    )
    cellmatched_targets = set(
        matched_contexts.index[matched_contexts.map(lambda value: value == expected)].astype(str)
    )
    merged["cell_count_matched_response_row"] = (
        merged["primary_response_row"]
        & in_overlap
        & merged["perturbed_gene_id"].astype(str).isin(cellmatched_targets)
    )

    kd_guide = _column(kd, ("index", "Unnamed: 0", "sgRNA", "guide_id", "guide"))
    kd_effective: set[str] = set()
    if kd_guide is not None and "signif_knockdown" in kd:
        kd_effective = set(
            kd.loc[_bool(kd["signif_knockdown"]), kd_guide].astype(str)
        )
    merged["effective_knockdown_secondary"] = merged["guide_id"].astype(str).isin(
        kd_effective
    )

    guide_rows = []
    for guide_id, frame in merged.groupby("guide_id", observed=True, sort=True):
        first = frame.iloc[0]
        eligible = frame.loc[frame["primary_response_row"]]
        guide_rows.append(
            {
                "guide_id": str(guide_id),
                "guide_type": str(first["guide_type"]),
                "perturbed_gene_id": str(first["perturbed_gene_id"]),
                "perturbed_gene_name": str(first["perturbed_gene_name"]),
                "primary_eligible_rows": int(frame["primary_response_row"].sum()),
                "complete_12_contexts": int(
                    eligible.drop_duplicates(["donor_id", "culture_condition"]).shape[0]
                )
                == 12,
                "problematic_library_annotation": bool(problematic.loc[frame.index].any()),
                "effective_knockdown_secondary": bool(
                    frame["effective_knockdown_secondary"].any()
                ),
            }
        )
    eligible_guides = pd.DataFrame(guide_rows)
    eligible_guides.to_csv(output / "eligible_guides.csv", index=False)

    target_rows = []
    primary = merged.loc[merged["primary_response_row"]].copy()
    primary["perturbed_gene_id_text"] = primary["perturbed_gene_id"].astype(str)
    for target_id, frame in primary.groupby(
        "perturbed_gene_id_text", observed=True, sort=True
    ):
        target_rows.append(
            {
                "perturbed_gene_id": target_id,
                "perturbed_gene_name": str(frame["perturbed_gene_name"].iloc[0]),
                "eligible_guides": int(frame["guide_id"].nunique()),
                "eligible_rows": int(len(frame)),
                "complete_contexts": int(
                    frame.drop_duplicates(["donor_id", "culture_condition"]).shape[0]
                ),
                "primary_eligible": True,
                "cell_count_matched_eligible": target_id in cellmatched_targets,
                "effective_knockdown_guides": int(
                    frame.loc[frame["effective_knockdown_secondary"], "guide_id"].nunique()
                ),
                "intervention_block": hash_block(target_id),
            }
        )
    eligible_targets = pd.DataFrame(target_rows)
    eligible_targets.to_csv(output / "eligible_targets.csv", index=False)

    ntc_manifest = merged.loc[
        merged["ntc_eligible"],
        [
            "pseudobulk_id",
            "guide_id",
            "donor_id",
            "culture_condition",
            "10xrun_id",
            "n_cells",
            "total_counts",
        ],
    ].copy()
    ntc_manifest.to_csv(output / "ntc_manifest.csv", index=False)
    return {
        "obs": merged,
        "eligible_targets": eligible_targets,
        "complete_targets": sorted(complete_targets),
        "cellmatched_targets": sorted(cellmatched_targets),
        "overlap_low": overlap_low,
        "overlap_high": overlap_high,
        "donors": donors,
        "states": states,
        "library_guide_column": library_guide,
    }


def _strict_gene_manifest(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    eligible_targets: pd.DataFrame,
    *,
    chunk_genes: int,
    chunk_rows: int,
    cpm_scale: float,
    output: Path,
) -> pd.DataFrame:
    used = obs["primary_response_row"] | obs["ntc_eligible"]
    rows = np.flatnonzero(used.to_numpy())
    library_sizes = obs.loc[used, "total_counts"].to_numpy(dtype=np.float64)
    if np.any(library_sizes <= 0):
        raise ValueError("Eligible pseudobulks contain non-positive total_counts")
    sums = np.zeros(adata.n_vars, dtype=np.float64)
    squares = np.zeros(adata.n_vars, dtype=np.float64)
    columns = np.arange(adata.n_vars)
    for row_start in range(0, len(rows), chunk_rows):
        row_stop = min(row_start + chunk_rows, len(rows))
        matrix = _normalized_chunk(
            adata,
            rows[row_start:row_stop],
            columns,
            library_sizes[row_start:row_stop],
            cpm_scale,
        )
        if sparse.issparse(matrix):
            sums += np.asarray(matrix.sum(axis=0)).ravel()
            squares += np.asarray(matrix.power(2).sum(axis=0)).ravel()
        else:
            sums += matrix.sum(axis=0, dtype=np.float64)
            squares += np.square(matrix, dtype=np.float64).sum(axis=0)
    variance = np.maximum(squares / len(rows) - np.square(sums / len(rows)), 0.0)
    var = adata.var.reset_index(names="variable_id").copy()
    gene_id_column = _column(
        var, ("gene_id", "gene_ids", "ensembl_id", "variable_id")
    ) or "variable_id"
    gene_name_column = _column(var, ("gene_name", "gene_symbol", "symbol"))
    gene_id = var[gene_id_column].astype(str)
    gene_name = (
        var[gene_name_column].astype(str) if gene_name_column else var["variable_id"].astype(str)
    )
    targets = set(eligible_targets["perturbed_gene_id"].astype(str))
    target_names = set(eligible_targets["perturbed_gene_name"].astype(str))
    is_target = gene_id.isin(targets) | gene_name.isin(target_names)
    is_mito = gene_name.str.upper().str.startswith("MT-")
    upper = gene_name.str.upper()
    is_technical = upper.str.startswith(
        ("__", "TOTAL", "HTO", "GUIDE", "SGRNA", "PURO")
    ) | gene_id.str.upper().str.startswith("CUSTOM")
    zero_variance = variance <= 1.0e-12
    manifest = pd.DataFrame(
        {
            "gene_index": np.arange(adata.n_vars),
            "gene_id": gene_id,
            "gene_name": gene_name,
            "normalized_variance": variance,
            "excluded_perturbation_target": is_target,
            "excluded_mitochondrial": is_mito,
            "excluded_technical": is_technical,
            "excluded_zero_variance": zero_variance,
        }
    )
    manifest["strict_trans_eligible"] = ~(
        is_target | is_mito | is_technical | zero_variance
    )
    manifest.to_csv(output / "strict_trans_genes.csv", index=False)
    ordered = manifest.loc[manifest["strict_trans_eligible"], "gene_id"].astype(str).tolist()
    (output / "strict_trans_gene_hash.txt").write_text(
        _sha256_lines(ordered) + "\n", encoding="utf-8"
    )
    return manifest


def _response_weights(
    obs: pd.DataFrame,
    targets: list[str],
    donors: list[str],
    states: list[str],
    *,
    response_flag: str,
) -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    target_position = {target: index for index, target in enumerate(targets)}
    donor_position = {donor: index for index, donor in enumerate(donors)}
    state_position = {state: index for index, state in enumerate(states)}
    ntc_lookup = {
        key: frame.index.to_numpy(dtype=np.int64)
        for key, frame in obs.loc[obs["ntc_eligible"]].groupby(
            ["donor_id", "culture_condition", "10xrun_id"], observed=True
        )
    }
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    nuisance = np.full((len(targets), 4, 3), np.nan, dtype=np.float32)
    metadata_rows = []
    selected = obs.loc[obs[response_flag]].copy()
    for (target, donor, state), frame in selected.groupby(
        ["perturbed_gene_id", "donor_id", "culture_condition"], observed=True
    ):
        target, donor, state = str(target), str(donor), str(state)
        if target not in target_position:
            continue
        unit = (target_position[target] * 4 + donor_position[donor]) * 3 + state_position[state]
        guides = sorted(frame["guide_id"].astype(str).unique())
        positive_weights: dict[int, float] = {}
        for guide in guides:
            guide_frame = frame.loc[frame["guide_id"].astype(str).eq(guide)]
            weight = 1.0 / len(guides) / len(guide_frame)
            for index in guide_frame.index:
                positive_weights[int(index)] = weight
        ntc_weights: dict[int, float] = {}
        for index, weight in positive_weights.items():
            item = obs.loc[index]
            key = (item["donor_id"], item["culture_condition"], item["10xrun_id"])
            controls = ntc_lookup[key]
            for control in controls:
                ntc_weights[int(control)] = ntc_weights.get(int(control), 0.0) - weight / len(controls)
        for index, weight in (*positive_weights.items(), *ntc_weights.items()):
            row_indices.append(unit)
            col_indices.append(index)
            data.append(weight)
        nuisance[target_position[target], donor_position[donor], state_position[state]] = sum(
            obs.loc[index, "n_cells"] * weight for index, weight in positive_weights.items()
        )
        metadata_rows.append(
            {
                "perturbed_gene_id": target,
                "donor_id": donor,
                "culture_condition": state,
                "eligible_guides": len(guides),
                "targeting_rows": len(positive_weights),
                "matched_ntc_rows": len(ntc_weights),
                "mean_target_n_cells_equal_guide_run": nuisance[
                    target_position[target], donor_position[donor], state_position[state]
                ],
            }
        )
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (row_indices, col_indices)),
        shape=(len(targets) * 12, len(obs)),
    )
    if not np.isfinite(nuisance).all() or matrix.shape[0] != len(metadata_rows):
        raise ValueError(f"Incomplete response tensor for {response_flag}")
    if not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 0.0, atol=1e-6):
        raise ValueError("Target-minus-NTC weights do not sum to zero")
    return matrix, nuisance, pd.DataFrame(metadata_rows)


def _materialize_response(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    strict: pd.DataFrame,
    weights: sparse.csr_matrix,
    *,
    targets: list[str],
    chunk_genes: int,
    chunk_rows: int,
    cpm_scale: float,
    path: Path,
) -> np.memmap:
    genes = strict.loc[strict["strict_trans_eligible"], "gene_index"].to_numpy(dtype=np.int64)
    used = np.unique(weights.indices)
    local_weights = weights[:, used]
    library_sizes = obs.loc[used, "total_counts"].to_numpy(dtype=np.float64)
    response = np.memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(targets), 4, 3, len(genes)),
    )
    flat = response.reshape(len(targets) * 12, len(genes))
    flat[:] = 0.0
    for row_start in range(0, len(used), chunk_rows):
        row_stop = min(row_start + chunk_rows, len(used))
        normalized = _normalized_chunk(
            adata,
            used[row_start:row_stop],
            genes,
            library_sizes[row_start:row_stop],
            cpm_scale,
        )
        row_weights = local_weights[:, row_start:row_stop]
        for output_start in range(0, len(genes), chunk_genes):
            output_stop = min(output_start + chunk_genes, len(genes))
            block = row_weights @ normalized[:, output_start:output_stop]
            dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
            flat[:, output_start:output_stop] += dense
    response.flush()
    return response


def run_prepare(root: Path, config_path: Path) -> dict[str, Any]:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = root / config["storage"]["raw_directory"]
    processed = root / config["storage"]["processed_directory"]
    output = root / config["outputs"]["directory"]
    processed.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    adata = ad.read_h5ad(raw / "GWCD4i.pseudobulk_merged.h5ad", backed="r")
    obs = adata.obs.reset_index(names="pseudobulk_id").copy()
    library = pd.read_csv(raw / "sgrna_library_metadata.suppl_table.csv", low_memory=False)
    kd = pd.read_csv(raw / "guide_kd_efficiency.suppl_table.csv", low_memory=False)
    frozen = _freeze_cohorts(
        obs,
        library,
        kd,
        states=list(map(str, config["frozen_states"])),
        primary_flags=list(config["eligibility"]["primary_observation_flags"]),
        output=output,
        overlap_quantiles=tuple(config["eligibility"]["cell_count_overlap_quantiles"]),
    )
    obs = frozen["obs"]
    strict = _strict_gene_manifest(
        adata,
        obs,
        frozen["eligible_targets"],
        chunk_genes=int(config["normalization"]["chunk_genes"]),
        chunk_rows=int(config["normalization"]["chunk_rows"]),
        cpm_scale=float(config["normalization"]["cpm_scale"]),
        output=output,
    )
    artifacts = {}
    for cohort, flag, targets in (
        ("primary", "primary_response_row", frozen["complete_targets"]),
        ("cell_count_matched", "cell_count_matched_response_row", frozen["cellmatched_targets"]),
    ):
        weights, nuisance, response_manifest = _response_weights(
            obs,
            targets,
            frozen["donors"],
            frozen["states"],
            response_flag=flag,
        )
        response_manifest.to_csv(output / f"{cohort}_response_units.csv", index=False)
        nuisance_path = processed / f"{cohort}_n_cells.npy"
        np.save(nuisance_path, nuisance)
        response_path = processed / f"{cohort}_delta.float32.mmap"
        _materialize_response(
            adata,
            obs,
            strict,
            weights,
            targets=targets,
            chunk_genes=int(config["normalization"]["chunk_genes"]),
            chunk_rows=int(config["normalization"]["chunk_rows"]),
            cpm_scale=float(config["normalization"]["cpm_scale"]),
            path=response_path,
        )
        artifacts[cohort] = {
            "targets": len(targets),
            "target_ids": targets,
            "response_path": str(response_path.relative_to(root)),
            "response_shape": [len(targets), 4, 3, int(strict["strict_trans_eligible"].sum())],
            "nuisance_path": str(nuisance_path.relative_to(root)),
        }
    manifest = {
        "git_provenance": _git(root),
        "normalization": "log1p(CPM)",
        "matched_ntc_level": "donor_state_run",
        "equal_ntc_guide_weighting": True,
        "equal_target_guide_weighting": True,
        "cell_count_weighting": False,
        "strict_trans_gene_hash": (output / "strict_trans_gene_hash.txt").read_text().strip(),
        "strict_trans_genes": int(strict["strict_trans_eligible"].sum()),
        "cell_count_overlap": [frozen["overlap_low"], frozen["overlap_high"]],
        "artifacts": artifacts,
        "complete_matrix_toarray_called": False,
        "peak_rss_bytes": int(_windows_memory_bytes()[1]),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "normalized_response_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    adata.file.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    print(json.dumps(run_prepare(root, config), indent=2))


if __name__ == "__main__":
    main()
