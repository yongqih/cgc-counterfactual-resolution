"""Counts-based Feng targeted pseudobulks with an explicit observation mask."""

from __future__ import annotations

import ctypes
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from igc_virtual_cell.data.materialize_feng import _normalized_cell_ids
from igc_virtual_cell.phase1.materialize import _feng_maps, _ntc_matrix, _write_delta


TARGETED_DELTA_FILE = "feng_targeted_counts_deltas.parquet"
TARGETED_MASK_FILE = "feng_targeted_observation_mask.parquet"


def _process_peak_ram_bytes() -> int:
    if not hasattr(ctypes, "windll"):
        return 0

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize) if ok else 0


def targeted_observation_table(root: Path) -> pd.DataFrame:
    """Return the full 444 x 19 grid and mark the one unobserved combination."""
    counts = pd.read_csv(root / "results/tables/feng_2026_cell_counts_by_perturbation_context.csv")
    observed = counts.loc[counts["screen"].eq("targeted")].copy()
    observed["context_id"] = observed["context_id"].astype(str)
    observed["perturbation_id"] = observed["perturbation_id"].astype(str)
    observed = observed.sort_values(["context_id", "perturbation_id"]).reset_index(drop=True)
    observed["response_row_index"] = np.arange(len(observed), dtype=np.int64)
    contexts = sorted(observed["context_id"].unique())
    perturbations = sorted(observed["perturbation_id"].unique())
    full = pd.MultiIndex.from_product(
        [contexts, perturbations], names=["context_id", "perturbation_id"]
    ).to_frame(index=False)
    full = full.merge(
        observed[["context_id", "perturbation_id", "cell_count", "response_row_index"]],
        on=["context_id", "perturbation_id"],
        how="left",
        validate="one_to_one",
    )
    full["observed"] = full["response_row_index"].notna()
    full["cell_count"] = full["cell_count"].fillna(0).astype(np.int64)
    full["response_row_index"] = full["response_row_index"].astype("Int64")
    if len(observed) != 8_435 or len(full) != 8_436 or int((~full["observed"]).sum()) != 1:
        raise ValueError(
            f"Unexpected Feng targeted grid: observed={len(observed)}, full={len(full)}, "
            f"missing={int((~full['observed']).sum())}"
        )
    return full


def materialize_feng_targeted_counts(root: Path, output: Path) -> tuple[Path, Path, dict[str, object]]:
    """Aggregate the official processed UMI counts into 8,435 response pseudobulks.

    Cells are consumed only during aggregation.  The returned artifact contains one
    row per observed perturbation x context combination and is the only expression
    object used by the tensorized Phase-I implementation.
    """
    output.mkdir(parents=True, exist_ok=True)
    destination = output / TARGETED_DELTA_FILE
    mask_path = output / TARGETED_MASK_FILE
    profile_path = output / "feng_targeted_counts_materialization.json"
    mask = targeted_observation_table(root)
    mask.to_parquet(mask_path, index=False, compression="zstd")
    if destination.exists() and profile_path.exists():
        return destination, mask_path, json.loads(profile_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    raw = root / "data/raw/feng_2026"
    manifest = pd.read_csv(root / "data/manifests/feng_2026_targeted.csv", low_memory=False)
    ntc = _ntc_matrix(
        root / "data/processed/feng_2026/feng_2026_targeted_ntc_log1p_cpm.parquet",
        manifest,
    )
    genes = list(map(str, ntc.columns))
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    context_map, donor_map = _feng_maps(root, "targeted")

    observed = mask.loc[mask["observed"]].copy()
    observed = observed.sort_values("response_row_index").reset_index(drop=True)
    group_lookup = {
        (str(row.context_id), str(row.perturbation_id)): int(row.response_row_index)
        for row in observed.itertuples(index=False)
    }
    metadata = pd.read_csv(raw / "TargetedScreen_Cell-Metadata.tsv.gz", sep="\t")
    metadata["count_cell_id"] = _normalized_cell_ids(metadata, "TargetedScreen")
    metadata = metadata.set_index("count_cell_id")

    count_path = raw / "TargetedScreen_RNA-UMI-Counts.csv.gz"
    sums = np.zeros((len(observed), len(genes)), dtype=np.float64)
    observed_cells = np.zeros(len(observed), dtype=np.int64)
    genes_used = 0
    with gzip.open(count_path, "rb") as handle:
        header = handle.readline().rstrip(b"\r\n").decode("utf-8").split(",")[1:]
        aligned = metadata.reindex(header)
        # The released count matrix has one extra column with no metadata row;
        # the authors' NTC materializer likewise treats it as unmatched.
        matched_columns = aligned["Cell_ID"].notna().to_numpy()
        contexts = aligned["Cell_Line"].astype(str).map(context_map)
        guide = aligned["Guide_Call"].fillna("").astype(str)
        perturbations = guide.str.rsplit("_", n=1).str[0]
        codes = np.fromiter(
            (
                group_lookup.get((str(context), str(perturbation)), -1)
                for context, perturbation in zip(contexts, perturbations, strict=True)
            ),
            dtype=np.int64,
            count=len(header),
        )
        selected = (codes >= 0) & matched_columns
        selected_codes = codes[selected]
        observed_cells = np.bincount(selected_codes, minlength=len(observed))
        for line in handle:
            comma = line.find(b",")
            if comma < 0:
                continue
            label = line[:comma].decode("utf-8")
            parts = label.split(":")
            gene = parts[1] if len(parts) > 1 else parts[0]
            gene_index = gene_lookup.get(gene)
            if gene_index is None:
                continue
            values = np.fromstring(line[comma + 1 :], dtype=np.float64, sep=",")
            if len(values) != len(header):
                raise ValueError(f"Count row length mismatch for {gene}")
            sums[:, gene_index] += np.bincount(
                selected_codes,
                weights=values[selected],
                minlength=len(observed),
            )
            genes_used += 1

    expected_cells = observed["cell_count"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_cells, expected_cells):
        bad = np.flatnonzero(observed_cells != expected_cells)[:10]
        raise ValueError(f"Feng targeted group cell counts disagree at rows {bad.tolist()}")
    libraries = sums.sum(axis=1)
    if np.any(libraries <= 0):
        raise ValueError("Zero library in Feng targeted perturbation pseudobulk")
    normalized = np.log1p(sums / libraries[:, None] * 1_000_000.0)
    baselines = ntc.loc[observed["context_id"], genes].to_numpy(dtype=np.float64)
    deltas = normalized - baselines
    row_metadata = pd.DataFrame(
        {
            "study": "feng_targeted",
            "context_id": observed["context_id"].astype(str),
            "donor_id": observed["context_id"].astype(str).map(donor_map),
            "perturbation_id": observed["perturbation_id"].astype(str),
            "cell_count": expected_cells,
            "response_source": "processed_UMI_pseudobulk_minus_NTC_log1p_CPM",
        }
    )
    _write_delta(row_metadata, deltas, genes, destination)
    missing = mask.loc[~mask["observed"], ["context_id", "perturbation_id"]].iloc[0]
    profile: dict[str, object] = {
        "observed_response_vectors": len(observed),
        "theoretical_response_vectors": len(mask),
        "missing_combinations": 1,
        "missing_context_id": str(missing["context_id"]),
        "missing_perturbation_id": str(missing["perturbation_id"]),
        "genes": len(genes),
        "count_columns_cells": len(header),
        "count_columns_without_metadata": int((~matched_columns).sum()),
        "assigned_perturbed_cells": int(selected.sum()),
        "cell_counts_match_audit_table": True,
        "genes_used": genes_used,
        "peak_ram_bytes": _process_peak_ram_bytes(),
        "seconds": time.perf_counter() - started,
        "per_cell_prediction": False,
    }
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return destination, mask_path, profile
