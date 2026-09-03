"""Stream official Tahoe Plate 6/14 CSR counts into frozen-core pseudobulks.

The implementation never materializes a dense cell-by-gene matrix.  It reads
contiguous cell chunks from the backed HDF5 CSR representation, aggregates in
float64, validates, and stores compact float32 pseudobulk counts in Zarr.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from numcodecs import Blosc


PLATES = ("plate6", "plate14")
CHUNK_ROWS = 8_192
SPOT_CHECK_SEED = 2_601
SPOT_CHECK_COUNT = 100

RAW_FILENAMES = {
    "plate6": "plate6_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    "plate14": "plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _categorical(store: h5py.File, column: str) -> tuple[list[str], np.ndarray]:
    group = store[f"obs/{column}"]
    categories = _decode(group["categories"][:])
    codes = np.asarray(group["codes"][:], dtype=np.int16)
    if np.any(codes < 0):
        raise RuntimeError(f"Missing categorical values in obs/{column}")
    return categories, codes


def _sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def verify_downloads(root: Path) -> dict[str, Any]:
    """Verify exact size and official MD5 before any large computation."""

    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0i"
    source = json.loads((result_dir / "raw_source_manifest.json").read_text(encoding="utf-8"))
    expected = {item["plate"]: item for item in source["objects"]}
    hashes: dict[str, Any] = {}
    for plate in PLATES:
        path = root / "data/tahoe100m_plate6_14_raw" / RAW_FILENAMES[plate]
        if not path.exists():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(expected[plate]["size_bytes"]):
            raise RuntimeError(f"{plate} size mismatch: {size}")
        observed_md5 = _md5(path)
        if observed_md5 != str(expected[plate]["md5_hex"]):
            raise RuntimeError(f"{plate} official MD5 mismatch")
        hashes[plate] = {
            "path": str(path.relative_to(root)),
            "size_bytes": size,
            "algorithm": "md5",
            "expected_hex": expected[plate]["md5_hex"],
            "observed_hex": observed_md5,
            "local_verified": True,
        }
    hashes["verified_at"] = _now()
    _write_json(result_dir / "raw_source_hashes.json", hashes)
    return hashes


def _frozen_design(root: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    core = pd.read_csv(root / "results/cgc_tahoe_0b/selected_replicate_core.csv")
    contexts = sorted(core["cell_line_id"].astype(str).unique())
    interventions = sorted(core["intervention_id"].astype(str).unique())
    if len(contexts) != 50 or len(interventions) != 93 or len(core) != 50 * 93:
        raise RuntimeError("Frozen Tahoe 0B core changed")
    if not core["present_both_plates"].astype(bool).all():
        raise RuntimeError("Frozen Tahoe 0B core is not plate-complete")
    design = pd.read_csv(root / "results/cgc_tahoe_0b/sample_design.csv")
    design["plate"] = design["plate"].astype(str)
    design["sample"] = design["sample"].astype(str)
    selected = design[
        design["intervention_id"].astype(str).isin(interventions)
        | design["is_control"].astype(bool)
    ].copy()
    if selected.groupby("plate")["sample"].nunique().to_dict() != {"plate14": 95, "plate6": 95}:
        raise RuntimeError("Unexpected frozen sample count")
    return contexts, interventions, selected


def _aggregate_plate(
    x: h5py.Group,
    retained_sample_axis: np.ndarray,
    sample_codes: np.ndarray,
    cell_line_codes: np.ndarray,
    source_row_filter: np.ndarray,
    gene_count: int,
    selected_cells: set[int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Aggregate a plate in one sequential pass despite interleaved sample rows."""

    groups = 95 * 50
    rows = len(sample_codes)
    accumulator = np.zeros((groups, gene_count), dtype=np.float64)
    cell_counts = np.zeros(groups, dtype=np.int64)
    checks: list[dict[str, Any]] = []
    global_indptr = x["indptr"]
    for chunk_index, row_start in enumerate(range(0, rows, CHUNK_ROWS)):
        row_stop = min(rows, row_start + CHUNK_ROWS)
        pointers = np.asarray(global_indptr[row_start : row_stop + 1], dtype=np.int64)
        nnz_start, nnz_stop = int(pointers[0]), int(pointers[-1])
        local_indptr = pointers - nnz_start
        indices = np.asarray(x["indices"][nnz_start:nnz_stop], dtype=np.int64)
        raw_data = np.asarray(x["data"][nnz_start:nnz_stop])
        if np.any(raw_data < 0) or np.any(raw_data != np.floor(raw_data)):
            raise RuntimeError("TAHOE_RAW_COUNT_SOURCE_INVALID")
        data = raw_data.astype(np.float64, copy=False)
        matrix = sp.csr_matrix(
            (data, indices, local_indptr),
            shape=(row_stop - row_start, gene_count),
        )
        local_contexts = cell_line_codes[row_start:row_stop]
        local_samples = retained_sample_axis[sample_codes[row_start:row_stop]]
        valid = (local_samples >= 0) & source_row_filter[row_start:row_stop]
        group_codes = local_samples[valid].astype(np.int64) * 50 + local_contexts[valid]
        cell_counts += np.bincount(group_codes, minlength=groups)
        unique_groups, inverse = np.unique(group_codes, return_inverse=True)
        valid_columns = np.flatnonzero(valid)
        grouping = sp.csr_matrix(
            (
                np.ones(len(valid_columns), dtype=np.float64),
                (inverse, valid_columns),
            ),
            shape=(len(unique_groups), row_stop - row_start),
        )
        accumulator[unique_groups] += (grouping @ matrix).toarray()

        for global_row in sorted(selected_cells.intersection(range(row_start, row_stop))):
            local_row = global_row - row_start
            left, right = int(local_indptr[local_row]), int(local_indptr[local_row + 1])
            row_indices = indices[left:right]
            row_values = raw_data[left:right]
            pick = 0 if len(row_indices) else None
            checks.append(
                {
                    "cell_row": global_row,
                    "cell_line_code": int(local_contexts[local_row]),
                    "sample_axis": int(local_samples[local_row]),
                    "row_sum_from_x": float(row_values.sum(dtype=np.float64)),
                    "nonzero_genes": int(len(row_indices)),
                    "spot_gene_index": None if pick is None else int(row_indices[pick]),
                    "spot_gene_count": None if pick is None else float(row_values[pick]),
                }
            )
        if (chunk_index + 1) % 64 == 0 or row_stop == rows:
            print(f"plate extraction cells {row_stop}/{rows}", flush=True)
    return accumulator, cell_counts, checks


def extract_pseudobulks(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0i"
    data_dir = root / "data/tahoe100m_plate6_14_raw"
    contexts, interventions, design = _frozen_design(root)
    context_index = {value: index for index, value in enumerate(contexts)}
    intervention_set = set(interventions)

    gene_metadata = pd.read_parquet(root / "data/tahoe100m_metadata/gene_metadata.parquet").copy()
    if gene_metadata.shape != (62_710, 3):
        raise RuntimeError("Official gene metadata changed")
    if not gene_metadata["gene_symbol"].astype(str).is_unique:
        raise RuntimeError("Ambiguous duplicate gene symbols in official mapping")
    gene_metadata.insert(0, "raw_gene_index", np.arange(len(gene_metadata), dtype=np.int64))
    gene_metadata["mapping_status"] = "unique_official_gene_symbol_ensembl_token"
    gene_metadata.to_csv(result_dir / "gene_metadata_frozen.csv", index=False)

    store_path = result_dir / "pseudobulk_raw_counts.zarr"
    if store_path.exists() and (result_dir / "extraction_manifest.json").exists():
        raise RuntimeError(f"Refusing to overwrite existing extraction: {store_path}")
    # A store without an extraction manifest is a failed/partial, explicitly
    # regenerable extraction. Zarr mode='w' clears it before the audited rerun.
    group = zarr.open_group(store_path, mode="w", zarr_format=2)
    counts_store = group.create_array(
        "raw_counts",
        shape=(2, 95, 50, len(gene_metadata)),
        chunks=(1, 1, 10, 2_048),
        dtype="float32",
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        fill_value=0,
    )
    group.attrs.update(
        {
            "plates": list(PLATES),
            "contexts": contexts,
            "genes_sha256": hashlib.sha256(
                "\n".join(gene_metadata["gene_symbol"].astype(str)).encode("utf-8")
            ).hexdigest(),
            "accumulation_dtype": "float64",
            "storage_dtype": "float32",
        }
    )

    rng = np.random.default_rng(SPOT_CHECK_SEED)
    metadata_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    max_count = 0.0
    plate_stats: dict[str, Any] = {}
    try:
        for plate_axis, plate in enumerate(PLATES):
            raw_path = data_dir / RAW_FILENAMES[plate]
            with h5py.File(raw_path, "r") as store:
                genes = _decode(store["var/gene_name"][:])
                if genes != gene_metadata["gene_symbol"].astype(str).tolist():
                    raise RuntimeError(f"{plate} gene axis differs from official gene metadata")
                sample_categories, sample_codes = _categorical(store, "sample")
                cell_categories, cell_codes = _categorical(store, "cell_line")
                filter_categories, filter_codes = _categorical(store, "pass_filter")
                if filter_categories.count("full") != 1:
                    raise RuntimeError(f"{plate}: unique pass_filter='full' category required")
                source_row_filter = filter_codes == filter_categories.index("full")
                if set(cell_categories) != set(contexts):
                    raise RuntimeError(f"{plate} cell-line axis differs from frozen 0B core")
                remap = np.array([context_index[v] for v in cell_categories], dtype=np.int16)
                frozen_cell_codes = remap[cell_codes]
                sample_code = {value: index for index, value in enumerate(sample_categories)}
                plate_design = design[design["plate"] == plate].copy()
                plate_design = plate_design.sort_values(
                    ["is_control", "intervention_id", "pair_match_id"], kind="stable"
                ).reset_index(drop=True)
                if len(plate_design) != 95:
                    raise RuntimeError(f"{plate}: expected 95 retained samples")
                retained_sample_axis = np.full(len(sample_categories), -1, dtype=np.int16)
                for sample_axis, sample in enumerate(plate_design["sample"].astype(str)):
                    retained_sample_axis[sample_code[sample]] = sample_axis
                retained_cell_rows = np.flatnonzero(
                    (retained_sample_axis[sample_codes] >= 0) & source_row_filter
                )
                selected_rows = set(
                    map(
                        int,
                        rng.choice(
                            retained_cell_rows,
                            size=SPOT_CHECK_COUNT,
                            replace=False,
                        ),
                    )
                )
                matrix, cell_counts, checks = _aggregate_plate(
                    store["X"],
                    retained_sample_axis,
                    sample_codes,
                    frozen_cell_codes,
                    source_row_filter,
                    len(gene_metadata),
                    selected_rows,
                )
                if matrix.max(initial=0) >= 2**24:
                    raise RuntimeError("Pseudobulk count exceeds exact float32 integer range")
                max_count = max(max_count, float(matrix.max(initial=0)))
                reshaped_matrix = matrix.reshape(95, 50, len(gene_metadata))
                reshaped_cells = cell_counts.reshape(95, 50)
                counts_store[plate_axis, :, :, :] = reshaped_matrix.astype(np.float32)
                observed_cells = int(cell_counts.sum())
                for sample_axis, row in plate_design.iterrows():
                    sample = str(row["sample"])
                    for context_axis, context in enumerate(contexts):
                        metadata_rows.append(
                            {
                                "plate": plate,
                                "plate_axis": plate_axis,
                                "sample": sample,
                                "sample_axis": sample_axis,
                                "cell_line_id": context,
                                "context_axis": context_axis,
                                "intervention_id": str(row["intervention_id"]),
                                "is_control": bool(row["is_control"]),
                                "control_well": str(row["pair_match_id"]) if bool(row["is_control"]) else "",
                                "drug": str(row["parsed_drug"]),
                                "concentration": float(row["concentration"]),
                                "concentration_unit": str(row["concentration_unit"]),
                                "n_cells": int(reshaped_cells[sample_axis, context_axis]),
                                "library_size": float(reshaped_matrix[sample_axis, context_axis].sum(dtype=np.float64)),
                            }
                        )
                for check in checks:
                    context_axis = check["cell_line_code"]
                    sample_axis = check["sample_axis"]
                    gene_axis = check["spot_gene_index"]
                    sample = str(plate_design.iloc[sample_axis]["sample"])
                    check.update(
                        {
                            "plate": plate,
                            "sample": sample,
                            "cell_line_id": contexts[context_axis],
                            "pseudobulk_contains_spot_count": True
                            if gene_axis is None
                            else bool(reshaped_matrix[sample_axis, context_axis, gene_axis] >= check["spot_gene_count"]),
                            "integer_nonnegative": bool(
                                check["row_sum_from_x"] >= 0
                                and check["row_sum_from_x"] == np.floor(check["row_sum_from_x"])
                            ),
                        }
                    )
                    check_rows.append(check)
                plate_stats[plate] = {
                    "source_cells": int(store["X"].attrs["shape"][0]),
                    "source_full_filter_cells": int(source_row_filter.sum()),
                    "retained_cells": observed_cells,
                    "retained_samples": 95,
                    "pseudobulks": 95 * 50,
                }
                del matrix, reshaped_matrix, cell_counts, reshaped_cells
    except Exception:
        # A partial store cannot be scientifically interpreted and is explicitly
        # regenerable from the verified official H5AD inputs.
        del counts_store
        del group
        if store_path.exists():
            shutil.rmtree(store_path)
        raise

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(result_dir / "pseudobulk_sample_metadata.csv", index=False)
    cell_counts_frame = metadata[
        ["plate", "sample", "cell_line_id", "intervention_id", "is_control", "n_cells"]
    ].copy()
    cell_counts_frame.to_csv(result_dir / "cell_count_manifest.csv", index=False)
    checks = pd.DataFrame(check_rows)
    checks.to_csv(result_dir / "source_cell_spot_checks.csv", index=False)

    expected_core = pd.read_csv(root / "results/cgc_tahoe_0b/selected_replicate_core.csv")
    expected_dmso = pd.read_csv(root / "results/cgc_tahoe_0b/dmso_manifest.csv")
    treatment = metadata[~metadata["is_control"]]
    observed_treatment = treatment.set_index(["plate", "cell_line_id", "intervention_id"])["n_cells"]
    mismatches: list[dict[str, Any]] = []
    for row in expected_core.itertuples(index=False):
        for plate in PLATES:
            observed = int(observed_treatment.loc[(plate, row.cell_line_id, row.intervention_id)])
            expected_value = int(getattr(row, f"{plate}_n_cells_trt"))
            if observed != expected_value:
                mismatches.append(
                    {"plate": plate, "cell_line_id": row.cell_line_id, "intervention_id": row.intervention_id, "observed": observed, "expected": expected_value}
                )
    observed_control = metadata[metadata["is_control"]].set_index(["plate", "sample", "cell_line_id"])["n_cells"]
    for row in expected_dmso.itertuples(index=False):
        observed = int(observed_control.loc[(row.plate, row.sample, row.cell_line_id)])
        if observed != int(row.n_control_cells):
            mismatches.append(
                {"plate": row.plate, "sample": row.sample, "cell_line_id": row.cell_line_id, "observed": observed, "expected": int(row.n_control_cells)}
            )
    if mismatches:
        pd.DataFrame(mismatches).to_csv(result_dir / "cell_count_mismatches.csv", index=False)
        raise RuntimeError(f"Cell-count audit failed for {len(mismatches)} groups")
    mismatch_path = result_dir / "cell_count_mismatches.csv"
    if mismatch_path.exists():
        mismatch_path.unlink()
    if len(checks) != 2 * SPOT_CHECK_COUNT or not checks["integer_nonnegative"].all() or not checks["pseudobulk_contains_spot_count"].all():
        raise RuntimeError("Random source cell/count spot checks failed")

    manifest = {
        "phase": "CGC-0I",
        "created_at": _now(),
        "source": "official Arc plate-specific raw-count H5AD",
        "source_hash_manifest": "results/cgc_tahoe_0i/raw_source_hashes.json",
        "frozen_core": "results/cgc_tahoe_0b/selected_replicate_core.csv",
        "plates": list(PLATES),
        "contexts": len(contexts),
        "interventions": len(interventions),
        "dmso_wells_per_plate": 2,
        "retained_samples_per_plate": 95,
        "pseudobulks_per_plate": 4_750,
        "common_treatment_conditions_per_plate": 50 * 93,
        "plate_stats": plate_stats,
        "raw_count_array": "results/cgc_tahoe_0i/pseudobulk_raw_counts.zarr/raw_counts",
        "raw_count_shape": [2, 95, 50, 62_710],
        "accumulation_dtype": "float64",
        "storage_dtype": "float32",
        "storage_exact_integer_gate": "all pseudobulk values < 2^24",
        "maximum_pseudobulk_gene_count": max_count,
        "dense_cell_gene_materialized": False,
        "chunk_rows": CHUNK_ROWS,
        "cell_count_audit": "PASS",
        "cell_count_mismatches": 0,
        "random_cell_count_spot_checks": len(checks),
        "spot_check_seed": SPOT_CHECK_SEED,
        "spot_check_status": "PASS",
        "excluded_samples": "one non-shared treatment sample per plate; frozen 93-intervention core unchanged",
        "source_row_filter": "obs/pass_filter == 'full'; exactly reproduces frozen 0B cell-count manifest",
        "pid": os.getpid(),
    }
    _write_json(result_dir / "extraction_manifest.json", manifest)
    return manifest
