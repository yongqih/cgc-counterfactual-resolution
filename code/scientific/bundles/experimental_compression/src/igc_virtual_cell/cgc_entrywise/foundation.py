from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr


GENE_READ_CHUNK = 1_024
PRECISION_SEED = 202_608_226
PRECISION_CHECKS = 96
PRECISION_RELATIVE_TOLERANCE = 1e-4


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_information(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "results/cgc_entrywise_compression/ENTRYWISE_INFORMATION_SET.json").read_text(encoding="utf-8")
    )


def _cache_paths(root: Path) -> dict[str, Path]:
    cache = root / "data/cgc_entrywise_cache"
    return {
        "cache": cache,
        "same6": cache / "same_plate6_float32.npy",
        "same14": cache / "same_plate14_float32.npy",
        "cross": cache / "cross_plate6_plate14_float32.npy",
        "sums": cache / "sample_gene_sums_float64.npy",
        "manifest": root / "results/cgc_entrywise_compression/ENTRYWISE_FOUNDATION_MANIFEST.json",
        "precision": root / "results/cgc_entrywise_compression/ENTRYWISE_GRAM_PRECISION.csv",
    }


def _write_precision(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_foundation(root: Path, source_root: Path | None = None, force: bool = False) -> dict[str, Any]:
    root = root.resolve()
    information = _load_information(root)
    source_root = Path(source_root or information["source_root_runtime"]).resolve()
    paths = _cache_paths(root)
    required = [paths["same6"], paths["same14"], paths["cross"], paths["sums"]]
    if not force and paths["manifest"].exists() and all(path.exists() for path in required):
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("status") == "PASS":
            return manifest
    paths["cache"].mkdir(parents=True, exist_ok=True)

    store_path = source_root / information["response_store_logical_path"]
    indices_path = source_root / information["gene_indices_logical_path"]
    gene_indices = np.load(indices_path).astype(np.int64)
    if len(gene_indices) != information["gene_count"] or not np.all(np.diff(gene_indices) > 0):
        raise RuntimeError("G_PRIMARY index axis changed")
    group = zarr.open_group(store_path, mode="r")
    response = group[information["response_array"]]
    if list(response.shape) != information["response_shape_full"]:
        raise RuntimeError("Response shape changed")
    if list(group.attrs["contexts"]) != information["contexts"]:
        raise RuntimeError("Context axis changed")
    if list(group.attrs["interventions"]) != information["interventions"]:
        raise RuntimeError("Intervention axis changed")

    gene_count = len(gene_indices)
    values = np.empty((2, 50 * 93, gene_count), dtype=np.float32)
    selected_position = 0
    for start in range(0, response.shape[-1], GENE_READ_CHUNK):
        stop = min(response.shape[-1], start + GENE_READ_CHUNK)
        left = int(np.searchsorted(gene_indices, start, side="left"))
        right = int(np.searchsorted(gene_indices, stop, side="left"))
        if right == left:
            continue
        local = gene_indices[left:right] - start
        block = np.asarray(response[:, :, :, start:stop], dtype=np.float32)
        picked = block[..., local].reshape(2, 50 * 93, right - left)
        values[:, :, selected_position : selected_position + right - left] = picked
        selected_position += right - left
        if stop % (8 * GENE_READ_CHUNK) == 0 or stop == response.shape[-1]:
            print(f"entrywise foundation genes {stop}/{response.shape[-1]}", flush=True)
    if selected_position != gene_count or not np.isfinite(values).all():
        raise RuntimeError("Selected response loading failed")

    sums = values.sum(axis=2, dtype=np.float64)
    np.save(paths["sums"], sums)
    gram_specs = (
        ("same6", values[0], values[0].T),
        ("same14", values[1], values[1].T),
        ("cross", values[0], values[1].T),
    )
    grams: dict[str, np.ndarray] = {}
    for name, left, right in gram_specs:
        print(f"entrywise foundation matmul {name}", flush=True)
        product = np.asarray(left @ right, dtype=np.float32)
        np.save(paths[name], product)
        grams[name] = product

    rng = np.random.default_rng(PRECISION_SEED)
    rows: list[dict[str, Any]] = []
    max_relative = 0.0
    for check in range(PRECISION_CHECKS):
        row = int(rng.integers(0, 4_650))
        col = int(rng.integers(0, 4_650))
        kind = ("same6", "same14", "cross")[check % 3]
        if kind == "same6":
            expected = float(np.dot(values[0, row].astype(np.float64), values[0, col].astype(np.float64)))
        elif kind == "same14":
            expected = float(np.dot(values[1, row].astype(np.float64), values[1, col].astype(np.float64)))
        else:
            expected = float(np.dot(values[0, row].astype(np.float64), values[1, col].astype(np.float64)))
        observed = float(grams[kind][row, col])
        relative = abs(observed - expected) / max(abs(expected), 1e-12)
        max_relative = max(max_relative, relative)
        rows.append(
            {
                "check": check,
                "gram": kind,
                "row": row,
                "column": col,
                "float32_blas": observed,
                "float64_direct": expected,
                "relative_error": relative,
            }
        )
    _write_precision(paths["precision"], rows)
    if max_relative > PRECISION_RELATIVE_TOLERANCE:
        raise RuntimeError("ENTRYWISE_GRAM_PRECISION_FAIL")

    del values
    manifest = {
        "phase": "CGC-EC-2",
        "created_at": _now(),
        "status": "PASS",
        "source_store": str(store_path),
        "source_array": information["response_array"],
        "source_indices": str(indices_path),
        "source_indices_sha256": _sha256(indices_path),
        "selected_shape": [2, 4_650, gene_count],
        "sample_order": "context-major then intervention-major",
        "gene_count": gene_count,
        "gram_dtype": "float32 BLAS with frozen float64 direct-dot precision gate",
        "sample_sum_dtype": "float64",
        "precision_checks": PRECISION_CHECKS,
        "precision_seed": PRECISION_SEED,
        "precision_max_relative_error": max_relative,
        "precision_tolerance": PRECISION_RELATIVE_TOLERANCE,
        "gpu_used": False,
        "threading": os.environ.get("OPENBLAS_NUM_THREADS", "runtime default"),
        "cache_files": {
            key: {
                "path": str(paths[key].relative_to(root)),
                "size_bytes": paths[key].stat().st_size,
                "sha256": _sha256(paths[key]),
            }
            for key in ("same6", "same14", "cross", "sums")
        },
        "raw_h5ad_read": False,
        "response_matrix_persisted": False,
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def load_foundation(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    root = root.resolve()
    paths = _cache_paths(root)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("Foundation gate did not pass")
    same6 = np.load(paths["same6"], mmap_mode="r")
    same14 = np.load(paths["same14"], mmap_mode="r")
    cross = np.load(paths["cross"], mmap_mode="r")
    sums = np.load(paths["sums"], mmap_mode="r")
    if same6.shape != (4_650, 4_650) or same14.shape != same6.shape or cross.shape != same6.shape:
        raise RuntimeError("Foundation cache shape changed")
    return same6, same14, cross, sums, manifest
