"""Range-extract the frozen matched-340 ARCHS4 representation.

Only selected GSE207049 GSM columns and the frozen 10,110-gene intersection are
materialized locally.  Per-sample library size is nevertheless computed over
all 67,186 ARCHS4 genes, as preregistered.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.audit_lea_archs4_coverage import ARCHS4_URL, HTTPRangeFile, remote_head


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "level2_lea" / "archs4_robustness"
CACHE = ROOT / "data" / "level2_lea_archs4_matched"
EXPECTED_ETAG = '"350f7e2f1096e77b8b8da50c6a91c509-9130"'
EXPECTED_GENE_ROWS = 67_186
EXPECTED_GSMS = 1_026
EXPECTED_MATCHED_GENES = 10_110


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def log1p_cpm(counts: np.ndarray, gene_indices: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply the frozen per-sample transform; the denominator uses every gene."""
    library = float(np.asarray(counts).sum(dtype=np.uint64))
    if not np.isfinite(library) or library <= 0:
        raise RuntimeError(f"Invalid full-gene library size: {library}")
    transformed = np.log1p(np.asarray(counts)[gene_indices].astype(np.float64) * (1_000_000.0 / library))
    if not np.isfinite(transformed).all():
        raise RuntimeError("Non-finite log1p-CPM")
    return transformed.astype(np.float32), library


def equal_version_mean(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise RuntimeError("Cannot aggregate an empty GSM group")
    return np.mean(vectors, axis=0, dtype=np.float64).astype(np.float32)


def _valid_shard(path: Path, gsms: list[str], positions: list[int]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            return (
                _decode(saved["gsms"]) == gsms
                and saved["positions"].astype(int).tolist() == positions
                and saved["values"].shape == (len(gsms), EXPECTED_MATCHED_GENES)
                and np.isfinite(saved["values"]).all()
                and (saved["library_sizes"] > 0).all()
            )
    except Exception:
        return False


def _worker(payload: dict[str, object]) -> dict[str, object]:
    worker = int(payload["worker"])
    rows = list(payload["rows"])
    gene_indices = np.asarray(payload["gene_indices"], dtype=np.int64)
    cache = Path(str(payload["cache"]))
    remote_size = int(payload["remote_size"])
    shard_size = int(payload["shard_size"])
    sessions = []
    pending: list[tuple[int, list[tuple[str, int]]]] = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        subset = rows[start : start + shard_size]
        path = cache / f"gsm_values_w{worker:02d}_{shard_index:03d}.npz"
        gsms = [str(gsm) for gsm, _ in subset]
        positions = [int(position) for _, position in subset]
        if not _valid_shard(path, gsms, positions):
            pending.append((shard_index, subset))
    if not pending:
        return {"worker": worker, "status": "CACHED", "sessions": []}

    started = time.perf_counter()
    reader = HTTPRangeFile(
        ARCHS4_URL,
        remote_size,
        block_size=256 * 1024,
        max_blocks=512,
    )
    peak_rss = 0
    try:
        import psutil

        process = psutil.Process(os.getpid())
    except ImportError:
        process = None
    with h5py.File(reader, "r", driver="fileobj") as handle:
        expression = handle["data/expression"]
        if expression.shape[0] != EXPECTED_GENE_ROWS:
            raise RuntimeError(f"ARCHS4 gene row drift: {expression.shape}")
        for shard_index, subset in pending:
            values = np.empty((len(subset), len(gene_indices)), dtype=np.float32)
            libraries = np.empty(len(subset), dtype=np.float64)
            for local_index, (_, position) in enumerate(subset):
                counts = np.asarray(expression[:, int(position)])
                transformed, library = log1p_cpm(counts, gene_indices)
                values[local_index] = transformed
                libraries[local_index] = library
                if process is not None:
                    peak_rss = max(peak_rss, int(process.memory_info().rss))
            path = cache / f"gsm_values_w{worker:02d}_{shard_index:03d}.npz"
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    gsms=np.asarray([gsm for gsm, _ in subset]),
                    positions=np.asarray([position for _, position in subset], dtype=np.int64),
                    library_sizes=libraries,
                    values=values,
                )
            os.replace(temporary, path)
            print(
                f"worker={worker} shard={shard_index} samples={len(subset)} "
                f"transferred_gb={reader.bytes_transferred / 1e9:.3f}",
                flush=True,
            )
    sessions.append(
        {
            "timestamp": now(),
            "worker": worker,
            "range_requests": reader.request_count,
            "bytes_transferred": reader.bytes_transferred,
            "peak_rss_bytes": peak_rss,
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    return {"worker": worker, "status": "COMPLETE", "sessions": sessions}


def scan_positions(target_gsms: set[str], head: dict[str, object]) -> tuple[dict[str, int], dict[str, object]]:
    path = CACHE / "ARCHS4_SELECTED_GSM_POSITIONS.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("etag") == head.get("etag") and set(saved.get("positions", {})) == target_gsms:
            return {gsm: int(position) for gsm, position in saved["positions"].items()}, saved["range_stats"]

    reader = HTTPRangeFile(ARCHS4_URL, int(head["content_length"]), block_size=4 * 1024 * 1024, max_blocks=64)
    positions: dict[str, int] = {}
    with h5py.File(reader, "r", driver="fileobj") as handle:
        series = handle["meta/samples/series_id"]
        accessions = handle["meta/samples/geo_accession"]
        step = int(series.chunks[0]) * 16
        for start in range(0, len(series), step):
            stop = min(len(series), start + step)
            study_relative = np.flatnonzero(series[start:stop] == b"GSE207049")
            if not len(study_relative):
                continue
            absolute = start + study_relative
            for position, gsm in zip(absolute.tolist(), _decode(accessions[absolute]), strict=True):
                if gsm in target_gsms:
                    positions[gsm] = int(position)
    if set(positions) != target_gsms:
        missing = sorted(target_gsms - set(positions))
        raise RuntimeError(f"Selected GSM position gate failed; missing={missing[:10]}")
    stats = {"range_requests": reader.request_count, "bytes_transferred": reader.bytes_transferred}
    path.write_text(
        json.dumps({"created_at": now(), "etag": head.get("etag"), "positions": positions, "range_stats": stats}, indent=2) + "\n",
        encoding="utf-8",
    )
    return positions, stats


def selected_gsms_and_indices() -> tuple[list[str], np.ndarray]:
    manifest = pd.read_csv(OUT / "ARCHS4_MATCHED_340_FOLD_MANIFEST.csv")
    manifest = manifest.loc[manifest["included_matched_340"]]
    target: set[str] = set()
    for column in ["ARCHS4_ETOH_GSM_group", "ARCHS4_DEX_GSM_group"]:
        for group in manifest[column].astype(str):
            target.update(part.strip() for part in group.split("|") if part.strip())
    if len(target) != EXPECTED_GSMS:
        raise RuntimeError(f"Frozen GSM count drift: {len(target)}")
    axis = pd.read_csv(OUT / "ARCHS4_GENE_AXIS_RECONCILIATION.csv")
    selected = axis.loc[axis["included_primary_axis"]].copy()
    if len(selected) != EXPECTED_MATCHED_GENES:
        raise RuntimeError(f"Frozen matched-gene count drift: {len(selected)}")
    return sorted(target), selected["ARCHS4_row_index"].to_numpy(dtype=np.int64)


def aggregate_representation(positions: dict[str, int], extraction_stats: dict[str, object]) -> Path:
    shard_paths = sorted(CACHE.glob("gsm_values_w*.npz"))
    values: dict[str, np.ndarray] = {}
    libraries: dict[str, float] = {}
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as saved:
            gsms = _decode(saved["gsms"])
            for index, gsm in enumerate(gsms):
                if gsm in values:
                    raise RuntimeError(f"GSM duplicated across extraction shards: {gsm}")
                values[gsm] = saved["values"][index].astype(np.float32, copy=True)
                libraries[gsm] = float(saved["library_sizes"][index])
    if set(values) != set(positions):
        raise RuntimeError(f"Extraction completeness failed: have={len(values)} expected={len(positions)}")

    manifest = pd.read_csv(OUT / "ARCHS4_MATCHED_340_FOLD_MANIFEST.csv")
    manifest = manifest.loc[manifest["included_matched_340"]].sort_values("historical_row_index")
    baseline = []
    treated = []
    for row in manifest.itertuples(index=False):
        etoh = [part.strip() for part in str(row.ARCHS4_ETOH_GSM_group).split("|")]
        dex = [part.strip() for part in str(row.ARCHS4_DEX_GSM_group).split("|")]
        baseline.append(equal_version_mean([values[gsm] for gsm in etoh]))
        treated.append(equal_version_mean([values[gsm] for gsm in dex]))
    axis = pd.read_csv(OUT / "ARCHS4_GENE_AXIS_RECONCILIATION.csv")
    genes = axis.loc[axis["included_primary_axis"], "historical_GeneID"].astype(str).to_numpy()
    path = CACHE / "ARCHS4_MATCHED_340_REPRESENTATION.npz"
    np.savez_compressed(
        path,
        baseline=np.asarray(baseline, dtype=np.float32),
        treated=np.asarray(treated, dtype=np.float32),
        folds=manifest["historical_fold"].to_numpy(dtype=np.int64),
        lines=np.asarray(manifest["lcl_id"].astype(str).tolist(), dtype=str),
        strata=np.asarray(manifest["ancestry"].astype(str).tolist(), dtype=str),
        genes=np.asarray(genes.tolist(), dtype=str),
    )
    library_frame = pd.DataFrame(
        {"gsm": sorted(libraries), "sample_position": [positions[gsm] for gsm in sorted(libraries)], "full_67186_gene_library_size": [libraries[gsm] for gsm in sorted(libraries)]}
    )
    library_frame.to_csv(OUT / "ARCHS4_SELECTED_GSM_LIBRARY_SIZES.csv", index=False)
    extraction_stats["representation_path"] = str(path)
    extraction_stats["representation_size_bytes"] = path.stat().st_size
    extraction_stats["shard_cache_size_bytes"] = sum(item.stat().st_size for item in shard_paths)
    extraction_stats["selected_gsms"] = len(values)
    extraction_stats["matched_genes"] = len(genes)
    extraction_stats["library_size_min"] = min(libraries.values())
    extraction_stats["library_size_max"] = max(libraries.values())
    (OUT / "ARCHS4_EXTRACTION_MANIFEST.json").write_text(json.dumps(extraction_stats, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    target_gsms, gene_indices = selected_gsms_and_indices()
    head = remote_head(ARCHS4_URL)
    if head.get("etag") != EXPECTED_ETAG:
        raise RuntimeError(f"ARCHS4 object identity changed: {head}")
    started = time.perf_counter()
    positions, scan_stats = scan_positions(set(target_gsms), head)
    ordered = sorted(positions.items(), key=lambda item: item[1])
    workers = max(1, min(args.workers, len(ordered)))
    partitions = np.array_split(np.asarray(ordered, dtype=object), workers)
    payloads = [
        {
            "worker": index,
            "rows": [(str(gsm), int(position)) for gsm, position in partition.tolist()],
            "gene_indices": gene_indices.tolist(),
            "cache": str(CACHE),
            "remote_size": int(head["content_length"]),
            "shard_size": args.shard_size,
        }
        for index, partition in enumerate(partitions)
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        worker_results = pool.map(_worker, payloads)
    sessions = [session for result in worker_results for session in result["sessions"]]
    prior_manifest_path = OUT / "ARCHS4_EXTRACTION_MANIFEST.json"
    prior = json.loads(prior_manifest_path.read_text(encoding="utf-8")) if prior_manifest_path.exists() else {}
    if not sessions and prior.get("expression_bytes_transferred"):
        historical_sessions = prior.get("worker_sessions", [])
        expression_requests = int(prior["expression_range_requests"])
        expression_bytes = int(prior["expression_bytes_transferred"])
        peak_rss = int(prior.get("peak_worker_rss_bytes", 0))
        initial_runtime = float(prior.get("initial_extraction_runtime_seconds", prior.get("runtime_seconds", 0.0)))
    else:
        historical_sessions = sessions
        expression_requests = sum(int(session["range_requests"]) for session in sessions)
        expression_bytes = sum(int(session["bytes_transferred"]) for session in sessions)
        peak_rss = max([int(session["peak_rss_bytes"]) for session in sessions] or [0])
        initial_runtime = time.perf_counter() - started
    current_runtime = time.perf_counter() - started
    stats = {
        "created_at": now(),
        "remote_url": ARCHS4_URL,
        "remote_head": head,
        "normalization": "natural log1p of CPM; library sum over all 67,186 genes",
        "aggregation": "equal arithmetic mean of all confidently assigned GSM versions within LCL-by-treatment",
        "position_scan": scan_stats,
        "worker_sessions": historical_sessions,
        "expression_range_requests": expression_requests,
        "expression_bytes_transferred": expression_bytes,
        "total_bytes_transferred_this_run": int(scan_stats["bytes_transferred"]) + expression_bytes,
        "peak_worker_rss_bytes": peak_rss,
        "runtime_seconds": initial_runtime,
        "initial_extraction_runtime_seconds": initial_runtime,
        "latest_cache_rebuild_runtime_seconds": current_runtime,
        "full_hdf5_downloaded": False,
        "raw_sequencing_downloaded": False,
    }
    path = aggregate_representation(positions, stats)
    print(json.dumps({"representation": str(path), **stats}, indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
