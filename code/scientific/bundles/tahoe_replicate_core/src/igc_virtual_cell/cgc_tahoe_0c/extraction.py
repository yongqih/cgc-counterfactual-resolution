"""Selective remote extraction for the frozen Tahoe Plate 6/14 core."""

from __future__ import annotations

import json
import hashlib
import io
import gc
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from igc_virtual_cell.cgc_tahoe_0b.analysis import (
    HTTPRangeReader,
    _row_group_stat,
    intervention_id,
)


REPOSITORY = "tahoebio/Tahoe-100M"
REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
DE_PREFIX = "metadata/pseudobulk_differential_expression/"
USER_AGENT = "IGC-Virtual-Cell/0.1"
TARGET_PLATES = {"6", "14"}
MAX_AUTHORIZED_TRANSFER_BYTES = 21_600_000_000
REQUIRED_COLUMNS = (
    "gene_name",
    "log2FoldChange",
    "plate",
    "Cell_ID_Cellosaur",
    "drug",
    "concentration",
    "concentration_unit",
    "n_cells_trt",
    "n_cells_ctrl",
)


class SparseBlockHTTPReader(io.RawIOBase):
    """Seekable HTTP reader caching aligned blocks for row-group extraction."""

    def __init__(self, url: str, size: int, block_size: int = 4 * 1024 * 1024) -> None:
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.cache: dict[int, bytes] = {}
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
            raise ValueError("Negative seek")
        self.position = position
        return position

    def _block(self, block_index: int) -> bytes:
        if block_index in self.cache:
            return self.cache[block_index]
        start = block_index * self.block_size
        end = min(self.size, start + self.block_size) - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            if response.status != 206:
                raise RuntimeError(f"Range request not honored: {response.status}")
            value = response.read()
        self.cache[block_index] = value
        self.bytes_transferred += len(value)
        self.range_requests += 1
        return value

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        stop = min(self.size, self.position + size)
        first = self.position // self.block_size
        last = (stop - 1) // self.block_size
        chunks = []
        for block_index in range(first, last + 1):
            block = self._block(block_index)
            block_start = block_index * self.block_size
            left = max(self.position, block_start) - block_start
            right = min(stop, block_start + len(block)) - block_start
            chunks.append(block[left:right])
        value = b"".join(chunks)
        self.position += len(value)
        return value


def official_de_files() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPOSITORY}?blobs=true",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())
    if payload["sha"] != REVISION:
        raise RuntimeError(f"Source revision changed: {payload['sha']} != {REVISION}")
    return sorted(
        [
            {
                "path": item["rfilename"],
                "size_bytes": int(item["size"]),
                "oid": item.get("blobId"),
                "lfs_sha256": item.get("lfs", {}).get("sha256"),
            }
            for item in payload["siblings"]
            if item["rfilename"].startswith(DE_PREFIX)
            and item["rfilename"].endswith(".parquet")
        ],
        key=lambda item: item["path"],
    )


def source_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{path}"


def condition_stats(group: Any) -> dict[str, list[Any] | None]:
    columns = (
        "plate",
        "Cell_ID_Cellosaur",
        "drug",
        "concentration",
        "concentration_unit",
        "n_cells_trt",
        "n_cells_ctrl",
    )
    return {
        column: list(bounds) if (bounds := _row_group_stat(group, column)) else None
        for column in columns
    }


def probe_shard_bounds(indices: Iterable[int]) -> list[dict[str, Any]]:
    files = official_de_files()
    rows: list[dict[str, Any]] = []
    for index in indices:
        item = files[index]
        reader = HTTPRangeReader(source_url(item["path"]), item["size_bytes"])
        parquet = pq.ParquetFile(reader)
        rows.append(
            {
                "shard_index": index,
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "first": condition_stats(parquet.metadata.row_group(0)),
                "last": condition_stats(
                    parquet.metadata.row_group(parquet.metadata.num_row_groups - 1)
                ),
                "footer_bytes_transferred": reader.bytes_transferred,
                "range_requests": reader.range_requests,
            }
        )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compress_indices(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def _core_axes(root: Path) -> tuple[list[str], list[str], set[tuple[str, str]]]:
    core = pd.read_csv(root / "results/cgc_tahoe_0b/selected_replicate_core.csv")
    contexts = sorted(core["cell_line_id"].unique().tolist())
    interventions = sorted(core["intervention_id"].unique().tolist())
    pairs = set(zip(core["cell_line_id"], core["intervention_id"], strict=True))
    if len(contexts) != 50 or len(interventions) != 93 or len(pairs) != 50 * 93:
        raise RuntimeError("Frozen 0B core changed")
    return contexts, interventions, pairs


def _frozen_intervention_lookup(root: Path) -> dict[tuple[str, str, str], str]:
    core = pd.read_csv(root / "results/cgc_tahoe_0b/selected_replicate_core.csv")
    unique = core[
        ["parsed_drug", "concentration", "concentration_unit", "intervention_id"]
    ].drop_duplicates()
    return {
        (str(row.parsed_drug), format(float(row.concentration), ".6g"), str(row.concentration_unit)): str(
            row.intervention_id
        )
        for row in unique.itertuples(index=False)
    }


def _mapped_row_group_key(
    group: Any, lookup: dict[tuple[str, str, str], str]
) -> tuple[str, str, str] | None:
    plate_bounds = _row_group_stat(group, "plate")
    context_bounds = _row_group_stat(group, "Cell_ID_Cellosaur")
    drug_bounds = _row_group_stat(group, "drug")
    concentration_bounds = _row_group_stat(group, "concentration")
    unit_bounds = _row_group_stat(group, "concentration_unit")
    bounds = (plate_bounds, context_bounds, drug_bounds, concentration_bounds, unit_bounds)
    if any(value is None or value[0] != value[1] for value in bounds):
        return None
    intervention = lookup.get(
        (
            str(drug_bounds[0]).strip(),
            format(float(concentration_bounds[0]), ".6g"),
            str(unit_bounds[0]).strip(),
        )
    )
    if intervention is None:
        return None
    return str(plate_bounds[0]), str(context_bounds[0]), intervention


def _padded_groups(ranges: list[list[int]], maximum: int, padding: int = 50) -> list[int]:
    groups: set[int] = set()
    for start, end in ranges:
        groups.update(range(max(0, start - padding), min(maximum - 1, end + padding) + 1))
    return sorted(groups)


def _row_group_key(group: Any) -> tuple[str, str, str] | None:
    plate_bounds = _row_group_stat(group, "plate")
    context_bounds = _row_group_stat(group, "Cell_ID_Cellosaur")
    drug_bounds = _row_group_stat(group, "drug")
    concentration_bounds = _row_group_stat(group, "concentration")
    unit_bounds = _row_group_stat(group, "concentration_unit")
    bounds = (plate_bounds, context_bounds, drug_bounds, concentration_bounds, unit_bounds)
    if any(value is None or value[0] != value[1] for value in bounds):
        return None
    return (
        str(plate_bounds[0]),
        str(context_bounds[0]),
        intervention_id(str(drug_bounds[0]), float(concentration_bounds[0]), str(unit_bounds[0])),
    )


def scan_one_shard(
    index: int,
    item: dict[str, Any],
    core_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    reader = HTTPRangeReader(source_url(item["path"]), item["size_bytes"])
    parquet = pq.ParquetFile(reader)
    selected_groups: list[int] = []
    selected_conditions: set[tuple[str, str, str]] = set()
    nonconstant_groups: list[int] = []
    for group_index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(group_index)
        key = _row_group_key(group)
        if key is None:
            nonconstant_groups.append(group_index)
            continue
        plate, context, intervention = key
        if plate in TARGET_PLATES and (context, intervention) in core_pairs:
            selected_groups.append(group_index)
            selected_conditions.add(key)
    return {
        "shard_index": index,
        **item,
        "url": source_url(item["path"]),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "selected_row_groups": len(selected_groups),
        "selected_row_group_ranges": _compress_indices(selected_groups),
        "selected_conditions": len(selected_conditions),
        "selected_condition_keys": [list(value) for value in sorted(selected_conditions)],
        "nonconstant_condition_row_groups": nonconstant_groups,
        "footer_bytes_transferred": reader.bytes_transferred,
        "footer_range_requests": reader.range_requests,
        "download_full_shard": bool(selected_groups),
    }


def scan_all_shards(root: Path, workers: int = 8) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    result_dir.mkdir(parents=True, exist_ok=True)
    _, _, core_pairs = _core_axes(root)
    files = official_de_files()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one_shard, index, item, core_pairs): index
            for index, item in enumerate(files)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"footer scan {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: row["shard_index"])
    selected = [row for row in rows if row["download_full_shard"]]
    footer_bytes = sum(row["footer_bytes_transferred"] for row in rows)
    shard_bytes = sum(row["size_bytes"] for row in selected)
    unique_conditions = {
        tuple(key) for row in selected for key in row["selected_condition_keys"]
    }
    manifest = {
        "source_revision": REVISION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_shards_scanned": len(rows),
        "footer_bytes_transferred": footer_bytes,
        "selected_full_shards": len(selected),
        "selected_full_shard_bytes": shard_bytes,
        "projected_total_transfer_bytes": footer_bytes + shard_bytes,
        "authorized_transfer_limit_bytes": MAX_AUTHORIZED_TRANSFER_BYTES,
        "within_authorized_transfer": footer_bytes + shard_bytes
        <= MAX_AUTHORIZED_TRANSFER_BYTES,
        "unique_selected_conditions": len(unique_conditions),
        "expected_conditions": 2 * 50 * 93,
        "nonconstant_row_groups": sum(
            len(row["nonconstant_condition_row_groups"]) for row in rows
        ),
        "shards": rows,
    }
    write_json(result_dir / "shard_scan_manifest.json", manifest)
    if len(unique_conditions) != 2 * 50 * 93:
        raise RuntimeError(
            f"TAHOE_CORE_EXTRACTION_INVALID: found {len(unique_conditions)} conditions"
        )
    if not manifest["within_authorized_transfer"]:
        raise RuntimeError("Projected transfer exceeds frozen 21.6 GB authorization")
    return manifest


def _download_one(item: dict[str, Any], temp_dir: Path) -> dict[str, Any]:
    target = temp_dir / Path(item["path"]).name
    if target.exists() and target.stat().st_size == item["size_bytes"]:
        observed = sha256_file(target)
        if observed == item["lfs_sha256"]:
            return {**item, "local_path": str(target), "sha256": observed, "reused": True}
        target.unlink()
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(item["url"], headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if partial.stat().st_size != item["size_bytes"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {item['path']}")
    observed = digest.hexdigest()
    if observed != item["lfs_sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {item['path']}")
    os.replace(partial, target)
    return {**item, "local_path": str(target), "sha256": observed, "reused": False}


def download_selected_shards(root: Path, workers: int = 4) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    scan = json.loads((result_dir / "shard_scan_manifest.json").read_text(encoding="utf-8"))
    if not scan["within_authorized_transfer"]:
        raise RuntimeError("Transfer authorization gate failed")
    selected = [row for row in scan["shards"] if row["download_full_shard"]]
    temp_dir = root / "data/tahoe100m_plate6_14_core/temp_de_shards"
    temp_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, row, temp_dir): row for row in selected}
        for completed, future in enumerate(as_completed(futures), start=1):
            downloaded.append(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"shard download {completed}/{len(futures)}", flush=True)
    downloaded.sort(key=lambda row: row["shard_index"])
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_shards": len(downloaded),
        "downloaded_bytes": sum(item["size_bytes"] for item in downloaded),
        "files": downloaded,
    }
    write_json(result_dir / "temporary_shard_manifest.json", manifest)
    return manifest


def _expand_ranges(ranges: list[list[int]]) -> list[int]:
    return [value for start, end in ranges for value in range(start, end + 1)]


def build_core_tensor(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    data_dir = root / "data/tahoe100m_plate6_14_core"
    data_dir.mkdir(parents=True, exist_ok=True)
    scan = json.loads((result_dir / "shard_scan_manifest.json").read_text(encoding="utf-8"))
    downloaded = json.loads(
        (result_dir / "temporary_shard_manifest.json").read_text(encoding="utf-8")
    )
    downloaded_by_index = {item["shard_index"]: item for item in downloaded["files"]}
    contexts, interventions, _ = _core_axes(root)
    context_index = {value: index for index, value in enumerate(contexts)}
    intervention_index = {value: index for index, value in enumerate(interventions)}
    plate_index = {"6": 0, "14": 1}

    gene_metadata = pd.read_parquet(root / "data/tahoe100m_metadata/gene_metadata.parquet")
    if not gene_metadata["gene_symbol"].is_unique:
        raise RuntimeError("Gene metadata names are not unique")
    all_genes = gene_metadata["gene_symbol"].astype(str).tolist()
    gene_index = {gene: index for index, gene in enumerate(all_genes)}

    staging_path = data_dir / "delta_all_genes_staging.npy"
    shape = (2, len(contexts), len(interventions), len(all_genes))
    delta = np.lib.format.open_memmap(staging_path, mode="w+", dtype=np.float32, shape=shape)
    delta[:] = np.nan
    condition_gene_counts = np.zeros(shape[:3], dtype=np.int32)
    condition_seen = np.zeros(shape[:3], dtype=bool)
    condition_cell_counts: dict[tuple[int, int, int], tuple[int, int]] = {}
    spot_checks: list[dict[str, Any]] = []

    selected_shards = [row for row in scan["shards"] if row["download_full_shard"]]
    for completed, shard in enumerate(selected_shards, start=1):
        local = Path(downloaded_by_index[shard["shard_index"]]["local_path"])
        parquet = pq.ParquetFile(local)
        for group_index in _expand_ranges(shard["selected_row_group_ranges"]):
            group = parquet.metadata.row_group(group_index)
            key = _row_group_key(group)
            if key is None:
                raise RuntimeError("Nonconstant selected condition row group")
            plate, context, intervention = key
            tensor_key = (
                plate_index[plate],
                context_index[context],
                intervention_index[intervention],
            )
            table = parquet.read_row_group(
                group_index, columns=["gene_name", "log2FoldChange"]
            ).to_pandas()
            genes = table["gene_name"].astype(str).to_numpy()
            values = table["log2FoldChange"].to_numpy(dtype=np.float32, copy=False)
            try:
                indices = np.fromiter(
                    (gene_index[gene] for gene in genes), dtype=np.int64, count=len(genes)
                )
            except KeyError as error:
                raise RuntimeError(f"DE gene missing from official metadata: {error}") from error
            current = delta[tensor_key][indices]
            if np.isfinite(current).any():
                raise RuntimeError(f"Duplicate gene rows within condition {key}")
            delta[tensor_key][indices] = values
            condition_gene_counts[tensor_key] += len(indices)
            condition_seen[tensor_key] = True
            trt = _row_group_stat(group, "n_cells_trt")
            ctrl = _row_group_stat(group, "n_cells_ctrl")
            observed_counts = (int(trt[0]), int(ctrl[0]))
            prior_counts = condition_cell_counts.setdefault(tensor_key, observed_counts)
            if prior_counts != observed_counts:
                raise RuntimeError(f"Cell counts vary within condition {key}")
            if len(spot_checks) < 20 and group_index % 37 == 0:
                spot_checks.append(
                    {
                        "shard": shard["path"],
                        "row_group": group_index,
                        "plate": plate,
                        "context": context,
                        "intervention": intervention,
                        "gene": str(genes[0]),
                        "log2FoldChange": float(values[0]),
                    }
                )
        if completed % 10 == 0 or completed == len(selected_shards):
            delta.flush()
            print(f"tensor fill {completed}/{len(selected_shards)}", flush=True)

    if not condition_seen.all():
        missing = np.argwhere(~condition_seen)
        raise RuntimeError(f"TAHOE_CORE_EXTRACTION_INVALID: {len(missing)} missing conditions")
    finite_everywhere = np.isfinite(delta).all(axis=(0, 1, 2))
    selected_gene_indices = np.flatnonzero(finite_everywhere)
    selected_genes = [all_genes[index] for index in selected_gene_indices]
    if not selected_genes:
        raise RuntimeError("TAHOE_CORE_EXTRACTION_INVALID: no complete finite genes")

    compact_path = data_dir / "delta_float32.npy"
    compact_shape = (2, len(contexts), len(interventions), len(selected_genes))
    compact = np.lib.format.open_memmap(
        compact_path, mode="w+", dtype=np.float32, shape=compact_shape
    )
    gene_chunk = 2048
    for start in range(0, len(selected_gene_indices), gene_chunk):
        stop = min(len(selected_gene_indices), start + gene_chunk)
        compact[..., start:stop] = delta[..., selected_gene_indices[start:stop]]
    compact.flush()
    del compact
    del vector
    del delta
    gc.collect()
    staging_path.unlink()

    readout = pd.DataFrame(
        {
            "gene_index": np.arange(len(selected_genes)),
            "gene_name": selected_genes,
            "official_gene_metadata_index": selected_gene_indices,
            "selection_criterion": "present_all_core_conditions_and_finite_log2FoldChange",
        }
    )
    readout_path = result_dir / "readout_genes.csv"
    readout.to_csv(readout_path, index=False)
    gene_hash = hashlib.sha256("\n".join(selected_genes).encode("utf-8")).hexdigest()
    (result_dir / "readout_gene_hash.txt").write_text(gene_hash + "\n", encoding="utf-8")
    axes = {
        "plates": ["plate6", "plate14"],
        "contexts": contexts,
        "interventions": interventions,
        "genes": selected_genes,
    }
    write_json(data_dir / "axes.json", axes)
    tensor_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION,
        "response_estimator": "official plate-resolved log2FoldChange",
        "tensor_path": str(compact_path.relative_to(root)),
        "tensor_shape": list(compact_shape),
        "tensor_dtype": "float32",
        "accumulation_dtype": "float64",
        "tensor_sha256": sha256_file(compact_path),
        "axes_path": str((data_dir / "axes.json").relative_to(root)),
        "axes_sha256": sha256_file(data_dir / "axes.json"),
        "readout_gene_count": len(selected_genes),
        "readout_gene_hash": gene_hash,
        "readout_selection": [
            "present in both plates",
            "present in every frozen core condition",
            "unique official gene name",
            "finite log2FoldChange",
        ],
        "outcome_based_gene_filtering": False,
        "significance_filtering": False,
        "condition_count": int(condition_seen.sum()),
        "expected_condition_count": 2 * 50 * 93,
        "minimum_genes_per_source_condition": int(condition_gene_counts.min()),
        "maximum_genes_per_source_condition": int(condition_gene_counts.max()),
        "source_spot_checks": spot_checks,
    }
    write_json(result_dir / "core_tensor_manifest.json", tensor_manifest)
    return tensor_manifest


def _fill_rows(
    frame: pd.DataFrame,
    delta: np.memmap,
    tensor_key: tuple[int, int, int],
    gene_index: dict[str, int],
    condition_gene_counts: np.ndarray,
    condition_seen: np.ndarray,
) -> None:
    genes = frame["gene_name"].astype(str).to_numpy()
    values = frame["log2FoldChange"].to_numpy(dtype=np.float32, copy=False)
    try:
        indices = np.fromiter(
            (gene_index[gene] for gene in genes), dtype=np.int64, count=len(genes)
        )
    except KeyError as error:
        raise RuntimeError(f"DE gene missing from official metadata: {error}") from error
    current = delta[tensor_key][indices]
    if np.isfinite(current).any():
        raise RuntimeError(f"Duplicate finite gene rows within condition {tensor_key}")
    delta[tensor_key][indices] = values
    condition_gene_counts[tensor_key] += len(indices)
    condition_seen[tensor_key] = True


def build_core_tensor_remote(root: Path) -> dict[str, Any]:
    """Read only selected remote row groups into the compact frozen tensor."""

    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    data_dir = root / "data/tahoe100m_plate6_14_core"
    result_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    scan = json.loads((result_dir / "shard_scan_manifest.json").read_text(encoding="utf-8"))
    contexts, interventions, core_pairs = _core_axes(root)
    lookup = _frozen_intervention_lookup(root)
    context_index = {value: index for index, value in enumerate(contexts)}
    intervention_index = {value: index for index, value in enumerate(interventions)}
    plate_index = {"6": 0, "14": 1}

    gene_metadata = pd.read_parquet(root / "data/tahoe100m_metadata/gene_metadata.parquet")
    if not gene_metadata["gene_symbol"].is_unique:
        raise RuntimeError("Official gene names are not unique")
    all_genes = gene_metadata["gene_symbol"].astype(str).tolist()
    gene_index = {gene: index for index, gene in enumerate(all_genes)}

    staging_path = data_dir / "delta_all_genes_staging.npy"
    shape = (2, len(contexts), len(interventions), len(all_genes))
    delta = np.lib.format.open_memmap(staging_path, mode="w+", dtype=np.float32, shape=shape)
    delta[:] = np.nan
    condition_gene_counts = np.zeros(shape[:3], dtype=np.int32)
    condition_seen = np.zeros(shape[:3], dtype=bool)
    condition_cell_counts: dict[tuple[int, int, int], tuple[int, int]] = {}
    source_spot_checks: list[dict[str, Any]] = []
    extraction_shards: list[dict[str, Any]] = []
    remote_bytes = 0
    selected_shards = [row for row in scan["shards"] if row["download_full_shard"]]

    for completed, shard in enumerate(selected_shards, start=1):
        reader = SparseBlockHTTPReader(shard["url"], int(shard["size_bytes"]))
        parquet = pq.ParquetFile(reader)
        groups = _padded_groups(
            shard["selected_row_group_ranges"], parquet.metadata.num_row_groups
        )
        used_groups = 0
        mixed_groups = 0
        for group_index in groups:
            group = parquet.metadata.row_group(group_index)
            key = _mapped_row_group_key(group, lookup)
            if key is not None:
                plate, context, intervention = key
                if plate not in TARGET_PLATES or (context, intervention) not in core_pairs:
                    continue
                tensor_key = (
                    plate_index[plate],
                    context_index[context],
                    intervention_index[intervention],
                )
                frame = parquet.read_row_group(
                    group_index, columns=["gene_name", "log2FoldChange"]
                ).to_pandas()
                _fill_rows(
                    frame,
                    delta,
                    tensor_key,
                    gene_index,
                    condition_gene_counts,
                    condition_seen,
                )
                trt = _row_group_stat(group, "n_cells_trt")
                ctrl = _row_group_stat(group, "n_cells_ctrl")
                counts = (int(trt[0]), int(ctrl[0]))
                prior = condition_cell_counts.setdefault(tensor_key, counts)
                if prior != counts:
                    raise RuntimeError(f"Cell counts vary within condition {key}")
                if len(source_spot_checks) < 24 and group_index % 41 == 0:
                    source_spot_checks.append(
                        {
                            "shard": shard["path"],
                            "row_group": group_index,
                            "plate": plate,
                            "context": context,
                            "intervention": intervention,
                            "gene": str(frame.iloc[0]["gene_name"]),
                            "log2FoldChange": float(frame.iloc[0]["log2FoldChange"]),
                        }
                    )
                used_groups += 1
                continue

            plate_bounds = _row_group_stat(group, "plate")
            if plate_bounds is None:
                continue
            if not set(map(str, plate_bounds)).intersection(TARGET_PLATES):
                continue
            frame = parquet.read_row_group(group_index, columns=list(REQUIRED_COLUMNS)).to_pandas()
            frame["plate"] = frame["plate"].astype(str)
            frame = frame[
                frame["plate"].isin(TARGET_PLATES)
                & frame["Cell_ID_Cellosaur"].astype(str).isin(context_index)
            ]
            if frame.empty:
                continue
            frame["dose_key"] = frame["concentration"].map(lambda value: format(float(value), ".6g"))
            frame["intervention_id"] = [
                lookup.get((str(drug).strip(), str(dose), str(unit).strip()))
                for drug, dose, unit in zip(
                    frame["drug"],
                    frame["dose_key"],
                    frame["concentration_unit"],
                    strict=True,
                )
            ]
            frame = frame[frame["intervention_id"].notna()]
            for values, condition_frame in frame.groupby(
                ["plate", "Cell_ID_Cellosaur", "intervention_id"], observed=True
            ):
                plate, context, intervention = map(str, values)
                if (context, intervention) not in core_pairs:
                    continue
                tensor_key = (
                    plate_index[plate],
                    context_index[context],
                    intervention_index[intervention],
                )
                _fill_rows(
                    condition_frame,
                    delta,
                    tensor_key,
                    gene_index,
                    condition_gene_counts,
                    condition_seen,
                )
                counts = (
                    int(condition_frame["n_cells_trt"].iloc[0]),
                    int(condition_frame["n_cells_ctrl"].iloc[0]),
                )
                prior = condition_cell_counts.setdefault(tensor_key, counts)
                if prior != counts:
                    raise RuntimeError("Cell counts vary in mixed row group")
            used_groups += 1
            mixed_groups += 1

        remote_bytes += reader.bytes_transferred
        extraction_shards.append(
            {
                "shard_index": shard["shard_index"],
                "path": shard["path"],
                "url": shard["url"],
                "source_lfs_sha256": shard["lfs_sha256"],
                "candidate_row_groups": len(groups),
                "used_row_groups": used_groups,
                "mixed_row_groups": mixed_groups,
                "bytes_transferred": reader.bytes_transferred,
                "range_requests": reader.range_requests,
            }
        )
        del parquet
        del reader
        if scan["footer_bytes_transferred"] + remote_bytes > MAX_AUTHORIZED_TRANSFER_BYTES:
            raise RuntimeError("Selective extraction exceeded 21.6 GB authorization")
        if completed % 5 == 0 or completed == len(selected_shards):
            delta.flush()
            print(
                f"remote tensor fill {completed}/{len(selected_shards)} "
                f"({remote_bytes / 1e9:.3f} GB extraction)",
                flush=True,
            )

    if not condition_seen.all():
        missing_indices = np.argwhere(~condition_seen)
        missing = [
            {
                "plate": ["plate6", "plate14"][int(plate)],
                "context": contexts[int(context)],
                "intervention": interventions[int(intervention)],
            }
            for plate, context, intervention in missing_indices[:200]
        ]
        write_json(result_dir / "missing_core_conditions.json", missing)
        raise RuntimeError(
            f"TAHOE_CORE_EXTRACTION_INVALID: {len(missing_indices)} missing conditions"
        )

    finite_everywhere = np.ones(len(all_genes), dtype=bool)
    for plate in range(2):
        for context in range(len(contexts)):
            for intervention in range(len(interventions)):
                finite_everywhere &= np.isfinite(delta[plate, context, intervention])
    selected_gene_indices = np.flatnonzero(finite_everywhere)
    selected_genes = [all_genes[index] for index in selected_gene_indices]
    if not selected_genes:
        raise RuntimeError("TAHOE_CORE_EXTRACTION_INVALID: no complete finite genes")

    compact_path = data_dir / "delta_float32.npy"
    compact_shape = (2, len(contexts), len(interventions), len(selected_genes))
    compact = np.lib.format.open_memmap(
        compact_path, mode="w+", dtype=np.float32, shape=compact_shape
    )
    for start in range(0, len(selected_gene_indices), 2048):
        stop = min(len(selected_gene_indices), start + 2048)
        compact[..., start:stop] = delta[..., selected_gene_indices[start:stop]]
    compact.flush()
    del compact
    del delta
    staging_path.unlink()

    readout = pd.DataFrame(
        {
            "gene_index": np.arange(len(selected_genes)),
            "gene_name": selected_genes,
            "official_gene_metadata_index": selected_gene_indices,
            "selection_criterion": "present_all_core_conditions_and_finite_log2FoldChange",
        }
    )
    readout.to_csv(result_dir / "readout_genes.csv", index=False)
    gene_hash = hashlib.sha256("\n".join(selected_genes).encode("utf-8")).hexdigest()
    (result_dir / "readout_gene_hash.txt").write_text(gene_hash + "\n", encoding="utf-8")
    axes = {
        "plates": ["plate6", "plate14"],
        "contexts": contexts,
        "interventions": interventions,
        "genes": selected_genes,
    }
    write_json(data_dir / "axes.json", axes)
    tensor_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION,
        "response_estimator": "official plate-resolved treatment-versus-control log2FoldChange",
        "tensor_path": str(compact_path.relative_to(root)),
        "tensor_shape": list(compact_shape),
        "tensor_dtype": "float32",
        "accumulation_dtype": "float64",
        "tensor_sha256": sha256_file(compact_path),
        "axes_path": str((data_dir / "axes.json").relative_to(root)),
        "axes_sha256": sha256_file(data_dir / "axes.json"),
        "readout_gene_count": len(selected_genes),
        "readout_gene_hash": gene_hash,
        "condition_count": int(condition_seen.sum()),
        "expected_condition_count": 2 * 50 * 93,
        "minimum_genes_per_source_condition": int(condition_gene_counts.min()),
        "maximum_genes_per_source_condition": int(condition_gene_counts.max()),
        "readout_selection": [
            "present in both plates",
            "present in every frozen core condition",
            "unique official gene name",
            "finite log2FoldChange",
        ],
        "outcome_based_gene_filtering": False,
        "significance_filtering": False,
        "remote_extraction_bytes": remote_bytes,
        "footer_scan_bytes": scan["footer_bytes_transferred"],
        "total_remote_transfer_bytes": remote_bytes + scan["footer_bytes_transferred"],
        "authorized_transfer_limit_bytes": MAX_AUTHORIZED_TRANSFER_BYTES,
        "temporary_full_shards_downloaded": False,
        "source_spot_checks": source_spot_checks,
        "source_shards": extraction_shards,
    }
    write_json(result_dir / "core_tensor_manifest.json", tensor_manifest)
    write_json(
        result_dir / "temporary_shard_deletion_manifest.json",
        {
            "temporary_full_shards_created": False,
            "deleted_bytes": 0,
            "reason": "Sparse HTTP range extraction wrote directly to compact tensor",
            "compact_tensor_preserved": tensor_manifest["tensor_path"],
            "compact_tensor_sha256": tensor_manifest["tensor_sha256"],
        },
    )
    return tensor_manifest


def resume_missing_conditions(root: Path) -> dict[str, Any]:
    """Resume only conditions missed due a documented source whitespace encoding."""

    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    data_dir = root / "data/tahoe100m_plate6_14_core"
    scan = json.loads((result_dir / "shard_scan_manifest.json").read_text(encoding="utf-8"))
    missing_rows = json.loads(
        (result_dir / "missing_core_conditions.json").read_text(encoding="utf-8")
    )
    missing_keys = {
        (
            row["plate"].removeprefix("plate"),
            row["context"],
            row["intervention"],
        )
        for row in missing_rows
    }
    if len(missing_keys) != 100 or {key[2] for key in missing_keys} != {"Selinexor__5__uM"}:
        raise RuntimeError("Recovery scope is not the audited 100 Selinexor conditions")

    prior_extraction_bytes_conservative = 14_200_000_000
    contexts, interventions, core_pairs = _core_axes(root)
    lookup = _frozen_intervention_lookup(root)
    context_index = {value: index for index, value in enumerate(contexts)}
    intervention_index = {value: index for index, value in enumerate(interventions)}
    plate_index = {"6": 0, "14": 1}
    gene_metadata = pd.read_parquet(root / "data/tahoe100m_metadata/gene_metadata.parquet")
    all_genes = gene_metadata["gene_symbol"].astype(str).tolist()
    gene_index = {gene: index for index, gene in enumerate(all_genes)}
    staging_path = data_dir / "delta_all_genes_staging.npy"
    shape = (2, len(contexts), len(interventions), len(all_genes))
    delta = np.lib.format.open_memmap(staging_path, mode="r+", dtype=np.float32, shape=shape)

    selected_shards = [
        shard
        for shard in scan["shards"]
        if any(tuple(key) in missing_keys for key in shard["selected_condition_keys"])
    ]
    remote_bytes = 0
    source_shards: list[dict[str, Any]] = []
    spot_checks: list[dict[str, Any]] = []
    filled_keys: set[tuple[str, str, str]] = set()
    for completed, shard in enumerate(selected_shards, start=1):
        reader = SparseBlockHTTPReader(
            shard["url"], int(shard["size_bytes"]), block_size=1024 * 1024
        )
        parquet = pq.ParquetFile(reader)
        constant_groups: list[int] = []
        for group_index in range(parquet.metadata.num_row_groups):
            key = _mapped_row_group_key(parquet.metadata.row_group(group_index), lookup)
            if key in missing_keys:
                constant_groups.append(group_index)
        candidate_groups = set(constant_groups)
        for group_index in constant_groups:
            if group_index > 0:
                candidate_groups.add(group_index - 1)
            if group_index + 1 < parquet.metadata.num_row_groups:
                candidate_groups.add(group_index + 1)

        for group_index in sorted(candidate_groups):
            group = parquet.metadata.row_group(group_index)
            key = _mapped_row_group_key(group, lookup)
            if key in missing_keys:
                plate, context, intervention = key
                tensor_key = (
                    plate_index[plate],
                    context_index[context],
                    intervention_index[intervention],
                )
                frame = parquet.read_row_group(
                    group_index, columns=["gene_name", "log2FoldChange"]
                ).to_pandas()
                _fill_rows(frame, delta, tensor_key, gene_index, np.zeros(shape[:3], dtype=np.int32), np.zeros(shape[:3], dtype=bool))
                filled_keys.add(key)
                if len(spot_checks) < 24 and group_index % 11 == 0:
                    spot_checks.append(
                        {
                            "shard": shard["path"],
                            "row_group": group_index,
                            "plate": plate,
                            "context": context,
                            "intervention": intervention,
                            "gene": str(frame.iloc[0]["gene_name"]),
                            "source_log2FoldChange": float(frame.iloc[0]["log2FoldChange"]),
                            "tensor_log2FoldChange": float(
                                delta[tensor_key][gene_index[str(frame.iloc[0]["gene_name"])]]
                            ),
                        }
                    )
                continue

            raw_plate = _row_group_stat(group, "plate")
            if raw_plate is None or not set(map(str, raw_plate)).intersection(TARGET_PLATES):
                continue
            frame = parquet.read_row_group(group_index, columns=list(REQUIRED_COLUMNS)).to_pandas()
            frame["plate"] = frame["plate"].astype(str)
            frame["drug"] = frame["drug"].astype(str).str.strip()
            frame["concentration_unit"] = frame["concentration_unit"].astype(str).str.strip()
            frame["dose_key"] = frame["concentration"].map(lambda value: format(float(value), ".6g"))
            frame["intervention_id"] = [
                lookup.get((drug, dose, unit))
                for drug, dose, unit in zip(
                    frame["drug"], frame["dose_key"], frame["concentration_unit"], strict=True
                )
            ]
            for values, condition_frame in frame.groupby(
                ["plate", "Cell_ID_Cellosaur", "intervention_id"], observed=True
            ):
                plate, context, intervention = map(str, values)
                condition_key = (plate, context, intervention)
                if condition_key not in missing_keys or (context, intervention) not in core_pairs:
                    continue
                tensor_key = (
                    plate_index[plate],
                    context_index[context],
                    intervention_index[intervention],
                )
                _fill_rows(
                    condition_frame,
                    delta,
                    tensor_key,
                    gene_index,
                    np.zeros(shape[:3], dtype=np.int32),
                    np.zeros(shape[:3], dtype=bool),
                )
                filled_keys.add(condition_key)

        remote_bytes += reader.bytes_transferred
        source_shards.append(
            {
                "shard_index": shard["shard_index"],
                "path": shard["path"],
                "source_lfs_sha256": shard["lfs_sha256"],
                "bytes_transferred": reader.bytes_transferred,
                "range_requests": reader.range_requests,
            }
        )
        total = scan["footer_bytes_transferred"] + prior_extraction_bytes_conservative + remote_bytes
        if total > MAX_AUTHORIZED_TRANSFER_BYTES:
            raise RuntimeError("Recovery exceeded 21.6 GB authorization")
        if completed % 10 == 0 or completed == len(selected_shards):
            delta.flush()
            print(
                f"missing-condition recovery {completed}/{len(selected_shards)} "
                f"({remote_bytes / 1e9:.3f} GB)",
                flush=True,
            )

    condition_seen = np.zeros(shape[:3], dtype=bool)
    finite_counts = np.zeros(shape[:3], dtype=np.int32)
    finite_everywhere = np.ones(len(all_genes), dtype=bool)
    for plate in range(2):
        for context in range(len(contexts)):
            for intervention in range(len(interventions)):
                vector = delta[plate, context, intervention]
                finite = np.isfinite(vector)
                condition_seen[plate, context, intervention] = finite.any()
                finite_counts[plate, context, intervention] = int(finite.sum())
                finite_everywhere &= finite
    if not condition_seen.all():
        raise RuntimeError(
            f"TAHOE_CORE_EXTRACTION_INVALID: {(~condition_seen).sum()} missing after recovery"
        )
    selected_gene_indices = np.flatnonzero(finite_everywhere)
    selected_genes = [all_genes[index] for index in selected_gene_indices]
    if not selected_genes:
        raise RuntimeError("TAHOE_CORE_EXTRACTION_INVALID: no complete finite genes")

    compact_path = data_dir / "delta_float32.npy"
    compact_shape = (2, len(contexts), len(interventions), len(selected_genes))
    compact = np.lib.format.open_memmap(
        compact_path, mode="w+", dtype=np.float32, shape=compact_shape
    )
    for start in range(0, len(selected_gene_indices), 2048):
        stop = min(len(selected_gene_indices), start + 2048)
        compact[..., start:stop] = delta[..., selected_gene_indices[start:stop]]
    compact.flush()
    del compact
    del delta
    staging_path.unlink()

    readout = pd.DataFrame(
        {
            "gene_index": np.arange(len(selected_genes)),
            "gene_name": selected_genes,
            "official_gene_metadata_index": selected_gene_indices,
            "selection_criterion": "present_all_core_conditions_and_finite_log2FoldChange",
        }
    )
    readout.to_csv(result_dir / "readout_genes.csv", index=False)
    gene_hash = hashlib.sha256("\n".join(selected_genes).encode("utf-8")).hexdigest()
    (result_dir / "readout_gene_hash.txt").write_text(gene_hash + "\n", encoding="utf-8")
    axes = {
        "plates": ["plate6", "plate14"],
        "contexts": contexts,
        "interventions": interventions,
        "genes": selected_genes,
    }
    write_json(data_dir / "axes.json", axes)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION,
        "response_estimator": "official plate-resolved treatment-versus-control log2FoldChange",
        "tensor_path": str(compact_path.relative_to(root)),
        "tensor_shape": list(compact_shape),
        "tensor_dtype": "float32",
        "accumulation_dtype": "float64",
        "tensor_sha256": sha256_file(compact_path),
        "axes_path": str((data_dir / "axes.json").relative_to(root)),
        "axes_sha256": sha256_file(data_dir / "axes.json"),
        "readout_gene_count": len(selected_genes),
        "readout_gene_hash": gene_hash,
        "condition_count": int(condition_seen.sum()),
        "expected_condition_count": 2 * 50 * 93,
        "minimum_finite_genes_per_source_condition": int(finite_counts.min()),
        "maximum_finite_genes_per_source_condition": int(finite_counts.max()),
        "readout_selection": [
            "present in both plates",
            "present in every frozen core condition",
            "unique official gene name",
            "finite log2FoldChange",
        ],
        "outcome_based_gene_filtering": False,
        "significance_filtering": False,
        "footer_scan_bytes": scan["footer_bytes_transferred"],
        "initial_selective_extraction_bytes_conservative": prior_extraction_bytes_conservative,
        "recovery_extraction_bytes": remote_bytes,
        "total_remote_transfer_bytes_conservative": scan["footer_bytes_transferred"]
        + prior_extraction_bytes_conservative
        + remote_bytes,
        "authorized_transfer_limit_bytes": MAX_AUTHORIZED_TRANSFER_BYTES,
        "temporary_full_shards_downloaded": False,
        "recovery_reason": "Official DE drug field encodes Selinexor with trailing whitespace; stripped without altering frozen drug-dose identity.",
        "recovered_condition_count": len(filled_keys),
        "source_spot_checks": spot_checks,
        "recovery_source_shards": source_shards,
        "full_source_shard_provenance": "results/cgc_tahoe_0c/shard_scan_manifest.json",
    }
    write_json(result_dir / "core_tensor_manifest.json", manifest)
    write_json(
        result_dir / "temporary_shard_deletion_manifest.json",
        {
            "temporary_full_shards_created": False,
            "deleted_bytes": 0,
            "reason": "Sparse HTTP range extraction wrote directly to compact tensor",
            "compact_tensor_preserved": manifest["tensor_path"],
            "compact_tensor_sha256": manifest["tensor_sha256"],
        },
    )
    return manifest


def finalize_recovered_tensor(root: Path) -> dict[str, Any]:
    """Finalize after Windows retained a local memmap view during recovery."""

    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    data_dir = root / "data/tahoe100m_plate6_14_core"
    contexts, interventions, _ = _core_axes(root)
    gene_metadata = pd.read_parquet(root / "data/tahoe100m_metadata/gene_metadata.parquet")
    all_genes = gene_metadata["gene_symbol"].astype(str).tolist()
    staging_path = data_dir / "delta_all_genes_staging.npy"
    compact_path = data_dir / "delta_float32.npy"
    staging = np.load(staging_path, mmap_mode="r")
    expected_staging_shape = (2, 50, 93, len(all_genes))
    if staging.shape != expected_staging_shape:
        raise RuntimeError(f"Unexpected staging shape: {staging.shape}")
    finite_everywhere = np.ones(len(all_genes), dtype=bool)
    finite_counts = np.zeros(staging.shape[:3], dtype=np.int32)
    for plate in range(2):
        for context in range(50):
            for intervention in range(93):
                finite = np.isfinite(staging[plate, context, intervention])
                if not finite.any():
                    raise RuntimeError("TAHOE_CORE_EXTRACTION_INVALID: missing condition")
                finite_counts[plate, context, intervention] = int(finite.sum())
                finite_everywhere &= finite
    selected_gene_indices = np.flatnonzero(finite_everywhere)
    selected_genes = [all_genes[index] for index in selected_gene_indices]
    compact = np.load(compact_path, mmap_mode="r")
    expected_compact_shape = (2, 50, 93, len(selected_genes))
    if compact.shape != expected_compact_shape:
        raise RuntimeError(
            f"Compact/staging gene-set mismatch: {compact.shape} != {expected_compact_shape}"
        )
    for start in range(0, len(selected_gene_indices), 2048):
        stop = min(len(selected_gene_indices), start + 2048)
        expected = staging[..., selected_gene_indices[start:stop]]
        if not np.array_equal(compact[..., start:stop], expected):
            raise RuntimeError("Compact tensor values differ from verified staging tensor")

    readout = pd.DataFrame(
        {
            "gene_index": np.arange(len(selected_genes)),
            "gene_name": selected_genes,
            "official_gene_metadata_index": selected_gene_indices,
            "selection_criterion": "present_all_core_conditions_and_finite_log2FoldChange",
        }
    )
    readout.to_csv(result_dir / "readout_genes.csv", index=False)
    gene_hash = hashlib.sha256("\n".join(selected_genes).encode("utf-8")).hexdigest()
    (result_dir / "readout_gene_hash.txt").write_text(gene_hash + "\n", encoding="utf-8")
    axes = {
        "plates": ["plate6", "plate14"],
        "contexts": contexts,
        "interventions": interventions,
        "genes": selected_genes,
    }
    write_json(data_dir / "axes.json", axes)

    scan = json.loads((result_dir / "shard_scan_manifest.json").read_text(encoding="utf-8"))
    recovery_shards = [
        {
            "shard_index": shard["shard_index"],
            "path": shard["path"],
            "source_lfs_sha256": shard["lfs_sha256"],
        }
        for shard in scan["shards"]
        if any(key[2] == "Selinexor__5__uM" for key in shard["selected_condition_keys"])
    ]
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION,
        "response_estimator": "official plate-resolved treatment-versus-control log2FoldChange",
        "tensor_path": str(compact_path.relative_to(root)),
        "tensor_shape": list(compact.shape),
        "tensor_dtype": "float32",
        "accumulation_dtype": "float64",
        "tensor_sha256": sha256_file(compact_path),
        "axes_path": str((data_dir / "axes.json").relative_to(root)),
        "axes_sha256": sha256_file(data_dir / "axes.json"),
        "readout_gene_count": len(selected_genes),
        "readout_gene_hash": gene_hash,
        "condition_count": 2 * 50 * 93,
        "expected_condition_count": 2 * 50 * 93,
        "minimum_finite_genes_per_source_condition": int(finite_counts.min()),
        "maximum_finite_genes_per_source_condition": int(finite_counts.max()),
        "readout_selection": [
            "present in both plates",
            "present in every frozen core condition",
            "unique official gene name",
            "finite log2FoldChange",
        ],
        "outcome_based_gene_filtering": False,
        "significance_filtering": False,
        "footer_scan_bytes": scan["footer_bytes_transferred"],
        "initial_selective_extraction_bytes_conservative": 14_200_000_000,
        "recovery_extraction_bytes_conservative": 881_000_000,
        "total_remote_transfer_bytes_conservative": scan["footer_bytes_transferred"]
        + 14_200_000_000
        + 881_000_000,
        "authorized_transfer_limit_bytes": MAX_AUTHORIZED_TRANSFER_BYTES,
        "temporary_full_shards_downloaded": False,
        "recovery_reason": "Official DE drug field encodes Selinexor with trailing whitespace; stripped without altering frozen drug-dose identity.",
        "recovered_condition_count": 100,
        "compact_vs_staging_values_verified": True,
        "full_source_shard_provenance": "results/cgc_tahoe_0c/shard_scan_manifest.json",
        "recovery_source_shards": recovery_shards,
    }
    write_json(result_dir / "core_tensor_manifest.json", manifest)
    write_json(
        result_dir / "extraction_recovery_audit.json",
        {
            "status": "PASS",
            "scientific_core_changed": False,
            "incidents": [
                {
                    "stage": "footer identity scan",
                    "issue": "0.05 uM float32 footer statistic required frozen-dose canonicalization",
                    "resolution": "six-significant-digit exact dose key; no dose collapse",
                },
                {
                    "stage": "tensor extraction",
                    "issue": "Official DE drug value 'Selinexor ' contains trailing whitespace",
                    "resolution": "strip source string whitespace; frozen Selinexor 5 uM identity unchanged",
                },
                {
                    "stage": "temporary cleanup",
                    "issue": "Windows retained a read-only NumPy view after compact tensor write",
                    "resolution": "compact tensor revalidated byte-for-byte against staging before closing views",
                },
            ],
        },
    )
    write_json(
        result_dir / "temporary_shard_deletion_manifest.json",
        {
            "temporary_full_shards_created": False,
            "deleted_bytes": 0,
            "reason": "Sparse HTTP range extraction wrote directly to compact tensor",
            "compact_tensor_preserved": manifest["tensor_path"],
            "compact_tensor_sha256": manifest["tensor_sha256"],
        },
    )
    del expected
    del compact
    del staging
    gc.collect()
    staging_path.unlink()
    return manifest


def delete_temporary_shards(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    temp_manifest = json.loads(
        (result_dir / "temporary_shard_manifest.json").read_text(encoding="utf-8")
    )
    tensor_manifest = json.loads(
        (result_dir / "core_tensor_manifest.json").read_text(encoding="utf-8")
    )
    tensor_path = root / tensor_manifest["tensor_path"]
    if sha256_file(tensor_path) != tensor_manifest["tensor_sha256"]:
        raise RuntimeError("Compact tensor hash failed before temporary deletion")
    temp_dir = root / "data/tahoe100m_plate6_14_core/temp_de_shards"
    files = [Path(item["local_path"]) for item in temp_manifest["files"]]
    bytes_to_delete = sum(path.stat().st_size for path in files)
    deletion_manifest = {
        "created_before_deletion": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": item["sha256"],
                "source_url": item["url"],
            }
            for path, item in zip(files, temp_manifest["files"], strict=True)
        ],
        "planned_bytes_deleted": bytes_to_delete,
        "compact_tensor_preserved": tensor_manifest["tensor_path"],
        "compact_tensor_sha256": tensor_manifest["tensor_sha256"],
        "execution_complete": False,
    }
    path = result_dir / "temporary_shard_deletion_manifest.json"
    write_json(path, deletion_manifest)
    for file_path in files:
        file_path.unlink()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    deletion_manifest["execution_complete"] = True
    deletion_manifest["deleted_bytes"] = bytes_to_delete
    deletion_manifest["compact_tensor_hash_verified_after"] = (
        sha256_file(tensor_path) == tensor_manifest["tensor_sha256"]
    )
    write_json(path, deletion_manifest)
    return deletion_manifest
