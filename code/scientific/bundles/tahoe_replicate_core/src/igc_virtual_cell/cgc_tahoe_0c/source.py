"""Freeze and verify official Tahoe sources for CGC-SUPPORT-0C."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from igc_virtual_cell.cgc_tahoe_0c.extraction import (
    MAX_AUTHORIZED_TRANSFER_BYTES,
    REPOSITORY,
    REVISION,
    USER_AGENT,
    official_de_files,
    source_url,
    write_json,
)
from igc_virtual_cell.cgc_tahoe_0b.analysis import HTTPRangeReader


README_PATH = "README.md"
PAPER_DOI = "10.1101/2025.02.20.639398"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    data_dir = root / "data/tahoe100m_plate6_14_core"
    result_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    readme_url = (
        f"https://huggingface.co/datasets/{REPOSITORY}/raw/{REVISION}/{README_PATH}"
    )
    readme = _read_url(readme_url)
    readme_text = readme.decode("utf-8")
    control_statement = (
        "DMSO_TF marks vehicle controls, use DMSO_TF along with plate to get plate matched controls"
    )
    if control_statement not in readme_text:
        raise RuntimeError("TAHOE_DE_ESTIMATOR_NOT_VERIFIED: control statement absent")

    files = official_de_files()
    probe = files[0]
    reader = HTTPRangeReader(source_url(probe["path"]), probe["size_bytes"])
    parquet = pq.ParquetFile(reader)
    schema_columns = parquet.schema_arrow.names
    required = {
        "gene_name",
        "log2FoldChange",
        "plate",
        "n_cells_trt",
        "n_cells_ctrl",
        "Cell_ID_Cellosaur",
        "drug",
        "concentration",
        "concentration_unit",
    }
    if not required.issubset(schema_columns):
        raise RuntimeError("TAHOE_DE_ESTIMATOR_NOT_VERIFIED: official DE schema incomplete")

    biorxiv_api = f"https://api.biorxiv.org/details/biorxiv/{PAPER_DOI}"
    paper_payload = json.loads(_read_url(biorxiv_api))
    versions = paper_payload.get("collection", [])
    if not versions:
        raise RuntimeError("TAHOE_DE_ESTIMATOR_NOT_VERIFIED: paper record absent")

    frozen_paths = {
        "selected_core": root / "results/cgc_tahoe_0b/selected_replicate_core.csv",
        "verdict": root / "results/cgc_tahoe_0b/verdict.json",
        "source_revision": root / "results/cgc_tahoe_0b/source_revision.json",
        "replicate_definition": root / "results/cgc_tahoe_0b/replicate_definition.json",
    }
    disk = shutil.disk_usage(root)
    manifest: dict[str, Any] = {
        "phase": "CGC-SUPPORT-0C",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "official_dataset_url": f"https://huggingface.co/datasets/{REPOSITORY}",
        "official_readme_url": readme_url,
        "official_readme_sha256": hashlib.sha256(readme).hexdigest(),
        "paper_doi": PAPER_DOI,
        "paper_versions": [
            {"version": item["version"], "date": item["date"], "jatsxml": item["jatsxml"]}
            for item in versions
        ],
        "estimator_verified": True,
        "estimator": "official plate-resolved treatment-versus-control log2FoldChange",
        "estimator_evidence": [
            "The official repository labels the release pseudobulk_differential_expression.",
            "Its schema contains log2FoldChange, plate, n_cells_trt, and n_cells_ctrl.",
            "The official README identifies DMSO_TF plus plate as the plate-matched vehicle control.",
        ],
        "plate_pooling": False,
        "alternative_estimator_recomputed": False,
        "official_de_schema": schema_columns,
        "probe_shard": probe,
        "probe_footer_bytes": reader.bytes_transferred,
        "full_de_release_bytes": sum(item["size_bytes"] for item in files),
        "full_de_release_downloaded": False,
        "authorized_transfer_bytes": MAX_AUTHORIZED_TRANSFER_BYTES,
        "free_disk_before_bytes": disk.free,
        "frozen_0b_hashes": {
            name: {"path": str(path.relative_to(root)), "sha256": _hash(path)}
            for name, path in frozen_paths.items()
        },
        "model_training_calls": 0,
        "virtual_cell_checkpoint_calls": 0,
    }
    write_json(result_dir / "source_manifest.json", manifest)
    return manifest
