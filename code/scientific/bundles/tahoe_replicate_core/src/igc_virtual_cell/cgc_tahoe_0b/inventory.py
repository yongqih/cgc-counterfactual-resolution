from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = "tahoebio/Tahoe-100M"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}?blobs=true"
SELECTED_METADATA = (
    "metadata/sample_metadata.parquet",
    "metadata/cell_line_metadata.parquet",
    "metadata/drug_metadata.parquet",
    "metadata/gene_metadata.parquet",
)


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "IGC-Virtual-Cell/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_and_download_small_metadata(repo: Path) -> dict[str, Any]:
    accessed = datetime.now(timezone.utc).isoformat()
    api = json.loads(_fetch(API_URL))
    revision = api["sha"]
    siblings = {item["rfilename"]: item for item in api["siblings"]}
    missing = [item for item in SELECTED_METADATA if item not in siblings]
    if missing:
        raise RuntimeError(f"Official metadata files are absent: {missing}")

    de_files = [
        item
        for name, item in siblings.items()
        if name.startswith("metadata/pseudobulk_differential_expression/")
        and name.endswith(".parquet")
    ]
    obs = siblings["metadata/obs_metadata.parquet"]
    source_revision = {
        "repository": REPOSITORY,
        "revision": revision,
        "accessed_utc": accessed,
        "api_url": API_URL,
        "last_modified": api.get("lastModified"),
        "license": api.get("cardData", {}).get("license"),
        "official_metadata_file_count": sum(name.startswith("metadata/") for name in siblings),
        "pseudobulk_de_shards": len(de_files),
        "pseudobulk_de_total_bytes": sum(int(item.get("size", 0)) for item in de_files),
        "obs_metadata_bytes": int(obs.get("size", 0)),
    }
    results = repo / "results/cgc_tahoe_0b"
    data = repo / "data/tahoe100m_metadata"
    results.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    (results / "source_revision.json").write_text(
        json.dumps(source_revision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    before = shutil.disk_usage(repo).free
    files: list[dict[str, Any]] = []
    for remote_path in SELECTED_METADATA:
        metadata = siblings[remote_path]
        size = int(metadata.get("size", 0))
        if size > 10_000_000_000:
            raise RuntimeError(f"Hard download gate exceeded by {remote_path}: {size}")
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{remote_path}?download=true"
        destination = data / Path(remote_path).name
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(_fetch(url))
        if temporary.stat().st_size != size:
            raise RuntimeError(
                f"Size mismatch for {remote_path}: {temporary.stat().st_size} != {size}"
            )
        os.replace(temporary, destination)
        observed_hash = _sha256(destination)
        expected_hash = (metadata.get("lfs") or {}).get("sha256", "")
        if expected_hash and observed_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {remote_path}")
        files.append(
            {
                "remote_path": remote_path,
                "url": url,
                "size_bytes": size,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "local_repo_path": destination.relative_to(repo).as_posix(),
                "downloaded": True,
            }
        )
    after = shutil.disk_usage(repo).free
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "accessed_utc": accessed,
        "free_disk_before_bytes": before,
        "free_disk_after_bytes": after,
        "downloaded_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "explicitly_not_downloaded": {
            "metadata/obs_metadata.parquet": int(obs.get("size", 0)),
            "metadata/pseudobulk_differential_expression": source_revision[
                "pseudobulk_de_total_bytes"
            ],
            "full_expression_atlas": True,
        },
    }
    (results / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
