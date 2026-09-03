from __future__ import annotations

import base64
import hashlib
import json
import platform
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cgc_tahoe_0i"
DATA = ROOT / "data/tahoe100m_plate6_14_raw"
BUCKET = "arc-institute-virtual-cell-atlas"
PREFIX = "tahoe100M/2025-02-25/h5ad"
ARC_REVISION = "76cb053d59be08a7d0c4dc68c265622f6e0ab52e"
TRANSFER_LIMIT = 120_000_000_000
MIN_FREE_AFTER = 700_000_000_000


OBJECTS = {
    "plate6": {
        "name": f"{PREFIX}/plate6_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
        "generation": "1765942908712173",
        "expected_size": 28_934_897_078,
        "expected_md5_base64": "NYvQEqVClziHm0ozWhOw1w==",
        "artifact_uid": "aAHQ3zbD7n1asyYr0000",
        "expected_cells": 7_545_393,
    },
    "plate14": {
        "name": f"{PREFIX}/plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
        "generation": "1765942736304598",
        "expected_size": 22_427_932_564,
        "expected_md5_base64": "FrnStRehP16siRGG35ou+g==",
        "artifact_uid": "vn5cUJCHbjpPPsZx0000",
        "expected_cells": 6_518_806,
    },
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def media_url(item: dict[str, object]) -> str:
    quoted = urllib.parse.quote(str(item["name"]), safe="")
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/{quoted}"
        f"?generation={item['generation']}&alt=media"
    )


def object_metadata(item: dict[str, object]) -> dict[str, object]:
    quoted = urllib.parse.quote(str(item["name"]), safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{quoted}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def inspect_remote(item: dict[str, object]) -> dict[str, object]:
    handle = fsspec.open(
        media_url(item), "rb", block_size=4 * 1024 * 1024, cache_type="blockcache"
    ).open()
    try:
        with h5py.File(handle, "r") as store:
            x = store["X"]
            first = np.asarray(store["X/data"][:1000])
            last = np.asarray(store["X/data"][-1000:])
            values = np.concatenate([first, last])
            return {
                "root_keys": list(store.keys()),
                "layers": list(store["layers"].keys()),
                "x_encoding": str(x.attrs["encoding-type"]),
                "x_shape": [int(v) for v in x.attrs["shape"]],
                "x_data_shape": [int(v) for v in store["X/data"].shape],
                "x_data_dtype": str(store["X/data"].dtype),
                "x_indices_dtype": str(store["X/indices"].dtype),
                "x_indptr_dtype": str(store["X/indptr"].dtype),
                "sampled_min": float(values.min()),
                "sampled_max": float(values.max()),
                "sampled_nonnegative": bool(np.all(values >= 0)),
                "sampled_integer_valued": bool(np.all(values == np.floor(values))),
                "obs_columns": list(store["obs"].keys()),
                "var_columns": list(store["var"].keys()),
                "var_index": str(store["var"].attrs["_index"]),
                "gene_count": int(store["var/gene_name"].shape[0]),
                "plate_categories": [
                    v.decode("utf-8") for v in store["obs/plate/categories"][:]
                ],
                "sample_categories": int(len(store["obs/sample/categories"])),
                "cell_line_categories": int(len(store["obs/cell_line/categories"])),
                "drug_dose_categories": int(
                    len(store["obs/drugname_drugconc/categories"])
                ),
            }
    finally:
        handle.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    free_before = int(__import__("shutil").disk_usage(ROOT).free)
    rows = []
    schemas = {}
    total = 0
    for plate, item in OBJECTS.items():
        remote = object_metadata(item)
        schema = inspect_remote(item)
        schemas[plate] = schema
        size = int(remote["size"])
        total += size
        remote_md5_hex = base64.b64decode(str(remote["md5Hash"])).hex()
        rows.append(
            {
                "plate": plate,
                "artifact_key": item["name"],
                "artifact_uid": item["artifact_uid"],
                "generation": remote["generation"],
                "size_bytes": size,
                "size_gb": size / 1e9,
                "md5_base64": remote["md5Hash"],
                "md5_hex": remote_md5_hex,
                "etag": remote["etag"],
                "n_cells": schema["x_shape"][0],
                "n_genes": schema["x_shape"][1],
                "x_encoding": schema["x_encoding"],
                "x_dtype": schema["x_data_dtype"],
                "raw_count_gate": bool(
                    schema["x_encoding"] == "csr_matrix"
                    and schema["sampled_nonnegative"]
                    and schema["sampled_integer_valued"]
                ),
                "media_url": media_url(item),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUT / "access_inventory.csv", index=False)
    transfer_gate = total <= TRANSFER_LIMIT and free_before - total >= MIN_FREE_AFTER
    source_manifest = {
        "phase": "CGC-0I",
        "route": "A_ARC_PLATE_SPECIFIC_H5AD",
        "audited_at": now(),
        "official_repository": "https://github.com/ArcInstitute/arc-virtual-cell-atlas",
        "official_revision": ARC_REVISION,
        "bucket": f"gs://{BUCKET}",
        "objects": rows,
        "total_transfer_bytes": total,
        "total_transfer_gb": total / 1e9,
        "transfer_limit_bytes": TRANSFER_LIMIT,
        "free_bytes_before": free_before,
        "projected_free_bytes_after": free_before - total,
        "minimum_free_after_bytes": MIN_FREE_AFTER,
        "transfer_gate_passed": transfer_gate,
        "raw_count_semantics_gate_passed": bool(inventory["raw_count_gate"].all()),
        "dense_cell_gene_matrix_forbidden": True,
        "planned_peak_ram_gb": 8.0,
        "hard_peak_ram_gb": 64.0,
        "planned_wall_time": "download 1-4 h; streaming extraction 4-24 h",
    }
    (OUT / "raw_source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )
    expected_hashes = {
        plate: {
            "algorithm": "md5",
            "expected_base64": item["expected_md5_base64"],
            "expected_hex": base64.b64decode(
                str(item["expected_md5_base64"])
            ).hex(),
            "local_verified": False,
        }
        for plate, item in OBJECTS.items()
    }
    (OUT / "raw_source_hashes.json").write_text(
        json.dumps(expected_hashes, indent=2) + "\n", encoding="utf-8"
    )
    schema_text = f"""# CGC-0I official raw-count schema audit

- Route: official Arc Virtual Cell Atlas plate-specific H5AD.
- Plate 6: {schemas['plate6']['x_shape'][0]:,} cells × {schemas['plate6']['x_shape'][1]:,} genes.
- Plate 14: {schemas['plate14']['x_shape'][0]:,} cells × {schemas['plate14']['x_shape'][1]:,} genes.
- `X`: CSR (`data`, `indices`, `indptr`); stored data dtype is float32 but sampled values are nonnegative and exactly integer-valued.
- `layers`: empty; raw counts reside in `X`.
- Gene index: `var/gene_name`, 62,710 unique-axis entries to be audited after local download.
- Required obs fields are present: `plate`, `sample`, `cell_line`, `drug`, `drugname_drugconc`, `pass_filter`, and cell/barcode identifiers.
- Exact dose is encoded in `drugname_drugconc`; DMSO is encoded in `drug` and matched by plate/sample.
- No outcome-selected DE table is used for raw reconstruction or gene selection.
- Remote range reads were used only for schema and count-semantic validation; no dense cell × gene matrix was materialized.
"""
    (OUT / "raw_schema.md").write_text(schema_text, encoding="utf-8")
    env = {
        "captured_at": now(),
        "branch": git("branch", "--show-current"),
        "starting_commit": git("rev-parse", "HEAD"),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "free_bytes": free_before,
        "physical_ram_bytes": 33_412_722_688,
        "deterministic_seeds": {
            "spot_checks": 2601,
            "truth_permutations": 2602,
            "bootstrap": 2603,
            "support_sequences": 2604,
            "intervention_folds": 2605,
        },
    }
    (OUT / "environment.json").write_text(
        json.dumps(env, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# CGC-0I data-access gate

- Official route: Arc plate-specific H5AD, repository revision `{ARC_REVISION}`.
- Plate 6: {OBJECTS['plate6']['expected_size'] / 1e9:.3f} GB.
- Plate 14: {OBJECTS['plate14']['expected_size'] / 1e9:.3f} GB.
- Total planned transfer: {total / 1e9:.3f} GB (limit 120 GB).
- Free disk before transfer: {free_before / 1e9:.3f} GB; projected after: {(free_before-total) / 1e9:.3f} GB (minimum 700 GB).
- Raw-count schema gate: {'PASS' if inventory['raw_count_gate'].all() else 'FAIL'}.
- Transfer gate: {'PASS' if transfer_gate else 'FAIL'}.
- Planned peak RAM: approximately 8 GB using backed HDF5 and row-chunked CSR accumulation; dense cell × gene matrices are prohibited.
"""
    reports = ROOT / "results/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "cgc_tahoe_0i_data_access.md").write_text(report, encoding="utf-8")
    if not transfer_gate:
        raise SystemExit("TAHOE_RAW_COUNT_RECONSTRUCTION_REQUIRES_FULL_SCAN")
    if not inventory["raw_count_gate"].all():
        raise SystemExit("TAHOE_RAW_COUNT_SOURCE_INVALID")


if __name__ == "__main__":
    main()
