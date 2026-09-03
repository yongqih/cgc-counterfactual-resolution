"""Coverage-only gene universes and raw-count Tahoe response estimators."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


GENE_CHUNK = 1_024


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _hash_genes(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def construct_responses(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    extraction = json.loads((out / "extraction_manifest.json").read_text(encoding="utf-8"))
    if extraction["cell_count_audit"] != "PASS":
        raise RuntimeError("Extraction audit did not pass")
    metadata = pd.read_csv(out / "pseudobulk_sample_metadata.csv")
    genes = pd.read_csv(out / "gene_metadata_frozen.csv")
    gene_names = genes["gene_symbol"].astype(str).tolist()
    raw_group = zarr.open_group(out / "pseudobulk_raw_counts.zarr", mode="r")
    raw = raw_group["raw_counts"]
    if raw.shape != (2, 95, 50, 62_710):
        raise RuntimeError("Unexpected raw pseudobulk shape")

    response_path = out / "response_tensors.zarr"
    if response_path.exists():
        raise RuntimeError(f"Refusing to overwrite {response_path}")
    response_group = zarr.open_group(response_path, mode="w", zarr_format=2)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arrays = {}
    for name in (
        "delta_primary",
        "delta_sum_control",
        "delta_cell_weighted_control",
        "delta_log2fc",
    ):
        arrays[name] = response_group.create_array(
            name,
            shape=(2, 50, 93, len(gene_names)),
            chunks=(1, 10, 10, GENE_CHUNK),
            dtype="float32",
            compressor=compressor,
            fill_value=np.nan,
        )
    controls = response_group.create_array(
        "primary_dmso_cpm",
        shape=(2, 50, len(gene_names)),
        chunks=(1, 10, GENE_CHUNK),
        dtype="float32",
        compressor=compressor,
        fill_value=np.nan,
    )

    universe_hits = {
        "G_PRIMARY": np.zeros(len(gene_names), dtype=np.int16),
        "G_BROAD": np.zeros(len(gene_names), dtype=np.int16),
        "G_STRICT": np.zeros(len(gene_names), dtype=np.int16),
    }
    plate_interventions: list[list[str]] = []
    for plate_axis, plate in enumerate(("plate6", "plate14")):
        plate_meta = metadata[metadata["plate"] == plate]
        sample_meta = plate_meta.drop_duplicates("sample_axis").sort_values("sample_axis")
        treatment_samples = sample_meta[~sample_meta["is_control"].astype(bool)]
        control_samples = sample_meta[sample_meta["is_control"].astype(bool)]
        if treatment_samples["sample_axis"].tolist() != list(range(93)):
            raise RuntimeError(f"{plate}: treatment sample axes are not frozen 0..92")
        if control_samples["sample_axis"].tolist() != [93, 94]:
            raise RuntimeError(f"{plate}: DMSO wells are not preserved separately")
        plate_interventions.append(treatment_samples["intervention_id"].astype(str).tolist())
        context_meta = plate_meta.sort_values(["sample_axis", "context_axis"])
        libraries = context_meta["library_size"].to_numpy(np.float64).reshape(95, 50)
        cells = context_meta["n_cells"].to_numpy(np.float64).reshape(95, 50)
        if np.any(libraries <= 0) or np.any(cells <= 0):
            raise RuntimeError(f"{plate}: empty frozen pseudobulk")

        for start in range(0, len(gene_names), GENE_CHUNK):
            stop = min(len(gene_names), start + GENE_CHUNK)
            block = np.asarray(raw[plate_axis, :, :, start:stop], dtype=np.float64)
            cpm = block / libraries[:, :, None] * 1_000_000.0
            treatment_cpm = cpm[:93].transpose(1, 0, 2)
            well_cpm = cpm[93:95]
            primary_cpm = well_cpm.mean(axis=0)
            primary_control = np.log1p(well_cpm).mean(axis=0)
            sum_cpm = block[93:95].sum(axis=0) / libraries[93:95].sum(axis=0)[:, None] * 1_000_000.0
            sum_control = np.log1p(sum_cpm)
            cell_weights = cells[93:95] / cells[93:95].sum(axis=0, keepdims=True)
            weighted_control = np.sum(np.log1p(well_cpm) * cell_weights[:, :, None], axis=0)
            treatment_log = np.log1p(treatment_cpm)
            arrays["delta_primary"][plate_axis, :, :, start:stop] = (
                treatment_log - primary_control[:, None, :]
            ).astype(np.float32)
            arrays["delta_sum_control"][plate_axis, :, :, start:stop] = (
                treatment_log - sum_control[:, None, :]
            ).astype(np.float32)
            arrays["delta_cell_weighted_control"][plate_axis, :, :, start:stop] = (
                treatment_log - weighted_control[:, None, :]
            ).astype(np.float32)
            arrays["delta_log2fc"][plate_axis, :, :, start:stop] = np.log2(
                (treatment_cpm + 1.0) / (primary_cpm[:, None, :] + 1.0)
            ).astype(np.float32)
            controls[plate_axis, :, start:stop] = primary_cpm.astype(np.float32)
            universe_hits["G_PRIMARY"][start:stop] += (primary_cpm >= 0.5).sum(axis=0)
            universe_hits["G_BROAD"][start:stop] += (primary_cpm >= 0.1).sum(axis=0)
            universe_hits["G_STRICT"][start:stop] += (primary_cpm >= 1.0).sum(axis=0)
            print(f"{plate}: response genes {stop}/{len(gene_names)}", flush=True)

    if plate_interventions[0] != plate_interventions[1]:
        raise RuntimeError("Plate intervention axes differ after frozen-core sorting")
    interventions = plate_interventions[0]
    response_group.attrs.update(
        {
            "plates": ["plate6", "plate14"],
            "contexts": metadata.sort_values("context_axis").drop_duplicates("context_axis")["cell_line_id"].astype(str).tolist(),
            "interventions": interventions,
            "genes_sha256": _hash_genes(gene_names),
            "primary_estimator": "treatment log1p(CPM) minus equal-weight mean of two independently normalized DMSO-well log1p(CPM) profiles",
            "outcome_based_gene_filtering": False,
        }
    )

    technical_ok = np.ones(len(gene_names), dtype=bool)
    mito = np.array([name.upper().startswith("MT-") for name in gene_names])
    ribo = np.array([name.upper().startswith(("RPL", "RPS")) for name in gene_names])
    masks = {
        "G_ALL_MAPPED": technical_ok,
        "G_PRIMARY": technical_ok & (universe_hits["G_PRIMARY"] >= 10),
        "G_BROAD": technical_ok & (universe_hits["G_BROAD"] >= 5),
        "G_STRICT": technical_ok & (universe_hits["G_STRICT"] >= 20),
    }
    masks["G_PRIMARY_NO_MITO"] = masks["G_PRIMARY"] & ~mito
    masks["G_PRIMARY_NO_MITO_RIBO"] = masks["G_PRIMARY"] & ~mito & ~ribo
    manifest_rows: list[dict[str, Any]] = []
    hash_manifest: dict[str, Any] = {}
    for universe, mask in masks.items():
        selected = np.flatnonzero(mask)
        selected_names = [gene_names[index] for index in selected]
        hash_value = _hash_genes(selected_names)
        np.save(out / f"gene_indices_{universe.lower()}.npy", selected.astype(np.int32))
        hash_manifest[universe] = {
            "gene_count": len(selected),
            "sha256_newline_gene_symbols": hash_value,
            "indices_path": f"results/cgc_tahoe_0i/gene_indices_{universe.lower()}.npy",
        }
        criterion = {
            "G_ALL_MAPPED": "all unique officially mapped raw-matrix genes",
            "G_PRIMARY": "CPM >=0.5 in >=10/100 primary plate-context DMSO profiles",
            "G_BROAD": "CPM >=0.1 in >=5/100 primary plate-context DMSO profiles",
            "G_STRICT": "CPM >=1 in >=20/100 primary plate-context DMSO profiles",
            "G_PRIMARY_NO_MITO": "G_PRIMARY excluding symbols prefixed MT-",
            "G_PRIMARY_NO_MITO_RIBO": "G_PRIMARY excluding MT-, RPL*, and RPS* symbols",
        }[universe]
        for rank, index in enumerate(selected):
            manifest_rows.append(
                {
                    "universe": universe,
                    "universe_gene_rank": rank,
                    "raw_gene_index": int(index),
                    "gene_symbol": gene_names[index],
                    "ensembl_id": genes.iloc[index]["ensembl_id"],
                    "token_id": genes.iloc[index]["token_id"],
                    "selection_criterion": criterion,
                    "selection_uses_treated_response": False,
                    "universe_sha256": hash_value,
                }
            )
    pd.DataFrame(manifest_rows).to_csv(out / "gene_universe_manifest.csv", index=False)
    hash_manifest["frozen_at"] = _now()
    hash_manifest["selection_inputs"] = [
        "official unique gene mapping",
        "primary DMSO CPM profiles only",
        "technical symbol class for explicit sensitivities",
    ]
    hash_manifest["treated_response_used"] = False
    hash_manifest["official_de_significance_used"] = False
    hash_manifest["protein_coding_sensitivity"] = "NOT_RUN_BIOTYPE_NOT_AVAILABLE_IN_OFFICIAL_GENE_METADATA"
    _write_json(out / "gene_universe_hashes.json", hash_manifest)

    manifest = {
        "created_at": _now(),
        "response_store": "results/cgc_tahoe_0i/response_tensors.zarr",
        "shape_each_response": [2, 50, 93, 62_710],
        "storage_dtype": "float32",
        "normalization_computation_dtype": "float64",
        "primary": response_group.attrs["primary_estimator"],
        "secondary_sum_control": "sum two raw DMSO wells, normalize once, log1p(CPM), subtract",
        "secondary_cell_weighted_control": "cell-count-weighted mean of independently normalized DMSO-well log1p(CPM)",
        "secondary_log2fc": "log2((treatment CPM+1)/(equal-well primary DMSO CPM+1))",
        "gene_universes": {key: value["gene_count"] for key, value in hash_manifest.items() if isinstance(value, dict) and "gene_count" in value},
        "treated_response_used_for_selection": False,
        "de_pvalue_used_for_selection": False,
    }
    _write_json(out / "response_manifest.json", manifest)
    return manifest

