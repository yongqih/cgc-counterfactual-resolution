"""Pre-outcome guide-disjoint tensor materialization for CGC-TCELL-0A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from igc_virtual_cell.cgc_tcell.inventory import _ntc_mask, _targeting_mask
from igc_virtual_cell.cgc_tcell.prepare import (
    _bool,
    _materialize_response,
    _response_weights,
)


def _git(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _guide_order(label: str) -> bytes:
    return hashlib.sha256(str(label).encode()).digest()


def _materialize_pair(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    strict: pd.DataFrame,
    weights_a: sparse.csr_matrix,
    weights_b: sparse.csr_matrix,
    *,
    targets: list[str],
    chunk_genes: int,
    chunk_rows: int,
    cpm_scale: float,
    path_a: Path,
    path_b: Path,
) -> None:
    from igc_virtual_cell.cgc_tcell.prepare import _normalized_chunk

    genes = strict.loc[strict["strict_trans_eligible"], "gene_index"].to_numpy(dtype=np.int64)
    used = np.union1d(weights_a.indices, weights_b.indices)
    local_a, local_b = weights_a[:, used], weights_b[:, used]
    libraries = obs.loc[used, "total_counts"].to_numpy(dtype=np.float64)
    shape = (len(targets), 4, 3, len(genes))
    response_a = np.memmap(path_a, mode="w+", dtype=np.float32, shape=shape)
    response_b = np.memmap(path_b, mode="w+", dtype=np.float32, shape=shape)
    flat_a, flat_b = response_a.reshape(len(targets) * 12, len(genes)), response_b.reshape(
        len(targets) * 12, len(genes)
    )
    flat_a[:] = 0.0
    flat_b[:] = 0.0
    for row_start in range(0, len(used), chunk_rows):
        row_stop = min(row_start + chunk_rows, len(used))
        normalized = _normalized_chunk(
            adata,
            used[row_start:row_stop],
            genes,
            libraries[row_start:row_stop],
            cpm_scale,
        )
        row_a, row_b = local_a[:, row_start:row_stop], local_b[:, row_start:row_stop]
        for gene_start in range(0, len(genes), chunk_genes):
            gene_stop = min(gene_start + chunk_genes, len(genes))
            expression = normalized[:, gene_start:gene_stop]
            block_a, block_b = row_a @ expression, row_b @ expression
            flat_a[:, gene_start:gene_stop] += (
                block_a.toarray() if sparse.issparse(block_a) else np.asarray(block_a)
            )
            flat_b[:, gene_start:gene_stop] += (
                block_b.toarray() if sparse.issparse(block_b) else np.asarray(block_b)
            )
    response_a.flush()
    response_b.flush()


def run_materialize(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = root / config["storage"]["raw_directory"]
    processed = root / config["storage"]["processed_directory"]
    output = root / config["outputs"]["directory"]
    guide = pd.read_csv(output / "eligible_guides.csv", low_memory=False)
    complete = guide.loc[
        guide["guide_type"].astype(str).eq("targeting")
        & _bool(guide["complete_12_contexts"])
        & guide["primary_eligible_rows"].gt(0)
        & ~_bool(guide["problematic_library_annotation"])
    ].copy()
    groups = {
        str(target): sorted(frame["guide_id"].astype(str), key=_guide_order)
        for target, frame in complete.groupby("perturbed_gene_id", observed=True)
        if frame["guide_id"].nunique() >= 2
    }
    threshold = int(config["guide_disjoint"]["minimum_complete_targets"])
    if len(groups) < threshold:
        manifest = {
            "git_provenance": _git(root),
            "status": "GUIDE_DISJOINT_FULL_CONTEXT_NOT_FEASIBLE",
            "complete_targets_post_qc": len(groups),
            "frozen_threshold": threshold,
        }
        (output / "guide_disjoint_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    targets = sorted(groups)
    assignment_rows = []
    assignment: dict[str, str] = {}
    for target in targets:
        for index, guide_id in enumerate(groups[target]):
            split = "A" if index % 2 == 0 else "B"
            assignment[guide_id] = split
            assignment_rows.append(
                {
                    "perturbed_gene_id": target,
                    "guide_id": guide_id,
                    "guide_split": split,
                    "assignment_rank": index,
                    "assignment_rule": "sha256_guide_order_alternating",
                }
            )
    pd.DataFrame(assignment_rows).to_csv(output / "guide_disjoint_assignment.csv", index=False)

    adata = ad.read_h5ad(raw / "GWCD4i.pseudobulk_merged.h5ad", backed="r")
    obs = adata.obs.reset_index(names="pseudobulk_id").copy()
    targeting, ntc = _targeting_mask(obs), _ntc_mask(obs)
    targeting_qc = targeting
    for flag in config["eligibility"]["primary_observation_flags"]:
        targeting_qc &= _bool(obs[flag])
    obs["ntc_eligible"] = ntc & _bool(obs["keep_min_cells"]) & _bool(
        obs["keep_total_counts"]
    )
    guide_split = obs["guide_id"].astype(str).map(assignment)
    obs["guide_a_response_row"] = targeting_qc & guide_split.eq("A")
    obs["guide_b_response_row"] = targeting_qc & guide_split.eq("B")
    donors = sorted(obs["donor_id"].astype(str).unique())
    states = list(map(str, config["frozen_states"]))
    weights_a, nuisance_a, units_a = _response_weights(
        obs, targets, donors, states, response_flag="guide_a_response_row"
    )
    weights_b, nuisance_b, units_b = _response_weights(
        obs, targets, donors, states, response_flag="guide_b_response_row"
    )
    units_a.assign(guide_split="A").to_csv(
        output / "guide_disjoint_a_response_units.csv", index=False
    )
    units_b.assign(guide_split="B").to_csv(
        output / "guide_disjoint_b_response_units.csv", index=False
    )
    nuisance_path = processed / "guide_disjoint_n_cells.npy"
    np.save(nuisance_path, (nuisance_a + nuisance_b) * 0.5)
    strict = pd.read_csv(output / "strict_trans_genes.csv")
    path_a = processed / "guide_disjoint_a_delta.float32.mmap"
    path_b = processed / "guide_disjoint_b_delta.float32.mmap"
    _materialize_pair(
        adata,
        obs,
        strict,
        weights_a,
        weights_b,
        targets=targets,
        chunk_genes=int(config["normalization"]["chunk_genes"]),
        chunk_rows=int(config["normalization"]["chunk_rows"]),
        cpm_scale=float(config["normalization"]["cpm_scale"]),
        path_a=path_a,
        path_b=path_b,
    )
    adata.file.close()
    shape = [len(targets), 4, 3, int(strict["strict_trans_eligible"].sum())]
    manifest = {
        "git_provenance": _git(root),
        "status": "GUIDE_DISJOINT_FULL_CONTEXT_MATERIALIZED",
        "complete_targets_post_qc": len(targets),
        "frozen_threshold": threshold,
        "target_ids": targets,
        "response_shape": shape,
        "response_a_path": str(path_a.relative_to(root)),
        "response_b_path": str(path_b.relative_to(root)),
        "nuisance_path": str(nuisance_path.relative_to(root)),
    }
    (output / "guide_disjoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    result = run_materialize(root, config)
    print(json.dumps({key: value for key, value in result.items() if key != "target_ids"}, indent=2))


if __name__ == "__main__":
    main()
