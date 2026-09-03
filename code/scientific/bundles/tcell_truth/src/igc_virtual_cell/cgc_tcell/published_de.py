"""Independent published-DE confirmation for confirmed CGC-TCELL-0A signal."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import shutil
import subprocess
from typing import Any

from anndata.io import read_elem, sparse_dataset
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from igc_virtual_cell.cgc_tcell.core import (
    crossmeasurement_context_distances,
    crossmeasurement_permutation_null,
    stratified_permutation_indices,
)


def _git(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _modalities(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(map(str, handle["mod"].keys()))


def _metadata(path: Path, modality: str) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, int]]:
    with h5py.File(path, "r") as handle:
        group = handle["mod"][modality]
        obs = read_elem(group["obs"])
        var = read_elem(group["var"])
        shape = tuple(group["layers"]["log_fc"].attrs.get("shape", ()))
        if not shape:
            shape = tuple(group["layers"]["log_fc"].shape)
    return obs, var, shape


def _matrix(group: h5py.Group) -> Any:
    item = group["layers"]["log_fc"]
    return item if isinstance(item, h5py.Dataset) else sparse_dataset(item)


def _complete_targets(obs: pd.DataFrame, states: list[str]) -> set[str]:
    grouped = obs.groupby("target_contrast", observed=True)["culture_condition"].agg(
        lambda values: set(map(str, values))
    )
    expected = set(states)
    return set(grouped.index[grouped.map(lambda value: value == expected)].astype(str))


def _load_modality_tensor(
    path: Path,
    modality: str,
    targets: list[str],
    genes: list[str],
    states: list[str],
    *,
    row_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["mod"][modality]
        obs = read_elem(group["obs"])
        var = read_elem(group["var"])
        gene_column = "gene_ids" if "gene_ids" in var else var.columns[0]
        gene_position = {str(gene): index for index, gene in enumerate(var[gene_column])}
        missing = set(genes) - set(gene_position)
        if missing:
            raise ValueError(f"Published DE modality missing {len(missing)} frozen genes")
        gene_indices = np.asarray([gene_position[gene] for gene in genes], dtype=np.int64)
        row_position = {
            (str(row.target_contrast), str(row.culture_condition)): index
            for index, row in enumerate(obs.itertuples(index=False))
        }
        tensor = np.empty((len(targets), len(states), len(genes)), dtype=np.float32)
        nuisance = np.empty((len(states), len(targets)), dtype=np.float64)
        matrix = _matrix(group)
        for state_index, state in enumerate(states):
            rows = np.asarray([row_position[(target, state)] for target in targets], dtype=np.int64)
            if "n_cells_target" in obs:
                nuisance[state_index] = obs.iloc[rows]["n_cells_target"].to_numpy(dtype=float)
            else:
                nuisance[state_index] = 1.0
            for start in range(0, len(rows), row_chunk):
                stop = min(start + row_chunk, len(rows))
                selected = rows[start:stop]
                order = np.argsort(selected, kind="stable")
                raw = matrix[selected[order], :]
                if sparse.issparse(raw):
                    block = raw[:, gene_indices].toarray()
                else:
                    block = np.asarray(raw)[:, gene_indices]
                tensor[start:stop, state_index] = np.asarray(block, dtype=np.float32)[
                    np.argsort(order)
                ]
    if not np.isfinite(tensor).all():
        raise ValueError(f"Non-finite published log_fc in modality {modality}")
    return tensor, nuisance


def _null_summary(
    left: np.ndarray,
    right: np.ndarray,
    nuisance: np.ndarray,
    *,
    draws: int,
    bins: int,
    seed: int,
    batch_draws: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    observed = crossmeasurement_context_distances(left, right)
    maps = stratified_permutation_indices(
        nuisance, draws=draws, bins=bins, seed=seed
    )
    null = crossmeasurement_permutation_null(
        left, right, maps, batch_draws=batch_draws, device="cuda"
    )
    return observed, null


def _update_download_manifest(
    root: Path, output: Path, raw: Path, specs: dict[str, int]
) -> None:
    manifest_path = output / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {item["file"] for item in manifest["files"]}
    for name, expected in specs.items():
        path = raw / name
        if path.stat().st_size != expected:
            raise ValueError(f"Conditional DE download has wrong size: {name}")
        if name not in existing:
            manifest["files"].append(
                {
                    "file": name,
                    "path": str(path.relative_to(root)),
                    "url": f"https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/{name}",
                    "expected_bytes": expected,
                    "actual_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "conditional_authorization": "primary_truth_signal_confirmed",
                }
            )
    manifest["git_provenance"] = _git(root)
    manifest["de_files_downloaded"] = True
    manifest["free_disk_gb_after_download"] = round(shutil.disk_usage(root).free / 1e9, 3)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_confirmation(root: Path, config_path: Path) -> pd.DataFrame:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = root / config["storage"]["raw_directory"]
    output = root / config["outputs"]["directory"]
    verdict = json.loads((output / "verdict.json").read_text(encoding="utf-8"))
    if verdict["verdict"] != "TCELL_STIMULATION_OPERATOR_SIGNAL_CONFIRMED":
        raise RuntimeError("Published DE files are not authorized without confirmed primary signal")
    specs = {
        "GWCD4i.DE_stats.by_guide.h5mu": 29424424894,
        "GWCD4i.DE_stats.by_donors.h5mu": 16866278447,
    }
    for name, expected in specs.items():
        path = raw / name
        if not path.exists() or path.stat().st_size != expected:
            raise FileNotFoundError(f"Missing conditional DE file: {path}")
    _update_download_manifest(root, output, raw, specs)
    strict = pd.read_csv(output / "strict_trans_genes.csv")
    strict_ids = strict.loc[strict["strict_trans_eligible"], "gene_id"].astype(str).tolist()
    states = list(map(str, config["frozen_states"]))
    draws = int(config["null"]["permutations"])
    bins = int(config["null"]["cell_count_bins"])
    batch = int(config["null"]["gpu_batch_size"])
    rows: list[dict[str, Any]] = []

    guide_path = raw / "GWCD4i.DE_stats.by_guide.h5mu"
    guide_modalities = _modalities(guide_path)
    if guide_modalities != ["guide_1", "guide_2"]:
        raise ValueError(f"Unexpected guide modalities: {guide_modalities}")
    guide_metadata = [_metadata(guide_path, modality)[0] for modality in guide_modalities]
    guide_targets = sorted(
        set.intersection(*[_complete_targets(obs, states) for obs in guide_metadata])
    )
    var = _metadata(guide_path, "guide_1")[1]
    published_genes = set(var["gene_ids"].astype(str))
    donor_path = raw / "GWCD4i.DE_stats.by_donors.h5mu"
    donor_modalities = _modalities(donor_path)
    donor_var = _metadata(donor_path, donor_modalities[0])[1]
    donor_genes = set(donor_var["gene_ids"].astype(str))
    genes = [
        gene for gene in strict_ids if gene in published_genes and gene in donor_genes
    ]
    guide_1, cells_1 = _load_modality_tensor(
        guide_path, "guide_1", guide_targets, genes, states
    )
    guide_2, cells_2 = _load_modality_tensor(
        guide_path, "guide_2", guide_targets, genes, states
    )
    observed, null = _null_summary(
        guide_1,
        guide_2,
        (cells_1 + cells_2) * 0.5,
        draws=draws,
        bins=bins,
        seed=int(config["random_seed"]) + 5,
        batch_draws=batch,
    )
    pair_labels = list(combinations(states, 2))
    for index, (state_a, state_b) in enumerate(pair_labels):
        value = float(observed.iloc[index]["crossvalidated_distance"])
        q95 = float(np.quantile(null[:, index], 0.95))
        rows.append(
            {
                "analysis": "published_guide_specific_DESeq2",
                "replicate": "guide_1_x_guide_2",
                "state_a": state_a,
                "state_b": state_b,
                "targets": len(guide_targets),
                "strict_trans_genes": len(genes),
                "crossvalidated_distance": value,
                "null_q95": q95,
                "empirical_p": (1 + int(np.count_nonzero(null[:, index] >= value)))
                / (draws + 1),
                "exceeds_null_q95": value > q95,
            }
        )

    donors = json.loads((output / "dataset_inventory.json").read_text())["donors"]
    modality_sets = {frozenset(name.split("_")): name for name in donor_modalities}
    partition_pairs = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]
    donor_nulls, donor_values = [], []
    for partition_index, (left_indices, right_indices) in enumerate(partition_pairs):
        left_name = modality_sets[frozenset(donors[index] for index in left_indices)]
        right_name = modality_sets[frozenset(donors[index] for index in right_indices)]
        left_obs = _metadata(donor_path, left_name)[0]
        right_obs = _metadata(donor_path, right_name)[0]
        targets = sorted(
            _complete_targets(left_obs, states) & _complete_targets(right_obs, states)
        )
        left, left_cells = _load_modality_tensor(
            donor_path, left_name, targets, genes, states
        )
        right, right_cells = _load_modality_tensor(
            donor_path, right_name, targets, genes, states
        )
        observed, null = _null_summary(
            left,
            right,
            (left_cells + right_cells) * 0.5,
            draws=draws,
            bins=bins,
            seed=int(config["random_seed"]) + 10 + partition_index,
            batch_draws=batch,
        )
        donor_values.append(observed["crossvalidated_distance"].to_numpy())
        donor_nulls.append(null)
        for index, (state_a, state_b) in enumerate(pair_labels):
            value = float(observed.iloc[index]["crossvalidated_distance"])
            q95 = float(np.quantile(null[:, index], 0.95))
            rows.append(
                {
                    "analysis": "published_donor_pair_DESeq2",
                    "replicate": f"{left_name}__x__{right_name}",
                    "partition_index": partition_index,
                    "state_a": state_a,
                    "state_b": state_b,
                    "targets": len(targets),
                    "strict_trans_genes": len(genes),
                    "crossvalidated_distance": value,
                    "null_q95": q95,
                    "empirical_p": (1 + int(np.count_nonzero(null[:, index] >= value)))
                    / (draws + 1),
                    "exceeds_null_q95": value > q95,
                }
            )
    result = pd.DataFrame(rows)
    result.insert(0, "git_provenance", _git(root))
    result.to_csv(output / "independent_de_confirmation.csv", index=False)
    summary = {
        "guide_overall_mean": float(np.mean(result.loc[result["analysis"].str.contains("guide"), "crossvalidated_distance"])),
        "guide_all_contrasts_positive": bool((result.loc[result["analysis"].str.contains("guide"), "crossvalidated_distance"] > 0).all()),
        "guide_all_contrasts_exceed_null": bool(result.loc[result["analysis"].str.contains("guide"), "exceeds_null_q95"].all()),
        "donor_overall_mean": float(np.mean(donor_values)),
        "donor_positive_fraction": float(np.mean(np.asarray(donor_values) > 0)),
        "donor_null_q95_overall": float(
            np.quantile(np.mean(np.stack(donor_nulls), axis=(0, 2)), 0.95)
        ),
        "donor_empirical_p_overall": float(
            (
                1
                + np.count_nonzero(
                    np.mean(np.stack(donor_nulls), axis=(0, 2))
                    >= np.mean(donor_values)
                )
            )
            / (draws + 1)
        ),
    }
    (output / "independent_de_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    result = run_confirmation(root, config)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
