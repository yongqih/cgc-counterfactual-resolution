"""Strict sparse LOCO module vocabulary for the full-transcriptome residual."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import zarr
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score

from igc_virtual_cell.cgc_tahoe_0g.gate_a import _bh
from igc_virtual_cell.cgc_tahoe_0i.scaling import _folds, _support_orders


CHUNK = 256
TOP_K = 20
SEED = 2_609


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _residual_and_hard_sets(root: Path) -> tuple[zarr.Array, np.ndarray, np.ndarray, list[str], list[str]]:
    out = root / "results/cgc_tahoe_0i"
    source = zarr.open_group(out / "response_tensors.zarr", mode="r")
    response = source["delta_primary"]
    contexts = list(source.attrs["contexts"])
    interventions = list(source.attrs["interventions"])
    gene_indices = np.load(out / "gene_indices_g_primary.npy")
    folds = _folds(root, interventions)
    orders = _support_orders(root, contexts)
    weights = np.load(out / "primary_q49_affine_weights.npy")
    store_path = out / "loco_q49_residual.zarr"
    group = zarr.open_group(store_path, mode="w")
    residual = group.create_array(
        "residual", shape=(2, 50, 93, len(gene_indices)), chunks=(1, 10, 93, CHUNK), dtype="f4"
    )
    hard_values = np.full((2, 50 * 93, TOP_K), -np.inf, dtype=np.float64)
    hard_genes = np.full((2, 50 * 93, TOP_K), -1, dtype=np.int32)
    for local_start in range(0, len(gene_indices), CHUNK):
        local_stop = min(len(gene_indices), local_start + CHUNK)
        selected = gene_indices[local_start:local_stop]
        block = np.asarray(response[:, :, :, selected], dtype=np.float64)
        observed = np.zeros_like(block)
        for fold in range(5):
            evaluation = np.flatnonzero(folds == fold)
            for target in range(50):
                supports = orders[target, 0]
                for fit_plate in range(2):
                    support_block = block[:, supports, :, :][:, :, evaluation, :]
                    prediction = np.einsum(
                        "s,pseg->peg", weights[fold, fit_plate, target], support_block, optimize=True
                    )
                    observed[:, target, evaluation] += block[:, target, evaluation] - prediction
        observed *= 0.5
        residual[:, :, :, local_start:local_stop] = observed.astype(np.float32)
        absolute = np.abs(observed).reshape(2, 50 * 93, local_stop - local_start)
        indices = np.broadcast_to(np.arange(local_start, local_stop, dtype=np.int32), absolute.shape)
        combined_values = np.concatenate([hard_values, absolute], axis=2)
        combined_genes = np.concatenate([hard_genes, indices], axis=2)
        keep = np.argpartition(combined_values, -TOP_K, axis=2)[:, :, -TOP_K:]
        hard_values = np.take_along_axis(combined_values, keep, axis=2)
        hard_genes = np.take_along_axis(combined_genes, keep, axis=2)
        print(f"LOCO residual genes {local_stop}/{len(gene_indices)}", flush=True)
    order = np.argsort(hard_values, axis=2)[:, :, ::-1]
    hard_genes = np.take_along_axis(hard_genes, order, axis=2)
    np.save(out / "loco_hard_top20.npy", hard_genes)
    group.attrs.update({"gene_indices": gene_indices.tolist(), "top_k": TOP_K, "source_weight_directions_averaged": 2})
    return residual, hard_genes, gene_indices, contexts, interventions


def _pair_statistics(hard: np.ndarray, rows: np.ndarray, genes: int) -> tuple[pd.DataFrame, np.ndarray]:
    selected = hard[rows]
    degrees = np.bincount(selected.ravel(), minlength=genes).astype(np.float64)
    left, right = np.triu_indices(TOP_K, 1)
    a = selected[:, left].ravel()
    b = selected[:, right].ravel()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    codes = lo.astype(np.int64) * genes + hi
    unique, counts = np.unique(codes, return_counts=True)
    i, j = (unique // genes).astype(np.int32), (unique % genes).astype(np.int32)
    n = len(rows)
    expected = degrees[i] * degrees[j] * (TOP_K - 1) / (n * TOP_K - 1)
    variance = np.maximum(expected * (1 - expected / max(n, 1)), 1e-12)
    z = (counts - expected) / np.sqrt(variance)
    p = norm.sf(z)
    # Correct over the entire possible gene-pair family, including unobserved pairs.
    ranked = np.argsort(p)
    adjusted = np.empty_like(p)
    m = genes * (genes - 1) / 2
    adjusted[ranked] = np.minimum.accumulate((p[ranked] * m / np.arange(1, len(p) + 1))[::-1])[::-1]
    frame = pd.DataFrame({"i": i, "j": j, "observed": counts, "expected": expected, "excess": counts - expected, "z": z, "p": p, "fdr": np.minimum(adjusted, 1.0)})
    return frame, degrees


def _discover(frame: pd.DataFrame, genes: int) -> list[list[int]]:
    edges = frame.loc[(frame.observed > frame.expected) & (frame.fdr < 0.05)]
    if edges.empty:
        return []
    graph = ig.Graph(n=genes, edges=list(zip(edges.i.astype(int), edges.j.astype(int))), directed=False)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights=np.maximum(edges.excess.to_numpy(float), 1e-9).tolist(), resolution_parameter=1.0, seed=0,
    )
    return [sorted(group) for group in partition if 2 <= len(group) <= 50]


def _validate_modules(modules: list[list[int]], opposite: pd.DataFrame, degrees: np.ndarray, rows: int, genes: int) -> list[list[int]]:
    lookup = {(int(r.i), int(r.j)): float(r.observed) for r in opposite.itertuples(index=False)}
    valid = []
    for module in modules:
        excess = []
        for a, b in itertools.combinations(module, 2):
            expected = degrees[a] * degrees[b] * (TOP_K - 1) / (rows * TOP_K - 1)
            excess.append(lookup.get((min(a, b), max(a, b)), 0.0) - expected)
        if excess and float(np.mean(excess)) > 0:
            valid.append(module)
    return valid


def _loadings(residual: zarr.Array, source_plate: int, train: np.ndarray, modules: list[list[int]]) -> np.ndarray:
    columns = []
    for module in modules:
        values = np.asarray(residual[source_plate, train, :, module], dtype=np.float64).reshape(-1, len(module))
        values -= values.mean(axis=0)
        _, _, vt = np.linalg.svd(values, full_matrices=False)
        loading = vt[0]
        if loading[np.argmax(np.abs(loading))] < 0:
            loading = -loading
        columns.append((np.asarray(module), loading))
    return columns


def run_loco(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    residual, hard, gene_indices, contexts, _ = _residual_and_hard_sets(root)
    genes = len(gene_indices)
    metadata = pd.read_csv(out / "gene_metadata_frozen.csv")
    symbols = metadata.iloc[gene_indices]["gene_symbol"].astype(str).tolist()
    manifest_rows, coverage_rows, stability_rows, fold_modules = [], [], [], {0: [], 1: []}
    for heldout in range(50):
        train = np.delete(np.arange(50), heldout)
        rows = np.concatenate([np.arange(c * 93, (c + 1) * 93) for c in train])
        stats = []
        degrees = []
        for plate in range(2):
            table, degree = _pair_statistics(hard[plate], rows, genes)
            stats.append(table); degrees.append(degree)
        denominator = float(np.sum(np.asarray(residual[0, heldout], dtype=np.float64) * np.asarray(residual[1, heldout], dtype=np.float64)))
        for direction in range(2):
            discovered = _discover(stats[direction], genes)
            modules = _validate_modules(discovered, stats[1 - direction], degrees[1 - direction], len(rows), genes)
            fold_modules[direction].append(modules)
            loadings = _loadings(residual, direction, train, modules)
            numerator = 0.0
            held6 = np.asarray(residual[0, heldout], dtype=np.float64)
            held14 = np.asarray(residual[1, heldout], dtype=np.float64)
            for module_index, (module, loading) in enumerate(loadings):
                score6 = held6[:, module] @ loading
                score14 = held14[:, module] @ loading
                numerator += float(score6 @ score14)
                manifest_rows.append({
                    "heldout_context_index": heldout, "heldout_context_id": contexts[heldout],
                    "direction": "plate6_to_plate14" if direction == 0 else "plate14_to_plate6",
                    "module_index": module_index, "module_size": len(module),
                    "member_gene_indices_g_primary": ";".join(map(str, module)),
                    "member_genes": ";".join(symbols[i] for i in module),
                    "discovery_training_contexts": 49, "heldout_outcome_loaded_in_discovery": False,
                    "degree_conditioned_null": True, "leiden_resolution": 1.0,
                })
            coverage_rows.append({
                "heldout_context_index": heldout, "heldout_context_id": contexts[heldout],
                "direction": "plate6_to_plate14" if direction == 0 else "plate14_to_plate6",
                "module_count": len(modules), "module_sizes": ";".join(map(str, map(len, modules))),
                "numerator": numerator, "denominator": denominator,
                "heldout_module_signal_coverage": numerator / denominator if denominator else np.nan,
                "positive_coverage": numerator > 0,
            })
        print(f"LOCO module fold {heldout + 1}/50", flush=True)
    labels = np.full((2, 50, genes), -1, dtype=np.int32)
    for direction in range(2):
        for fold, modules in enumerate(fold_modules[direction]):
            for index, module in enumerate(modules): labels[direction, fold, module] = index
        for first, second in itertools.combinations(range(50), 2):
            active = (labels[direction, first] >= 0) | (labels[direction, second] >= 0)
            stability_rows.append({"direction": direction, "fold_i": first, "fold_j": second,
                                   "ari": adjusted_rand_score(labels[direction, first, active], labels[direction, second, active]) if active.any() else np.nan})
    manifest = pd.DataFrame(manifest_rows)
    coverage = pd.DataFrame(coverage_rows)
    stability = pd.DataFrame(stability_rows)
    manifest.to_csv(out / "loco_module_manifest.csv", index=False)
    coverage.to_csv(out / "loco_module_coverage.csv", index=False)
    stability.to_csv(out / "loco_module_stability.csv", index=False)
    per_context = coverage.groupby("heldout_context_index", as_index=False).agg(numerator=("numerator", "sum"), denominator=("denominator", "sum"))
    per_context["coverage"] = per_context.numerator / per_context.denominator
    rng = np.random.default_rng(SEED)
    boot = np.empty(10_000)
    for draw in range(len(boot)):
        sample = rng.integers(0, 50, 50)
        boot[draw] = per_context.numerator.iloc[sample].sum() / per_context.denominator.iloc[sample].sum()
    summary = {
        "created_at": _now(), "strict_loco_folds": 50, "directions": 2, "top_k": TOP_K,
        "degree_conditioned_null": "configuration expectation with family-wide BH",
        "leiden_resolution": 1.0, "module_size_range": [2, 50],
        "pooled_coverage": float(coverage.numerator.sum() / coverage.denominator.sum()),
        "bootstrap_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "positive_contexts": int((per_context.coverage > 0).sum()),
        "mean_module_count": float(coverage.module_count.mean()),
        "module_sizes": sorted(manifest.module_size.unique().astype(int).tolist()) if not manifest.empty else [],
        "mean_stability_ari": float(stability.ari.mean()),
        "heldout_outcome_used_in_discovery": False,
    }
    _write_json(out / "loco_module_summary.json", summary)
    return summary
