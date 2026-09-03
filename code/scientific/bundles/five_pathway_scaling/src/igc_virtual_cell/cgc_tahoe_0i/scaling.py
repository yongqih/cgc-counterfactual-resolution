"""Tensorized full-transcriptome context-support scaling from Gram statistics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from igc_virtual_cell.cgc_tahoe_0i.truth import UNIVERSES


SUPPORT_SIZES = (2, 4, 8, 16, 24, 32, 40, 49)
SUPPORT_SEQUENCES = 128
GENE_CHUNK = 256
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 2_606
NULL_DRAWS = 5_000
NULL_SEED = 2_607
LAMBDAS = np.asarray([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], dtype=np.float64)
RCOND = 1e-10


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _folds(root: Path, interventions: list[str]) -> np.ndarray:
    frame = pd.read_csv(root / "results/cgc_tahoe_0c/intervention_folds.csv")
    mapping = frame.set_index("intervention_id")["fold"].to_dict()
    values = np.asarray([mapping[value] for value in interventions], dtype=np.int8)
    if set(values.tolist()) != set(range(5)):
        raise RuntimeError("Frozen intervention folds changed")
    return values


def _support_orders(root: Path, contexts: list[str]) -> np.ndarray:
    frame = pd.read_csv(root / "results/cgc_tahoe_0c/support_sequences.csv")
    mapping = {value: index for index, value in enumerate(contexts)}
    expected = len(contexts) * SUPPORT_SEQUENCES * (len(contexts) - 1)
    if len(frame) != expected:
        raise RuntimeError("Frozen support-sequence manifest changed")
    orders = frame["support_context"].map(mapping).to_numpy().reshape(50, 128, 49)
    for target in range(50):
        if np.any(orders[target] == target):
            raise RuntimeError("Held-out context leaked into support sequence")
    return orders


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=2, keepdims=True, dtype=np.float64)


def _batched_affine(gram: np.ndarray, cross: np.ndarray) -> np.ndarray:
    batch, support, _ = gram.shape
    # Solve in an orthonormal zero-sum basis instead of applying a relative
    # cutoff to the scale-mismatched KKT matrix [G, 1; 1', 0].  The latter can
    # discard the affine constraint when G is large even though the constrained
    # problem itself is well defined.
    origin = np.full(support, 1.0 / support, dtype=np.float64)
    contrasts = np.vstack((np.eye(support - 1), -np.ones((1, support - 1))))
    basis, _ = np.linalg.qr(contrasts, mode="reduced")
    reduced_gram = np.einsum("si,bst,tj->bij", basis, gram, basis, optimize=True)
    gram_origin = np.einsum("bst,t->bs", gram, origin, optimize=True)
    reduced_cross = np.einsum("si,bs->bi", basis, cross - gram_origin, optimize=True)
    u, singular, vh = np.linalg.svd(reduced_gram, full_matrices=False)
    projected = np.einsum("bij,bj->bi", u.transpose(0, 2, 1), reduced_cross, optimize=True)
    cutoff = RCOND * singular[:, :1]
    projected = np.divide(projected, singular, out=np.zeros_like(projected), where=singular > cutoff)
    coordinates = np.einsum("bij,bj->bi", vh.transpose(0, 2, 1), projected, optimize=True)
    weights = origin + np.einsum("si,bi->bs", basis, coordinates, optimize=True)
    if np.max(np.abs(weights.sum(axis=1) - 1)) > 1e-10:
        raise RuntimeError("Minimum-norm affine constraint failed")
    return weights


def _centered_subset_gram(
    fold_products: np.ndarray,
    fold_sum_products: np.ndarray,
    selected_folds: list[int],
    fold_sizes: np.ndarray,
) -> np.ndarray:
    n = int(fold_sizes[selected_folds].sum())
    first = fold_products[selected_folds].sum(axis=0)
    second = fold_sum_products[np.ix_(selected_folds, selected_folds)].sum(axis=(0, 1))
    return first - second / n


def build_gram_cache(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    truth = json.loads((out / "truth_gate.json").read_text(encoding="utf-8"))
    if not truth["primary_truth_passed"]:
        raise RuntimeError("Truth gate did not authorize support scaling")
    group = zarr.open_group(out / "response_tensors.zarr", mode="r")
    response = group["delta_primary"]
    contexts = list(group.attrs["contexts"])
    interventions = list(group.attrs["interventions"])
    folds = _folds(root, interventions)
    fold_sizes = np.bincount(folds, minlength=5)
    flags: dict[str, np.ndarray] = {}
    for universe in UNIVERSES:
        flag = np.zeros(response.shape[-1], dtype=bool)
        flag[np.load(out / f"gene_indices_{universe.lower()}.npy")] = True
        flags[universe] = flag
    cache = {
        universe: {
            "fold_products": np.zeros((2, 5, 50, 50), dtype=np.float64),
            "fold_sum_products": np.zeros((2, 5, 5, 50, 50), dtype=np.float64),
            "eval_same": np.zeros((2, 93, 50, 50), dtype=np.float64),
            "eval_cross": np.zeros((93, 50, 50), dtype=np.float64),
            "gene_count": 0,
        }
        for universe in UNIVERSES
    }
    for start in range(0, response.shape[-1], GENE_CHUNK):
        stop = min(response.shape[-1], start + GENE_CHUNK)
        block = np.asarray(response[:, :, :, start:stop], dtype=np.float64)
        for universe in UNIVERSES:
            local = flags[universe][start:stop]
            if not local.any():
                continue
            values = block[..., local]
            item = cache[universe]
            item["gene_count"] += int(local.sum())
            fold_sums = np.zeros((2, 5, 50, int(local.sum())), dtype=np.float64)
            for fold in range(5):
                indices = np.flatnonzero(folds == fold)
                selected = values[:, :, indices, :]
                item["fold_products"][:, fold] += np.einsum(
                    "rcpg,rdpg->rcd", selected, selected, optimize=True
                )
                fold_sums[:, fold] = selected.sum(axis=2, dtype=np.float64)
                centered = _center(selected)
                item["eval_same"][:, indices] += np.einsum(
                    "rcpg,rdpg->rpcd", centered, centered, optimize=True
                )
                item["eval_cross"][indices] += np.einsum(
                    "cpg,dpg->pcd", centered[0], centered[1], optimize=True
                )
            item["fold_sum_products"] += np.einsum(
                "rfcg,rhdg->rfhcd", fold_sums, fold_sums, optimize=True
            )
        print(f"support Gram genes {stop}/{response.shape[-1]}", flush=True)

    paths = {}
    for universe, item in cache.items():
        expected = int(flags[universe].sum())
        if item["gene_count"] != expected:
            raise RuntimeError(f"{universe}: support Gram gene count mismatch")
        path = out / f"support_grams_{universe.lower()}.npz"
        np.savez_compressed(
            path,
            fold_products=item["fold_products"],
            fold_sum_products=item["fold_sum_products"],
            eval_same=item["eval_same"],
            eval_cross=item["eval_cross"],
            fold_sizes=fold_sizes,
            folds=folds,
            gene_count=np.asarray(item["gene_count"]),
        )
        paths[universe] = str(path.relative_to(root))
    manifest = {
        "created_at": _now(),
        "method": "streaming sufficient Gram statistics; no dense case-by-gene design matrix",
        "response_estimator": "delta_primary",
        "gene_chunk": GENE_CHUNK,
        "folds": 5,
        "cache_paths": paths,
        "intervention_fold_source": "results/cgc_tahoe_0c/intervention_folds.csv",
        "support_sequence_source": "results/cgc_tahoe_0c/support_sequences.csv",
    }
    _write_json(out / "support_gram_manifest.json", manifest)
    return manifest


def _residual_from_gram(
    gram: np.ndarray,
    target: int,
    supports: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    left = gram[supports, target]
    right = gram[target, supports]
    support_gram = gram[supports[:, :, None], supports[:, None, :]]
    return (
        gram[target, target]
        - np.einsum("bn,bn->b", weights, left)
        - np.einsum("bn,bn->b", weights, right)
        + np.einsum("bi,bij,bj->b", weights, support_gram, weights, optimize=True)
    )


def _bootstrap_utility(utility: np.ndarray, draws: int = BOOTSTRAPS) -> np.ndarray:
    # utility: support x context x intervention x [base,residual,...]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty((draws, utility.shape[0]), dtype=np.float64)
    for start in range(0, draws, 100):
        stop = min(draws, start + 100)
        batch = stop - start
        contexts = rng.integers(0, 50, size=(batch, 50))
        interventions = rng.integers(0, 93, size=(batch, 50, 93))
        expanded = np.broadcast_to(contexts[:, :, None], interventions.shape)
        for size in range(utility.shape[0]):
            base = utility[size, ..., 0][expanded, interventions].sum(axis=(1, 2))
            residual = utility[size, ..., 1][expanded, interventions].sum(axis=(1, 2))
            values[start:stop, size] = residual / base
    return values


def _fieller_context_interval(base: np.ndarray, residual: np.ndarray) -> tuple[float, float, str]:
    """Classical Fieller interval using 50 context-level paired totals."""

    n = len(base)
    mean_x, mean_y = float(base.mean()), float(residual.mean())
    covariance = np.cov(np.stack([base, residual]), ddof=1) / n
    z2 = 1.959963984540054**2
    a = mean_x * mean_x - z2 * covariance[0, 0]
    b = -2 * (mean_x * mean_y - z2 * covariance[0, 1])
    c = mean_y * mean_y - z2 * covariance[1, 1]
    discriminant = b * b - 4 * a * c
    if a <= 0 or discriminant < 0:
        return float("-inf"), float("inf"), "UNBOUNDED"
    roots = sorted(((-b - np.sqrt(discriminant)) / (2 * a), (-b + np.sqrt(discriminant)) / (2 * a)))
    return float(roots[0]), float(roots[1]), "BOUNDED"


def _run_universe(
    root: Path,
    universe: str,
    contexts: list[str],
    interventions: list[str],
    support_orders: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]], dict[str, Any], np.ndarray]:
    out = root / "results/cgc_tahoe_0i"
    cached = np.load(out / f"support_grams_{universe.lower()}.npz")
    fold_products = cached["fold_products"]
    fold_sum_products = cached["fold_sum_products"]
    eval_same = cached["eval_same"]
    eval_cross = cached["eval_cross"]
    folds = cached["folds"]
    fold_sizes = cached["fold_sizes"]
    gene_count = int(cached["gene_count"])
    utility_sum = np.zeros((8, 50, 93, 4), dtype=np.float64)
    utility_count = np.zeros((8, 50, 93), dtype=np.int32)
    max_weights = np.zeros((5, 2, 50, 49), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []

    outer_grams = np.zeros((5, 2, 50, 50), dtype=np.float64)
    for fold in range(5):
        selected = [value for value in range(5) if value != fold]
        for plate in range(2):
            outer_grams[fold, plate] = _centered_subset_gram(
                fold_products[plate], fold_sum_products[plate], selected, fold_sizes
            )
    for fold in range(5):
        evaluation = np.flatnonzero(folds == fold)
        for target in range(50):
            for size_axis, support_size in enumerate(SUPPORT_SIZES):
                sequence_count = 1 if support_size == 49 else SUPPORT_SEQUENCES
                supports = support_orders[target, :sequence_count, :support_size]
                base_weights = np.full((sequence_count, support_size), 1.0 / support_size)
                for direction in range(2):
                    gram = outer_grams[fold, direction]
                    support_gram = gram[supports[:, :, None], supports[:, None, :]]
                    support_target = gram[supports, target]
                    weights = _batched_affine(support_gram, support_target)
                    if support_size == 49:
                        max_weights[fold, direction, target] = weights[0]
                    for intervention in evaluation:
                        cross = eval_cross[intervention] / gene_count
                        same6 = eval_same[0, intervention] / gene_count
                        same14 = eval_same[1, intervention] / gene_count
                        base_cross = _residual_from_gram(cross, target, supports, base_weights)
                        residual_cross = _residual_from_gram(cross, target, supports, weights)
                        base_observed = (
                            _residual_from_gram(same6, target, supports, base_weights)
                            + _residual_from_gram(same14, target, supports, base_weights)
                        ) / 2
                        residual_observed = (
                            _residual_from_gram(same6, target, supports, weights)
                            + _residual_from_gram(same14, target, supports, weights)
                        ) / 2
                        utility_sum[size_axis, target, intervention] += np.asarray(
                            [base_cross.sum(), residual_cross.sum(), base_observed.sum(), residual_observed.sum()]
                        )
                        utility_count[size_axis, target, intervention] += sequence_count
                    diagnostics.append(
                        {
                            "universe": universe,
                            "outer_fold": fold,
                            "heldout_context": contexts[target],
                            "support_size": support_size,
                            "fit_plate": ("plate6", "plate14")[direction],
                            "sequence_count": sequence_count,
                            "median_weight_norm": float(np.median(np.linalg.norm(weights, axis=1))),
                            "median_max_abs_weight": float(np.median(np.max(np.abs(weights), axis=1))),
                            "median_negative_weight_fraction": float(np.median(np.mean(weights < 0, axis=1))),
                        }
                    )
        print(f"{universe}: support fold {fold + 1}/5", flush=True)
    if np.any(utility_count == 0):
        raise RuntimeError(f"{universe}: incomplete OOF support utility")
    utility = utility_sum / utility_count[..., None]
    rows = []
    for size_axis, support_size in enumerate(SUPPORT_SIZES):
        context_base = utility[size_axis, ..., 0].sum(axis=1)
        context_residual = utility[size_axis, ..., 1].sum(axis=1)
        context_q = context_residual / context_base
        rows.append(
            {
                "universe": universe,
                "support_size": support_size,
                "q_pooled": float(context_residual.sum() / context_base.sum()),
                "median_context_q": float(np.median(context_q)),
                "context_q_iqr_lower": float(np.quantile(context_q, 0.25)),
                "context_q_iqr_upper": float(np.quantile(context_q, 0.75)),
                "pooled_v_base": float(utility[size_axis, ..., 0].mean()),
                "pooled_v_residual": float(utility[size_axis, ..., 1].mean()),
                "raw_r2": float(1 - utility[size_axis, ..., 3].sum() / utility[size_axis, ..., 2].sum()),
                "positive_residual_contexts": int(np.count_nonzero(context_residual > 0)),
            }
        )
    curve = pd.DataFrame(rows)
    bootstrap = _bootstrap_utility(utility)
    for size_axis in range(8):
        low, high = np.quantile(bootstrap[:, size_axis], [0.025, 0.975])
        curve.loc[size_axis, "bootstrap_lower_95"] = low
        curve.loc[size_axis, "bootstrap_upper_95"] = high
    x = np.log2(np.asarray(SUPPORT_SIZES, dtype=np.float64))
    centered_x = x - x.mean()
    q = curve["q_pooled"].to_numpy()
    slopes = (bootstrap @ centered_x) / np.sum(centered_x * centered_x)
    differences = bootstrap[:, 0] - bootstrap[:, -1]
    trend = {
        "universe": universe,
        "beta1": float(np.dot(q, centered_x) / np.dot(centered_x, centered_x)),
        "beta1_lower_95": float(np.quantile(slopes, 0.025)),
        "beta1_upper_95": float(np.quantile(slopes, 0.975)),
        "q2_minus_q49": float(q[0] - q[-1]),
        "q2_minus_q49_lower_95": float(np.quantile(differences, 0.025)),
        "q2_minus_q49_upper_95": float(np.quantile(differences, 0.975)),
    }
    np.savez_compressed(out / f"support_utility_{universe.lower()}.npz", utility=utility, count=utility_count)
    return utility, curve, diagnostics, trend, max_weights


def _regularized_unconstrained_q49(
    root: Path,
    support_orders: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = root / "results/cgc_tahoe_0i"
    cached = np.load(out / "support_grams_g_primary.npz")
    a, b = cached["fold_products"], cached["fold_sum_products"]
    same, cross = cached["eval_same"], cached["eval_cross"]
    folds, fold_sizes = cached["folds"], cached["fold_sizes"]
    gene_count = int(cached["gene_count"])
    utility_sum = np.zeros((50, 93, 4), dtype=np.float64)
    utility_count = np.zeros((50, 93), dtype=np.int16)
    selected_lambdas = np.zeros((5, 2, 50), dtype=np.float64)
    for outer in range(5):
        evaluation = np.flatnonzero(folds == outer)
        outer_folds = [f for f in range(5) if f != outer]
        for direction in range(2):
            outer_gram = _centered_subset_gram(a[direction], b[direction], outer_folds, fold_sizes)
            for target in range(50):
                support = support_orders[target, 0, :49]
                errors = np.zeros(len(LAMBDAS), dtype=np.float64)
                for inner in outer_folds:
                    training_folds = [f for f in outer_folds if f != inner]
                    train_gram = _centered_subset_gram(a[direction], b[direction], training_folds, fold_sizes)
                    validation = same[direction, folds == inner].sum(axis=0) / gene_count
                    gss = train_gram[np.ix_(support, support)]
                    gst = train_gram[support, target]
                    for index, lam in enumerate(LAMBDAS):
                        weight = np.linalg.solve(gss + lam * np.eye(49), gst)
                        errors[index] += float(_residual_from_gram(validation, target, support[None, :], weight[None, :])[0])
                lam = float(LAMBDAS[np.argmin(errors)])
                selected_lambdas[outer, direction, target] = lam
                weight = np.linalg.solve(
                    outer_gram[np.ix_(support, support)] + lam * np.eye(49),
                    outer_gram[support, target],
                )[None, :]
                supports = support[None, :]
                base_weight = np.full((1, 49), 1 / 49)
                for intervention in evaluation:
                    g = cross[intervention] / gene_count
                    g6, g14 = same[0, intervention] / gene_count, same[1, intervention] / gene_count
                    utility_sum[target, intervention] += [
                        _residual_from_gram(g, target, supports, base_weight)[0],
                        _residual_from_gram(g, target, supports, weight)[0],
                        (_residual_from_gram(g6, target, supports, base_weight)[0] + _residual_from_gram(g14, target, supports, base_weight)[0]) / 2,
                        (_residual_from_gram(g6, target, supports, weight)[0] + _residual_from_gram(g14, target, supports, weight)[0]) / 2,
                    ]
                    utility_count[target, intervention] += 1
        print(f"G_PRIMARY unconstrained ridge outer fold {outer + 1}/5", flush=True)
    utility = utility_sum / utility_count[..., None]
    frame = pd.DataFrame(
        [
            {
                "estimator": "unconstrained_ridge",
                "q49": float(utility[..., 1].sum() / utility[..., 0].sum()),
                "raw_r2": float(1 - utility[..., 3].sum() / utility[..., 2].sum()),
                "median_selected_lambda": float(np.median(selected_lambdas)),
                "lambda_grid": ",".join(map(str, LAMBDAS.tolist())),
                "selection": "training-intervention inner folds only",
            }
        ]
    )
    return frame, selected_lambdas


def _q49_residual_null(
    root: Path,
    max_weights: np.ndarray,
    support_orders: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = root / "results/cgc_tahoe_0i"
    group = zarr.open_group(out / "response_tensors.zarr", mode="r")
    response = group["delta_primary"]
    indices = np.load(out / "gene_indices_g_primary.npy")
    flag = np.zeros(response.shape[-1], dtype=bool)
    flag[indices] = True
    folds = _folds(root, list(group.attrs["interventions"]))
    matrices = {fold: np.zeros((np.sum(folds == fold), np.sum(folds == fold)), dtype=np.float64) for fold in range(5)}
    for start in range(0, response.shape[-1], GENE_CHUNK):
        stop = min(response.shape[-1], start + GENE_CHUNK)
        local = flag[start:stop]
        if not local.any():
            continue
        block = np.asarray(response[:, :, :, start:stop], dtype=np.float64)[..., local]
        for fold in range(5):
            evaluation = np.flatnonzero(folds == fold)
            centered = _center(block[:, :, evaluation, :])
            for direction in range(2):
                weight_matrix = np.zeros((50, 50), dtype=np.float64)
                for target in range(50):
                    weight_matrix[target, support_orders[target, 0, :49]] = max_weights[fold, direction, target]
                prediction6 = np.einsum("cd,dpg->cpg", weight_matrix, centered[0], optimize=True)
                prediction14 = np.einsum("cd,dpg->cpg", weight_matrix, centered[1], optimize=True)
                residual6 = centered[0] - prediction6
                residual14 = centered[1] - prediction14
                matrices[fold] += np.einsum("cpg,cqg->pq", residual6, residual14, optimize=True)
        print(f"q49 null genes {stop}/{response.shape[-1]}", flush=True)
    rng = np.random.default_rng(NULL_SEED)
    values = np.empty(NULL_DRAWS, dtype=np.float64)
    denominator = 50 * 2 * 93 * len(indices)
    for draw in range(NULL_DRAWS):
        total = 0.0
        for fold in range(5):
            matrix = matrices[fold]
            total += matrix[np.arange(len(matrix)), rng.permutation(len(matrix))].sum()
        values[draw] = total / denominator
    utility = np.load(out / "support_utility_g_primary.npz")["utility"]
    observed = float(utility[-1, ..., 1].mean())
    summary = {
        "observed_residual_energy": observed,
        "null_q95": float(np.quantile(values, 0.95)),
        "empirical_p": float((1 + np.count_nonzero(values >= observed)) / (1 + NULL_DRAWS)),
    }
    frame = pd.DataFrame(
        {"support_size": 49, "permutation": np.arange(NULL_DRAWS), "seed": NULL_SEED, "residual_energy": values}
    )
    return frame, summary


def run_scaling(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    group = zarr.open_group(out / "response_tensors.zarr", mode="r")
    contexts = list(group.attrs["contexts"])
    interventions = list(group.attrs["interventions"])
    orders = _support_orders(root, contexts)
    curve_frames = []
    diagnostic_rows: list[dict[str, Any]] = []
    trends = []
    utilities: dict[str, np.ndarray] = {}
    primary_max_weights = None
    for universe in UNIVERSES:
        utility, universe_curve, diagnostics, trend, max_weights = _run_universe(root, universe, contexts, interventions, orders)
        utilities[universe] = utility
        curve_frames.append(universe_curve)
        diagnostic_rows.extend(diagnostics)
        trends.append(trend)
        if universe == "G_PRIMARY":
            primary_max_weights = max_weights
    curve = pd.concat(curve_frames, ignore_index=True)
    curve.to_csv(out / "support_q_curve.csv", index=False)
    pd.DataFrame(trends).to_csv(out / "support_trend.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(out / "support_fit_stability.csv", index=False)

    primary = utilities["G_PRIMARY"]
    names = pd.read_csv(root / "results/cgc_tahoe_0b/cell_line_manifest.csv").set_index("cell_line_id")["cell_name"].to_dict()
    max_rows = []
    for context, context_id in enumerate(contexts):
        base = float(primary[-1, context, :, 0].sum())
        residual = float(primary[-1, context, :, 1].sum())
        max_rows.append(
            {
                "context_id": context_id,
                "context_name": names.get(context_id, context_id),
                "support_size": 49,
                "q49": residual / base,
                "v_base": base / 93,
                "v_residual": residual / 93,
                "raw_r2": 1 - primary[-1, context, :, 3].sum() / primary[-1, context, :, 2].sum(),
                "positive_residual_signal": residual > 0,
            }
        )
    pd.DataFrame(max_rows).to_csv(out / "maximal_support_summary.csv", index=False)

    base_context = primary[-1, ..., 0].sum(axis=1)
    residual_context = primary[-1, ..., 1].sum(axis=1)
    fieller_low, fieller_high, fieller_status = _fieller_context_interval(base_context, residual_context)
    ridge, lambdas = _regularized_unconstrained_q49(root, orders)
    ridge.to_csv(out / "regularized_affine_summary.csv", index=False)
    np.save(out / "regularized_affine_selected_lambdas.npy", lambdas)
    null_frame, null_summary = _q49_residual_null(root, primary_max_weights, orders)
    null_frame.to_csv(out / "residual_shuffle_null.csv", index=False)
    np.save(out / "primary_q49_affine_weights.npy", primary_max_weights)

    primary_curve = curve[curve["universe"] == "G_PRIMARY"].set_index("support_size")
    q49 = float(primary_curve.loc[49, "q_pooled"])
    result = {
        "created_at": _now(),
        "primary_universe": "G_PRIMARY",
        "q_values": {str(size): float(primary_curve.loc[size, "q_pooled"]) for size in SUPPORT_SIZES},
        "q49_bootstrap_lower_95": float(primary_curve.loc[49, "bootstrap_lower_95"]),
        "q49_bootstrap_upper_95": float(primary_curve.loc[49, "bootstrap_upper_95"]),
        "q49_fieller_lower_95": fieller_low,
        "q49_fieller_upper_95": fieller_high,
        "q49_fieller_status": fieller_status,
        "q2_minus_q49": float(primary_curve.loc[2, "q_pooled"] - q49),
        "positive_residual_contexts_q49": int(primary_curve.loc[49, "positive_residual_contexts"]),
        "residual_null": null_summary,
        "regularized_best_valid_estimator": ridge.iloc[0].to_dict(),
        "support_sequences": 128,
        "intervention_folds": 5,
        "plate_directions": ["plate6_fit_apply_both", "plate14_fit_apply_both"],
    }
    _write_json(out / "scaling_summary.json", result)
    report = f"""# CGC-0I full-transcriptome context-support scaling

- Primary G_PRIMARY q2/q8/q16/q32/q49: {result['q_values']['2']:.4f} / {result['q_values']['8']:.4f} / {result['q_values']['16']:.4f} / {result['q_values']['32']:.4f} / {result['q_values']['49']:.4f}.
- q49 hierarchical-bootstrap 95% CI: [{result['q49_bootstrap_lower_95']:.4f}, {result['q49_bootstrap_upper_95']:.4f}].
- q49 Fieller interval: [{fieller_low:.4f}, {fieller_high:.4f}] ({fieller_status}).
- Positive residual contexts at maximal support: {result['positive_residual_contexts_q49']}/50.
- Residual identity-shuffle p: {null_summary['empirical_p']:.6g}; null q95: {null_summary['null_q95']:.8g}.
- Training-only selected 0H best valid regularized estimator q49: {float(ridge.iloc[0]['q49']):.4f}.

All computations used the frozen 0C support sequences and intervention-disjoint folds. Utilities were computed from streamed gene-space Gram statistics; no per-cell prediction or dense case-by-gene fit was performed.
"""
    (root / "results/reports/cgc_tahoe_0i_support_scaling.md").write_text(report, encoding="utf-8")
    return result
