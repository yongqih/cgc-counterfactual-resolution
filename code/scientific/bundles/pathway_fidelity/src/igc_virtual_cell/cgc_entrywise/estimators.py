from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from igc_virtual_cell.cgc_entrywise.core import affine_ridge_weights, lowrank_context_weights
from igc_virtual_cell.cgc_entrywise.metrics import excess_geometry


ALPHAS = np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
LAMBDAS = np.asarray(
    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    dtype=np.float64,
)
RANKS = np.asarray([1, 2, 4, 8, 16, 24, 32], dtype=np.int64)


@dataclass(frozen=True)
class SelectedParameters:
    alpha: float
    ridge: float
    rank: int
    alpha_loss: float
    ridge_loss: float
    rank_loss: float
    pseudo_targets: int
    validation_cases: int


def sentinel_matrix(order: np.ndarray, count: int) -> np.ndarray:
    """One row per hidden intervention; target identity is removed first."""

    order = np.asarray(order, dtype=np.int64)
    if count == 0:
        return np.empty((len(order), 0), dtype=np.int64)
    result = np.empty((len(order), count), dtype=np.int64)
    for hidden in range(len(order)):
        result[hidden] = order[order != hidden][:count]
        if hidden in result[hidden]:
            raise RuntimeError("SEALED_INTERVENTION_IN_SENTINELS")
    return result


def _batch_sum(
    items: np.ndarray,
    order: np.ndarray,
    count: int,
    episodes: np.ndarray | None = None,
) -> np.ndarray:
    """Sum first count order items after removing each row's hidden identity."""

    intervention_count = len(order)
    shape = (intervention_count,) + items.shape[1:]
    if count == 0:
        return np.zeros(shape, dtype=np.float64)
    if episodes is not None:
        result = np.full(shape, np.nan, dtype=np.float64)
        for hidden in map(int, episodes):
            selected = order[order != hidden][:count]
            result[hidden] = items[selected].sum(axis=0, dtype=np.float64)
        return result
    base = np.asarray(items[order[:count]].sum(axis=0), dtype=np.float64)
    result = np.broadcast_to(base, shape).copy()
    position = np.empty(intervention_count, dtype=np.int64)
    position[order] = np.arange(intervention_count)
    affected = np.flatnonzero(position < count)
    # Do not form a sum that ever contains the sealed item and subtract it
    # afterward. That algebraic shortcut leaks through finite-precision
    # cancellation for adversarially large hidden vectors. Reconstruct only the
    # affected rows from an information set that excludes the target ab initio.
    for hidden in affected:
        selected = order[order != hidden][:count]
        result[hidden] = items[selected].sum(axis=0, dtype=np.float64)
    return result


def fit_batches(
    same_by_intervention: np.ndarray,
    full_same4: np.ndarray,
    target: int,
    sources: np.ndarray,
    order: np.ndarray,
    k: int,
    ridge: float,
    rank: int,
    context_null: int,
    sentinel_null_map: np.ndarray,
    episodes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit all 93 hidden-intervention episodes without accessing a hidden row."""

    sources = np.asarray(sources, dtype=np.int64)
    support = len(sources)
    uniform = np.full(support, 1.0 / support, dtype=np.float64)
    episode_ids = np.arange(93, dtype=np.int64) if episodes is None else np.asarray(episodes, dtype=np.int64)
    if k == 0 or support == 1:
        equal = np.broadcast_to(uniform, (93, support)).copy()
        return {
            "M0": equal,
            "M2": equal.copy(),
            "M3": equal.copy(),
            "NULL_CONTEXT": equal.copy(),
            "NULL_SENTINEL": equal.copy(),
            "NULL_REFERENCE": equal[:, ::-1].copy() if support > 1 else equal.copy(),
        }
    item_gram = same_by_intervention[:, sources][:, :, sources]
    item_cross = same_by_intervention[:, sources, target]
    grams = _batch_sum(item_gram, order, k, episode_ids if episodes is not None else None)
    crosses = _batch_sum(item_cross, order, k, episode_ids if episodes is not None else None)
    weights = np.full((93, support), np.nan, dtype=np.float64)
    lowrank = np.full_like(weights, np.nan)
    reference = item_gram.sum(axis=0, dtype=np.float64)
    for hidden in episode_ids:
        weights[hidden] = affine_ridge_weights(grams[hidden], crosses[hidden], ridge)
        lowrank[hidden] = lowrank_context_weights(
            reference, grams[hidden], crosses[hidden], rank
        )

    context_cross_items = same_by_intervention[:, sources, context_null]
    context_crosses = _batch_sum(
        context_cross_items, order, k, episode_ids if episodes is not None else None
    )
    context_weights = np.full_like(weights, np.nan)
    for hidden in episode_ids:
        context_weights[hidden] = affine_ridge_weights(
            grams[hidden], context_crosses[hidden], ridge
        )

    null_gram_items = np.empty_like(item_gram)
    null_cross_items = np.empty_like(item_cross)
    for intervention in range(93):
        mapped = int(sentinel_null_map[intervention])
        null_gram_items[intervention] = same_by_intervention[mapped][np.ix_(sources, sources)]
        null_cross_items[intervention] = np.asarray(
            [full_same4[source, mapped, target, intervention] for source in sources],
            dtype=np.float64,
        )
    null_grams = _batch_sum(
        null_gram_items, order, k, episode_ids if episodes is not None else None
    )
    null_crosses = _batch_sum(
        null_cross_items, order, k, episode_ids if episodes is not None else None
    )
    sentinel_weights = np.full_like(weights, np.nan)
    for hidden in episode_ids:
        sentinel_weights[hidden] = affine_ridge_weights(
            null_grams[hidden], null_crosses[hidden], ridge
        )

    return {
        "M0": np.broadcast_to(uniform, (93, support)).copy(),
        "M2": weights,
        "M3": lowrank,
        "NULL_CONTEXT": context_weights,
        "NULL_SENTINEL": sentinel_weights,
        "NULL_REFERENCE": np.roll(weights, 1, axis=1),
    }


def _validation_energy(
    matched: np.ndarray,
    validation: np.ndarray,
    pseudo_target: int,
    sources: np.ndarray,
    weights: np.ndarray,
) -> float:
    block = matched[validation][:, sources][:, :, sources].sum(axis=0, dtype=np.float64)
    cross = matched[validation][:, sources, pseudo_target].sum(axis=0, dtype=np.float64)
    truth = float(matched[validation, pseudo_target, pseudo_target].sum(dtype=np.float64))
    return float(truth - 2.0 * weights @ cross + weights @ block @ weights)


def select_parameters(
    matched: np.ndarray,
    full_same4: np.ndarray,
    sample_sums: np.ndarray,
    target: int,
    m: int,
    k: int,
    support_orders: np.ndarray,
    sentinel_orders: np.ndarray,
    folds: np.ndarray,
    geometry_cache: dict[tuple[int, tuple[int, ...]], object] | None = None,
    support_sequence: int | None = None,
) -> SelectedParameters:
    """Select within the episode's observed reference-context support.

    Pooling paths is allowed only when their reference information sets are
    identical (in particular m=49). Otherwise the caller must name one path.
    """

    if m <= 1 or k == 0:
        return SelectedParameters(0.0, float(LAMBDAS[0]), 0, np.nan, np.nan, np.nan, 0, 0)
    if support_sequence is None:
        allowed = set(map(int, support_orders[0, :m]))
        if any(set(map(int, row[:m])) != allowed for row in support_orders):
            raise ValueError("BUDGET_LEAKAGE: select one support_sequence before tuning")
        sequences = range(support_orders.shape[0])
    else:
        if not 0 <= support_sequence < support_orders.shape[0]:
            raise ValueError("Invalid support_sequence")
        sequences = (support_sequence,)
    for sequence in sequences:
        support_set = support_orders[sequence, :m]
        if len(set(map(int, support_set))) != m or target in support_set:
            raise ValueError("Invalid or target-contaminated reference support")
    alpha_losses = np.zeros(len(ALPHAS), dtype=np.float64)
    ridge_losses = np.zeros(len(LAMBDAS), dtype=np.float64)
    rank_losses = np.zeros(len(RANKS), dtype=np.float64)
    rank_cases = np.zeros(len(RANKS), dtype=np.int64)
    pseudo_count = 0
    validation_cases = 0
    for sequence in sequences:
        support_set = np.asarray(support_orders[sequence, :m], dtype=np.int64)
        order = np.asarray(sentinel_orders[sequence], dtype=np.int64)
        for pseudo_target in support_set[: min(5, len(support_set))]:
            sources = support_set[support_set != pseudo_target]
            key = (int(pseudo_target), tuple(map(int, sources)))
            if geometry_cache is not None and key in geometry_cache:
                geometry = geometry_cache[key]
            else:
                geometry = excess_geometry(full_same4, sample_sums, int(pseudo_target), sources)
                if geometry_cache is not None:
                    geometry_cache[key] = geometry
            pseudo_count += 1
            for fold in range(5):
                training = np.flatnonzero(folds != fold)
                validation = np.flatnonzero(folds == fold)
                chosen = order[np.isin(order, training)][: min(k, len(training))]
                if not len(chosen) or not len(validation):
                    continue
                gram = matched[chosen][:, sources][:, :, sources].sum(axis=0, dtype=np.float64)
                cross = matched[chosen][:, sources, pseudo_target].sum(axis=0, dtype=np.float64)
                for index, value in enumerate(LAMBDAS):
                    weights = affine_ridge_weights(gram, cross, float(value))
                    ridge_losses[index] += _validation_energy(
                        matched, validation, int(pseudo_target), sources, weights
                    )
                reference = matched[training][:, sources][:, :, sources].sum(
                    axis=0, dtype=np.float64
                )
                for index, rank in enumerate(RANKS):
                    if rank > len(sources) - 1:
                        continue
                    weights = lowrank_context_weights(
                        reference, gram, cross, int(rank)
                    )
                    rank_losses[index] += _validation_energy(
                        matched, validation, int(pseudo_target), sources, weights
                    )
                    rank_cases[index] += 1
                offset_norm = float(geometry.gram[np.ix_(chosen, chosen)].mean())
                truth_dot_offset = geometry.gram[np.ix_(validation, chosen)].mean(axis=1)
                truth_norm = np.diag(geometry.gram)[validation]
                for index, alpha in enumerate(ALPHAS):
                    alpha_losses[index] += float(
                        np.sum(truth_norm - 2.0 * alpha * truth_dot_offset + alpha * alpha * offset_norm)
                    )
                validation_cases += len(validation)
    if validation_cases == 0:
        return SelectedParameters(0.0, float(LAMBDAS[0]), 0, np.nan, np.nan, np.nan, pseudo_count, 0)
    alpha_index = int(np.argmin(alpha_losses))
    ridge_index = int(np.argmin(ridge_losses))
    valid_ranks = np.flatnonzero(rank_cases > 0)
    if len(valid_ranks):
        rank_index = int(valid_ranks[np.argmin(rank_losses[valid_ranks])])
        selected_rank = int(RANKS[rank_index])
        selected_rank_loss = float(rank_losses[rank_index])
    else:
        selected_rank = 0
        selected_rank_loss = np.nan
    return SelectedParameters(
        alpha=float(ALPHAS[alpha_index]),
        ridge=float(LAMBDAS[ridge_index]),
        rank=selected_rank,
        alpha_loss=float(alpha_losses[alpha_index]),
        ridge_loss=float(ridge_losses[ridge_index]),
        rank_loss=selected_rank_loss,
        pseudo_targets=pseudo_count,
        validation_cases=validation_cases,
    )
