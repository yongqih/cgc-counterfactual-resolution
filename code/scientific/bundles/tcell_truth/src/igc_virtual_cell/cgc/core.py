"""Numerically stable, model-free primitives for the CGC theory audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


def two_way_interaction(delta: np.ndarray) -> np.ndarray:
    """Return intervention-by-context interactions for a complete P x C x G tensor."""
    values = np.asarray(delta, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("two_way_interaction requires a finite P x C x G tensor")
    p_mean = values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    c_mean = values.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    grand = values.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    return values - p_mean - c_mean + grand


def masked_two_way_interaction(delta: np.ndarray) -> np.ndarray:
    """Secondary direct mean-formula interaction retaining missing entries as NaN."""
    values = np.asarray(delta, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        p_mean = np.nanmean(values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        c_mean = np.nanmean(values, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        grand = np.nanmean(values, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    result = values - p_mean - c_mean + grand
    result[~np.isfinite(values)] = np.nan
    return result


def context_main_effect_removed(delta: np.ndarray) -> np.ndarray:
    """Return U_c = Delta_c - mean_p Delta_c, before centering across contexts."""
    values = np.asarray(delta, dtype=np.float32)
    return values - values.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)


def flatten_contexts(tensor: np.ndarray) -> np.ndarray:
    """View P x C x F as C x (P F), copying only the axis rearrangement."""
    values = np.asarray(tensor)
    return np.ascontiguousarray(values.transpose(1, 0, 2)).reshape(values.shape[1], -1)


def gram_from_context_tensor(
    left: np.ndarray, right: np.ndarray | None = None, *, block_features: int = 200_000
) -> np.ndarray:
    """Compute a context Gram matrix without making a full float64 feature copy."""
    x = flatten_contexts(left)
    y = x if right is None else flatten_contexts(right)
    if x.shape != y.shape:
        raise ValueError("Cross-Gram tensors must have identical shapes")
    result = np.zeros((x.shape[0], y.shape[0]), dtype=np.float64)
    for start in range(0, x.shape[1], block_features):
        stop = min(start + block_features, x.shape[1])
        xb = x[:, start:stop].astype(np.float64, copy=False)
        yb = y[:, start:stop].astype(np.float64, copy=False)
        result += xb @ yb.T
    return result


def centered_gram(gram: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    """Select contexts and center their feature rows using only the selected contexts."""
    matrix = np.asarray(gram, dtype=np.float64)
    if indices is not None:
        matrix = matrix[np.ix_(indices, indices)]
    n = matrix.shape[0]
    h = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    return h @ matrix @ h


def distances_from_gram(gram: np.ndarray, *, feature_count: int) -> dict[str, np.ndarray]:
    matrix = np.asarray(gram, dtype=np.float64)
    diagonal = np.diag(matrix)
    squared = np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * matrix, 0.0)
    frobenius = np.sqrt(squared / float(feature_count))
    denominator = np.sqrt(np.maximum(diagonal[:, None] * diagonal[None, :], 0.0))
    cosine = np.ones_like(matrix)
    np.divide(matrix, denominator, out=cosine, where=denominator > 0)
    cosine = 1.0 - cosine
    np.fill_diagonal(frobenius, 0.0)
    np.fill_diagonal(cosine, 0.0)
    return {"normalized_frobenius": frobenius, "cosine": cosine, "correlation": cosine.copy()}


def average_intervention_distance(gamma: np.ndarray) -> np.ndarray:
    values = np.asarray(gamma, dtype=np.float32)
    contexts = values.shape[1]
    result = np.zeros((contexts, contexts), dtype=np.float64)
    for left in range(contexts):
        difference = values[:, left + 1 :, :] - values[:, left : left + 1, :]
        if difference.size == 0:
            continue
        per_perturbation = np.sqrt(np.mean(np.square(difference, dtype=np.float64), axis=2))
        result[left, left + 1 :] = per_perturbation.mean(axis=0)
    return result + result.T


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(np.asarray(matrix).shape[0], 1)
    return np.asarray(matrix, dtype=np.float64)[indices]


def geometry_reliability(distance_a: np.ndarray, distance_b: np.ndarray, *, k: int = 3) -> dict[str, float]:
    a = np.asarray(distance_a, dtype=np.float64)
    b = np.asarray(distance_b, dtype=np.float64)
    upper_a, upper_b = upper_triangle(a), upper_triangle(b)
    local = []
    overlaps = []
    nearest = []
    for context in range(a.shape[0]):
        keep = np.arange(a.shape[0]) != context
        local.append(float(spearmanr(a[context, keep], b[context, keep]).statistic))
        neighbors_a = set(np.flatnonzero(keep)[np.argsort(a[context, keep])[:k]].tolist())
        neighbors_b = set(np.flatnonzero(keep)[np.argsort(b[context, keep])[:k]].tolist())
        overlaps.append(len(neighbors_a & neighbors_b) / k)
        nearest.append(int(np.argmin(np.where(keep, a[context], np.inf))) == int(np.argmin(np.where(keep, b[context], np.inf))))
    return {
        "spearman": float(spearmanr(upper_a, upper_b).statistic),
        "pearson": float(pearsonr(upper_a, upper_b).statistic),
        "local_rank_correlation": float(np.nanmean(local)),
        "knn3_overlap": float(np.mean(overlaps)),
        "nearest_context_agreement": float(np.mean(nearest)),
    }


@dataclass(frozen=True)
class SpectrumSummary:
    pc1_fraction: float
    pc50: int
    pc80: int
    pc90: int
    participation_ratio: float
    entropy_rank: float


def spectrum_summary(eigenvalues: np.ndarray, *, positive_only: bool = True) -> SpectrumSummary:
    values = np.asarray(eigenvalues, dtype=np.float64)
    weights = np.maximum(values, 0.0) if positive_only else values
    total = float(weights.sum())
    if total <= 0:
        return SpectrumSummary(float("nan"), 0, 0, 0, float("nan"), float("nan"))
    fractions = weights / total
    cumulative = np.cumsum(fractions)
    count = lambda threshold: int(np.searchsorted(cumulative, threshold, side="left") + 1)
    nonzero = fractions[fractions > 0]
    participation = total * total / float(np.square(weights).sum())
    entropy = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    return SpectrumSummary(float(fractions[0]), count(0.5), count(0.8), count(0.9), participation, entropy)


def eigenspectrum(gram: np.ndarray) -> np.ndarray:
    matrix = (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T) * 0.5
    return np.linalg.eigvalsh(matrix)[::-1]


def oracle_projection_residual(
    base_gram: np.ndarray, target: int, references: Iterable[int], *, rcond: float = 1.0e-10
) -> float:
    """Leakage-safe oracle residual using reference-only operator centering.

    ``base_gram`` is the Gram matrix of U_c = Delta_c - mean_p Delta_c.  The
    target operator and all reference operators are oriented relative to the
    reference-only mean, avoiding the trivial sum-to-zero containment that would
    result from globally centering with the held-out target included.
    """
    q = np.asarray(base_gram, dtype=np.float64)
    refs = np.asarray(list(references), dtype=np.int64)
    if target in refs or len(refs) < 2:
        raise ValueError("Oracle references must exclude target and contain at least two contexts")
    qrr = q[np.ix_(refs, refs)]
    h = np.eye(len(refs)) - np.ones((len(refs), len(refs))) / len(refs)
    train = h @ qrr @ h
    mean_target = float(q[refs, target].mean())
    mean_mean = float(qrr.mean())
    target_norm = float(q[target, target] - 2.0 * mean_target + mean_mean)
    raw_cross = q[refs, target] - qrr.mean(axis=1) - mean_target + mean_mean
    cross = h @ raw_cross
    projection = float(cross @ np.linalg.pinv(train, rcond=rcond) @ cross)
    if target_norm <= 0:
        return float("nan")
    return float(np.sqrt(max(target_norm - projection, 0.0) / target_norm))


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    xr = rankdata(np.asarray(x, dtype=np.float64))
    yr = rankdata(np.asarray(y, dtype=np.float64))
    z = np.asarray(controls, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    design = np.column_stack([np.ones(len(xr)), z])
    x_residual = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_residual = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(pearsonr(x_residual, y_residual).statistic)


def permutation_alignment(
    distance_reference: np.ndarray,
    distance_candidate: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    reference = upper_triangle(distance_reference)
    observed = float(spearmanr(reference, upper_triangle(distance_candidate)).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        order = rng.permutation(distance_candidate.shape[0])
        permuted = distance_candidate[np.ix_(order, order)]
        null[draw] = spearmanr(reference, upper_triangle(permuted)).statistic
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (draws + 1))
    return observed, p_value, null
