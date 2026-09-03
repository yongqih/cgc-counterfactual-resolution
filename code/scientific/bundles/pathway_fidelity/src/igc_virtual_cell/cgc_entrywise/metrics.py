from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METRIC_NAMES = ("pearson", "cosine", "gene_r2", "mse", "variance_retention")


@dataclass
class ExcessGeometry:
    gram: np.ndarray
    truth_dot_excess: np.ndarray
    truth_sum: np.ndarray
    excess_sum: np.ndarray


def metric_arrays(
    truth_norm: np.ndarray,
    prediction_norm: np.ndarray,
    truth_dot_prediction: np.ndarray,
    truth_sum: np.ndarray,
    prediction_sum: np.ndarray,
    gene_count: int,
) -> np.ndarray:
    truth_norm = np.asarray(truth_norm, dtype=np.float64)
    prediction_norm = np.asarray(prediction_norm, dtype=np.float64)
    dot = np.asarray(truth_dot_prediction, dtype=np.float64)
    truth_sum = np.asarray(truth_sum, dtype=np.float64)
    prediction_sum = np.asarray(prediction_sum, dtype=np.float64)
    error = truth_norm + prediction_norm - 2.0 * dot
    truth_var = truth_norm - truth_sum * truth_sum / gene_count
    pred_var = prediction_norm - prediction_sum * prediction_sum / gene_count
    covariance = dot - truth_sum * prediction_sum / gene_count
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = covariance / np.sqrt(truth_var * pred_var)
        cosine = dot / np.sqrt(truth_norm * prediction_norm)
        gene_r2 = 1.0 - error / truth_var
        mse = error / gene_count
        retention = pred_var / truth_var
    pearson[(truth_var <= 0) | (pred_var <= 0)] = np.nan
    cosine[(truth_norm <= 0) | (prediction_norm <= 0)] = np.nan
    gene_r2[truth_var <= 0] = np.nan
    retention[truth_var <= 0] = np.nan
    return np.stack((pearson, cosine, gene_r2, mse, retention), axis=-1)


def excess_geometry(
    gram4: np.ndarray,
    sample_sums: np.ndarray,
    target: int,
    sources: np.ndarray,
) -> ExcessGeometry:
    """Full 93-by-93 Gram of target-minus-equal-source-mean responses."""

    sources = np.asarray(sources, dtype=np.int64)
    count = len(sources)
    target_target = np.asarray(gram4[target, :, target, :], dtype=np.float64)
    target_source = np.zeros_like(target_target)
    source_target = np.zeros_like(target_target)
    source_source = np.zeros_like(target_target)
    for source in sources:
        target_source += np.asarray(gram4[target, :, source, :], dtype=np.float64)
        source_target += np.asarray(gram4[source, :, target, :], dtype=np.float64)
        for other in sources:
            source_source += np.asarray(gram4[source, :, other, :], dtype=np.float64)
    gram = target_target - target_source / count - source_target / count + source_source / (count * count)
    truth_dot_excess = target_target - target_source / count
    target_sum = np.asarray(sample_sums[target * 93 : (target + 1) * 93], dtype=np.float64)
    source_sum = np.zeros(93, dtype=np.float64)
    for source in sources:
        source_sum += np.asarray(sample_sums[source * 93 : (source + 1) * 93], dtype=np.float64)
    excess_sum = target_sum - source_sum / count
    return ExcessGeometry(
        gram=gram,
        truth_dot_excess=truth_dot_excess,
        truth_sum=target_sum,
        excess_sum=excess_sum,
    )


def matched_context_gram(gram4: np.ndarray) -> np.ndarray:
    result = np.empty((93, 50, 50), dtype=np.float64)
    for intervention in range(93):
        result[intervention] = np.asarray(
            gram4[:, intervention, :, intervention], dtype=np.float64
        )
    return result


def matched_residual_cross(
    cross_by_intervention: np.ndarray,
    target: int,
    sources: np.ndarray,
    weights6: np.ndarray,
    weights14: np.ndarray,
) -> np.ndarray:
    """Cross-plate dot of target minus weighted-source residual, per intervention."""

    sources = np.asarray(sources, dtype=np.int64)
    block = cross_by_intervention[:, sources][:, :, sources]
    source_target = cross_by_intervention[:, sources, target]
    target_source = cross_by_intervention[:, target, sources]
    return (
        cross_by_intervention[:, target, target]
        - np.einsum("pi,pi->p", weights6, source_target, optimize=True)
        - np.einsum("pi,pi->p", weights14, target_source, optimize=True)
        + np.einsum("pi,pij,pj->p", weights6, block, weights14, optimize=True)
    )


def weighted_same_stats(
    same_by_intervention: np.ndarray,
    sample_sums: np.ndarray,
    target: int,
    sources: np.ndarray,
    weights: np.ndarray,
    truth_excess: ExcessGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-response and excess-response metric arrays for context weights."""

    sources = np.asarray(sources, dtype=np.int64)
    uniform = np.full(len(sources), 1.0 / len(sources), dtype=np.float64)
    block = same_by_intervention[:, sources][:, :, sources]
    target_source = same_by_intervention[:, target, sources]
    truth_norm = same_by_intervention[:, target, target]
    prediction_norm = np.einsum("pi,pij,pj->p", weights, block, weights, optimize=True)
    truth_dot_prediction = np.einsum("pi,pi->p", weights, target_source, optimize=True)
    source_sums = np.stack(
        [sample_sums[source * 93 : (source + 1) * 93] for source in sources], axis=1
    ).astype(np.float64)
    prediction_sum = np.einsum("pi,pi->p", weights, source_sums, optimize=True)
    # Return raw sufficient statistics; the caller supplies the frozen gene
    # count when converting them to metrics.
    delta_weights = weights - uniform
    excess_prediction_norm = np.einsum(
        "pi,pij,pj->p", delta_weights, block, delta_weights, optimize=True
    )
    truth_dot_excess_prediction = np.einsum(
        "pi,pi->p",
        delta_weights,
        target_source - np.einsum("i,pij->pj", uniform, block, optimize=True),
        optimize=True,
    )
    excess_prediction_sum = np.einsum("pi,pi->p", delta_weights, source_sums, optimize=True)
    raw_full = np.stack(
        (
            truth_norm,
            prediction_norm,
            truth_dot_prediction,
            truth_excess.truth_sum,
            prediction_sum,
        ),
        axis=-1,
    )
    raw_excess = np.stack(
        (
            np.diag(truth_excess.gram),
            excess_prediction_norm,
            truth_dot_excess_prediction,
            truth_excess.excess_sum,
            excess_prediction_sum,
        ),
        axis=-1,
    )
    return raw_full, raw_excess


def metrics_from_raw(raw: np.ndarray, gene_count: int) -> np.ndarray:
    return metric_arrays(raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4], gene_count)


def offset_stats(
    geometry: ExcessGeometry,
    sentinels: np.ndarray,
    alpha: float,
    truth_full_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M1 raw full/excess stats and same-plate residual norm for all targets."""

    intervention_count = geometry.gram.shape[0]
    truth_norm_excess = np.diag(geometry.gram)
    truth_norm_full = np.asarray(truth_full_norm, dtype=np.float64)
    full = np.empty((intervention_count, 5), dtype=np.float64)
    excess = np.empty_like(full)
    residual_norm = np.empty(intervention_count, dtype=np.float64)
    for intervention in range(intervention_count):
        selected = sentinels[intervention]
        if len(selected):
            truth_dot_offset = float(geometry.gram[intervention, selected].mean())
            offset_norm = float(geometry.gram[np.ix_(selected, selected)].mean())
            offset_sum = float(geometry.excess_sum[selected].mean())
            truth_full_dot_offset = float(geometry.truth_dot_excess[intervention, selected].mean())
        else:
            truth_dot_offset = offset_norm = offset_sum = truth_full_dot_offset = 0.0
        prediction_excess_norm = alpha * alpha * offset_norm
        truth_dot_prediction_excess = alpha * truth_dot_offset
        prediction_excess_sum = alpha * offset_sum
        residual_norm[intervention] = (
            truth_norm_excess[intervention]
            - 2.0 * truth_dot_prediction_excess
            + prediction_excess_norm
        )
        excess[intervention] = (
            truth_norm_excess[intervention],
            prediction_excess_norm,
            truth_dot_prediction_excess,
            geometry.excess_sum[intervention],
            prediction_excess_sum,
        )
        truth_dot_error = (
            geometry.truth_dot_excess[intervention, intervention]
            - alpha * truth_full_dot_offset
        )
        prediction_full_sum = geometry.truth_sum[intervention] - (
            geometry.excess_sum[intervention] - prediction_excess_sum
        )
        full[intervention] = (
            truth_norm_full[intervention],
            truth_norm_full[intervention] - 2.0 * truth_dot_error + residual_norm[intervention],
            truth_norm_full[intervention] - truth_dot_error,
            geometry.truth_sum[intervention],
            prediction_full_sum,
        )
    return full, excess, residual_norm


def offset_residual_cross(
    cross_excess: np.ndarray,
    sentinels: np.ndarray,
    alpha6: float,
    alpha14: float,
) -> np.ndarray:
    result = np.empty(cross_excess.shape[0], dtype=np.float64)
    for intervention in range(cross_excess.shape[0]):
        selected = sentinels[intervention]
        if len(selected):
            right = float(cross_excess[intervention, selected].mean())
            left = float(cross_excess[selected, intervention].mean())
            offset = float(cross_excess[np.ix_(selected, selected)].mean())
        else:
            right = left = offset = 0.0
        result[intervention] = (
            cross_excess[intervention, intervention]
            - alpha14 * right
            - alpha6 * left
            + alpha6 * alpha14 * offset
        )
    return result
