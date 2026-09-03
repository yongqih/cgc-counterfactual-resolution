from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


RCOND = 1e-10


def sample_index(context: int, intervention: int, intervention_count: int = 93) -> int:
    return int(context) * intervention_count + int(intervention)


def _as_index(values: Iterable[int] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.int64)


@dataclass(frozen=True)
class LinearPrediction:
    """Sparse linear combination of observed response rows."""

    indices: np.ndarray
    weights: np.ndarray

    @classmethod
    def combine(cls, indices: Iterable[int], weights: Iterable[float]) -> "LinearPrediction":
        merged: dict[int, float] = {}
        for index, weight in zip(indices, weights, strict=True):
            merged[int(index)] = merged.get(int(index), 0.0) + float(weight)
        ordered = sorted(index for index, value in merged.items() if value != 0.0)
        return cls(
            indices=np.asarray(ordered, dtype=np.int64),
            weights=np.asarray([merged[index] for index in ordered], dtype=np.float64),
        )

    def materialize(self, responses: np.ndarray) -> np.ndarray:
        return np.einsum("i,ig->g", self.weights, responses[self.indices], optimize=True)

    def signature(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        return tuple(map(int, self.indices)), tuple(map(float, self.weights))


class SealedGramView:
    """Episode-local fit view that refuses every hidden-row access.

    The global Gram cache is a read-only acceleration artifact. Estimators only
    receive this view, which checks every selected row and column before exposing
    a statistic. Evaluation code owns truth separately.
    """

    def __init__(
        self,
        same: np.ndarray,
        hidden_context: int,
        hidden_intervention: int,
        intervention_count: int = 93,
    ) -> None:
        if same.ndim != 2 or same.shape[0] != same.shape[1]:
            raise ValueError("same-plate Gram must be square")
        self._same = same
        self.hidden_index = sample_index(hidden_context, hidden_intervention, intervention_count)
        self.intervention_count = intervention_count
        self.accessed: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def gram(self, rows: Iterable[int] | np.ndarray, cols: Iterable[int] | np.ndarray) -> np.ndarray:
        row = _as_index(rows)
        col = _as_index(cols)
        if np.any(row == self.hidden_index) or np.any(col == self.hidden_index):
            raise RuntimeError("SEALED_TARGET_ACCESS_ATTEMPT")
        self.accessed.append((tuple(map(int, row)), tuple(map(int, col))))
        return np.asarray(self._same[np.ix_(row, col)], dtype=np.float64)

    def dot(self, rows: Iterable[int] | np.ndarray, col: int) -> np.ndarray:
        return self.gram(rows, np.asarray([col], dtype=np.int64))[:, 0]


def affine_basis(support: int) -> tuple[np.ndarray, np.ndarray]:
    if support < 1:
        raise ValueError("support must be positive")
    origin = np.full(support, 1.0 / support, dtype=np.float64)
    if support == 1:
        return origin, np.empty((1, 0), dtype=np.float64)
    contrasts = np.vstack((np.eye(support - 1), -np.ones((1, support - 1))))
    basis, _ = np.linalg.qr(contrasts, mode="reduced")
    return origin, basis


def affine_ridge_weights(gram: np.ndarray, cross: np.ndarray, ridge: float) -> np.ndarray:
    support = int(len(cross))
    origin, basis = affine_basis(support)
    if support == 1:
        return origin
    reduced = basis.T @ np.asarray(gram, dtype=np.float64) @ basis
    rhs = basis.T @ (np.asarray(cross, dtype=np.float64) - gram @ origin)
    matrix = reduced + float(ridge) * np.eye(support - 1)
    try:
        coordinates = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        coordinates = np.linalg.pinv(matrix, rcond=RCOND) @ rhs
    weights = origin + basis @ coordinates
    weights += (1.0 - weights.sum()) / support
    if abs(weights.sum() - 1.0) > 1e-9:
        raise RuntimeError("AFFINE_CONSTRAINT_FAIL")
    return weights


def lowrank_context_weights(
    reference_gram: np.ndarray,
    sentinel_gram: np.ndarray,
    sentinel_cross: np.ndarray,
    rank: int,
) -> np.ndarray:
    support = int(len(sentinel_cross))
    origin, affine = affine_basis(support)
    if support == 1 or rank <= 0:
        return origin
    covariance = affine.T @ np.asarray(reference_gram, dtype=np.float64) @ affine
    values, vectors = np.linalg.eigh(covariance)
    keep = np.flatnonzero(values > max(float(values.max(initial=0.0)), 1.0) * RCOND)
    if not len(keep):
        return origin
    keep = keep[np.argsort(values[keep])[::-1][: min(int(rank), len(keep))]]
    subspace = affine @ vectors[:, keep]
    reduced = subspace.T @ np.asarray(sentinel_gram, dtype=np.float64) @ subspace
    rhs = subspace.T @ (np.asarray(sentinel_cross, dtype=np.float64) - sentinel_gram @ origin)
    coordinates = np.linalg.pinv(reduced, rcond=RCOND) @ rhs
    weights = origin + subspace @ coordinates
    weights += (1.0 - weights.sum()) / support
    if abs(weights.sum() - 1.0) > 1e-9:
        raise RuntimeError("LOWRANK_AFFINE_CONSTRAINT_FAIL")
    return weights


def matched_fit_statistics(
    view: SealedGramView,
    source_contexts: np.ndarray,
    target_context: int,
    sentinels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support = len(source_contexts)
    gram = np.zeros((support, support), dtype=np.float64)
    cross = np.zeros(support, dtype=np.float64)
    for intervention in map(int, sentinels):
        source = np.asarray(
            [sample_index(context, intervention, view.intervention_count) for context in source_contexts],
            dtype=np.int64,
        )
        target = sample_index(target_context, intervention, view.intervention_count)
        gram += view.gram(source, source)
        cross += view.dot(source, target)
    return gram, cross


def reference_context_gram(
    view: SealedGramView,
    source_contexts: np.ndarray,
    interventions: np.ndarray,
) -> np.ndarray:
    support = len(source_contexts)
    result = np.zeros((support, support), dtype=np.float64)
    for intervention in map(int, interventions):
        rows = np.asarray(
            [sample_index(context, intervention, view.intervention_count) for context in source_contexts],
            dtype=np.int64,
        )
        result += view.gram(rows, rows)
    return result


def support_mean_prediction(
    source_contexts: np.ndarray,
    intervention: int,
    intervention_count: int = 93,
) -> LinearPrediction:
    indices = [sample_index(context, intervention, intervention_count) for context in source_contexts]
    return LinearPrediction.combine(indices, np.full(len(indices), 1.0 / len(indices)))


def weighted_context_prediction(
    source_contexts: np.ndarray,
    intervention: int,
    weights: np.ndarray,
    intervention_count: int = 93,
) -> LinearPrediction:
    indices = [sample_index(context, intervention, intervention_count) for context in source_contexts]
    return LinearPrediction.combine(indices, weights)


def sentinel_offset_prediction(
    source_contexts: np.ndarray,
    target_context: int,
    target_intervention: int,
    sentinels: np.ndarray,
    alpha: float,
    intervention_count: int = 93,
) -> LinearPrediction:
    support = len(source_contexts)
    indices: list[int] = []
    weights: list[float] = []
    for context in source_contexts:
        indices.append(sample_index(int(context), target_intervention, intervention_count))
        weights.append(1.0 / support)
    if len(sentinels):
        scale = float(alpha) / len(sentinels)
        for intervention in map(int, sentinels):
            indices.append(sample_index(target_context, intervention, intervention_count))
            weights.append(scale)
            for context in source_contexts:
                indices.append(sample_index(int(context), intervention, intervention_count))
                weights.append(-scale / support)
    prediction = LinearPrediction.combine(indices, weights)
    hidden = sample_index(target_context, target_intervention, intervention_count)
    if hidden in set(map(int, prediction.indices)):
        raise RuntimeError("SEALED_TARGET_IN_PREDICTION")
    return prediction
