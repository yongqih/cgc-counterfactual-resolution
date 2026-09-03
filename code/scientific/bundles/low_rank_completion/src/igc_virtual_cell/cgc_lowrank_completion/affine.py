from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from igc_virtual_cell.cgc_entrywise.core import affine_ridge_weights

from .core import CONTEXTS, ENTRIES, INTERVENTIONS, prediction_weights


@dataclass(frozen=True)
class FrozenAffineParameters:
    ridge: np.ndarray


def load_frozen_affine_parameters(source_root: Path) -> FrozenAffineParameters:
    ridge = np.full((2, CONTEXTS), np.nan, dtype=np.float64)
    cache = source_root.resolve() / "results/cgc_entrywise_compression/_cache"
    paths = sorted(cache.glob("target_*.npz"))
    if len(paths) != CONTEXTS:
        raise RuntimeError("LOW_RANK_AFFINE_TARGET_CACHE_COUNT_FAIL")
    for context, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            rows = json.loads(str(archive["parameter_rows_json"]))
        selected = [row for row in rows if int(row["m"]) == 49 and int(row["k"]) == 92]
        if len(selected) != 2:
            raise RuntimeError("LOW_RANK_AFFINE_PARAMETER_COUNT_FAIL")
        for row in selected:
            if bool(row["target_outcome_used"]):
                raise RuntimeError("LOW_RANK_AFFINE_PARAMETER_LEAKAGE")
            plate = ("plate6", "plate14").index(str(row["plate"]))
            ridge[plate, context] = float(row["ridge_lambda"])
    if not np.isfinite(ridge).all():
        raise RuntimeError("LOW_RANK_AFFINE_PARAMETER_NONFINITE")
    return FrozenAffineParameters(ridge)


class AdditiveCompletedGram:
    """Same-plate Gram accessor that never reads a hidden row or column."""

    def __init__(self, full_gram: np.ndarray, train: np.ndarray, missing: np.ndarray):
        self.full_gram = full_gram
        self.train = np.asarray(train, dtype=np.int64)
        self.missing = np.asarray(missing, dtype=np.int64)
        self.observed_position = np.full(ENTRIES, -1, dtype=np.int64)
        self.observed_position[self.train] = np.arange(len(self.train), dtype=np.int64)
        self.missing_position = np.full(ENTRIES, -1, dtype=np.int64)
        self.missing_position[self.missing] = np.arange(len(self.missing), dtype=np.int64)
        self.impute = prediction_weights(self.train, self.missing, None).additive
        train_gram = np.asarray(full_gram[np.ix_(self.train, self.train)], dtype=np.float64)
        self.missing_observed = np.asarray(self.impute @ train_gram, dtype=np.float64)
        self.missing_missing = np.asarray(self.missing_observed @ self.impute.T, dtype=np.float64)

    def coefficients(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        result = np.zeros((len(indices), len(self.train)), dtype=np.float64)
        observed = self.observed_position[indices] >= 0
        if np.any(observed):
            result[np.flatnonzero(observed), self.observed_position[indices[observed]]] = 1.0
        if np.any(~observed):
            missing = self.missing_position[indices[~observed]]
            if np.any(missing < 0):
                raise RuntimeError("LOW_RANK_COMPLETED_GRAM_UNKNOWN_ENTRY")
            result[~observed] = self.impute[missing]
        return result

    def gram(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        left_obs = self.observed_position[left] >= 0
        right_obs = self.observed_position[right] >= 0
        result = np.empty((len(left), len(right)), dtype=np.float64)
        if np.any(left_obs) and np.any(right_obs):
            # Both sides are certified observed before the frozen Gram is read.
            result[np.ix_(left_obs, right_obs)] = np.asarray(
                self.full_gram[np.ix_(left[left_obs], right[right_obs])], dtype=np.float64
            )
        if np.any(~left_obs) and np.any(right_obs):
            lm = self.missing_position[left[~left_obs]]
            ro = self.observed_position[right[right_obs]]
            result[np.ix_(~left_obs, right_obs)] = self.missing_observed[np.ix_(lm, ro)]
        if np.any(left_obs) and np.any(~right_obs):
            lo = self.observed_position[left[left_obs]]
            rm = self.missing_position[right[~right_obs]]
            result[np.ix_(left_obs, ~right_obs)] = self.missing_observed[np.ix_(rm, lo)].T
        if np.any(~left_obs) and np.any(~right_obs):
            lm = self.missing_position[left[~left_obs]]
            rm = self.missing_position[right[~right_obs]]
            result[np.ix_(~left_obs, ~right_obs)] = self.missing_missing[np.ix_(lm, rm)]
        return result


def matched_affine_prediction_weights(
    full_gram: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    ridge_by_context: np.ndarray,
) -> np.ndarray:
    """Rerun the frozen affine estimator under one balanced outer mask."""

    train = np.asarray(train, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    completed = AdditiveCompletedGram(full_gram, train, targets)
    result = np.zeros((len(targets), len(train)), dtype=np.float64)
    for row, target in enumerate(map(int, targets)):
        context = target // INTERVENTIONS
        intervention = target % INTERVENTIONS
        sources = np.asarray(
            [other for other in range(CONTEXTS) if other != context], dtype=np.int64
        )
        gram = np.zeros((CONTEXTS - 1, CONTEXTS - 1), dtype=np.float64)
        cross = np.zeros(CONTEXTS - 1, dtype=np.float64)
        for sentinel in range(INTERVENTIONS):
            if sentinel == intervention:
                continue
            source_entries = sources * INTERVENTIONS + sentinel
            target_sentinel = np.asarray([context * INTERVENTIONS + sentinel], dtype=np.int64)
            gram += completed.gram(source_entries, source_entries)
            cross += completed.gram(source_entries, target_sentinel)[:, 0]
        weights = affine_ridge_weights(gram, cross, float(ridge_by_context[context]))
        prediction_entries = sources * INTERVENTIONS + intervention
        positions = completed.observed_position[prediction_entries]
        if np.any(positions < 0):
            raise RuntimeError("LOW_RANK_AFFINE_TARGET_COLUMN_SUPPORT_FAIL")
        result[row, positions] = weights
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-9, rtol=0):
        raise RuntimeError("LOW_RANK_AFFINE_CONSTRAINT_FAIL")
    return result


def historical_one_entry_reconciliation(
    source_root: Path,
    full_gram: np.ndarray,
    target: int,
    plate: int,
    parameters: FrozenAffineParameters,
) -> float:
    train = np.asarray([index for index in range(ENTRIES) if index != target], dtype=np.int64)
    observed = matched_affine_prediction_weights(
        full_gram, train, np.asarray([target], dtype=np.int64), parameters.ridge[plate]
    )[0]
    full_observed = np.zeros(ENTRIES, dtype=np.float64)
    full_observed[train] = observed
    weights_path = source_root / "data/cgc_resolution2_replay/m49_k92_frozen_weights.npz"
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = np.asarray(archive["weights"], dtype=np.float64)
        sources = np.asarray(archive["sources"], dtype=np.int64)
    context = target // INTERVENTIONS
    intervention = target % INTERVENTIONS
    expected = np.zeros(ENTRIES, dtype=np.float64)
    expected[sources[0, context] * INTERVENTIONS + intervention] = weights[
        plate, 0, context, intervention
    ]
    return float(np.max(np.abs(full_observed - expected)))
