from __future__ import annotations

import numpy as np

from igc_virtual_cell.cgc_lowrank_completion.core import (
    CONTEXTS,
    INTERVENTIONS,
    FactorFit,
    additive_design,
    prediction_weights,
    weighted_squared_error,
)
from igc_virtual_cell.cgc_lowrank_completion.design import frozen_masks


def test_additive_design_is_full_rank_on_near_complete_mask() -> None:
    indices = np.arange(CONTEXTS * INTERVENTIONS, dtype=np.int64)
    keep = indices[(indices % 100) != 0]
    design = additive_design(keep)
    assert design.shape == (len(keep), 142)
    assert np.linalg.matrix_rank(design) == 142


def test_prediction_weights_do_not_require_target_outcomes() -> None:
    train = np.asarray([i for i in range(CONTEXTS * INTERVENTIONS) if i not in {0, 101}], dtype=np.int64)
    targets = np.asarray([0, 101], dtype=np.int64)
    rng = np.random.default_rng(7)
    fit = FactorFit(
        rank=2,
        ridge=0.01,
        seed=7,
        u=rng.normal(size=(CONTEXTS, 2)),
        v=rng.normal(size=(INTERVENTIONS, 2)),
        steps=1,
        converged=True,
        objective_initial=0.0,
        objective_final=0.0,
        objective_best=0.0,
        relative_change_window=0.0,
        gradient_norm=0.0,
        device="cpu",
        peak_memory_bytes=0,
    )
    first = prediction_weights(train, targets, fit)
    second = prediction_weights(train, targets, fit)
    assert np.array_equal(first.additive, second.additive)
    assert np.array_equal(first.interaction, second.interaction)
    assert np.allclose(first.full.sum(axis=1), 1.0, atol=1e-10)


def test_gram_squared_error_matches_vectors() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(12, 9))
    gram = values @ values.T
    train = np.arange(9)
    target = np.arange(9, 12)
    weights = rng.normal(size=(3, 9))
    observed = weighted_squared_error(gram, train, target, weights)
    expected = np.sum((values[target] - weights @ values[train]) ** 2, axis=1)
    assert np.allclose(observed, expected, atol=1e-10)


def test_frozen_masks_are_balanced_and_complete() -> None:
    masks = frozen_masks()
    seen = np.bincount(masks.outer_fold, minlength=100)
    assert set(seen.tolist()) == {46, 47}
    assert sum(seen) == CONTEXTS * INTERVENTIONS
