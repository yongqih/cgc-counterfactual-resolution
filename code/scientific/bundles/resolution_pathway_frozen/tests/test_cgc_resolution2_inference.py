from __future__ import annotations

import numpy as np

from igc_virtual_cell.cgc_resolution_poc.inference import bootstrap_g, hierarchical_weights, simultaneous_rows


def test_hierarchical_weights_have_frozen_draw_size() -> None:
    weights = hierarchical_weights(draws=7, seed=3)
    assert weights.shape == (7, 50 * 93)
    assert np.all(weights.sum(axis=1) == 50 * 93)
    assert np.array_equal(weights, hierarchical_weights(draws=7, seed=3))


def test_simultaneous_interval_family_is_shared_and_conservative() -> None:
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(10_000, 2))
    rows = simultaneous_rows(["a", "b"], np.zeros(2), draws, "test")
    assert rows[0]["simultaneous_critical_95"] == rows[1]["simultaneous_critical_95"]
    assert rows[0]["simultaneous_critical_95"] > 1.96


def test_bootstrap_pools_trailing_coordinates_by_ratio_of_sums() -> None:
    weights = np.ones((1, 50 * 93), dtype=np.int16)
    truth = np.ones((50, 93, 2))
    residual = np.stack((np.full((50, 93), 0.5), np.full((50, 93), 0.25)), axis=-1)
    assert bootstrap_g(weights, truth, residual).shape == (1, 1)
    assert bootstrap_g(weights, truth, residual).item() == 0.625
