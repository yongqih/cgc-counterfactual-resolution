import numpy as np

from igc_virtual_cell.cgc_state_1.analysis import (
    bootstrap_recovery,
    correspondence_null,
    leave_one_context_mean,
    pairwise_context_geometry,
    recovery_statistics,
)


def test_leave_one_context_mean_excludes_target():
    tensor = np.arange(3 * 2 * 1, dtype=float).reshape(3, 2, 1)
    got = leave_one_context_mean(tensor)
    assert np.allclose(got[0], (tensor[1] + tensor[2]) / 2)
    assert np.allclose(got[2], (tensor[0] + tensor[1]) / 2)


def test_recovery_identity_and_zero_baseline():
    truth = np.arange(1, 13, dtype=float).reshape(2, 2, 3)
    assert np.isclose(recovery_statistics(truth, truth)["g"], 1.0)
    assert np.isclose(recovery_statistics(truth, np.zeros_like(truth))["g"], 0.0)


def test_bootstrap_operates_on_frozen_table():
    truth = np.ones((3, 4, 2))
    pred = truth * 0.5
    values = bootstrap_recovery(truth, pred, draws=20, seed=1)
    assert np.allclose(values, 0.75)


def test_null_shapes():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(4, 6, 8))
    pred = rng.normal(size=(4, 6, 8))
    assert correspondence_null(truth, pred, "context", 12, 1).shape == (12,)
    assert correspondence_null(truth, pred, "intervention", 12, 1).shape == (12,)


def test_pairwise_context_geometry_shape():
    tensor = np.zeros((5, 7, 3))
    pairs, distances = pairwise_context_geometry(tensor)
    assert pairs.shape == (10, 2)
    assert distances.shape == (7, 10)
