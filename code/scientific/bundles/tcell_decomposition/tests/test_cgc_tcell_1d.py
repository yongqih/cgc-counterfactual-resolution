from __future__ import annotations

import numpy as np

from igc_virtual_cell.cgc_tcell_1d.decompose import decomposition, scalar_metrics


def test_decomposition_is_orthogonal() -> None:
    stats = {
        "t2": np.array([5.0]), "s2": np.array([4.0]), "ts": np.array([2.0]),
        "p2": np.array([9.0]), "pt": np.array([3.0]), "ps": np.array([6.0]),
    }
    result = decomposition(stats)
    assert np.isclose(result["beta_t"], 0.5)
    assert np.isclose(result["beta_p"], 1.5)
    assert np.isclose(result["nt_s_dot"].sum(), 0.0)
    assert np.isclose(result["np_s_dot"].sum(), 0.0)


def test_perfect_adaptation_metrics() -> None:
    x = np.array([1.0, 2.0, 3.0])
    result = scalar_metrics(x, x, x)
    assert result["cosine"] == 1.0
    assert result["alpha"] == 1.0
    assert result["kappa"] == 1.0


def test_two_way_requires_global_intervention_axis() -> None:
    rng = np.random.default_rng(19)
    values = rng.normal(size=(17, 3, 8)).astype(np.float32)
    global_result = values.copy()
    global_result -= values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    global_result -= values.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    global_result += values.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    from igc_virtual_cell.cgc_tcell_1b.benchmark import two_way
    assert np.allclose(global_result, two_way(values))
