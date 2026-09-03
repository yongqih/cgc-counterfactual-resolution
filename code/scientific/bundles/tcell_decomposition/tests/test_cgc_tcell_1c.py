from __future__ import annotations

from pathlib import Path

import numpy as np

from igc_virtual_cell.cgc_tcell_1c.adjudicate import (
    _analyze_matrix,
    _metric_row,
    _predicted_sufficient,
)


def test_parallel_orthogonal_energy_identities() -> None:
    truth2 = np.array([2.0, 3.0])
    pred2 = np.array([1.0, 4.0])
    dot = np.array([0.5, 1.0])
    row = _metric_row("M", 0, 0, "D", truth2, pred2, dot)
    assert np.isclose(row["kappa_c"], row["e_parallel"] + row["e_perpendicular"])
    assert np.isclose(row["f_parallel_pred"], row["cosine"] ** 2)


def test_low_rank_prediction_sufficient_statistics() -> None:
    rng = np.random.default_rng(3)
    p, states, rank = 7, 3, 4
    truth = rng.normal(size=(p, states, rank)).astype(np.float32)
    anchor = rng.normal(size=(p, states, rank)).astype(np.float32)
    residual = rng.normal(size=(p, states, rank)).astype(np.float32)
    base = {
        "truth2": np.square(truth, dtype=np.float64).sum(axis=(1, 2)),
        "anchor2": np.square(anchor, dtype=np.float64).sum(axis=(1, 2)),
        "dot": np.multiply(truth, anchor, dtype=np.float64).sum(axis=(1, 2)),
        "truth_proj": truth,
        "anchor_proj": anchor,
    }
    rgamma, pred2, dot = _predicted_sufficient(base, residual)
    prediction = anchor + rgamma
    assert np.allclose(pred2, np.square(prediction, dtype=np.float64).sum(axis=(1, 2)))
    assert np.allclose(dot, np.multiply(truth, prediction, dtype=np.float64).sum(axis=(1, 2)))


def test_identity_similarity_retrieves_interventions(tmp_path: Path) -> None:
    matrix = np.eye(9, dtype=np.float32)
    summary, rows = _analyze_matrix(matrix, [f"P{i}" for i in range(9)], 100, 17)
    assert summary["top1_accuracy"] == 1.0
    assert summary["median_true_match_rank"] == 1.0
    assert summary["median_paired_matched_minus_row_mismatch"] == 1.0
    assert rows["true_match_rank"].eq(1).all()
