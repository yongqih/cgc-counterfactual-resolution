from __future__ import annotations

from igc_virtual_cell.cgc_tcell_1b.audit import (
    inner_folds,
    outer_folds,
    training_examples,
    verify_split_leakage,
)
from igc_virtual_cell.cgc_tcell_1b.benchmark import FoldEvaluator
import numpy as np


DONORS = ["D1", "D2", "D3", "D4"]
STATES = ["Rest", "Stim8hr", "Stim48hr"]


def test_outer_and_inner_folds_are_strictly_donor_disjoint() -> None:
    outer = outer_folds(DONORS)
    inner = inner_folds(DONORS)
    assert len(outer) == 4
    assert len(inner) == 12
    for row in outer.itertuples(index=False):
        assert row.held_out_donor not in row.training_donors.split(";")
    for row in inner.itertuples(index=False):
        assert row.held_out_donor != row.inner_validation_donor
        assert row.held_out_donor not in row.inner_training_donors.split(";")
        assert row.inner_validation_donor not in row.inner_training_donors.split(";")


def test_held_out_responses_are_never_loaded_for_training() -> None:
    examples = training_examples(DONORS, STATES, ["P1", "P2"])
    checks = verify_split_leakage(examples, inner_folds(DONORS))
    assert all(checks.values())
    assert not examples.loc[
        examples.donor_id.eq(examples.held_out_donor),
        "perturbation_response_loaded_for_training",
    ].any()
    assert len(checks) == 8
    assert set(checks) == {
        "held_out_perturbation_rows_absent_from_training",
        "held_out_perturbation_rows_absent_from_validation",
        "target_normalization_has_no_cross_donor_fit",
        "context_pca_fit_on_outer_training_donors_only",
        "intervention_anchors_fit_on_outer_training_donors_only",
        "all_states_use_same_outer_checkpoint",
        "held_out_truth_unloaded_during_model_inference",
        "outer_test_metrics_not_selection_objective",
    }


def test_context_blind_operator_is_zero_and_cosine_is_missing() -> None:
    rng = np.random.default_rng(7)
    delta = rng.normal(size=(5, 4, 3, 7)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(7, 3)))
    components = q.T.astype(np.float32)
    coeff = np.einsum("pdsg,rg->pdsr", delta, components)
    ntc = rng.normal(size=(4, 3, 7)).astype(np.float32)
    evaluator = FoldEvaluator(
        delta, coeff, components, ntc, [1, 2, 3], 0, np.arange(7) % 5, 3
    )
    metrics, _, _ = evaluator.evaluate(
        np.zeros((5, 3, 3), dtype=np.float32),
        anchor_kind="CONTEXT_BLIND_ANCHOR",
    )
    assert metrics["kappa_c"] == 0.0
    assert metrics["alpha"] == 0.0
    assert np.isnan(metrics["cosine"])
    assert np.isfinite(metrics["response_metrics"]["mse"])
