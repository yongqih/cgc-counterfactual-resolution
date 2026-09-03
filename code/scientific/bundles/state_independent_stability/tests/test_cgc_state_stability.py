import numpy as np

from scripts.cgc_state_stability_audit import (
    adjudicate,
    mutation_invariance,
    strict_leave_one_context_mean,
)


def test_frozen_state_stability_verdict_rules():
    assert adjudicate(np.array([0.2, 0.3, 0.4, 0.5, -0.05])) == "STATE_OPERATOR_RECOVERY_BROADLY_STABLE"
    assert adjudicate(np.array([0.2, 0.3, -0.2, -0.3, 0.0])) == "STATE_OPERATOR_RECOVERY_HETEROGENEOUSLY_REPLICATED"
    assert adjudicate(np.array([0.05, 0.02, -0.01, 0.0, -0.02])) == "STATE_OPERATOR_RECOVERY_NOT_REPLICATED"


def test_truth_mutation_does_not_change_target_excluding_objects():
    truth = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    prediction = {"ST_SE_TAHOE": truth * 0.5, "ST_HVG_TAHOE": truth * 0.2}
    audit = mutation_invariance(truth, prediction)
    assert (audit["status"] == "PASS").all()


def test_strict_loco_mean_masks_target_before_arithmetic():
    truth = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    before = strict_leave_one_context_mean(truth)
    mutated = truth.copy()
    mutated[2] = 1e10
    after = strict_leave_one_context_mean(mutated)
    assert np.array_equal(before[2], after[2])
