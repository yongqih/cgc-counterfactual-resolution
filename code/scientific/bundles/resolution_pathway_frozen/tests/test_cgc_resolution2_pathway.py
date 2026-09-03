from __future__ import annotations

import numpy as np

from igc_virtual_cell.cgc_resolution_poc.pathway import PATHWAYS, recovery, utility


def test_pathway_panel_is_exactly_frozen() -> None:
    assert PATHWAYS == ("MAPK", "PI3K", "JAK-STAT", "p53", "NFkB")


def test_replicate_stable_utility_and_recovery() -> None:
    truth = np.asarray([[[[[2.0]]]], [[[[3.0]]]]])
    prediction = np.asarray([[[[[1.0]]]], [[[[2.0]]]]])
    vtruth, vafter = utility(truth, prediction)
    assert vtruth.item() == 6.0
    assert vafter.item() == 1.0
    assert recovery(vtruth, vafter) == 1.0 - 1.0 / 6.0


def test_invalid_denominator_is_fail_closed() -> None:
    with np.testing.assert_raises_regex(RuntimeError, "DENOMINATOR"):
        recovery(np.asarray([-1.0]), np.asarray([0.0]))


def test_random_candidate_recovery_allows_negative_cross_plate_denominator() -> None:
    truth = np.asarray([-2.0, -1.0])
    residual = np.asarray([-1.0, -0.5])
    assert recovery(truth, residual, require_positive=False) == 0.5
