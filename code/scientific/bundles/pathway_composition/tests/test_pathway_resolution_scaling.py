from __future__ import annotations

import numpy as np
import pandas as pd

from igc_virtual_cell.pathway_resolution_scaling import (
    PANEL_SIZE,
    all_but_one_deltas,
    dominance_probability,
    iter_subsets,
    pareto_frontier,
    span_inverse,
)


def test_exhaustive_panel_family_is_complete() -> None:
    subsets = list(iter_subsets())
    assert len(subsets) == 2**PANEL_SIZE - 1
    assert subsets[0] == (1, (0,))
    assert subsets[-1] == (2**PANEL_SIZE - 1, tuple(range(PANEL_SIZE)))


def test_all_but_one_identity_matches_explicit_sources() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(2, 50, 3, 4))
    observed = all_but_one_deltas(values)
    expected = np.empty_like(values)
    for target in range(50):
        source = [index for index in range(50) if index != target]
        expected[:, target] = values[:, target] - values[:, source].mean(axis=1)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)


def test_small_gram_inverse_matches_explicit_orthogonal_projector() -> None:
    rng = np.random.default_rng(19)
    weights = rng.normal(size=(40, 4))
    weights /= np.linalg.norm(weights, axis=0)
    inverse, audit = span_inverse(weights.T @ weights)
    explicit = weights @ inverse @ weights.T
    q, _ = np.linalg.qr(weights, mode="reduced")
    np.testing.assert_allclose(explicit, q @ q.T, atol=2e-14, rtol=0)
    assert audit["rank"] == 4


def test_raw_score_recovery_and_orthogonal_span_are_distinct_when_axes_overlap() -> None:
    weights = np.asarray([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    weights /= np.linalg.norm(weights, axis=0)
    inverse, _ = span_inverse(weights.T @ weights)
    orthogonal = weights @ inverse @ weights.T
    raw = weights @ weights.T
    assert not np.allclose(orthogonal, raw)


def test_dominance_probability_is_exact() -> None:
    assert dominance_probability(np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])) == 1.0
    assert dominance_probability(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])) == 0.0


def test_pareto_frontier_flags_only_nondominated_rows() -> None:
    table = pd.DataFrame(
        {
            "subset_id": ["a", "b", "c", "d"],
            "subset_mask": [1, 2, 3, 4],
            "pathways": ["a", "b", "c", "d"],
            "pathway_count": [1, 1, 2, 2],
            "effective_rank": [1, 1, 2, 2],
            "biological_fidelity_F": [0.1, 0.2, 0.3, 0.4],
            "recoverability_g": [0.9, 0.7, 0.8, 0.6],
            "historical_five": [False, False, True, False],
        }
    )
    observed = pareto_frontier(table).set_index("subset_id")
    assert observed.loc["a", "pareto_frontier"]
    assert not observed.loc["b", "pareto_frontier"]
    assert observed.loc["c", "pareto_frontier"]
    assert observed.loc["d", "pareto_frontier"]

