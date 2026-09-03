import pandas as pd
import numpy as np

from igc_virtual_cell.cgc_tcell.core import (
    donor_disjoint_distances,
    donor_interactions,
    donor_partitions,
    hierarchical_target_bootstrap,
    null_term_plan,
    crossmeasurement_context_distances,
    crossmeasurement_permutation_null,
    permutation_null_from_crossgrams,
    stratified_permutation_indices,
)
from igc_virtual_cell.cgc_tcell.inventory import _ntc_mask, _targeting_mask
from igc_virtual_cell.cgc_tcell.conditional import (
    _anova_components,
    factorial_signal_decomposition,
)


def test_guide_type_masks_are_explicit_and_disjoint():
    obs = pd.DataFrame(
        {"guide_type": ["targeting", "non-targeting", "NTC", "targeting"]}
    )
    targeting = _targeting_mask(obs)
    ntc = _ntc_mask(obs)
    assert targeting.tolist() == [True, False, False, True]
    assert ntc.tolist() == [False, True, True, False]
    assert not (targeting & ntc).any()


def test_three_unique_balanced_donor_partitions():
    partitions = donor_partitions(["D1", "D2", "D3", "D4"])
    assert partitions == [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def test_donor_interactions_remove_target_and_state_means():
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(7, 4, 3, 11)).astype(np.float32)
    gamma = donor_interactions(delta)
    np.testing.assert_allclose(gamma.mean(axis=0), 0, atol=2e-7)
    np.testing.assert_allclose(gamma.mean(axis=2), 0, atol=2e-7)


def test_per_target_contributions_reconstruct_distances_and_bootstrap():
    rng = np.random.default_rng(5)
    delta = rng.normal(size=(9, 4, 3, 13)).astype(np.float32)
    states = ["Rest", "Stim8hr", "Stim48hr"]
    frame, contributions = donor_disjoint_distances(
        delta, donors=["D1", "D2", "D3", "D4"], states=states
    )
    reconstructed = np.concatenate(
        [values.mean(axis=0) for values in contributions]
    )
    np.testing.assert_allclose(reconstructed, frame["crossvalidated_distance"])
    bootstrap = hierarchical_target_bootstrap(
        contributions, states=states, draws=100, seed=7
    )
    assert len(bootstrap) == 4
    assert np.isfinite(bootstrap["bootstrap_mean"]).all()


def test_stratified_permutations_preserve_bins_and_are_deterministic():
    nuisance = np.asarray([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]], dtype=float)
    first = stratified_permutation_indices(nuisance, draws=7, bins=3, seed=13)
    second = stratified_permutation_indices(nuisance, draws=7, bins=3, seed=13)
    np.testing.assert_array_equal(first, second)
    for unit in range(2):
        order = np.argsort(nuisance[unit], kind="stable")
        labels = np.empty(6, dtype=int)
        labels[order] = np.arange(6) * 3 // 6
        for draw in range(7):
            np.testing.assert_array_equal(labels[first[unit, draw]], labels)


def test_crossgram_null_matches_literal_pipeline():
    rng = np.random.default_rng(19)
    delta = rng.normal(size=(7, 4, 3, 5)).astype(np.float32)
    centered = delta - delta.mean(axis=0, keepdims=True)
    permutations = stratified_permutation_indices(
        np.ones((12, 7)), draws=11, bins=1, seed=29
    )
    fast = permutation_null_from_crossgrams(
        centered, permutations, batch_draws=4, device="cpu"
    )
    literal = np.empty_like(fast)
    donors = ["D1", "D2", "D3", "D4"]
    states = ["Rest", "Stim8hr", "Stim48hr"]
    for draw in range(permutations.shape[1]):
        permuted = np.empty_like(delta)
        for donor in range(4):
            for state in range(3):
                unit = donor * 3 + state
                permuted[:, donor, state] = delta[
                    permutations[unit, draw], donor, state
                ]
        distances, _ = donor_disjoint_distances(
            permuted, donors=donors, states=states
        )
        literal[draw] = distances["crossvalidated_distance"].to_numpy()
    np.testing.assert_allclose(fast, literal, rtol=2e-5, atol=2e-6)


def test_null_plan_has_only_cross_donor_terms():
    terms, coefficients = null_term_plan()
    assert len(terms) == 54
    assert coefficients.shape == (54, 9)
    assert all(left_donor < right_donor for left_donor, _, right_donor, _ in terms)


def test_crossmeasurement_null_matches_literal_context_geometry():
    rng = np.random.default_rng(31)
    left = rng.normal(size=(6, 4, 8)).astype(np.float32)
    right = rng.normal(size=(6, 4, 8)).astype(np.float32)
    maps = stratified_permutation_indices(np.ones((4, 6)), draws=5, bins=1, seed=37)
    fast = crossmeasurement_permutation_null(left, right, maps, batch_draws=2)
    literal = np.empty_like(fast)
    for draw in range(5):
        a = np.empty_like(left)
        b = np.empty_like(right)
        for context in range(4):
            a[:, context] = left[maps[context, draw], context]
            b[:, context] = right[maps[context, draw], context]
        literal[draw] = crossmeasurement_context_distances(a, b)[
            "crossvalidated_distance"
        ]
    np.testing.assert_allclose(fast, literal, rtol=2e-5, atol=2e-6)


def test_factorial_components_are_orthogonal_and_recover_injected_terms():
    rng = np.random.default_rng(41)
    p, d, s, g = 8, 4, 3, 5
    pd = rng.normal(size=(p, d, 1, g)).astype(np.float32)
    pd -= pd.mean(axis=0, keepdims=True)
    pd -= pd.mean(axis=1, keepdims=True)
    ps = rng.normal(size=(p, 1, s, g)).astype(np.float32)
    ps -= ps.mean(axis=0, keepdims=True)
    ps -= ps.mean(axis=2, keepdims=True)
    tensor = pd + ps
    estimated_pd, estimated_ps, estimated_pds = _anova_components(tensor)
    np.testing.assert_allclose(estimated_pd, pd, atol=2e-6)
    np.testing.assert_allclose(estimated_ps, ps, atol=2e-6)
    np.testing.assert_allclose(estimated_pds, 0, atol=2e-6)
    result = factorial_signal_decomposition(
        tensor, tensor, chunk_genes=2, bootstrap_draws=20, seed=43
    )
    assert result.loc[
        result["component"].eq("intervention_x_donor_x_stimulation"),
        "reliable_crossguide_energy",
    ].iloc[0] < 1e-10
