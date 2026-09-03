from __future__ import annotations

import numpy as np

from igc_virtual_cell.cgc_entrywise.core import (
    SealedGramView,
    affine_ridge_weights,
    lowrank_context_weights,
    matched_fit_statistics,
    reference_context_gram,
    sample_index,
    sentinel_offset_prediction,
    support_mean_prediction,
    weighted_context_prediction,
)
from igc_virtual_cell.cgc_entrywise.estimators import fit_batches, sentinel_matrix
from igc_virtual_cell.cgc_entrywise.metrics import matched_context_gram


def _episode(values: np.ndarray, replacement: np.ndarray) -> dict[str, object]:
    contexts, interventions, _ = values.shape
    target_context, target_intervention = 2, 3
    modified = values.copy()
    modified[target_context, target_intervention] = replacement
    flat = modified.reshape(contexts * interventions, -1)
    same = flat @ flat.T
    view = SealedGramView(same, target_context, target_intervention, interventions)
    sources = np.asarray([0, 1, 3], dtype=np.int64)
    sentinels = np.asarray([0, 1, 2, 4], dtype=np.int64)
    fit_gram, fit_cross = matched_fit_statistics(view, sources, target_context, sentinels)
    reference = reference_context_gram(view, sources, np.arange(interventions))
    affine = affine_ridge_weights(fit_gram, fit_cross, 1e-3)
    lowrank = lowrank_context_weights(reference, fit_gram, fit_cross, 2)
    predictions = {
        "M0": support_mean_prediction(sources, target_intervention, interventions),
        "M1": sentinel_offset_prediction(
            sources, target_context, target_intervention, sentinels, 0.5, interventions
        ),
        "M2": weighted_context_prediction(sources, target_intervention, affine, interventions),
        "M3": weighted_context_prediction(sources, target_intervention, lowrank, interventions),
    }
    return {
        "support": tuple(map(int, sources)),
        "sentinels": tuple(map(int, sentinels)),
        "lambda": 1e-3,
        "rank": 2,
        "affine": affine,
        "lowrank": lowrank,
        "signatures": {name: prediction.signature() for name, prediction in predictions.items()},
        "predictions": {name: prediction.materialize(flat) for name, prediction in predictions.items()},
        "accessed": tuple(view.accessed),
    }


def test_entrywise_prediction_invariant_to_hidden_target() -> None:
    rng = np.random.default_rng(202608227)
    values = rng.normal(size=(5, 7, 31))
    original = values[2, 3].copy()
    variants = [
        rng.normal(size=31),
        np.full(31, 1e12),
        original[rng.permutation(31)],
        values[2, 5].copy(),
    ]
    baseline = _episode(values, original)
    hidden = sample_index(2, 3, 7)
    assert all(hidden not in rows and hidden not in cols for rows, cols in baseline["accessed"])
    for replacement in variants:
        observed = _episode(values, replacement)
        assert observed["support"] == baseline["support"]
        assert observed["sentinels"] == baseline["sentinels"]
        assert observed["lambda"] == baseline["lambda"]
        assert observed["rank"] == baseline["rank"]
        np.testing.assert_array_equal(observed["affine"], baseline["affine"])
        np.testing.assert_array_equal(observed["lowrank"], baseline["lowrank"])
        assert observed["signatures"] == baseline["signatures"]
        for name in baseline["predictions"]:
            np.testing.assert_array_equal(observed["predictions"][name], baseline["predictions"][name])


def test_sealed_view_rejects_hidden_access() -> None:
    gram = np.eye(12)
    view = SealedGramView(gram, hidden_context=1, hidden_intervention=2, intervention_count=4)
    hidden = sample_index(1, 2, 4)
    try:
        view.gram([0, hidden], [0, 1])
    except RuntimeError as exc:
        assert str(exc) == "SEALED_TARGET_ACCESS_ATTEMPT"
    else:
        raise AssertionError("hidden target access was not rejected")


def test_budget_extreme_is_all_but_one() -> None:
    assert 93 * 49 + 92 == 4_649


def test_full_batch_hidden_episode_is_invariant_to_hidden_gram_rows() -> None:
    rng = np.random.default_rng(202608228)
    contexts, interventions, genes = 5, 7, 41
    values = rng.normal(size=(contexts, interventions, genes))
    target, hidden = 2, 3
    sources = np.asarray([0, 1, 3, 4])
    order = np.asarray([5, 3, 0, 6, 1, 4, 2])
    null_map = np.asarray([1, 2, 3, 4, 5, 6, 0])

    def fit(replacement: np.ndarray) -> dict[str, np.ndarray]:
        changed = values.copy()
        changed[target, hidden] = replacement
        flat = changed.reshape(contexts * interventions, genes)
        full = (flat @ flat.T).reshape(contexts, interventions, contexts, interventions)
        matched = np.empty((interventions, contexts, contexts))
        for intervention in range(interventions):
            matched[intervention] = full[:, intervention, :, intervention]
        # fit_batches is frozen for the 93-intervention production axis. Pad the
        # synthetic fixture without changing any of its first seven entries.
        padded_matched = np.zeros((93, 50, 50))
        padded_matched[:interventions, :contexts, :contexts] = matched
        padded_full = np.zeros((50, 93, 50, 93))
        padded_full[:contexts, :interventions, :contexts, :interventions] = full
        padded_order = np.concatenate((order, np.arange(interventions, 93)))
        padded_null = np.arange(93)
        padded_null[:interventions] = null_map
        result = fit_batches(
            padded_matched,
            padded_full,
            target,
            sources,
            padded_order,
            4,
            1e-3,
            2,
            4,
            padded_null,
            np.asarray([hidden]),
        )
        return {key: value[hidden].copy() for key, value in result.items()}

    baseline = fit(values[target, hidden])
    for replacement in (
        rng.normal(size=genes),
        np.full(genes, 1e9),
        values[target, hidden][::-1],
        values[target, 1],
    ):
        observed = fit(replacement)
        for key in baseline:
            np.testing.assert_array_equal(observed[key], baseline[key])


def test_sentinel_matrix_excludes_only_sealed_identity() -> None:
    order = np.arange(93)[::-1]
    selected = sentinel_matrix(order, 92)
    assert selected.shape == (93, 92)
    for hidden in range(93):
        assert hidden not in selected[hidden]
        assert set(selected[hidden]) == set(range(93)) - {hidden}
