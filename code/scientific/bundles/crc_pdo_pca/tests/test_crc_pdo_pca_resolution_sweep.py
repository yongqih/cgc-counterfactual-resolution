from __future__ import annotations

import numpy as np

from igc_virtual_cell.crc_pdo_personalized.modeling import PlatformData
from igc_virtual_cell.crc_pdo_personalized.pca_resolution_sweep import (
    _best_alpha,
    _best_k_alpha,
    bootstrap_primary_contrasts,
    deterministic_verdict,
    fit_nested_pca_oof,
    selection_is_coherent,
)


def synthetic_data(seed: int = 4) -> PlatformData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(15, 12))
    weights = rng.normal(size=(12, 24))
    y = x @ weights + 0.05 * rng.normal(size=(15, 24))
    return PlatformData(
        platform="RNAseq",
        patient_ids=np.asarray([f"Pt{index + 1}" for index in range(15)]),
        x=x,
        y=y,
        drug_names=np.asarray([f"D{index + 1}" for index in range(24)]),
        eligible_pdo_ids=tuple((f"PDO{index + 1}",) for index in range(15)),
    )


def test_frozen_tie_breaks() -> None:
    assert _best_alpha({0.1: 1.0, 1.0: 1.0}) == 1.0
    assert _best_k_alpha({(4, 0.1): 1.0, (2, 0.1): 1.0, (2, 1.0): 1.0}) == (2, 1.0)


def test_nested_sweep_is_complete_and_outcome_sealed() -> None:
    data = synthetic_data()
    fit = fit_nested_pca_oof(data, [2, 4, 8], [0.01, 0.1, 1.0], 21)
    assert set(fit.predictions) == {"PCA2", "PCA4", "PCA8", "PCA_STAR", "FULL_RNA_REOPT"}
    assert all(value.shape == data.y.shape for value in fit.predictions.values())
    assert all(np.isfinite(value).all() for value in fit.predictions.values())
    assert fit.selections.held_outcome_used_in_selection.eq(False).all()
    assert fit.selections.patient_overlap.eq(0).all()
    assert fit.selections.all_inner_splits_sealed.all()
    assert fit.grid_manifest.groupby("held_patient").selected_jointly.sum().eq(1).all()
    assert fit.variance_capture.training_expression_variance_explained.between(0, 1).all()


def test_primary_bootstrap_is_deterministic_and_simultaneous() -> None:
    data = synthetic_data()
    population = np.repeat(np.mean(data.y, axis=0, keepdims=True), len(data.y), axis=0)
    pca = data.y + 0.1
    full = data.y + 0.2
    first, draws_first = bootstrap_primary_contrasts(
        data.y, pca, full, population, 200, 33, 0.005, 0.01
    )
    second, draws_second = bootstrap_primary_contrasts(
        data.y, pca, full, population, 200, 33, 0.005, 0.01
    )
    assert np.array_equal(draws_first, draws_second)
    assert first.equals(second)
    assert len(first) == 2
    assert first.max_t_critical.nunique() == 1
    assert first.resampling_unit.eq("patient_profile").all()


def test_selection_coherence_and_verdict_rules() -> None:
    coherent, summary = selection_is_coherent(
        np.asarray([8] * 20 + [14] * 10), 0.2, 16
    )
    assert coherent
    assert summary["mode_k"] == 8
    inference = np.asarray(
        [
            ("PCA_STAR_minus_FULL_RNA_REOPT_PG", 0.01, 0.001, 0.02, True, True, 0.005),
            ("PCA_STAR_minus_FULL_RNA_REOPT_g_func", 0.03, 0.002, 0.05, True, True, 0.014),
        ],
        dtype=[
            ("contrast", "U50"),
            ("estimate", "f8"),
            ("simultaneous_95_ci_lower", "f8"),
            ("simultaneous_95_ci_upper", "f8"),
            ("lower_exceeds_zero", "?"),
            ("lower_exceeds_negative_margin", "?"),
            ("noninferiority_margin", "f8"),
        ],
    )
    import pandas as pd

    assert deterministic_verdict(pd.DataFrame(inference), coherent, True) == "COARSER_CONDITIONING_RESOLUTION_IMPROVES_TRANSFER"
