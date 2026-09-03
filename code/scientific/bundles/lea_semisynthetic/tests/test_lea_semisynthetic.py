from __future__ import annotations

import numpy as np
import pandas as pd

from igc_virtual_cell.lea_semisynthetic import (
    adjudicate,
    generate_observed_replicates,
    generate_seed_components,
    scheduled_models,
)


def test_generator_uses_real_shape_and_matches_reliability() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.normal(size=(342, 101)).astype(np.float32)
    components = generate_seed_components(baseline, seed=207049, rank=4)
    assert components.z_x.shape == components.z_h.shape == (342, 4)
    assert components.w_y.shape == (4, 101)
    a, b, calibration = generate_observed_replicates(
        components, 0.5, 207049, 4, 0.7443
    )
    assert a.shape == b.shape == baseline.shape
    assert abs(calibration["realized_reliability"] - 0.7443) <= 0.015


def test_tiered_schedule_is_frozen() -> None:
    config = {
        "tier_1_models": ["ridge", "rbf_kernel_ridge", "deep_residual_mlp"],
        "tier_2_models": [
            "ridge",
            "pca_ridge",
            "pilot_mlp",
            "rbf_kernel_ridge",
            "hist_gradient_boosting",
            "deep_residual_mlp",
        ],
        "tier_2_lambdas": [0.0, 0.25, 1.0],
    }
    assert len(scheduled_models(config, 0.1)) == 3
    assert len(scheduled_models(config, 0.25)) == 6


def _adjudication_fixture(pass_control: bool) -> pd.DataFrame:
    rows = []
    lambdas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    tier1 = ["ridge", "rbf_kernel_ridge", "deep_residual_mlp"]
    all_models = ["ridge", "pca_ridge", "pilot_mlp", "rbf_kernel_ridge", "hist_gradient_boosting", "deep_residual_mlp"]
    for seed in range(10):
        for model in tier1:
            for value in lambdas:
                endpoint = 0.45 * value if pass_control and model == "ridge" else 0.01 * value
                rows.append({"seed": seed, "model": model, "lambda": value, "pooled_full_gene_oof_r2": endpoint})
        for model in set(all_models) - set(tier1):
            for value in (0.0, 0.25, 1.0):
                rows.append({"seed": seed, "model": model, "lambda": value, "pooled_full_gene_oof_r2": 0.005 * value})
    return pd.DataFrame(rows)


def test_positive_control_adjudication_pass_and_partial() -> None:
    config = {
        "lambda_grid": [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
        "tier_1_models": ["ridge", "rbf_kernel_ridge", "deep_residual_mlp"],
        "adjudication": {
            "positive_control_min_model_median_r2": 0.25,
            "clear_separation_min_paired_median_difference": 0.20,
            "monotonicity_min_spearman": 0.90,
            "negative_control_max_median_seedwise_best_r2": 0.02,
        },
    }
    assert adjudicate(_adjudication_fixture(True), config)[0] == "PIPELINE_POSITIVE_CONTROL_PASS"
    assert adjudicate(_adjudication_fixture(False), config)[0] == "PIPELINE_POSITIVE_CONTROL_PARTIAL"
