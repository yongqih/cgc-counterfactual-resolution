from __future__ import annotations

import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.dense_resolution_plateau import (
    BootstrapPerformance,
    deterministic_verdict,
    gain_and_slope_tables,
    plateau_band,
    saturation_thresholds,
)


def simple_bootstrap() -> BootstrapPerformance:
    names = ("PCA2", "PCA16", "PCA32", "FULL_RNA_REOPT")
    observed = np.asarray(
        [
            [0.01, 0.02],
            [0.09, 0.18],
            [0.10, 0.20],
            [0.10, 0.20],
        ]
    )
    draws = np.repeat(observed[None, :, :], 20, axis=0)
    indices = np.zeros((20, 4), dtype=int)
    return BootstrapPerformance(names, ("PG_macro", "g_func"), observed, draws, indices)


def test_saturation_thresholds_are_smallest_joint_dimension() -> None:
    dimensions = np.asarray([2, 16, 32])
    observed = np.asarray([[0.1, 0.1], [0.9, 0.9], [1.0, 1.0]])
    boot = np.repeat(observed[None, :, :], 20, axis=0)
    table = saturation_thresholds(dimensions, observed, boot, [0.90, 0.95], 19421)
    indexed = table.set_index(["retention_threshold", "metric"])
    assert indexed.loc[(0.90, "PG"), "observed_k"] == 16
    assert indexed.loc[(0.90, "g_func"), "observed_k"] == 16
    assert indexed.loc[(0.90, "joint"), "observed_k"] == 16
    assert indexed.loc[(0.95, "joint"), "observed_k"] == 32
    assert indexed.loc[(0.90, "joint"), "representation_dimension_compression_ratio"] == 19421 / 16


def test_nonpositive_denominators_are_documented_as_undefined() -> None:
    dimensions = np.asarray([2, 16])
    observed = np.asarray([[0.5, 0.5], [1.0, 1.0]])
    boot = np.asarray(
        [
            [[0.5, 0.5], [1.0, 1.0]],
            [[np.nan, 0.5], [np.nan, 1.0]],
        ]
    )
    table = saturation_thresholds(dimensions, observed, boot, [0.90], 19421)
    pg = table.query("metric == 'PG'").iloc[0]
    assert pg.bootstrap_nonpositive_denominator_draws == 1
    assert pg.bootstrap_defined_draws == 1


def test_gain_and_piecewise_slopes_use_frozen_regions() -> None:
    names = tuple(f"PCA{k}" for k in [2, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]) + ("FULL_RNA_REOPT",)
    dimensions = np.asarray([2, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    early_then_flat = np.minimum(dimensions / 16, 1.0)
    observed = np.column_stack([0.1 * early_then_flat, 0.2 * early_then_flat])
    observed = np.vstack([observed, [0.1, 0.2]])
    boot = np.repeat(observed[None, :, :], 30, axis=0)
    bundle = BootstrapPerformance(names, ("PG_macro", "g_func"), observed, boot, np.zeros((30, 2), int))
    table = gain_and_slope_tables(bundle, dimensions, list(names[:-1]), "FULL_RNA_REOPT")
    indexed = table.set_index(["metric", "quantity"])
    assert indexed.loc[("PG_macro", "gain_PCA2_to_PCA16"), "estimate"] > 0
    assert abs(indexed.loc[("PG_macro", "gain_PCA16_to_PCA32"), "estimate"]) < 1e-12
    assert indexed.loc[("PG_macro", "beta_early_minus_beta_late"), "estimate"] > 0


def test_verdict_requires_every_saturation_gate() -> None:
    assert deterministic_verdict(True, True, True, True, True, False, True) == "PREDICTIVE_INFORMATION_SATURATION_SUPPORTED"
    assert deterministic_verdict(True, True, False, True, True, False, True) == "COMPACT_PREDICTIVE_STRUCTURE_SUPPORTED_WITHOUT_CLEAR_SATURATION"
    assert deterministic_verdict(False, False, False, False, False, True, True) == "PREDICTIVE_PERFORMANCE_CONTINUES_TO_SCALE_WITH_DIMENSION"
    assert deterministic_verdict(False, False, False, False, False, False, True) == "NO_REPRODUCIBLE_RESOLUTION_STRUCTURE"


def test_plateau_band_contains_fixed_k_only() -> None:
    performance = pd.DataFrame(
        {
            "representation": ["PCA16", "PCA24", "PCA_STAR_DENSE", "FULL_RNA_REOPT"],
            "representation_family": ["fixed_PCA", "fixed_PCA", "nested_PCA", "full_RNA"],
            "dimension": [16, 24, 24, 19421],
            "PG_macro": [0.09, 0.10, 0.10, 0.10],
            "g_func": [0.18, 0.20, 0.20, 0.20],
        }
    )
    result = plateau_band(performance, 0.10, 0.20, 0.01, 0.03)
    assert result.representation.tolist() == ["PCA16", "PCA24"]
