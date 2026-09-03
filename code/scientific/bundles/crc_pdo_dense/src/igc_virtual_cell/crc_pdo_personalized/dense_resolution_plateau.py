from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .modeling import PlatformData, _core_metrics


@dataclass(frozen=True)
class BootstrapPerformance:
    representation_order: tuple[str, ...]
    metric_order: tuple[str, ...]
    observed: np.ndarray
    draws: np.ndarray
    indices: np.ndarray


def bootstrap_performance(
    data: PlatformData,
    predictions: dict[str, np.ndarray],
    population: np.ndarray,
    draws: int,
    seed: int,
) -> BootstrapPerformance:
    names = tuple(predictions)
    metrics = ("PG_macro", "g_func")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data.patient_ids), size=(draws, len(data.patient_ids)))
    observed = np.empty((len(names), 2), dtype=np.float64)
    values = np.empty((draws, len(names), 2), dtype=np.float64)
    for representation_index, name in enumerate(names):
        core = _core_metrics(data.y, predictions[name], population, 1e-12)
        observed[representation_index] = [core["PG_macro"], core["g_func"]]
        values[:, representation_index, 0] = np.mean(core["pg_patient"][indices], axis=1)
        values[:, representation_index, 1] = 1.0 - (
            np.mean(core["mse_model"][indices], axis=1)
            / np.mean(core["mse_population"][indices], axis=1)
        )
    return BootstrapPerformance(names, metrics, observed, values, indices)


def performance_interval_frame(bootstrap: BootstrapPerformance) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation_index, representation in enumerate(bootstrap.representation_order):
        for metric_index, metric in enumerate(bootstrap.metric_order):
            values = bootstrap.draws[:, representation_index, metric_index]
            rows.append(
                {
                    "representation": representation,
                    "metric": metric,
                    "estimate": bootstrap.observed[representation_index, metric_index],
                    "pointwise_95_ci_lower": float(np.quantile(values, 0.025)),
                    "pointwise_95_ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_draws": len(values),
                    "resampling_unit": "patient_profile",
                }
            )
    return pd.DataFrame.from_records(rows)


def retention_frame(
    bootstrap: BootstrapPerformance,
    fixed_names: Iterable[str],
    full_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    names = list(bootstrap.representation_order)
    full_index = names.index(full_name)
    rows: list[dict[str, object]] = []
    fixed_names = list(fixed_names)
    draw_retention = np.full((len(bootstrap.draws), len(fixed_names), 2), np.nan)
    for fixed_index, name in enumerate(fixed_names):
        representation_index = names.index(name)
        observed_ratios = bootstrap.observed[representation_index] / bootstrap.observed[full_index]
        for metric_index, metric in enumerate(bootstrap.metric_order):
            denominator = bootstrap.draws[:, full_index, metric_index]
            valid = denominator > 0
            draw_retention[valid, fixed_index, metric_index] = (
                bootstrap.draws[valid, representation_index, metric_index] / denominator[valid]
            )
            values = draw_retention[:, fixed_index, metric_index]
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "representation": name,
                    "dimension": int(name.removeprefix("PCA")),
                    "metric": metric,
                    "retention": observed_ratios[metric_index],
                    "bootstrap_median": float(np.median(finite)),
                    "bootstrap_q25": float(np.quantile(finite, 0.25)),
                    "bootstrap_q75": float(np.quantile(finite, 0.75)),
                    "bootstrap_90_range_lower": float(np.quantile(finite, 0.05)),
                    "bootstrap_90_range_upper": float(np.quantile(finite, 0.95)),
                    "valid_bootstrap_draws": len(finite),
                    "nonpositive_full_denominator_draws": int(np.sum(~valid)),
                }
            )
    return pd.DataFrame.from_records(rows), draw_retention


def saturation_thresholds(
    fixed_dimensions: np.ndarray,
    observed_retention: np.ndarray,
    bootstrap_retention: np.ndarray,
    thresholds: Iterable[float],
    nominal_dimensions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = ("PG", "g_func", "joint")
    for threshold in thresholds:
        for label_index, label in enumerate(labels):
            if label_index < 2:
                observed_mask = observed_retention[:, label_index] >= threshold
                valid_draws = np.isfinite(bootstrap_retention[:, :, label_index]).all(axis=1)
                bootstrap_mask = bootstrap_retention[:, :, label_index] >= threshold
            else:
                observed_mask = np.all(observed_retention >= threshold, axis=1)
                valid_draws = np.isfinite(bootstrap_retention).all(axis=(1, 2))
                bootstrap_mask = np.all(bootstrap_retention >= threshold, axis=2)
            observed_k = float(fixed_dimensions[np.flatnonzero(observed_mask)[0]]) if np.any(observed_mask) else np.nan
            draw_k = np.full(len(bootstrap_retention), np.nan)
            for draw in np.flatnonzero(valid_draws):
                reached = np.flatnonzero(bootstrap_mask[draw])
                if len(reached):
                    draw_k[draw] = fixed_dimensions[reached[0]]
            defined = draw_k[np.isfinite(draw_k)]
            rows.append(
                {
                    "retention_threshold": threshold,
                    "metric": label,
                    "observed_k": observed_k,
                    "representation_dimension_compression_ratio": nominal_dimensions / observed_k if np.isfinite(observed_k) else np.nan,
                    "bootstrap_defined_draws": len(defined),
                    "bootstrap_undefined_draws": int(len(draw_k) - len(defined)),
                    "bootstrap_nonpositive_denominator_draws": int(np.sum(~valid_draws)),
                    "bootstrap_median_k": float(np.median(defined)) if len(defined) else np.nan,
                    "bootstrap_q25_k": float(np.quantile(defined, 0.25)) if len(defined) else np.nan,
                    "bootstrap_q75_k": float(np.quantile(defined, 0.75)) if len(defined) else np.nan,
                    "bootstrap_90_range_lower_k": float(np.quantile(defined, 0.05)) if len(defined) else np.nan,
                    "bootstrap_90_range_upper_k": float(np.quantile(defined, 0.95)) if len(defined) else np.nan,
                    "resampling_unit": "patient_profile",
                }
            )
    return pd.DataFrame.from_records(rows)


def _slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x)
    return np.sum((y - np.mean(y, axis=-1, keepdims=True)) * centered, axis=-1) / np.sum(centered**2)


def gain_and_slope_tables(
    bootstrap: BootstrapPerformance,
    dimensions: np.ndarray,
    fixed_names: list[str],
    full_name: str,
) -> pd.DataFrame:
    names = list(bootstrap.representation_order)
    fixed_indices = np.asarray([names.index(name) for name in fixed_names], dtype=int)
    full_index = names.index(full_name)
    rows: list[dict[str, object]] = []
    early_positions = np.flatnonzero((dimensions >= 2) & (dimensions <= 16))
    late_positions = np.flatnonzero((dimensions >= 16) & (dimensions <= 32))
    position_2 = int(np.flatnonzero(dimensions == 2)[0])
    position_16 = int(np.flatnonzero(dimensions == 16)[0])
    position_32 = int(np.flatnonzero(dimensions == 32)[0])
    for metric_index, metric in enumerate(bootstrap.metric_order):
        observed_fixed = bootstrap.observed[fixed_indices, metric_index]
        draw_fixed = bootstrap.draws[:, fixed_indices, metric_index]
        observed_full = bootstrap.observed[full_index, metric_index]
        draw_full = bootstrap.draws[:, full_index, metric_index]
        quantities = {
            "gain_PCA2_to_PCA16": (
                observed_fixed[position_16] - observed_fixed[position_2],
                draw_fixed[:, position_16] - draw_fixed[:, position_2],
            ),
            "gain_PCA16_to_PCA32": (
                observed_fixed[position_32] - observed_fixed[position_16],
                draw_fixed[:, position_32] - draw_fixed[:, position_16],
            ),
            "early_capture_fraction_of_PCA2_to_FULL": (
                (observed_fixed[position_16] - observed_fixed[position_2])
                / (observed_full - observed_fixed[position_2]),
                (draw_fixed[:, position_16] - draw_fixed[:, position_2])
                / (draw_full - draw_fixed[:, position_2]),
            ),
            "beta_early_per_dimension": (
                float(_slope(dimensions[early_positions], observed_fixed[early_positions][None, :])[0]),
                _slope(dimensions[early_positions], draw_fixed[:, early_positions]),
            ),
            "beta_late_per_dimension": (
                float(_slope(dimensions[late_positions], observed_fixed[late_positions][None, :])[0]),
                _slope(dimensions[late_positions], draw_fixed[:, late_positions]),
            ),
        }
        early_slope = quantities["beta_early_per_dimension"]
        late_slope = quantities["beta_late_per_dimension"]
        quantities["beta_early_minus_beta_late"] = (
            early_slope[0] - late_slope[0], early_slope[1] - late_slope[1]
        )
        for quantity, (estimate, values) in quantities.items():
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "metric": metric,
                    "quantity": quantity,
                    "estimate": estimate,
                    "pointwise_95_ci_lower": float(np.quantile(finite, 0.025)),
                    "pointwise_95_ci_upper": float(np.quantile(finite, 0.975)),
                    "valid_bootstrap_draws": len(finite),
                    "bootstrap_draws": len(values),
                    "resampling_unit": "patient_profile",
                }
            )
    return pd.DataFrame.from_records(rows)


def adjacent_gain_table(
    bootstrap: BootstrapPerformance,
    dimensions: np.ndarray,
    fixed_names: list[str],
    practical_pg: float,
    practical_g: float,
) -> pd.DataFrame:
    names = list(bootstrap.representation_order)
    indices = np.asarray([names.index(name) for name in fixed_names], dtype=int)
    rows: list[dict[str, object]] = []
    for start in range(16, 32, 2):
        left = int(np.flatnonzero(dimensions == start)[0])
        right = int(np.flatnonzero(dimensions == start + 2)[0])
        for metric_index, metric in enumerate(bootstrap.metric_order):
            estimate = bootstrap.observed[indices[right], metric_index] - bootstrap.observed[indices[left], metric_index]
            values = bootstrap.draws[:, indices[right], metric_index] - bootstrap.draws[:, indices[left], metric_index]
            practical = practical_pg if metric == "PG_macro" else practical_g
            rows.append(
                {
                    "from_k": start,
                    "to_k": start + 2,
                    "metric": metric,
                    "estimate": estimate,
                    "pointwise_95_ci_lower": float(np.quantile(values, 0.025)),
                    "pointwise_95_ci_upper": float(np.quantile(values, 0.975)),
                    "practical_gain_scale": practical,
                    "positive_gain_reaches_practical_scale": bool(estimate >= practical),
                    "bootstrap_draws": len(values),
                    "resampling_unit": "patient_profile",
                }
            )
    return pd.DataFrame.from_records(rows)


def full_increment_table(
    bootstrap: BootstrapPerformance,
    fixed_names: list[str],
    dimensions: Iterable[int],
    full_name: str,
) -> pd.DataFrame:
    names = list(bootstrap.representation_order)
    full_index = names.index(full_name)
    rows: list[dict[str, object]] = []
    for k in dimensions:
        pca_index = names.index(f"PCA{k}")
        for metric_index, metric in enumerate(bootstrap.metric_order):
            estimate = bootstrap.observed[full_index, metric_index] - bootstrap.observed[pca_index, metric_index]
            values = bootstrap.draws[:, full_index, metric_index] - bootstrap.draws[:, pca_index, metric_index]
            rows.append(
                {
                    "contrast": f"FULL_RNA_REOPT_minus_PCA{k}",
                    "k": k,
                    "metric": metric,
                    "estimate": estimate,
                    "pointwise_95_ci_lower": float(np.quantile(values, 0.025)),
                    "pointwise_95_ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_draws": len(values),
                    "resampling_unit": "patient_profile",
                }
            )
    return pd.DataFrame.from_records(rows)


def plateau_band(
    performance: pd.DataFrame,
    reference_pg: float,
    reference_g: float,
    pg_scale: float,
    g_scale: float,
) -> pd.DataFrame:
    if "representation_family" in performance.columns:
        local = performance.query(
            "representation_family == 'fixed_PCA' and 16 <= dimension <= 32"
        ).copy()
    else:
        local = performance.query("16 <= dimension <= 32").copy()
    local["reference_PG"] = reference_pg
    local["reference_g_func"] = reference_g
    local["PG_deficit_from_reference"] = reference_pg - local["PG_macro"]
    local["g_func_deficit_from_reference"] = reference_g - local["g_func"]
    local["within_PG_practical_scale"] = local["PG_macro"] >= reference_pg - pg_scale
    local["within_g_func_practical_scale"] = local["g_func"] >= reference_g - g_scale
    local["in_plateau_band"] = local["within_PG_practical_scale"] & local["within_g_func_practical_scale"]
    return local


def deterministic_verdict(
    early_gain_pass: bool,
    joint_k90_by_32: bool,
    broad_plateau_pass: bool,
    dense_selection_pass: bool,
    random_structure_pass: bool,
    continued_scaling_pass: bool,
    qa_pass: bool,
) -> str:
    if not qa_pass:
        return "NO_REPRODUCIBLE_RESOLUTION_STRUCTURE"
    if all([early_gain_pass, joint_k90_by_32, broad_plateau_pass, dense_selection_pass, random_structure_pass]):
        return "PREDICTIVE_INFORMATION_SATURATION_SUPPORTED"
    if joint_k90_by_32 and dense_selection_pass and random_structure_pass:
        return "COMPACT_PREDICTIVE_STRUCTURE_SUPPORTED_WITHOUT_CLEAR_SATURATION"
    if continued_scaling_pass:
        return "PREDICTIVE_PERFORMANCE_CONTINUES_TO_SCALE_WITH_DIMENSION"
    return "NO_REPRODUCIBLE_RESOLUTION_STRUCTURE"
