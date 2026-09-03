from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_entrywise.analysis import K_VALUES, M_VALUES, MODELS, NULLS


BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 202_608_225
PRIMARY_MODEL = "M2_AFFINE_RIDGE"
THRESHOLDS = (0.0, 0.25, 0.50, 0.80, 0.95)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _hierarchical_weights(rng: np.random.Generator, draws: int) -> np.ndarray:
    weights = np.zeros((draws, 50 * 93), dtype=np.float32)
    for draw in range(draws):
        sampled_contexts = rng.integers(0, 50, size=50)
        sampled_interventions = rng.integers(0, 93, size=(50, 93))
        for occurrence, context in enumerate(sampled_contexts):
            indices, counts = np.unique(sampled_interventions[occurrence], return_counts=True)
            weights[draw, context * 93 + indices] += counts.astype(np.float32)
    return weights


def _bootstrap_ratios(
    truth: np.ndarray,
    residual_tables: np.ndarray,
    full_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return context-specific and full-response g draws.

    truth has shape grid-by-context-by-intervention. residual_tables has
    surface-by-grid-by-context-by-intervention.
    """

    grid = truth.shape[0]
    surfaces = residual_tables.shape[0]
    truth_flat = truth.reshape(grid, -1).astype(np.float64)
    residual_flat = residual_tables.reshape(surfaces * grid, -1).astype(np.float64)
    full_flat = np.broadcast_to(full_truth.reshape(1, -1), (grid, 50 * 93)).astype(np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    g_context = np.empty((surfaces, BOOTSTRAPS, grid), dtype=np.float64)
    g_full = np.empty_like(g_context)
    batch_size = 25
    for start in range(0, BOOTSTRAPS, batch_size):
        stop = min(BOOTSTRAPS, start + batch_size)
        weights = _hierarchical_weights(rng, stop - start).astype(np.float64)
        denominator = weights @ truth_flat.T
        denominator_full = weights @ full_flat.T
        numerator = (weights @ residual_flat.T).reshape(stop - start, surfaces, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_context[:, start:stop] = np.moveaxis(1.0 - numerator / denominator[:, None, :], 1, 0)
            g_full[:, start:stop] = np.moveaxis(1.0 - numerator / denominator_full[:, None, :], 1, 0)
        if stop % 500 == 0:
            print(f"entrywise bootstrap {stop}/{BOOTSTRAPS}", flush=True)
    return g_context, g_full


def _simultaneous_band(estimate: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    standard_error = np.std(draws, axis=0, ddof=1)
    scale = np.maximum(standard_error, 1e-12)
    statistic = np.max((estimate[None, :] - draws) / scale[None, :], axis=1)
    critical = float(np.quantile(statistic, 0.95))
    return estimate - critical * scale, estimate + critical * scale, critical


def run_inference(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_entrywise_compression"
    utility = np.load(out / "_cache/frozen_utility_table.npz", allow_pickle=False)
    vtruth_m = np.asarray(utility["vtruth"], dtype=np.float64)
    vafter = np.asarray(utility["vafter"], dtype=np.float64)
    nulls = np.asarray(utility["nulls"], dtype=np.float64)
    full_truth = np.asarray(utility["full_truth"], dtype=np.float64)
    primary_index = MODELS.index(PRIMARY_MODEL)

    truth_grid = np.repeat(vtruth_m[:, None], len(K_VALUES), axis=1).reshape(90, 50, 93)
    observed = vafter[primary_index].reshape(90, 50, 93)
    residuals = np.concatenate((observed[None], nulls.reshape(len(NULLS), 90, 50, 93)), axis=0)
    context_draws, full_draws = _bootstrap_ratios(truth_grid, residuals, full_truth)
    cache_path = out / "_cache/bootstrap_draws.npz"
    np.savez(
        cache_path,
        observed_context=context_draws[0],
        null_context=context_draws[1:],
        observed_full=full_draws[0],
    )

    truth_sum = truth_grid.sum(axis=(1, 2))
    observed_estimate = 1.0 - observed.sum(axis=(1, 2)) / truth_sum
    observed_full_estimate = 1.0 - observed.sum(axis=(1, 2)) / full_truth.sum()
    null_estimate = 1.0 - nulls.reshape(len(NULLS), 90, 50, 93).sum(axis=(2, 3)) / truth_sum[None]
    simultaneous_low, simultaneous_high, critical = _simultaneous_band(
        observed_estimate, context_draws[0]
    )
    point_low, point_high = np.quantile(context_draws[0], [0.025, 0.975], axis=0)
    full_low, full_high = np.quantile(full_draws[0], [0.025, 0.975], axis=0)
    null_low = np.quantile(context_draws[1:], 0.025, axis=1)
    null_high = np.quantile(context_draws[1:], 0.975, axis=1)

    point_rows: list[dict[str, Any]] = []
    simultaneous_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    grid_index = 0
    for m in M_VALUES:
        for k in K_VALUES:
            budget = 93 * m + k
            point_rows.append(
                {
                    "model": PRIMARY_MODEL,
                    "m": m,
                    "k": k,
                    "budget": budget,
                    "matrix_fraction": budget / 4_650,
                    "g_context_specific": observed_estimate[grid_index],
                    "pointwise_lower_95": point_low[grid_index],
                    "pointwise_upper_95": point_high[grid_index],
                    "g_full_response": observed_full_estimate[grid_index],
                    "full_pointwise_lower_95": full_low[grid_index],
                    "full_pointwise_upper_95": full_high[grid_index],
                }
            )
            simultaneous_rows.append(
                {
                    "model": PRIMARY_MODEL,
                    "m": m,
                    "k": k,
                    "budget": budget,
                    "matrix_fraction": budget / 4_650,
                    "estimate": observed_estimate[grid_index],
                    "simultaneous_lower_95": simultaneous_low[grid_index],
                    "simultaneous_upper_95": simultaneous_high[grid_index],
                    "max_t_critical": critical,
                    "family_size": 90,
                }
            )
            maximum_null_upper = float(np.max(null_high[:, grid_index]))
            for null_index, null_name in enumerate(NULLS):
                null_rows.append(
                    {
                        "null": null_name,
                        "model": PRIMARY_MODEL,
                        "m": m,
                        "k": k,
                        "budget": budget,
                        "observed_g": observed_estimate[grid_index],
                        "null_g": null_estimate[null_index, grid_index],
                        "null_lower_95": null_low[null_index, grid_index],
                        "null_upper_95": null_high[null_index, grid_index],
                        "observed_exceeds_null_upper_95": bool(
                            observed_estimate[grid_index] > null_high[null_index, grid_index]
                        ),
                        "all_nulls_separated": bool(
                            observed_estimate[grid_index] > maximum_null_upper
                        ),
                    }
                )
            grid_index += 1
    pd.DataFrame(point_rows).to_csv(out / "ENTRYWISE_RECOVERY_SURFACE_POINTWISE_CI.csv", index=False)
    pd.DataFrame(simultaneous_rows).to_csv(out / "ENTRYWISE_RECOVERY_SURFACE_SIMULTANEOUS_CI.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out / "ENTRYWISE_NULL_CONTROLS.csv", index=False)

    simultaneous_frame = pd.DataFrame(simultaneous_rows).sort_values(
        ["budget", "m", "k"], kind="stable"
    )
    threshold_rows: list[dict[str, Any]] = []
    labels = ("B_detect_95", "B25_95", "B50_95", "B80_95", "B95_95")
    for label, threshold in zip(labels, THRESHOLDS, strict=True):
        if threshold == 0.0:
            eligible = simultaneous_frame[simultaneous_frame["simultaneous_lower_95"] > 0]
        else:
            eligible = simultaneous_frame[
                simultaneous_frame["simultaneous_lower_95"] >= threshold
            ]
        if len(eligible):
            row = eligible.iloc[0]
            budget: int | str = int(row["budget"])
            fraction: float | str = float(row["matrix_fraction"])
            m_value: int | str = int(row["m"])
            k_value: int | str = int(row["k"])
            estimate: float | str = float(row["estimate"])
            lower: float | str = float(row["simultaneous_lower_95"])
        else:
            budget = fraction = m_value = k_value = estimate = lower = "NOT_REACHED_WITHIN_FULL_GRID"
        threshold_rows.append(
            {
                "threshold": label,
                "target_recovery": threshold,
                "budget": budget,
                "matrix_fraction": fraction,
                "m": m_value,
                "k": k_value,
                "estimate": estimate,
                "simultaneous_lower_95": lower,
                "model": PRIMARY_MODEL,
                "support_selection": "NESTED_RANDOM_PRIMARY",
            }
        )
    pd.DataFrame(threshold_rows).to_csv(out / "ENTRYWISE_COMPRESSION_THRESHOLDS.csv", index=False)

    frontier = simultaneous_frame.copy()
    frontier["pareto_nondominated"] = False
    best = -np.inf
    for index, row in frontier.iterrows():
        if float(row["estimate"]) > best:
            frontier.loc[index, "pareto_nondominated"] = True
            best = float(row["estimate"])
    frontier["cumulative_best_estimate"] = frontier["estimate"].cummax()
    frontier["cumulative_best_simultaneous_lower_95"] = frontier[
        "simultaneous_lower_95"
    ].cummax()
    frontier.to_csv(out / "ENTRYWISE_COMPRESSION_FRONTIER.csv", index=False)
    return {
        "created_at": _now(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "simultaneous_family": 90,
        "max_t_critical": critical,
        "thresholds": threshold_rows,
    }
