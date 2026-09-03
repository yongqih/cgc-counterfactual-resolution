from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_m1_secondary import (
    GENE_COUNT,
    K_VALUES,
    M_VALUES,
    _geometry_from_sums_simple,
    _orders,
    _same_excess,
    _sentinel_matrix,
    _simultaneous,
    load_frozen_inputs,
)


TARGET_M = 2
TARGET_K = 92
TARGET_BUDGET = 278
TARGET_GRID_INDEX = M_VALUES.index(TARGET_M) * len(K_VALUES) + K_VALUES.index(TARGET_K)


def adjudicate(
    observed_lower_95: float,
    null_passes: np.ndarray,
    plate6_to_plate14: float,
    plate14_to_plate6: float,
) -> str:
    confirmed = (
        observed_lower_95 > 0
        and bool(np.asarray(null_passes, dtype=bool).all())
        and plate6_to_plate14 >= 0
        and plate14_to_plate6 >= 0
    )
    return (
        "M1_M2K92_CORRESPONDENCE_CONFIRMED"
        if confirmed
        else "M1_M2K92_CORRESPONDENCE_NOT_CONFIRMED"
    )


def _focused_directional(inputs: Any) -> tuple[pd.DataFrame, dict[str, float]]:
    truth_6_to_14 = np.zeros((50, 93), dtype=np.float64)
    residual_6_to_14 = np.zeros_like(truth_6_to_14)
    truth_14_to_6 = np.zeros_like(truth_6_to_14)
    residual_14_to_6 = np.zeros_like(truth_6_to_14)
    counts = np.zeros_like(truth_6_to_14, dtype=np.int16)

    for target in range(50):
        for sequence in range(8):
            support_order, sentinel_order = _orders(inputs.split, target, sequence)
            sources = support_order[:TARGET_M]
            sentinels = _sentinel_matrix(sentinel_order, TARGET_K)
            same6 = _same_excess(inputs.same6_4, target, sources)
            same14 = _same_excess(inputs.same14_4, target, sources)
            cross = _geometry_from_sums_simple(inputs.cross4, target, sources)
            alpha6 = inputs.alpha[(target, "plate6", TARGET_M, TARGET_K)]
            alpha14 = inputs.alpha[(target, "plate14", TARGET_M, TARGET_K)]
            for intervention in range(93):
                selected = sentinels[intervention]
                a6_norm = float(same6[np.ix_(selected, selected)].mean())
                a14_norm = float(same14[np.ix_(selected, selected)].mean())
                a6_to_truth14 = float(cross[selected, intervention].mean())
                truth6_to_a14 = float(cross[intervention, selected].mean())
                v14 = float(same14[intervention, intervention])
                v6 = float(same6[intervention, intervention])
                r6_to_14 = v14 - 2.0 * alpha6 * a6_to_truth14 + alpha6 * alpha6 * a6_norm
                r14_to_6 = v6 - 2.0 * alpha14 * truth6_to_a14 + alpha14 * alpha14 * a14_norm
                truth_6_to_14[target, intervention] += v14 / GENE_COUNT
                residual_6_to_14[target, intervention] += r6_to_14 / GENE_COUNT
                truth_14_to_6[target, intervention] += v6 / GENE_COUNT
                residual_14_to_6[target, intervention] += r14_to_6 / GENE_COUNT
                counts[target, intervention] += 1

    for value in (truth_6_to_14, residual_6_to_14, truth_14_to_6, residual_14_to_6):
        value /= counts
    frame = pd.DataFrame(
        {
            "context_index": np.repeat(np.arange(50), 93),
            "context_id": np.repeat(np.asarray(inputs.split["contexts"]), 93),
            "intervention_index": np.tile(np.arange(93), 50),
            "intervention_id": np.tile(np.asarray(inputs.split["interventions"]), 50),
            "plate6_to_plate14_truth": truth_6_to_14.ravel(),
            "plate6_to_plate14_residual": residual_6_to_14.ravel(),
            "plate14_to_plate6_truth": truth_14_to_6.ravel(),
            "plate14_to_plate6_residual": residual_14_to_6.ravel(),
        }
    )
    summary = {
        "plate6_to_plate14_g": float(1.0 - residual_6_to_14.sum() / truth_6_to_14.sum()),
        "plate14_to_plate6_g": float(1.0 - residual_14_to_6.sum() / truth_14_to_6.sum()),
    }
    return frame, summary


def run_confirmation(root: Path, upstream_root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_m1_secondary_audit"
    inputs = load_frozen_inputs(root, upstream_root.resolve())
    bootstrap = np.load(out / "_cache/m1_bootstrap_draws.npz", allow_pickle=False)
    model_draws = np.asarray(bootstrap["model"], dtype=np.float64)
    if model_draws.shape != (4, 10_000, 90):
        raise RuntimeError("M1_M2K92_CONFIRMATION_NOT_EXECUTABLE_FROM_FROZEN_ARTIFACTS")

    truth = np.repeat(inputs.utility["vtruth"][:, None], len(K_VALUES), axis=1).reshape(90, 50, 93)
    residual = np.asarray(inputs.utility["vafter"][1], dtype=np.float64).reshape(90, 50, 93)
    estimates = 1.0 - residual.sum(axis=(1, 2)) / truth.sum(axis=(1, 2))
    m1_low, m1_high, m1_critical = _simultaneous(estimates, model_draws[1])

    estimator_family = pd.read_csv(out / "M1_ESTIMATOR_FAMILY_CORRECTION.csv")
    strict_observed = estimator_family[
        (estimator_family.model == "M1_SENTINEL_OFFSET")
        & (estimator_family.m == TARGET_M)
        & (estimator_family.k == TARGET_K)
    ].iloc[0]
    nulls = pd.read_csv(out / "M1_CORRESPONDENCE_NULLS.csv")
    exact_nulls = nulls[(nulls.m == TARGET_M) & (nulls.k == TARGET_K)].copy()
    if len(exact_nulls) != 4 or not (exact_nulls.family_size == 360).all():
        raise RuntimeError("M1_M2K92_CONFIRMATION_NOT_EXECUTABLE_FROM_FROZEN_ARTIFACTS")

    directional, directional_summary = _focused_directional(inputs)
    pooled_truth = np.asarray(inputs.utility["vtruth"][M_VALUES.index(TARGET_M)], dtype=np.float64)
    pooled_residual = np.asarray(
        inputs.utility["vafter"][1, M_VALUES.index(TARGET_M), K_VALUES.index(TARGET_K)],
        dtype=np.float64,
    )
    pooled_g = float(1.0 - pooled_residual.sum() / pooled_truth.sum())
    context_g = 1.0 - pooled_residual.sum(axis=1) / pooled_truth.sum(axis=1)
    intervention_g = 1.0 - pooled_residual.sum(axis=0) / pooled_truth.sum(axis=0)
    context_positive_fraction = float(np.mean(context_g > 0))
    intervention_positive_fraction = float(np.mean(intervention_g > 0))

    exact_nulls.insert(0, "audit_point", "m=2,k=92")
    exact_nulls["primary_family"] = "ALL_FROZEN_M1_GRID_X_4_NULLS"
    exact_nulls["primary_family_size"] = 360
    exact_nulls["postselection_pass"] = (
        (exact_nulls.observed_minus_null > 0)
        & (exact_nulls.familywise_p_4x90 <= 0.05)
    )
    exact_nulls.to_csv(out / "M1_M2K92_NULL_COMPARISON.csv", index=False)

    inference_rows = [
        {
            "endpoint": "OBSERVED_M1_G",
            "family": "ALL_FROZEN_M1_GRID",
            "family_size": 90,
            "status": "EXECUTED",
            "estimate": estimates[TARGET_GRID_INDEX],
            "simultaneous_lower_95": m1_low[TARGET_GRID_INDEX],
            "simultaneous_upper_95": m1_high[TARGET_GRID_INDEX],
            "max_t_critical": m1_critical,
            "passed": bool(m1_low[TARGET_GRID_INDEX] > 0),
            "reason": "primary post-selection observed-signal family",
        },
        {
            "endpoint": "OBSERVED_M1_G",
            "family": "ALL_ESTIMATORS_X_ALL_FROZEN_GRID",
            "family_size": 360,
            "status": "EXECUTED_STRICTER_SENSITIVITY",
            "estimate": float(strict_observed.estimate),
            "simultaneous_lower_95": float(strict_observed.simultaneous_lower_95),
            "simultaneous_upper_95": float(strict_observed.simultaneous_upper_95),
            "max_t_critical": float(strict_observed.max_t_critical),
            "passed": bool(strict_observed.simultaneous_lower_95 > 0),
            "reason": "stricter frozen estimator-selection sensitivity",
        },
        {
            "endpoint": "NULL_CONTRASTS",
            "family": "ALL_FROZEN_M1_GRID_X_4_NULLS",
            "family_size": 360,
            "status": "EXECUTED_PRIMARY",
            "estimate": np.nan,
            "simultaneous_lower_95": np.nan,
            "simultaneous_upper_95": np.nan,
            "max_t_critical": np.nan,
            "passed": bool(exact_nulls.postselection_pass.all()),
            "reason": "single-step paired max-statistic family; see null comparison",
        },
        {
            "endpoint": "NULL_CONTRASTS",
            "family": "ALL_ESTIMATORS_X_ALL_FROZEN_GRID_X_4_NULLS",
            "family_size": 1_440,
            "status": "NOT_EXECUTABLE_FROM_FROZEN_STATS",
            "estimate": np.nan,
            "simultaneous_lower_95": np.nan,
            "simultaneous_upper_95": np.nan,
            "max_t_critical": np.nan,
            "passed": False,
            "reason": "M0 and M3 null residual tables were not frozen; no model/null rerun allowed",
        },
    ]
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(out / "M1_M2K92_POSTSELECTION_INFERENCE.csv", index=False)

    verdict = adjudicate(
        float(m1_low[TARGET_GRID_INDEX]),
        exact_nulls.postselection_pass.to_numpy(dtype=bool),
        directional_summary["plate6_to_plate14_g"],
        directional_summary["plate14_to_plate6_g"],
    )
    directional_contradiction = (
        directional_summary["plate6_to_plate14_g"] < 0
        or directional_summary["plate14_to_plate6_g"] < 0
    )
    null_table = exact_nulls[
        [
            "null", "observed_m1_g", "null_g", "observed_minus_null",
            "difference_pointwise_lower_95", "difference_pointwise_upper_95",
            "familywise_p_4x90", "postselection_pass",
        ]
    ].to_csv(index=False)
    report = f"""# M1 m=2,k=92 Frozen Correspondence-Null Confirmation

Identity: post-hoc confirmation of the frozen descriptive M1 maximum. No estimator,
prediction, alpha, support/sentinel set, grid, seed, bootstrap unit, or null was
refit, regenerated, changed, or selected in this confirmation.

## Exact point

- `m=2`, `k=92`, budget `278/4650` ({TARGET_BUDGET / 4650:.6%}).
- Pooled cross-product g: `{pooled_g:.9f}`.
- M1-grid (90-member) simultaneous 95% CI:
  `[{m1_low[TARGET_GRID_INDEX]:.9f}, {m1_high[TARGET_GRID_INDEX]:.9f}]`.
- Stricter estimator x grid (360-member) simultaneous 95% CI:
  `[{float(strict_observed.simultaneous_lower_95):.9f}, {float(strict_observed.simultaneous_upper_95):.9f}]`.

The positive observed M1 signal survives both frozen observed-surface families.
That alone is not correspondence confirmation.

## Primary post-selection null family

The primary family is all 90 frozen M1 grid cells x four frozen nulls (360
contrasts) using the frozen 10,000-draw paired hierarchical bootstrap and
single-step maximum statistic.

```csv
{null_table.strip()}
```

Only the context-identity contrast passes. Intervention identity is higher than
observed, sentinel identity is also higher than observed, and the equal-weight
M1 estimator is exactly invariant to the frozen cyclic reference-context shift.
Therefore all-four-null confirmation fails.

The stricter all-estimator x grid x null family cannot be constructed from
frozen statistics because M0 and M3 null residual tables were never frozen.
It is explicitly marked `NOT_EXECUTABLE_FROM_FROZEN_STATS`; the protocol
forbids rerunning models or null predictions to manufacture that sensitivity.

## Direction and heterogeneity

- Plate6 to Plate14 g: `{directional_summary['plate6_to_plate14_g']:.9f}`.
- Plate14 to Plate6 g: `{directional_summary['plate14_to_plate6_g']:.9f}`.
- Pooled cross-product g: `{pooled_g:.9f}`.
- Positive target contexts: `{int(np.sum(context_g > 0))}/50 = {context_positive_fraction:.6%}`.
- Positive interventions: `{int(np.sum(intervention_g > 0))}/93 = {intervention_positive_fraction:.6%}`.

The pooled signal is broad, but both conventional cross-plate directions are
negative. This is a clear directional contradiction, not a minor power loss.

## Final adjudication

The observed M1 maximum is real as a frozen pooled association, but it is not
correspondence-specific: three of four jointly corrected null contrasts fail,
and both directional transfers are negative. Per protocol, M1 is closed after
this adjudication and is not promoted as a CGC remedy.

{verdict}
"""
    (out / "M1_M2K92_CORRESPONDENCE_CONFIRMATION.md").write_text(report, encoding="utf-8")
    return {
        "verdict": verdict,
        "pooled_g": pooled_g,
        "m1_grid_simultaneous_lower_95": float(m1_low[TARGET_GRID_INDEX]),
        "m1_grid_simultaneous_upper_95": float(m1_high[TARGET_GRID_INDEX]),
        "strict_estimator_grid_lower_95": float(strict_observed.simultaneous_lower_95),
        "strict_estimator_grid_upper_95": float(strict_observed.simultaneous_upper_95),
        "plate6_to_plate14_g": directional_summary["plate6_to_plate14_g"],
        "plate14_to_plate6_g": directional_summary["plate14_to_plate6_g"],
        "context_positive_fraction": context_positive_fraction,
        "intervention_positive_fraction": intervention_positive_fraction,
        "directional_contradiction": bool(directional_contradiction),
    }
