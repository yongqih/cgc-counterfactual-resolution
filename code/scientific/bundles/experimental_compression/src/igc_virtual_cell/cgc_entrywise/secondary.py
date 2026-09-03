from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_entrywise.analysis import K_VALUES, M_VALUES, MODELS
from igc_virtual_cell.cgc_entrywise.inference import (
    BOOTSTRAPS,
    BOOTSTRAP_SEED,
    PRIMARY_MODEL,
    THRESHOLDS,
    _hierarchical_weights,
    _simultaneous_band,
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_rna_orders(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "data/tahoe100m_control_state/control_wells_log1p_cpm_float32.npy"
    axes_path = root / "data/tahoe100m_control_state/axes.json"
    axes = json.loads(axes_path.read_text(encoding="utf-8"))
    values = np.load(data, mmap_mode="r")
    if values.shape != (4, 50, 42_523):
        raise RuntimeError("Baseline RNA axis changed")
    baseline = np.asarray(values.mean(axis=0, dtype=np.float64), dtype=np.float64)
    baseline -= baseline.mean(axis=0, keepdims=True)
    scale = baseline.std(axis=0, ddof=1)
    keep = np.isfinite(scale) & (scale > 1e-8)
    standardized = baseline[:, keep] / scale[keep]
    u, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    scores = u[:, :32] * singular[:32]
    distances = np.sqrt(
        np.maximum(
            np.sum(scores * scores, axis=1)[:, None]
            + np.sum(scores * scores, axis=1)[None, :]
            - 2 * scores @ scores.T,
            0,
        )
    )
    contexts = list(axes["contexts"])
    orders: dict[str, list[str]] = {}
    for target, context in enumerate(contexts):
        candidates = [index for index in range(50) if index != target]
        candidates.sort(key=lambda index: (float(distances[target, index]), contexts[index]))
        orders[context] = [contexts[index] for index in candidates]
    result = {
        "phase": "CGC-EC-2",
        "created_at": _now(),
        "method": "rank-32 SVD of gene-standardized mean four-well baseline log1p CPM; Euclidean nearest contexts",
        "treated_outcome_used": False,
        "baseline_shape": list(values.shape),
        "nonconstant_genes": int(keep.sum()),
        "rank": 32,
        "baseline_sha256": _sha256(data),
        "axes_sha256": _sha256(axes_path),
        "contexts": contexts,
        "orders": orders,
    }
    path = root / "results/cgc_entrywise_compression/ENTRYWISE_RNA_SUPPORT_ORDERS.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _threshold_rows(estimate: np.ndarray, lower: np.ndarray) -> list[dict[str, Any]]:
    grid = []
    index = 0
    for m in M_VALUES:
        for k in K_VALUES:
            grid.append(
                {
                    "index": index,
                    "m": m,
                    "k": k,
                    "budget": 93 * m + k,
                    "matrix_fraction": (93 * m + k) / 4_650,
                    "estimate": estimate[index],
                    "lower": lower[index],
                }
            )
            index += 1
    frame = pd.DataFrame(grid).sort_values(["budget", "m", "k"], kind="stable")
    labels = ("B_detect_95", "B25_95", "B50_95", "B80_95", "B95_95")
    rows: list[dict[str, Any]] = []
    for label, threshold in zip(labels, THRESHOLDS, strict=True):
        eligible = frame[frame.lower > 0] if threshold == 0 else frame[frame.lower >= threshold]
        if len(eligible):
            value = eligible.iloc[0]
            rows.append(
                {
                    "threshold": label,
                    "target_recovery": threshold,
                    "budget": int(value.budget),
                    "matrix_fraction": float(value.matrix_fraction),
                    "m": int(value.m),
                    "k": int(value.k),
                    "estimate": float(value.estimate),
                    "simultaneous_lower_95": float(value.lower),
                }
            )
        else:
            rows.append(
                {
                    "threshold": label,
                    "target_recovery": threshold,
                    "budget": "NOT_REACHED_WITHIN_FULL_GRID",
                    "matrix_fraction": "NOT_REACHED_WITHIN_FULL_GRID",
                    "m": "NOT_REACHED_WITHIN_FULL_GRID",
                    "k": "NOT_REACHED_WITHIN_FULL_GRID",
                    "estimate": "NOT_REACHED_WITHIN_FULL_GRID",
                    "simultaneous_lower_95": "NOT_REACHED_WITHIN_FULL_GRID",
                }
            )
    return rows


def run_rna_inference(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_entrywise_compression"
    random = np.load(out / "_cache/frozen_utility_table.npz", allow_pickle=False)
    rna = np.load(out / "_cache/rna_utility_table.npz", allow_pickle=False)
    primary = MODELS.index(PRIMARY_MODEL)
    random_truth = np.repeat(random["vtruth"][:, None], len(K_VALUES), axis=1).reshape(90, 50, 93)
    rna_truth = np.repeat(rna["vtruth"][:, None], len(K_VALUES), axis=1).reshape(90, 50, 93)
    random_residual = random["vafter"][primary].reshape(90, 50, 93)
    rna_residual = rna["vafter"][primary].reshape(90, 50, 93)
    random_estimate = 1 - random_residual.sum(axis=(1, 2)) / random_truth.sum(axis=(1, 2))
    rna_estimate = 1 - rna_residual.sum(axis=(1, 2)) / rna_truth.sum(axis=(1, 2))
    random_draws = np.empty((BOOTSTRAPS, 90), dtype=np.float64)
    rna_draws = np.empty_like(random_draws)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    batch = 25
    for start in range(0, BOOTSTRAPS, batch):
        stop = min(BOOTSTRAPS, start + batch)
        weights = _hierarchical_weights(rng, stop - start).astype(np.float64)
        for table_truth, table_residual, destination in (
            (random_truth, random_residual, random_draws),
            (rna_truth, rna_residual, rna_draws),
        ):
            denominator = weights @ table_truth.reshape(90, -1).T
            numerator = weights @ table_residual.reshape(90, -1).T
            destination[start:stop] = 1 - numerator / denominator
    lower, upper, critical = _simultaneous_band(rna_estimate, rna_draws)
    point_low, point_high = np.quantile(rna_draws, [0.025, 0.975], axis=0)
    difference = rna_draws - random_draws
    difference_low, difference_high = np.quantile(difference, [0.025, 0.975], axis=0)

    surface = pd.read_csv(out / "RNA_INFORMED_COMPRESSION_RESULTS.csv")
    surface = surface[surface.model == PRIMARY_MODEL].sort_values(["m", "k"], kind="stable").reset_index(drop=True)
    surface["pointwise_lower_95"] = point_low
    surface["pointwise_upper_95"] = point_high
    surface["simultaneous_lower_95"] = lower
    surface["simultaneous_upper_95"] = upper
    surface["max_t_critical"] = critical
    surface["delta_g_vs_random"] = rna_estimate - random_estimate
    surface["delta_pointwise_lower_95"] = difference_low
    surface["delta_pointwise_upper_95"] = difference_high
    surface["rna_significantly_better_at_budget"] = difference_low > 0
    surface.to_csv(out / "RNA_INFORMED_COMPRESSION_RESULTS.csv", index=False)
    thresholds = _threshold_rows(rna_estimate, lower)
    random_thresholds = pd.read_csv(out / "ENTRYWISE_COMPRESSION_THRESHOLDS.csv", dtype=str).set_index("threshold")
    for row in thresholds:
        random_budget = random_thresholds.loc[row["threshold"], "budget"]
        row["random_budget"] = random_budget
        if isinstance(row["budget"], int) and random_budget != "NOT_REACHED_WITHIN_FULL_GRID":
            row["budget_reduction"] = int(float(random_budget)) - row["budget"]
        else:
            row["budget_reduction"] = "NOT_ESTABLISHED"
    pd.DataFrame(thresholds).to_csv(out / "RNA_INFORMED_COMPRESSION_THRESHOLDS.csv", index=False)
    significant_any = bool(surface.rna_significantly_better_at_budget.any())
    return {
        "created_at": _now(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "simultaneous_max_t_critical": critical,
        "thresholds": thresholds,
        "rna_significantly_better_at_any_frozen_budget": significant_any,
    }
