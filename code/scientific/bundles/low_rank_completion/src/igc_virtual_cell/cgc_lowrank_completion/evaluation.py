from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import CONTEXTS, INTERVENTIONS


@dataclass(frozen=True)
class FoldEvaluation:
    prediction_rows: list[dict[str, object]]
    diagnostic_rows: list[dict[str, object]]


def _paired_bilinear(left: np.ndarray, gram: np.ndarray, right: np.ndarray, device: str) -> np.ndarray:
    import torch

    dev = torch.device(device)
    with torch.no_grad():
        k = torch.as_tensor(np.asarray(gram, dtype=np.float32), device=dev)
        l = torch.as_tensor(np.asarray(left, dtype=np.float32), device=dev)
        r = torch.as_tensor(np.asarray(right, dtype=np.float32), device=dev)
        value = torch.sum(l * (k @ r.T).T, dim=1)
        return value.cpu().numpy().astype(np.float64)


def _baseline_weights(train: np.ndarray, targets: np.ndarray) -> np.ndarray:
    position = np.full(CONTEXTS * INTERVENTIONS, -1, dtype=np.int64)
    position[train] = np.arange(len(train), dtype=np.int64)
    result = np.zeros((len(targets), len(train)), dtype=np.float64)
    for row, target in enumerate(map(int, targets)):
        context = target // INTERVENTIONS
        intervention = target % INTERVENTIONS
        source = np.asarray(
            [other * INTERVENTIONS + intervention for other in range(CONTEXTS) if other != context],
            dtype=np.int64,
        )
        locations = position[source]
        if np.any(locations < 0):
            raise RuntimeError("LOW_RANK_CONTEXT_BASELINE_SUPPORT_FAIL")
        result[row, locations] = 1.0 / (CONTEXTS - 1)
    return result


def _truth_cross(
    cross: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    baseline: np.ndarray,
    device: str,
) -> np.ndarray:
    direct = np.asarray(np.diag(cross[np.ix_(targets, targets)]), dtype=np.float64)
    left = np.einsum(
        "ij,ji->i", baseline, np.asarray(cross[np.ix_(train, targets)], dtype=np.float64), optimize=True
    )
    right = np.einsum(
        "ij,ij->i", baseline, np.asarray(cross[np.ix_(targets, train)], dtype=np.float64), optimize=True
    )
    base = _paired_bilinear(
        baseline, np.asarray(cross[np.ix_(train, train)], dtype=np.float32), baseline, device
    )
    return direct - left - right + base


def _same_plate_diagnostics(
    gram: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_prediction = prediction - baseline
    direct = np.asarray(np.diag(gram[np.ix_(targets, targets)]), dtype=np.float64)
    truth_base = np.einsum(
        "ij,ji->i", baseline, np.asarray(gram[np.ix_(train, targets)], dtype=np.float64), optimize=True
    )
    base_norm = _paired_bilinear(
        baseline, np.asarray(gram[np.ix_(train, train)], dtype=np.float32), baseline, device
    )
    truth_norm = direct - 2.0 * truth_base + base_norm
    pred_norm = _paired_bilinear(
        d_prediction, np.asarray(gram[np.ix_(train, train)], dtype=np.float32), d_prediction, device
    )
    truth_dot_pred = np.einsum(
        "ij,ji->i",
        d_prediction,
        np.asarray(gram[np.ix_(train, targets)], dtype=np.float64),
        optimize=True,
    ) - _paired_bilinear(
        baseline,
        np.asarray(gram[np.ix_(train, train)], dtype=np.float32),
        d_prediction,
        device,
    )
    return truth_norm, pred_norm, truth_dot_pred


def evaluate_fold(
    same6: np.ndarray,
    same14: np.ndarray,
    cross: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    model_weights: dict[str, tuple[np.ndarray, np.ndarray]],
    fold: int,
    *,
    device: str = "cuda",
) -> FoldEvaluation:
    train = np.asarray(train, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    baseline = _baseline_weights(train, targets)
    cross_train = np.asarray(cross[np.ix_(train, train)], dtype=np.float32)
    full_truth = np.asarray(np.diag(cross[np.ix_(targets, targets)]), dtype=np.float64)
    context_truth = _truth_cross(cross, train, targets, baseline, device)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for model, (weights6, weights14) in model_weights.items():
        pred6_truth14 = np.einsum(
            "ij,ji->i", weights6, np.asarray(cross[np.ix_(train, targets)], dtype=np.float64), optimize=True
        )
        truth6_pred14 = np.einsum(
            "ij,ij->i", weights14, np.asarray(cross[np.ix_(targets, train)], dtype=np.float64), optimize=True
        )
        pred_cross = _paired_bilinear(weights6, cross_train, weights14, device)
        full_after = full_truth - pred6_truth14 - truth6_pred14 + pred_cross

        d6 = weights6 - baseline
        d14 = weights14 - baseline
        d6_truth14 = np.einsum(
            "ij,ji->i", d6, np.asarray(cross[np.ix_(train, targets)], dtype=np.float64), optimize=True
        ) - _paired_bilinear(d6, cross_train, baseline, device)
        truth6_d14 = np.einsum(
            "ij,ij->i", d14, np.asarray(cross[np.ix_(targets, train)], dtype=np.float64), optimize=True
        ) - _paired_bilinear(baseline, cross_train, d14, device)
        d_cross = _paired_bilinear(d6, cross_train, d14, device)
        context_after = context_truth - d6_truth14 - truth6_d14 + d_cross
        if not np.allclose(full_after, context_after, atol=1e-2, rtol=2e-5):
            raise RuntimeError("LOW_RANK_FULL_CONTEXT_RESIDUAL_RECONCILIATION_FAIL")

        plate_stats = []
        for plate, (gram, weights) in enumerate(((same6, weights6), (same14, weights14))):
            truth_norm, pred_norm, truth_dot = _same_plate_diagnostics(
                gram, train, targets, baseline, weights, device
            )
            plate_stats.append((truth_norm, pred_norm, truth_dot))
            for i, target in enumerate(map(int, targets)):
                diagnostics.append(
                    {
                        "outer_fold": fold,
                        "target_index": target,
                        "context_index": target // INTERVENTIONS,
                        "intervention_index": target % INTERVENTIONS,
                        "model": model,
                        "component": "context_specific_prediction",
                        "plate": plate,
                        "truth_norm": truth_norm[i],
                        "predicted_norm": pred_norm[i],
                        "truth_dot_prediction": truth_dot[i],
                        "truth_aligned_amplitude": truth_dot[i] / truth_norm[i] if truth_norm[i] > 0 else np.nan,
                        "cosine": truth_dot[i] / np.sqrt(truth_norm[i] * pred_norm[i]) if truth_norm[i] > 0 and pred_norm[i] > 0 else np.nan,
                        "scale_ratio": np.sqrt(pred_norm[i] / truth_norm[i]) if truth_norm[i] > 0 and pred_norm[i] >= 0 else np.nan,
                    }
                )
        for i, target in enumerate(map(int, targets)):
            rows.append(
                {
                    "outer_fold": fold,
                    "target_index": target,
                    "context_index": target // INTERVENTIONS,
                    "intervention_index": target % INTERVENTIONS,
                    "model": model,
                    "full_truth_cross": full_truth[i],
                    "full_residual_cross": full_after[i],
                    "context_truth_cross": context_truth[i],
                    "context_residual_cross": context_after[i],
                    "context_predicted_cross": d_cross[i],
                    "context_truth_dot_prediction_symmetric": 0.5 * (d6_truth14[i] + truth6_d14[i]),
                    "plate6_truth_norm": plate_stats[0][0][i],
                    "plate6_predicted_norm": plate_stats[0][1][i],
                    "plate6_truth_dot_prediction": plate_stats[0][2][i],
                    "plate14_truth_norm": plate_stats[1][0][i],
                    "plate14_predicted_norm": plate_stats[1][1][i],
                    "plate14_truth_dot_prediction": plate_stats[1][2][i],
                }
            )

    low6, low14 = model_weights["LOW_RANK_INTERACTION"]
    add6, add14 = model_weights["ADDITIVE_MAIN_EFFECT"]
    interaction6 = low6 - add6
    interaction14 = low14 - add14
    interaction_cross = _paired_bilinear(interaction6, cross_train, interaction14, device)
    low_cross = _paired_bilinear(low6 - baseline, cross_train, low14 - baseline, device)
    for i, target in enumerate(map(int, targets)):
        diagnostics.append(
            {
                "outer_fold": fold,
                "target_index": target,
                "context_index": target // INTERVENTIONS,
                "intervention_index": target % INTERVENTIONS,
                "model": "LOW_RANK_INTERACTION",
                "component": "cp_interaction_only",
                "plate": "cross_plate",
                "truth_norm": context_truth[i],
                "predicted_norm": interaction_cross[i],
                "truth_dot_prediction": np.nan,
                "truth_aligned_amplitude": np.nan,
                "cosine": np.nan,
                "scale_ratio": np.nan,
                "predicted_interaction_energy_fraction": interaction_cross[i] / low_cross[i] if low_cross[i] != 0 else np.nan,
            }
        )
    return FoldEvaluation(rows, diagnostics)
