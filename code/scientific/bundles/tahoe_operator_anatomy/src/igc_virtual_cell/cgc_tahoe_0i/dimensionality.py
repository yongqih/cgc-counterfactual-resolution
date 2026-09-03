"""Cross-fitted signal-capture dimensionality of the maximal-support residual."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr

from igc_virtual_cell.cgc_tahoe_0i.scaling import _folds, _support_orders


RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
SEED = 2_608


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _weight_matrix(weights: np.ndarray, orders: np.ndarray, fold: int, direction: int) -> np.ndarray:
    matrix = np.zeros((50, 50), dtype=np.float32)
    for target in range(50):
        matrix[target, orders[target, 0, :49]] = weights[fold, direction, target]
    return matrix


def run_dimensionality(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    scaling = json.loads((out / "scaling_summary.json").read_text(encoding="utf-8"))
    if scaling["residual_null"]["empirical_p"] >= 0.05:
        raise RuntimeError("Maximal residual was not significant; dimensionality not authorized")
    group = zarr.open_group(out / "response_tensors.zarr", mode="r")
    indices = np.load(out / "gene_indices_g_primary.npy")
    selection = (slice(None), slice(None), slice(None), indices)
    delta = np.asarray(group["delta_primary"].get_orthogonal_selection(selection), dtype=np.float32)
    contexts = list(group.attrs["contexts"])
    interventions = list(group.attrs["interventions"])
    folds = _folds(root, interventions)
    orders = _support_orders(root, contexts)
    weights = np.load(out / "primary_q49_affine_weights.npy")
    if weights.shape != (5, 2, 50, 49):
        raise RuntimeError("Unexpected maximal-support weight tensor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum = min(1024, len(indices), 50 * int(93 - np.max(np.bincount(folds))))
    tested = [rank for rank in RANKS if rank <= maximum]
    if not tested:
        raise RuntimeError("No requested dimensionality rank is feasible")
    capture_sum = np.zeros(len(tested), dtype=np.float64)
    denominator_sum = 0.0
    component_signal_sum = np.zeros(maximum, dtype=np.float64)
    rows = []
    peak_gpu = 0
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        evaluation = np.flatnonzero(folds == fold)
        train_values = delta[:, :, train] - delta[:, :, train].mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
        eval_values = delta[:, :, evaluation] - delta[:, :, evaluation].mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
        for direction in range(2):
            w = _weight_matrix(weights, orders, fold, direction)
            train_fit = train_values[direction]
            residual_train = train_fit - np.einsum("cd,dpg->cpg", w, train_fit, optimize=True)
            residual6 = eval_values[0] - np.einsum("cd,dpg->cpg", w, eval_values[0], optimize=True)
            residual14 = eval_values[1] - np.einsum("cd,dpg->cpg", w, eval_values[1], optimize=True)
            matrix = torch.from_numpy(residual_train.reshape(-1, len(indices))).to(device)
            # Training plate and interventions only. q=maximum is frozen; two
            # power iterations balance accuracy and the 72-hour compute gate.
            _, singular, basis = torch.pca_lowrank(matrix, q=maximum, center=False, niter=2)
            eval6 = torch.from_numpy(residual6.reshape(-1, len(indices))).to(device)
            eval14 = torch.from_numpy(residual14.reshape(-1, len(indices))).to(device)
            coefficient6 = eval6 @ basis
            coefficient14 = eval14 @ basis
            component = torch.sum(
                coefficient6.to(torch.float64) * coefficient14.to(torch.float64), axis=0
            ).detach().cpu().numpy()
            denominator = float(
                torch.sum(eval6.to(torch.float64) * eval14.to(torch.float64)).detach().cpu()
            )
            cumulative = np.cumsum(component)
            denominator_sum += denominator
            component_signal_sum[: len(component)] += component
            for index, rank in enumerate(tested):
                capture_sum[index] += cumulative[rank - 1]
                rows.append(
                    {
                        "fold": fold,
                        "basis_fit_plate": ("plate6", "plate14")[direction],
                        "rank": rank,
                        "captured_crossplate_signal": float(cumulative[rank - 1]),
                        "total_crossplate_signal": denominator,
                        "capture_fraction_secondary": float(cumulative[rank - 1] / denominator),
                        "training_interventions": len(train),
                        "evaluation_interventions": len(evaluation),
                        "gene_count": len(indices),
                        "basis_fit_uses_heldout_interventions": False,
                    }
                )
            if device.type == "cuda":
                peak_gpu = max(peak_gpu, int(torch.cuda.max_memory_allocated()))
                torch.cuda.empty_cache()
            del matrix, singular, basis, eval6, eval14, coefficient6, coefficient14
        print(f"dimensionality fold {fold + 1}/5", flush=True)

    capture = capture_sum / denominator_sum
    summary_rows = [
        {
            "scope": "pooled_cross_fitted",
            "rank": rank,
            "signal_capture_fraction": float(value),
            "captured_crossplate_signal": float(capture_sum[index]),
            "total_crossplate_signal": float(denominator_sum),
        }
        for index, (rank, value) in enumerate(zip(tested, capture, strict=True))
    ]
    pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(rows)], ignore_index=True).to_csv(
        out / "lowrank_capture.csv", index=False
    )
    positive = np.maximum(component_signal_sum, 0)
    if positive.sum() > 0:
        probability = positive / positive.sum()
        participation = float(positive.sum() ** 2 / np.sum(positive * positive))
        entropy_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
    else:
        participation = float("nan")
        entropy_rank = float("nan")

    def threshold(value: float) -> int | str:
        for rank, fraction in zip(tested, capture, strict=True):
            if fraction >= value:
                return rank
        return f">{tested[-1]}"

    result = {
        "gene_count": len(indices),
        "ranks_tested": tested,
        "capture": {str(rank): float(value) for rank, value in zip(tested, capture, strict=True)},
        "k50": threshold(0.50),
        "k80": threshold(0.80),
        "k90": threshold(0.90),
        "participation_ratio_positive_aligned_components": participation,
        "entropy_rank_positive_aligned_components": entropy_rank,
        "interpretation_limit": "cross-fitted signal-capture rank, not full mathematical intrinsic dimension",
        "basis_training": "one plate and training interventions only",
        "evaluation": "paired opposite-plate held-out-intervention residual signal",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": peak_gpu,
        "seed": SEED,
    }
    _write_json(out / "dimensionality_summary.json", result)
    return result
