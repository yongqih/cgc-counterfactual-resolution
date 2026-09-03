"""Cross-fitted affine context-span scaling on the compact Tahoe tensor."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_tahoe_0c.analysis import load_core
from igc_virtual_cell.cgc_tahoe_0c.design import SUPPORT_SEQUENCES, SUPPORT_SIZES
from igc_virtual_cell.cgc_tahoe_0c.extraction import write_json


RCOND = 1e-10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202608204
RESIDUAL_NULL_SEED = 202608205
RESIDUAL_NULL_DRAWS = {2: 2_000, 4: 2_000, 8: 2_000, 16: 2_000, 24: 2_000, 32: 2_000, 40: 2_000, 49: 5_000}


def _center_interventions(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=2, keepdims=True)


def _batched_affine_svd(
    gram: np.ndarray, cross: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch, support, _ = gram.shape
    augmented = np.zeros((batch, support + 1, support + 1), dtype=np.float64)
    augmented[:, :support, :support] = gram
    augmented[:, :support, support] = 1.0
    augmented[:, support, :support] = 1.0
    rhs = np.empty((batch, support + 1), dtype=np.float64)
    rhs[:, :support] = cross
    rhs[:, support] = 1.0
    u, singular, vh = np.linalg.svd(augmented, full_matrices=False)
    projected = np.einsum("bij,bj->bi", u.transpose(0, 2, 1), rhs, optimize=True)
    cutoff = RCOND * singular[:, :1]
    projected = np.divide(
        projected,
        singular,
        out=np.zeros_like(projected),
        where=singular > cutoff,
    )
    solution = np.einsum(
        "bij,bj->bi", vh.transpose(0, 2, 1), projected, optimize=True
    )
    weights = solution[:, :support]

    gram_singular = np.linalg.svd(gram, compute_uv=False)
    gram_cutoff = RCOND * gram_singular[:, :1]
    rank = (gram_singular > gram_cutoff).sum(axis=1)
    smallest = gram_singular[:, -1]
    condition = np.divide(
        gram_singular[:, 0],
        smallest,
        out=np.full(batch, np.inf),
        where=smallest > gram_cutoff[:, 0],
    )
    return weights, rank, condition, gram_singular


def _residual_energy(gram: np.ndarray, target: int, supports: np.ndarray, weights: np.ndarray) -> np.ndarray:
    support_target = gram[supports, target]
    support_gram = gram[supports[:, :, None], supports[:, None, :]]
    return (
        gram[target, target]
        - 2 * np.einsum("bn,bn->b", weights, support_target)
        + np.einsum("bi,bij,bj->b", weights, support_gram, weights, optimize=True)
    )


def _weight_diagnostics(weights: np.ndarray) -> tuple[np.ndarray, ...]:
    absolute = np.abs(weights)
    total = absolute.sum(axis=1, keepdims=True)
    probabilities = np.divide(absolute, total, out=np.zeros_like(absolute), where=total > 0)
    entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0), axis=1
    )
    if weights.shape[1] > 1:
        entropy /= math.log(weights.shape[1])
    effective = np.divide(
        1.0,
        np.sum(probabilities * probabilities, axis=1),
        out=np.zeros(weights.shape[0]),
        where=np.sum(probabilities * probabilities, axis=1) > 0,
    )
    return (
        np.max(absolute, axis=1),
        np.mean(weights < 0, axis=1),
        entropy,
        effective,
    )


def _cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _support_orders(root: Path, contexts: list[str]) -> np.ndarray:
    frame = pd.read_csv(root / "results/cgc_tahoe_0c/support_sequences.csv")
    mapping = {value: index for index, value in enumerate(contexts)}
    expected_rows = len(contexts) * SUPPORT_SEQUENCES * (len(contexts) - 1)
    if len(frame) != expected_rows:
        raise RuntimeError("Frozen support manifest row count changed")
    values = frame["support_context"].map(mapping).to_numpy()
    orders = values.reshape(len(contexts), SUPPORT_SEQUENCES, len(contexts) - 1)
    for heldout in range(len(contexts)):
        if np.any(orders[heldout] == heldout):
            raise RuntimeError("Held-out context leaked into support sequence")
    return orders


def _folds(root: Path, interventions: list[str]) -> np.ndarray:
    frame = pd.read_csv(root / "results/cgc_tahoe_0c/intervention_folds.csv")
    mapping = frame.set_index("intervention_id")["fold"].to_dict()
    values = np.array([mapping[value] for value in interventions], dtype=np.int8)
    if set(values.tolist()) != set(range(5)):
        raise RuntimeError("Intervention folds changed")
    return values


def _support_hash(contexts: list[str], row: np.ndarray) -> str:
    return hashlib.sha256("|".join(contexts[index] for index in row).encode("utf-8")).hexdigest()


def _metric_contributions(
    target6: np.ndarray,
    target14: np.ndarray,
    prediction6: np.ndarray,
    prediction14: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual6 = target6 - prediction6
    residual14 = target14 - prediction14
    cross = np.mean(residual6 * residual14, axis=2)
    observed = (np.mean(residual6 * residual6, axis=2) + np.mean(residual14 * residual14, axis=2)) / 2
    return cross, observed


def run_scaling(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    truth = json.loads((result_dir / "truth_gate.json").read_text(encoding="utf-8"))
    if truth["truth_label"] != "TAHOE_REPLICATE_STABLE_CONTEXT_OPERATOR_CONFIRMED":
        raise RuntimeError("Truth gate did not authorize scaling")
    design = json.loads((result_dir / "design_manifest.json").read_text(encoding="utf-8"))
    if not design["frozen_before_coordinate_fitting"]:
        raise RuntimeError("Support design was not frozen")

    delta, axes = load_core(root)
    contexts = axes["contexts"]
    interventions = axes["interventions"]
    context_count = len(contexts)
    intervention_count = len(interventions)
    gene_count = len(axes["genes"])
    support_orders = _support_orders(root, contexts)
    intervention_folds = _folds(root, interventions)
    size_count = len(SUPPORT_SIZES)

    utility_sum = np.zeros((size_count, context_count, intervention_count, 4), dtype=np.float64)
    nearest_sum = np.zeros_like(utility_sum)
    utility_count = np.zeros((size_count, context_count, intervention_count), dtype=np.int32)
    null_cross: dict[tuple[int, int], np.ndarray] = {}

    coordinate_path = root / "data/tahoe100m_plate6_14_core/coordinate_staging.npy"
    coordinate_shape = (2, 5, context_count, size_count, SUPPORT_SEQUENCES, 49)
    coordinates = np.lib.format.open_memmap(
        coordinate_path, mode="w+", dtype=np.float32, shape=coordinate_shape
    )
    coordinates[:] = np.nan
    ranks = np.zeros(coordinate_shape[:-1], dtype=np.int8)
    conditions = np.full(coordinate_shape[:-1], np.nan, dtype=np.float32)

    fit_path = result_dir / "affine_fit_summary.csv"
    fieldnames = [
        "heldout_context",
        "support_size",
        "sequence",
        "evaluation_fold",
        "fit_plate",
        "matrix_rank",
        "condition_number",
        "max_abs_weight",
        "negative_weight_fraction",
        "weight_entropy",
        "effective_weight_count",
        "v_base",
        "v_residual",
        "q_ratio_secondary",
        "raw_r2",
        "nearest_context",
        "nearest_v_residual",
        "nearest_raw_r2",
    ]
    names = (
        pd.read_csv(root / "results/cgc_tahoe_0b/cell_line_manifest.csv")
        .set_index("cell_line_id")["cell_name"]
        .to_dict()
    )
    fit_rows = 0
    with fit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold in range(5):
            evaluation_indices = np.flatnonzero(intervention_folds == fold)
            training_indices = np.flatnonzero(intervention_folds != fold)
            q_training = _center_interventions(delta[:, :, training_indices, :])
            q_evaluation = _center_interventions(delta[:, :, evaluation_indices, :])
            training_grams = np.einsum(
                "rcpg,rdpg->rcd", q_training, q_training, optimize=True
            )
            print(f"scaling fold {fold + 1}/5", flush=True)
            for heldout in range(context_count):
                for size_index, support_size in enumerate(SUPPORT_SIZES):
                    sequence_count = 1 if support_size == 49 else SUPPORT_SEQUENCES
                    supports = support_orders[heldout, :sequence_count, :support_size]
                    support_hashes = [_support_hash(contexts, row) for row in supports]
                    support_q6 = q_evaluation[0][supports]
                    support_q14 = q_evaluation[1][supports]
                    shared6 = support_q6.mean(axis=1)
                    shared14 = support_q14.mean(axis=1)
                    target6 = np.broadcast_to(q_evaluation[0, heldout], shared6.shape)
                    target14 = np.broadcast_to(q_evaluation[1, heldout], shared14.shape)
                    base_cross, base_observed = _metric_contributions(
                        target6, target14, shared6, shared14
                    )

                    for direction in range(2):
                        full_gram = training_grams[direction]
                        support_gram = full_gram[
                            supports[:, :, None], supports[:, None, :]
                        ]
                        support_target = full_gram[supports, heldout]
                        weights, rank, condition_number, _ = _batched_affine_svd(
                            support_gram, support_target
                        )
                        coordinates[
                            direction,
                            fold,
                            heldout,
                            size_index,
                            :sequence_count,
                            :support_size,
                        ] = weights.astype(np.float32)
                        ranks[
                            direction, fold, heldout, size_index, :sequence_count
                        ] = rank.astype(np.int8)
                        conditions[
                            direction, fold, heldout, size_index, :sequence_count
                        ] = condition_number.astype(np.float32)

                        prediction6 = np.einsum(
                            "bn,bneg->beg", weights, support_q6, optimize=True
                        )
                        prediction14 = np.einsum(
                            "bn,bneg->beg", weights, support_q14, optimize=True
                        )
                        residual_cross, residual_observed = _metric_contributions(
                            target6, target14, prediction6, prediction14
                        )

                        diagonal = np.diag(full_gram)
                        cosine = support_target / np.sqrt(
                            np.maximum(diagonal[supports] * diagonal[heldout], 0)
                        )
                        nearest_position = np.nanargmax(cosine, axis=1)
                        nearest = supports[np.arange(sequence_count), nearest_position]
                        nearest6 = q_evaluation[0][nearest]
                        nearest14 = q_evaluation[1][nearest]
                        nearest_cross, nearest_observed = _metric_contributions(
                            target6, target14, nearest6, nearest14
                        )

                        utility_sum[
                            size_index, heldout, evaluation_indices, 0
                        ] += base_cross.sum(axis=0)
                        utility_sum[
                            size_index, heldout, evaluation_indices, 1
                        ] += residual_cross.sum(axis=0)
                        utility_sum[
                            size_index, heldout, evaluation_indices, 2
                        ] += base_observed.sum(axis=0)
                        utility_sum[
                            size_index, heldout, evaluation_indices, 3
                        ] += residual_observed.sum(axis=0)
                        nearest_sum[
                            size_index, heldout, evaluation_indices, 0
                        ] += base_cross.sum(axis=0)
                        nearest_sum[
                            size_index, heldout, evaluation_indices, 1
                        ] += nearest_cross.sum(axis=0)
                        nearest_sum[
                            size_index, heldout, evaluation_indices, 2
                        ] += base_observed.sum(axis=0)
                        nearest_sum[
                            size_index, heldout, evaluation_indices, 3
                        ] += nearest_observed.sum(axis=0)
                        utility_count[
                            size_index, heldout, evaluation_indices
                        ] += sequence_count

                        residual6 = target6 - prediction6
                        residual14 = target14 - prediction14
                        key = (size_index, fold)
                        if key not in null_cross:
                            null_cross[key] = np.zeros(
                                (len(evaluation_indices), len(evaluation_indices)),
                                dtype=np.float64,
                            )
                        null_cross[key] += np.einsum(
                            "beg,bfg->ef", residual6, residual14, optimize=True
                        )

                        max_abs, negative_fraction, entropy, effective = _weight_diagnostics(weights)
                        vbase_fit = base_cross.mean(axis=1)
                        vres_fit = residual_cross.mean(axis=1)
                        ebase_fit = base_observed.mean(axis=1)
                        eres_fit = residual_observed.mean(axis=1)
                        nearest_vres = nearest_cross.mean(axis=1)
                        nearest_eres = nearest_observed.mean(axis=1)
                        for sequence in range(sequence_count):
                            writer.writerow(
                                {
                                    "heldout_context": contexts[heldout],
                                    "support_size": support_size,
                                    "sequence": sequence,
                                    "evaluation_fold": fold,
                                    "fit_plate": ["plate6", "plate14"][direction],
                                    "matrix_rank": int(rank[sequence]),
                                    "condition_number": float(condition_number[sequence]),
                                    "max_abs_weight": float(max_abs[sequence]),
                                    "negative_weight_fraction": float(
                                        negative_fraction[sequence]
                                    ),
                                    "weight_entropy": float(entropy[sequence]),
                                    "effective_weight_count": float(effective[sequence]),
                                    "v_base": float(vbase_fit[sequence]),
                                    "v_residual": float(vres_fit[sequence]),
                                    "q_ratio_secondary": float(
                                        vres_fit[sequence] / vbase_fit[sequence]
                                    )
                                    if vbase_fit[sequence] != 0
                                    else np.nan,
                                    "raw_r2": float(
                                        1 - eres_fit[sequence] / ebase_fit[sequence]
                                    )
                                    if ebase_fit[sequence] != 0
                                    else np.nan,
                                    "nearest_context": contexts[int(nearest[sequence])],
                                    "nearest_v_residual": float(
                                        nearest_vres[sequence]
                                    ),
                                    "nearest_raw_r2": float(
                                        1 - nearest_eres[sequence] / ebase_fit[sequence]
                                    )
                                    if ebase_fit[sequence] != 0
                                    else np.nan,
                                }
                            )
                            fit_rows += 1
                    del support_q6, support_q14, shared6, shared14
                if (heldout + 1) % 10 == 0:
                    coordinates.flush()
                    print(
                        f"fold {fold + 1}: heldout {heldout + 1}/{context_count}",
                        flush=True,
                    )

    utility_average = np.divide(
        utility_sum,
        utility_count[..., None],
        out=np.full_like(utility_sum, np.nan),
        where=utility_count[..., None] > 0,
    )
    nearest_average = np.divide(
        nearest_sum,
        utility_count[..., None],
        out=np.full_like(nearest_sum, np.nan),
        where=utility_count[..., None] > 0,
    )
    utility_rows = []
    for size_index, support_size in enumerate(SUPPORT_SIZES):
        for context_index, context in enumerate(contexts):
            for intervention_index, intervention in enumerate(interventions):
                values = utility_average[size_index, context_index, intervention_index]
                utility_rows.append(
                    {
                        "support_size": support_size,
                        "context_id": context,
                        "intervention_id": intervention,
                        "v_base": values[0],
                        "v_residual": values[1],
                        "e_observed": values[2],
                        "e_residual_observed": values[3],
                        "fit_replicates": int(
                            utility_count[size_index, context_index, intervention_index]
                        ),
                    }
                )
    frozen_utility = pd.DataFrame(utility_rows)
    frozen_utility.to_parquet(result_dir / "frozen_oof_utility.parquet", index=False)

    curve_rows = []
    raw_rows = []
    recoverable_rows = []
    budget_rows = []
    context_rows = []
    nearest_rows = []
    for size_index, support_size in enumerate(SUPPORT_SIZES):
        values = utility_average[size_index]
        vbase = float(np.nansum(values[..., 0]))
        vres = float(np.nansum(values[..., 1]))
        eobs = float(np.nansum(values[..., 2]))
        eres = float(np.nansum(values[..., 3]))
        q = vres / vbase
        raw_r2 = 1 - eres / eobs
        context_q = np.nansum(values[..., 1], axis=1) / np.nansum(values[..., 0], axis=1)
        curve_rows.append(
            {
                "support_size": support_size,
                "q_pooled": q,
                "median_context_q": float(np.nanmedian(context_q)),
                "context_q_iqr_lower": float(np.nanquantile(context_q, 0.25)),
                "context_q_iqr_upper": float(np.nanquantile(context_q, 0.75)),
                "pooled_v_base": vbase / values[..., 0].size,
                "pooled_v_residual": vres / values[..., 1].size,
            }
        )
        recoverable_rows.append(
            {"support_size": support_size, "recoverable_fraction": 1 - q}
        )
        raw_rows.append({"support_size": support_size, "raw_r2_pooled": raw_r2})
        captured = vbase - vres
        measurement = eobs - vbase
        budget_rows.append(
            {
                "support_size": support_size,
                "e_observed": eobs / values[..., 2].size,
                "v_base": vbase / values[..., 0].size,
                "v_captured": captured / values[..., 0].size,
                "v_residual": vres / values[..., 1].size,
                "v_measurement_unresolved": measurement / values[..., 2].size,
                "captured_fraction_of_observed": captured / eobs,
                "residual_fraction_of_observed": vres / eobs,
                "measurement_fraction_of_observed": measurement / eobs,
                "fraction_sum": (captured + vres + measurement) / eobs,
            }
        )
        nearest_values = nearest_average[size_index]
        nearest_vbase = float(np.nansum(nearest_values[..., 0]))
        nearest_vres = float(np.nansum(nearest_values[..., 1]))
        nearest_eobs = float(np.nansum(nearest_values[..., 2]))
        nearest_eres = float(np.nansum(nearest_values[..., 3]))
        nearest_rows.append(
            {
                "support_size": support_size,
                "affine_q": q,
                "nearest_context_q": nearest_vres / nearest_vbase,
                "affine_raw_r2": raw_r2,
                "nearest_context_raw_r2": 1 - nearest_eres / nearest_eobs,
                "affine_minus_nearest_recoverable_fraction": (1 - q)
                - (1 - nearest_vres / nearest_vbase),
            }
        )
        for context_index, context in enumerate(contexts):
            context_values = values[context_index]
            context_vbase = float(np.nansum(context_values[:, 0]))
            context_vres = float(np.nansum(context_values[:, 1]))
            context_eobs = float(np.nansum(context_values[:, 2]))
            context_eres = float(np.nansum(context_values[:, 3]))
            context_rows.append(
                {
                    "context_id": context,
                    "context_name": names.get(context, context),
                    "support_size": support_size,
                    "q_context": context_vres / context_vbase,
                    "v_base": context_vbase / intervention_count,
                    "v_residual": context_vres / intervention_count,
                    "raw_r2": 1 - context_eres / context_eobs,
                }
            )
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(result_dir / "support_q_curve.csv", index=False)
    pd.DataFrame(recoverable_rows).to_csv(
        result_dir / "support_recoverable_fraction.csv", index=False
    )
    pd.DataFrame(raw_rows).to_csv(result_dir / "support_raw_r2.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(result_dir / "measurement_budget.csv", index=False)
    context_frame = pd.DataFrame(context_rows)
    context_frame.to_csv(result_dir / "context_heterogeneity.csv", index=False)
    pd.DataFrame(nearest_rows).to_csv(
        result_dir / "nearest_context_baseline.csv", index=False
    )

    maximal = context_frame[context_frame["support_size"] == 49].copy()
    maximal = maximal.rename(columns={"q_context": "q49"})
    maximal["positive_residual_signal"] = maximal["v_residual"] > 0
    maximal.to_csv(result_dir / "maximal_support_summary.csv", index=False)

    stability_rows = []
    for heldout, context in enumerate(contexts):
        for size_index, support_size in enumerate(SUPPORT_SIZES):
            sequence_count = 1 if support_size == 49 else SUPPORT_SEQUENCES
            for sequence in range(sequence_count):
                cross_plate = []
                for fold in range(5):
                    first = coordinates[0, fold, heldout, size_index, sequence, :support_size]
                    second = coordinates[1, fold, heldout, size_index, sequence, :support_size]
                    cross_plate.append(float(_cosine_rows(first[None], second[None])[0]))
                fold_stability = []
                for direction in range(2):
                    for first_fold, second_fold in combinations(range(5), 2):
                        first = coordinates[
                            direction, first_fold, heldout, size_index, sequence, :support_size
                        ]
                        second = coordinates[
                            direction, second_fold, heldout, size_index, sequence, :support_size
                        ]
                        fold_stability.append(
                            float(_cosine_rows(first[None], second[None])[0])
                        )
                all_weights = coordinates[
                    :, :, heldout, size_index, sequence, :support_size
                ].reshape(10, support_size)
                max_abs, negative, entropy, effective = _weight_diagnostics(all_weights)
                stability_rows.append(
                    {
                        "context_id": context,
                        "support_size": support_size,
                        "sequence": sequence,
                        "cross_plate_coordinate_cosine": float(np.nanmean(cross_plate)),
                        "fold_coordinate_cosine": float(np.nanmean(fold_stability)),
                        "median_effective_rank": float(
                            np.median(ranks[:, :, heldout, size_index, sequence])
                        ),
                        "median_condition_number": float(
                            np.nanmedian(conditions[:, :, heldout, size_index, sequence])
                        ),
                        "median_max_abs_weight": float(np.median(max_abs)),
                        "mean_negative_weight_fraction": float(np.mean(negative)),
                        "mean_weight_entropy": float(np.mean(entropy)),
                        "median_effective_weight_count": float(np.median(effective)),
                    }
                )
    pd.DataFrame(stability_rows).to_csv(
        result_dir / "coordinate_stability.csv", index=False
    )

    null_rng = np.random.default_rng(RESIDUAL_NULL_SEED)
    null_rows = []
    null_summary: dict[int, dict[str, float]] = {}
    for size_index, support_size in enumerate(SUPPORT_SIZES):
        draws = RESIDUAL_NULL_DRAWS[support_size]
        values = np.empty(draws, dtype=np.float64)
        sequence_count = 1 if support_size == 49 else SUPPORT_SEQUENCES
        denominator = context_count * sequence_count * 2 * intervention_count * gene_count
        for draw in range(draws):
            total = 0.0
            for fold in range(5):
                matrix = null_cross[(size_index, fold)]
                row = np.arange(matrix.shape[0])
                permutation = null_rng.permutation(matrix.shape[0])
                total += matrix[row, permutation].sum()
            values[draw] = total / denominator
        observed = float(curve.loc[curve["support_size"] == support_size, "pooled_v_residual"].iloc[0])
        q95 = float(np.quantile(values, 0.95))
        empirical_p = float((1 + np.count_nonzero(values >= observed)) / (1 + draws))
        null_summary[support_size] = {
            "observed": observed,
            "q95": q95,
            "empirical_p": empirical_p,
        }
        null_rows.extend(
            {
                "support_size": support_size,
                "permutation": draw,
                "seed": RESIDUAL_NULL_SEED,
                "residual_energy": value,
            }
            for draw, value in enumerate(values)
        )
    pd.DataFrame(null_rows).to_csv(result_dir / "residual_shuffle_null.csv", index=False)

    bootstrap_q = np.empty((BOOTSTRAP_DRAWS, size_count), dtype=np.float64)
    bootstrap_max_residual = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAP_DRAWS, 100):
        stop = min(BOOTSTRAP_DRAWS, start + 100)
        batch = stop - start
        sampled_contexts = rng.integers(0, context_count, size=(batch, context_count))
        sampled_interventions = rng.integers(
            0, intervention_count, size=(batch, context_count, intervention_count)
        )
        expanded_contexts = np.broadcast_to(
            sampled_contexts[:, :, None], sampled_interventions.shape
        )
        for size_index in range(size_count):
            base = utility_average[size_index, ..., 0][
                expanded_contexts, sampled_interventions
            ].sum(axis=(1, 2))
            residual = utility_average[size_index, ..., 1][
                expanded_contexts, sampled_interventions
            ].sum(axis=(1, 2))
            bootstrap_q[start:stop, size_index] = residual / base
            if SUPPORT_SIZES[size_index] == 49:
                bootstrap_max_residual[start:stop] = residual / (
                    context_count * intervention_count
                )
    x = np.log2(np.array(SUPPORT_SIZES, dtype=np.float64))
    centered_x = x - x.mean()
    bootstrap_beta = np.sum(
        bootstrap_q * centered_x[None, :], axis=1
    ) / np.sum(centered_x * centered_x)
    bootstrap_difference = bootstrap_q[:, 0] - bootstrap_q[:, -1]
    bootstrap_rows = []
    for size_index, support_size in enumerate(SUPPORT_SIZES):
        low, high = np.quantile(bootstrap_q[:, size_index], [0.025, 0.975])
        estimate = float(curve.loc[curve["support_size"] == support_size, "q_pooled"].iloc[0])
        bootstrap_rows.append(
            {
                "metric": f"q{support_size}",
                "estimate": estimate,
                "lower_95": float(low),
                "upper_95": float(high),
                "draws": BOOTSTRAP_DRAWS,
            }
        )
    q_values = curve["q_pooled"].to_numpy()
    beta = float(np.sum(q_values * centered_x) / np.sum(centered_x * centered_x))
    beta_low, beta_high = np.quantile(bootstrap_beta, [0.025, 0.975])
    difference = float(q_values[0] - q_values[-1])
    difference_low, difference_high = np.quantile(
        bootstrap_difference, [0.025, 0.975]
    )
    max_low, max_high = np.quantile(bootstrap_max_residual, [0.025, 0.975])
    max_estimate = float(curve.loc[curve["support_size"] == 49, "pooled_v_residual"].iloc[0])
    bootstrap_rows.extend(
        [
            {
                "metric": "q2_minus_q49",
                "estimate": difference,
                "lower_95": float(difference_low),
                "upper_95": float(difference_high),
                "draws": BOOTSTRAP_DRAWS,
            },
            {
                "metric": "trend_beta1",
                "estimate": beta,
                "lower_95": float(beta_low),
                "upper_95": float(beta_high),
                "draws": BOOTSTRAP_DRAWS,
            },
            {
                "metric": "max_support_residual_energy",
                "estimate": max_estimate,
                "lower_95": float(max_low),
                "upper_95": float(max_high),
                "draws": BOOTSTRAP_DRAWS,
            },
        ]
    )
    pd.DataFrame(bootstrap_rows).to_csv(result_dir / "bootstrap_summary.csv", index=False)

    q2_context = context_frame[context_frame["support_size"] == 2].set_index("context_id")[
        "q_context"
    ]
    q49_context = context_frame[context_frame["support_size"] == 49].set_index("context_id")[
        "q_context"
    ]
    contexts_declining = int((q49_context < q2_context).sum())
    trend_gates = {
        "beta_negative": bool(beta < 0),
        "beta_upper_ci_negative": bool(beta_high < 0),
        "q49_below_q2": bool(difference > 0),
        "paired_difference_ci_positive": bool(difference_low > 0),
        "at_least_35_contexts_decline": bool(contexts_declining >= 35),
    }
    trend_label = (
        "REPRODUCIBLE_RESIDUAL_DECLINES_WITH_CONTEXT_SUPPORT"
        if all(trend_gates.values())
        else "CONTEXT_SUPPORT_DECLINE_NOT_ESTABLISHED"
    )
    max_null = null_summary[49]
    positive_max_contexts = int(maximal["positive_residual_signal"].sum())
    novelty_gates = {
        "pooled_residual_positive": bool(max_estimate > 0),
        "bootstrap_lower_positive": bool(max_low > 0),
        "residual_above_shuffle_q95": bool(max_estimate > max_null["q95"]),
        "empirical_p_below_0_05": bool(max_null["empirical_p"] < 0.05),
        "at_least_30_positive_contexts": bool(positive_max_contexts >= 30),
    }
    if all(novelty_gates.values()):
        maximal_label = "NOVEL_REPRODUCIBLE_OPERATOR_PERSISTS_AT_MAX_SUPPORT"
    elif max_estimate <= max_null["q95"] or max_null["empirical_p"] >= 0.05:
        maximal_label = "HELDOUT_OPERATOR_APPROACHES_MEASUREMENT_FLOOR"
    else:
        maximal_label = "MAX_SUPPORT_RESIDUAL_UNRESOLVED"
    q49 = q_values[-1]
    if q49 <= 0.25:
        coverage_label = "CONTEXT_OPERATOR_SUPPORT_COVERAGE_STRONG"
    elif q49 < 0.75:
        coverage_label = "CONTEXT_OPERATOR_SUPPORT_COVERAGE_PARTIAL"
    else:
        coverage_label = "CONTEXT_OPERATOR_SUPPORT_COVERAGE_WEAK"

    pd.DataFrame(
        [
            {
                "beta0": float(q_values.mean() - beta * x.mean()),
                "beta1": beta,
                "beta1_lower_95": float(beta_low),
                "beta1_upper_95": float(beta_high),
                "q2_minus_q49": difference,
                "q2_minus_q49_lower_95": float(difference_low),
                "q2_minus_q49_upper_95": float(difference_high),
                "contexts_q49_below_q2": contexts_declining,
                "total_contexts": context_count,
                "trend_label": trend_label,
            }
        ]
    ).to_csv(result_dir / "support_trend.csv", index=False)

    verdict = {
        "phase": "CGC-SUPPORT-0C",
        "status": "complete_pending_integrity",
        "truth_label": truth["truth_label"],
        "trend_label": trend_label,
        "maximal_support_label": maximal_label,
        "coverage_label": coverage_label,
        "truth_gates": truth["gates"],
        "trend_gates": trend_gates,
        "novelty_gates": novelty_gates,
        "q_values": {
            str(row.support_size): float(row.q_pooled)
            for row in curve.itertuples(index=False)
        },
        "q2_minus_q49": difference,
        "trend_beta1": beta,
        "trend_beta1_lower_95": float(beta_low),
        "trend_beta1_upper_95": float(beta_high),
        "contexts_q49_below_q2": contexts_declining,
        "max_support_residual_energy": max_estimate,
        "max_support_residual_lower_95": float(max_low),
        "max_support_residual_upper_95": float(max_high),
        "max_support_null_q95": max_null["q95"],
        "max_support_empirical_p": max_null["empirical_p"],
        "positive_max_support_contexts": positive_max_contexts,
        "fit_rows": fit_rows,
        "models_trained": 0,
        "virtual_cell_checkpoint_calls": 0,
    }
    write_json(result_dir / "verdict.json", verdict)
    write_json(
        result_dir / "scaling_run_stats.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fit_rows": fit_rows,
            "support_sizes": list(SUPPORT_SIZES),
            "support_sequences": SUPPORT_SEQUENCES,
            "intervention_folds": 5,
            "plate_directions": 2,
            "rcond": RCOND,
            "coordinate_solver": "batched SVD minimum-norm affine KKT",
            "gpu_used": False,
            "models_trained": 0,
        },
    )
    del all_weights
    del first
    del second
    del coordinates
    gc.collect()
    coordinate_path.unlink()
    return verdict
