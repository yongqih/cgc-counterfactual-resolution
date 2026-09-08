from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_entrywise.estimators import (
    SelectedParameters,
    fit_batches,
    select_parameters,
    sentinel_matrix,
)
from igc_virtual_cell.cgc_entrywise.foundation import load_foundation
from igc_virtual_cell.cgc_entrywise.metrics import (
    METRIC_NAMES,
    excess_geometry,
    matched_context_gram,
    matched_residual_cross,
    metrics_from_raw,
    offset_residual_cross,
    offset_stats,
    weighted_same_stats,
)


M_VALUES = (1, 2, 4, 8, 16, 24, 32, 40, 49)
K_VALUES = (0, 1, 2, 4, 8, 16, 32, 64, 80, 92)
MODELS = ("M0_SUPPORT_MEAN", "M1_SENTINEL_OFFSET", "M2_AFFINE_RIDGE", "M3_LOWRANK_CONTEXT")
NULLS = ("INTERVENTION_IDENTITY", "CONTEXT_IDENTITY", "SENTINEL_IDENTITY", "REFERENCE_CONTEXT_IDENTITY")
TARGETS = ("FULL_RESPONSE", "CONTEXT_SPECIFIC_EXCESS")
PLATES = ("plate6", "plate14")
GENE_COUNT = 25_695


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _load_split(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )


def _axes(split: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {value: index for index, value in enumerate(split["contexts"])},
        {value: index for index, value in enumerate(split["interventions"])},
    )


def _orders(
    split: dict[str, Any],
    target: int,
    root: Path | None = None,
    support_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    context_index, intervention_index = _axes(split)
    context = split["contexts"][target]
    if support_mode == "random":
        support = np.asarray(
            [
                [context_index[value] for value in split["context_support_orders"][context][str(sequence)]["order"]]
                for sequence in split["support_sequence_ids"]
            ],
            dtype=np.int64,
        )
    elif support_mode == "rna":
        if root is None:
            raise ValueError("root is required for RNA support")
        rna = json.loads(
            (root / "results/cgc_entrywise_compression/ENTRYWISE_RNA_SUPPORT_ORDERS.json").read_text(encoding="utf-8")
        )
        order = [context_index[value] for value in rna["orders"][context]]
        support = np.broadcast_to(np.asarray(order, dtype=np.int64), (8, 49)).copy()
    else:
        raise ValueError(f"Unknown support mode: {support_mode}")
    sentinel = np.asarray(
        [
            [intervention_index[value] for value in split["sentinel_orders"][context][str(sequence)]["full_order"]]
            for sequence in split["support_sequence_ids"]
        ],
        dtype=np.int64,
    )
    return support, sentinel


def _null_maps(split: dict[str, Any], target: int, sequence: int) -> tuple[int, np.ndarray, np.ndarray]:
    context_index, intervention_index = _axes(split)
    context = split["contexts"][target]
    context_null = context_index[split["context_identity_null_maps"][str(sequence)][context]]
    intervention_mapping = split["null_maps"][context][str(sequence)]["intervention_identity"]
    sentinel_mapping = split["null_maps"][context][str(sequence)]["sentinel_identity"]
    intervention_null = np.asarray(
        [intervention_index[intervention_mapping[value]] for value in split["interventions"]],
        dtype=np.int64,
    )
    sentinel_null = np.asarray(
        [intervention_index[sentinel_mapping[value]] for value in split["interventions"]],
        dtype=np.int64,
    )
    return context_null, intervention_null, sentinel_null


def _intervention_null_residual(
    cross4: np.ndarray,
    cross_by_intervention: np.ndarray,
    target: int,
    sources: np.ndarray,
    weights6: np.ndarray,
    weights14: np.ndarray,
    mapping: np.ndarray,
    truth_excess: np.ndarray,
) -> np.ndarray:
    uniform = np.full(len(sources), 1.0 / len(sources), dtype=np.float64)
    delta6 = weights6 - uniform
    delta14 = weights14 - uniform
    result = np.empty(93, dtype=np.float64)
    for intervention in range(93):
        mapped = int(mapping[intervention])
        d6_to_pred14 = np.asarray(
            [
                cross4[target, intervention, source, mapped]
                - np.mean(cross4[sources, intervention, source, mapped], dtype=np.float64)
                for source in sources
            ],
            dtype=np.float64,
        )
        pred6_to_d14 = np.asarray(
            [
                cross4[source, mapped, target, intervention]
                - np.mean(cross4[source, mapped, sources, intervention], dtype=np.float64)
                for source in sources
            ],
            dtype=np.float64,
        )
        prediction_cross = float(
            delta6[intervention]
            @ cross_by_intervention[mapped][np.ix_(sources, sources)]
            @ delta14[intervention]
        )
        result[intervention] = (
            truth_excess[intervention]
            - float(d6_to_pred14 @ delta14[intervention])
            - float(delta6[intervention] @ pred6_to_d14)
            + prediction_cross
        )
    return result


def _accumulate_metrics(
    metric_sum: np.ndarray,
    metric_count: np.ndarray,
    model_index: int,
    m_index: int,
    k_index: int,
    target_index: int,
    plate_index: int,
    values: np.ndarray,
) -> None:
    finite = np.isfinite(values)
    metric_sum[model_index, m_index, k_index, target_index, plate_index] += np.nansum(
        values, axis=0
    )
    metric_count[model_index, m_index, k_index, target_index, plate_index] += finite.sum(axis=0)


def _target_cache_path(root: Path, target: int, support_mode: str = "random") -> Path:
    prefix = "target" if support_mode == "random" else f"target_{support_mode}"
    return root / "results/cgc_entrywise_compression/_cache" / f"{prefix}_{target:02d}.npz"


def _parameter_row(
    target: int,
    plate: int,
    m: int,
    k: int,
    selected: SelectedParameters,
    support_sequence: int = -1,
) -> dict[str, Any]:
    return {
        "target_context_index": target,
        "plate": PLATES[plate],
        "m": m,
        "k": k,
        "support_sequence": support_sequence,
        "selection_scope": "identical_reference_support_pool" if support_sequence == -1 else "episode_reference_support_only",
        "alpha": selected.alpha,
        "ridge_lambda": selected.ridge,
        "rank": selected.rank,
        "alpha_cv_loss": selected.alpha_loss,
        "ridge_cv_loss": selected.ridge_loss,
        "rank_cv_loss": selected.rank_loss,
        "pseudo_target_cases": selected.pseudo_targets,
        "validation_interventions": selected.validation_cases,
        "target_outcome_used": False,
    }


def run_target(
    root: Path,
    target: int,
    overwrite: bool = False,
    support_mode: str = "random",
) -> Path:
    root = root.resolve()
    cache_path = _target_cache_path(root, target, support_mode)
    if cache_path.exists() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as existing:
            if "tuning_protocol" not in existing or str(existing["tuning_protocol"]) != "EPISODE_REFERENCE_SUPPORT_ONLY_V2":
                raise RuntimeError("Stale pooled-support cache; archive then rerun with overwrite=True")
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    split = _load_split(root)
    support_orders, sentinel_orders = _orders(split, target, root, support_mode)
    folds = np.asarray(
        [split["inner_intervention_folds"][value] for value in split["interventions"]],
        dtype=np.int64,
    )
    same6_raw, same14_raw, cross_raw, sums, _ = load_foundation(root)
    same_raw = (same6_raw, same14_raw)
    same4 = tuple(value.reshape(50, 93, 50, 93) for value in same_raw)
    cross4 = cross_raw.reshape(50, 93, 50, 93)
    matched_same = tuple(matched_context_gram(value) for value in same4)
    matched_cross = matched_context_gram(cross4)

    vtruth_sum = np.zeros((len(M_VALUES), 93), dtype=np.float64)
    vtruth_count = np.zeros((len(M_VALUES), 93), dtype=np.int32)
    vafter_sum = np.zeros((len(MODELS), len(M_VALUES), len(K_VALUES), 93), dtype=np.float64)
    vafter_count = np.zeros((len(M_VALUES), len(K_VALUES), 93), dtype=np.int32)
    null_sum = np.zeros((len(NULLS), len(M_VALUES), len(K_VALUES), 93), dtype=np.float64)
    null_count = np.zeros((len(M_VALUES), len(K_VALUES), 93), dtype=np.int32)
    metric_sum = np.zeros((len(MODELS), len(M_VALUES), len(K_VALUES), 2, 2, len(METRIC_NAMES)), dtype=np.float64)
    metric_count = np.zeros_like(metric_sum, dtype=np.int64)
    parameter_rows: list[dict[str, Any]] = []
    allbut_metrics = np.full((len(MODELS), 93, 2, 2, len(METRIC_NAMES)), np.nan, dtype=np.float64)
    allbut_vtruth = np.full(93, np.nan, dtype=np.float64)
    allbut_vafter = np.full((len(MODELS), 93), np.nan, dtype=np.float64)

    selected: dict[tuple[int, int, int, int], SelectedParameters] = {}
    for plate in range(2):
        geometry_cache: dict[tuple[int, tuple[int, ...]], object] = {}
        for m_index, m in enumerate(M_VALUES):
            for k_index, k in enumerate(K_VALUES):
                identical_support = all(
                    set(map(int, row[:m])) == set(map(int, support_orders[0, :m]))
                    for row in support_orders
                )
                sequences = (-1,) if identical_support or m == 1 or k == 0 else range(8)
                for sequence in sequences:
                    value = select_parameters(
                        matched_same[plate],
                        same4[plate],
                        np.asarray(sums[plate], dtype=np.float64),
                        target, m, k, support_orders, sentinel_orders, folds,
                        geometry_cache,
                        support_sequence=None if sequence == -1 else sequence,
                    )
                    for destination in range(8) if sequence == -1 else (sequence,):
                        selected[(plate, m_index, k_index, destination)] = value
                    parameter_rows.append(_parameter_row(target, plate, m, k, value, sequence))
            print(f"entrywise tune target {target + 1}/50 {PLATES[plate]} m={m}", flush=True)

    full_truth_cross = np.asarray(
        [cross4[target, intervention, target, intervention] for intervention in range(93)],
        dtype=np.float64,
    )
    for m_index, m in enumerate(M_VALUES):
        for sequence in range(8):
            sources = np.asarray(support_orders[sequence, :m], dtype=np.int64)
            same_geometry = (
                excess_geometry(same4[0], np.asarray(sums[0]), target, sources),
                excess_geometry(same4[1], np.asarray(sums[1]), target, sources),
            )
            cross_geometry = excess_geometry(cross4, np.asarray(sums[0]), target, sources)
            truth_excess = np.diag(cross_geometry.gram)
            vtruth_sum[m_index] += truth_excess / GENE_COUNT
            vtruth_count[m_index] += 1
            for k_index, k in enumerate(K_VALUES):
                sequence_count = 1 if (m == 49 and k == 92) else 8
                if sequence >= sequence_count:
                    continue
                sentinels = sentinel_matrix(sentinel_orders[sequence], k)
                context_null, intervention_null, sentinel_null = _null_maps(split, target, sequence)
                batches = []
                for plate in range(2):
                    params = selected[(plate, m_index, k_index, sequence)]
                    batches.append(
                        fit_batches(
                            matched_same[plate],
                            same4[plate],
                            target,
                            sources,
                            sentinel_orders[sequence],
                            k,
                            params.ridge,
                            params.rank,
                            context_null,
                            sentinel_null,
                        )
                    )

                weights_by_model = (
                    (batches[0]["M0"], batches[1]["M0"]),
                    None,
                    (batches[0]["M2"], batches[1]["M2"]),
                    (batches[0]["M3"], batches[1]["M3"]),
                )
                for model_index in (0, 2, 3):
                    weights6, weights14 = weights_by_model[model_index]
                    residual = matched_residual_cross(
                        matched_cross, target, sources, weights6, weights14
                    )
                    vafter_sum[model_index, m_index, k_index] += residual / GENE_COUNT
                    for plate in range(2):
                        raw_full, raw_excess = weighted_same_stats(
                            matched_same[plate],
                            np.asarray(sums[plate]),
                            target,
                            sources,
                            weights_by_model[model_index][plate],
                            same_geometry[plate],
                        )
                        full_metrics = metrics_from_raw(raw_full, GENE_COUNT)
                        excess_metrics = metrics_from_raw(raw_excess, GENE_COUNT)
                        _accumulate_metrics(metric_sum, metric_count, model_index, m_index, k_index, 0, plate, full_metrics)
                        _accumulate_metrics(metric_sum, metric_count, model_index, m_index, k_index, 1, plate, excess_metrics)

                alpha6 = selected[(0, m_index, k_index, sequence)].alpha
                alpha14 = selected[(1, m_index, k_index, sequence)].alpha
                offset_residual = offset_residual_cross(
                    cross_geometry.gram, sentinels, alpha6, alpha14
                )
                vafter_sum[1, m_index, k_index] += offset_residual / GENE_COUNT
                for plate, alpha in enumerate((alpha6, alpha14)):
                    truth_norm = matched_same[plate][:, target, target]
                    raw_full, raw_excess, _ = offset_stats(
                        same_geometry[plate], sentinels, alpha, truth_norm
                    )
                    _accumulate_metrics(
                        metric_sum,
                        metric_count,
                        1,
                        m_index,
                        k_index,
                        0,
                        plate,
                        metrics_from_raw(raw_full, GENE_COUNT),
                    )
                    _accumulate_metrics(
                        metric_sum,
                        metric_count,
                        1,
                        m_index,
                        k_index,
                        1,
                        plate,
                        metrics_from_raw(raw_excess, GENE_COUNT),
                    )

                normal6 = batches[0]["M2"]
                normal14 = batches[1]["M2"]
                null_sum[0, m_index, k_index] += _intervention_null_residual(
                    cross4,
                    matched_cross,
                    target,
                    sources,
                    normal6,
                    normal14,
                    intervention_null,
                    truth_excess,
                ) / GENE_COUNT
                for null_index, key in enumerate(
                    ("NULL_CONTEXT", "NULL_SENTINEL", "NULL_REFERENCE"), start=1
                ):
                    null_sum[null_index, m_index, k_index] += matched_residual_cross(
                        matched_cross,
                        target,
                        sources,
                        batches[0][key],
                        batches[1][key],
                    ) / GENE_COUNT
                vafter_count[m_index, k_index] += 1
                null_count[m_index, k_index] += 1

                if m == 49 and k == 92 and sequence == 0:
                    allbut_vtruth[:] = truth_excess / GENE_COUNT
                    for model_index in range(len(MODELS)):
                        allbut_vafter[model_index] = vafter_sum[
                            model_index, m_index, k_index
                        ]
                    for model_index in (0, 2, 3):
                        for plate in range(2):
                            raw_full, raw_excess = weighted_same_stats(
                                matched_same[plate],
                                np.asarray(sums[plate]),
                                target,
                                sources,
                                weights_by_model[model_index][plate],
                                same_geometry[plate],
                            )
                            allbut_metrics[model_index, :, 0, plate] = metrics_from_raw(raw_full, GENE_COUNT)
                            allbut_metrics[model_index, :, 1, plate] = metrics_from_raw(raw_excess, GENE_COUNT)
                    for plate, alpha in enumerate((alpha6, alpha14)):
                        raw_full, raw_excess, _ = offset_stats(
                            same_geometry[plate],
                            sentinels,
                            alpha,
                            matched_same[plate][:, target, target],
                        )
                        allbut_metrics[1, :, 0, plate] = metrics_from_raw(raw_full, GENE_COUNT)
                        allbut_metrics[1, :, 1, plate] = metrics_from_raw(raw_excess, GENE_COUNT)
            print(
                f"entrywise surface target {target + 1}/50 m={m} sequence={sequence + 1}/8",
                flush=True,
            )

    payload = {
        "tuning_protocol": np.asarray("EPISODE_REFERENCE_SUPPORT_ONLY_V2"),
        "vtruth_sum": vtruth_sum,
        "vtruth_count": vtruth_count,
        "vafter_sum": vafter_sum,
        "vafter_count": vafter_count,
        "null_sum": null_sum,
        "null_count": null_count,
        "metric_sum": metric_sum,
        "metric_count": metric_count,
        "allbut_metrics": allbut_metrics,
        "allbut_vtruth": allbut_vtruth,
        "allbut_vafter": allbut_vafter,
        "full_truth_cross": full_truth_cross / GENE_COUNT,
        "parameter_rows_json": np.asarray(json.dumps(parameter_rows, allow_nan=True)),
        "target": np.asarray(target),
        "created_at": np.asarray(_now()),
        "support_mode": np.asarray(support_mode),
    }
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)
    return cache_path


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def aggregate_targets(root: Path) -> dict[str, Any]:
    root = root.resolve()
    split = _load_split(root)
    paths = [_target_cache_path(root, target) for target in range(50)]
    if not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise RuntimeError(f"Missing target caches: {missing[:5]}")
    vtruth = np.empty((len(M_VALUES), 50, 93), dtype=np.float64)
    vafter = np.empty((len(MODELS), len(M_VALUES), len(K_VALUES), 50, 93), dtype=np.float64)
    nulls = np.empty((len(NULLS), len(M_VALUES), len(K_VALUES), 50, 93), dtype=np.float64)
    metric_sum = np.zeros((len(MODELS), len(M_VALUES), len(K_VALUES), 2, 2, len(METRIC_NAMES)), dtype=np.float64)
    metric_count = np.zeros_like(metric_sum, dtype=np.int64)
    full_truth = np.empty((50, 93), dtype=np.float64)
    parameter_rows: list[dict[str, Any]] = []
    allbut_rows: list[dict[str, Any]] = []
    for target, path in enumerate(paths):
        cache = np.load(path, allow_pickle=False)
        if "tuning_protocol" not in cache or str(cache["tuning_protocol"]) != "EPISODE_REFERENCE_SUPPORT_ONLY_V2":
            raise RuntimeError(f"Mixed or stale tuning protocols at {path}")
        vtruth[:, target] = _safe_divide(cache["vtruth_sum"], cache["vtruth_count"])
        vafter[:, :, :, target] = _safe_divide(
            cache["vafter_sum"], cache["vafter_count"][None]
        )
        nulls[:, :, :, target] = _safe_divide(
            cache["null_sum"], cache["null_count"][None]
        )
        metric_sum += cache["metric_sum"]
        metric_count += cache["metric_count"]
        full_truth[target] = cache["full_truth_cross"]
        parameter_rows.extend(json.loads(str(cache["parameter_rows_json"])))
        allbut_metrics = cache["allbut_metrics"]
        for model_index, model in enumerate(MODELS):
            for intervention, intervention_id in enumerate(split["interventions"]):
                row = {
                    "context_index": target,
                    "context_id": split["contexts"][target],
                    "intervention_index": intervention,
                    "intervention_id": intervention_id,
                    "model": model,
                    "m": 49,
                    "k": 92,
                    "budget": 4_649,
                    "matrix_fraction": 4_649 / 4_650,
                    "v_truth": float(cache["allbut_vtruth"][intervention]),
                    "v_after": float(cache["allbut_vafter"][model_index, intervention]),
                }
                for target_index, target_name in enumerate(TARGETS):
                    for plate_index, plate in enumerate(PLATES):
                        for metric_index, metric in enumerate(METRIC_NAMES):
                            row[f"{target_name.lower()}_{plate}_{metric}"] = float(
                                allbut_metrics[model_index, intervention, target_index, plate_index, metric_index]
                            )
                allbut_rows.append(row)

    cache_dir = root / "results/cgc_entrywise_compression/_cache"
    np.savez(
        cache_dir / "frozen_utility_table.npz",
        vtruth=vtruth,
        vafter=vafter,
        nulls=nulls,
        full_truth=full_truth,
        models=np.asarray(MODELS),
        null_names=np.asarray(NULLS),
        m_values=np.asarray(M_VALUES),
        k_values=np.asarray(K_VALUES),
    )
    out = root / "results/cgc_entrywise_compression"
    pd.DataFrame(parameter_rows).to_csv(out / "ENTRYWISE_HYPERPARAMETERS.csv", index=False)
    pd.DataFrame(allbut_rows).to_csv(out / "ALL_BUT_ONE_ENTRY_RESULTS.csv", index=False)

    mean_metrics = _safe_divide(metric_sum, metric_count)
    surface_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        for m_index, m in enumerate(M_VALUES):
            denominator = float(vtruth[m_index].sum())
            for k_index, k in enumerate(K_VALUES):
                numerator = float(vafter[model_index, m_index, k_index].sum())
                row = {
                    "model": model,
                    "m": m,
                    "k": k,
                    "budget": 93 * m + k,
                    "matrix_fraction": (93 * m + k) / 4_650,
                    "truth_sum": denominator,
                    "residual_sum": numerator,
                    "g_context_specific": 1.0 - numerator / denominator,
                    "g_full_response": 1.0 - numerator / float(full_truth.sum()),
                }
                for target_index, target_name in enumerate(TARGETS):
                    for plate_index, plate in enumerate(PLATES):
                        for metric_index, metric in enumerate(METRIC_NAMES):
                            row[f"mean_{target_name.lower()}_{plate}_{metric}"] = float(
                                mean_metrics[model_index, m_index, k_index, target_index, plate_index, metric_index]
                            )
                surface_rows.append(row)
    pd.DataFrame(surface_rows).to_csv(out / "ENTRYWISE_RECOVERY_SURFACE.csv", index=False)
    return {
        "created_at": _now(),
        "targets": 50,
        "entries": 4_650,
        "models": list(MODELS),
        "grid_points": len(M_VALUES) * len(K_VALUES),
        "utility_cache": str((cache_dir / "frozen_utility_table.npz").relative_to(root)),
    }


def aggregate_rna_targets(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = [_target_cache_path(root, target, "rna") for target in range(50)]
    if not all(path.exists() for path in paths):
        raise RuntimeError("RNA target caches are incomplete")
    vtruth = np.empty((len(M_VALUES), 50, 93), dtype=np.float64)
    vafter = np.empty((len(MODELS), len(M_VALUES), len(K_VALUES), 50, 93), dtype=np.float64)
    metric_sum = np.zeros((len(MODELS), len(M_VALUES), len(K_VALUES), 2, 2, len(METRIC_NAMES)), dtype=np.float64)
    metric_count = np.zeros_like(metric_sum, dtype=np.int64)
    full_truth = np.empty((50, 93), dtype=np.float64)
    parameter_rows: list[dict[str, Any]] = []
    for target, path in enumerate(paths):
        cache = np.load(path, allow_pickle=False)
        vtruth[:, target] = _safe_divide(cache["vtruth_sum"], cache["vtruth_count"])
        if "tuning_protocol" not in cache or str(cache["tuning_protocol"]) != "EPISODE_REFERENCE_SUPPORT_ONLY_V2":
            raise RuntimeError(f"Mixed or stale RNA tuning protocols at {path}")
        vafter[:, :, :, target] = _safe_divide(
            cache["vafter_sum"], cache["vafter_count"][None]
        )
        metric_sum += cache["metric_sum"]
        metric_count += cache["metric_count"]
        full_truth[target] = cache["full_truth_cross"]
        parameter_rows.extend(json.loads(str(cache["parameter_rows_json"])))
    out = root / "results/cgc_entrywise_compression"
    np.savez(
        out / "_cache/rna_utility_table.npz",
        vtruth=vtruth,
        vafter=vafter,
        full_truth=full_truth,
        models=np.asarray(MODELS),
        m_values=np.asarray(M_VALUES),
        k_values=np.asarray(K_VALUES),
    )
    pd.DataFrame(parameter_rows).to_csv(out / "RNA_INFORMED_HYPERPARAMETERS.csv", index=False)
    mean_metrics = _safe_divide(metric_sum, metric_count)
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        for m_index, m in enumerate(M_VALUES):
            denominator = float(vtruth[m_index].sum())
            for k_index, k in enumerate(K_VALUES):
                numerator = float(vafter[model_index, m_index, k_index].sum())
                row = {
                    "model": model,
                    "support_selection": "RNA_NEAREST_SECONDARY",
                    "m": m,
                    "k": k,
                    "budget": 93 * m + k,
                    "matrix_fraction": (93 * m + k) / 4_650,
                    "truth_sum": denominator,
                    "residual_sum": numerator,
                    "g_context_specific": 1.0 - numerator / denominator,
                    "g_full_response": 1.0 - numerator / float(full_truth.sum()),
                }
                for target_index, target_name in enumerate(TARGETS):
                    for plate_index, plate in enumerate(PLATES):
                        for metric_index, metric in enumerate(METRIC_NAMES):
                            row[f"mean_{target_name.lower()}_{plate}_{metric}"] = float(
                                mean_metrics[model_index, m_index, k_index, target_index, plate_index, metric_index]
                            )
                rows.append(row)
    pd.DataFrame(rows).to_csv(out / "RNA_INFORMED_COMPRESSION_RESULTS.csv", index=False)
    return {"created_at": _now(), "targets": 50, "support_mode": "rna"}


def run_surface(
    root: Path,
    overwrite: bool = False,
    target_start: int = 0,
    target_stop: int = 50,
    support_mode: str = "random",
) -> dict[str, Any]:
    root = root.resolve()
    for target in range(target_start, target_stop):
        run_target(root, target, overwrite=overwrite, support_mode=support_mode)
    if target_start == 0 and target_stop == 50:
        return aggregate_targets(root) if support_mode == "random" else aggregate_rna_targets(root)
    return {"targets_run": [target_start, target_stop], "complete": False}
