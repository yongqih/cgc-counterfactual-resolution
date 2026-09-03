from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


@dataclass(frozen=True)
class PlatformData:
    platform: str
    patient_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    drug_names: np.ndarray
    eligible_pdo_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class KernelFit:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    test_kernel: np.ndarray
    retained_features: int


@dataclass(frozen=True)
class OuterFoldCache:
    held_index: int
    train_indices: np.ndarray
    patient_id: str
    y_train: np.ndarray
    y_mean: np.ndarray
    y_true: np.ndarray
    kernel: KernelFit
    ridge_alpha: float


PAIR_I, PAIR_J = np.triu_indices(24, k=1)


def _natural_patient_key(patient_id: str) -> int:
    return int(patient_id.removeprefix("Pt"))


def load_platform_data(processed_dir: Path, platform: str) -> PlatformData:
    expression_name = {
        "RNAseq": "RNAseq_PDO_log2CPM1.npz",
        "HTA2.0": "HTA2_PDO_public_processed.npz",
    }[platform]
    expression = np.load(processed_dir / expression_name)
    dss = np.load(processed_dir / "PRIMARY_DSS.npz")

    expression_samples = [str(value) for value in expression["sample_ids"]]
    expression_patients = [str(value) for value in expression["patient_ids"]]
    dss_samples = [str(value) for value in dss["sample_ids"]]
    dss_lookup = {sample_id: index for index, sample_id in enumerate(dss_samples)}
    joint_samples = [sample_id for sample_id in expression_samples if sample_id in dss_lookup]
    if not joint_samples:
        raise ValueError(f"No joint expression/DSS samples for {platform}")

    expression_lookup = {sample_id: index for index, sample_id in enumerate(expression_samples)}
    patient_samples: dict[str, list[str]] = {}
    for sample_id, patient_id in zip(expression_samples, expression_patients, strict=True):
        if sample_id in dss_lookup:
            patient_samples.setdefault(patient_id, []).append(sample_id)

    patient_ids = sorted(patient_samples, key=_natural_patient_key)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    pdo_ids: list[tuple[str, ...]] = []
    for patient_id in patient_ids:
        samples = sorted(patient_samples[patient_id])
        x_rows.append(np.mean(expression["values"][[expression_lookup[value] for value in samples]], axis=0))
        y_rows.append(np.mean(dss["values"][[dss_lookup[value] for value in samples]], axis=0))
        pdo_ids.append(tuple(samples))

    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite patient matrix for {platform}")
    return PlatformData(
        platform=platform,
        patient_ids=np.asarray(patient_ids, dtype="U16"),
        x=x,
        y=y,
        drug_names=dss["drug_names"].astype("U32"),
        eligible_pdo_ids=tuple(pdo_ids),
    )


def _standardized_kernel(x_train: np.ndarray, x_test: np.ndarray) -> KernelFit:
    mean = np.mean(x_train, axis=0)
    variance = np.var(x_train, axis=0)
    keep = np.isfinite(variance) & (variance > 1e-12)
    retained = int(keep.sum())
    if retained == 0:
        raise ValueError("No nonzero-variance training features")
    scale = np.sqrt(variance[keep])
    norm = math.sqrt(retained)
    z_train = ((x_train[:, keep] - mean[keep]) / scale) / norm
    z_test = ((x_test[:, keep] - mean[keep]) / scale) / norm
    kernel = z_train @ z_train.T
    test_kernel = z_test @ z_train.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return KernelFit(eigenvalues, eigenvectors, test_kernel, retained)


def _predict_from_kernel(
    fit: KernelFit,
    y_residual: np.ndarray,
    alpha: float,
    *,
    components: int | None = None,
    permutation: np.ndarray | None = None,
) -> np.ndarray:
    eigenvalues = fit.eigenvalues
    eigenvectors = fit.eigenvectors
    test_kernel = fit.test_kernel
    if permutation is not None:
        eigenvectors = eigenvectors[permutation]
        test_kernel = test_kernel[:, permutation]
    positive = eigenvalues > 1e-12
    if components is not None:
        indices = np.flatnonzero(positive)[:components]
    else:
        indices = np.flatnonzero(positive)
    if len(indices) == 0:
        return np.zeros((test_kernel.shape[0], y_residual.shape[1]), dtype=np.float64)
    values = eigenvalues[indices]
    vectors = eigenvectors[:, indices]
    left = test_kernel @ vectors
    right = vectors.T @ y_residual
    return (left / (values + alpha)) @ right


def _kendall_tau_b_rows(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_diff = truth[:, PAIR_I] - truth[:, PAIR_J]
    pred_diff = prediction[:, PAIR_I] - prediction[:, PAIR_J]
    truth_sign = np.sign(truth_diff)
    pred_sign = np.sign(pred_diff)
    numerator = np.sum(truth_sign * pred_sign, axis=1)
    truth_n = np.sum(truth_sign != 0, axis=1)
    pred_n = np.sum(pred_sign != 0, axis=1)
    denominator = np.sqrt(truth_n * pred_n)
    return np.divide(
        numerator,
        denominator,
        out=np.full(truth.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _reversal_patient_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    population: np.ndarray,
    tolerance: float,
) -> dict[str, np.ndarray]:
    true_diff = truth[:, PAIR_I] - truth[:, PAIR_J]
    pred_diff = prediction[:, PAIR_I] - prediction[:, PAIR_J]
    pop_diff = population[:, PAIR_I] - population[:, PAIR_J]
    true_sign = np.where(np.abs(true_diff) <= tolerance, 0, np.sign(true_diff))
    pred_sign = np.where(np.abs(pred_diff) <= tolerance, 0, np.sign(pred_diff))
    pop_sign = np.where(np.abs(pop_diff) <= tolerance, 0, np.sign(pop_diff))
    valid = (true_sign != 0) & (pop_sign != 0)
    actual = valid & (true_sign == -pop_sign)
    predicted = valid & (pred_sign == -pop_sign)
    non_actual = valid & ~actual
    tp = np.sum(actual & predicted, axis=1)
    fn = np.sum(actual & ~predicted, axis=1)
    fp = np.sum(non_actual & predicted, axis=1)
    tn = np.sum(non_actual & ~predicted, axis=1)
    recall = np.divide(tp, tp + fn, out=np.full(len(tp), np.nan), where=(tp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.full(len(tp), np.nan), where=(tp + fp) > 0)
    specificity = np.divide(tn, tn + fp, out=np.full(len(tp), np.nan), where=(tn + fp) > 0)
    balanced_accuracy = (recall + specificity) / 2.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.full(len(tp), np.nan),
        where=denominator > 0,
    )
    return {
        "true_reversals": np.sum(actual, axis=1),
        "valid_pairs": np.sum(valid, axis=1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def _core_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    population: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    tau_model = _kendall_tau_b_rows(truth, prediction)
    tau_population = _kendall_tau_b_rows(truth, population)
    mse_model = np.mean((truth - prediction) ** 2, axis=1)
    mse_population = np.mean((truth - population) ** 2, axis=1)
    reversal = _reversal_patient_metrics(truth, prediction, population, tolerance)
    return {
        "tau_model": tau_model,
        "tau_population": tau_population,
        "pg_patient": tau_model - tau_population,
        "mse_model": mse_model,
        "mse_population": mse_population,
        "g_func_patient": 1.0 - mse_model / mse_population,
        "PG_macro": float(np.nanmean(tau_model - tau_population)),
        "g_func": float(1.0 - np.mean(mse_model) / np.mean(mse_population)),
        "reversal_balanced_accuracy": float(np.nanmean(reversal["balanced_accuracy"])),
        "reversal": reversal,
    }


def _inner_folds(patient_ids: np.ndarray, seed: int, n_splits: int = 5) -> list[np.ndarray]:
    keyed = []
    for index, patient_id in enumerate(patient_ids):
        token = hashlib.sha256(f"{seed}|{patient_id}".encode()).hexdigest()
        keyed.append((token, index))
    order = np.asarray([index for _, index in sorted(keyed)], dtype=int)
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, n_splits)]


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    alphas: Iterable[float],
    *,
    components: int | None,
    seed: int,
) -> tuple[float, dict[float, float]]:
    alpha_values = [float(value) for value in alphas]
    score_lists: dict[float, list[float]] = {value: [] for value in alpha_values}
    for val_indices in _inner_folds(patient_ids, seed):
        train_indices = np.setdiff1d(np.arange(len(patient_ids)), val_indices, assume_unique=True)
        fit = _standardized_kernel(x[train_indices], x[val_indices])
        y_mean = np.mean(y[train_indices], axis=0)
        residual = y[train_indices] - y_mean
        for alpha in alpha_values:
            prediction = y_mean + _predict_from_kernel(fit, residual, alpha, components=components)
            score_lists[alpha].extend(_kendall_tau_b_rows(y[val_indices], prediction).tolist())
    scores = {alpha: float(np.nanmean(values)) for alpha, values in score_lists.items()}
    selected = max(alpha_values, key=lambda alpha: (scores[alpha], alpha))
    return selected, scores


def fit_oof_platform(
    data: PlatformData,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[OuterFoldCache]]:
    model = config["model"]
    alphas = model["alpha_grid"]
    seed = int(model["seed"])
    components = int(model["pca_components"])
    n_patients, n_drugs = data.y.shape
    predictions = {
        "Ridge": np.full((n_patients, n_drugs), np.nan, dtype=np.float64),
        f"PCA{components}+Ridge": np.full((n_patients, n_drugs), np.nan, dtype=np.float64),
    }
    population = np.full_like(data.y, np.nan)
    fold_records: list[dict[str, Any]] = []
    caches: list[OuterFoldCache] = []

    for held_index in range(n_patients):
        train_indices = np.delete(np.arange(n_patients), held_index)
        x_train = data.x[train_indices]
        y_train = data.y[train_indices]
        train_patients = data.patient_ids[train_indices]
        ridge_alpha, ridge_scores = _select_alpha(
            x_train,
            y_train,
            train_patients,
            alphas,
            components=None,
            seed=seed + held_index,
        )
        pca_alpha, pca_scores = _select_alpha(
            x_train,
            y_train,
            train_patients,
            alphas,
            components=components,
            seed=seed + held_index,
        )
        fit = _standardized_kernel(x_train, data.x[[held_index]])
        y_mean = np.mean(y_train, axis=0)
        residual = y_train - y_mean
        population[held_index] = y_mean
        predictions["Ridge"][held_index] = (
            y_mean + _predict_from_kernel(fit, residual, ridge_alpha, components=None)[0]
        )
        predictions[f"PCA{components}+Ridge"][held_index] = (
            y_mean + _predict_from_kernel(fit, residual, pca_alpha, components=components)[0]
        )
        fold_records.append(
            {
                "platform": data.platform,
                "held_patient": str(data.patient_ids[held_index]),
                "held_patient_index": held_index,
                "held_pdo_count": len(data.eligible_pdo_ids[held_index]),
                "held_pdo_ids": ";".join(data.eligible_pdo_ids[held_index]),
                "training_patients": len(train_indices),
                "patient_overlap": 0,
                "retained_features": fit.retained_features,
                "ridge_alpha": ridge_alpha,
                "pca_components": components,
                "pca_ridge_alpha": pca_alpha,
                "ridge_inner_tau": ridge_scores[ridge_alpha],
                "pca_ridge_inner_tau": pca_scores[pca_alpha],
            }
        )
        caches.append(
            OuterFoldCache(
                held_index=held_index,
                train_indices=train_indices,
                patient_id=str(data.patient_ids[held_index]),
                y_train=y_train,
                y_mean=y_mean,
                y_true=data.y[held_index],
                kernel=fit,
                ridge_alpha=ridge_alpha,
            )
        )
    predictions["Population"] = population
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError(f"Non-finite OOF prediction for {data.platform}")
    return predictions, pd.DataFrame.from_records(fold_records), caches


def run_primary_nulls(
    data: PlatformData,
    caches: list[OuterFoldCache],
    observed_prediction: np.ndarray,
    population: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    n_draws = int(config["nulls"]["refit_permutations"])
    seed = int(config["nulls"]["seed"])
    tolerance = float(config["evaluation"]["pair_tolerance"])
    records: list[dict[str, Any]] = []

    def append_metrics(null_family: str, draw: int, prediction: np.ndarray) -> None:
        metrics = _core_metrics(data.y, prediction, population, tolerance)
        records.append(
            {
                "null_family": null_family,
                "draw": draw,
                "PG_macro": metrics["PG_macro"],
                "g_func": metrics["g_func"],
                "reversal_balanced_accuracy": metrics["reversal_balanced_accuracy"],
                "reversal_balanced_accuracy_minus_0.5": metrics["reversal_balanced_accuracy"] - 0.5,
            }
        )

    append_metrics("observed_Ridge", 0, observed_prediction)
    append_metrics("population_only", 0, population)
    append_metrics("baseline_free_drug_model", 0, population)
    for draw in range(n_draws):
        baseline_prediction = np.full_like(data.y, np.nan)
        response_prediction = np.full_like(data.y, np.nan)
        for cache in caches:
            fold_rng = np.random.default_rng(seed + draw * 1009 + cache.held_index * 9176)
            baseline_permutation = fold_rng.permutation(len(cache.train_indices))
            response_permutation = fold_rng.permutation(len(cache.train_indices))
            residual = cache.y_train - cache.y_mean
            baseline_prediction[cache.held_index] = cache.y_mean + _predict_from_kernel(
                cache.kernel,
                residual,
                cache.ridge_alpha,
                permutation=baseline_permutation,
            )[0]
            response_prediction[cache.held_index] = cache.y_mean + _predict_from_kernel(
                cache.kernel,
                residual[response_permutation],
                cache.ridge_alpha,
            )[0]
        append_metrics("baseline_RNA_shuffle", draw + 1, baseline_prediction)
        append_metrics("patient_response_profile_shuffle", draw + 1, response_prediction)
    return pd.DataFrame.from_records(records)


def primary_inference(
    truth: np.ndarray,
    prediction: np.ndarray,
    population: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tolerance = float(config["evaluation"]["pair_tolerance"])
    metrics = _core_metrics(truth, prediction, population, tolerance)
    estimates = np.asarray(
        [
            metrics["PG_macro"],
            metrics["g_func"],
            metrics["reversal_balanced_accuracy"] - 0.5,
        ],
        dtype=np.float64,
    )
    names = ["PG_macro", "g_func", "reversal_balanced_accuracy_minus_0.5"]
    n_patients = len(truth)
    inference = config["inference"]
    rng = np.random.default_rng(int(inference["seed"]))
    bootstrap_draws = int(inference["bootstrap_draws"])
    bootstrap = np.empty((bootstrap_draws, 3), dtype=np.float64)
    for draw in range(bootstrap_draws):
        indices = rng.integers(0, n_patients, size=n_patients)
        boot_metrics = _core_metrics(truth[indices], prediction[indices], population[indices], tolerance)
        bootstrap[draw] = (
            boot_metrics["PG_macro"],
            boot_metrics["g_func"],
            boot_metrics["reversal_balanced_accuracy"] - 0.5,
        )
    standard_errors = np.std(bootstrap, axis=0, ddof=1)
    standardized_deviation = np.divide(
        np.abs(bootstrap - estimates),
        standard_errors,
        out=np.zeros_like(bootstrap),
        where=standard_errors > 0,
    )
    critical_value = float(np.quantile(np.max(standardized_deviation, axis=1), 0.95))
    lower = estimates - critical_value * standard_errors
    upper = estimates + critical_value * standard_errors

    contributions = np.column_stack(
        [
            metrics["pg_patient"],
            metrics["mse_population"] - metrics["mse_model"],
            metrics["reversal"]["balanced_accuracy"] - 0.5,
        ]
    )
    valid = np.isfinite(contributions).all(axis=1)
    contributions = contributions[valid]
    contribution_se = np.std(contributions, axis=0, ddof=1) / math.sqrt(len(contributions))
    observed_mean = np.mean(contributions, axis=0)
    observed_t = np.divide(
        observed_mean,
        contribution_se,
        out=np.where(observed_mean > 0, np.inf, np.where(observed_mean < 0, -np.inf, 0.0)),
        where=contribution_se > 0,
    )
    signflip_draws = int(inference["signflip_draws"])
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(signflip_draws, len(contributions)))
    signed_contributions = signs[:, :, None] * contributions[None, :, :]
    permuted_mean = np.mean(signed_contributions, axis=1)
    permuted_se = np.std(signed_contributions, axis=1, ddof=1) / math.sqrt(len(contributions))
    permuted_t = np.divide(
        permuted_mean,
        permuted_se,
        out=np.zeros_like(permuted_mean),
        where=permuted_se > 0,
    )
    max_t = np.max(permuted_t, axis=1)
    adjusted_p = np.asarray(
        [(1.0 + np.sum(max_t >= value)) / (signflip_draws + 1.0) for value in observed_t]
    )
    table = pd.DataFrame(
        {
            "metric": names,
            "estimate": estimates,
            "simultaneous_95_ci_lower": lower,
            "simultaneous_95_ci_upper": upper,
            "bootstrap_se": standard_errors,
            "max_deviation_critical_value": critical_value,
            "one_sided_maxT_adjusted_p": adjusted_p,
            "positive_after_joint_correction": lower > 0,
            "patients": n_patients,
            "bootstrap_draws": bootstrap_draws,
            "signflip_draws": signflip_draws,
        }
    )
    details = {
        "critical_value": critical_value,
        "bootstrap_draws": bootstrap_draws,
        "signflip_draws": signflip_draws,
        "all_primary_lower_bounds_positive": bool(np.all(lower > 0)),
    }
    return table, details


def build_output_tables(
    platform_results: list[tuple[PlatformData, dict[str, np.ndarray]]],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    tolerance = float(config["evaluation"]["pair_tolerance"])
    prediction_records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    population_records: list[dict[str, Any]] = []
    gain_records: list[dict[str, Any]] = []
    recovery_records: list[dict[str, Any]] = []
    reversal_records: list[dict[str, Any]] = []
    near_tie_records: list[dict[str, Any]] = []
    topk_records: list[dict[str, Any]] = []

    for data, predictions in platform_results:
        population = predictions["Population"]
        true_rank = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, data.y)
        pop_rank = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, population)
        for patient_index, patient_id in enumerate(data.patient_ids):
            for drug_index, drug_name in enumerate(data.drug_names):
                population_records.append(
                    {
                        "platform": data.platform,
                        "patient_id": patient_id,
                        "drug_name": drug_name,
                        "true_DSS": data.y[patient_index, drug_index],
                        "population_DSS": population[patient_index, drug_index],
                        "true_rank": true_rank[patient_index, drug_index],
                        "population_rank": pop_rank[patient_index, drug_index],
                    }
                )

        for estimator, prediction in predictions.items():
            if estimator == "Population":
                continue
            metrics = _core_metrics(data.y, prediction, population, tolerance)
            model_rank = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, prediction)
            spearman_model = np.asarray(
                [spearmanr(data.y[index], prediction[index]).statistic for index in range(len(data.y))]
            )
            spearman_population = np.asarray(
                [spearmanr(data.y[index], population[index]).statistic for index in range(len(data.y))]
            )
            reversal = metrics["reversal"]
            for patient_index, patient_id in enumerate(data.patient_ids):
                gain_records.append(
                    {
                        "platform": data.platform,
                        "estimator": estimator,
                        "patient_id": patient_id,
                        "pdo_count": len(data.eligible_pdo_ids[patient_index]),
                        "kendall_model": metrics["tau_model"][patient_index],
                        "kendall_population": metrics["tau_population"][patient_index],
                        "PG_i": metrics["pg_patient"][patient_index],
                        "spearman_model": spearman_model[patient_index],
                        "spearman_population": spearman_population[patient_index],
                    }
                )
                recovery_records.append(
                    {
                        "platform": data.platform,
                        "estimator": estimator,
                        "patient_id": patient_id,
                        "row_type": "patient",
                        "mse_model": metrics["mse_model"][patient_index],
                        "mse_population": metrics["mse_population"][patient_index],
                        "g_func": metrics["g_func_patient"][patient_index],
                    }
                )
                for drug_index, drug_name in enumerate(data.drug_names):
                    prediction_records.append(
                        {
                            "platform": data.platform,
                            "estimator": estimator,
                            "patient_id": patient_id,
                            "drug_name": drug_name,
                            "true_DSS": data.y[patient_index, drug_index],
                            "predicted_DSS": prediction[patient_index, drug_index],
                            "population_DSS": population[patient_index, drug_index],
                            "residual_truth": data.y[patient_index, drug_index] - population[patient_index, drug_index],
                            "residual_prediction": prediction[patient_index, drug_index] - population[patient_index, drug_index],
                        }
                    )
                    ranking_records.append(
                        {
                            "platform": data.platform,
                            "estimator": estimator,
                            "patient_id": patient_id,
                            "drug_name": drug_name,
                            "true_rank": true_rank[patient_index, drug_index],
                            "model_rank": model_rank[patient_index, drug_index],
                            "population_rank": pop_rank[patient_index, drug_index],
                        }
                    )
                for k in config["evaluation"]["top_k"]:
                    truth_top = set(np.argsort(-data.y[patient_index])[:k])
                    model_top = set(np.argsort(-prediction[patient_index])[:k])
                    pop_top = set(np.argsort(-population[patient_index])[:k])
                    topk_records.append(
                        {
                            "platform": data.platform,
                            "estimator": estimator,
                            "patient_id": patient_id,
                            "k": k,
                            "model_overlap_count": len(truth_top & model_top),
                            "model_overlap_fraction": len(truth_top & model_top) / k,
                            "population_overlap_count": len(truth_top & pop_top),
                            "population_overlap_fraction": len(truth_top & pop_top) / k,
                            "model_exact_set_recovery": truth_top == model_top,
                            "population_exact_set_recovery": truth_top == pop_top,
                        }
                    )

                pop_diff = population[patient_index, PAIR_I] - population[patient_index, PAIR_J]
                true_diff = data.y[patient_index, PAIR_I] - data.y[patient_index, PAIR_J]
                pred_diff = prediction[patient_index, PAIR_I] - prediction[patient_index, PAIR_J]
                pop_sign = np.where(np.abs(pop_diff) <= tolerance, 0, np.sign(pop_diff))
                true_sign = np.where(np.abs(true_diff) <= tolerance, 0, np.sign(true_diff))
                pred_sign = np.where(np.abs(pred_diff) <= tolerance, 0, np.sign(pred_diff))
                valid = (pop_sign != 0) & (true_sign != 0)
                margin = np.abs(pop_diff)
                q1, q2 = np.quantile(margin, config["evaluation"]["near_tie_quantiles"])
                bins = np.where(margin <= q1, "low", np.where(margin <= q2, "middle", "high"))
                for pair_index in np.flatnonzero(valid):
                    true_reversal = true_sign[pair_index] == -pop_sign[pair_index]
                    predicted_reversal = pred_sign[pair_index] == -pop_sign[pair_index]
                    reversal_records.append(
                        {
                            "platform": data.platform,
                            "estimator": estimator,
                            "patient_id": patient_id,
                            "drug_A": data.drug_names[PAIR_I[pair_index]],
                            "drug_B": data.drug_names[PAIR_J[pair_index]],
                            "population_margin": pop_diff[pair_index],
                            "true_margin": true_diff[pair_index],
                            "predicted_margin": pred_diff[pair_index],
                            "population_margin_bin": bins[pair_index],
                            "true_reversal": true_reversal,
                            "predicted_reversal": predicted_reversal,
                            "reversal_correct": true_reversal == predicted_reversal,
                        }
                    )
            recovery_records.append(
                {
                    "platform": data.platform,
                    "estimator": estimator,
                    "patient_id": "MACRO",
                    "row_type": "patient_balanced_macro",
                    "mse_model": float(np.mean(metrics["mse_model"])),
                    "mse_population": float(np.mean(metrics["mse_population"])),
                    "g_func": metrics["g_func"],
                }
            )

            estimator_reversals = pd.DataFrame.from_records(reversal_records)
            estimator_reversals = estimator_reversals[
                estimator_reversals["platform"].eq(data.platform)
                & estimator_reversals["estimator"].eq(estimator)
            ]
            for margin_bin in ("low", "middle", "high"):
                subset = estimator_reversals[estimator_reversals["population_margin_bin"].eq(margin_bin)]
                actual = subset["true_reversal"].to_numpy(bool)
                predicted_reversal = subset["predicted_reversal"].to_numpy(bool)
                tp = int(np.sum(actual & predicted_reversal))
                fn = int(np.sum(actual & ~predicted_reversal))
                fp = int(np.sum(~actual & predicted_reversal))
                tn = int(np.sum(~actual & ~predicted_reversal))
                recall = tp / (tp + fn) if tp + fn else np.nan
                precision = tp / (tp + fp) if tp + fp else np.nan
                specificity = tn / (tn + fp) if tn + fp else np.nan
                denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                near_tie_records.append(
                    {
                        "platform": data.platform,
                        "estimator": estimator,
                        "population_margin_bin": margin_bin,
                        "valid_pairs": len(subset),
                        "true_reversals": int(actual.sum()),
                        "reversal_prevalence": float(actual.mean()) if len(actual) else np.nan,
                        "reversal_recall": recall,
                        "reversal_precision": precision,
                        "balanced_accuracy": (recall + specificity) / 2 if np.isfinite(recall + specificity) else np.nan,
                        "mcc": (tp * tn - fp * fn) / denominator if denominator else np.nan,
                    }
                )

    return {
        "HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv": pd.DataFrame.from_records(prediction_records),
        "HELD_PATIENT_DRUG_RANKINGS.csv": pd.DataFrame.from_records(ranking_records),
        "POPULATION_REFERENCE_RANKINGS.csv": pd.DataFrame.from_records(population_records),
        "PERSONALIZATION_GAIN.csv": pd.DataFrame.from_records(gain_records),
        "FUNCTIONAL_PERSONALIZATION_RECOVERY.csv": pd.DataFrame.from_records(recovery_records),
        "PREFERENCE_REVERSALS.csv": pd.DataFrame.from_records(reversal_records),
        "NEAR_TIE_REVERSAL_ANALYSIS.csv": pd.DataFrame.from_records(near_tie_records),
        "TOPK_RECOVERY.csv": pd.DataFrame.from_records(topk_records),
    }
