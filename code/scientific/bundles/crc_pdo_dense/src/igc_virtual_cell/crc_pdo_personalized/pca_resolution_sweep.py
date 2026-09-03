from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .conditioning_resolution import (
    _pca_scores_from_gram,
    standardize_genes,
)
from .modeling import (
    PlatformData,
    _core_metrics,
    _inner_folds,
    _predict_from_kernel,
    _standardized_kernel,
)


@dataclass(frozen=True)
class PCASweepFit:
    predictions: dict[str, np.ndarray]
    population: np.ndarray
    selections: pd.DataFrame
    grid_manifest: pd.DataFrame
    variance_capture: pd.DataFrame


def _profile_mse(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """One equally weighted full-drug-profile loss per patient."""
    return np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2, axis=1)


def _best_alpha(scores: dict[float, float]) -> float:
    """Frozen tie break: minimum MSE, then larger alpha."""
    return min(scores, key=lambda alpha: (scores[alpha], -alpha))


def _best_k_alpha(scores: dict[tuple[int, float], float]) -> tuple[int, float]:
    """Frozen tie break: minimum MSE, smaller k, then larger alpha."""
    return min(scores, key=lambda item: (scores[item], item[0], -item[1]))


def _inner_mse_scores(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    k_grid: tuple[int, ...],
    alphas: tuple[float, ...],
    seed: int,
) -> tuple[dict[tuple[int, float], float], dict[float, float], bool]:
    pca_sum = {(k, alpha): 0.0 for k in k_grid for alpha in alphas}
    pca_count = {(k, alpha): 0 for k in k_grid for alpha in alphas}
    full_sum = {alpha: 0.0 for alpha in alphas}
    full_count = {alpha: 0 for alpha in alphas}
    sealed = True
    maximum_k = max(k_grid)

    for validation in _inner_folds(patient_ids, seed):
        training = np.setdiff1d(np.arange(len(patient_ids)), validation, assume_unique=True)
        y_mean = np.mean(y[training], axis=0)
        residual = y[training] - y_mean

        full_fit = _standardized_kernel(x[training], x[validation])
        for alpha in alphas:
            prediction = y_mean + _predict_from_kernel(full_fit, residual, alpha)
            losses = _profile_mse(y[validation], prediction)
            full_sum[alpha] += float(np.sum(losses))
            full_count[alpha] += len(losses)

        z_training, z_validation, _ = standardize_genes(x[training], x[validation])
        pca_training, pca_validation, available_rank = _pca_scores_from_gram(
            z_training, z_validation, maximum_k
        )
        for k in k_grid:
            if k > available_rank:
                raise ValueError(f"Requested k={k} exceeds inner-training PCA rank {available_rank}")
            fit = _standardized_kernel(pca_training[:, :k], pca_validation[:, :k])
            for alpha in alphas:
                prediction = y_mean + _predict_from_kernel(fit, residual, alpha)
                losses = _profile_mse(y[validation], prediction)
                pca_sum[(k, alpha)] += float(np.sum(losses))
                pca_count[(k, alpha)] += len(losses)
        sealed &= np.intersect1d(training, validation).size == 0

    pca_scores = {
        item: pca_sum[item] / pca_count[item]
        for item in pca_sum
    }
    full_scores = {
        alpha: full_sum[alpha] / full_count[alpha]
        for alpha in alphas
    }
    return pca_scores, full_scores, sealed


def fit_nested_pca_oof(
    data: PlatformData,
    k_grid: Iterable[int],
    alphas: Iterable[float],
    seed: int,
) -> PCASweepFit:
    k_values = tuple(int(value) for value in k_grid)
    alpha_values = tuple(float(value) for value in alphas)
    if tuple(sorted(set(k_values))) != k_values:
        raise ValueError("PCA k grid must be strictly increasing and unique")
    if max(k_values) >= len(data.patient_ids) - 1:
        raise ValueError("Maximum k is not estimable within every outer-training fold")

    n_patients, n_drugs = data.y.shape
    predictions = {
        **{f"PCA{k}": np.full((n_patients, n_drugs), np.nan) for k in k_values},
        "PCA_STAR": np.full((n_patients, n_drugs), np.nan),
        "FULL_RNA_REOPT": np.full((n_patients, n_drugs), np.nan),
    }
    population = np.full_like(data.y, np.nan, dtype=np.float64)
    selection_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []

    for held_index, held_patient in enumerate(data.patient_ids):
        training = np.delete(np.arange(n_patients), held_index)
        x_training = data.x[training]
        y_training = data.y[training]
        patient_training = data.patient_ids[training]
        pca_scores, full_scores, sealed = _inner_mse_scores(
            x_training,
            y_training,
            patient_training,
            k_values,
            alpha_values,
            seed + held_index,
        )
        selected_k, selected_alpha = _best_k_alpha(pca_scores)
        full_alpha = _best_alpha(full_scores)
        fixed_alphas = {
            k: _best_alpha({alpha: pca_scores[(k, alpha)] for alpha in alpha_values})
            for k in k_values
        }

        z_training, z_held, keep = standardize_genes(x_training, data.x[[held_index]])
        pca_training, pca_held, available_rank = _pca_scores_from_gram(
            z_training, z_held, max(k_values)
        )
        if max(k_values) > available_rank:
            raise ValueError(f"Requested k exceeds outer-training PCA rank {available_rank}")
        eigenvalues = np.linalg.eigvalsh((z_training @ z_training.T + (z_training @ z_training.T).T) / 2)
        eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
        total_variance = float(np.sum(eigenvalues))
        y_mean = np.mean(y_training, axis=0)
        residual = y_training - y_mean
        population[held_index] = y_mean

        full_fit = _standardized_kernel(x_training, data.x[[held_index]])
        predictions["FULL_RNA_REOPT"][held_index] = y_mean + _predict_from_kernel(
            full_fit, residual, full_alpha
        )[0]

        for k in k_values:
            fit = _standardized_kernel(pca_training[:, :k], pca_held[:, :k])
            prediction = y_mean + _predict_from_kernel(fit, residual, fixed_alphas[k])[0]
            predictions[f"PCA{k}"][held_index] = prediction
            if k == selected_k:
                predictions["PCA_STAR"][held_index] = prediction
            variance_rows.append(
                {
                    "held_patient": str(held_patient),
                    "held_patient_index": held_index,
                    "k": k,
                    "training_genes_retained": int(np.sum(keep)),
                    "available_training_rank": available_rank,
                    "training_expression_variance_explained": float(np.sum(eigenvalues[:k]) / total_variance),
                    "preprocessing_scope": "outer_training_only",
                }
            )
            for alpha in alpha_values:
                grid_rows.append(
                    {
                        "held_patient": str(held_patient),
                        "held_patient_index": held_index,
                        "representation": f"PCA{k}",
                        "k": k,
                        "alpha": alpha,
                        "inner_patient_profile_mse": pca_scores[(k, alpha)],
                        "selected_for_fixed_k": alpha == fixed_alphas[k],
                        "selected_jointly": k == selected_k and alpha == selected_alpha,
                        "inner_preprocessing_refit": True,
                        "outer_outcome_used": False,
                    }
                )
        for alpha in alpha_values:
            grid_rows.append(
                {
                    "held_patient": str(held_patient),
                    "held_patient_index": held_index,
                    "representation": "FULL_RNA_REOPT",
                    "k": np.nan,
                    "alpha": alpha,
                    "inner_patient_profile_mse": full_scores[alpha],
                    "selected_for_fixed_k": alpha == full_alpha,
                    "selected_jointly": False,
                    "inner_preprocessing_refit": True,
                    "outer_outcome_used": False,
                }
            )
        selection_rows.append(
            {
                "held_patient": str(held_patient),
                "held_patient_index": held_index,
                "training_patients": len(training),
                "patient_overlap": 0,
                "selected_k": selected_k,
                "selected_alpha": selected_alpha,
                "selected_inner_patient_profile_mse": pca_scores[(selected_k, selected_alpha)],
                "full_rna_reopt_alpha": full_alpha,
                "full_rna_reopt_inner_patient_profile_mse": full_scores[full_alpha],
                "all_inner_splits_sealed": sealed,
                "pca_fit_scope": "outer_and_inner_training_only",
                "held_outcome_used_in_selection": False,
            }
        )

    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Non-finite PCA sweep prediction")
    return PCASweepFit(
        predictions=predictions,
        population=population,
        selections=pd.DataFrame.from_records(selection_rows),
        grid_manifest=pd.DataFrame.from_records(grid_rows),
        variance_capture=pd.DataFrame.from_records(variance_rows),
    )


def _random_projection_inner_scores(
    x: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    basis_stack: np.ndarray,
    draws: int,
    maximum_k: int,
    selected_k: int,
    alphas: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, bool]:
    sums = np.zeros((draws, len(alphas)), dtype=np.float64)
    counts = np.zeros((draws, len(alphas)), dtype=np.int64)
    sealed = True
    for validation in _inner_folds(patients, seed):
        training = np.setdiff1d(np.arange(len(patients)), validation, assume_unique=True)
        z_training, z_validation, keep = standardize_genes(x[training], x[validation])
        active = basis_stack[keep]
        projected_training = (z_training @ active).reshape(len(training), draws, maximum_k)
        projected_validation = (z_validation @ active).reshape(len(validation), draws, maximum_k)
        y_mean = np.mean(y[training], axis=0)
        residual = y[training] - y_mean
        for draw in range(draws):
            fit = _standardized_kernel(
                projected_training[:, draw, :selected_k],
                projected_validation[:, draw, :selected_k],
            )
            for alpha_index, alpha in enumerate(alphas):
                prediction = y_mean + _predict_from_kernel(fit, residual, alpha)
                losses = _profile_mse(y[validation], prediction)
                sums[draw, alpha_index] += float(np.sum(losses))
                counts[draw, alpha_index] += len(losses)
        sealed &= np.intersect1d(training, validation).size == 0
    return sums / counts, sealed


def fit_random_matched_kstar_oof(
    data: PlatformData,
    bases: np.ndarray,
    selected_k_by_fold: np.ndarray,
    alphas: Iterable[float],
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fold-local alpha tuning for outcome-blind gene-space orthonormal controls."""
    alpha_values = tuple(float(value) for value in alphas)
    draws, genes, maximum_k = bases.shape
    if genes != data.x.shape[1]:
        raise ValueError("Random basis gene axis does not match baseline RNA")
    if len(selected_k_by_fold) != len(data.patient_ids):
        raise ValueError("One selected k is required for every outer fold")
    basis_stack = bases.transpose(1, 0, 2).reshape(genes, draws * maximum_k)
    predictions = np.full((draws, *data.y.shape), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []

    for held_index, held_patient in enumerate(data.patient_ids):
        training = np.delete(np.arange(len(data.patient_ids)), held_index)
        selected_k = int(selected_k_by_fold[held_index])
        score_matrix, sealed = _random_projection_inner_scores(
            data.x[training],
            data.y[training],
            data.patient_ids[training],
            basis_stack,
            draws,
            maximum_k,
            selected_k,
            alpha_values,
            seed + held_index,
        )
        selected_alpha_indices = np.asarray(
            [
                min(range(len(alpha_values)), key=lambda index: (score_matrix[draw, index], -alpha_values[index]))
                for draw in range(draws)
            ],
            dtype=int,
        )
        z_training, z_held, keep = standardize_genes(data.x[training], data.x[[held_index]])
        active = basis_stack[keep]
        projected_training = (z_training @ active).reshape(len(training), draws, maximum_k)
        projected_held = (z_held @ active).reshape(1, draws, maximum_k)
        y_mean = np.mean(data.y[training], axis=0)
        residual = data.y[training] - y_mean
        for draw in range(draws):
            alpha = alpha_values[selected_alpha_indices[draw]]
            fit = _standardized_kernel(
                projected_training[:, draw, :selected_k],
                projected_held[:, draw, :selected_k],
            )
            predictions[draw, held_index] = y_mean + _predict_from_kernel(fit, residual, alpha)[0]
            fold_rows.append(
                {
                    "draw": draw,
                    "held_patient": str(held_patient),
                    "held_patient_index": held_index,
                    "matched_selected_k": selected_k,
                    "ridge_alpha": alpha,
                    "inner_patient_profile_mse": score_matrix[draw, selected_alpha_indices[draw]],
                    "all_inner_splits_sealed": sealed,
                    "outcome_blind_projection": True,
                    "held_outcome_used_in_selection": False,
                }
            )
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite random matched-k prediction")
    return predictions, pd.DataFrame.from_records(fold_rows)


def bootstrap_primary_contrasts(
    truth: np.ndarray,
    pca_star: np.ndarray,
    full_reopt: np.ndarray,
    population: np.ndarray,
    draws: int,
    seed: int,
    pg_margin: float,
    g_margin: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    pca_core = _core_metrics(truth, pca_star, population, 1e-12)
    full_core = _core_metrics(truth, full_reopt, population, 1e-12)
    pg_difference = pca_core["pg_patient"] - full_core["pg_patient"]
    pca_mse = pca_core["mse_model"]
    full_mse = full_core["mse_model"]
    population_mse = pca_core["mse_population"]
    observed = np.asarray(
        [
            np.mean(pg_difference),
            (np.mean(full_mse) - np.mean(pca_mse)) / np.mean(population_mse),
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(truth), size=(draws, len(truth)))
    bootstrap = np.column_stack(
        [
            np.mean(pg_difference[indices], axis=1),
            (
                np.mean(full_mse[indices], axis=1)
                - np.mean(pca_mse[indices], axis=1)
            )
            / np.mean(population_mse[indices], axis=1),
        ]
    )
    standard_error = np.std(bootstrap, axis=0, ddof=1)
    safe = np.where(standard_error > 0, standard_error, 1.0)
    centered_t = (bootstrap - observed) / safe
    max_abs = np.max(np.abs(centered_t), axis=1)
    critical = float(np.quantile(max_abs, 0.95))
    lower = observed - critical * standard_error
    upper = observed + critical * standard_error
    maximum_t = np.max(centered_t, axis=1)
    rows = []
    for index, (name, margin) in enumerate(
        [
            ("PCA_STAR_minus_FULL_RNA_REOPT_PG", pg_margin),
            ("PCA_STAR_minus_FULL_RNA_REOPT_g_func", g_margin),
        ]
    ):
        observed_t = observed[index] / safe[index]
        adjusted_p = float((1 + np.sum(maximum_t >= observed_t)) / (draws + 1))
        rows.append(
            {
                "contrast": name,
                "estimate": observed[index],
                "bootstrap_se": standard_error[index],
                "simultaneous_95_ci_lower": lower[index],
                "simultaneous_95_ci_upper": upper[index],
                "max_t_critical": critical,
                "max_t_adjusted_one_sided_p": adjusted_p,
                "noninferiority_margin": margin,
                "lower_exceeds_zero": bool(lower[index] > 0),
                "lower_exceeds_negative_margin": bool(lower[index] > -margin),
                "resampling_unit": "patient_profile",
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame.from_records(rows), bootstrap


def bootstrap_performance_curves(
    data: PlatformData,
    predictions: dict[str, np.ndarray],
    population: np.ndarray,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    names = list(predictions)
    metric_names = ("PG_macro", "g_func")
    cores = {
        name: _core_metrics(data.y, predictions[name], population, 1e-12)
        for name in names
    }
    observed = {
        name: {"PG_macro": cores[name]["PG_macro"], "g_func": cores[name]["g_func"]}
        for name in names
    }
    values = np.empty((draws, len(names), len(metric_names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(data.patient_ids), size=(draws, len(data.patient_ids))
    )
    for name_index, name in enumerate(names):
        core = cores[name]
        values[:, name_index, 0] = np.mean(core["pg_patient"][indices], axis=1)
        values[:, name_index, 1] = 1.0 - (
            np.mean(core["mse_model"][indices], axis=1)
            / np.mean(core["mse_population"][indices], axis=1)
        )
    rows = []
    for name_index, name in enumerate(names):
        for metric_index, metric in enumerate(metric_names):
            rows.append(
                {
                    "representation": name,
                    "metric": metric,
                    "estimate": observed[name][metric],
                    "pointwise_95_ci_lower": float(np.quantile(values[:, name_index, metric_index], 0.025)),
                    "pointwise_95_ci_upper": float(np.quantile(values[:, name_index, metric_index], 0.975)),
                    "bootstrap_draws": draws,
                    "resampling_unit": "patient_profile",
                }
            )
    return pd.DataFrame.from_records(rows)


def selection_is_coherent(
    selected_k: np.ndarray,
    mode_fraction_threshold: float,
    maximum_iqr_width: float,
) -> tuple[bool, dict[str, float]]:
    values, counts = np.unique(selected_k.astype(int), return_counts=True)
    mode_index = int(np.argmax(counts))
    mode = int(values[mode_index])
    mode_fraction = float(counts[mode_index] / len(selected_k))
    q25, q75 = np.quantile(selected_k, [0.25, 0.75])
    iqr_width = float(q75 - q25)
    coherent = mode_fraction >= mode_fraction_threshold and iqr_width <= maximum_iqr_width
    return coherent, {
        "mode_k": mode,
        "mode_fraction": mode_fraction,
        "q25_k": float(q25),
        "median_k": float(np.median(selected_k)),
        "q75_k": float(q75),
        "iqr_width": iqr_width,
    }


def deterministic_verdict(
    inference: pd.DataFrame,
    coherent_selection: bool,
    qa_pass: bool,
) -> str:
    if not qa_pass:
        return "PCA_RESOLUTION_AUDIT_NOT_INTERPRETABLE"
    indexed = inference.set_index("contrast")
    pg = indexed.loc["PCA_STAR_minus_FULL_RNA_REOPT_PG"]
    g = indexed.loc["PCA_STAR_minus_FULL_RNA_REOPT_g_func"]
    both_superior = bool(pg["lower_exceeds_zero"] and g["lower_exceeds_zero"])
    both_noninferior = bool(
        pg["lower_exceeds_negative_margin"] and g["lower_exceeds_negative_margin"]
    )
    ranking_only = bool(
        pg["lower_exceeds_zero"] and not g["lower_exceeds_zero"] and not g["lower_exceeds_negative_margin"]
    )
    full_superior = bool(
        pg["simultaneous_95_ci_upper"] < 0
        and g["simultaneous_95_ci_upper"] < 0
        and pg["estimate"] < -pg["noninferiority_margin"]
        and g["estimate"] < -g["noninferiority_margin"]
    )
    if both_superior and coherent_selection:
        return "COARSER_CONDITIONING_RESOLUTION_IMPROVES_TRANSFER"
    if both_noninferior and coherent_selection:
        return "COMPACT_CONDITIONING_REPRESENTATION_IS_SUFFICIENT"
    if ranking_only:
        return "PCA_RESOLUTION_IMPROVES_RANKING_ONLY"
    if full_superior:
        return "FULL_RNA_REMAINS_SUPERIOR"
    return "NO_STABLE_CONDITIONING_RESOLUTION_OPTIMUM"
