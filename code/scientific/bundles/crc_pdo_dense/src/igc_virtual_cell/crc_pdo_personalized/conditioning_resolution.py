from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .modeling import (
    KernelFit,
    PlatformData,
    _core_metrics,
    _inner_folds,
    _kendall_tau_b_rows,
    _predict_from_kernel,
    _standardized_kernel,
)


@dataclass(frozen=True)
class ProgenyMapping:
    pathway_order: tuple[str, ...]
    raw_weights: np.ndarray
    normalized_weights: np.ndarray
    q: np.ndarray
    r: np.ndarray
    rank: int
    mapping_audit: pd.DataFrame
    pathway_summary: pd.DataFrame


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    family: str
    dimensions: int | None = None


@dataclass(frozen=True)
class RepresentationFit:
    train_features: np.ndarray
    test_features: np.ndarray
    retained_genes: int
    residual_rank: int | None = None


@dataclass(frozen=True)
class OOFEvaluation:
    predictions: np.ndarray
    population: np.ndarray
    folds: pd.DataFrame
    coefficients: pd.DataFrame | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    q, r = np.linalg.qr(np.asarray(matrix, dtype=np.float64), mode="reduced")
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0, -1.0, 1.0)
    q = q * signs
    r = signs[:, None] * r
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    return q, r, rank


def build_progeny_mapping(
    feature_ids: np.ndarray,
    feature_symbols: np.ndarray,
    weights_path: Path,
    pathway_order: Iterable[str],
) -> ProgenyMapping:
    pathways = tuple(pathway_order)
    weights = pd.read_csv(weights_path)
    required = {"gene", "pathway", "weight"}
    if not required.issubset(weights.columns):
        raise ValueError(f"PROGENy weights missing columns: {sorted(required - set(weights.columns))}")
    if weights.duplicated(["gene", "pathway"]).any():
        raise ValueError("Ambiguous duplicate PROGENy gene/pathway weights")
    observed_pathways = set(weights["pathway"].astype(str))
    if set(pathways) != observed_pathways:
        raise ValueError(f"Pathway vocabulary mismatch: {sorted(observed_pathways)}")

    symbols = np.asarray(feature_symbols, dtype=str)
    ids = np.asarray(feature_ids, dtype=str)
    if len(np.unique(symbols)) != len(symbols):
        raise ValueError("Baseline feature symbols are not one-to-one")
    weight_lookup = {
        pathway: frame.set_index(frame["gene"].astype(str))["weight"].astype(float).to_dict()
        for pathway, frame in weights.groupby("pathway", sort=False)
    }
    raw = np.zeros((len(symbols), len(pathways)), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for gene_index, (feature_id, symbol) in enumerate(zip(ids, symbols, strict=True)):
        represented = False
        for pathway_index, pathway in enumerate(pathways):
            value = float(weight_lookup[pathway].get(symbol, 0.0))
            raw[gene_index, pathway_index] = value
            represented = represented or value != 0.0
            records.append(
                {
                    "gene_index": gene_index,
                    "feature_id": feature_id,
                    "gene_symbol": symbol,
                    "pathway": pathway,
                    "weight": value,
                    "mapping_status": "EXACT_SYMBOL" if value != 0.0 else "NOT_IN_PATHWAY_MODEL",
                    "ambiguous_mapping": False,
                }
            )
        if not represented:
            continue
    norms = np.linalg.norm(raw, axis=0)
    if np.any(norms <= 0):
        raise ValueError("At least one PROGENy pathway has no mapped nonzero weights")
    normalized = raw / norms
    q, r, rank = _stable_qr(normalized)
    summary = []
    for index, pathway in enumerate(pathways):
        column = raw[:, index]
        summary.append(
            {
                "pathway": pathway,
                "baseline_genes": len(symbols),
                "mapped_nonzero_genes": int(np.sum(column != 0.0)),
                "mapped_fraction": float(np.mean(column != 0.0)),
                "positive_weights": int(np.sum(column > 0)),
                "negative_weights": int(np.sum(column < 0)),
                "raw_l2_norm": float(norms[index]),
                "normalized_l2_norm": float(np.linalg.norm(normalized[:, index])),
            }
        )
    return ProgenyMapping(
        pathway_order=pathways,
        raw_weights=raw,
        normalized_weights=normalized,
        q=q,
        r=r,
        rank=rank,
        mapping_audit=pd.DataFrame.from_records(records),
        pathway_summary=pd.DataFrame.from_records(summary),
    )


def standardize_genes(
    x_train: np.ndarray,
    x_test: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    variance = np.var(x_train, axis=0)
    keep = np.isfinite(variance) & (variance > tolerance)
    if not np.any(keep):
        raise ValueError("No nonzero-variance genes in training split")
    scale = np.sqrt(variance[keep])
    train = (x_train[:, keep] - mean[keep]) / scale
    test = (x_test[:, keep] - mean[keep]) / scale
    return train, test, keep


def _pca_scores_from_gram(
    train: np.ndarray,
    test: np.ndarray,
    components: int,
    subtract_train: np.ndarray | None = None,
    subtract_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    gram = train @ train.T
    cross = test @ train.T
    if subtract_train is not None:
        gram = gram - subtract_train @ subtract_train.T
        if subtract_test is None:
            raise ValueError("subtract_test is required with subtract_train")
        cross = cross - subtract_test @ subtract_train.T
    gram = (gram + gram.T) / 2.0
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    positive = np.flatnonzero(values > 1e-10)
    rank = int(len(positive))
    indices = positive[: min(int(components), rank)]
    if len(indices) == 0:
        return (
            np.zeros((len(train), 0), dtype=np.float64),
            np.zeros((len(test), 0), dtype=np.float64),
            rank,
        )
    singular = np.sqrt(values[indices])
    train_scores = vectors[:, indices] * singular
    test_scores = (cross @ vectors[:, indices]) / singular
    return train_scores, test_scores, rank


def representation_fit(
    x_train: np.ndarray,
    x_test: np.ndarray,
    spec: RepresentationSpec,
    q: np.ndarray,
) -> RepresentationFit:
    if spec.family == "FULL_RNA":
        return RepresentationFit(x_train, x_test, int(np.sum(np.var(x_train, axis=0) > 1e-12)))

    standardized_train, standardized_test, keep = standardize_genes(x_train, x_test)
    q_active = q[keep]
    coarse_train = standardized_train @ q_active
    coarse_test = standardized_test @ q_active
    if spec.family == "PROGENY_SPAN":
        return RepresentationFit(coarse_train, coarse_test, int(np.sum(keep)))
    if spec.family == "PCA":
        train_scores, test_scores, rank = _pca_scores_from_gram(
            standardized_train, standardized_test, int(spec.dimensions or 0)
        )
        return RepresentationFit(train_scores, test_scores, int(np.sum(keep)), rank)
    if spec.family == "COARSE_PLUS_RESIDUAL":
        k = int(spec.dimensions or 0)
        if k == 0:
            return RepresentationFit(coarse_train, coarse_test, int(np.sum(keep)), 0)
        residual_train, residual_test, rank = _pca_scores_from_gram(
            standardized_train,
            standardized_test,
            k,
            subtract_train=coarse_train,
            subtract_test=coarse_test,
        )
        return RepresentationFit(
            np.column_stack([coarse_train, residual_train]),
            np.column_stack([coarse_test, residual_test]),
            int(np.sum(keep)),
            rank,
        )
    if spec.family == "RESIDUAL_ONLY":
        residual_train, residual_test, rank = _pca_scores_from_gram(
            standardized_train,
            standardized_test,
            int(spec.dimensions or 0),
            subtract_train=coarse_train,
            subtract_test=coarse_test,
        )
        return RepresentationFit(residual_train, residual_test, int(np.sum(keep)), rank)
    raise ValueError(f"Unknown representation family {spec.family}")


def _select_alpha_for_representation(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    spec: RepresentationSpec,
    q: np.ndarray,
    alphas: Iterable[float],
    seed: int,
) -> tuple[float, dict[float, float], bool]:
    alpha_values = [float(value) for value in alphas]
    score_lists = {value: [] for value in alpha_values}
    all_inner_sealed = True
    for val_indices in _inner_folds(patient_ids, seed):
        train_indices = np.setdiff1d(np.arange(len(patient_ids)), val_indices, assume_unique=True)
        transformed = representation_fit(x[train_indices], x[val_indices], spec, q)
        fit = _standardized_kernel(transformed.train_features, transformed.test_features)
        y_mean = np.mean(y[train_indices], axis=0)
        residual = y[train_indices] - y_mean
        for alpha in alpha_values:
            prediction = y_mean + _predict_from_kernel(fit, residual, alpha)
            score_lists[alpha].extend(_kendall_tau_b_rows(y[val_indices], prediction).tolist())
        all_inner_sealed &= not np.intersect1d(train_indices, val_indices).size
    scores = {alpha: float(np.nanmean(values)) for alpha, values in score_lists.items()}
    selected = max(alpha_values, key=lambda alpha: (scores[alpha], alpha))
    return selected, scores, all_inner_sealed


def _standardized_primal_coefficient(
    train_features: np.ndarray,
    y_residual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = np.mean(train_features, axis=0)
    variance = np.var(train_features, axis=0)
    keep = variance > 1e-12
    if not np.all(keep):
        raise ValueError("PROGENy-span coordinate became constant")
    scale = np.sqrt(variance)
    z = ((train_features - mean) / scale) / math.sqrt(train_features.shape[1])
    dual = np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y_residual)
    standardized_beta = z.T @ dual
    return standardized_beta / (scale[:, None] * math.sqrt(train_features.shape[1]))


def fit_representation_oof(
    data: PlatformData,
    spec: RepresentationSpec,
    q: np.ndarray,
    r: np.ndarray,
    pathway_order: tuple[str, ...],
    alphas: Iterable[float],
    seed: int,
) -> OOFEvaluation:
    predictions = np.full_like(data.y, np.nan, dtype=np.float64)
    population = np.full_like(data.y, np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    for held_index in range(len(data.patient_ids)):
        train_indices = np.delete(np.arange(len(data.patient_ids)), held_index)
        x_train = data.x[train_indices]
        y_train = data.y[train_indices]
        train_patients = data.patient_ids[train_indices]
        alpha, alpha_scores, inner_sealed = _select_alpha_for_representation(
            x_train,
            y_train,
            train_patients,
            spec,
            q,
            alphas,
            seed + held_index,
        )
        transformed = representation_fit(x_train, data.x[[held_index]], spec, q)
        fit = _standardized_kernel(transformed.train_features, transformed.test_features)
        y_mean = np.mean(y_train, axis=0)
        residual = y_train - y_mean
        population[held_index] = y_mean
        predictions[held_index] = y_mean + _predict_from_kernel(fit, residual, alpha)[0]
        fold_records.append(
            {
                "platform": data.platform,
                "representation": spec.name,
                "held_patient": str(data.patient_ids[held_index]),
                "held_patient_index": held_index,
                "training_patients": len(train_indices),
                "patient_overlap": 0,
                "held_outcome_used_in_transform": False,
                "inner_preprocessing_refit": True,
                "all_inner_splits_sealed": inner_sealed,
                "retained_genes": transformed.retained_genes,
                "representation_dimensions": transformed.train_features.shape[1],
                "available_residual_rank": transformed.residual_rank,
                "ridge_alpha": alpha,
                "ridge_inner_tau": alpha_scores[alpha],
            }
        )
        if spec.name == "PROGENY14":
            coefficient_q = _standardized_primal_coefficient(transformed.train_features, residual, alpha)
            coefficient_pathway = np.linalg.solve(r, coefficient_q)
            for pathway_index, pathway in enumerate(pathway_order):
                for drug_index, drug in enumerate(data.drug_names):
                    coefficient_records.append(
                        {
                            "platform": data.platform,
                            "held_patient": str(data.patient_ids[held_index]),
                            "pathway": pathway,
                            "drug_name": str(drug),
                            "coefficient": float(coefficient_pathway[pathway_index, drug_index]),
                            "coordinate_system": "normalized_signed_PROGENy_weight_columns",
                        }
                    )
    if not np.isfinite(predictions).all() or not np.isfinite(population).all():
        raise ValueError(f"Non-finite OOF output for {data.platform}/{spec.name}")
    coefficients = pd.DataFrame.from_records(coefficient_records) if coefficient_records else None
    return OOFEvaluation(predictions, population, pd.DataFrame.from_records(fold_records), coefficients)


def random_orthonormal_bases(genes: int, dimensions: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bases = np.empty((draws, genes, dimensions), dtype=np.float64)
    for draw in range(draws):
        q, _, rank = _stable_qr(rng.normal(size=(genes, dimensions)))
        if rank != dimensions:
            raise ValueError("Random projection was rank deficient")
        bases[draw] = q
    return bases


def random_pathway_bases(weights: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    genes, dimensions = weights.shape
    bases = np.empty((draws, genes, dimensions), dtype=np.float64)
    for draw in range(draws):
        permuted = np.zeros_like(weights)
        for column_index in range(dimensions):
            values = weights[:, column_index]
            nonzero_values = values[values != 0.0]
            destinations = rng.permutation(genes)[: len(nonzero_values)]
            permuted[destinations, column_index] = nonzero_values
        q, _, rank = _stable_qr(permuted)
        if rank != dimensions:
            raise ValueError("Random pathway vocabulary was rank deficient")
        bases[draw] = q
    return bases


def _random_family_inner_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    patients: np.ndarray,
    bases: np.ndarray,
    alphas: list[float],
    seed: int,
) -> tuple[np.ndarray, bool]:
    draws, _, dimensions = bases.shape
    basis_stack = bases.transpose(1, 0, 2).reshape(bases.shape[1], draws * dimensions)
    scores = np.zeros((draws, len(alphas)), dtype=np.float64)
    counts = np.zeros((draws, len(alphas)), dtype=np.int64)
    sealed = True
    for val_indices in _inner_folds(patients, seed):
        train_indices = np.setdiff1d(np.arange(len(patients)), val_indices, assume_unique=True)
        z_train_gene, z_val_gene, _ = standardize_genes(x_train[train_indices], x_train[val_indices])
        # All bases are defined on the full gene axis; restore the active rows.
        variance = np.var(x_train[train_indices], axis=0)
        keep = np.isfinite(variance) & (variance > 1e-12)
        active_stack = basis_stack[keep]
        projected_train = (z_train_gene @ active_stack).reshape(len(train_indices), draws, dimensions)
        projected_val = (z_val_gene @ active_stack).reshape(len(val_indices), draws, dimensions)
        y_mean = np.mean(y_train[train_indices], axis=0)
        residual = y_train[train_indices] - y_mean
        for draw in range(draws):
            fit = _standardized_kernel(projected_train[:, draw], projected_val[:, draw])
            for alpha_index, alpha in enumerate(alphas):
                prediction = y_mean + _predict_from_kernel(fit, residual, alpha)
                tau = _kendall_tau_b_rows(y_train[val_indices], prediction)
                scores[draw, alpha_index] += np.nansum(tau)
                counts[draw, alpha_index] += np.sum(np.isfinite(tau))
        sealed &= not np.intersect1d(train_indices, val_indices).size
    return np.divide(scores, counts, out=np.full_like(scores, np.nan), where=counts > 0), sealed


def fit_random_family_oof(
    data: PlatformData,
    bases: np.ndarray,
    alphas: Iterable[float],
    seed: int,
    family: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    alpha_values = [float(value) for value in alphas]
    draws, _, dimensions = bases.shape
    predictions = np.full((draws, len(data.patient_ids), len(data.drug_names)), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    basis_stack = bases.transpose(1, 0, 2).reshape(bases.shape[1], draws * dimensions)
    for held_index in range(len(data.patient_ids)):
        train_indices = np.delete(np.arange(len(data.patient_ids)), held_index)
        x_train = data.x[train_indices]
        y_train = data.y[train_indices]
        score_matrix, sealed = _random_family_inner_scores(
            x_train,
            y_train,
            data.patient_ids[train_indices],
            bases,
            alpha_values,
            seed + held_index,
        )
        selected_indices = np.asarray(
            [max(range(len(alpha_values)), key=lambda index: (score_matrix[draw, index], alpha_values[index])) for draw in range(draws)],
            dtype=int,
        )
        z_train_gene, z_test_gene, keep = standardize_genes(x_train, data.x[[held_index]])
        active_stack = basis_stack[keep]
        projected_train = (z_train_gene @ active_stack).reshape(len(train_indices), draws, dimensions)
        projected_test = (z_test_gene @ active_stack).reshape(1, draws, dimensions)
        y_mean = np.mean(y_train, axis=0)
        residual = y_train - y_mean
        for draw in range(draws):
            alpha = alpha_values[selected_indices[draw]]
            fit = _standardized_kernel(projected_train[:, draw], projected_test[:, draw])
            predictions[draw, held_index] = y_mean + _predict_from_kernel(fit, residual, alpha)[0]
            fold_records.append(
                {
                    "family": family,
                    "draw": draw,
                    "held_patient": str(data.patient_ids[held_index]),
                    "patient_overlap": 0,
                    "held_outcome_used_in_transform": False,
                    "inner_preprocessing_refit": True,
                    "all_inner_splits_sealed": sealed,
                    "dimensions": dimensions,
                    "ridge_alpha": alpha,
                    "ridge_inner_tau": score_matrix[draw, selected_indices[draw]],
                }
            )
    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite random-family output for {family}")
    return predictions, pd.DataFrame.from_records(fold_records)


def pooled_reversal_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    population: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    from .modeling import PAIR_I, PAIR_J

    truth_difference = truth[:, PAIR_I] - truth[:, PAIR_J]
    prediction_difference = prediction[:, PAIR_I] - prediction[:, PAIR_J]
    population_difference = population[:, PAIR_I] - population[:, PAIR_J]
    truth_sign = np.where(np.abs(truth_difference) <= tolerance, 0, np.sign(truth_difference))
    prediction_sign = np.where(np.abs(prediction_difference) <= tolerance, 0, np.sign(prediction_difference))
    population_sign = np.where(np.abs(population_difference) <= tolerance, 0, np.sign(population_difference))
    valid = (truth_sign != 0) & (population_sign != 0)
    actual = valid & (truth_sign == -population_sign)
    predicted = valid & (prediction_sign == -population_sign)
    negative = valid & ~actual
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(negative & predicted))
    tn = int(np.sum(negative & ~predicted))
    recall = tp / (tp + fn) if tp + fn else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "reversal_recall": recall,
        "reversal_precision": precision,
        "reversal_balanced_accuracy_pooled": (recall + specificity) / 2.0,
        "reversal_mcc": (tp * tn - fp * fn) / denominator if denominator else math.nan,
    }


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    population: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    core = _core_metrics(truth, prediction, population, tolerance)
    spearman_model = np.nanmean(
        [spearmanr(truth[index], prediction[index]).statistic for index in range(len(truth))]
    )
    spearman_population = np.nanmean(
        [spearmanr(truth[index], population[index]).statistic for index in range(len(truth))]
    )
    return {
        "population_kendall_tau_b": float(np.nanmean(core["tau_population"])),
        "personalized_kendall_tau_b": float(np.nanmean(core["tau_model"])),
        "population_spearman_rho": float(spearman_population),
        "personalized_spearman_rho": float(spearman_model),
        "PG_macro": float(core["PG_macro"]),
        "g_func": float(core["g_func"]),
        "reversal_balanced_accuracy_patient_mean": float(core["reversal_balanced_accuracy"]),
        **pooled_reversal_metrics(truth, prediction, population, tolerance),
    }


def oof_table(
    data: PlatformData,
    representation: str,
    prediction: np.ndarray,
    population: np.ndarray,
) -> pd.DataFrame:
    records = []
    for patient_index, patient_id in enumerate(data.patient_ids):
        for drug_index, drug in enumerate(data.drug_names):
            records.append(
                {
                    "platform": data.platform,
                    "representation": representation,
                    "patient_id": str(patient_id),
                    "drug_name": str(drug),
                    "true_DSS": float(data.y[patient_index, drug_index]),
                    "predicted_DSS": float(prediction[patient_index, drug_index]),
                    "population_DSS": float(population[patient_index, drug_index]),
                    "predicted_personalized_residual": float(
                        prediction[patient_index, drug_index] - population[patient_index, drug_index]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def bootstrap_primary_inference(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    population: np.ndarray,
    draws: int,
    seed: int,
    pg_margin: float,
    g_margin: float,
    fine_levels: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    level_names = [f"COARSE_R{level}" for level in fine_levels if int(level) > 0]
    primary_names = [
        "PROGENY14_minus_FULL_RNA_PG",
        "PROGENY14_minus_FULL_RNA_g_func",
        "PROGENY14_minus_PCA14_PG",
        *[f"{name}_minus_R0_PG" for name in level_names],
    ]
    secondary_names = [f"{name}_minus_R0_g_func" for name in level_names]
    contrast_names = [*primary_names, *secondary_names]

    def estimates(indices: np.ndarray) -> np.ndarray:
        local_truth = truth[indices]
        local_population = population[indices]
        bundles = {
            name: metrics(local_truth, prediction[indices], local_population)
            for name, prediction in predictions.items()
        }
        return np.asarray(
            [
                bundles["PROGENY14"]["PG_macro"] - bundles["FULL_RNA"]["PG_macro"],
                bundles["PROGENY14"]["g_func"] - bundles["FULL_RNA"]["g_func"],
                bundles["PROGENY14"]["PG_macro"] - bundles["PCA14"]["PG_macro"],
                *[
                    bundles[name]["PG_macro"] - bundles["PROGENY14"]["PG_macro"]
                    for name in level_names
                ],
                *[
                    bundles[name]["g_func"] - bundles["PROGENY14"]["g_func"]
                    for name in level_names
                ],
            ]
        )

    observed = estimates(np.arange(len(truth)))
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(observed)), dtype=np.float64)
    for draw in range(draws):
        boot[draw] = estimates(rng.integers(0, len(truth), size=len(truth)))
    standard_error = np.std(boot, axis=0, ddof=1)
    primary_count = len(primary_names)
    lower = np.empty_like(observed)
    upper = np.empty_like(observed)
    critical_values = np.empty_like(observed)
    for indices in (np.arange(primary_count), np.arange(primary_count, len(observed))):
        family_se = standard_error[indices]
        safe_se = np.where(family_se > 0, family_se, 1.0)
        max_t = np.max(np.abs((boot[:, indices] - observed[indices]) / safe_se), axis=1)
        critical = float(np.quantile(max_t, 0.95))
        lower[indices] = observed[indices] - critical * family_se
        upper[indices] = observed[indices] + critical * family_se
        critical_values[indices] = critical
    rows = []
    for index, name in enumerate(contrast_names):
        margin = pg_margin if name == "PROGENY14_minus_FULL_RNA_PG" else g_margin if name == "PROGENY14_minus_FULL_RNA_g_func" else 0.0
        rows.append(
            {
                "contrast": name,
                "estimate": observed[index],
                "bootstrap_se": standard_error[index],
                "simultaneous_95_ci_lower": lower[index],
                "simultaneous_95_ci_upper": upper[index],
                "max_t_critical": critical_values[index],
                "inference_family": "PRIMARY_7" if index < primary_count else "SECONDARY_FINE_G_4",
                "noninferiority_margin": margin,
                "lower_exceeds_negative_margin": bool(lower[index] > -margin),
                "lower_exceeds_zero": bool(lower[index] > 0.0),
                "bootstrap_draws": draws,
                "resampling_unit": "patient_profile",
            }
        )
    return pd.DataFrame.from_records(rows), {"observed": observed, "bootstrap": boot, "lower": lower, "upper": upper}


def random_family_summary(
    data: PlatformData,
    predictions: np.ndarray,
    population: np.ndarray,
    family: str,
) -> pd.DataFrame:
    rows = []
    for draw in range(len(predictions)):
        rows.append({"family": family, "draw": draw, **metrics(data.y, predictions[draw], population)})
    return pd.DataFrame.from_records(rows)


def deterministic_verdict(
    inference: pd.DataFrame,
    performance: pd.DataFrame,
    random14: pd.DataFrame,
    random_pathway: pd.DataFrame,
    random_pathway_p: float,
    material_fine_pg: float,
    material_fine_g: float,
) -> tuple[str, dict[str, bool]]:
    indexed = inference.set_index("contrast")
    perf = performance.set_index("representation")
    retain_pg = bool(indexed.loc["PROGENY14_minus_FULL_RNA_PG", "lower_exceeds_negative_margin"])
    retain_g = bool(indexed.loc["PROGENY14_minus_FULL_RNA_g_func", "lower_exceeds_negative_margin"])
    pathway_specific = bool(random_pathway_p <= 0.05)
    fine_rows = indexed.loc[
        [name for name in indexed.index if name.startswith("COARSE_R") and name.endswith("_PG")]
    ]
    fine_pg = bool(
        np.any(
            (fine_rows["estimate"].to_numpy(float) >= material_fine_pg)
            & fine_rows["lower_exceeds_zero"].to_numpy(bool)
        )
    )
    fine_g_rows = indexed.loc[
        [name for name in indexed.index if name.startswith("COARSE_R") and name.endswith("_g_func")]
    ]
    fine_g = bool(
        np.any(
            (fine_g_rows["estimate"].to_numpy(float) >= material_fine_g)
            & fine_g_rows["lower_exceeds_zero"].to_numpy(bool)
        )
    )
    prog_full_positive = bool(indexed.loc["PROGENY14_minus_FULL_RNA_PG", "lower_exceeds_zero"])
    prog_pca_positive = bool(indexed.loc["PROGENY14_minus_PCA14_PG", "lower_exceeds_zero"])
    random14_central = bool(
        np.quantile(random14.PG_macro, 0.025)
        <= perf.loc["PROGENY14", "PG_macro"]
        <= np.quantile(random14.PG_macro, 0.975)
    )
    pca_equivalent = bool(
        indexed.loc["PROGENY14_minus_PCA14_PG", "simultaneous_95_ci_lower"] <= 0
        <= indexed.loc["PROGENY14_minus_PCA14_PG", "simultaneous_95_ci_upper"]
    )
    gates = {
        "retains_PG": retain_pg,
        "retains_g_func": retain_g,
        "exceeds_random_pathway_on_PG": pathway_specific,
        "material_fine_PG_gain": fine_pg,
        "material_fine_g_func_gain_descriptive": fine_g,
        "PROGENY14_PG_exceeds_FULL": prog_full_positive,
        "PROGENY14_PG_exceeds_PCA14": prog_pca_positive,
        "PROGENY14_within_RANDOM14_central95": random14_central,
        "PROGENY14_minus_PCA14_CI_includes_zero": pca_equivalent,
    }
    if retain_pg and retain_g and pathway_specific and not fine_pg:
        if prog_full_positive and prog_pca_positive:
            return "COARSE_BIOLOGICAL_REPRESENTATION_SUPERIOR", gates
        return "CONDITIONING_RESOLUTION_MATCHING_SUPPORTED", gates
    if fine_pg or fine_g:
        return "FINE_BASELINE_DETAIL_ADDS_PREDICTIVE_VALUE", gates
    if not retain_pg or not retain_g:
        return "COARSE_REPRESENTATION_LOSES_PERSONALIZATION", gates
    if (not pathway_specific) and pca_equivalent and random14_central:
        return "GENERIC_DIMENSIONALITY_REDUCTION_EXPLAINS_COARSE_ADVANTAGE", gates
    return "CONDITIONING_RESOLUTION_AUDIT_NOT_INTERPRETABLE", gates
