"""Training-only PCA or signed-NMF perturbation-response programs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, PCA


def _signed_features(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.maximum(values, 0), np.maximum(-values, 0)], axis=1)


@dataclass
class ResponseProgramModel:
    method: str
    genes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    estimator: PCA | NMF

    @classmethod
    def fit(
        cls,
        response_matrix: pd.DataFrame,
        *,
        method: str = "pca",
        n_components: int = 10,
        standardize: bool = True,
        random_state: int = 0,
    ) -> "ResponseProgramModel":
        if response_matrix.empty:
            raise ValueError("Cannot fit response programs to an empty matrix")
        values = response_matrix.to_numpy(dtype=float)
        mean = values.mean(axis=0) if standardize else np.zeros(values.shape[1])
        scale = values.std(axis=0, ddof=0) if standardize else np.ones(values.shape[1])
        scale[scale == 0] = 1.0
        standardized = (values - mean) / scale
        normalized_method = method.strip().lower()
        max_components = min(values.shape[0], values.shape[1])
        components = min(int(n_components), max_components)
        if components < 1:
            raise ValueError("n_components must be positive")
        if normalized_method == "pca":
            estimator: PCA | NMF = PCA(
                n_components=components, svd_solver="full", random_state=random_state
            ).fit(standardized)
        elif normalized_method == "nmf":
            estimator = NMF(
                n_components=components,
                init="nndsvda",
                max_iter=1000,
                random_state=random_state,
            ).fit(_signed_features(standardized))
        else:
            raise ValueError("method must be 'pca' or 'nmf'")
        return cls(normalized_method, tuple(map(str, response_matrix.columns)), mean, scale, estimator)

    def transform(self, response_matrix: pd.DataFrame) -> pd.DataFrame:
        standardized = (
            response_matrix.loc[:, self.genes].to_numpy(dtype=float) - self.mean
        ) / self.scale
        features = standardized if self.method == "pca" else _signed_features(standardized)
        scores = self.estimator.transform(features)
        columns = [f"response_program_{index + 1}" for index in range(scores.shape[1])]
        return pd.DataFrame(scores, index=response_matrix.index, columns=columns)

    def loadings(self) -> pd.DataFrame:
        components = self.estimator.components_
        if self.method == "nmf":
            gene_count = len(self.genes)
            components = components[:, :gene_count] - components[:, gene_count:]
        columns = list(self.genes)
        index = [f"response_program_{number + 1}" for number in range(components.shape[0])]
        return pd.DataFrame(components, index=index, columns=columns)

    def reconstruct(self, scores: pd.DataFrame) -> pd.DataFrame:
        score_values = scores.to_numpy(dtype=float)
        if self.method == "pca":
            standardized = self.estimator.inverse_transform(score_values)
        else:
            signed = self.estimator.inverse_transform(score_values)
            gene_count = len(self.genes)
            standardized = signed[:, :gene_count] - signed[:, gene_count:]
        values = standardized * self.scale + self.mean
        return pd.DataFrame(values, index=scores.index, columns=self.genes)


def reconstruction_fraction(
    model: ResponseProgramModel, response_matrix: pd.DataFrame
) -> float:
    """Return one minus normalized squared reconstruction error."""
    reconstructed = model.reconstruct(model.transform(response_matrix)).to_numpy()
    truth = response_matrix.loc[:, model.genes].to_numpy(dtype=float)
    denominator = np.square(truth - truth.mean(axis=0)).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.square(truth - reconstructed).sum() / denominator)


def program_stability(
    response_matrix: pd.DataFrame,
    *,
    method: str,
    n_components: int,
    n_bootstraps: int = 20,
    random_state: int = 0,
) -> pd.DataFrame:
    """Estimate component stability by matched absolute loading correlation."""
    if len(response_matrix) < 3:
        return pd.DataFrame(columns=["bootstrap", "mean_matched_abs_correlation"])
    reference = ResponseProgramModel.fit(
        response_matrix,
        method=method,
        n_components=n_components,
        random_state=random_state,
    ).loadings().to_numpy()
    rng = np.random.default_rng(random_state)
    records: list[dict[str, float | int]] = []
    for bootstrap in range(n_bootstraps):
        indices = rng.integers(0, len(response_matrix), size=len(response_matrix))
        sampled = response_matrix.iloc[indices].reset_index(drop=True)
        candidate = ResponseProgramModel.fit(
            sampled,
            method=method,
            n_components=n_components,
            random_state=random_state + bootstrap + 1,
        ).loadings().to_numpy()
        correlations = np.corrcoef(reference, candidate)[: len(reference), len(reference) :]
        correlations = np.nan_to_num(np.abs(correlations), nan=0.0)
        left, right = linear_sum_assignment(-correlations)
        records.append(
            {
                "bootstrap": bootstrap,
                "mean_matched_abs_correlation": float(correlations[left, right].mean()),
            }
        )
    return pd.DataFrame.from_records(records)

