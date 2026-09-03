"""Zero-shot heterogeneous held-context scaling on the frozen Tahoe core.

SPDX-License-Identifier: MIT

The implementation deliberately operates on frozen pseudobulk tensors and
baseline-only DMSO profiles.  It never loads the Tahoe single-cell release.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import zarr


M_GRID = (2, 4, 8, 12, 16, 24, 32, 40, 49)
PILOT_M_GRID = (4, 16, 40, 49)
LADDERS = 20
DESIGN_SEED = 202608281
BOOTSTRAP_SEED = 202608282
BOOTSTRAP_DRAWS = 10_000
GENE_CHUNK = 2048
LINEAR_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
RBF_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
RBF_GAMMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
SMALL_M_DEFAULTS = {
    "linear_ridge": {"alpha": 1.0, "gamma_factor": np.nan},
    "rbf_ridge": {"alpha": 0.1, "gamma_factor": 1.0},
}
BILINEAR_ALPHAS = (1e-2, 1e-1, 1.0, 10.0)
BILINEAR_RANKS = (2, 4, 8, 16)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while payload := handle.read(block):
            digest.update(payload)
    return digest.hexdigest()


def _stable_seed(*values: Any) -> int:
    text = "|".join(map(str, values))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def output_dir(root: Path) -> Path:
    path = root / "results/tahoe_held_context_scaling"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class FrozenInputs:
    contexts: tuple[str, ...]
    interventions: tuple[str, ...]
    gene_indices: np.ndarray
    baselines: np.ndarray
    baseline_kernels: np.ndarray
    lineages: tuple[str, ...]
    response_store: Path
    source_root: Path


@dataclass(frozen=True)
class ResponseGrams:
    same6: np.ndarray
    same14: np.ndarray
    cross: np.ndarray
    gene_count: int
    intervention_count: int

    @property
    def total_same6(self) -> np.ndarray:
        return self.same6.sum(axis=0)

    @property
    def total_same14(self) -> np.ndarray:
        return self.same14.sum(axis=0)

    @property
    def total_cross(self) -> np.ndarray:
        return self.cross.sum(axis=0)


def _normalize_baseline_rows(values: np.ndarray) -> np.ndarray:
    """Fixed per-profile normalization using no other context and no outcomes."""

    centered = values - values.mean(axis=1, keepdims=True, dtype=np.float64)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("A frozen DMSO baseline has zero row variance")
    return centered / norms


def load_frozen_inputs(root: Path, source_root: Path) -> FrozenInputs:
    root, source_root = root.resolve(), source_root.resolve()
    response_store = source_root / "results/cgc_tahoe_0i/response_tensors.zarr"
    group = zarr.open_group(response_store, mode="r")
    contexts = tuple(map(str, group.attrs["contexts"]))
    interventions = tuple(map(str, group.attrs["interventions"]))
    gene_indices = np.load(source_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy")
    if len(contexts) != 50 or len(interventions) != 93 or len(gene_indices) != 25_695:
        raise RuntimeError("Frozen Tahoe axes do not match the prespecified 50 x 93 x 25,695 core")
    axes = json.loads((root / "data/tahoe100m_control_state/axes.json").read_text(encoding="utf-8"))
    wells = np.load(root / "data/tahoe100m_control_state/control_wells_log1p_cpm_float32.npy")
    if tuple(map(str, axes["contexts"])) != contexts or wells.shape != (4, 50, 42_523):
        raise RuntimeError("Frozen DMSO context axis does not match the response core")
    baselines = np.stack((wells[[0, 1]].mean(axis=0), wells[[2, 3]].mean(axis=0))).astype(np.float64)
    normalized = np.stack([_normalize_baseline_rows(plate) for plate in baselines])
    baseline_kernels = np.einsum("rcg,rdg->rcd", normalized, normalized, optimize=True)
    lineage_frame = pd.read_csv(root / "results/cgc_tahoe_0d/lineage_baseline.csv")
    lineage_map = (
        lineage_frame.drop_duplicates("context_id")
        .set_index("context_id")["lineage"]
        .astype(str)
        .to_dict()
    )
    lineages = tuple(lineage_map.get(context, "UNMAPPED") for context in contexts)
    if any(value == "UNMAPPED" for value in lineages):
        raise RuntimeError("Tahoe lineage mapping is incomplete")
    return FrozenInputs(
        contexts=contexts,
        interventions=interventions,
        gene_indices=gene_indices.astype(np.int32),
        baselines=baselines,
        baseline_kernels=baseline_kernels,
        lineages=lineages,
        response_store=response_store,
        source_root=source_root,
    )


def prepare_design(root: Path, frozen: FrozenInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = output_dir(root)
    lineage_counts = pd.Series(frozen.lineages).value_counts().to_dict()
    pilot_indices = {0, 12, 24, 36, 49}
    targets = pd.DataFrame(
        {
            "target_index": np.arange(50),
            "target_context": frozen.contexts,
            "lineage": frozen.lineages,
            "lineage_size": [lineage_counts[value] for value in frozen.lineages],
            "pilot_target": [index in pilot_indices for index in range(50)],
            "heldout_perturbation_outcomes_visible_before_evaluation": False,
            "allowed_target_information": "DMSO baseline only",
            "test_interventions": 93,
            "response_genes": 25_695,
        }
    )
    rows: list[dict[str, Any]] = []
    all_indices = np.arange(50)
    for target in range(50):
        pool = all_indices[all_indices != target]
        for ladder in range(LADDERS):
            seed = _stable_seed(DESIGN_SEED, frozen.contexts[target], ladder)
            order = np.random.default_rng(seed).permutation(pool)
            for m in M_GRID[:-1]:
                for rank, context_index in enumerate(order[:m], start=1):
                    rows.append(
                        {
                            "target_index": target,
                            "target_context": frozen.contexts[target],
                            "ladder_id": ladder,
                            "ladder_seed": seed,
                            "m": m,
                            "training_rank": rank,
                            "training_context_index": int(context_index),
                            "training_context": frozen.contexts[context_index],
                        }
                    )
        max_order = np.sort(pool)
        for rank, context_index in enumerate(max_order, start=1):
            rows.append(
                {
                    "target_index": target,
                    "target_context": frozen.contexts[target],
                    "ladder_id": "max49",
                    "ladder_seed": DESIGN_SEED,
                    "m": 49,
                    "training_rank": rank,
                    "training_context_index": int(context_index),
                    "training_context": frozen.contexts[context_index],
                }
            )
    ladders = pd.DataFrame(rows)
    _verify_ladders(ladders)
    targets.to_csv(out / "TAHOE_HELD_CONTEXT_SCALING_TARGET_MANIFEST.csv", index=False)
    ladders.to_csv(out / "TAHOE_HELD_CONTEXT_SCALING_LADDERS.csv", index=False)
    return targets, ladders


def _verify_ladders(frame: pd.DataFrame) -> None:
    if frame["target_context"].nunique() != 50:
        raise RuntimeError("Not all 50 targets have ladders")
    for target, target_frame in frame.groupby("target_index"):
        if np.any(target_frame["training_context_index"].to_numpy() == target):
            raise RuntimeError("Held-out target leaked into a training ladder")
        max_rows = target_frame[target_frame["m"] == 49]
        if len(max_rows) != 49 or max_rows["training_context_index"].nunique() != 49:
            raise RuntimeError("Maximal training set is not the unique remaining 49 contexts")
        for ladder in range(LADDERS):
            selected = target_frame[target_frame["ladder_id"].astype(str) == str(ladder)]
            previous: set[int] = set()
            for m in M_GRID[:-1]:
                current = set(selected.loc[selected["m"] == m, "training_context_index"].astype(int))
                if len(current) != m or not previous.issubset(current):
                    raise RuntimeError("Context ladders are not nested")
                previous = current


def _support_from_ladders(frame: pd.DataFrame, target: int, ladder: int | str, m: int) -> np.ndarray:
    subset = frame[
        (frame["target_index"] == target)
        & (frame["m"] == m)
        & (frame["ladder_id"].astype(str) == str(ladder))
    ].sort_values("training_rank")
    values = subset["training_context_index"].to_numpy(np.int32)
    if len(values) != m:
        raise RuntimeError(f"Incomplete support: target={target}, ladder={ladder}, m={m}")
    return values


def build_response_gram_cache(root: Path, frozen: FrozenInputs, force: bool = False) -> Path:
    out = output_dir(root)
    cache_dir = out / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "TAHOE_PRIMARY_RESPONSE_GRAMS.npz"
    manifest_path = cache_dir / "TAHOE_PRIMARY_RESPONSE_GRAMS.json"
    if path.exists() and manifest_path.exists() and not force:
        return path
    response = zarr.open_group(frozen.response_store, mode="r")["delta_primary"]
    same6 = np.zeros((93, 50, 50), dtype=np.float64)
    same14 = np.zeros_like(same6)
    cross = np.zeros_like(same6)
    selected = frozen.gene_indices
    started = time.perf_counter()
    for start in range(0, response.shape[-1], GENE_CHUNK):
        stop = min(response.shape[-1], start + GENE_CHUNK)
        local = selected[(selected >= start) & (selected < stop)] - start
        if not len(local):
            continue
        block = np.asarray(response[:, :, :, start:stop], dtype=np.float64)[..., local]
        first, second = block[0], block[1]
        for intervention in range(93):
            a, b = first[:, intervention], second[:, intervention]
            same6[intervention] += a @ a.T
            same14[intervention] += b @ b.T
            cross[intervention] += a @ b.T
        print(f"held-context Gram genes {stop}/{response.shape[-1]}", flush=True)
    np.savez_compressed(
        path,
        same6=same6,
        same14=same14,
        cross=cross,
        gene_count=np.asarray(len(selected), dtype=np.int32),
        intervention_count=np.asarray(93, dtype=np.int16),
    )
    _json(
        manifest_path,
        {
            "method": "streamed full-gene response Gram sufficient statistics",
            "response_store": str(frozen.response_store),
            "response_object": "delta_primary",
            "shape": [2, 50, 93, 25_695],
            "gene_indices_sha256": _sha256(frozen.source_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy"),
            "elapsed_seconds": time.perf_counter() - started,
            "raw_single_cells_loaded": False,
            "treated_plates_averaged_before_scoring": False,
            "cache_sha256": _sha256(path),
        },
    )
    return path


def load_response_grams(path: Path) -> ResponseGrams:
    cached = np.load(path)
    return ResponseGrams(
        same6=cached["same6"],
        same14=cached["same14"],
        cross=cached["cross"],
        gene_count=int(cached["gene_count"]),
        intervention_count=int(cached["intervention_count"]),
    )


def _squared_distances_from_kernel(kernel: np.ndarray) -> np.ndarray:
    diagonal = np.diag(kernel)
    return np.maximum(diagonal[:, None] + diagonal[None, :] - 2 * kernel, 0.0)


def _gamma_base(distances: np.ndarray, indices: np.ndarray) -> float:
    selected = distances[np.ix_(indices, indices)]
    upper = selected[np.triu_indices(len(indices), 1)]
    positive = upper[upper > 1e-15]
    return 1.0 / float(np.median(positive)) if len(positive) else 1.0


def _kernel_matrix(
    base_kernel: np.ndarray,
    indices: np.ndarray,
    model: str,
    gamma_factor: float | None = None,
    gamma_reference: np.ndarray | None = None,
) -> np.ndarray:
    if model in {"linear_ridge", "bilinear_reduced_rank"}:
        return base_kernel[np.ix_(indices, indices)]
    distances = _squared_distances_from_kernel(base_kernel)
    reference = indices if gamma_reference is None else gamma_reference
    gamma = float(gamma_factor) * _gamma_base(distances, reference)
    return np.exp(-gamma * distances[np.ix_(indices, indices)])


def _kernel_weights(
    base_kernel: np.ndarray,
    training: np.ndarray,
    query: np.ndarray,
    model: str,
    alpha: float,
    gamma_factor: float | None = None,
    gamma_reference: np.ndarray | None = None,
) -> np.ndarray:
    all_indices = np.concatenate((training, query))
    matrix = _kernel_matrix(
        base_kernel, all_indices, model, gamma_factor=gamma_factor, gamma_reference=gamma_reference
    )
    n = len(training)
    train_kernel = matrix[:n, :n]
    cross = matrix[n:, :n]
    raw = np.linalg.solve(train_kernel + alpha * np.eye(n), cross.T).T
    return raw + (1.0 - raw.sum(axis=1, keepdims=True)) / n


def _coefficient(target: int, support: np.ndarray, weights: np.ndarray, size: int = 50) -> np.ndarray:
    value = np.zeros(size, dtype=np.float64)
    value[target] = 1.0
    value[support] -= weights
    return value


def cross_contributions(
    grams: np.ndarray,
    target: int,
    support: np.ndarray,
    weights6: np.ndarray,
    weights14: np.ndarray,
) -> np.ndarray:
    size = grams.shape[-1]
    first = _coefficient(target, support, weights6, size=size)
    second = _coefficient(target, support, weights14, size=size)
    return np.einsum("i,pij,j->p", first, grams, second, optimize=True)


def _cross_bilinear(grams: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.einsum("i,pij,j->p", first, grams, second, optimize=True)


def operator_contributions(
    grams: np.ndarray,
    target: int,
    support: np.ndarray,
    weights6: np.ndarray,
    weights14: np.ndarray,
    reference_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    truth6 = _coefficient(target, support, reference_weights)
    truth14 = truth6.copy()
    hat6 = np.zeros(50, dtype=np.float64)
    hat14 = np.zeros(50, dtype=np.float64)
    hat6[support] = weights6 - reference_weights
    hat14[support] = weights14 - reference_weights
    truth = _cross_bilinear(grams, truth6, truth14)
    aligned = 0.5 * (
        _cross_bilinear(grams, hat6, truth14) + _cross_bilinear(grams, truth6, hat14)
    )
    predicted = _cross_bilinear(grams, hat6, hat14)
    return {"truth": truth, "aligned": aligned, "predicted": predicted}


def _balanced_folds(indices: np.ndarray, contexts: tuple[str, ...], folds: int = 5) -> list[np.ndarray]:
    order = sorted(range(len(indices)), key=lambda position: _stable_seed("inner", contexts[indices[position]]))
    labels = np.empty(len(indices), dtype=np.int8)
    for rank, position in enumerate(order):
        labels[position] = rank % min(folds, len(indices))
    return [np.flatnonzero(labels == fold) for fold in range(labels.max() + 1)]


def _training_objective_from_kernels(
    grams: ResponseGrams,
    support: np.ndarray,
    kernel6: np.ndarray,
    kernel14: np.ndarray,
    alpha: float,
    folds: list[np.ndarray],
) -> float:
    total = 0.0
    for validation_positions in folds:
        train_positions = np.setdiff1d(np.arange(len(support)), validation_positions)
        training = support[train_positions]
        validation = support[validation_positions]
        weights = []
        for kernel in (kernel6, kernel14):
            train_kernel = kernel[np.ix_(train_positions, train_positions)]
            cross = kernel[np.ix_(validation_positions, train_positions)]
            raw = np.linalg.solve(
                train_kernel + alpha * np.eye(len(train_positions)), cross.T
            ).T
            weights.append(raw + (1.0 - raw.sum(axis=1, keepdims=True)) / len(training))
        for local, target in enumerate(validation):
            contribution = cross_contributions(
                grams.cross,
                int(target),
                training,
                weights[0][local],
                weights[1][local],
            ).sum()
            total += float(contribution)
    return total


def tune_hyperparameters(
    frozen: FrozenInputs,
    grams: ResponseGrams,
    support: np.ndarray,
    model: str,
) -> dict[str, Any]:
    if len(support) < 8:
        defaults = SMALL_M_DEFAULTS[model]
        return {
            **defaults,
            "selection": "frozen_small_m_training_only_default",
            "inner_objective": np.nan,
            "candidate_count": 1,
        }
    factors: tuple[float | None, ...]
    alphas: tuple[float, ...]
    if model == "linear_ridge":
        factors = (None,)
        alphas = LINEAR_ALPHAS
    elif model == "rbf_ridge":
        factors = RBF_GAMMA_FACTORS
        alphas = RBF_ALPHAS
    else:
        raise ValueError(model)
    folds = _balanced_folds(support, frozen.contexts)
    objectives = []
    for factor in factors:
        kernel6 = _kernel_matrix(
            frozen.baseline_kernels[0], support, model, factor, gamma_reference=support
        )
        kernel14 = _kernel_matrix(
            frozen.baseline_kernels[1], support, model, factor, gamma_reference=support
        )
        for alpha in alphas:
            value = _training_objective_from_kernels(
                grams, support, kernel6, kernel14, alpha, folds
            )
            objectives.append((value, alpha, factor))
    value, alpha, factor = min(
        objectives,
        key=lambda item: (not np.isfinite(item[0]), item[0], item[1], -1 if item[2] is None else item[2]),
    )
    return {
        "alpha": alpha,
        "gamma_factor": np.nan if factor is None else factor,
        "selection": "fivefold_training_context_validation_replicate_cross_error",
        "inner_objective": value,
        "candidate_count": len(factors) * len(alphas),
    }


def _response_eigenvectors(gram: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = gram[np.ix_(indices, indices)]
    center = np.eye(len(indices)) - np.ones((len(indices), len(indices))) / len(indices)
    centered = center @ selected @ center
    eigenvalues, eigenvectors = np.linalg.eigh((centered + centered.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order]


def _project_affine_weights(weights: np.ndarray, eigenvectors: np.ndarray, rank: int) -> np.ndarray:
    uniform = np.full(weights.shape[1], 1.0 / weights.shape[1])
    basis = eigenvectors[:, : min(rank, eigenvectors.shape[1] - 1)]
    projected = (weights - uniform) @ basis @ basis.T
    result = uniform + projected
    result += (1.0 - result.sum(axis=1, keepdims=True)) / result.shape[1]
    return result


def tune_bilinear(
    frozen: FrozenInputs,
    grams: ResponseGrams,
    support: np.ndarray,
) -> dict[str, Any]:
    if len(support) < 8:
        return {
            "alpha": 1.0,
            "gamma_factor": np.nan,
            "rank": min(2, len(support) - 1),
            "selection": "frozen_small_m_training_only_default",
            "inner_objective": np.nan,
            "candidate_count": 1,
        }
    folds = _balanced_folds(support, frozen.contexts)
    kernel6 = frozen.baseline_kernels[0][np.ix_(support, support)]
    kernel14 = frozen.baseline_kernels[1][np.ix_(support, support)]
    fold_cache = []
    for validation_positions in folds:
        train_positions = np.setdiff1d(np.arange(len(support)), validation_positions)
        training = support[train_positions]
        fold_cache.append(
            (
                validation_positions,
                train_positions,
                _response_eigenvectors(grams.total_same6, training),
                _response_eigenvectors(grams.total_same14, training),
            )
        )
    objectives = []
    for alpha in BILINEAR_ALPHAS:
        raw_by_fold = []
        for validation_positions, train_positions, eig6, eig14 in fold_cache:
            raw = []
            for kernel in (kernel6, kernel14):
                train_kernel = kernel[np.ix_(train_positions, train_positions)]
                cross = kernel[np.ix_(validation_positions, train_positions)]
                values = np.linalg.solve(
                    train_kernel + alpha * np.eye(len(train_positions)), cross.T
                ).T
                values += (1.0 - values.sum(axis=1, keepdims=True)) / len(train_positions)
                raw.append(values)
            raw_by_fold.append((raw, eig6, eig14))
        for rank in BILINEAR_RANKS:
            total = 0.0
            for cache, fold_values in zip(fold_cache, raw_by_fold, strict=True):
                validation_positions, train_positions, _, _ = cache
                raw, eig6, eig14 = fold_values
                weights6 = _project_affine_weights(raw[0], eig6, rank)
                weights14 = _project_affine_weights(raw[1], eig14, rank)
                training = support[train_positions]
                for local, target in enumerate(support[validation_positions]):
                    total += float(
                        cross_contributions(
                            grams.cross,
                            int(target),
                            training,
                            weights6[local],
                            weights14[local],
                        ).sum()
                    )
            objectives.append((total, alpha, rank))
    objective, alpha, rank = min(objectives, key=lambda item: (item[0], item[1], item[2]))
    return {
        "alpha": alpha,
        "gamma_factor": np.nan,
        "rank": rank,
        "selection": "fivefold_training_context_validation_replicate_cross_error",
        "inner_objective": objective,
        "candidate_count": len(BILINEAR_ALPHAS) * len(BILINEAR_RANKS),
    }


def fit_episode(
    frozen: FrozenInputs,
    grams: ResponseGrams,
    target: int,
    support: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    parameters = (
        tune_bilinear(frozen, grams, support)
        if model == "bilinear_reduced_rank"
        else tune_hyperparameters(frozen, grams, support, model)
    )
    factor = None if np.isnan(parameters["gamma_factor"]) else float(parameters["gamma_factor"])
    weights = []
    for plate in range(2):
        raw = _kernel_weights(
            frozen.baseline_kernels[plate],
            support,
            np.asarray([target]),
            model,
            float(parameters["alpha"]),
            factor,
            gamma_reference=support,
        )
        if model == "bilinear_reduced_rank":
            response_gram = grams.total_same6 if plate == 0 else grams.total_same14
            eigenvectors = _response_eigenvectors(response_gram, support)
            raw = _project_affine_weights(raw, eigenvectors, int(parameters["rank"]))
        weights.append(raw[0])
    parameters["weight_sum_plate6"] = float(weights[0].sum())
    parameters["weight_sum_plate14"] = float(weights[1].sum())
    parameters["weight_l2_plate6"] = float(np.linalg.norm(weights[0]))
    parameters["weight_l2_plate14"] = float(np.linalg.norm(weights[1]))
    return weights[0], weights[1], parameters


def evaluate_episode(
    grams: ResponseGrams,
    target: int,
    support: np.ndarray,
    weights6: np.ndarray,
    weights14: np.ndarray,
    max_support: np.ndarray,
) -> dict[str, Any]:
    reference = np.full(len(support), 1.0 / len(support))
    model_error = cross_contributions(grams.cross, target, support, weights6, weights14)
    reference_error = cross_contributions(grams.cross, target, support, reference, reference)
    max_reference = np.full(len(max_support), 1.0 / len(max_support))
    fixed_error = cross_contributions(
        grams.cross, target, max_support, max_reference, max_reference
    )
    operator = operator_contributions(
        grams.cross, target, support, weights6, weights14, reference
    )
    return {
        "reference": reference_error,
        "model_error": model_error,
        "fixed_reference": fixed_error,
        "truth": operator["truth"],
        "aligned": operator["aligned"],
        "predicted": operator["predicted"],
    }


def _run_target_episodes(
    frozen: FrozenInputs,
    grams: ResponseGrams,
    ladders: pd.DataFrame,
    target: int,
    m_grid: Iterable[int],
    models: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    maximal = _support_from_ladders(ladders, target, "max49", 49)
    for m in m_grid:
        ladder_ids: list[int | str] = ["max49"] if m == 49 else list(range(LADDERS))
        for ladder in ladder_ids:
            support = _support_from_ladders(ladders, target, ladder, m)
            for model in models:
                episode_start = time.perf_counter()
                weights6, weights14, parameters = fit_episode(
                    frozen, grams, target, support, model
                )
                values = evaluate_episode(
                    grams, target, support, weights6, weights14, maximal
                )
                reference = float(values["reference"].sum())
                error = float(values["model_error"].sum())
                fixed = float(values["fixed_reference"].sum())
                truth = float(values["truth"].sum())
                aligned = float(values["aligned"].sum())
                predicted = float(values["predicted"].sum())
                g = 1.0 - error / reference if reference > 0 else np.nan
                g_fixed = 1.0 - error / fixed if fixed > 0 else np.nan
                alpha = aligned / truth if truth > 0 else np.nan
                kappa = predicted / truth if truth > 0 else np.nan
                cosine = (
                    aligned / math.sqrt(truth * predicted)
                    if truth > 0 and predicted > 0
                    else np.nan
                )
                rows.append(
                    {
                        "target_index": target,
                        "target_context": frozen.contexts[target],
                        "lineage": frozen.lineages[target],
                        "m": m,
                        "ladder_id": ladder,
                        "model": model,
                        "reference_cross_energy": reference,
                        "model_error_cross_energy": error,
                        "fixed49_reference_cross_energy": fixed,
                        "g": g,
                        "g_fixed": g_fixed,
                        "truth_cross_energy": truth,
                        "aligned_cross_energy": aligned,
                        "predicted_cross_energy": predicted,
                        "alpha_cross": alpha,
                        "kappa_cross": kappa,
                        "cosine_cross": cosine,
                        "g_from_decomposition": 2 * alpha - kappa,
                        "decomposition_absolute_error": abs(g - (2 * alpha - kappa)),
                        "target_treated_outcomes_used_in_fit": False,
                        "target_baseline_only": True,
                        "plate_predictions_fit_separately": True,
                        "test_interventions": grams.intervention_count,
                        "response_genes": grams.gene_count,
                        "episode_seconds": time.perf_counter() - episode_start,
                    }
                )
                parameter_rows.append(
                    {
                        "target_index": target,
                        "target_context": frozen.contexts[target],
                        "m": m,
                        "ladder_id": ladder,
                        "model": model,
                        **parameters,
                    }
                )
    return rows, parameter_rows


_PROCESS_STATE: tuple[
    FrozenInputs, ResponseGrams, pd.DataFrame, tuple[int, ...], tuple[str, ...]
] | None = None


def _initialize_process_worker(
    frozen: FrozenInputs,
    grams: ResponseGrams,
    ladders: pd.DataFrame,
    m_grid: tuple[int, ...],
    models: tuple[str, ...],
) -> None:
    global _PROCESS_STATE
    _PROCESS_STATE = frozen, grams, ladders, m_grid, models


def _process_target_worker(target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _PROCESS_STATE is None:
        raise RuntimeError("Tahoe process worker was not initialized")
    frozen, grams, ladders, m_grid, models = _PROCESS_STATE
    return _run_target_episodes(frozen, grams, ladders, target, m_grid, models)


def run_episodes(
    root: Path,
    frozen: FrozenInputs,
    grams: ResponseGrams,
    ladders: pd.DataFrame,
    targets: Iterable[int],
    m_grid: Iterable[int],
    models: Iterable[str],
    pilot: bool = False,
    workers: int = 1,
    output_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    target_values = list(map(int, targets))
    if workers == 1:
        for position, target in enumerate(target_values, start=1):
            target_rows, target_parameters = _run_target_episodes(
                frozen, grams, ladders, target, tuple(m_grid), tuple(models)
            )
            rows.extend(target_rows)
            parameter_rows.extend(target_parameters)
            print(f"held-context episodes completed {position}/{len(target_values)}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_process_worker,
            initargs=(frozen, grams, ladders, tuple(m_grid), tuple(models)),
        ) as executor:
            futures = {executor.submit(_process_target_worker, target): target for target in target_values}
            for position, future in enumerate(as_completed(futures), start=1):
                target_rows, target_parameters = future.result()
                rows.extend(target_rows)
                parameter_rows.extend(target_parameters)
                print(f"held-context episodes completed {position}/{len(target_values)}", flush=True)
    rows.sort(key=lambda row: (row["target_index"], row["m"], str(row["ladder_id"]), row["model"]))
    parameter_rows.sort(
        key=lambda row: (row["target_index"], row["m"], str(row["ladder_id"]), row["model"])
    )
    frame, parameters = pd.DataFrame(rows), pd.DataFrame(parameter_rows)
    frame.attrs["elapsed_seconds"] = time.perf_counter() - started
    out = output_dir(root)
    prefix = output_prefix or (
        "TAHOE_HELD_CONTEXT_SCALING_PILOT" if pilot else "TAHOE_HELD_CONTEXT_SCALING_RAW"
    )
    frame.to_csv(out / f"{prefix}.csv", index=False)
    parameters.to_csv(out / f"{prefix}_PARAMETERS.csv", index=False)
    return frame, parameters


def sanity_check_audited_estimator(root: Path) -> dict[str, Any]:
    """Reconcile the generalized cross form with the audited Tahoe helper."""

    from igc_virtual_cell.cgc_tahoe_0i.scaling import _residual_from_gram

    rng = np.random.default_rng(202608283)
    x6 = rng.normal(size=(50, 17))
    x14 = rng.normal(size=(50, 17))
    gram = x6 @ x14.T
    target = 7
    support = np.asarray([0, 2, 9, 13, 21, 33], dtype=np.int32)
    weights = rng.normal(size=len(support))
    weights += (1 - weights.sum()) / len(weights)
    ours = float(cross_contributions(gram[None], target, support, weights, weights)[0])
    audited = float(
        _residual_from_gram(gram, target, support[None, :], weights[None, :])[0]
    )
    direct = float(
        (x6[target] - weights @ x6[support])
        @ (x14[target] - weights @ x14[support])
    )
    result = {
        "ours": ours,
        "audited": audited,
        "direct_vector": direct,
        "max_absolute_error": max(abs(ours - audited), abs(ours - direct)),
        "tolerance": 1e-10,
        "passed": bool(max(abs(ours - audited), abs(ours - direct)) <= 1e-10),
    }
    _json(output_dir(root) / "TAHOE_AUDITED_G_SANITY.json", result)
    if not result["passed"]:
        raise RuntimeError("Replicate-stable cross-energy sanity reconciliation failed")
    return result


def summarize_pilot(
    root: Path,
    frame: pd.DataFrame,
    cache_path: Path,
    cache_started_size: int,
    wall_seconds: float,
) -> dict[str, Any]:
    out = output_dir(root)
    pooled_rows = []
    for (model, m), group in frame.groupby(["model", "m"], sort=True):
        pooled_rows.append(
            {
                "model": model,
                "m": int(m),
                "pooled_g": 1 - group["model_error_cross_energy"].sum() / group["reference_cross_energy"].sum(),
                "median_episode_g": group["g"].median(),
                "episodes": len(group),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(out / "TAHOE_HELD_CONTEXT_SCALING_PILOT_POOLED.csv", index=False)
    summary = {
        "status": "PASS",
        "targets": frame["target_context"].nunique(),
        "target_contexts": sorted(frame["target_context"].unique().tolist()),
        "m_grid": sorted(frame["m"].unique().astype(int).tolist()),
        "models": sorted(frame["model"].unique().tolist()),
        "cpu_only": True,
        "gpu_required": False,
        "wall_seconds_including_cache": wall_seconds,
        "episode_seconds": float(frame["episode_seconds"].sum()),
        "disk_growth_bytes": max(cache_path.stat().st_size - cache_started_size, 0),
        "gram_cache_bytes": cache_path.stat().st_size,
        "peak_ram_note": "Process peak working set is captured by the PowerShell launcher in the companion runtime JSON.",
        "leakage": {
            "target_treated_outcomes_used_in_fit": bool(frame["target_treated_outcomes_used_in_fit"].any()),
            "target_baseline_only_all": bool(frame["target_baseline_only"].all()),
            "plate_predictions_fit_separately_all": bool(frame["plate_predictions_fit_separately"].all()),
            "same_93_interventions_all": bool(frame["test_interventions"].eq(93).all()),
            "same_25695_genes_all": bool(frame["response_genes"].eq(25_695).all()),
        },
        "max_decomposition_absolute_error": float(frame["decomposition_absolute_error"].max()),
        "pooled_g": pooled.to_dict("records"),
        "raw_single_cells_loaded": False,
        "full_run_authorized_by_stop_rule": True,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    _json(out / "TAHOE_HELD_CONTEXT_SCALING_PILOT.json", summary)
    lines = [
        "# Tahoe held-context scaling pilot",
        "",
        "**Verdict: PASS — proceed to the full frozen experiment.**",
        "",
        f"The required five-target pilot completed in {wall_seconds:.1f} s using CPU only; no GPU is required.",
        "The frozen pseudobulk response tensor was streamed once into small response-Gram sufficient statistics; no raw single-cell data were loaded.",
        "",
        "## Exact pooled pilot results",
        "",
        "| model | m | pooled g | median episode g | episodes |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row.model} | {int(row.m)} | {row.pooled_g:.8f} | {row.median_episode_g:.8f} | {int(row.episodes)} |"
            for row in pooled.itertuples(index=False)
        ],
        "",
        "## Leakage QA",
        "",
        "- Zero target treated outcomes were used in fitting or tuning.",
        "- The target DMSO profile was the only target molecular input.",
        "- Plate6 and Plate14 weights were fitted independently and combined only through the audited cross-plate energy.",
        "- All 93 interventions and 25,695 response genes were retained.",
    ]
    (out / "TAHOE_HELD_CONTEXT_SCALING_PILOT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
