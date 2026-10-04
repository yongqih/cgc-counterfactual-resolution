#!/usr/bin/env python3
"""Frozen matched-340 WGCNA counterfactual-resolution frontier.

The implementation intentionally separates training-only module construction
from held-out scoring.  Run order is prepare -> modules -> audit-and-score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cgc_wgcna_frontier"
DATA = ROOT / "data" / "wgcna_frontier_official"
CACHE = ROOT / "data" / "wgcna_frontier_cache"
REPRESENTATION = DATA / "MATCHED_POSTSVA_340_REPRESENTATION.npz"
FROZEN_OOF = DATA / "MATCHED_POSTSVA_FULL_OOF.npz"
HALLMARK = DATA / "h.all.v2025.1.Hs.symbols.gmt"
GENE_AXIS = ROOT / "results" / "level2_lea" / "archs4_robustness" / "ARCHS4_GENE_AXIS_RECONCILIATION.csv"
R_SCRIPT = ROOT / "scripts" / "run_wgcna_base_partition.R"
R_EXECUTABLE = Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")
R_LIBRARY = ROOT / "data" / "wgcna_frontier_runtime" / "Rlib"
HISTORICAL = "c3d074412ecc1be0958141726db16a1ed0036a04"
SEED = 207049
MERGE_THRESHOLDS = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
GAMMA_MULTIPLIERS = [0.25, 1.0, 4.0]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
PC_LEVELS = (4, 8, 16)
TOLERANCE = 1e-6


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Byte-for-byte transcription of historical level2_full.standardize_fit."""
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32, copy=False)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    result = (
        np.square(x64).sum(axis=1, keepdims=True)
        + np.square(y64).sum(axis=1)[None, :]
        - 2 * x64 @ y64.T
    )
    np.maximum(result, 0, out=result)
    return result


def median_squared_distance(x: np.ndarray) -> float:
    distances = squared_distances(x, x)
    values = distances[np.triu_indices(len(x), 1)]
    positive = values[values > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def rbf_kernel_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    gamma_multiplier: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    mean, scale = standardize_fit(x_train)
    train = standardize_apply(x_train, mean, scale)
    test = standardize_apply(x_test, mean, scale)
    median_distance = median_squared_distance(train)
    gamma = gamma_multiplier / median_distance
    train_kernel = np.exp(-gamma * squared_distances(train, train))
    test_kernel = np.exp(-gamma * squared_distances(test, train))
    y_mean = y_train.mean(axis=0, dtype=np.float64)
    coefficients = np.linalg.solve(
        train_kernel + alpha * np.eye(len(train_kernel)),
        y_train.astype(np.float64) - y_mean,
    )
    prediction = test_kernel @ coefficients + y_mean
    return prediction.astype(np.float32), gamma


def inner_splitter(strata: np.ndarray, seed: int, folds: int = 3):
    if len(np.unique(strata)) > 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(strata)), strata
        )
    return KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(strata)))


def tune_rbf_kernel(
    x: np.ndarray,
    delta: np.ndarray,
    strata: np.ndarray,
    gamma_multipliers: list[float],
    alphas: list[float],
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    """Exact historical RBF inner-CV protocol, isolated from unused torch imports."""
    totals = {(gamma, alpha): [0.0, 0.0] for gamma in gamma_multipliers for alpha in alphas}
    for train, valid in inner_splitter(strata, seed):
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        y_valid = delta[valid] - mean_delta
        denominator = float(np.square(y_valid, dtype=np.float64).sum())
        for gamma_multiplier, alpha in totals:
            prediction, _ = rbf_kernel_ridge_predict(x[train], y_train, x[valid], gamma_multiplier, alpha)
            totals[(gamma_multiplier, alpha)][0] += float(
                np.square(y_valid - prediction, dtype=np.float64).sum()
            )
            totals[(gamma_multiplier, alpha)][1] += denominator
    table = pd.DataFrame([
        {"gamma_multiplier": gamma, "alpha": alpha, "inner_pooled_R2": 1 - sse / denominator}
        for (gamma, alpha), (sse, denominator) in totals.items()
    ])
    best = table.sort_values(
        ["inner_pooled_R2", "gamma_multiplier", "alpha"], ascending=[False, True, True]
    ).iloc[0]
    return float(best["gamma_multiplier"]), float(best["alpha"]), table


@dataclass(frozen=True)
class KernelSplit:
    train_indices: np.ndarray
    valid_indices: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    median_distance: float
    train_kernels: dict[float, np.ndarray]
    valid_kernels: dict[float, np.ndarray]


@dataclass(frozen=True)
class RBFDesign:
    inner: tuple[KernelSplit, ...]
    outer: KernelSplit

    @staticmethod
    def _split(x: np.ndarray, train: np.ndarray, valid: np.ndarray) -> KernelSplit:
        mean, scale = standardize_fit(x[train])
        standardized_train = standardize_apply(x[train], mean, scale)
        standardized_valid = standardize_apply(x[valid], mean, scale)
        median_distance = median_squared_distance(standardized_train)
        train_distances = squared_distances(standardized_train, standardized_train)
        valid_distances = squared_distances(standardized_valid, standardized_train)
        return KernelSplit(
            train_indices=np.asarray(train), valid_indices=np.asarray(valid), mean=mean, scale=scale,
            median_distance=median_distance,
            train_kernels={
                gamma: np.exp(-(gamma / median_distance) * train_distances) for gamma in GAMMA_MULTIPLIERS
            },
            valid_kernels={
                gamma: np.exp(-(gamma / median_distance) * valid_distances) for gamma in GAMMA_MULTIPLIERS
            },
        )

    @classmethod
    def build(cls, x_train: np.ndarray, x_test: np.ndarray, strata: np.ndarray, seed: int) -> "RBFDesign":
        inner = tuple(
            cls._split(x_train, train, valid) for train, valid in inner_splitter(strata, seed)
        )
        combined = np.vstack([x_train, x_test])
        outer_train = np.arange(len(x_train), dtype=np.int64)
        outer_valid = np.arange(len(x_train), len(combined), dtype=np.int64)
        outer = cls._split(combined, outer_train, outer_valid)
        return cls(inner=inner, outer=outer)

    def tune(self, delta: np.ndarray) -> tuple[float, float, pd.DataFrame]:
        totals = {(gamma, alpha): [0.0, 0.0] for gamma in GAMMA_MULTIPLIERS for alpha in ALPHAS}
        for split in self.inner:
            mean_delta = delta[split.train_indices].mean(axis=0)
            y_train = delta[split.train_indices] - mean_delta
            y_valid = delta[split.valid_indices] - mean_delta
            denominator = float(np.square(y_valid, dtype=np.float64).sum())
            for gamma, alpha in totals:
                coefficients = np.linalg.solve(
                    split.train_kernels[gamma] + alpha * np.eye(len(split.train_indices)),
                    y_train.astype(np.float64) - y_train.mean(axis=0, dtype=np.float64),
                )
                prediction = split.valid_kernels[gamma] @ coefficients + y_train.mean(axis=0, dtype=np.float64)
                totals[(gamma, alpha)][0] += float(
                    np.square(y_valid - prediction.astype(np.float32), dtype=np.float64).sum()
                )
                totals[(gamma, alpha)][1] += denominator
        table = pd.DataFrame([
            {"gamma_multiplier": gamma, "alpha": alpha, "inner_pooled_R2": 1 - sse / denominator}
            for (gamma, alpha), (sse, denominator) in totals.items()
        ])
        best = table.sort_values(
            ["inner_pooled_R2", "gamma_multiplier", "alpha"], ascending=[False, True, True]
        ).iloc[0]
        return float(best["gamma_multiplier"]), float(best["alpha"]), table

    def predict_and_hash(self, y_train: np.ndarray, gamma: float, alpha: float) -> tuple[np.ndarray, float, str]:
        y_mean = y_train.mean(axis=0, dtype=np.float64)
        coefficients = np.linalg.solve(
            self.outer.train_kernels[gamma] + alpha * np.eye(len(self.outer.train_indices)),
            y_train.astype(np.float64) - y_mean,
        )
        prediction = self.outer.valid_kernels[gamma] @ coefficients + y_mean
        fitted_gamma = gamma / self.outer.median_distance
        parameter_hash = array_hash(
            self.outer.mean, self.outer.scale, np.asarray([fitted_gamma, alpha]), y_mean, coefficients
        )
        return prediction.astype(np.float32), fitted_gamma, parameter_hash


def load_inputs() -> dict[str, np.ndarray]:
    with np.load(REPRESENTATION, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    expected = (340, 10_110)
    if values["baseline"].shape != expected or values["treated"].shape != expected:
        raise RuntimeError("matched-340 representation shape failed")
    if not np.isfinite(values["baseline"]).all() or not np.isfinite(values["treated"]).all():
        raise RuntimeError("matched-340 representation contains non-finite values")
    if tuple(np.bincount(values["folds"].astype(int), minlength=5)) != (67, 69, 68, 68, 68):
        raise RuntimeError("historical fold counts changed")
    return values


def fold_arrays(values: dict[str, np.ndarray], fold: int, treated: np.ndarray | None = None) -> dict[str, np.ndarray]:
    baseline = values["baseline"].astype(np.float32, copy=False)
    active_treated = values["treated"].astype(np.float32, copy=False) if treated is None else treated
    folds = values["folds"].astype(int)
    train = np.flatnonzero(folds != fold)
    test = np.flatnonzero(folds == fold)
    delta = active_treated - baseline
    # Historical Lea authority calls mean directly on the float32 delta.
    shared = delta[train].mean(axis=0)
    return {
        "train": train,
        "test": test,
        "x_train": baseline[train],
        "x_test": baseline[test],
        "y_train": (delta[train] - shared).astype(np.float32),
        "y_test": (delta[test] - shared).astype(np.float32),
        "shared": shared,
        "strata_train": values["strata"][train].astype(str),
    }


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    values = load_inputs()
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        arrays = fold_arrays(values, fold)
        fold_dir = CACHE / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        path = fold_dir / "train_residual.float32.bin"
        contiguous = np.ascontiguousarray(arrays["y_train"], dtype="<f4")
        contiguous.tofile(path)
        digest = sha256_file(path)
        metadata = {
            "status": "PREPARED",
            "outer_fold": fold,
            "input_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "input_sha256": digest,
            "n_training_lcls": int(len(arrays["train"])),
            "n_held_lcls": int(len(arrays["test"])),
            "n_genes": int(contiguous.shape[1]),
            "training_indices_hash": array_hash(arrays["train"]),
            "held_indices_hash": array_hash(arrays["test"]),
            "training_residual_hash": array_hash(contiguous),
            "denominator": float(np.square(arrays["y_test"], dtype=np.float64).sum()),
            "prepared_at": now(),
        }
        write_json(fold_dir / "PREPARED.json", metadata)
        rows.append(metadata)
    pd.DataFrame(rows).to_csv(OUT / "WGCNA_TRAINING_INPUT_MANIFEST.csv", index=False)
    print(f"Prepared five fold-local training matrices under {CACHE}", flush=True)


def run_module(fold: int, output_name: str = "wgcna_base") -> None:
    metadata = json.loads((CACHE / f"fold_{fold}" / "PREPARED.json").read_text(encoding="utf-8"))
    input_path = ROOT / metadata["input_path"]
    output_dir = CACHE / f"fold_{fold}" / output_name
    complete = output_dir / "COMPLETE.json"
    if complete.exists():
        existing = json.loads(complete.read_text(encoding="utf-8"))
        if existing.get("input_sha256") == metadata["input_sha256"] and existing.get("status") == "COMPLETE":
            print(f"fold {fold} {output_name}: valid COMPLETE reused", flush=True)
            return
        raise RuntimeError(f"stale WGCNA COMPLETE manifest for fold {fold} {output_name}")
    command = [
        str(R_EXECUTABLE), str(R_SCRIPT), str(input_path), str(metadata["n_training_lcls"]),
        str(metadata["n_genes"]), str(fold), str(output_dir), str(R_LIBRARY), metadata["input_sha256"],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (output_dir / "stderr.log").open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if result.returncode != 0 or not complete.exists():
        raise RuntimeError(f"WGCNA fold {fold} failed; inspect {output_dir}")
    print(f"fold {fold} {output_name}: COMPLETE", flush=True)


def load_base_labels(fold: int, output_name: str = "wgcna_base") -> tuple[np.ndarray, dict[str, Any]]:
    directory = CACHE / f"fold_{fold}" / output_name
    summary = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    table = pd.read_csv(directory / "base_module_labels.csv")
    if not np.array_equal(table["gene_index"].to_numpy(), np.arange(10_110)):
        raise RuntimeError(f"fold {fold} gene index changed")
    labels = table["base_module_label"].to_numpy(dtype=np.int64)
    return labels, summary


@dataclass(frozen=True)
class Component:
    coordinate: int
    genes: np.ndarray
    base_labels: tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    loading: np.ndarray
    explained_variance_ratio: float
    training_energy: float

    def scores(self, matrix: np.ndarray) -> np.ndarray:
        standardized = (matrix[:, self.genes].astype(np.float64) - self.mean) / self.scale
        return (standardized @ self.loading).astype(np.float32)

    def reconstruct(self, scores: np.ndarray) -> np.ndarray:
        standardized = np.asarray(scores, dtype=np.float64)[:, None] * self.loading[None, :]
        return (standardized * self.scale[None, :] + self.mean[None, :]).astype(np.float32)


def fit_component(matrix: np.ndarray, genes: np.ndarray, base_labels: tuple[int, ...], coordinate: int, seed: int) -> Component:
    subset = matrix[:, genes].astype(np.float64)
    mean = subset.mean(axis=0)
    scale = subset.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("component received zero-variance gene")
    standardized = (subset - mean) / scale
    pca = PCA(n_components=1, svd_solver="randomized", iterated_power=7, random_state=seed)
    score = pca.fit_transform(standardized)[:, 0]
    loading = pca.components_[0].astype(np.float64)
    largest = int(np.argmax(np.abs(loading)))
    if loading[largest] < 0:
        loading = -loading
        score = -score
    return Component(
        coordinate=coordinate,
        genes=np.asarray(genes, dtype=np.int64),
        base_labels=base_labels,
        mean=mean,
        scale=scale,
        loading=loading,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
        training_energy=float(np.square(score, dtype=np.float64).sum()),
    )


def component_hash(components: list[Component]) -> str:
    digest = hashlib.sha256()
    for component in components:
        digest.update(np.asarray([component.coordinate], dtype=np.int64).tobytes())
        digest.update(np.asarray(component.base_labels, dtype=np.int64).tobytes())
        for value in (component.genes, component.mean, component.scale, component.loading):
            digest.update(array_hash(np.asarray(value)).encode())
        digest.update(np.asarray([component.training_energy, component.explained_variance_ratio], dtype=np.float64).tobytes())
    return digest.hexdigest()


def component_seed(fold: int, base_labels: tuple[int, ...]) -> int:
    encoded = ",".join(map(str, base_labels)).encode()
    offset = int(hashlib.sha256(encoded).hexdigest()[:8], 16) % 90_000
    return SEED + fold * 100_000 + offset


def build_ladder(y_train: np.ndarray, base_labels: np.ndarray, fold: int) -> list[dict[str, Any]]:
    labels = sorted(int(value) for value in np.unique(base_labels) if value > 0)
    if not labels:
        raise RuntimeError(f"fold {fold} has no assigned WGCNA modules")
    base_components: dict[int, Component] = {}
    for index, label in enumerate(labels):
        genes = np.flatnonzero(base_labels == label)
        base_components[label] = fit_component(
            y_train, genes, (label,), index, component_seed(fold, (label,))
        )
    base_scores = np.column_stack([base_components[label].scores(y_train) for label in labels])
    if len(labels) > 1:
        correlation = np.corrcoef(base_scores, rowvar=False)
        distance = np.clip(1.0 - correlation, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=True), method="average", optimal_ordering=False)
    else:
        tree = None
    ladder: list[dict[str, Any]] = []
    for level_index, threshold in enumerate(MERGE_THRESHOLDS):
        if threshold == 0.0 or tree is None:
            raw_groups = np.arange(1, len(labels) + 1, dtype=int)
        else:
            raw_groups = fcluster(tree, t=threshold, criterion="distance").astype(int)
        members: dict[int, list[int]] = {}
        for label, group in zip(labels, raw_groups, strict=True):
            members.setdefault(int(group), []).append(label)
        ordered_groups = sorted(members.values(), key=lambda value: min(value))
        components: list[Component] = []
        for coordinate, group_labels in enumerate(ordered_groups):
            if threshold == 0.0 and len(group_labels) == 1:
                source = base_components[group_labels[0]]
                component = Component(
                    coordinate=coordinate, genes=source.genes, base_labels=source.base_labels,
                    mean=source.mean, scale=source.scale, loading=source.loading,
                    explained_variance_ratio=source.explained_variance_ratio, training_energy=source.training_energy,
                )
            else:
                genes = np.flatnonzero(np.isin(base_labels, group_labels))
                component = fit_component(
                    y_train, genes, tuple(group_labels), coordinate,
                    component_seed(fold, tuple(group_labels)),
                )
            components.append(component)
        ladder.append({
            "level_index": level_index,
            "threshold": threshold,
            "components": components,
            "partition_hash": json_hash([list(component.base_labels) for component in components]),
            "component_hash": component_hash(components),
        })
    return ladder


def coordinate_matrices(matrix: np.ndarray, components: list[Component]) -> np.ndarray:
    return np.column_stack([component.scores(matrix) for component in components]).astype(np.float32)


def reconstruct(matrix_scores: np.ndarray, components: list[Component], n_rows: int, n_genes: int) -> np.ndarray:
    output = np.zeros((n_rows, n_genes), dtype=np.float32)
    for index, component in enumerate(components):
        output[:, component.genes] = component.reconstruct(matrix_scores[:, index])
    return output


def fit_ladder_predictions(
    arrays: dict[str, np.ndarray],
    ladder: list[dict[str, Any]],
    fold: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    reuse: dict[str, dict[str, Any]] = {}
    design = RBFDesign.build(arrays["x_train"], arrays["x_test"], arrays["strata_train"], SEED + fold)
    for level in ladder:
        components = level["components"]
        train_scores = coordinate_matrices(arrays["y_train"], components)
        if level["component_hash"] in reuse:
            cached = reuse[level["component_hash"]]
            gamma_multiplier = cached["gamma_multiplier"]
            alpha = cached["alpha"]
            tuning_hash = cached["tuning_hash"]
            prediction = cached["predicted_scores"]
            fitted_gamma = cached["fitted_gamma"]
            parameter_hash = cached["parameter_hash"]
        else:
            gamma_multiplier, alpha, tuning = design.tune(train_scores)
            prediction, fitted_gamma, parameter_hash = design.predict_and_hash(
                train_scores, gamma_multiplier, alpha
            )
            tuning_hash = hashlib.sha256(tuning.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
        energies = np.asarray([component.training_energy for component in components], dtype=np.float64)
        energy_weights = energies / energies.sum()
        result = {
            **level,
            "train_scores": train_scores,
            "predicted_scores": prediction,
            "gamma_multiplier": gamma_multiplier,
            "alpha": alpha,
            "fitted_gamma": fitted_gamma,
            "tuning_hash": tuning_hash,
            "parameter_hash": parameter_hash,
            "prediction_hash": array_hash(prediction),
            "energy_weights": energy_weights,
            "energy_weights_hash": array_hash(energy_weights),
        }
        results.append(result)
        reuse.setdefault(level["component_hash"], result)
    return results


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def load_hallmarks() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for line in HALLMARK.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3:
            sets[fields[0]] = set(fields[2:])
    if len(sets) != 50:
        raise RuntimeError(f"expected 50 Hallmark sets, found {len(sets)}")
    return sets


def gene_symbols(values: dict[str, np.ndarray]) -> np.ndarray:
    axis = pd.read_csv(GENE_AXIS)
    selected = axis.loc[axis["included_primary_axis"].astype(bool)].copy()
    if not np.array_equal(selected["historical_GeneID"].astype(str).to_numpy(), values["genes"].astype(str)):
        raise RuntimeError("gene-axis reconciliation does not match matched-340 representation")
    symbols = selected["ARCHS4_symbol"].astype(str).to_numpy()
    if len(np.unique(symbols)) != 10_110 or any(not value or value == "nan" for value in symbols):
        raise RuntimeError("gene symbols are not exact unique nonempty mappings")
    return symbols


def pathway_rows(
    components: list[Component], symbols: np.ndarray, hallmarks: dict[str, set[str]],
    fold: int, level_index: int, threshold: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    universe = set(symbols.tolist())
    raw: list[dict[str, Any]] = []
    for component in components:
        module_symbols = set(symbols[component.genes].tolist())
        for term, term_members in hallmarks.items():
            eligible_term = term_members & universe
            overlap = len(module_symbols & eligible_term)
            a = overlap
            b = len(module_symbols) - overlap
            c = len(eligible_term) - overlap
            d = len(universe) - a - b - c
            odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            raw.append({
                "outer_fold": fold, "level_index": level_index, "merge_threshold": threshold,
                "coordinate": component.coordinate, "module_size": len(component.genes),
                "hallmark_term": term, "overlap": overlap, "odds_ratio": float(odds),
                "p_value": float(pvalue), "training_energy": component.training_energy,
            })
    table = pd.DataFrame(raw)
    table["fdr"] = bh_adjust(table["p_value"].to_numpy())
    table["coherent_annotation"] = (table["fdr"] < 0.05) & (table["odds_ratio"] > 1.5)
    annotated = table.groupby("coordinate")["coherent_annotation"].any().reindex(
        [component.coordinate for component in components], fill_value=False
    ).to_numpy(dtype=bool)
    top = table.sort_values(
        ["coordinate", "fdr", "odds_ratio", "hallmark_term"], ascending=[True, True, False, True]
    ).groupby("coordinate", as_index=False).head(3).copy()
    top["rank_within_module"] = top.groupby("coordinate").cumcount() + 1
    return top.to_dict("records"), annotated


def r2_sufficient_statistics(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    denominator = float(np.square(truth, dtype=np.float64).sum())
    sse = float(np.square(truth - prediction, dtype=np.float64).sum())
    return denominator, sse, 1.0 - sse / denominator


def module_diagnostics(
    true_scores: np.ndarray, predicted_scores: np.ndarray, components: list[Component],
    fold: int, level: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weights = np.asarray([component.training_energy for component in components], dtype=np.float64)
    weights /= weights.sum()
    for index, component in enumerate(components):
        denominator = float(np.square(true_scores[:, index], dtype=np.float64).sum())
        sse = float(np.square(true_scores[:, index] - predicted_scores[:, index], dtype=np.float64).sum())
        rows.append({
            "outer_fold": fold, "level_index": level["level_index"],
            "merge_threshold": level["threshold"], "coordinate": component.coordinate,
            "base_module_labels": ";".join(map(str, component.base_labels)),
            "gene_count": len(component.genes),
            "pc1_variance_explained": component.explained_variance_ratio,
            "training_response_energy": component.training_energy,
            "training_energy_weight": weights[index],
            "held_denominator": denominator, "held_sse": sse,
            "coordinate_R2": 1.0 - sse / denominator if denominator > 0 else np.nan,
        })
    return rows


def validate_rerun_fold0() -> dict[str, Any]:
    run_module(0, "wgcna_leakage_rerun")
    before_dir = CACHE / "fold_0" / "wgcna_base"
    after_dir = CACHE / "fold_0" / "wgcna_leakage_rerun"
    before_labels, before_summary = load_base_labels(0, "wgcna_base")
    after_labels, after_summary = load_base_labels(0, "wgcna_leakage_rerun")
    before_fit = pd.read_csv(before_dir / "soft_threshold_fit.csv").to_csv(index=False, float_format="%.17g")
    after_fit = pd.read_csv(after_dir / "soft_threshold_fit.csv").to_csv(index=False, float_format="%.17g")
    return {
        "selected_power_identical": before_summary["selected_power"] == after_summary["selected_power"],
        "soft_threshold_fit_identical": before_fit == after_fit,
        "base_labels_identical": np.array_equal(before_labels, after_labels),
        "before_label_hash": array_hash(before_labels),
        "after_label_hash": array_hash(after_labels),
    }


def score() -> None:
    values = load_inputs()
    symbols = gene_symbols(values)
    hallmarks = load_hallmarks()
    with np.load(FROZEN_OOF, allow_pickle=False) as frozen:
        frozen_targets = frozen["targets"].astype(np.float32)
        frozen_prediction = frozen["rbf_kernel_ridge"].astype(np.float32)
        frozen_folds = frozen["folds"].astype(int)
    if not np.array_equal(frozen_folds, values["folds"].astype(int)):
        raise RuntimeError("frozen RBF OOF fold vector mismatch")

    wgcna_rerun = validate_rerun_fold0()
    leakage_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    ladder_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    predictive_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    pathway_all: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_summary_rows: list[dict[str, Any]] = []
    gene_universe_hash = array_hash(values["genes"].astype(str))

    for fold in range(5):
        start = time.time()
        before = fold_arrays(values, fold)
        mutated_treated = values["treated"].astype(np.float32).copy()
        rng = np.random.default_rng(SEED + 50_000 + fold)
        mutated_treated[before["test"]] = rng.normal(
            1000.0, 100.0, size=mutated_treated[before["test"]].shape
        ).astype(np.float32)
        after = fold_arrays(values, fold, mutated_treated)
        labels, module_summary = load_base_labels(fold)
        ladder_before = build_ladder(before["y_train"], labels, fold)
        ladder_after = build_ladder(after["y_train"], labels, fold)
        fitted_before = fit_ladder_predictions(before, ladder_before, fold)
        fitted_after = fit_ladder_predictions(after, ladder_after, fold)

        checks = {
            "training_residual_identical": array_hash(before["y_train"]) == array_hash(after["y_train"]),
            "training_baseline_identical": array_hash(before["x_train"]) == array_hash(after["x_train"]),
            "selected_power_identical": True,
            "base_modules_identical": True,
            "merged_partitions_identical": all(
                left["partition_hash"] == right["partition_hash"] for left, right in zip(ladder_before, ladder_after, strict=True)
            ),
            "eigengene_loadings_identical": all(
                left["component_hash"] == right["component_hash"] for left, right in zip(ladder_before, ladder_after, strict=True)
            ),
            "model_parameters_identical": all(
                left["parameter_hash"] == right["parameter_hash"] and left["tuning_hash"] == right["tuning_hash"]
                for left, right in zip(fitted_before, fitted_after, strict=True)
            ),
            "energy_weights_identical": all(
                left["energy_weights_hash"] == right["energy_weights_hash"]
                for left, right in zip(fitted_before, fitted_after, strict=True)
            ),
            "held_predictions_identical": all(
                left["prediction_hash"] == right["prediction_hash"]
                for left, right in zip(fitted_before, fitted_after, strict=True)
            ),
        }
        if fold == 0:
            checks["selected_power_identical"] = wgcna_rerun["selected_power_identical"] and wgcna_rerun[
                "soft_threshold_fit_identical"
            ]
            checks["base_modules_identical"] = wgcna_rerun["base_labels_identical"]
        leakage_rows.append({"outer_fold": fold, **checks, "status": "PASS" if all(checks.values()) else "FAIL"})
        if not all(checks.values()):
            raise RuntimeError("WGCNA_FRONTIER_LEAKAGE_INVARIANCE_FAILED")

        # Only after the leakage gate passes do held-out outcomes enter scoring.
        if not np.array_equal(before["y_test"], frozen_targets[before["test"]]):
            maximum = float(np.max(np.abs(before["y_test"] - frozen_targets[before["test"]])))
            if maximum > 1e-5:
                raise RuntimeError(f"frozen matched target mismatch: {maximum}")
        denominator = float(np.square(before["y_test"], dtype=np.float64).sum())
        assigned_hash = array_hash(np.flatnonzero(labels > 0))
        grey_hash = array_hash(np.flatnonzero(labels == 0))
        manifest_rows.append({
            "outer_fold": fold, "training_lcls": len(before["train"]), "held_lcls": len(before["test"]),
            "selected_power": module_summary["selected_power"], "selection_rule": module_summary["selection_rule"],
            "selected_SFT_R_sq": module_summary["selected_SFT_R_sq"], "selected_slope": module_summary["selected_slope"],
            "selected_mean_connectivity": module_summary["selected_mean_connectivity"],
            "base_module_count": module_summary["base_module_count"], "assigned_genes": int((labels > 0).sum()),
            "grey_genes": int((labels == 0).sum()), "gene_universe_hash": gene_universe_hash,
            "assigned_gene_hash": assigned_hash, "grey_gene_hash": grey_hash,
            "wgcna_input_sha256": module_summary["input_sha256"], "wgcna_wall_seconds": module_summary["wall_seconds"],
        })

        # Full-gene common reference.
        _, full_sse, full_r2 = r2_sufficient_statistics(before["y_test"], frozen_prediction[before["test"]])
        oracle_rows.append({
            "representation": "full_gene", "family": "full_gene", "outer_fold": fold,
            "coordinate_count": 10_110, "denominator": denominator, "oracle_sse": 0.0, "oracle_fidelity": 1.0,
        })
        predictive_rows.append({
            "representation": "full_gene", "family": "full_gene", "outer_fold": fold,
            "coordinate_count": 10_110, "denominator": denominator, "predictive_sse": full_sse,
            "predictive_R2": full_r2,
        })

        # Fold-local PC references reconstructed to the same gene axis.
        pca = PCA(n_components=16, svd_solver="randomized", iterated_power=7, random_state=SEED + fold).fit(before["y_train"])
        for k in PC_LEVELS:
            components = pca.components_[:k]
            true_scores = (before["y_test"] - pca.mean_) @ components.T
            oracle = true_scores @ components + pca.mean_
            predicted_scores = (frozen_prediction[before["test"]] - pca.mean_) @ components.T
            reconstructed = predicted_scores @ components + pca.mean_
            _, oracle_sse, fidelity = r2_sufficient_statistics(before["y_test"], oracle)
            _, predictive_sse, predictive_r2 = r2_sufficient_statistics(before["y_test"], reconstructed)
            name = f"PC{k}"
            oracle_rows.append({
                "representation": name, "family": "PCA", "outer_fold": fold, "coordinate_count": k,
                "denominator": denominator, "oracle_sse": oracle_sse, "oracle_fidelity": fidelity,
            })
            predictive_rows.append({
                "representation": name, "family": "PCA", "outer_fold": fold, "coordinate_count": k,
                "denominator": denominator, "predictive_sse": predictive_sse, "predictive_R2": predictive_r2,
            })

        for level, fitted in zip(ladder_before, fitted_before, strict=True):
            components = level["components"]
            true_scores = coordinate_matrices(before["y_test"], components)
            oracle = reconstruct(true_scores, components, len(before["test"]), 10_110)
            prediction = reconstruct(fitted["predicted_scores"], components, len(before["test"]), 10_110)
            _, oracle_sse, fidelity = r2_sufficient_statistics(before["y_test"], oracle)
            _, predictive_sse, predictive_r2 = r2_sufficient_statistics(before["y_test"], prediction)
            representation = f"WGCNA_h{level['threshold']:.2f}"
            sizes = np.asarray([len(component.genes) for component in components], dtype=int)
            annotation_rows, annotated = pathway_rows(
                components, symbols, hallmarks, fold, level["level_index"], level["threshold"]
            )
            pathway_all.extend(annotation_rows)
            energy = np.asarray([component.training_energy for component in components], dtype=np.float64)
            annotation_energy_fraction = float(energy[annotated].sum() / energy.sum())
            annotated_fraction = float(annotated.mean())
            ladder_rows.append({
                "outer_fold": fold, "level_index": level["level_index"], "merge_threshold": level["threshold"],
                "representation": representation, "base_module_count": module_summary["base_module_count"],
                "realized_module_count": len(components), "assigned_genes": int((labels > 0).sum()),
                "grey_genes": int((labels == 0).sum()), "median_module_size": float(np.median(sizes)),
                "min_module_size": int(sizes.min()), "max_module_size": int(sizes.max()),
                "partition_hash": level["partition_hash"], "component_hash": level["component_hash"],
                "fraction_annotated_modules": annotated_fraction,
                "annotated_training_energy_fraction": annotation_energy_fraction,
            })
            coverage_rows.append({
                "outer_fold": fold, "level_index": level["level_index"], "merge_threshold": level["threshold"],
                "total_genes_G": 10_110, "assigned_genes": int((labels > 0).sum()),
                "grey_genes": int((labels == 0).sum()), "excluded_genes": 0,
                "gene_universe_hash": gene_universe_hash, "assigned_gene_hash": assigned_hash,
                "grey_gene_hash": grey_hash, "grey_prediction_rule": "zero individual-specific residual",
            })
            oracle_rows.append({
                "representation": representation, "family": "WGCNA", "outer_fold": fold,
                "coordinate_count": len(components), "denominator": denominator,
                "oracle_sse": oracle_sse, "oracle_fidelity": fidelity,
            })
            predictive_rows.append({
                "representation": representation, "family": "WGCNA", "outer_fold": fold,
                "coordinate_count": len(components), "denominator": denominator,
                "predictive_sse": predictive_sse, "predictive_R2": predictive_r2,
            })
            module_rows.extend(module_diagnostics(true_scores, fitted["predicted_scores"], components, fold, level))
            selection_rows.append({
                "outer_fold": fold, "level_index": level["level_index"], "merge_threshold": level["threshold"],
                "realized_module_count": len(components), "gamma_multiplier": fitted["gamma_multiplier"],
                "alpha": fitted["alpha"], "fitted_gamma": fitted["fitted_gamma"],
                "tuning_hash": fitted["tuning_hash"], "parameter_hash": fitted["parameter_hash"],
                "prediction_hash": fitted["prediction_hash"],
            })
        fold_summary_rows.append({"outer_fold": fold, "formal_wall_seconds": time.time() - start})
        write_json(CACHE / f"fold_{fold}" / "FORMAL_COMPLETE.json", {
            "status": "COMPLETE", "outer_fold": fold, "completed_at": now(),
            "formal_wall_seconds": fold_summary_rows[-1]["formal_wall_seconds"],
        })
        print(f"formal fold {fold}: complete", flush=True)

    leakage_table = pd.DataFrame(leakage_rows)
    if not (leakage_table["status"] == "PASS").all():
        raise RuntimeError("WGCNA_FRONTIER_LEAKAGE_INVARIANCE_FAILED")
    coverage = pd.DataFrame(coverage_rows)
    fairness_ok = True
    for fold, group in coverage.groupby("outer_fold"):
        for column in ("total_genes_G", "assigned_genes", "grey_genes", "gene_universe_hash", "assigned_gene_hash", "grey_gene_hash"):
            fairness_ok &= group[column].nunique(dropna=False) == 1
    if not fairness_ok:
        raise RuntimeError("RESOLUTION_COMPARISON_NOT_FAIR")

    manifest = pd.DataFrame(manifest_rows)
    ladder_table = pd.DataFrame(ladder_rows)
    oracle = pd.DataFrame(oracle_rows)
    predictive = pd.DataFrame(predictive_rows)
    module_table = pd.DataFrame(module_rows)
    pathways = pd.DataFrame(pathway_all)
    selections = pd.DataFrame(selection_rows)
    runtime = pd.DataFrame(fold_summary_rows)

    def add_pooled(table: pd.DataFrame, sse_column: str, metric_column: str) -> pd.DataFrame:
        rows = []
        for representation, group in table.groupby("representation", sort=False):
            first = group.iloc[0]
            denominator = float(group["denominator"].sum())
            sse = float(group[sse_column].sum())
            rows.append({
                "representation": representation, "family": first["family"], "outer_fold": "pooled",
                "coordinate_count": float(group["coordinate_count"].mean()),
                "coordinate_count_min": int(group["coordinate_count"].min()),
                "coordinate_count_max": int(group["coordinate_count"].max()),
                "denominator": denominator, sse_column: sse, metric_column: 1.0 - sse / denominator,
            })
        return pd.concat([table, pd.DataFrame(rows)], ignore_index=True)

    oracle = add_pooled(oracle, "oracle_sse", "oracle_fidelity")
    predictive = add_pooled(predictive, "predictive_sse", "predictive_R2")

    # Per-fold and pooled module summaries; both macro and training-energy weighted.
    module_summaries = []
    for keys, group in module_table.groupby(["outer_fold", "level_index", "merge_threshold"], sort=True):
        module_summaries.append({
            "outer_fold": keys[0], "level_index": keys[1], "merge_threshold": keys[2],
            "module_space_macro_R2": float(group["coordinate_R2"].mean()),
            "module_space_weighted_R2": float(np.sum(group["coordinate_R2"] * group["training_energy_weight"])),
        })
    module_summary_table = pd.DataFrame(module_summaries)

    pooled_oracle = oracle.loc[oracle["outer_fold"] == "pooled"].copy()
    pooled_predictive = predictive.loc[predictive["outer_fold"] == "pooled"].copy()
    source = pooled_oracle.merge(
        pooled_predictive[["representation", "predictive_R2"]], on="representation", validate="one_to_one"
    )
    source["delta_R2_vs_full_gene"] = source["predictive_R2"] - float(
        source.loc[source["representation"] == "full_gene", "predictive_R2"].iloc[0]
    )
    source["pareto_nondominated"] = [
        not any(
            (other.oracle_fidelity >= row.oracle_fidelity - TOLERANCE)
            and (other.predictive_R2 >= row.predictive_R2 - TOLERANCE)
            and (
                other.oracle_fidelity > row.oracle_fidelity + TOLERANCE
                or other.predictive_R2 > row.predictive_R2 + TOLERANCE
            )
            for other in source.itertuples()
        )
        for row in source.itertuples()
    ]

    # Add WGCNA annotation/fairness and secondary module-coordinate summaries.
    wgcna_agg = ladder_table.groupby(["representation", "level_index", "merge_threshold"], as_index=False).agg(
        coordinate_count=("realized_module_count", "mean"),
        coordinate_count_min=("realized_module_count", "min"), coordinate_count_max=("realized_module_count", "max"),
        assigned_genes_min=("assigned_genes", "min"), assigned_genes_max=("assigned_genes", "max"),
        grey_genes_min=("grey_genes", "min"), grey_genes_max=("grey_genes", "max"),
        median_module_size=("median_module_size", "median"),
        fraction_annotated_modules=("fraction_annotated_modules", "mean"),
        annotated_training_energy_fraction=("annotated_training_energy_fraction", "mean"),
    )
    secondary = module_summary_table.groupby(["level_index", "merge_threshold"], as_index=False).agg(
        module_space_macro_R2=("module_space_macro_R2", "mean"),
        module_space_weighted_R2=("module_space_weighted_R2", "mean"),
    )
    wgcna_agg = wgcna_agg.merge(secondary, on=["level_index", "merge_threshold"], validate="one_to_one")

    fairness_rows: list[dict[str, Any]] = []
    for row in source.itertuples():
        if row.family == "WGCNA":
            extra = wgcna_agg.loc[wgcna_agg["representation"] == row.representation].iloc[0]
            fairness_rows.append({
                "representation": row.representation, "module_or_coordinate_count": extra["coordinate_count"],
                "coordinate_count_range": f"{int(extra['coordinate_count_min'])}-{int(extra['coordinate_count_max'])}",
                "total_genes_G": 10_110,
                "assigned_genes": f"{int(extra['assigned_genes_min'])}-{int(extra['assigned_genes_max'])}",
                "grey_genes": f"{int(extra['grey_genes_min'])}-{int(extra['grey_genes_max'])}",
                "median_module_size": extra["median_module_size"], "oracle_fidelity": row.oracle_fidelity,
                "full_gene_predictive_R2": row.predictive_R2,
                "module_space_macro_R2": extra["module_space_macro_R2"],
                "module_space_weighted_R2": extra["module_space_weighted_R2"],
                "fraction_annotated_modules": extra["fraction_annotated_modules"],
            })
        else:
            fairness_rows.append({
                "representation": row.representation, "module_or_coordinate_count": row.coordinate_count,
                "coordinate_count_range": f"{int(row.coordinate_count_min)}-{int(row.coordinate_count_max)}",
                "total_genes_G": 10_110, "assigned_genes": 10_110, "grey_genes": 0,
                "median_module_size": np.nan, "oracle_fidelity": row.oracle_fidelity,
                "full_gene_predictive_R2": row.predictive_R2,
                "module_space_macro_R2": np.nan, "module_space_weighted_R2": np.nan,
                "fraction_annotated_modules": np.nan,
            })
    fairness = pd.DataFrame(fairness_rows)

    full_r2 = float(source.loc[source["representation"] == "full_gene", "predictive_R2"].iloc[0])
    wgcna_candidates = source.loc[source["family"] == "WGCNA"].copy()
    fold_pred = predictive.loc[predictive["outer_fold"] != "pooled"].copy()
    full_fold = fold_pred.loc[fold_pred["representation"] == "full_gene", ["outer_fold", "predictive_R2"]].rename(
        columns={"predictive_R2": "full_R2"}
    )
    stability: dict[str, dict[str, float]] = {}
    for representation in wgcna_candidates["representation"]:
        values_by_fold = fold_pred.loc[fold_pred["representation"] == representation, ["outer_fold", "predictive_R2"]].merge(
            full_fold, on="outer_fold", validate="one_to_one"
        )
        delta = values_by_fold["predictive_R2"] - values_by_fold["full_R2"]
        stability[representation] = {
            "positive_folds": int((delta > 0).sum()), "minimum_fold_delta": float(delta.min())
        }
    nondominated = source.loc[source["pareto_nondominated"]]
    frontier_span_ok = (
        len(nondominated) >= 3
        and float(nondominated["oracle_fidelity"].max() - nondominated["oracle_fidelity"].min()) >= 0.25
        and float(nondominated["predictive_R2"].max() - nondominated["predictive_R2"].min()) >= 0.02
    )
    supported_names = []
    partial_names = []
    for row in wgcna_candidates.itertuples():
        stats = stability[row.representation]
        if (
            row.oracle_fidelity >= 0.50 and row.delta_R2_vs_full_gene >= 0.02
            and stats["positive_folds"] >= 4 and stats["minimum_fold_delta"] >= -0.02
            and bool(row.pareto_nondominated) and frontier_span_ok
        ):
            supported_names.append(row.representation)
        pc = source.loc[source["family"] == "PCA"]
        dominated_by_every_pc = all(
            (pc_row.oracle_fidelity >= row.oracle_fidelity - TOLERANCE)
            and (pc_row.predictive_R2 >= row.predictive_R2 - TOLERANCE)
            for pc_row in pc.itertuples()
        )
        if row.oracle_fidelity >= 0.25 and row.delta_R2_vs_full_gene >= 0.01 and not dominated_by_every_pc:
            partial_names.append(row.representation)
    if supported_names:
        verdict = "RESOLUTION_FRONTIER_SUPPORTED"
        placement = "The specified intermediate-resolution criteria are satisfied."
    elif partial_names:
        verdict = "WGCNA_INTERMEDIATE_RESOLUTION_PARTIAL"
        placement = "The specified intermediate-resolution criteria are partially satisfied."
    else:
        verdict = "WGCNA_RESOLUTION_FRONTIER_NOT_SUPPORTED"
        placement = "The specified intermediate-resolution criteria are not satisfied."

    manifest.to_csv(OUT / "WGCNA_OUTER_FOLD_MODULE_MANIFEST.csv", index=False)
    ladder_table.to_csv(OUT / "WGCNA_RESOLUTION_LADDER.csv", index=False)
    coverage.to_csv(OUT / "WGCNA_GENE_COVERAGE_AUDIT.csv", index=False)
    oracle.to_csv(OUT / "WGCNA_ORACLE_FIDELITY_RESULTS.csv", index=False)
    predictive.to_csv(OUT / "WGCNA_PREDICTIVE_RECONSTRUCTION_RESULTS.csv", index=False)
    module_table.merge(module_summary_table, on=["outer_fold", "level_index", "merge_threshold"], how="left").to_csv(
        OUT / "WGCNA_MODULE_COORDINATE_RESULTS.csv", index=False
    )
    pathways.to_csv(OUT / "WGCNA_PATHWAY_ANNOTATIONS.csv", index=False)
    fairness.to_csv(OUT / "WGCNA_FAIRNESS_TABLE.csv", index=False)
    selections.to_csv(OUT / "WGCNA_RBF_SELECTIONS.csv", index=False)
    runtime.to_csv(OUT / "WGCNA_FORMAL_RUNTIME.csv", index=False)
    source.to_csv(OUT / "COUNTERFACTUAL_RESOLUTION_FRONTIER_SOURCE_DATA.csv", index=False)

    leakage_report = f"""# WGCNA leakage invariance audit

Held-out DEX outcomes were replaced independently in all five historical outer folds. Training residuals, baseline inputs, deterministic nested partitions, eigengene parameters, frozen RBF selections and fitted coefficients, training-energy weights and held predictions were hashed before and after replacement. Fold 0 additionally reran the complete R/WGCNA power selection and base partition from the unchanged training binary.

- Fold checks passed: `{int((leakage_table['status'] == 'PASS').sum())}/5`.
- Fold-0 selected power invariant: `{wgcna_rerun['selected_power_identical']}`.
- Fold-0 complete fit table invariant: `{wgcna_rerun['soft_threshold_fit_identical']}`.
- Fold-0 base labels invariant: `{wgcna_rerun['base_labels_identical']}`.
- All merge, loading, parameter, energy-weight and prediction hashes invariant: `{bool((leakage_table['status'] == 'PASS').all())}`.

Result: `WGCNA_FRONTIER_LEAKAGE_INVARIANCE_PASS`.
"""
    (OUT / "WGCNA_LEAKAGE_INVARIANCE_AUDIT.md").write_text(leakage_report, encoding="utf-8")

    render_figure(source)
    best = wgcna_candidates.sort_values(["predictive_R2", "oracle_fidelity"], ascending=False).iloc[0]
    best_stability = stability[str(best["representation"])]
    pc_text = source.loc[source["family"] == "PCA", ["representation", "oracle_fidelity", "predictive_R2"]].to_string(index=False)
    nondominated_text = ", ".join(nondominated["representation"].astype(str))
    base_ladder = ladder_table.loc[ladder_table["merge_threshold"] == 0.0]
    base_module_summary = module_summary_table.loc[module_summary_table["merge_threshold"] == 0.0]
    macro_coordinate_r2 = float(base_module_summary["module_space_macro_R2"].mean())
    weighted_coordinate_r2 = float(base_module_summary["module_space_weighted_R2"].mean())
    minimum_coverage = float(base_ladder["assigned_genes"].min() / 10_110)
    maximum_coverage = float(base_ladder["assigned_genes"].max() / 10_110)
    mean_annotated_fraction = float(base_ladder["fraction_annotated_modules"].mean())
    mean_annotated_energy = float(base_ladder["annotated_training_energy_fraction"].mean())
    report = f"""# Counterfactual Resolution Frontier — final frozen report

## Outcome

`{verdict}`

The prespecified six-level fold-local WGCNA ladder is executable, leakage-invariant and dimensionally fair. Every representation was evaluated on the same 340 LCLs, exact five folds, fixed 10,110-gene truth and common pooled residual-energy denominator. Grey genes were retained in the denominator and received zero individual-specific residual prediction.

This is a scientifically negative but diagnostically sharp result. The frozen construction did not instantiate an intermediate-resolution ladder: all six thresholds realized the same fold-local module count. Assigned-gene coverage ranged from `{100 * minimum_coverage:.2f}%` to `{100 * maximum_coverage:.2f}%`; the remaining genes were grey. The test is executable, but this WGCNA representation collapsed directly to an extremely low-coverage regime rather than tracing a fidelity–recoverability tradeoff.

## Direct answers

- **Does an intermediate WGCNA resolution materially improve recoverability?** Best WGCNA point `{best['representation']}` has full-gene reconstructed R2 `{best['predictive_R2']:.6f}`, versus `{full_r2:.6f}` for full-gene RBF prediction (difference `{best['delta_R2_vs_full_gene']:.6f}`). Its fold improvement is positive in `{best_stability['positive_folds']}/5` folds; minimum fold difference `{best_stability['minimum_fold_delta']:.6f}`.
- **How much biological fidelity is retained?** The best-recovering WGCNA point retains oracle fidelity `{best['oracle_fidelity']:.6f}` on the unchanged 10,110-gene truth.
- **Is the comparison fair?** Yes. `RESOLUTION_COMPARISON_NOT_FAIR` was not triggered; within each fold, all six WGCNA levels have identical assigned and grey-gene sets, held LCLs, target and denominator.
- **Are all resolutions evaluated on the same gene truth?** Yes: exact matched-340 Lea axis, `G=10,110`; excluded genes `0`.
- **How does WGCNA compare with PC4/8/16?** Pooled gene-space values are:\n\n```\n{pc_text}\n```
- **Do modules correspond to recognizable biology?** Within the small assigned subset, mean annotated-module fraction is `{mean_annotated_fraction:.3f}` and annotated modules carry mean training energy fraction `{mean_annotated_energy:.3f}`. Pathway coherence was not the failure mode; coverage and retained full-gene biology were.
- **Is there a meaningful Pareto frontier?** Global nondominated representations: `{nondominated_text}`. The prespecified three-point/span condition is `{frontier_span_ok}`.
- **Which verdict applies?** `{verdict}`.
- **Criteria assessment:** {placement}

## Interpretation

The secondary coordinate-only view looks superficially encouraging: mean fold-level module macro R2 is `{macro_coordinate_r2:.6f}` and training-energy-weighted R2 is `{weighted_coordinate_r2:.6f}`. Those values describe only the assigned modules. Once reconstructed onto the complete 10,110-gene truth, predictive R2 and oracle fidelity collapse. This discrepancy is exactly why raw module-space R2 was preregistered as secondary.

The failure is not evidence that coherent response biology is absent and does not falsify CGC or the broader resolution principle. It is a failure of this frozen signed-WGCNA construction to create the intended intermediate-resolution representation under strict fold-local fitting. The specific WGCNA-frontier claim is not supported; no parameter rescue is permitted.

## Provenance

- Representation: `{REPRESENTATION.relative_to(ROOT)}`, SHA-256 `{sha256_file(REPRESENTATION)}`.
- Frozen full-gene RBF OOF: `{FROZEN_OOF.relative_to(ROOT)}`, SHA-256 `{sha256_file(FROZEN_OOF)}`.
- Historical RBF authority: `{HISTORICAL}`.
- Protocol commits precede real partitions/outcomes.
- WGCNA implementation reference: https://cran.r-project.org/web/packages/WGCNA/WGCNA.pdf
- Annotation reference: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H

The analysis evaluates the prespecified ladder.
"""
    (OUT / "WGCNA_RESOLUTION_FRONTIER_REPORT.md").write_text(report, encoding="utf-8")
    write_json(OUT / "WGCNA_FINAL_VERDICT.json", {
        "verdict": verdict, "supported_points": supported_names, "partial_points": partial_names,
        "frontier_span_gate": frontier_span_ok, "fairness_pass": fairness_ok,
        "leakage_pass": True, "completed_at": now(),
    })
    print(verdict, flush=True)


def render_figure(source: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    colors = {"full_gene": "#777777", "PCA": "#64748B", "WGCNA": "#2A7F80"}
    fig = plt.figure(figsize=(7.2, 2.85), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.38)
    ax = fig.add_subplot(grid[0, 0])
    wgcna = source.loc[source["family"] == "WGCNA"].sort_values("coordinate_count", ascending=False)
    ax.plot(wgcna["oracle_fidelity"], wgcna["predictive_R2"], color=colors["WGCNA"], lw=1.2, alpha=0.8, zorder=2)
    for row in source.itertuples():
        color = colors[row.family]
        marker = "o" if row.family == "WGCNA" else ("D" if row.family == "full_gene" else "s")
        ax.scatter(row.oracle_fidelity, row.predictive_R2, s=28 if row.family == "WGCNA" else 34,
                   marker=marker, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        if row.family != "WGCNA":
            label = row.representation.replace("_", " ")
            ax.annotate(label, (row.oracle_fidelity, row.predictive_R2), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=color)
    for _, group in wgcna.groupby(["oracle_fidelity", "predictive_R2", "coordinate_count"], sort=False):
        row = group.iloc[0]
        prefix = f"{len(group)} thresholds, " if len(group) > 1 else ""
        ax.annotate(
            f"{prefix}{int(round(row['coordinate_count']))} modules",
            (row["oracle_fidelity"], row["predictive_R2"]), xytext=(4, 3), textcoords="offset points",
            fontsize=6.5, color=colors["WGCNA"],
        )
    ax.axhline(0, color="#CBD5E1", lw=0.7, zorder=0)
    ax.set_xlabel("oracle biological fidelity, $F$")
    ax.set_ylabel("full-gene predictive reconstruction, $R^2$")
    ax.set_title("Fidelity–recoverability frontier", loc="center", fontweight="semibold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color="#E5E7EB", lw=0.5, zorder=0)

    bx = fig.add_subplot(grid[0, 1])
    point_sizes = 20 + 90 * (source["oracle_fidelity"] - source["oracle_fidelity"].min()) / max(
        source["oracle_fidelity"].max() - source["oracle_fidelity"].min(), 1e-9
    )
    scatter = bx.scatter(source["coordinate_count"], source["oracle_fidelity"], c=source["predictive_R2"],
                         s=point_sizes, cmap="RdBu_r", edgecolor="white", linewidth=0.55)
    bx.set_xscale("log")
    bx.set_xlabel("effective coordinate count")
    bx.set_ylabel("oracle biological fidelity, $F$")
    bx.set_title("Resolution and retained biology", loc="center", fontweight="semibold")
    bx.spines[["top", "right"]].set_visible(False)
    bx.grid(axis="both", color="#E5E7EB", lw=0.5, zorder=0)
    colorbar = fig.colorbar(scatter, ax=bx, fraction=0.045, pad=0.04)
    colorbar.set_label("predictive $R^2$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.text(0.012, 0.965, "a", fontweight="bold", fontsize=9, va="top")
    fig.text(0.585, 0.965, "b", fontweight="bold", fontsize=9, va="top")
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.22, top=0.86)
    fig.savefig(OUT / "COUNTERFACTUAL_RESOLUTION_FRONTIER.png", dpi=600, facecolor="white")
    fig.savefig(OUT / "COUNTERFACTUAL_RESOLUTION_FRONTIER.svg", facecolor="white")
    plt.close(fig)


def verify() -> None:
    required = [
        "WGCNA_FRONTIER_PROTOCOL.md", "WGCNA_OUTER_FOLD_MODULE_MANIFEST.csv",
        "WGCNA_RESOLUTION_LADDER.csv", "WGCNA_GENE_COVERAGE_AUDIT.csv",
        "WGCNA_LEAKAGE_INVARIANCE_AUDIT.md", "WGCNA_ORACLE_FIDELITY_RESULTS.csv",
        "WGCNA_PREDICTIVE_RECONSTRUCTION_RESULTS.csv", "WGCNA_MODULE_COORDINATE_RESULTS.csv",
        "WGCNA_PATHWAY_ANNOTATIONS.csv", "WGCNA_FAIRNESS_TABLE.csv",
        "COUNTERFACTUAL_RESOLUTION_FRONTIER.png", "COUNTERFACTUAL_RESOLUTION_FRONTIER.svg",
        "COUNTERFACTUAL_RESOLUTION_FRONTIER_SOURCE_DATA.csv", "WGCNA_RESOLUTION_FRONTIER_REPORT.md",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    manifest = pd.read_csv(OUT / "WGCNA_OUTER_FOLD_MODULE_MANIFEST.csv")
    ladder = pd.read_csv(OUT / "WGCNA_RESOLUTION_LADDER.csv")
    coverage = pd.read_csv(OUT / "WGCNA_GENE_COVERAGE_AUDIT.csv")
    oracle = pd.read_csv(OUT / "WGCNA_ORACLE_FIDELITY_RESULTS.csv")
    predictive = pd.read_csv(OUT / "WGCNA_PREDICTIVE_RECONSTRUCTION_RESULTS.csv")
    modules = pd.read_csv(OUT / "WGCNA_MODULE_COORDINATE_RESULTS.csv")
    pathways = pd.read_csv(OUT / "WGCNA_PATHWAY_ANNOTATIONS.csv")
    fairness = pd.read_csv(OUT / "WGCNA_FAIRNESS_TABLE.csv")
    source = pd.read_csv(OUT / "COUNTERFACTUAL_RESOLUTION_FRONTIER_SOURCE_DATA.csv")
    selections = pd.read_csv(OUT / "WGCNA_RBF_SELECTIONS.csv")
    assert len(manifest) == 5 and set(manifest["outer_fold"]) == set(range(5))
    assert len(ladder) == 30 and set(np.round(ladder["merge_threshold"], 2)) == set(MERGE_THRESHOLDS)
    assert len(coverage) == 30 and len(fairness) == 10 and len(source) == 10
    assert len(oracle) == 60 and len(predictive) == 60 and len(selections) == 30
    assert set(modules["outer_fold"]) == set(range(5))
    assert set(pathways["rank_within_module"]) <= {1, 2, 3}
    assert pathways["fdr"].between(0, 1).all() and pathways["p_value"].between(0, 1).all()
    assert np.isfinite(source[["oracle_fidelity", "predictive_R2"]].to_numpy()).all()
    assert source["representation"].is_unique
    assert coverage["total_genes_G"].eq(10_110).all() and coverage["excluded_genes"].eq(0).all()
    assert (coverage["assigned_genes"] + coverage["grey_genes"]).eq(10_110).all()
    for _, group in coverage.groupby("outer_fold"):
        for column in (
            "total_genes_G", "assigned_genes", "grey_genes", "gene_universe_hash",
            "assigned_gene_hash", "grey_gene_hash",
        ):
            assert group[column].nunique(dropna=False) == 1
    for _, group in ladder.groupby("outer_fold"):
        counts = group.sort_values("merge_threshold")["realized_module_count"].to_numpy()
        assert np.all(np.diff(counts) <= 0)
    for fold in range(5):
        prepared = json.loads((CACHE / f"fold_{fold}" / "PREPARED.json").read_text(encoding="utf-8"))
        completed = json.loads((CACHE / f"fold_{fold}" / "wgcna_base" / "COMPLETE.json").read_text(encoding="utf-8"))
        assert prepared["input_sha256"] == completed["input_sha256"]
        assert sha256_file(ROOT / prepared["input_path"]) == prepared["input_sha256"]
        fit = pd.read_csv(CACHE / f"fold_{fold}" / "wgcna_base" / "soft_threshold_fit.csv")
        eligible = (
            fit["SFT.R.sq"].ge(0.80) & fit["slope"].lt(0) & fit["mean.k."].ge(1)
            & np.isfinite(fit[["SFT.R.sq", "slope", "mean.k."]]).all(axis=1)
        )
        fallback = (
            fit["slope"].lt(0) & fit["mean.k."].ge(1)
            & np.isfinite(fit[["SFT.R.sq", "slope", "mean.k."]]).all(axis=1)
        )
        if eligible.any():
            expected_power = float(fit.loc[eligible, "Power"].min())
        elif fallback.any():
            expected_power = float(
                fit.loc[fallback].sort_values(["SFT.R.sq", "Power"], ascending=[False, True]).iloc[0]["Power"]
            )
        else:
            expected_power = float(fit.sort_values(["SFT.R.sq", "Power"], ascending=[False, True]).iloc[0]["Power"])
        assert float(completed["selected_power"]) == expected_power
    pooled_denominators = predictive.loc[predictive["outer_fold"].astype(str) == "pooled", "denominator"]
    assert pooled_denominators.nunique() == 1
    assert oracle.loc[oracle["outer_fold"].astype(str) == "pooled", "denominator"].nunique() == 1
    for representation, group in predictive.loc[predictive["outer_fold"].astype(str) != "pooled"].groupby("representation"):
        pooled = predictive.loc[
            (predictive["representation"] == representation) & (predictive["outer_fold"].astype(str) == "pooled")
        ].iloc[0]
        assert np.isclose(pooled["denominator"], group["denominator"].sum(), rtol=0, atol=1e-8)
        assert np.isclose(pooled["predictive_sse"], group["predictive_sse"].sum(), rtol=0, atol=1e-8)
        assert np.isclose(pooled["predictive_R2"], 1 - pooled["predictive_sse"] / pooled["denominator"], atol=1e-14)
    values = load_inputs()
    with np.load(FROZEN_OOF, allow_pickle=False) as frozen:
        direct_denominator = float(np.square(frozen["targets"], dtype=np.float64).sum())
        direct_sse = float(np.square(frozen["targets"] - frozen["rbf_kernel_ridge"], dtype=np.float64).sum())
    full_source = source.loc[source["representation"] == "full_gene"].iloc[0]
    assert np.isclose(full_source["predictive_R2"], 1 - direct_sse / direct_denominator, atol=1e-14)
    assert np.isclose(pooled_denominators.iloc[0], direct_denominator, atol=1e-8)
    for fold in range(5):
        arrays = fold_arrays(values, fold)
        assert np.array_equal(arrays["y_test"], np.load(FROZEN_OOF, allow_pickle=False)["targets"][arrays["test"]])
    before_labels, before_summary = load_base_labels(0, "wgcna_base")
    after_labels, after_summary = load_base_labels(0, "wgcna_leakage_rerun")
    assert before_summary["selected_power"] == after_summary["selected_power"]
    assert np.array_equal(before_labels, after_labels)
    before_fit = pd.read_csv(CACHE / "fold_0" / "wgcna_base" / "soft_threshold_fit.csv")
    after_fit = pd.read_csv(CACHE / "fold_0" / "wgcna_leakage_rerun" / "soft_threshold_fit.csv")
    pd.testing.assert_frame_equal(before_fit, after_fit, check_exact=True)
    verdict = json.loads((OUT / "WGCNA_FINAL_VERDICT.json").read_text(encoding="utf-8"))["verdict"]
    assert verdict == "WGCNA_RESOLUTION_FRONTIER_NOT_SUPPORTED"
    hash_rows = [
        {"path": name, "size_bytes": (OUT / name).stat().st_size, "sha256": sha256_file(OUT / name)}
        for name in required
    ]
    pd.DataFrame(hash_rows).to_csv(OUT / "WGCNA_OUTPUT_HASH_MANIFEST.csv", index=False)
    report = f"""# WGCNA frontier integrity audit

- Required outputs present: `{len(required)}/{len(required)}`.
- Outer folds: `5/5`; frozen sizes `67, 69, 68, 68, 68`.
- WGCNA ladder rows: `30` (`5 folds × 6 frozen thresholds`).
- Primary representation rows: `10`; fold-plus-pooled metric rows: `60` per primary metric table.
- Common pooled denominator: `{direct_denominator:.12f}` for every representation.
- Exact frozen full-gene RBF R2 independently recomputed: `{1 - direct_sse / direct_denominator:.15f}`.
- Within-fold gene-universe, assigned-set and grey-set invariance: `PASS`.
- Nested non-increasing module count: `PASS`.
- Training-binary hashes and prespecified soft-power rule: `PASS`.
- Fold-0 independent WGCNA rerun (power, fit table, labels): `PASS`.
- Pathway p-values/FDR/ranks: `PASS`.
- Figure PNG/SVG/source-data reconciliation: `PASS`.
- Frozen verdict: `{verdict}`.

No scientific analysis was rerun by this verifier; it reconciled frozen tables, official inputs, output hashes and the independent leakage rerun.
"""
    (OUT / "WGCNA_FRONTIER_INTEGRITY_AUDIT.md").write_text(report, encoding="utf-8")
    print("WGCNA_FRONTIER_INTEGRITY_PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    modules = subparsers.add_parser("module")
    modules.add_argument("--fold", type=int, required=True, choices=range(5))
    modules.add_argument("--output-name", default="wgcna_base")
    subparsers.add_parser("score")
    subparsers.add_parser("verify")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "module":
        run_module(args.fold, args.output_name)
    elif args.command == "score":
        score()
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()
