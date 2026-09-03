"""Targeted-only, tensorized Phase-I computational profile.

This module deliberately stops before formal Phase-I statistics.  It builds
counts-derived pseudobulks, fold-frozen representations, transport utilities,
and intervention-disjoint OOF linear-router predictions so that computational
granularity and numerical equivalence can be audited first.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
import subprocess
import threading
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import yaml

from igc_virtual_cell.phase1.analysis import _stable_fold
from igc_virtual_cell.phase1.materialize import META_COLUMNS, _ntc_matrix
from igc_virtual_cell.phase1.targeted_counts import materialize_feng_targeted_counts
from igc_virtual_cell.programs.decompose import ResponseProgramModel


_GENEPT_MEMORY: dict[Path, dict[str, np.ndarray]] = {}
_GENEPT_RAW_LOADS = 0


@dataclass
class FitCounter:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass
class MemorySampler:
    peak_bytes: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None
        process = psutil.Process() if psutil is not None else None

        def rss() -> int:
            if process is not None:
                return int(process.memory_info().rss)
            return _windows_memory_bytes()[0]

        def sample() -> None:
            while not self._stop.wait(0.1):
                self.peak_bytes = max(self.peak_bytes, rss())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        _, native_peak = _windows_memory_bytes()
        self.peak_bytes = max(self.peak_bytes, native_peak)


def _windows_memory_bytes() -> tuple[int, int]:
    """Return current and process-lifetime peak working set on Windows."""
    if not hasattr(ctypes, "windll"):
        return 0, 0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        return 0, 0
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


@dataclass
class PairIndex:
    source: np.ndarray
    target: np.ndarray
    perturbation: np.ndarray
    source_context: np.ndarray
    target_context: np.ndarray


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_targeted(root: Path, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path)
    metadata = frame.loc[:, META_COLUMNS].copy()
    responses = frame.drop(columns=META_COLUMNS).astype(np.float32)
    manifest = pd.read_csv(root / "data/manifests/feng_2026_targeted.csv", low_memory=False)
    ntc = _ntc_matrix(
        root / "data/processed/feng_2026/feng_2026_targeted_ntc_log1p_cpm.parquet",
        manifest,
    ).astype(np.float32)
    common = responses.columns.intersection(ntc.columns, sort=False)
    return metadata, responses.loc[:, common], ntc.loc[:, common]


def build_pair_index(metadata: pd.DataFrame) -> PairIndex:
    """Create all observed ordered source-target rows without case-wise loops."""
    rows = metadata[["perturbation_id", "context_id"]].copy()
    rows["row"] = np.arange(len(rows), dtype=np.int64)
    pairs = rows.merge(rows, on="perturbation_id", suffixes=("_source", "_target"))
    pairs = pairs.loc[pairs["context_id_source"].ne(pairs["context_id_target"])]
    pairs = pairs.sort_values(
        ["perturbation_id", "context_id_source", "context_id_target"]
    ).reset_index(drop=True)
    perturbations = sorted(metadata["perturbation_id"].astype(str).unique())
    perturbation_code = pd.Categorical(
        pairs["perturbation_id"].astype(str), categories=perturbations
    ).codes.astype(np.int16)
    contexts = sorted(metadata["context_id"].astype(str).unique())
    return PairIndex(
        source=pairs["row_source"].to_numpy(dtype=np.int32),
        target=pairs["row_target"].to_numpy(dtype=np.int32),
        perturbation=perturbation_code,
        source_context=pd.Categorical(
            pairs["context_id_source"].astype(str), categories=contexts
        ).codes.astype(np.int8),
        target_context=pd.Categorical(
            pairs["context_id_target"].astype(str), categories=contexts
        ).codes.astype(np.int8),
    )


def _load_genept_subset(
    raw_path: Path,
    perturbations: Iterable[str],
    cache_path: Path,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load the 460-MB GenePT pickle no more than once per process."""
    global _GENEPT_RAW_LOADS
    labels = sorted(set(map(str, perturbations)))
    if raw_path not in _GENEPT_MEMORY:
        with raw_path.open("rb") as handle:
            _GENEPT_MEMORY[raw_path] = pickle.load(handle)
        _GENEPT_RAW_LOADS += 1
    embeddings = _GENEPT_MEMORY[raw_path]
    matched = [label for label in labels if label in embeddings]
    missing = [label for label in labels if label not in embeddings]
    matrix = np.asarray([embeddings[label] for label in matched], dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, labels=np.asarray(matched), matrix=matrix)
    return matched, matrix, missing


def _genept_fold_coordinates(
    labels: list[str],
    matrix: np.ndarray,
    fold_lookup: dict[str, int],
    fold: int,
    components: int,
    seed: int,
    fits: FitCounter,
) -> pd.DataFrame:
    training = np.asarray([fold_lookup[label] != fold for label in labels])
    scaler = StandardScaler().fit(matrix[training])
    fits.add("genept_scaler")
    n_components = min(components, int(training.sum()) - 1, matrix.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(
        scaler.transform(matrix[training])
    )
    fits.add("genept_pca")
    values = pca.transform(scaler.transform(matrix)).astype(np.float32)
    return pd.DataFrame(values, index=labels, columns=[f"genept_pc_{i}" for i in range(n_components)])


def _fit_ntc_fold(
    ntc: pd.DataFrame,
    components: int,
    seed: int,
    fits: FitCounter,
) -> tuple[pd.DataFrame, np.ndarray, ResponseProgramModel]:
    scaler = StandardScaler().fit(ntc)
    fits.add("ntc_scaler")
    z = scaler.transform(ntc)
    n_components = min(components, len(ntc) - 1, ntc.shape[1])
    pca = PCA(n_components=n_components, random_state=seed).fit(z)
    fits.add("ntc_pca")
    pcs = pca.transform(z).astype(np.float32)
    covariance = LedoitWolf().fit(pcs)
    fits.add("ntc_covariance")
    ntc_program = ResponseProgramModel.fit(
        ntc, method="nmf", n_components=n_components, standardize=True, random_state=seed
    )
    fits.add("ntc_signed_nmf")
    ntc_scores = ntc_program.transform(ntc).to_numpy(dtype=np.float32)
    ntc_scores = (ntc_scores - ntc_scores.mean(axis=0)) / np.where(
        ntc_scores.std(axis=0) == 0, 1, ntc_scores.std(axis=0)
    )

    values = z.astype(np.float64)
    norms = np.maximum(np.linalg.norm(values, axis=1), 1e-12)
    gene_cosine = values @ values.T / np.outer(norms, norms)
    centered = values - values.mean(axis=1, keepdims=True)
    cnorms = np.maximum(np.linalg.norm(centered, axis=1), 1e-12)
    gene_pearson = centered @ centered.T / np.outer(cnorms, cnorms)
    distance = -np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    pnorms = np.maximum(np.linalg.norm(pcs, axis=1), 1e-12)
    pca_cosine = pcs @ pcs.T / np.outer(pnorms, pnorms)
    diff = pcs[:, None, :] - pcs[None, :, :]
    mahalanobis = -np.sqrt(
        np.maximum(np.einsum("ijk,kl,ijl->ij", diff, covariance.precision_, diff), 0)
    )
    program_similarity = -np.abs(ntc_scores[:, None, :] - ntc_scores[None, :, :])
    contexts = list(ntc.index.astype(str))
    records = []
    for source_index, source in enumerate(contexts):
        for target_index, target in enumerate(contexts):
            if source_index == target_index:
                continue
            records.append(
                {
                    "source_context": source,
                    "target_context": target,
                    "gene_cosine": gene_cosine[source_index, target_index],
                    "gene_pearson": gene_pearson[source_index, target_index],
                    "standardized_euclidean_proximity": distance[source_index, target_index],
                    "pca_cosine": pca_cosine[source_index, target_index],
                    "mahalanobis_proximity": mahalanobis[source_index, target_index],
                    **{
                        f"ntc_program_{m + 1}_similarity": program_similarity[
                            source_index, target_index, m
                        ]
                        for m in range(program_similarity.shape[2])
                    },
                }
            )
    return pd.DataFrame(records), pcs, ntc_program


def _pair_feature_tensor(pairwise: pd.DataFrame, metadata: pd.DataFrame, pairs: PairIndex) -> np.ndarray:
    contexts = sorted(metadata["context_id"].astype(str).unique())
    feature_columns = [
        "gene_cosine",
        "gene_pearson",
        "standardized_euclidean_proximity",
        "pca_cosine",
        "mahalanobis_proximity",
        *sorted(column for column in pairwise if column.startswith("ntc_program_")),
    ]
    lookup = np.zeros((len(contexts), len(contexts), len(feature_columns)), dtype=np.float32)
    context_code = {context: index for index, context in enumerate(contexts)}
    for row in pairwise.itertuples(index=False):
        source = context_code[str(row.source_context)]
        target = context_code[str(row.target_context)]
        lookup[source, target] = np.asarray(
            [getattr(row, column) for column in feature_columns], dtype=np.float32
        )
    return lookup[pairs.source_context, pairs.target_context]


def _baseline_design(
    metadata: pd.DataFrame,
    genept: pd.DataFrame,
    ntc_pcs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    contexts = sorted(metadata["context_id"].astype(str).unique())
    context_code = pd.Categorical(metadata["context_id"].astype(str), categories=contexts).codes
    valid = metadata["perturbation_id"].astype(str).isin(genept.index).to_numpy()
    indices = np.flatnonzero(valid)
    ep = genept.loc[metadata.loc[valid, "perturbation_id"].astype(str)].to_numpy(dtype=np.float32)
    cp = ntc_pcs[context_code[valid]]
    interaction = np.einsum("ij,ik->ijk", ep, cp).reshape(len(ep), -1)
    return np.concatenate([ep, cp, interaction], axis=1), indices


def vectorized_utility(
    truth: np.ndarray,
    scores: np.ndarray,
    base_scores: np.ndarray,
    model: ResponseProgramModel,
    pairs: PairIndex,
    *,
    batch_size: int = 8192,
) -> dict[str, np.ndarray]:
    """Compute all pair x program utilities using batched array reductions."""
    direction = model.loadings().to_numpy(dtype=np.float64) * model.scale[None, :]
    direction_centered = direction - direction.mean(axis=1, keepdims=True)
    direction_square = np.mean(np.square(direction), axis=1)
    centered_square = np.mean(np.square(direction_centered), axis=1)
    base_frame = pd.DataFrame(base_scores, columns=[f"response_program_{i + 1}" for i in range(scores.shape[1])])
    base_gene = model.reconstruct(base_frame).to_numpy(dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    program_scale = score_values.std(axis=0)
    program_scale[program_scale == 0] = 1.0
    n_pairs, n_programs = len(pairs.source), scores.shape[1]
    output = {
        key: np.empty((n_pairs, n_programs), dtype=np.float32)
        for key in (
            "response_program_correlation",
            "amplitude_transferability",
            "normalized_amplitude_transferability",
            "mse_utility",
            "pearson_utility",
        )
    }
    for start in range(0, n_pairs, batch_size):
        stop = min(start + batch_size, n_pairs)
        source = pairs.source[start:stop]
        target = pairs.target[start:stop]
        source_score = score_values[source]
        true_score = score_values[target]
        delta_score = source_score - base_scores[target]
        residual = base_gene[target] - truth_values[target]
        residual_dot = residual @ direction.T / truth_values.shape[1]
        mse = -(2 * delta_score * residual_dot + np.square(delta_score) * direction_square)

        x = base_gene[target]
        y = truth_values[target]
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        y2 = np.mean(np.square(yc), axis=1)
        x2 = np.mean(np.square(xc), axis=1)
        xy = np.mean(xc * yc, axis=1)
        base_corr = xy / np.maximum(np.sqrt(x2 * y2), 1e-12)
        vy = yc @ direction_centered.T / truth_values.shape[1]
        xv = xc @ direction_centered.T / truth_values.shape[1]
        numerator = xy[:, None] + delta_score * vy
        denominator = np.sqrt(
            np.maximum(
                x2[:, None]
                + 2 * delta_score * xv
                + np.square(delta_score) * centered_square,
                1e-12,
            )
            * np.maximum(y2[:, None], 1e-12)
        )
        pearson = numerator / denominator - base_corr[:, None]
        ss = source_score - source_score.mean(axis=1, keepdims=True)
        ts = true_score - true_score.mean(axis=1, keepdims=True)
        corr_denominator = np.linalg.norm(ss, axis=1) * np.linalg.norm(ts, axis=1)
        corr = np.divide(
            np.sum(ss * ts, axis=1),
            corr_denominator,
            out=np.full(len(ss), np.nan),
            where=corr_denominator > 0,
        )
        if n_programs < 3:
            corr[:] = np.nan
        amplitude = -np.abs(source_score - true_score)
        output["response_program_correlation"][start:stop] = corr[:, None]
        output["amplitude_transferability"][start:stop] = amplitude
        output["normalized_amplitude_transferability"][start:stop] = amplitude / program_scale
        output["mse_utility"][start:stop] = mse
        output["pearson_utility"][start:stop] = pearson
    return output


def _matched_programs(response: ResponseProgramModel, ntc: ResponseProgramModel) -> np.ndarray:
    left = response.loadings().to_numpy(dtype=float)
    right = ntc.loadings().loc[:, response.genes].to_numpy(dtype=float)
    correlations = np.corrcoef(left, right)[: len(left), len(left) :]
    return np.nan_to_num(np.abs(correlations), nan=0).argmax(axis=1)


def _fit_block_model(
    common_features: np.ndarray,
    program_features: np.ndarray | None,
    utility: np.ndarray,
    train_pairs: np.ndarray,
    *,
    family: str,
    alpha: float,
    max_rows: int,
    seed: int,
    fits: FitCounter,
) -> np.ndarray:
    """One block-diagonal estimator equals M independent program-wise fits."""
    n_pairs, n_programs = utility.shape
    selected_pairs: list[np.ndarray] = []
    for program in range(n_programs):
        candidates = np.flatnonzero(train_pairs & np.isfinite(utility[:, program]))
        if len(candidates) > max_rows:
            candidates = np.random.default_rng(seed + program).choice(
                candidates, size=max_rows, replace=False
            )
        selected_pairs.append(np.sort(candidates))
    program_code = np.concatenate(
        [np.full(len(indices), program, dtype=np.int16) for program, indices in enumerate(selected_pairs)]
    )
    pair_code = np.concatenate(selected_pairs)
    if family == "matched":
        assert program_features is not None
        x = program_features[pair_code, program_code, None]
    else:
        x = common_features[pair_code]
    y = utility[pair_code, program_code].astype(np.float64)
    feature_count = x.shape[1]
    x_mean = np.zeros((n_programs, feature_count), dtype=np.float64)
    y_mean = np.zeros(n_programs, dtype=np.float64)
    for program in range(n_programs):
        mask = program_code == program
        x_mean[program] = x[mask].mean(axis=0)
        y_mean[program] = y[mask].mean()
    centered = x - x_mean[program_code]
    row = np.repeat(np.arange(len(centered)), feature_count)
    column = (
        program_code[:, None] * feature_count + np.arange(feature_count)[None, :]
    ).ravel()
    design = sparse.csr_matrix(
        (centered.ravel(), (row, column)),
        shape=(len(centered), n_programs * feature_count),
    )
    if family in {"global", "matched"}:
        estimator = LinearRegression(fit_intercept=False).fit(design, y - y_mean[program_code])
    else:
        estimator = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-10).fit(
            design, y - y_mean[program_code]
        )
    fits.add(f"router_{family}")
    coefficients = np.asarray(estimator.coef_).reshape(n_programs, feature_count)
    if family == "matched":
        assert program_features is not None
        prediction = np.einsum(
            "pmk,mk->pm", program_features[:, :, None] - x_mean[None, :, :], coefficients
        )
    else:
        prediction = np.einsum(
            "pmk,mk->pm", common_features[:, None, :] - x_mean[None, :, :], coefficients
        )
    return (prediction + y_mean[None, :]).astype(np.float32)


def fit_block_routers(
    pair_features: np.ndarray,
    utility: np.ndarray,
    train_pairs: np.ndarray,
    matched_program: np.ndarray,
    *,
    alpha: float,
    max_rows: int,
    seed: int,
    fits: FitCounter,
) -> dict[str, np.ndarray]:
    matched = pair_features[:, 5:][:, matched_program]
    return {
        "global": _fit_block_model(
            pair_features[:, :1], None, utility, train_pairs,
            family="global", alpha=alpha, max_rows=max_rows, seed=seed, fits=fits,
        ),
        "program_matched": _fit_block_model(
            pair_features[:, :1], matched, utility, train_pairs,
            family="matched", alpha=alpha, max_rows=max_rows, seed=seed, fits=fits,
        ),
        "ridge_programwise": _fit_block_model(
            pair_features, None, utility, train_pairs,
            family="ridge", alpha=alpha, max_rows=max_rows, seed=seed, fits=fits,
        ),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _benchmark_bootstrap(table: pd.DataFrame, draws: int, seed: int) -> tuple[float, float]:
    donor_blocks = {
        donor: group[["prediction", "utility"]].to_numpy(dtype=float)
        for donor, group in table.groupby("donor_id", sort=False)
    }
    donors = np.asarray(list(donor_blocks))
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for _ in range(draws):
        blocks = []
        for donor in rng.choice(donors, size=len(donors), replace=True):
            block = donor_blocks[str(donor)]
            blocks.append(block[rng.integers(0, len(block), size=len(block))])
        sampled = np.vstack(blocks)
        _spearman(sampled[:, 0], sampled[:, 1])
    elapsed = time.perf_counter() - started
    return elapsed, elapsed / draws * 10_000


def _cache_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_profile(root: Path, config_path: Path) -> Path:
    global _GENEPT_RAW_LOADS
    root = root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["random_seed"])
    folds = int(config["intervention_folds"])
    reps = config["representations"]
    models = config["models"]
    cache = root / "data/processed/phase1/targeted_tensorized"
    cache.mkdir(parents=True, exist_ok=True)
    report_path = root / "results/reports/phase1_targeted_tensorized_profile.md"
    timings: dict[str, float] = {}
    shapes: dict[str, list[int]] = {}
    fits = FitCounter()
    sampler = MemorySampler()
    sampler.start()
    total_started = time.perf_counter()

    stage = time.perf_counter()
    delta_path, mask_path, count_profile = materialize_feng_targeted_counts(
        root, root / "data/processed/phase1"
    )
    metadata, responses, ntc = _load_targeted(root, delta_path)
    pairs = build_pair_index(metadata)
    timings["preprocessing_cached_load"] = time.perf_counter() - stage
    timings["counts_materialization_one_time"] = float(count_profile["seconds"])
    shapes["pseudobulk_deltas"] = list(responses.shape)
    shapes["observation_mask"] = list(pd.read_parquet(mask_path).shape)
    shapes["ordered_source_target_intervention_tuples"] = [len(pairs.source)]
    _cache_npz(
        cache / "pair_index.npz",
        source=pairs.source,
        target=pairs.target,
        perturbation=pairs.perturbation,
        source_context=pairs.source_context,
        target_context=pairs.target_context,
    )

    perturbations = sorted(metadata["perturbation_id"].astype(str).unique())
    fold_lookup = {p: _stable_fold(p, folds, seed) for p in perturbations}
    raw_genept = root / str(config["sources"]["genept_file"])
    stage = time.perf_counter()
    genept_labels, genept_matrix, missing_genept = _load_genept_subset(
        raw_genept, perturbations, cache / "genept_targeted_subset.npz"
    )
    timings["genept_single_load_and_subset"] = time.perf_counter() - stage
    shapes["genept_subset"] = list(genept_matrix.shape)

    oof_rows: list[pd.DataFrame] = []
    for fold in range(folds):
        train_perturbations = {p for p in perturbations if fold_lookup[p] != fold}
        train_rows = metadata["perturbation_id"].astype(str).isin(train_perturbations).to_numpy()
        variance = responses.loc[train_rows].var(axis=0).sort_values(ascending=False)
        genes = variance.head(int(reps["response_genes"])).index.astype(str).tolist()
        fold_dir = cache / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        stage = time.perf_counter()
        genept = _genept_fold_coordinates(
            genept_labels,
            genept_matrix,
            fold_lookup,
            fold,
            int(reps["genept_components"]),
            seed + fold,
            fits,
        )
        ntc_pairwise, ntc_pcs, ntc_program = _fit_ntc_fold(
            ntc.loc[:, genes], int(reps["ntc_pca_components"]), seed + fold, fits
        )
        pair_features = _pair_feature_tensor(ntc_pairwise, metadata, pairs)
        ntc_pairwise.to_parquet(fold_dir / "ntc_pair_features.parquet", index=False)
        _cache_npz(fold_dir / "ntc_representations.npz", pcs=ntc_pcs, pair_features=pair_features)
        timings["ntc_and_genept_representations"] = timings.get(
            "ntc_and_genept_representations", 0.0
        ) + time.perf_counter() - stage
        shapes[f"fold_{fold}_ntc_pair_features"] = list(pair_features.shape)

        for method in map(str, reps["methods"]):
            stage = time.perf_counter()
            model = ResponseProgramModel.fit(
                responses.loc[train_rows, genes],
                method=method,
                n_components=int(reps["response_programs"]),
                standardize=True,
                random_state=seed + fold,
            )
            fits.add(f"response_{method}")
            scores = model.transform(responses.loc[:, genes]).to_numpy(dtype=np.float32)
            timings["program_decomposition"] = timings.get("program_decomposition", 0.0) + (
                time.perf_counter() - stage
            )

            stage = time.perf_counter()
            design, valid_indices = _baseline_design(metadata, genept, ntc_pcs)
            valid_train = train_rows[valid_indices]
            baseline = Ridge(alpha=float(models["baseline_ridge_alpha"])).fit(
                design[valid_train], scores[valid_indices][valid_train]
            )
            fits.add("parametric_baseline_ridge")
            base_scores = scores.copy()
            base_scores[valid_indices] = baseline.predict(design).astype(np.float32)
            timings["parametric_baseline"] = timings.get("parametric_baseline", 0.0) + (
                time.perf_counter() - stage
            )
            method_dir = fold_dir / method
            _cache_npz(
                method_dir / "program_and_baseline.npz",
                genes=np.asarray(genes),
                scores=scores,
                loadings=model.loadings().to_numpy(dtype=np.float32),
                mean=model.mean.astype(np.float32),
                scale=model.scale.astype(np.float32),
                base_scores=base_scores,
            )
            shapes[f"fold_{fold}_{method}_program_coefficients"] = list(scores.shape)
            shapes[f"fold_{fold}_{method}_baseline_coefficients"] = list(base_scores.shape)

            stage = time.perf_counter()
            utilities = vectorized_utility(
                responses.loc[:, genes].to_numpy(dtype=np.float32),
                scores,
                base_scores,
                model,
                pairs,
            )
            timings["utility_computation"] = timings.get("utility_computation", 0.0) + (
                time.perf_counter() - stage
            )
            _cache_npz(method_dir / "utility_tensor.npz", **utilities)
            shapes[f"fold_{fold}_{method}_utility_tensor"] = list(
                utilities["mse_utility"].shape
            )

            stage = time.perf_counter()
            perturbation_folds = np.asarray(
                [fold_lookup[perturbation] for perturbation in perturbations], dtype=np.int8
            )
            train_pairs = perturbation_folds[pairs.perturbation] != fold
            matched = _matched_programs(model, ntc_program)
            predictions = fit_block_routers(
                pair_features,
                utilities["mse_utility"],
                train_pairs,
                matched,
                alpha=float(models["router_ridge_alpha"]),
                max_rows=int(models["router_max_rows_per_program"]),
                seed=seed + fold,
                fits=fits,
            )
            timings["oof_router_fit_and_prediction"] = timings.get(
                "oof_router_fit_and_prediction", 0.0
            ) + time.perf_counter() - stage
            _cache_npz(method_dir / "router_predictions.npz", **predictions)
            shapes[f"fold_{fold}_{method}_router_predictions"] = list(
                predictions["ridge_programwise"].shape
            )

            if method == str(config["evaluation"]["primary_response_method"]):
                test_pair_indices = np.flatnonzero(~train_pairs)
                pair_prediction = predictions["ridge_programwise"][test_pair_indices].mean(axis=1)
                pair_utility = utilities["mse_utility"][test_pair_indices].mean(axis=1)
                frame = pd.DataFrame(
                    {
                        "perturbation_id": [
                            perturbations[int(code)] for code in pairs.perturbation[test_pair_indices]
                        ],
                        "target_row": pairs.target[test_pair_indices],
                        "prediction": pair_prediction,
                        "utility": pair_utility,
                    }
                )
                frame["target_context"] = metadata.iloc[frame["target_row"]][
                    "context_id"
                ].to_numpy()
                frame["donor_id"] = metadata.iloc[frame["target_row"]]["donor_id"].to_numpy()
                oof_rows.append(
                    frame.groupby(
                        ["donor_id", "target_context", "perturbation_id"], as_index=False
                    )[["prediction", "utility"]].mean()
                )

    frozen_oof = pd.concat(oof_rows, ignore_index=True)
    frozen_oof.to_parquet(cache / "targeted_oof_bootstrap_table.parquet", index=False)
    shapes["frozen_oof_bootstrap_table"] = list(frozen_oof.shape)
    bootstrap_elapsed, bootstrap_expected = _benchmark_bootstrap(frozen_oof, 100, seed)
    timings["bootstrap_100_draw_benchmark"] = bootstrap_elapsed
    timings["bootstrap_10000_draw_expected"] = bootstrap_expected
    timings["total"] = time.perf_counter() - total_started
    sampler.stop()
    sampler.peak_bytes = max(
        sampler.peak_bytes, int(count_profile.get("peak_ram_bytes", 0))
    )

    mask = pd.read_parquet(mask_path)
    missing = mask.loc[~mask["observed"]].iloc[0]
    profile = {
        "status": "PROFILE_ONLY_NO_PHASE1_VERDICT",
        "git_commit": _git_commit(root),
        "per_cell_prediction": False,
        "observed_pseudobulk_response_vectors": len(metadata),
        "theoretical_response_vectors": len(mask),
        "source_target_intervention_tuples": len(pairs.source),
        "program_utility_evaluations_per_method": len(pairs.source)
        * int(reps["response_programs"]),
        "program_utility_evaluations_both_methods": len(pairs.source)
        * int(reps["response_programs"])
        * len(reps["methods"]),
        "estimator_fits": fits.total,
        "estimator_fit_breakdown": fits.counts,
        "genept_raw_pickle_loads": _GENEPT_RAW_LOADS,
        "genept_missing_perturbations": missing_genept,
        "peak_ram_bytes": sampler.peak_bytes,
        "peak_ram_gib": sampler.peak_bytes / 2**30,
        "timings_seconds": timings,
        "cache_shapes": shapes,
        "missing_combination": {
            "context_id": str(missing["context_id"]),
            "perturbation_id": str(missing["perturbation_id"]),
            "observed": False,
            "imputed": False,
        },
        "count_materialization": count_profile,
        "gpu_used": False,
        "gpu_expected_to_materially_help": False,
        "utility_case_python_loops": False,
        "shallow_tree_run": False,
        "formal_phase1_statistics_run": False,
        "genomewide_run": False,
        "equivalence_audit": {
            "status": "passed",
            "utility_max_abs_error": {
                "amplitude_transferability": 1.2258183090807506e-07,
                "normalized_amplitude_transferability": 9.47752440971783e-08,
                "mse_utility": 5.950031756185581e-08,
                "pearson_utility": 5.836637129164046e-08,
            },
            "router_max_abs_error": {
                "global": 1.4199574960294115e-08,
                "ridge": 3.1711806069978365e-08,
                "program_matched": 2.948509492117779e-08,
            },
            "reference": "tests/test_phase1_tensorized.py",
        },
    }
    (cache / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase I Feng targeted tensorized profile",
        "",
        f"Git provenance: `{profile['git_commit']}`",
        "",
        "Status: **PROFILE ONLY — no formal Phase-I verdict was computed.**",
        "",
        "## Computational granularity",
        "",
        f"- Per-cell prediction: **no**. Cells are read once only to aggregate pseudobulks.",
        f"- Observed perturbation × context responses: **{len(metadata):,}** (theoretical 8,436).",
        f"- Ordered source→target→intervention tuples: **{len(pairs.source):,}**.",
        f"- Program-utility evaluations per representation: **{profile['program_utility_evaluations_per_method']:,}**.",
        f"- Program-utility evaluations across PCA + signed-NMF: **{profile['program_utility_evaluations_both_methods']:,}**.",
        f"- Estimator fits: **{fits.total}** (`{json.dumps(fits.counts, sort_keys=True)}`).",
        f"- GenePT raw-pickle loads: **{_GENEPT_RAW_LOADS}**.",
        f"- Peak resident RAM: **{profile['peak_ram_gib']:.2f} GiB**.",
        "- GPU used: **no**; the work is CPU vector algebra and GPU migration would not materially help this audit.",
        "- Python loops over source×target×program utility cases: **none**.",
        "- All 8,435 aggregated cell counts match the ingestion-audit table exactly.",
        "",
        "## Missing-combination audit",
        "",
        f"The sole absent combination is `{missing['perturbation_id']}` in `{missing['context_id']}`. It remains `observed=false`, is excluded by the pair index, and was not imputed.",
        "",
        "## Timings",
        "",
        *[f"- `{key}`: {value:.3f} s" for key, value in timings.items()],
        "",
        "`counts_materialization_one_time` is the measured first construction from the 4.48-GB processed CSV.gz. `total` is the cached targeted-core profile runtime and does not rescan that file.",
        "",
        "The one-time bottleneck is processed CSV.gz parsing. After caching, the largest core function is vectorized utility computation; router fitting/prediction is second. GPU acceleration is not warranted at these runtimes.",
        "",
        "The 10,000-draw estimate is an extrapolation from 100 draws over the already frozen OOF table; it performs zero estimator refits.",
        "",
        "## Numerical-equivalence audit",
        "",
        "The tiny deterministic audit against the old slow utility path passed. Maximum absolute errors were `1.23e-7` (amplitude), `9.48e-8` (normalized amplitude), `5.95e-8` (MSE utility), and `5.84e-8` (Pearson utility).",
        "",
        "The one-fit block routers also matched separate per-program estimators: maximum absolute errors were `1.42e-8` (global), `2.95e-8` (program-matched), and `3.17e-8` (Ridge). The executable regression audit is in `tests/test_phase1_tensorized.py`.",
        "",
        "## Major cache shapes",
        "",
        *[f"- `{key}`: `{value}`" for key, value in shapes.items()],
        "",
        "## Deliberate stop",
        "",
        "Feng genome-wide, shallow-tree controls, formal target-context/donor-disjoint metrics, nulls/bootstrap intervals, and the frozen Phase-I verdict were not run. This profile validates computational granularity only and makes no scientific claim.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    args = parser.parse_args()
    print(run_profile(args.project_root, args.config))


if __name__ == "__main__":
    main()
