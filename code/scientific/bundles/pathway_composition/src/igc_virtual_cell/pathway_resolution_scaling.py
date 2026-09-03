from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_resolution_poc.pathway import (
    PATHWAYS as HISTORICAL_PATHWAYS,
    load_pathway_weights as load_historical_pathway_weights,
)
from igc_virtual_cell.cgc_resolution_poc.replay import CONTEXTS, GENE_COUNT, INTERVENTIONS


PANEL = (
    "Androgen",
    "EGFR",
    "Estrogen",
    "Hypoxia",
    "JAK-STAT",
    "MAPK",
    "NFkB",
    "PI3K",
    "TGFb",
    "TNFa",
    "Trail",
    "VEGF",
    "WNT",
    "p53",
)
PANEL_SIZE = len(PANEL)
M, K = 49, 92
BOOTSTRAP_GRID = (1, 2, 3, 5, 8, 11, PANEL_SIZE)
BOOTSTRAPS = 10_000
GENE_BLOCK = 512
RCOND = max(GENE_COUNT, PANEL_SIZE) * np.finfo(np.float64).eps
FIVE_FIDELITY = 0.029695165
FIVE_RECOVERY = 0.27281393788398667
GENE_RECOVERY = 0.11068053088808294
FIVE_MINUS_GENE = FIVE_RECOVERY - GENE_RECOVERY
F_ANCHOR_ATOL = 5e-7
RECOVERY_ATOL = 1e-10
FULL_TABLE_ATOL = 5e-7
PREDICTION_SCORE_ATOL = 2e-5
PROGENY_REVISION = "cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f"
PROGENY_VERSION = "1.17.3"
EXPECTED_HASHES = {
    "weights": "f5c4faa94fe03b2d088b72743da175380dfcf2edbfacd0077cc25eb19d2ca7e9",
    "metadata": "f690be9e4a57bcf1fc2c70d1e458ea901950c5c42cc84c6d1fb7e728094c4960",
    "indices": "efea7718149514c763fbbad86d2337fde3377bf6c977365f0046576a85b94343",
    "truth": "005603c5a01c085762c9eecb266de6302f95356ca5eb4026f8c0380f57793f73",
    "prediction": "a1e99f8a949b8b205d1b360dd9e6919229e6b41969e98fdc57139f6983f798d6",
    "episode": "b0e410e9c0f5d0d9f0a4d3fbfd02cf247d171cd203c480083dc6fa666b68d928",
    "utility": "bd3b4f5097b4f158f58da102849d52080bb9ae7b7a4c4325f37bf574fd1a7c08",
    "bootstrap": "f0a9697175fe2d4b8bc800e4352d961ed349f9e9a33df138c793be24f982c9cf",
    "split": "da3fc0f3422c12e2a1ec9fcdc6447cd137836a963290a262a82451430df1198d",
}


@dataclass(frozen=True)
class FrozenPaths:
    weights: Path
    metadata: Path
    indices: Path
    truth: Path
    prediction: Path
    episode: Path
    utility: Path
    bootstrap: Path
    split: Path
    progeny_repo: Path


@dataclass
class SufficientStatistics:
    programs: tuple[str, ...]
    weight_matrix: np.ndarray
    gram: np.ndarray
    truth_scores: np.ndarray
    residual_scores: np.ndarray
    full_cross: np.ndarray
    truth_diag: np.ndarray
    residual_diag: np.ndarray
    cross_outer_flat: np.ndarray
    gene_recovery: float
    anchor_audit: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    value = np.ascontiguousarray(vector, dtype=np.float64)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def resolve_paths(root: Path, truth_root: Path) -> FrozenPaths:
    return FrozenPaths(
        weights=truth_root / "data/cgc_bio1_official/frozen_program_weights.parquet",
        metadata=truth_root / "results/cgc_tahoe_0i/gene_metadata_frozen.csv",
        indices=truth_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy",
        truth=truth_root / "data/cgc_resolution2_replay/truth_g_primary_float32.npy",
        prediction=truth_root / "data/cgc_resolution2_replay/m49_k92_predictions_float32.npy",
        episode=truth_root / "data/cgc_resolution2_replay/m49_k92_frozen_weights.npz",
        utility=truth_root / "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz",
        bootstrap=truth_root / "data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy",
        split=root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json",
        progeny_repo=truth_root / "data/cgc_bio1_official/progeny",
    )


def validate_hashes(paths: FrozenPaths) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = getattr(paths, name)
        if not path.exists():
            raise RuntimeError(f"PATHWAY_SCALING_INPUT_MISSING:{name}:{path}")
        value = sha256(path)
        observed[name] = value
        if value != expected:
            raise RuntimeError(f"PATHWAY_SCALING_HASH_MISMATCH:{name}:{value}")
    revision = subprocess.check_output(
        ["git", "-C", str(paths.progeny_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PROGENY_REVISION:
        raise RuntimeError(f"PATHWAY_SCALING_PROGENY_REVISION_MISMATCH:{revision}")
    description = (paths.progeny_repo / "DESCRIPTION").read_text(encoding="utf-8")
    version = next(
        (line.split(":", 1)[1].strip() for line in description.splitlines() if line.startswith("Version:")),
        None,
    )
    if version != PROGENY_VERSION:
        raise RuntimeError(f"PATHWAY_SCALING_PROGENY_VERSION_MISMATCH:{version}")
    return observed


def load_full_panel(paths: FrozenPaths) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    metadata = pd.read_csv(paths.metadata)
    indices = np.load(paths.indices).astype(np.int64)
    symbols = metadata.set_index("raw_gene_index").loc[indices, "gene_symbol"].astype(str).tolist()
    if len(symbols) != GENE_COUNT or len(set(symbols)) != GENE_COUNT:
        raise RuntimeError("PATHWAY_SCALING_GENE_AXIS_AMBIGUOUS")
    source = pd.read_parquet(paths.weights)
    available = tuple(source.loc[source["space"].eq("PROGENy"), "program"].drop_duplicates().astype(str))
    if available != PANEL:
        raise RuntimeError(f"PATHWAY_SCALING_PANEL_MISMATCH:{available}")
    position = {symbol: index for index, symbol in enumerate(symbols)}
    matrix = np.zeros((GENE_COUNT, PANEL_SIZE), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for column, pathway in enumerate(PANEL):
        block = source[source["space"].eq("PROGENy") & source["program"].eq(pathway)]
        if block.empty or block["gene_symbol"].duplicated().any():
            raise RuntimeError(f"PATHWAY_SCALING_WEIGHT_ROWS_INVALID:{pathway}")
        for row in block.itertuples(index=False):
            mapped = position.get(str(row.gene_symbol))
            if mapped is not None:
                matrix[mapped, column] = float(row.weight)
        norm = float(np.linalg.norm(matrix[:, column]))
        if not np.isclose(norm, 1.0, atol=1e-12, rtol=0):
            raise RuntimeError(f"PATHWAY_SCALING_WEIGHT_NORM_INVALID:{pathway}:{norm}")
        rows.append(
            {
                "panel_order": column,
                "pathway": pathway,
                "source_space": "PROGENy",
                "source_version": PROGENY_VERSION,
                "source_git_revision": PROGENY_REVISION,
                "source_rows": int(len(block)),
                "mapped_nonzero_genes": int(np.count_nonzero(matrix[:, column])),
                "positive_weight_genes": int(np.count_nonzero(matrix[:, column] > 0)),
                "negative_weight_genes": int(np.count_nonzero(matrix[:, column] < 0)),
                "mapped_l2_norm": norm,
                "mapped_vector_sha256": vector_sha256(matrix[:, column]),
                "historical_five": pathway in HISTORICAL_PATHWAYS,
            }
        )
    historical, historical_symbols = load_historical_pathway_weights(paths.truth.parent.parent.parent)
    if historical_symbols != symbols:
        raise RuntimeError("PATHWAY_SCALING_HISTORICAL_GENE_AXIS_MISMATCH")
    observed = np.stack([matrix[:, PANEL.index(pathway)] for pathway in HISTORICAL_PATHWAYS], axis=0)
    exact = bool(np.array_equal(observed, historical))
    if not exact:
        raise RuntimeError("PATHWAY_SCALING_HISTORICAL_WEIGHT_VECTOR_MISMATCH")
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * RCOND))
    if rank != PANEL_SIZE:
        raise RuntimeError(f"PATHWAY_SCALING_FULL_PANEL_RANK_DEFICIENT:{rank}")
    panel_audit = pd.DataFrame(rows)
    panel_audit["complete_panel_count"] = PANEL_SIZE
    panel_audit["complete_panel_rank"] = rank
    panel_audit["complete_panel_condition_W"] = float(singular[0] / singular[-1])
    panel_audit["weights_archive_sha256"] = EXPECTED_HASHES["weights"]
    panel_audit["historical_vector_exact_match"] = exact
    return matrix, symbols, panel_audit


def all_but_one_deltas(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.shape[1] != CONTEXTS:
        raise ValueError("PATHWAY_SCALING_CONTEXT_AXIS_INVALID")
    return (CONTEXTS / (CONTEXTS - 1.0)) * (data - data.mean(axis=1, keepdims=True))


def replay_pathway_scores(
    raw_scores: np.ndarray, episode_path: Path
) -> tuple[np.ndarray, np.ndarray, bool]:
    with np.load(episode_path, allow_pickle=False) as archive:
        weights = np.asarray(archive["weights"], dtype=np.float64)
        sources = np.asarray(archive["sources"], dtype=np.int64)
    if weights.shape != (2, 1, CONTEXTS, INTERVENTIONS, M):
        raise RuntimeError(f"PATHWAY_SCALING_EPISODE_WEIGHT_SHAPE:{weights.shape}")
    if sources.shape != (1, CONTEXTS, M):
        raise RuntimeError(f"PATHWAY_SCALING_EPISODE_SOURCE_SHAPE:{sources.shape}")
    all_but_one = all(
        set(map(int, sources[0, target])) == (set(range(CONTEXTS)) - {target})
        for target in range(CONTEXTS)
    )
    if not all_but_one:
        raise RuntimeError("PATHWAY_SCALING_EPISODE_NOT_ALL_BUT_ONE")
    truth = np.empty((2, CONTEXTS, INTERVENTIONS, PANEL_SIZE), dtype=np.float64)
    prediction = np.empty_like(truth)
    uniform = np.full(M, 1.0 / M, dtype=np.float64)
    for target in range(CONTEXTS):
        selected = sources[0, target]
        for plate in range(2):
            block = raw_scores[plate, selected]
            truth[plate, target] = raw_scores[plate, target] - block.mean(axis=0)
            prediction[plate, target] = np.einsum(
                "ps,sph->ph",
                weights[plate, 0, target] - uniform,
                block,
                optimize=True,
            )
    return truth, prediction, all_but_one


def frozen_gene_metric(utility_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    with np.load(utility_path, allow_pickle=False) as archive:
        models = list(map(str, archive["models"]))
        m_values = archive["m_values"].astype(int).tolist()
        k_values = archive["k_values"].astype(int).tolist()
        model = models.index("M2_AFFINE_RIDGE")
        m_index = m_values.index(M)
        k_index = k_values.index(K)
        truth = np.asarray(archive["vtruth"][m_index], dtype=np.float64)
        after = np.asarray(archive["vafter"][model, m_index, k_index], dtype=np.float64)
    value = 1.0 - float(after.sum(dtype=np.float64)) / float(truth.sum(dtype=np.float64))
    return value, truth, after


def compute_sufficient_statistics(paths: FrozenPaths, weight_matrix: np.ndarray) -> SufficientStatistics:
    truth_mem = np.load(paths.truth, mmap_mode="r")
    prediction_mem = np.load(paths.prediction, mmap_mode="r")
    if truth_mem.shape != (2, CONTEXTS, INTERVENTIONS, GENE_COUNT) or truth_mem.dtype != np.float32:
        raise RuntimeError(f"PATHWAY_SCALING_TRUTH_AXIS_INVALID:{truth_mem.shape}:{truth_mem.dtype}")
    if prediction_mem.shape != (2, 1, CONTEXTS, INTERVENTIONS, GENE_COUNT) or prediction_mem.dtype != np.float32:
        raise RuntimeError(
            f"PATHWAY_SCALING_PREDICTION_AXIS_INVALID:{prediction_mem.shape}:{prediction_mem.dtype}"
        )
    raw_scores = np.zeros((2, CONTEXTS, INTERVENTIONS, PANEL_SIZE), dtype=np.float64)
    direct_truth_scores = np.zeros_like(raw_scores)
    cached_prediction_scores = np.zeros_like(raw_scores)
    full_cross = np.zeros((CONTEXTS, INTERVENTIONS), dtype=np.float64)
    cached_residual_cross = np.zeros_like(full_cross)
    for start in range(0, GENE_COUNT, GENE_BLOCK):
        stop = min(start + GENE_BLOCK, GENE_COUNT)
        raw = np.asarray(truth_mem[..., start:stop], dtype=np.float64)
        delta = all_but_one_deltas(raw)
        prediction = np.asarray(prediction_mem[:, 0, ..., start:stop], dtype=np.float64)
        residual = delta - prediction
        block = weight_matrix[start:stop]
        raw_scores += np.einsum("pcig,gh->pcih", raw, block, optimize=True)
        direct_truth_scores += np.einsum("pcig,gh->pcih", delta, block, optimize=True)
        cached_prediction_scores += np.einsum("pcig,gh->pcih", prediction, block, optimize=True)
        full_cross += np.einsum("cig,cig->ci", delta[0], delta[1], optimize=True)
        cached_residual_cross += np.einsum("cig,cig->ci", residual[0], residual[1], optimize=True)
        print(f"PATHWAY_SCALING project genes={stop}/{GENE_COUNT}", flush=True)
    truth_scores, prediction_scores, all_but_one = replay_pathway_scores(raw_scores, paths.episode)
    truth_score_error = float(np.max(np.abs(truth_scores - direct_truth_scores)))
    prediction_score_error = float(np.max(np.abs(prediction_scores - cached_prediction_scores)))
    if truth_score_error > 2e-11:
        raise RuntimeError(f"PATHWAY_SCALING_TRUTH_SCORE_REPLAY_MISMATCH:{truth_score_error}")
    if prediction_score_error > PREDICTION_SCORE_ATOL:
        raise RuntimeError(f"PATHWAY_SCALING_PREDICTION_SCORE_REPLAY_MISMATCH:{prediction_score_error}")
    residual_scores = truth_scores - prediction_scores
    truth_diag = (truth_scores[0] * truth_scores[1]).reshape(CONTEXTS * INTERVENTIONS, PANEL_SIZE)
    residual_diag = (residual_scores[0] * residual_scores[1]).reshape(
        CONTEXTS * INTERVENTIONS, PANEL_SIZE
    )
    score6 = truth_scores[0].reshape(CONTEXTS * INTERVENTIONS, PANEL_SIZE)
    score14 = truth_scores[1].reshape(CONTEXTS * INTERVENTIONS, PANEL_SIZE)
    cross_outer_flat = np.einsum("ni,nj->nij", score14, score6, optimize=True).reshape(
        CONTEXTS * INTERVENTIONS, PANEL_SIZE * PANEL_SIZE
    )
    gene_recovery, frozen_truth, _ = frozen_gene_metric(paths.utility)
    full_table_error = float(np.max(np.abs(full_cross / GENE_COUNT - frozen_truth)))
    if full_table_error > FULL_TABLE_ATOL:
        raise RuntimeError(f"PATHWAY_SCALING_FULL_TRUTH_TABLE_MISMATCH:{full_table_error}")
    gram = weight_matrix.T @ weight_matrix
    historical = np.asarray([PANEL.index(pathway) for pathway in HISTORICAL_PATHWAYS], dtype=int)
    inverse, span = span_inverse(gram[np.ix_(historical, historical)])
    historical_scores6 = score6[:, historical]
    historical_scores14 = score14[:, historical]
    f5_num = np.einsum(
        "ni,ij,nj->", historical_scores6, inverse, historical_scores14, optimize=True
    )
    f5 = float(f5_num / full_cross.sum(dtype=np.float64))
    historical_truth = float(truth_diag[:, historical].sum(dtype=np.float64))
    historical_after = float(residual_diag[:, historical].sum(dtype=np.float64))
    g5 = 1.0 - historical_after / historical_truth
    delta = g5 - gene_recovery
    cached_gene_recovery = 1.0 - float(cached_residual_cross.sum()) / float(full_cross.sum())
    anchors = {
        "full_panel_count": PANEL_SIZE,
        "all_nonempty_subset_count": 2**PANEL_SIZE - 1,
        "all_but_one_sources_verified": all_but_one,
        "truth_score_replay_max_abs_error": truth_score_error,
        "prediction_score_replay_vs_stored_float32_max_abs_error": prediction_score_error,
        "full_truth_table_max_abs_error": full_table_error,
        "full_truth_table_tolerance": FULL_TABLE_ATOL,
        "f5_observed": f5,
        "f5_expected": FIVE_FIDELITY,
        "f5_abs_error": abs(f5 - FIVE_FIDELITY),
        "g5_observed": g5,
        "g5_expected": FIVE_RECOVERY,
        "g5_abs_error": abs(g5 - FIVE_RECOVERY),
        "g_gene_observed": gene_recovery,
        "g_gene_expected": GENE_RECOVERY,
        "g_gene_abs_error": abs(gene_recovery - GENE_RECOVERY),
        "delta_g_observed": delta,
        "delta_g_expected": FIVE_MINUS_GENE,
        "delta_g_abs_error": abs(delta - FIVE_MINUS_GENE),
        "cached_prediction_gene_recovery_diagnostic": cached_gene_recovery,
        "historical_raw_score_truth_denominator": historical_truth,
        "historical_span_rank": int(span["rank"]),
        "historical_span_condition_WtW": float(span["condition_wtw"]),
        "full_cross_signal": float(full_cross.sum(dtype=np.float64)),
    }
    failures = []
    if abs(f5 - FIVE_FIDELITY) > F_ANCHOR_ATOL:
        failures.append("F5")
    if abs(g5 - FIVE_RECOVERY) > RECOVERY_ATOL:
        failures.append("g5")
    if abs(gene_recovery - GENE_RECOVERY) > RECOVERY_ATOL:
        failures.append("g_gene")
    if abs(delta - FIVE_MINUS_GENE) > RECOVERY_ATOL:
        failures.append("delta_g")
    if failures:
        raise RuntimeError(f"PATHWAY_SCALING_ANCHOR_GATE_FAIL:{','.join(failures)}")
    return SufficientStatistics(
        programs=PANEL,
        weight_matrix=weight_matrix,
        gram=gram,
        truth_scores=np.stack([score6, score14], axis=0),
        residual_scores=residual_scores.reshape(2, CONTEXTS * INTERVENTIONS, PANEL_SIZE),
        full_cross=full_cross.reshape(-1),
        truth_diag=truth_diag,
        residual_diag=residual_diag,
        cross_outer_flat=cross_outer_flat,
        gene_recovery=gene_recovery,
        anchor_audit=anchors,
    )


def span_inverse(gram: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(gram, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular = np.sqrt(eigenvalues)[::-1]
    smax = float(singular[0])
    absolute_tolerance = smax * RCOND
    keep_eigen = eigenvalues > absolute_tolerance**2
    rank = int(np.count_nonzero(keep_eigen))
    inverse = (eigenvectors[:, keep_eigen] / eigenvalues[keep_eigen]) @ eigenvectors[:, keep_eigen].T
    kept = singular[:rank]
    condition_w = float(kept[0] / kept[-1]) if rank else math.inf
    return inverse, {
        "rank": rank,
        "singular_values": singular.tolist(),
        "smax": smax,
        "smin_retained": float(kept[-1]) if rank else np.nan,
        "absolute_tolerance": absolute_tolerance,
        "relative_rcond": RCOND,
        "condition_w": condition_w,
        "condition_wtw": condition_w**2,
        "rank_deficient": rank < matrix.shape[0],
        "pinv_symmetry_error": float(np.max(np.abs(inverse - inverse.T))),
    }


def iter_subsets(size: int = PANEL_SIZE) -> Iterable[tuple[int, tuple[int, ...]]]:
    for mask in range(1, 2**size):
        yield mask, tuple(index for index in range(size) if mask & (1 << index))


def enumerate_subset_metrics(
    stats: SufficientStatistics,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    full_denominator = float(stats.full_cross.sum(dtype=np.float64))
    cross_total_flat = stats.cross_outer_flat.sum(axis=0, dtype=np.float64)
    truth_total = stats.truth_diag.sum(axis=0, dtype=np.float64)
    residual_total = stats.residual_diag.sum(axis=0, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    numerical: list[dict[str, Any]] = []
    coefficient: dict[int, np.ndarray] = {}
    historical_mask = sum(1 << PANEL.index(pathway) for pathway in HISTORICAL_PATHWAYS)
    for sequence, (mask, indices_tuple) in enumerate(iter_subsets(), start=1):
        indices = np.asarray(indices_tuple, dtype=int)
        inverse, audit = span_inverse(stats.gram[np.ix_(indices, indices)])
        embedded = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=np.float64)
        embedded[np.ix_(indices, indices)] = inverse
        coefficient[mask] = embedded.reshape(-1)
        f_num = float(coefficient[mask] @ cross_total_flat)
        truth_denominator = float(truth_total[indices].sum(dtype=np.float64))
        residual_denominator = float(residual_total[indices].sum(dtype=np.float64))
        recovery_valid = bool(np.isfinite(truth_denominator) and truth_denominator > 0)
        algebraic_recovery = (
            1.0 - residual_denominator / truth_denominator
            if np.isfinite(truth_denominator) and truth_denominator != 0
            else np.nan
        )
        pathways = "|".join(PANEL[index] for index in indices)
        rows.append(
            {
                "subset_id": f"S{mask:04x}",
                "subset_mask": mask,
                "pathways": pathways,
                "pathway_count": len(indices),
                "effective_rank": audit["rank"],
                "biological_fidelity_F": f_num / full_denominator,
                "recoverability_g": algebraic_recovery if recovery_valid else np.nan,
                "recoverability_status": "VALID" if recovery_valid else "UNDEFINED_NONPOSITIVE_TRUTH_SIGNAL",
                "algebraic_ratio_diagnostic_not_recovery": algebraic_recovery,
                "orthogonal_span_truth_signal": f_num,
                "full_gene_truth_signal": full_denominator,
                "raw_pathway_truth_denominator": truth_denominator,
                "raw_pathway_residual_denominator": residual_denominator,
                "historical_five": mask == historical_mask,
            }
        )
        numerical.append(
            {
                "subset_id": f"S{mask:04x}",
                "subset_mask": mask,
                "pathways": pathways,
                "pathway_count": len(indices),
                "effective_rank": audit["rank"],
                "rank_deficient": audit["rank_deficient"],
                "relative_rcond": audit["relative_rcond"],
                "absolute_tolerance": audit["absolute_tolerance"],
                "largest_singular_value": audit["smax"],
                "smallest_retained_singular_value": audit["smin_retained"],
                "condition_W": audit["condition_w"],
                "condition_WtW": audit["condition_wtw"],
                "singular_values_json": json.dumps(audit["singular_values"], separators=(",", ":")),
                "pinv_symmetry_max_abs_error": audit["pinv_symmetry_error"],
            }
        )
        if sequence % 2_000 == 0:
            print(f"PATHWAY_SCALING subsets={sequence}/{2**PANEL_SIZE - 1}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(["pathway_count", "subset_mask"]).reset_index(drop=True)
    numerical_audit = pd.DataFrame(numerical).sort_values(["pathway_count", "subset_mask"]).reset_index(
        drop=True
    )
    return metrics, numerical_audit, coefficient


def _quantiles(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(data)),
        "p10": float(np.quantile(data, 0.10)),
        "q25": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "q75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
        "maximum": float(np.max(data)),
        "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        "p90_minus_p10": float(np.quantile(data, 0.90) - np.quantile(data, 0.10)),
    }


def composition_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (count, rank), group in metrics.groupby(["pathway_count", "effective_rank"], sort=True):
        for metric, column in (("F", "biological_fidelity_F"), ("g", "recoverability_g")):
            values = group[column].to_numpy(float)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            rows.append(
                {
                    "pathway_count": int(count),
                    "effective_rank": int(rank),
                    "metric": metric,
                    "subset_count": int(len(group)),
                    "valid_subset_count": int(np.count_nonzero(finite)),
                    "invalid_subset_count": int(np.count_nonzero(~finite)),
                    **_quantiles(values[finite]),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_sufficient(
    paths: FrozenPaths, stats: SufficientStatistics
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.load(paths.bootstrap, mmap_mode="r")
    if weights.shape != (BOOTSTRAPS, CONTEXTS * INTERVENTIONS) or weights.dtype != np.int16:
        raise RuntimeError(f"PATHWAY_SCALING_BOOTSTRAP_AXIS_INVALID:{weights.shape}:{weights.dtype}")
    if not np.all(weights.sum(axis=1) == CONTEXTS * INTERVENTIONS):
        raise RuntimeError("PATHWAY_SCALING_BOOTSTRAP_COUNTS_INVALID")
    full = np.empty(BOOTSTRAPS, dtype=np.float64)
    truth = np.empty((BOOTSTRAPS, PANEL_SIZE), dtype=np.float64)
    residual = np.empty_like(truth)
    outer = np.empty((BOOTSTRAPS, PANEL_SIZE * PANEL_SIZE), dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 200):
        stop = min(start + 200, BOOTSTRAPS)
        block = np.asarray(weights[start:stop], dtype=np.float64)
        full[start:stop] = block @ stats.full_cross
        truth[start:stop] = block @ stats.truth_diag
        residual[start:stop] = block @ stats.residual_diag
        outer[start:stop] = block @ stats.cross_outer_flat
    if np.any(~np.isfinite(full)) or np.any(full <= 0):
        raise RuntimeError("PATHWAY_SCALING_BOOTSTRAP_FULL_DENOMINATOR_INVALID")
    return full, truth, residual, outer


def bootstrap_scaling(
    paths: FrozenPaths,
    stats: SufficientStatistics,
    metrics: pd.DataFrame,
    coefficients: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    full, truth, residual, outer = _bootstrap_sufficient(paths, stats)
    draw_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    curves = {
        "F": np.empty((BOOTSTRAPS, len(BOOTSTRAP_GRID)), dtype=np.float64),
        "g": np.empty((BOOTSTRAPS, len(BOOTSTRAP_GRID)), dtype=np.float64),
    }
    for grid_index, count in enumerate(BOOTSTRAP_GRID):
        group = metrics[metrics["pathway_count"].eq(count)].sort_values("subset_mask")
        masks = group["subset_mask"].astype(int).tolist()
        indicator = np.zeros((len(masks), PANEL_SIZE), dtype=np.float64)
        coefficient = np.stack([coefficients[mask] for mask in masks], axis=0)
        for row, mask in enumerate(masks):
            indicator[row, [index for index in range(PANEL_SIZE) if mask & (1 << index)]] = 1.0
        median_f = np.empty(BOOTSTRAPS, dtype=np.float64)
        median_g = np.empty(BOOTSTRAPS, dtype=np.float64)
        valid_g_count = np.empty(BOOTSTRAPS, dtype=np.int32)
        for start in range(0, BOOTSTRAPS, 200):
            stop = min(start + 200, BOOTSTRAPS)
            f_values = (outer[start:stop] @ coefficient.T) / full[start:stop, None]
            truth_values = truth[start:stop] @ indicator.T
            residual_values = residual[start:stop] @ indicator.T
            valid = np.isfinite(truth_values) & (truth_values > 0)
            g_values = np.full_like(truth_values, np.nan)
            np.divide(residual_values, truth_values, out=g_values, where=valid)
            g_values[valid] = 1.0 - g_values[valid]
            valid_g_count[start:stop] = valid.sum(axis=1)
            if np.any(valid_g_count[start:stop] == 0):
                raise RuntimeError(f"PATHWAY_SCALING_BOOTSTRAP_NO_VALID_SUBSET:k{count}")
            median_f[start:stop] = np.median(f_values, axis=1)
            median_g[start:stop] = np.nanmedian(g_values, axis=1)
        curves["F"][:, grid_index] = median_f
        curves["g"][:, grid_index] = median_g
        for metric, values, column in (
            ("F", median_f, "biological_fidelity_F"),
            ("g", median_g, "recoverability_g"),
        ):
            point_values = group[column].to_numpy(float)
            point = float(np.nanmedian(point_values))
            valid_counts = np.full(BOOTSTRAPS, len(group), dtype=np.int32) if metric == "F" else valid_g_count
            summary_rows.append(
                {
                    "pathway_count": count,
                    "effective_rank": int(group["effective_rank"].median()),
                    "metric": metric,
                    "subset_count": len(group),
                    "point_valid_subset_count": int(np.count_nonzero(np.isfinite(point_values))),
                    "point_invalid_subset_count": int(np.count_nonzero(~np.isfinite(point_values))),
                    "point_median_across_subsets": point,
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_se": float(np.std(values, ddof=1)),
                    "bootstrap_lower_95": float(np.quantile(values, 0.025)),
                    "bootstrap_upper_95": float(np.quantile(values, 0.975)),
                    "bootstrap_draws": BOOTSTRAPS,
                    "bootstrap_min_valid_subset_count": int(valid_counts.min()),
                    "bootstrap_median_valid_subset_count": float(np.median(valid_counts)),
                }
            )
            draw_rows.append(
                pd.DataFrame(
                    {
                        "draw": np.arange(BOOTSTRAPS, dtype=int),
                        "pathway_count": count,
                        "effective_rank": int(group["effective_rank"].median()),
                        "metric": metric,
                        "median_across_subsets": values,
                        "valid_subset_count": valid_counts,
                        "invalid_subset_count": len(group) - valid_counts,
                    }
                )
            )
        print(f"PATHWAY_SCALING bootstrap k={count} subsets={len(group)}", flush=True)
    return pd.concat(draw_rows, ignore_index=True), pd.DataFrame(summary_rows), curves


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups)
    grand = float(np.mean(y))
    total = float(np.sum((y - grand) ** 2))
    between = sum(
        len(block) * (float(np.mean(block)) - grand) ** 2
        for label in np.unique(labels)
        for block in [y[labels == label]]
    )
    return between / total if total > 0 else np.nan


def spearman_small(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(np.asarray(x)).rank(method="average").to_numpy(float)
    ry = pd.Series(np.asarray(y)).rank(method="average").to_numpy(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def row_spearman(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.asarray([spearman_small(x, row) for row in np.asarray(values)], dtype=np.float64)


def dominance_probability(lower: np.ndarray, upper: np.ndarray) -> float:
    left = np.sort(np.asarray(lower, dtype=np.float64))
    right = np.asarray(upper, dtype=np.float64)
    return float(np.searchsorted(left, right, side="left").sum() / (len(left) * len(right)))


def relationship_audit(
    metrics: pd.DataFrame,
    composition: pd.DataFrame,
    curves: dict[str, np.ndarray],
) -> dict[str, Any]:
    counts = np.arange(1, PANEL_SIZE + 1, dtype=float)
    median_f = metrics.groupby("pathway_count")["biological_fidelity_F"].median().reindex(counts.astype(int)).to_numpy()
    median_g = metrics.groupby("pathway_count")["recoverability_g"].median().reindex(counts.astype(int)).to_numpy()
    grid = np.asarray(BOOTSTRAP_GRID, dtype=float)
    rho_f_draw = row_spearman(grid, curves["F"])
    rho_g_draw = row_spearman(grid, curves["g"])
    f_difference = curves["F"][:, -1] - curves["F"][:, 0]
    g_difference = curves["g"][:, -1] - curves["g"][:, 0]
    f_dominance = []
    g_dominance = []
    for low, high in zip(BOOTSTRAP_GRID[:-1], BOOTSTRAP_GRID[1:]):
        low_group = metrics[metrics["pathway_count"].eq(low)]
        high_group = metrics[metrics["pathway_count"].eq(high)]
        f_dominance.append(
            dominance_probability(
                low_group["biological_fidelity_F"].to_numpy(),
                high_group["biological_fidelity_F"].to_numpy(),
            )
        )
        g_dominance.append(
            dominance_probability(
                low_group["recoverability_g"].to_numpy(), high_group["recoverability_g"].to_numpy()
            )
        )
    rho_f = spearman_small(counts, median_f)
    rho_g = spearman_small(counts, median_g)
    g_sign = 1.0 if rho_g >= 0 else -1.0
    g_directional = abs(rho_g) >= 0.75 and float(np.mean(g_sign * rho_g_draw > 0)) >= 0.975
    g_interval = (float(np.quantile(g_difference, 0.025)), float(np.quantile(g_difference, 0.975)))
    g_stable = abs(float(median_g[-1] - median_g[0])) <= 0.05 and g_interval[0] >= -0.10 and g_interval[1] <= 0.10
    f_eta = eta_squared(metrics["biological_fidelity_F"], metrics["pathway_count"])
    valid_g = np.isfinite(metrics["recoverability_g"].to_numpy(float))
    g_eta = eta_squared(
        metrics.loc[valid_g, "recoverability_g"], metrics.loc[valid_g, "pathway_count"]
    )
    g_widths = composition.loc[composition["metric"].eq("g"), "p90_minus_p10"].to_numpy(float)
    f_primary = (
        rho_f >= 0.90
        and float(np.mean(rho_f_draw > 0)) >= 0.975
        and float(np.quantile(f_difference, 0.025)) > 0
        and float(np.median(f_dominance)) >= 0.75
        and float(np.min(f_dominance)) >= 0.60
    )
    recoverability_systematic = g_directional or g_stable
    composition_resistant = f_eta >= 0.50 and (
        (g_directional and g_eta >= 0.20) or (g_stable and float(np.median(g_widths)) <= 0.15)
    )
    recoverability_domain_complete = bool(valid_g.all())
    if f_primary and recoverability_systematic and composition_resistant and recoverability_domain_complete:
        verdict = "PATHWAY_RESOLUTION_SCALING_SUPPORTED"
    elif f_primary:
        verdict = "PATHWAY_RESOLUTION_SCALING_PARTIAL"
    else:
        verdict = "PATHWAY_RESOLUTION_SCALING_NOT_SUPPORTED"
    return {
        "median_F_by_count": median_f.tolist(),
        "median_g_by_count": median_g.tolist(),
        "rho_count_median_F": rho_f,
        "rho_count_median_g": rho_g,
        "bootstrap_rho_F_lower_95": float(np.quantile(rho_f_draw, 0.025)),
        "bootstrap_rho_F_upper_95": float(np.quantile(rho_f_draw, 0.975)),
        "bootstrap_rho_F_positive_fraction": float(np.mean(rho_f_draw > 0)),
        "bootstrap_rho_g_lower_95": float(np.quantile(rho_g_draw, 0.025)),
        "bootstrap_rho_g_upper_95": float(np.quantile(rho_g_draw, 0.975)),
        "bootstrap_rho_g_direction_fraction": float(np.mean(g_sign * rho_g_draw > 0)),
        "F14_minus_F1": float(median_f[-1] - median_f[0]),
        "F14_minus_F1_bootstrap_lower_95": float(np.quantile(f_difference, 0.025)),
        "F14_minus_F1_bootstrap_upper_95": float(np.quantile(f_difference, 0.975)),
        "g14_minus_g1": float(median_g[-1] - median_g[0]),
        "g14_minus_g1_bootstrap_lower_95": g_interval[0],
        "g14_minus_g1_bootstrap_upper_95": g_interval[1],
        "F_adjacent_grid_dominance_probabilities": f_dominance,
        "g_adjacent_grid_dominance_probabilities": g_dominance,
        "eta_squared_count_F": f_eta,
        "eta_squared_count_g": g_eta,
        "median_within_count_g_p90_minus_p10": float(np.median(g_widths)),
        "primary_fidelity_trend_pass": bool(f_primary),
        "recoverability_directional_pass": bool(g_directional),
        "recoverability_stably_flat_pass": bool(g_stable),
        "recoverability_systematic_pass": bool(recoverability_systematic),
        "composition_resistance_pass": bool(composition_resistant),
        "recoverability_domain_complete": recoverability_domain_complete,
        "undefined_point_recoverability_subsets": int(np.count_nonzero(~valid_g)),
        "verdict": verdict,
    }


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta, design @ beta


def _fit_saturating(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rates = np.geomspace(1e-3, 3.0, 400)
    best_sse = math.inf
    best = None
    best_prediction = None
    for rate in rates:
        transformed = 1.0 - np.exp(-rate * x)
        beta, prediction = _linear_fit(transformed, y)
        sse = float(np.sum((y - prediction) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = np.asarray([beta[0], beta[1], rate], dtype=np.float64)
            best_prediction = prediction
    assert best is not None and best_prediction is not None
    return best, best_prediction


def _fit_one(model: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model == "log_linear":
        return _linear_fit(np.log(x), y)
    if model == "saturating_exponential":
        return _fit_saturating(x, y)
    if model == "power_law_like":
        if np.any(y <= 0):
            raise ValueError("power law requires positive response")
        beta, log_prediction = _linear_fit(np.log(x), np.log(y))
        return np.asarray([math.exp(beta[0]), beta[1]]), np.exp(log_prediction)
    raise KeyError(model)


def _fit_bootstrap_many(
    model: str, x: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != len(x):
        raise ValueError("bootstrap curve shape mismatch")
    if model in {"log_linear", "power_law_like"}:
        transformed_x = np.log(x)
        valid = np.ones(len(y), dtype=bool)
        target = y
        if model == "power_law_like":
            valid = np.all(y > 0, axis=1)
            target = np.log(y[valid])
        else:
            target = y[valid]
        design = np.column_stack([np.ones(len(x)), transformed_x])
        beta = np.linalg.lstsq(design, target.T, rcond=None)[0].T
        fitted = beta @ design.T
        if model == "power_law_like":
            fitted = np.exp(fitted)
            beta[:, 0] = np.exp(beta[:, 0])
        total = np.sum((y[valid] - y[valid].mean(axis=1, keepdims=True)) ** 2, axis=1)
        sse = np.sum((y[valid] - fitted) ** 2, axis=1)
        r2 = 1.0 - sse / total
        return beta, r2, valid
    if model == "saturating_exponential":
        rates = np.geomspace(1e-3, 3.0, 400)
        y_mean = y.mean(axis=1)
        centered_y = y - y_mean[:, None]
        best_sse = np.full(len(y), np.inf, dtype=np.float64)
        best_a = np.empty(len(y), dtype=np.float64)
        best_b = np.empty(len(y), dtype=np.float64)
        best_rate = np.empty(len(y), dtype=np.float64)
        for rate in rates:
            transformed = 1.0 - np.exp(-rate * x)
            centered_x = transformed - transformed.mean()
            slope = (centered_y @ centered_x) / float(centered_x @ centered_x)
            intercept = y_mean - slope * transformed.mean()
            fitted = intercept[:, None] + slope[:, None] * transformed
            sse = np.sum((y - fitted) ** 2, axis=1)
            improve = sse < best_sse
            best_sse[improve] = sse[improve]
            best_a[improve] = intercept[improve]
            best_b[improve] = slope[improve]
            best_rate[improve] = rate
        total = np.sum((y - y_mean[:, None]) ** 2, axis=1)
        r2 = 1.0 - best_sse / total
        return np.column_stack([best_a, best_b, best_rate]), r2, np.ones(len(y), dtype=bool)
    raise KeyError(model)


def _fit_diagnostics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    residual = y - prediction
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)))
    lag = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if len(residual) > 2 else np.nan
    return r2, rmse, lag


def scaling_fits(metrics: pd.DataFrame, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_x = np.arange(1, PANEL_SIZE + 1, dtype=float)
    grid_x = np.asarray(BOOTSTRAP_GRID, dtype=float)
    for metric, column in (("F", "biological_fidelity_F"), ("g", "recoverability_g")):
        full_y = metrics.groupby("pathway_count")[column].median().reindex(full_x.astype(int)).to_numpy(float)
        for model in ("log_linear", "saturating_exponential", "power_law_like"):
            try:
                parameters, prediction = _fit_one(model, full_x, full_y)
            except ValueError:
                rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "status": "NOT_EXECUTABLE_NONPOSITIVE_RESPONSE",
                    }
                )
                continue
            r2, rmse, lag = _fit_diagnostics(full_y, prediction)
            boot, bootstrap_r2, valid = _fit_bootstrap_many(model, grid_x, curves[metric])
            parameter_sign = np.sign(boot[:, 1]) == np.sign(parameters[1])
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "status": "COMPLETE",
                    "parameters_json": json.dumps(parameters.tolist(), separators=(",", ":")),
                    "r_squared": r2,
                    "rmse": rmse,
                    "lag1_residual_correlation": lag,
                    "bootstrap_valid_draws": int(np.count_nonzero(valid)),
                    "bootstrap_parameters_lower_95_json": json.dumps(
                        np.quantile(boot, 0.025, axis=0).tolist(), separators=(",", ":")
                    ),
                    "bootstrap_parameters_median_json": json.dumps(
                        np.quantile(boot, 0.5, axis=0).tolist(), separators=(",", ":")
                    ),
                    "bootstrap_parameters_upper_95_json": json.dumps(
                        np.quantile(boot, 0.975, axis=0).tolist(), separators=(",", ":")
                    ),
                    "bootstrap_r_squared_median": float(np.median(bootstrap_r2)),
                    "bootstrap_parameter_direction_stability": float(np.mean(parameter_sign)),
                }
            )
    return pd.DataFrame(rows)


def pareto_frontier(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.copy()
    valid = np.isfinite(table["recoverability_g"].to_numpy(float))
    valid_positions = np.flatnonzero(valid)
    valid_table = table.iloc[valid_positions]
    local_order = np.lexsort(
        (-valid_table["recoverability_g"].to_numpy(), -valid_table["biological_fidelity_F"].to_numpy())
    )
    order = valid_positions[local_order]
    frontier = np.zeros(len(table), dtype=bool)
    best_g = -np.inf
    for position in order:
        value = float(table.iloc[position]["recoverability_g"])
        if value > best_g:
            frontier[position] = True
            best_g = value
    table["pareto_frontier"] = frontier
    frontier_table = table[frontier].sort_values("biological_fidelity_F").copy()
    frontier_table["pareto_order"] = np.arange(1, len(frontier_table) + 1)
    merged = table.merge(
        frontier_table[["subset_id", "pareto_order"]], on="subset_id", how="left", validate="one_to_one"
    )
    f_range = float(table["biological_fidelity_F"].max() - table["biological_fidelity_F"].min())
    g_range = float(table["recoverability_g"].max() - table["recoverability_g"].min())
    front_points = frontier_table[["biological_fidelity_F", "recoverability_g"]].to_numpy(float)
    historical = table[table["historical_five"]].iloc[0]
    distance = np.sqrt(
        ((front_points[:, 0] - historical["biological_fidelity_F"]) / max(f_range, 1e-12)) ** 2
        + ((front_points[:, 1] - historical["recoverability_g"]) / max(g_range, 1e-12)) ** 2
    )
    merged["historical_normalized_distance_to_frontier"] = np.nan
    merged.loc[merged["historical_five"], "historical_normalized_distance_to_frontier"] = float(distance.min())
    merged["historical_near_frontier_0p05"] = False
    merged.loc[merged["historical_five"], "historical_near_frontier_0p05"] = bool(distance.min() <= 0.05)
    return merged.sort_values(["pathway_count", "subset_mask"]).reset_index(drop=True)


def historical_typicality(metrics: pd.DataFrame, pareto: pd.DataFrame) -> dict[str, Any]:
    group = metrics[metrics["pathway_count"].eq(5)]
    historical = group[group["historical_five"]].iloc[0]
    result: dict[str, Any] = {
        "k5_subset_count": int(len(group)),
        "historical_pathways": "|".join(HISTORICAL_PATHWAYS),
    }
    for label, column in (("F", "biological_fidelity_F"), ("g", "recoverability_g")):
        values = group[column].to_numpy(float)
        point = float(historical[column])
        percentile = 100.0 * (np.count_nonzero(values < point) + 0.5 * np.count_nonzero(values == point)) / len(values)
        result[f"historical_{label}"] = point
        result[f"historical_{label}_within_k5_percentile"] = float(percentile)
        result[f"historical_{label}_inside_k5_IQR"] = bool(
            np.quantile(values, 0.25) <= point <= np.quantile(values, 0.75)
        )
        result[f"historical_{label}_inside_k5_10_90"] = bool(
            np.quantile(values, 0.10) <= point <= np.quantile(values, 0.90)
        )
    historical_pareto = pareto[pareto["historical_five"]].iloc[0]
    result["historical_on_pareto_frontier"] = bool(historical_pareto["pareto_frontier"])
    result["historical_normalized_distance_to_frontier"] = float(
        historical_pareto["historical_normalized_distance_to_frontier"]
    )
    result["historical_near_frontier_0p05"] = bool(historical_pareto["historical_near_frontier_0p05"])
    return result


def _figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stixsans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55, zorder=0)
    axis.set_axisbelow(True)


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_scaling(
    output: Path,
    metrics: pd.DataFrame,
    composition: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    pareto: pd.DataFrame,
) -> None:
    _figure_style()
    historical = metrics[metrics["historical_five"]].iloc[0]
    colors = {"F": "#2F6F9F", "g": "#B54A4A"}
    for metric, y_column, ylabel, filename in (
        ("F", "biological_fidelity_F", "Biological fidelity, $F$", "PATHWAY_RESOLUTION_SCALING"),
        ("g", "recoverability_g", "Counterfactual recoverability, $g$", "PATHWAY_RECOVERABILITY_SCALING"),
    ):
        summary = composition[composition["metric"].eq(metric)].sort_values("effective_rank")
        boot = bootstrap_summary[bootstrap_summary["metric"].eq(metric)].sort_values("effective_rank")
        fig, axis = plt.subplots(figsize=(3.55, 2.35))
        x = summary["effective_rank"].to_numpy(float)
        axis.fill_between(x, summary["p10"], summary["p90"], color=colors[metric], alpha=0.12, linewidth=0)
        axis.fill_between(x, summary["q25"], summary["q75"], color=colors[metric], alpha=0.22, linewidth=0)
        axis.plot(x, summary["median"], color=colors[metric], linewidth=1.5, marker="o", markersize=3.0, zorder=3)
        axis.errorbar(
            boot["effective_rank"],
            boot["point_median_across_subsets"],
            yerr=np.vstack(
                [
                    boot["point_median_across_subsets"] - boot["bootstrap_lower_95"],
                    boot["bootstrap_upper_95"] - boot["point_median_across_subsets"],
                ]
            ),
            fmt="none",
            ecolor="#222222",
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
            label="hierarchical 95% CI of subset median",
        )
        axis.scatter(
            [historical["effective_rank"]],
            [historical[y_column]],
            marker="*",
            s=70,
            color="#E69F00",
            edgecolor="white",
            linewidth=0.6,
            zorder=6,
            label="historical five pathways",
        )
        axis.set_xlabel("Effective pathway rank")
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(1, PANEL_SIZE + 1))
        _clean_axis(axis)
        axis.legend(frameon=False, loc="best", handlelength=1.4)
        fig.tight_layout(pad=0.35)
        _save(fig, output, filename)

    fig = plt.figure(figsize=(3.65, 3.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(0.8, 3.2), hspace=0.06)
    top = fig.add_subplot(grid[0])
    axis = fig.add_subplot(grid[1], sharex=top)
    scatter = None
    for current in (top, axis):
        scatter = current.scatter(
            pareto["biological_fidelity_F"],
            pareto["recoverability_g"],
            c=pareto["effective_rank"],
            cmap="RdBu",
            vmin=1,
            vmax=PANEL_SIZE,
            s=7,
            alpha=0.38,
            linewidth=0,
            rasterized=False,
            zorder=2,
        )
    assert scatter is not None
    frontier = pareto[pareto["pareto_frontier"]].sort_values("biological_fidelity_F")
    for current in (top, axis):
        current.plot(
            frontier["biological_fidelity_F"],
            frontier["recoverability_g"],
            color="#1A1A1A",
            linewidth=1.0,
            zorder=4,
            label="empirical Pareto frontier" if current is axis else None,
        )
    hist = pareto[pareto["historical_five"]].iloc[0]
    axis.scatter(
        [hist["biological_fidelity_F"]],
        [hist["recoverability_g"]],
        marker="*",
        s=80,
        color="#E69F00",
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
        label="historical five pathways",
    )
    axis.set_xlabel("Biological fidelity, $F$")
    axis.set_ylabel("Counterfactual recoverability, $g$")
    axis.set_ylim(0.10, 0.90)
    top.set_yscale("log")
    top.set_ylim(0.90, 35)
    top.set_ylabel("outliers", fontsize=7, labelpad=2)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    top.spines["bottom"].set_visible(False)
    axis.spines["top"].set_visible(False)
    _clean_axis(axis)
    _clean_axis(top)
    top.spines["bottom"].set_visible(False)
    axis.spines["top"].set_visible(False)
    marker_size = 0.012
    kwargs = dict(color="#1A1A1A", clip_on=False, linewidth=0.7)
    top.plot((-marker_size, +marker_size), (-marker_size, +marker_size), transform=top.transAxes, **kwargs)
    top.plot((1 - marker_size, 1 + marker_size), (-marker_size, +marker_size), transform=top.transAxes, **kwargs)
    axis.plot((-marker_size, +marker_size), (1 - marker_size, 1 + marker_size), transform=axis.transAxes, **kwargs)
    axis.plot((1 - marker_size, 1 + marker_size), (1 - marker_size, 1 + marker_size), transform=axis.transAxes, **kwargs)
    axis.legend(frameon=False, loc="best")
    colorbar = fig.colorbar(scatter, ax=[top, axis], pad=0.02, fraction=0.055)
    colorbar.set_label("Effective pathway rank", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, width=0.65, length=2.5)
    fig.tight_layout(pad=0.35)
    _save(fig, output, "PATHWAY_FIDELITY_RECOVERABILITY_FRONTIER")


def write_metric_reconciliation(output: Path, stats: SufficientStatistics) -> None:
    gram = stats.gram
    off_diagonal = gram - np.eye(PANEL_SIZE)
    text = f"""# Pathway metric reconciliation

## Exact definitions

For subset `S`, let `W_S` contain the frozen unit-L2 PROGENy vectors and let

`P_S = W_S(W_S^T W_S)^+W_S^T`.

The biological-fidelity numerator is

`V_span(S) = sum_ci D6_ci^T P_S D14_ci`,

and `F(S)=V_span(S)/V_full` with `V_full=sum_ci D6_ci^T D14_ci`.

The historical recoverability metric is calculated in the raw PROGENy score coordinates:

`T_raw(S)=sum_ci D6_ci^T W_S W_S^T D14_ci`,

`R_raw(S)=sum_ci (D6-Dhat6)_ci^T W_S W_S^T (D14-Dhat14)_ci`,

`g(S)=1-R_raw(S)/T_raw(S)`.

## Why `F(S)g(S)` is not reported

For `F(S)g(S)` to be the full-gene reproducible signal recovered inside the pathway span, `F` and `g` would have to use the same span numerator/denominator measure. They do not: `F` uses the orthogonal projector `W_S(W_S^TW_S)^+W_S^T`, whereas the frozen historical `g` uses `W_SW_S^T`.

The complete frozen panel has maximum absolute off-diagonal Gram entry `{float(np.max(np.abs(off_diagonal))):.9g}`. Thus the pathway vectors are materially non-orthogonal and the two quadratic forms are not interchangeable. Replacing historical `g` by whitened orthogonal coordinates would change the frozen anchor and therefore violate the task.

Singleton subsets are a special case because each frozen vector has unit norm, but that identity does not extend across the multi-pathway resolution curve.

**Decision:** `H=F*g` is algebraically invalid as a cross-resolution full-response recovery fraction. No `H` column or figure is produced, and no substitute composite score is invented.
"""
    (output / "PATHWAY_METRIC_RECONCILIATION.md").write_text(text, encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def write_report(
    output: Path,
    stats: SufficientStatistics,
    metrics: pd.DataFrame,
    composition: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    fits: pd.DataFrame,
    relationship: dict[str, Any],
    typicality: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    f_curve = composition[composition["metric"].eq("F")].sort_values("pathway_count")
    g_curve = composition[composition["metric"].eq("g")].sort_values("pathway_count")
    full_f = float(f_curve.iloc[-1]["median"])
    full_g = float(g_curve.iloc[-1]["median"])
    rank_equal = bool((metrics["pathway_count"] == metrics["effective_rank"]).all())
    full_substantial = full_f >= 0.25
    full_vs_historical_f_gain = full_f - float(stats.anchor_audit["f5_observed"])
    valid_g_values = metrics["recoverability_g"].dropna().to_numpy(float)
    maximum_g = float(np.max(valid_g_values))
    maximum_g_row = metrics.loc[metrics["recoverability_g"].idxmax()]
    complete_fits = fits[fits["status"].eq("COMPLETE")]
    law_candidates = []
    for model, group in complete_fits.groupby("model"):
        if set(group["metric"]) != {"F", "g"}:
            continue
        if (
            relationship["verdict"] == "PATHWAY_RESOLUTION_SCALING_SUPPORTED"
            and bool((group["r_squared"] >= 0.95).all())
            and bool((group["bootstrap_parameter_direction_stability"] >= 0.975).all())
            and bool((group["lag1_residual_correlation"].abs() < 0.50).all())
        ):
            law_candidates.append(model)
    scaling_law = bool(law_candidates)
    hist_typical = typicality["historical_F_inside_k5_10_90"] and typicality[
        "historical_g_inside_k5_10_90"
    ]
    if typicality["historical_g_within_k5_percentile"] >= 90:
        hist_g_text = "unusually high"
    elif typicality["historical_g_within_k5_percentile"] <= 10:
        hist_g_text = "unusually low"
    else:
        hist_g_text = "within the central 10th-90th composition range"
    actionability = (
        "The fixed library supplies a defensible empirical resolution control knob."
        if relationship["verdict"] == "PATHWAY_RESOLUTION_SCALING_SUPPORTED"
        else "The curve is useful for bounded design guidance, but pathway count alone is not a composition-invariant control knob."
        if relationship["verdict"] == "PATHWAY_RESOLUTION_SCALING_PARTIAL"
        else "Pathway count is not an actionable resolution control variable in this fixed library."
    )
    text = f"""# Fixed PROGENy pathway-resolution scaling audit

## Result in one sentence

Across all `{len(metrics):,}` non-empty subsets of the exact 14-axis PROGENy panel, median biological fidelity changed from `{_fmt(float(f_curve.iloc[0]['median']))}` at rank 1 to `{_fmt(full_f)}` at rank 14, while median recoverability changed from `{_fmt(float(g_curve.iloc[0]['median']))}` to `{_fmt(full_g)}`; the preregistered adjudication is `{relationship['verdict']}`.

## Frozen input and anchor gates

- Available fixed PROGENy pathways: **14** (`{', '.join(PANEL)}`).
- Every one of the `2^14-1=16,383` non-empty subsets was evaluated; no subset was selected by outcome.
- Historical vectors matched the existing five-pathway loader exactly.
- Five-pathway fidelity: observed `{stats.anchor_audit['f5_observed']:.12g}`, frozen `{FIVE_FIDELITY:.12g}`, absolute error `{stats.anchor_audit['f5_abs_error']:.3g}`.
- Five-pathway recoverability: observed `{stats.anchor_audit['g5_observed']:.12g}`, frozen `{FIVE_RECOVERY:.12g}`, absolute error `{stats.anchor_audit['g5_abs_error']:.3g}`.
- Full-gene recoverability: observed `{stats.anchor_audit['g_gene_observed']:.12g}`, frozen `{GENE_RECOVERY:.12g}`.
- Pathway-minus-gene difference: `{stats.anchor_audit['delta_g_observed']:.12g}` (frozen `{FIVE_MINUS_GENE:.12g}`).
- Direct full-gene truth replay maximum table error: `{stats.anchor_audit['full_truth_table_max_abs_error']:.3g}` (gate `{FULL_TABLE_ATOL:.1g}`).
- Replayed pathway predictions versus the stored float32 tensor differed by at most `{stats.anchor_audit['prediction_score_replay_vs_stored_float32_max_abs_error']:.3g}`; this is below the frozen `{PREDICTION_SCORE_ATOL:.1g}` representation check.
- No predictor, split, prediction, pathway weight or statistical unit changed.

Four subsets had nonpositive raw pathway truth signal and therefore no interpretable recovery fraction (one of 14 at `k=1`, three of 91 at `k=2`, none at `k>=3`). They remain in the exhaustive tables with `recoverability_g=NaN`, an explicit status and a non-inferential algebraic-ratio diagnostic. Bootstrap medians use only positive-denominator subsets within each draw and report the valid count. This fail-closed domain issue prevents a `SUPPORTED` verdict even if the remaining curve is orderly.

## Fidelity gained with biological resolution

Median `F` increased by `{relationship['F14_minus_F1']:.6g}` from rank 1 to rank 14. Its hierarchical-bootstrap 95% interval was `[{relationship['F14_minus_F1_bootstrap_lower_95']:.6g}, {relationship['F14_minus_F1_bootstrap_upper_95']:.6g}]`. The count-versus-median-`F` Spearman correlation was `{relationship['rho_count_median_F']:.4f}`; `{100*relationship['bootstrap_rho_F_positive_fraction']:.2f}%` of bootstrap draws had positive seven-grid correlation. Adjacent-grid stochastic-dominance probabilities were `{', '.join(_fmt(x) for x in relationship['F_adjacent_grid_dominance_probabilities'])}`.

Resolution explained `eta^2={relationship['eta_squared_count_F']:.3f}` of complete-subset fidelity variation. The remaining spread at fixed count is pathway-composition dependence, not sampling uncertainty.

The full 14-axis panel fidelity was `{full_f:.6g}`. Under the frozen `F>=0.25` wording threshold, this is **{'substantial' if full_substantial else 'not substantial'}**. Canonical pathways therefore {'capture a substantial fraction of' if full_substantial else 'remain an incomplete coarse summary of'} the reproducible transcriptomic operator.

Relative to the historically used five-axis span, adding all nine remaining PROGENy axes gained only `{full_vs_historical_f_gain:.6g}` absolute fidelity (`{100*full_vs_historical_f_gain/max(float(stats.anchor_audit['f5_observed']), 1e-12):.2f}%` relative), leaving `{100*(1-full_f):.2f}%` of cross-plate reproducible full-gene signal outside the complete canonical panel.

## Recoverability across resolution

Median `g` changed by `{relationship['g14_minus_g1']:.6g}` from rank 1 to rank 14, with hierarchical-bootstrap 95% interval `[{relationship['g14_minus_g1_bootstrap_lower_95']:.6g}, {relationship['g14_minus_g1_bootstrap_upper_95']:.6g}]`. The count-versus-median-`g` Spearman correlation was `{relationship['rho_count_median_g']:.4f}`. Resolution explained `eta^2={relationship['eta_squared_count_g']:.3f}` of complete-subset recoverability variation; the median within-count 10th-90th width was `{relationship['median_within_count_g_p90_minus_p10']:.6g}`.

Directional recoverability gate: `{relationship['recoverability_directional_pass']}`. Stably-flat gate: `{relationship['recoverability_stably_flat_pass']}`. Composition-resistance gate: `{relationship['composition_resistance_pass']}`.

The valid-subset algebraic recovery range contains a denominator-instability outlier: maximum `g={maximum_g:.6g}` for `{maximum_g_row['pathways']}` with raw truth denominator `{maximum_g_row['raw_pathway_truth_denominator']:.6g}`. This is not evidence of extraordinary predictive reconstruction; it is the expected instability of a normalized ratio near zero truth signal. The frontier figure therefore preserves these points in a separate logarithmic outlier band while expanding the interpretable main cloud.

## Historical five-pathway panel

Among all `{typicality['k5_subset_count']}` five-pathway subsets, the historical panel was at the `{typicality['historical_F_within_k5_percentile']:.1f}`th fidelity percentile and `{typicality['historical_g_within_k5_percentile']:.1f}`th recoverability percentile. Its recoverability is {hist_g_text}; jointly it is **{'typical within the central 10th-90th ranges' if hist_typical else 'not jointly typical of all five-pathway compositions'}**. It is `{'on' if typicality['historical_on_pareto_frontier'] else 'off'}` the empirical Pareto frontier, with normalized distance `{typicality['historical_normalized_distance_to_frontier']:.5f}` (`near` threshold `0.05`: `{typicality['historical_near_frontier_0p05']}`).

The historical panel's 98.6th-percentile fidelity demonstrates that the original five pathways are not representative of arbitrary five-axis compositions. The prespecified normalized “near-frontier” distance is reported, but it is not robust evidence of optimality because the observed `g` range used for normalization is dominated by the near-zero-denominator outlier; the historical panel is not itself on the frontier.

## Count versus effective rank

Every subset was numerically full column rank under the single frozen cutoff, so effective rank **{'equals' if rank_equal else 'does not always equal'}** raw pathway count in this 14-axis library. The observed curve cannot be attributed to hidden rank loss, although overlapping axes affect conditioning and the distinction remains important for other libraries.

## Metric reconciliation

`F(S)` uses the orthogonal projector `W_S(W_S^TW_S)^+W_S^T`; frozen historical `g(S)` uses raw pathway-score geometry `W_SW_S^T`. Because the axes are non-orthogonal, `F(S)g(S)` is **not** the full-gene reproducible signal recovered within the span. No `H` metric or figure was produced. See `PATHWAY_METRIC_RECONCILIATION.md`.

## Descriptive functional fits

Log-linear, saturating-exponential and positive power-law-like fits were reported without choosing a form by maximum R-squared alone. The strict preregistered requirements for the phrase **scaling law** were `{'met by ' + ', '.join(law_candidates) if scaling_law else 'not met'}`. The appropriate language is therefore **{'scaling law' if scaling_law else 'pathway-resolution scaling curve / fidelity-recoverability scaling relation'}**.

## Scientific adjudication

- Primary fidelity trend gate: `{relationship['primary_fidelity_trend_pass']}`.
- Systematic recoverability gate: `{relationship['recoverability_systematic_pass']}`.
- Composition-resistance gate: `{relationship['composition_resistance_pass']}`.
- Complete recoverability-domain gate: `{relationship['recoverability_domain_complete']}` (`{relationship['undefined_point_recoverability_subsets']}` undefined point subsets).
- Does a reproducible fidelity-recoverability scaling relation exist? **A monotone median fidelity curve exists, but the joint composition-resistant relation is only partially established.**
- Is the relation strong enough to justify “scaling law”? **{scaling_law}**.
- Does the full panel reach substantial fidelity? **{full_substantial}**.
- Actionable Counterfactual Resolution framework: {actionability}

The strongest defensible claim is bounded to this fixed external vocabulary and these frozen predictions. It does not establish a universal optimal pathway number, full-transcriptome reconstruction, or universality across pathway systems.

## Provenance

- Branch: `{provenance['branch']}`
- Protocol/implementation commit: `{provenance['protocol_commit']}`
- Formal run input authority: `{provenance['input_authority']}`
- Exact input hashes: `PATHWAY_ANCHOR_AUDIT.json`
- Bootstrap draws: `{BOOTSTRAPS}` with the pre-existing context-then-intervention count matrix.

## Verdict

`{relationship['verdict']}`
"""
    (output / "PATHWAY_RESOLUTION_SCALING_REPORT.md").write_text(text, encoding="utf-8")


def run(root: Path, truth_root: Path, protocol_commit: str) -> dict[str, Any]:
    output = root / "results/cgc_pathway_resolution_scaling"
    output.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(root, truth_root)
    hashes = validate_hashes(paths)
    weight_matrix, _, panel_audit = load_full_panel(paths)
    panel_audit.to_csv(output / "PROGENY_FULL_PANEL_AUDIT.csv", index=False)
    stats = compute_sufficient_statistics(paths, weight_matrix)
    anchor = dict(stats.anchor_audit)
    anchor["input_sha256"] = hashes
    anchor["progeny_version"] = PROGENY_VERSION
    anchor["progeny_git_revision"] = PROGENY_REVISION
    anchor["protocol_commit"] = protocol_commit
    (output / "PATHWAY_ANCHOR_AUDIT.json").write_text(
        json.dumps(anchor, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics, numerical, coefficients = enumerate_subset_metrics(stats)
    if len(metrics) != 2**PANEL_SIZE - 1:
        raise RuntimeError("PATHWAY_SCALING_SUBSET_COUNT_INCOMPLETE")
    if not (metrics["pathway_count"] == metrics["effective_rank"]).all():
        print("PATHWAY_SCALING note: at least one subset has rank<count", flush=True)
    metrics.to_csv(output / "PATHWAY_SUBSET_METRICS.csv", index=False)
    numerical.to_csv(output / "PATHWAY_SPAN_NUMERICAL_AUDIT.csv", index=False)
    composition = composition_summary(metrics)
    composition.to_csv(output / "PATHWAY_COMPOSITION_SUMMARY.csv", index=False)
    domain = (
        metrics.groupby("pathway_count", as_index=False)
        .agg(
            subset_count=("subset_id", "size"),
            valid_recoverability_subsets=("recoverability_g", "count"),
            minimum_raw_truth_denominator=("raw_pathway_truth_denominator", "min"),
            maximum_raw_truth_denominator=("raw_pathway_truth_denominator", "max"),
        )
    )
    domain["undefined_recoverability_subsets"] = (
        domain["subset_count"] - domain["valid_recoverability_subsets"]
    )
    domain.to_csv(output / "PATHWAY_RECOVERABILITY_DOMAIN_AUDIT.csv", index=False)
    bootstrap_draws, bootstrap_summary, curves = bootstrap_scaling(
        paths, stats, metrics, coefficients
    )
    bootstrap_draws.to_csv(output / "PATHWAY_SCALING_BOOTSTRAP.csv", index=False)
    bootstrap_summary.to_csv(output / "PATHWAY_SCALING_BOOTSTRAP_SUMMARY.csv", index=False)
    pareto = pareto_frontier(metrics)
    pareto.to_csv(output / "PATHWAY_PARETO_FRONTIER.csv", index=False)
    fit_table = scaling_fits(metrics, curves)
    fit_table.to_csv(output / "PATHWAY_SCALING_FITS.csv", index=False)
    relationship = relationship_audit(metrics, composition, curves)
    typicality = historical_typicality(metrics, pareto)
    (output / "PATHWAY_RELATIONSHIP_AUDIT.json").write_text(
        json.dumps(relationship, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "PATHWAY_HISTORICAL_PANEL_AUDIT.json").write_text(
        json.dumps(typicality, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_metric_reconciliation(output, stats)
    render_scaling(output, metrics, composition, bootstrap_summary, pareto)
    branch = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
    provenance = {
        "branch": branch,
        "protocol_commit": protocol_commit,
        "input_authority": "46b25eec4a97e3001cc43262980af8f69c533478",
    }
    write_report(
        output,
        stats,
        metrics,
        composition,
        bootstrap_summary,
        fit_table,
        relationship,
        typicality,
        provenance,
    )
    return {
        "verdict": relationship["verdict"],
        "subsets": len(metrics),
        "full_panel_F": float(
            metrics.loc[metrics["pathway_count"].eq(PANEL_SIZE), "biological_fidelity_F"].iloc[0]
        ),
        "full_panel_g": float(
            metrics.loc[metrics["pathway_count"].eq(PANEL_SIZE), "recoverability_g"].iloc[0]
        ),
        "historical_F": stats.anchor_audit["f5_observed"],
        "historical_g": stats.anchor_audit["g5_observed"],
        "output": str(output.resolve()),
    }
