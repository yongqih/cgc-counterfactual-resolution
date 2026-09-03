from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from igc_virtual_cell.cgc_resolution_poc.replay import BUDGETS, CONTEXTS, GENE_COUNT, INTERVENTIONS


PATHWAYS = ("MAPK", "PI3K", "JAK-STAT", "p53", "NFkB")
CANDIDATES = 5_000
MATCHED = 500
MASTER_SEED = 20_260_822


def _seed(*parts: object) -> int:
    text = "|".join(map(str, (MASTER_SEED,) + parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def load_pathway_weights(truth_root: Path) -> tuple[np.ndarray, list[str]]:
    metadata = pd.read_csv(truth_root / "results/cgc_tahoe_0i/gene_metadata_frozen.csv")
    indices = np.load(truth_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy").astype(int)
    symbols = metadata.set_index("raw_gene_index").loc[indices, "gene_symbol"].astype(str).tolist()
    if len(symbols) != GENE_COUNT or len(set(symbols)) != GENE_COUNT:
        raise RuntimeError("RESOLUTION2_PATHWAY_GENE_MAPPING_AMBIGUOUS")
    source = pd.read_parquet(truth_root / "data/cgc_bio1_official/frozen_program_weights.parquet")
    source = source[(source["space"] == "PROGENy") & source["program"].isin(PATHWAYS)]
    weights = np.zeros((len(PATHWAYS), GENE_COUNT), dtype=np.float64)
    position = {symbol: index for index, symbol in enumerate(symbols)}
    for h, pathway in enumerate(PATHWAYS):
        rows = source[source["program"] == pathway]
        if rows.empty or rows["gene_symbol"].duplicated().any():
            raise RuntimeError(f"RESOLUTION2_PATHWAY_MAPPING_FAIL:{pathway}")
        for row in rows.itertuples(index=False):
            if row.gene_symbol in position:
                weights[h, position[row.gene_symbol]] = float(row.weight)
        norm = float(np.linalg.norm(weights[h]))
        if not np.isclose(norm, 1.0, atol=1e-12, rtol=0):
            raise RuntimeError(f"RESOLUTION2_PATHWAY_NORM_FAIL:{pathway}:{norm}")
    return weights, symbols


def gene_matching_bins(truth_root: Path, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.load(truth_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy").astype(int)
    group = zarr.open_group(truth_root / "results/cgc_tahoe_0i/response_tensors.zarr", mode="r")
    baseline = np.asarray(group["primary_dmso_cpm"][:, :, indices], dtype=np.float64).mean(axis=(0, 1))
    variance = np.empty(GENE_COUNT, dtype=np.float64)
    for start in range(0, GENE_COUNT, 1_024):
        stop = min(start + 1_024, GENE_COUNT)
        block = np.asarray(values[..., start:stop], dtype=np.float64).reshape(-1, stop - start)
        variance[start:stop] = block.var(axis=0, ddof=0)
    def deciles(x: np.ndarray) -> np.ndarray:
        edges = np.quantile(x, np.linspace(0, 1, 11))
        return np.searchsorted(edges[1:-1], x, side="right").astype(np.int8)
    expr_bin, var_bin = deciles(baseline), deciles(variance)
    return expr_bin, var_bin, variance


def project_raw(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.empty((2, CONTEXTS, INTERVENTIONS, weights.shape[0]), dtype=np.float64)
    for plate in range(2):
        projected = np.asarray(values[plate], dtype=np.float64).reshape(
            CONTEXTS * INTERVENTIONS, GENE_COUNT
        ) @ weights.T
        result[plate] = projected.reshape(CONTEXTS, INTERVENTIONS, weights.shape[0])
    return result


def episode_scores(raw: np.ndarray, replay_root: Path, m: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    frozen = np.load(replay_root / f"m{m}_k{k}_frozen_weights.npz")
    weights = np.asarray(frozen["weights"], dtype=np.float64)
    sources = np.asarray(frozen["sources"], dtype=np.int64)
    sequence_count = sources.shape[0]
    coordinates = raw.shape[-1]
    truth = np.empty((2, sequence_count, CONTEXTS, INTERVENTIONS, coordinates), dtype=np.float64)
    prediction = np.empty_like(truth)
    uniform = np.full(m, 1.0 / m, dtype=np.float64)
    for sequence in range(sequence_count):
        for target in range(CONTEXTS):
            selected = sources[sequence, target]
            for plate in range(2):
                block = raw[plate, selected]
                truth[plate, sequence, target] = raw[plate, target] - block.mean(axis=0)
                prediction[plate, sequence, target] = np.einsum(
                    "ps,sph->ph", weights[plate, sequence, target] - uniform, block, optimize=True
                )
    return truth, prediction


def utility(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vtruth = (truth[0] * truth[1]).mean(axis=0)
    residual6 = truth[0] - prediction[0]
    residual14 = truth[1] - prediction[1]
    vafter = (residual6 * residual14).mean(axis=0)
    return vtruth, vafter


def recovery(vtruth: np.ndarray, vafter: np.ndarray, *, require_positive: bool = True) -> float:
    denominator = float(np.sum(vtruth, dtype=np.float64))
    if not np.isfinite(denominator) or denominator == 0 or (require_positive and denominator < 0):
        raise RuntimeError("RESOLUTION2_PATHWAY_DENOMINATOR_INVALID")
    return 1.0 - float(np.sum(vafter, dtype=np.float64)) / denominator


def _candidate_weights(
    real: np.ndarray, expr_bin: np.ndarray, var_bin: np.ndarray, pathway: str, candidate_ids: np.ndarray
) -> np.ndarray:
    result = np.zeros((GENE_COUNT, len(candidate_ids)), dtype=np.float32)
    joint = expr_bin.astype(int) * 10 + var_bin.astype(int)
    nonzero = np.flatnonzero(real != 0)
    for column, candidate in enumerate(map(int, candidate_ids)):
        rng = np.random.default_rng(_seed("pathway", pathway, candidate))
        for label in np.unique(joint[nonzero]):
            source_positions = nonzero[joint[nonzero] == label]
            universe = np.flatnonzero(joint == label)
            if len(universe) < len(source_positions):
                raise RuntimeError("RESOLUTION2_MATCHING_BIN_EXHAUSTED")
            chosen = rng.choice(universe, size=len(source_positions), replace=False)
            assigned = real[source_positions][rng.permutation(len(source_positions))]
            result[chosen, column] = assigned.astype(np.float32)
    return result


def build_candidate_raw_scores(
    values: np.ndarray,
    real_weights: np.ndarray,
    expr_bin: np.ndarray,
    var_bin: np.ndarray,
    cache_root: Path,
) -> dict[str, Path]:
    cache_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    x = [np.asarray(values[plate], dtype=np.float32).reshape(CONTEXTS * INTERVENTIONS, GENE_COUNT) for plate in range(2)]
    for h, pathway in enumerate(PATHWAYS):
        path = cache_root / f"{pathway}_candidate_raw_scores_float32.npy"
        paths[pathway] = path
        shape = (2, CONTEXTS, INTERVENTIONS, CANDIDATES)
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if existing.shape == shape and existing.dtype == np.float32:
                continue
            raise RuntimeError("RESOLUTION2_PATHWAY_NULL_CACHE_MISMATCH")
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, CANDIDATES, 100):
            stop = min(start + 100, CANDIDATES)
            candidate = _candidate_weights(real_weights[h], expr_bin, var_bin, pathway, np.arange(start, stop))
            for plate in range(2):
                output[plate, :, :, start:stop] = (x[plate] @ candidate).reshape(CONTEXTS, INTERVENTIONS, stop - start)
            output.flush()
            print(f"RESOLUTION2 pathway-null {pathway} candidates={stop}/{CANDIDATES}", flush=True)
    return paths


def _truth_only_scores(raw: np.ndarray, replay_root: Path, m: int, k: int) -> np.ndarray:
    frozen = np.load(replay_root / f"m{m}_k{k}_frozen_weights.npz")
    sources = np.asarray(frozen["sources"], dtype=np.int64)
    truth = np.empty((2, len(sources), CONTEXTS, INTERVENTIONS, raw.shape[-1]), dtype=np.float64)
    for sequence in range(len(sources)):
        for target in range(CONTEXTS):
            selected = sources[sequence, target]
            truth[:, sequence, target] = raw[:, target] - raw[:, selected].mean(axis=1)
    return truth


def _matching_summaries(truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    averaged_cross = (truth[0] * truth[1]).mean(axis=0)
    variance = averaged_cross.sum(axis=(0, 1), dtype=np.float64)
    x = truth[0].reshape(-1, truth.shape[-1])
    y = truth[1].reshape(-1, truth.shape[-1])
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    covariance = np.sum(x * y, axis=0, dtype=np.float64)
    reliability = covariance / np.sqrt(
        np.sum(x * x, axis=0, dtype=np.float64) * np.sum(y * y, axis=0, dtype=np.float64)
    )
    return variance, reliability


def select_matches(real_truth: np.ndarray, candidate_raw: np.ndarray, replay_root: Path, m: int, k: int) -> tuple[np.ndarray, pd.DataFrame]:
    real_variance, real_reliability = _matching_summaries(real_truth)
    variance = np.empty(CANDIDATES, dtype=np.float64)
    reliability = np.empty(CANDIDATES, dtype=np.float64)
    for start in range(0, CANDIDATES, 100):
        stop = min(start + 100, CANDIDATES)
        truth = _truth_only_scores(np.asarray(candidate_raw[..., start:stop], dtype=np.float64), replay_root, m, k)
        variance[start:stop], reliability[start:stop] = _matching_summaries(truth)
    rv, rr = float(real_variance[0]), float(real_reliability[0])
    relative = np.abs(variance - rv) / max(abs(rv), 1e-12)
    fisher = np.abs(np.arctanh(np.clip(reliability, -0.999999, 0.999999)) - np.arctanh(np.clip(rr, -0.999999, 0.999999)))
    exact = (relative <= 0.10) & (fisher <= 0.05)
    logged_variance = np.log(np.maximum(variance, 1e-12))
    scale_v = max(float(np.median(np.abs(logged_variance - np.median(logged_variance)))), 1e-9)
    scale_r = max(float(np.median(np.abs(np.arctanh(np.clip(reliability, -0.999999, 0.999999)) - np.median(np.arctanh(np.clip(reliability, -0.999999, 0.999999)))))), 1e-9)
    distance = np.sqrt((np.log(np.maximum(variance, 1e-12) / max(rv, 1e-12)) / scale_v) ** 2 + (fisher / scale_r) ** 2)
    exact_ids = np.flatnonzero(exact)
    if len(exact_ids) >= MATCHED:
        selected = exact_ids[np.lexsort((exact_ids, distance[exact_ids]))[:MATCHED]]
    else:
        selected = np.lexsort((np.arange(CANDIDATES), distance))[:MATCHED]
    table = pd.DataFrame({
        "candidate": selected,
        "truth_variance": variance[selected],
        "replicate_reliability": reliability[selected],
        "relative_variance_error": relative[selected],
        "fisher_reliability_error": fisher[selected],
        "exact_match": exact[selected],
        "exact_pool_size": int(exact.sum()),
        "distance": distance[selected],
    })
    return selected, table


def run_pathway_analysis(root: Path, truth_root: Path) -> dict[str, Any]:
    output = root / "results/cgc_resolution_poc_v2"
    replay_root = root / "data/cgc_resolution2_replay"
    cache_root = replay_root / "pathway_null"
    values = np.load(replay_root / "truth_g_primary_float32.npy", mmap_mode="r")
    weights, _ = load_pathway_weights(truth_root)
    expr_bin, var_bin, _ = gene_matching_bins(truth_root, values)
    real_raw = project_raw(values, weights)
    candidate_paths = build_candidate_raw_scores(values, weights, expr_bin, var_bin, cache_root)
    result_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}
    for m, k, _ in BUDGETS:
        real_truth, real_prediction = episode_scores(real_raw, replay_root, m, k)
        real_vtruth, real_vafter = utility(real_truth, real_prediction)
        saved[f"real_vtruth_m{m}_k{k}"] = real_vtruth
        saved[f"real_vafter_m{m}_k{k}"] = real_vafter
        for label, index in [("POOLED", None)] + [(p, i) for i, p in enumerate(PATHWAYS)]:
            vt = real_vtruth if index is None else real_vtruth[..., index]
            va = real_vafter if index is None else real_vafter[..., index]
            result_rows.append({"m": m, "k": k, "pathway": label, "g": recovery(vt, va),
                                "truth_denominator": float(vt.sum()), "n_interventions": INTERVENTIONS,
                                "analysis_status": "COMPLETE"})
        null_vtruth_panel = np.zeros((MATCHED, CONTEXTS, INTERVENTIONS), dtype=np.float64)
        null_vafter_panel = np.zeros_like(null_vtruth_panel)
        exact_by_pathway: list[np.ndarray] = []
        for h, pathway in enumerate(PATHWAYS):
            candidates = np.load(candidate_paths[pathway], mmap_mode="r")
            selected, matching = select_matches(real_truth[..., h : h + 1], candidates, replay_root, m, k)
            matching.insert(0, "pathway", pathway)
            matching.insert(0, "k", k)
            matching.insert(0, "m", m)
            exact_by_pathway.append(matching["exact_match"].to_numpy(bool))
            reliability_rows.extend(matching.to_dict("records"))
            selected_raw = np.asarray(candidates[..., selected], dtype=np.float64)
            truth, prediction = episode_scores(selected_raw, replay_root, m, k)
            vt, va = utility(truth, prediction)
            vt = np.moveaxis(vt, -1, 0)
            va = np.moveaxis(va, -1, 0)
            saved[f"null_vtruth_{pathway}_m{m}_k{k}"] = vt
            saved[f"null_vafter_{pathway}_m{m}_k{k}"] = va
            null_vtruth_panel += vt
            null_vafter_panel += va
            for rank, candidate in enumerate(selected):
                null_rows.append({"m": m, "k": k, "pathway": pathway, "match_rank": rank,
                                  "candidate": int(candidate), "g_random": recovery(
                                      vt[rank], va[rank], require_positive=False
                                  ),
                                  "exact_match": bool(matching.iloc[rank]["exact_match"])})
        saved[f"null_vtruth_m{m}_k{k}"] = null_vtruth_panel
        saved[f"null_vafter_m{m}_k{k}"] = null_vafter_panel
        for rank in range(MATCHED):
            null_rows.append({"m": m, "k": k, "pathway": "POOLED", "match_rank": rank,
                              "candidate": rank, "g_random": recovery(null_vtruth_panel[rank], null_vafter_panel[rank]),
                              "exact_match": bool(all(values[rank] for values in exact_by_pathway))})
    pd.DataFrame(result_rows).to_csv(output / "RESOLUTION2_PATHWAY_RESULTS.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output / "RESOLUTION2_PATHWAY_RANDOM_NULL.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output / "RESOLUTION2_RELIABILITY_MATCHING.csv", index=False)
    np.savez_compressed(cache_root / "pathway_frozen_utilities.npz", **saved)
    return {"results": result_rows, "null_rows": len(null_rows), "matching_rows": len(reliability_rows)}
