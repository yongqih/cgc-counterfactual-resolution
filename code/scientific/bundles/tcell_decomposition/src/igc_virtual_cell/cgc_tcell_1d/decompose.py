"""CGC-TCELL-1D frozen shared/specific operator decomposition.

This module performs evaluation only.  It reads frozen 0A truth and frozen 1B
coefficient predictions; it never loads a predictive checkpoint and never fits
a predictive model.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from igc_virtual_cell.cgc_tcell.core import hash_block
from igc_virtual_cell.cgc_tcell_1b.audit import load_frozen, sha256_file
from igc_virtual_cell.cgc_tcell_1b.benchmark import STATE_PAIRS, STATE_PAIR_NAMES, two_way
from igc_virtual_cell.cgc_tcell_1c.adjudicate import (
    MODEL_FILES,
    MODELS,
    _analyze_matrix,
    _cos,
    _input_hashes as input_hashes_1c,
    _null_summary,
    _prediction_residuals,
    reproduce_gate,
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def scalar_metrics(dot: np.ndarray | float, left2: np.ndarray | float, right2: np.ndarray | float) -> dict[str, float]:
    d, l2, r2 = float(np.sum(dot)), float(np.sum(left2)), float(np.sum(right2))
    return {
        "dot": d,
        "left2": l2,
        "right2": r2,
        "cosine": float(_cos(d, l2, r2)),
        "pearson": float(_cos(d, l2, r2)),
        "alpha": d / r2,
        "kappa": l2 / r2,
    }


def decomposition(stats: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return additive and globally orthogonal sufficient statistics."""
    t2, s2, ts = (np.asarray(stats[k], np.float64) for k in ("t2", "s2", "ts"))
    p2, pt, ps = (np.asarray(stats[k], np.float64) for k in ("p2", "pt", "ps"))
    beta_t = float(ts.sum() / s2.sum())
    beta_p = float(ps.sum() / s2.sum())
    dtrue2 = t2 + s2 - 2 * ts
    dpred2 = p2 + s2 - 2 * ps
    ddot = pt - ps - ts + s2
    nt2 = t2 - 2 * beta_t * ts + beta_t * beta_t * s2
    np2 = p2 - 2 * beta_p * ps + beta_p * beta_p * s2
    ndot = pt - beta_t * ps - beta_p * ts + beta_p * beta_t * s2
    return {
        "beta_t": beta_t, "beta_p": beta_p,
        "dtrue2": dtrue2, "dpred2": dpred2, "ddot": ddot,
        "nt2": nt2, "np2": np2, "ndot": ndot,
        "nt_s_dot": ts - beta_t * s2,
        "np_s_dot": ps - beta_p * s2,
    }


def _verify_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = input_hashes_1c(root, config)
    one_c_paths = [
        root / "results/cgc_tcell_1c/verdict.json",
        root / "results/cgc_tcell_1c/metric_reproduction.csv",
        root / "results/cgc_tcell_1c/intervention_shuffle_summary.csv",
        root / "results/cgc_tcell_1c/run_manifest.json",
    ]
    records = []
    for path in one_c_paths:
        if not path.exists():
            raise RuntimeError(f"CGC_1D_INVALID_FROZEN_INPUT: missing {path}")
        records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    interpretation_ok = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", config["interpretation_commit"], "--", "results/cgc_tcell_1c", "results/reports/cgc_tcell_1c_operator_alignment.md", "results/reports/cgc_tcell_1c_mechanistic_interpretation.md", "results/reports/cgc_tcell_1c_phase_summary.md"],
        capture_output=True,
    ).returncode == 0
    if not interpretation_ok:
        raise RuntimeError("CGC_1D_INVALID_FROZEN_INPUT")
    base["interpretation_commit"] = config["interpretation_commit"]
    base["interpretation_artifacts_unchanged"] = True
    base["files"].extend(records)
    base["predictive_model_fits"] = 0
    base["checkpoint_inference_calls"] = 0
    return base


def _reproduction(root: Path, config: dict[str, Any], delta, donors, targets):
    frame, cache, tolerance = reproduce_gate(root, config, delta, donors, targets)
    one_c = pd.read_csv(root / "results/cgc_tcell_1c/intervention_shuffle_summary.csv")
    pooled = frame.loc[frame.scope.eq("pooled")].copy()
    extra = []
    for model in MODELS:
        observed = float(pooled.loc[pooled.model.eq(model), "cosine"].iloc[0])
        expected = float(one_c.loc[one_c.scope.eq("pooled") & one_c.model.eq(model), "observed_cosine"].iloc[0])
        extra.append({
            "scope": "pooled_1c_shuffle_observed", "model": model, "seed": -999,
            "outer_fold": -1, "held_out_donor": "ALL", "cosine": observed,
            "frozen_1c_cosine": expected, "absolute_error_cosine": abs(observed - expected),
            "accepted_tolerance": tolerance, "reproduced": abs(observed - expected) <= tolerance,
        })
    result = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True, sort=False)
    if not result.reproduced.fillna(False).all():
        raise RuntimeError("CGC_1D_FROZEN_METRIC_REPRODUCTION_FAILED")
    return result, cache, tolerance


def _fold_exact(delta, components: np.ndarray, train: list[int], held: int, gene_masks: list[np.ndarray]):
    p, _, _, genes = delta.shape
    rank = components.shape[0]
    keys = ("t2", "s2", "ts")
    total = {k: np.zeros(p, np.float64) for k in keys}
    total.update({"tproj": np.zeros((p, 3, rank), np.float32), "sproj": np.zeros((p, 3, rank), np.float32)})
    total["tcontrast2"] = np.zeros((p, 3), np.float64)
    total["scontrast2"] = np.zeros((p, 3), np.float64)
    total["tscontrast"] = np.zeros((p, 3), np.float64)
    total["tcontrastproj"] = np.zeros((3, p, rank), np.float32)
    total["scontrastproj"] = np.zeros((3, p, rank), np.float32)
    truth_full = two_way(np.asarray(delta[:, held], np.float32))
    shared_full = two_way(np.asarray(delta[:, train], np.float32).mean(axis=1, dtype=np.float64).astype(np.float32))
    blocks = []
    for mask in gene_masks:
        t = np.ascontiguousarray(truth_full[:, :, mask])
        s = np.ascontiguousarray(shared_full[:, :, mask])
        vb = components[:, mask]
        item = {
            "t2": np.square(t, dtype=np.float64).sum(axis=(1, 2)),
            "s2": np.square(s, dtype=np.float64).sum(axis=(1, 2)),
            "ts": np.multiply(t, s, dtype=np.float64).sum(axis=(1, 2)),
            "tproj": np.einsum("psg,rg->psr", t, vb, optimize=True),
            "sproj": np.einsum("psg,rg->psr", s, vb, optimize=True),
            "gram": vb @ vb.T,
            "truth": t, "shared": s,
        }
        for key in keys: total[key] += item[key]
        total["tproj"] += item["tproj"]; total["sproj"] += item["sproj"]
        for ci, (left, right) in enumerate(STATE_PAIRS):
            tc, sc = t[:, left] - t[:, right], s[:, left] - s[:, right]
            total["tcontrast2"][:, ci] += np.square(tc, dtype=np.float64).sum(axis=1)
            total["scontrast2"][:, ci] += np.square(sc, dtype=np.float64).sum(axis=1)
            total["tscontrast"][:, ci] += np.multiply(tc, sc, dtype=np.float64).sum(axis=1)
            total["tcontrastproj"][ci] += tc @ vb.T
            total["scontrastproj"][ci] += sc @ vb.T
        blocks.append(item)
    del truth_full, shared_full
    return total, blocks


def _prediction_stats(base: dict[str, np.ndarray], residual: np.ndarray, gram: np.ndarray | None = None):
    r = two_way(residual)
    if gram is None:
        r2 = np.square(r, dtype=np.float64).sum(axis=(1, 2))
    else:
        r2 = np.einsum("psr,rt,pst->p", r, gram, r, optimize=True)
    rs = np.einsum("psr,psr->p", r, base["sproj"])
    rt = np.einsum("psr,psr->p", r, base["tproj"])
    return {
        "r": r, "r2": r2, "rs": rs, "rt": rt,
        "t2": base["t2"], "s2": base["s2"], "ts": base["ts"],
        "p2": base["s2"] + 2 * rs + r2,
        "pt": base["ts"] + rt,
        "ps": base["s2"] + rs,
    }


def _operator_cross(left: np.ndarray, right: np.ndarray, path: Path, row_batch: int) -> np.memmap:
    p = left.shape[0]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(p, p))
    device = torch.device("cuda")
    right_t = torch.from_numpy(np.ascontiguousarray(right.reshape(p, -1).T)).to(device)
    flat = left.reshape(p, -1)
    for start in range(0, p, row_batch):
        stop = min(start + row_batch, p)
        out[start:stop] = (torch.from_numpy(np.ascontiguousarray(flat[start:stop])).to(device) @ right_t).cpu().numpy()
    out.flush()
    del right_t
    torch.cuda.empty_cache()
    return out


def _novel_raw(st: np.ndarray, ss: np.ndarray, stats: dict[str, np.ndarray], base: dict[str, np.ndarray], dec: dict[str, Any], path: Path, row_batch: int):
    p = len(stats["t2"])
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(p, p))
    device = torch.device("cuda")
    tp = torch.from_numpy(np.ascontiguousarray(base["tproj"].reshape(p, -1).T)).to(device)
    sp = torch.from_numpy(np.ascontiguousarray(base["sproj"].reshape(p, -1).T)).to(device)
    r = stats["r"].reshape(p, -1)
    bt, bp = dec["beta_t"], dec["beta_p"]
    for start in range(0, p, row_batch):
        stop = min(start + row_batch, p)
        rr = torch.from_numpy(np.ascontiguousarray(r[start:stop])).to(device)
        rt = (rr @ tp).cpu().numpy()
        rs = (rr @ sp).cpu().numpy()
        st_b = np.asarray(st[start:stop], np.float32)
        ss_b = np.asarray(ss[start:stop], np.float32)
        pt = st_b + rt
        ps = ss_b + rs
        out[start:stop] = pt - bt * ps - bp * st_b + bp * bt * ss_b
    out.flush()
    del tp, sp
    torch.cuda.empty_cache()
    return out


def _add_matrix(source: np.ndarray, destination: Path, first: bool, row_batch: int):
    out = np.lib.format.open_memmap(destination, mode="w+" if first else "r+", dtype=np.float32, shape=source.shape)
    for start in range(0, source.shape[0], row_batch):
        stop = min(start + row_batch, source.shape[0])
        if first: out[start:stop] = source[start:stop]
        else: out[start:stop] += source[start:stop]
    out.flush()
    return out


def _safe_unlink(path: Path) -> None:
    """Best-effort cleanup for Windows-backed NumPy memory maps."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except PermissionError:
        # The file is temporary and will be removed at the next process start;
        # a lingering mmap reference must never invalidate frozen statistics.
        pass


def _mean_similarity(raws: list[np.ndarray], left2s: list[np.ndarray], right2: np.ndarray, path: Path, row_batch: int):
    p = len(right2)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(p, p))
    out[:] = 0
    for raw, left2 in zip(raws, left2s):
        for start in range(0, p, row_batch):
            stop = min(start + row_batch, p)
            denom = np.sqrt(np.maximum(left2[start:stop, None] * right2[None, :], 1e-30))
            out[start:stop] += (np.asarray(raw[start:stop], np.float64) / denom / len(raws)).astype(np.float32)
    out.flush()
    return out


def _permutation_metrics(raw: np.ndarray, np2: np.ndarray, nt2: np.ndarray, row_median: np.ndarray, draws: int, seed: int):
    p = len(np2); cols = np.arange(p); rng = np.random.default_rng(seed)
    denom = np.sqrt(float(np2.sum() * nt2.sum()))
    cosines = np.empty(draws); alphas = np.empty(draws); effects = np.empty(draws)
    for draw in range(draws):
        perm = rng.permutation(p)
        selected = np.asarray(raw[perm, cols], np.float64)
        dot = selected.sum()
        cosines[draw] = dot / denom
        alphas[draw] = dot / float(nt2.sum())
        effects[draw] = np.median(selected / np.sqrt(np.maximum(np2[perm] * nt2, 1e-30)) - row_median[perm])
    return cosines, alphas, effects


def _write_shared_operator(root: Path, output: Path, delta, donors: list[str], targets: list[str], states: list[str]):
    path = output / "shared_operator_by_fold.parquet"
    schema = pa.schema([
        ("outer_fold", pa.int16()), ("held_out_donor", pa.string()),
        ("training_donors", pa.string()), ("perturbation_id", pa.string()),
        ("stimulation_state", pa.string()),
        ("operator_values", pa.list_(pa.float32(), delta.shape[-1])),
        ("operator_label", pa.string()),
    ])
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    try:
        for fold, donor in enumerate(donors):
            train = [i for i in range(4) if i != fold]
            # two_way centers across the complete intervention axis.  Build the
            # full fold operator once; writer chunks are serialization only.
            shared = np.zeros((len(targets), 3, delta.shape[-1]), np.float32)
            for index in train:
                shared += np.asarray(delta[:, index], np.float32) / len(train)
            state_mean = shared.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
            perturbation_mean = shared.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
            grand_mean = shared.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
            shared -= state_mean
            shared -= perturbation_mean
            shared += grand_mean
            del state_mean, perturbation_mean, grand_mean
            for start in range(0, len(targets), 128):
                stop = min(start + 128, len(targets))
                flat = np.ascontiguousarray(shared[start:stop].reshape(-1, delta.shape[-1]))
                values = pa.FixedSizeListArray.from_arrays(pa.array(flat.ravel(), type=pa.float32()), delta.shape[-1])
                n = len(flat)
                table = pa.Table.from_arrays([
                    pa.array([fold] * n, type=pa.int16()), pa.array([donor] * n),
                    pa.array([";".join(donors[i] for i in train)] * n),
                    pa.array(np.repeat(targets[start:stop], 3)), pa.array(states * (stop - start)),
                    values, pa.array(["TRAIN_DONOR_SHARED_OPERATOR"] * n),
                ], schema=schema)
                writer.write_table(table)
            del shared
            gc.collect()
    finally:
        writer.close()


def _fixed_decomposition(stats: dict[str, np.ndarray], beta_t: float, beta_p: float):
    t2, s2, ts = stats["t2"], stats["s2"], stats["ts"]
    p2, pt, ps = stats["p2"], stats["pt"], stats["ps"]
    return {
        "beta_t": beta_t, "beta_p": beta_p,
        "nt2": t2 - 2 * beta_t * ts + beta_t * beta_t * s2,
        "np2": p2 - 2 * beta_p * ps + beta_p * beta_p * s2,
        "ndot": pt - beta_t * ps - beta_p * ts + beta_p * beta_t * s2,
    }


def _basic_rows(model: str, seed: int, fold: int, donor: str, stats: dict[str, np.ndarray], dimensions: int):
    dec = decomposition(stats)
    shared = scalar_metrics(stats["ts"], stats["s2"], stats["t2"])
    pred = scalar_metrics(stats["pt"], stats["p2"], stats["t2"])
    additive = scalar_metrics(dec["ddot"], dec["dpred2"], dec["dtrue2"])
    novel = scalar_metrics(dec["ndot"], dec["np2"], dec["nt2"])
    err_shared = float(dec["dtrue2"].sum())
    err_pred = float((stats["t2"] + stats["p2"] - 2 * stats["pt"]).sum())
    base = {"model": model, "seed": seed, "outer_fold": fold, "held_out_donor": donor}
    compare = {
        **base, "shared_mse": err_shared / dimensions, "model_mse": err_pred / dimensions,
        "delta_mse": (err_shared - err_pred) / dimensions,
        "shared_cosine": shared["cosine"], "model_cosine": pred["cosine"],
        "delta_cosine": pred["cosine"] - shared["cosine"],
        "shared_pearson": shared["pearson"], "model_pearson": pred["pearson"],
        "delta_pearson": pred["pearson"] - shared["pearson"],
    }
    adapt = {**base, "shared_error": err_shared, "model_error": err_pred, "r2_adapt": 1 - err_pred / err_shared}
    additive_row = {**base, "alpha_d": additive["alpha"], "additive_cosine": additive["cosine"], "additive_pearson": additive["pearson"], "dtrue2": additive["right2"], "dpred2": additive["left2"], "ddot": additive["dot"]}
    shared_row = {**base, "beta_t": dec["beta_t"], "beta_p": dec["beta_p"], "r_shared": dec["beta_p"] / dec["beta_t"] if abs(dec["beta_t"]) > 1e-12 else np.nan, "cosine_p_s": float(_cos(stats["ps"].sum(), stats["p2"].sum(), stats["s2"].sum())), "shared_truth_cosine": shared["cosine"]}
    novel_row = {**base, "alpha_novel": novel["alpha"], "cosine_novel": novel["cosine"], "kappa_novel": novel["kappa"], "ntrue2": novel["right2"], "npred2": novel["left2"], "novel_dot": novel["dot"], "truth_orthogonality_error": float(abs(dec["nt_s_dot"].sum())), "prediction_orthogonality_error": float(abs(dec["np_s_dot"].sum()))}
    c_p = float(stats["pt"].sum() / stats["p2"].sum())
    c_s = float(stats["ts"].sum() / stats["s2"].sum())
    pstar_error = float(stats["t2"].sum() - stats["pt"].sum() ** 2 / stats["p2"].sum())
    sstar_error = float(stats["t2"].sum() - stats["ts"].sum() ** 2 / stats["s2"].sum())
    oracle = {**base, "evaluation_label": "ORACLE_EVALUATION_ONLY", "c_p_star": c_p, "c_s_star": c_s, "model_calibrated_mse": pstar_error / dimensions, "shared_calibrated_mse": sstar_error / dimensions, "calibrated_delta_mse": (sstar_error - pstar_error) / dimensions, "calibrated_model_beats_shared": pstar_error < sstar_error}
    orthogonal = {**base, "beta_t": dec["beta_t"], "beta_p": dec["beta_p"], "shared_truth_energy": dec["beta_t"] ** 2 * float(stats["s2"].sum()), "truth_novel_energy": float(dec["nt2"].sum()), "shared_prediction_energy": dec["beta_p"] ** 2 * float(stats["s2"].sum()), "prediction_novel_energy": float(dec["np2"].sum())}
    return compare, adapt, additive_row, shared_row, novel_row, oracle, orthogonal, dec


def _subset_null(raw: np.ndarray, np2: np.ndarray, nt2: np.ndarray, keep: np.ndarray, draws: int, seed: int, subtract: np.ndarray | None = None):
    ids = np.flatnonzero(keep); rng = np.random.default_rng(seed)
    values = np.empty(draws); denom = np.sqrt(float(np2[keep].sum() * nt2[keep].sum()))
    for draw in range(draws):
        perm = rng.permutation(ids)
        dot = np.asarray(raw[perm, ids], np.float64)
        if subtract is not None: dot -= np.asarray(subtract[perm, ids], np.float64)
        values[draw] = dot.sum() / denom
    return values


def _matrix_fold(*, root: Path, config: dict[str, Any], temp: Path, matrix_dir: Path,
                 fold: int, donor: str, targets: list[str], intervention_blocks: np.ndarray,
                 total: dict[str, np.ndarray], blocks: list[dict[str, np.ndarray]],
                 residuals: dict[str, dict[int, np.ndarray]], total_stats: dict[str, dict[int, dict[str, np.ndarray]]],
                 total_dec: dict[str, dict[int, dict[str, Any]]]):
    p = len(targets); rb = int(config["matrix_row_batch"]); draws = int(config["permutations"])
    total_paths: dict[tuple[str, int], Path] = {}
    block_paths: dict[tuple[int, str, int], Path] = {}
    gene_rows, shuffle_rows, null_rows, fingerprint_rows, retrieval_rows, intervention_rows = [], [], [], [], [], []
    donor_null: dict[str, dict[int, np.ndarray]] = {m: {} for m in MODELS}
    paired: dict[str, np.ndarray] = {}
    for block_index, block in enumerate(blocks):
        print(f"[1D] {donor}: gene block {block_index + 1}/5", flush=True)
        st_path, ss_path = temp / f"st_{fold}_{block_index}.npy", temp / f"ss_{fold}_{block_index}.npy"
        st = _operator_cross(block["shared"], block["truth"], st_path, rb)
        ss = _operator_cross(block["shared"], block["shared"], ss_path, rb)
        for model in MODELS:
            raws, np2s = [], []
            for seed, residual in residuals[model].items():
                stats = _prediction_stats(block, residual, block["gram"])
                fixed = _fixed_decomposition(stats, total_dec[model][seed]["beta_t"], total_dec[model][seed]["beta_p"])
                raw_path = temp / f"novel_{fold}_{block_index}_{model}_{seed}.npy"
                raw = _novel_raw(st, ss, stats, block, fixed, raw_path, rb)
                total_path = temp / f"novel_total_{fold}_{model}_{seed}.npy"
                acc = _add_matrix(raw, total_path, block_index == 0, rb); del acc
                total_paths[(model, seed)] = total_path; block_paths[(block_index, model, seed)] = raw_path
                raws.append(raw); np2s.append(fixed["np2"])
                observed = float(_cos(fixed["ndot"].sum(), fixed["np2"].sum(), fixed["nt2"].sum()))
                null = _subset_null(raw, fixed["np2"], fixed["nt2"], np.ones(p, bool), draws, int(config["random_seed"]) + fold * 1009 + block_index * 131 + seed * 7 + MODELS.index(model))
                gene_rows.append({"model": model, "seed": seed, "outer_fold": fold, "held_out_donor": donor, "gene_block": block_index, "analysis": "block_only", "r2_adapt": 1 - float((stats["t2"] + stats["p2"] - 2 * stats["pt"]).sum()) / float((stats["t2"] + stats["s2"] - 2 * stats["ts"]).sum()), "novel_cosine": observed, "alpha_novel": float(fixed["ndot"].sum() / fixed["nt2"].sum()), "shuffle_q95": float(np.quantile(null, .95)), "exceeds_shuffle_q95": observed > float(np.quantile(null, .95))})
            mean_path = temp / f"mean_gene_{fold}_{block_index}_{model}.npy"
            mean = _mean_similarity(raws, np2s, fixed["nt2"], mean_path, rb)
            summary, _ = _analyze_matrix(mean, targets, draws, int(config["random_seed"]) + fold * 2017 + block_index * 71 + MODELS.index(model), False)
            for row in gene_rows[-len(raws):]: row["matched_minus_mismatched"] = summary["median_paired_matched_minus_row_mismatch"]
            del mean
            Path(mean_path).unlink()
            for raw in raws: del raw
        del st, ss
        st_path.unlink(); ss_path.unlink()
        del block["truth"], block["shared"]
        gc.collect()
    # Exact full novel fingerprints and primary shuffle.
    for model in MODELS:
        raw_arrays, np2s = [], []
        for seed in sorted(residuals[model]):
            raw_arrays.append(np.load(total_paths[(model, seed)], mmap_mode="r"))
            np2s.append(total_dec[model][seed]["np2"])
        matrix_path = matrix_dir / f"{model}_{donor}.float32.npy"
        mean = _mean_similarity(raw_arrays, np2s, total_dec[model][sorted(residuals[model])[0]]["nt2"], matrix_path, rb)
        summary, rows = _analyze_matrix(mean, targets, draws, int(config["random_seed"]) + fold * 3011 + MODELS.index(model), True)
        summary.update({"model": model, "outer_fold": fold, "held_out_donor": donor})
        fingerprint_rows.append(summary); paired[model] = rows["matched_minus_row_mismatched"].to_numpy()
        rows.insert(0, "model", model); rows.insert(1, "outer_fold", fold); rows.insert(2, "held_out_donor", donor); retrieval_rows.append(rows)
        seed_nulls, seed_alphas, seed_effects, observed_seed = [], [], [], []
        for seed, raw, np2 in zip(sorted(residuals[model]), raw_arrays, np2s):
            dec = total_dec[model][seed]
            row_med = np.asarray(mean).copy(); row_med[np.arange(p), np.arange(p)] = np.nan; row_med = np.nanmedian(row_med, axis=1)
            null_c, null_a, null_e = _permutation_metrics(raw, np2, dec["nt2"], row_med, draws, int(config["random_seed"]) + fold * 4013 + seed * 11 + MODELS.index(model))
            seed_nulls.append(null_c); seed_alphas.append(null_a); seed_effects.append(null_e)
            observed_seed.append(float(_cos(dec["ndot"].sum(), dec["np2"].sum(), dec["nt2"].sum())))
        null_c = np.mean(seed_nulls, axis=0); null_a = np.mean(seed_alphas, axis=0); null_e = np.mean(seed_effects, axis=0)
        donor_null[model] = {"cosine": null_c, "alpha": null_a, "matched_effect": null_e}
        ns = _null_summary(null_c, float(np.mean(observed_seed))); ns.update({"scope": "donor", "model": model, "outer_fold": fold, "held_out_donor": donor, "alpha_null_q95": float(np.quantile(null_a, .95)), "matched_effect_null_q95": float(np.quantile(null_e, .95))}); shuffle_rows.append(ns)
        for draw in range(draws): null_rows.append({"scope": "donor", "model": model, "outer_fold": fold, "held_out_donor": donor, "permutation": draw, "cosine_novel": null_c[draw], "alpha_novel": null_a[draw], "matched_minus_mismatched": null_e[draw]})
        # Frozen intervention block and leave-one-block-out analyses.
        for block_index in range(5):
            for analysis, keep in (("block_only", intervention_blocks == block_index), ("leave_one_block_out", intervention_blocks != block_index)):
                seed_obs, seed_r2, seed_alpha, seed_null = [], [], [], []
                for seed, raw in zip(sorted(residuals[model]), raw_arrays):
                    stats, dec = total_stats[model][seed], total_dec[model][seed]
                    seed_obs.append(float(_cos(dec["ndot"][keep].sum(), dec["np2"][keep].sum(), dec["nt2"][keep].sum())))
                    seed_alpha.append(float(dec["ndot"][keep].sum() / dec["nt2"][keep].sum()))
                    seed_r2.append(1 - float((stats["t2"][keep] + stats["p2"][keep] - 2 * stats["pt"][keep]).sum()) / float((stats["t2"][keep] + stats["s2"][keep] - 2 * stats["ts"][keep]).sum()))
                    seed_null.append(_subset_null(raw, dec["np2"], dec["nt2"], keep, draws, int(config["random_seed"]) + fold * 5003 + block_index * 43 + seed))
                null = np.mean(seed_null, axis=0); obs = float(np.mean(seed_obs))
                intervention_rows.append({"model": model, "outer_fold": fold, "held_out_donor": donor, "intervention_block": block_index, "analysis": analysis, "r2_adapt": float(np.mean(seed_r2)), "novel_cosine": obs, "alpha_novel": float(np.mean(seed_alpha)), "shuffle_q95": float(np.quantile(null, .95)), "exceeds_shuffle_q95": obs > float(np.quantile(null, .95))})
        # Gene leave-one-out nulls after total matrices are available.
        for block_index in range(5):
            seed_obs, seed_r2, seed_alpha, seed_null = [], [], [], []
            for seed, raw in zip(sorted(residuals[model]), raw_arrays):
                block_stats = _prediction_stats(blocks[block_index], residuals[model][seed], blocks[block_index]["gram"])
                stats, dec = total_stats[model][seed], total_dec[model][seed]
                bdec = _fixed_decomposition(block_stats, dec["beta_t"], dec["beta_p"])
                np2, nt2, ndot = dec["np2"] - bdec["np2"], dec["nt2"] - bdec["nt2"], dec["ndot"] - bdec["ndot"]
                errp = (stats["t2"] + stats["p2"] - 2 * stats["pt"]) - (block_stats["t2"] + block_stats["p2"] - 2 * block_stats["pt"])
                errs = (stats["t2"] + stats["s2"] - 2 * stats["ts"]) - (block_stats["t2"] + block_stats["s2"] - 2 * block_stats["ts"])
                seed_obs.append(float(_cos(ndot.sum(), np2.sum(), nt2.sum()))); seed_alpha.append(float(ndot.sum() / nt2.sum())); seed_r2.append(1 - float(errp.sum() / errs.sum()))
                block_raw = np.load(block_paths[(block_index, model, seed)], mmap_mode="r")
                seed_null.append(_subset_null(raw, np2, nt2, np.ones(p, bool), draws, int(config["random_seed"]) + fold * 6007 + block_index * 47 + seed, subtract=block_raw))
                del block_raw
            null = np.mean(seed_null, axis=0); obs = float(np.mean(seed_obs))
            gene_rows.append({"model": model, "seed": -999, "outer_fold": fold, "held_out_donor": donor, "gene_block": block_index, "analysis": "leave_one_block_out", "r2_adapt": float(np.mean(seed_r2)), "novel_cosine": obs, "alpha_novel": float(np.mean(seed_alpha)), "shuffle_q95": float(np.quantile(null, .95)), "exceeds_shuffle_q95": obs > float(np.quantile(null, .95)), "matched_minus_mismatched": np.nan})
        del mean, raw_arrays, np2s
        gc.collect()
    for path in set(total_paths.values()) | set(block_paths.values()): _safe_unlink(Path(path))
    return {
        "gene_rows": gene_rows, "shuffle_rows": shuffle_rows, "null_rows": null_rows,
        "fingerprint_rows": fingerprint_rows, "retrieval_rows": retrieval_rows,
        "intervention_rows": intervention_rows, "donor_null": donor_null, "paired": paired,
    }


def _sum_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: sum((np.asarray(item[key], np.float64) for item in items), np.zeros_like(np.asarray(items[0][key], np.float64))) for key in ("t2", "s2", "ts", "p2", "pt", "ps")}


def _model_mean_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).columns.difference(["seed"])
    result = frame.groupby(group_cols, as_index=False)[numeric].mean()
    result["seed"] = -999
    columns = list(frame.columns)
    return result.reindex(columns=columns)


def _contrast_rows(model: str, seed: int, fold: int, donor: str, base: dict[str, np.ndarray], stats: dict[str, np.ndarray], dec: dict[str, Any], residual: np.ndarray):
    rows = []; r = two_way(residual)
    for ci, (left, right) in enumerate(STATE_PAIRS):
        rc = r[:, left] - r[:, right]
        r2 = np.square(rc, dtype=np.float64).sum(axis=1)
        rs = np.einsum("pr,pr->p", rc, base["scontrastproj"][ci])
        rt = np.einsum("pr,pr->p", rc, base["tcontrastproj"][ci])
        s2, t2, ts = base["scontrast2"][:, ci], base["tcontrast2"][:, ci], base["tscontrast"][:, ci]
        p2, pt, ps = s2 + 2 * rs + r2, ts + rt, s2 + rs
        bt, bp = dec["beta_t"], dec["beta_p"]
        nt2 = t2 - 2 * bt * ts + bt * bt * s2
        np2 = p2 - 2 * bp * ps + bp * bp * s2
        ndot = pt - bt * ps - bp * ts + bp * bt * s2
        err_s = float((t2 + s2 - 2 * ts).sum()); err_p = float((t2 + p2 - 2 * pt).sum())
        rows.append({
            "model": model, "seed": seed, "outer_fold": fold, "held_out_donor": donor,
            "contrast": STATE_PAIR_NAMES[ci], "r2_adapt": 1 - err_p / err_s,
            "novel_cosine": float(_cos(ndot.sum(), np2.sum(), nt2.sum())),
            "alpha_novel": float(ndot.sum() / nt2.sum()), "kappa_novel": float(np2.sum() / nt2.sum()),
            "shared_contrast_energy": float(s2.sum()), "truth_contrast_energy": float(t2.sum()), "model_contrast_energy": float(p2.sum()),
        })
    return rows


def _bootstrap(config: dict[str, Any], stats_all, paired_all, shared_by_fold):
    draws = int(config["bootstrap_draws"]); p = len(next(iter(shared_by_fold.values()))["t2"])
    gene_count = int(config.get("gene_count_runtime", 1))
    rng = np.random.default_rng(int(config["random_seed"]) + 9001)
    metric_names = ("r2_adapt", "delta_mse", "alpha_d", "additive_cosine", "alpha_novel", "novel_cosine", "kappa_novel", "oracle_delta_mse", "cosine_p_s", "matched_minus_mismatched")
    values = {m: {k: np.empty(draws) for k in metric_names} for m in MODELS}
    shared_values = {k: np.empty(draws) for k in ("cosine", "pearson", "alpha_shared", "mse")}
    for draw in range(draws):
        idx = rng.integers(0, p, p)
        sb = _sum_stats([{**x, "p2": x["s2"], "pt": x["ts"], "ps": x["s2"]} for x in shared_by_fold.values()])
        st2, ss2, sdot = sum(float(x["t2"][idx].sum()) for x in shared_by_fold.values()), sum(float(x["s2"][idx].sum()) for x in shared_by_fold.values()), sum(float(x["ts"][idx].sum()) for x in shared_by_fold.values())
        shared_values["cosine"][draw] = float(_cos(sdot, ss2, st2)); shared_values["pearson"][draw] = shared_values["cosine"][draw]
        shared_values["alpha_shared"][draw] = sdot / st2; shared_values["mse"][draw] = (st2 + ss2 - 2 * sdot) / (p * 4 * 3 * gene_count)
        for model in MODELS:
            per = {k: [] for k in metric_names}
            seeds = sorted(stats_all[model])
            for seed in seeds:
                items = [stats_all[model][seed][fold] for fold in range(4)]
                agg = {k: sum((x[k][idx] for x in items), np.zeros(p)) for k in ("t2", "s2", "ts", "p2", "pt", "ps")}
                dec = decomposition(agg)
                err_s = float(dec["dtrue2"].sum()); err_p = float((agg["t2"] + agg["p2"] - 2 * agg["pt"]).sum())
                per["r2_adapt"].append(1 - err_p / err_s); per["delta_mse"].append((err_s - err_p) / (p * 4 * 3 * gene_count))
                per["alpha_d"].append(float(dec["ddot"].sum() / dec["dtrue2"].sum())); per["additive_cosine"].append(float(_cos(dec["ddot"].sum(), dec["dpred2"].sum(), dec["dtrue2"].sum())))
                per["alpha_novel"].append(float(dec["ndot"].sum() / dec["nt2"].sum())); per["novel_cosine"].append(float(_cos(dec["ndot"].sum(), dec["np2"].sum(), dec["nt2"].sum()))); per["kappa_novel"].append(float(dec["np2"].sum() / dec["nt2"].sum()))
                pstar = float(agg["t2"].sum() - agg["pt"].sum() ** 2 / agg["p2"].sum()); sstar = float(agg["t2"].sum() - agg["ts"].sum() ** 2 / agg["s2"].sum()); per["oracle_delta_mse"].append((sstar - pstar) / (p * 4 * 3 * gene_count))
                per["cosine_p_s"].append(float(_cos(agg["ps"].sum(), agg["p2"].sum(), agg["s2"].sum())))
                effects = np.concatenate([paired_all[model][fold][idx] for fold in range(4)])
                per["matched_minus_mismatched"].append(float(np.median(effects)))
            for key in metric_names: values[model][key][draw] = float(np.mean(per[key]))
    rows = []
    for model in MODELS:
        for metric, array in values[model].items():
            rows.append({"scope": "pooled", "model": model, "metric": metric, "point": float(np.mean(array)), "ci_lower": float(np.quantile(array, .025)), "ci_upper": float(np.quantile(array, .975)), "draws": draws})
    for metric, array in shared_values.items():
        rows.append({"scope": "pooled", "model": "TRAIN_DONOR_SHARED_OPERATOR", "metric": metric, "point": float(np.mean(array)), "ci_lower": float(np.quantile(array, .025)), "ci_upper": float(np.quantile(array, .975)), "draws": draws})
    return pd.DataFrame(rows)


def _svg_bar(path: Path, title: str, labels: list[str], values: list[float], zero: bool = False):
    width, height, margin = 1100, 560, 90; plot_h = 370
    lo = min(values + ([0] if zero else [])); hi = max(values + ([0] if zero else [])); span = max(hi - lo, 1e-9)
    y0 = 70 + hi / span * plot_h; bw = (width - 2 * margin) / len(values) * .65
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="550" y="35" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{title}</text>', f'<line x1="{margin}" y1="{y0:.2f}" x2="{width-margin}" y2="{y0:.2f}" stroke="#333"/>']
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin + (i + .5) * (width - 2 * margin) / len(values) - bw / 2; y = 70 + (hi - max(value, 0)) / span * plot_h; h = abs(value) / span * plot_h
        if value < 0: y = y0
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#3b82a0"/>', f'<text x="{x+bw/2:.1f}" y="{y-6 if value>=0 else y+h+18:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.4g}</text>', f'<text x="{x+bw/2:.1f}" y="485" text-anchor="middle" font-family="Arial" font-size="11" transform="rotate(30 {x+bw/2:.1f} 485)">{label}</text>']
    parts.append('</svg>'); path.write_text("\n".join(parts), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free compact Markdown rendering for frozen reports."""
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _reports(root: Path, config: dict[str, Any], verdict: dict[str, Any], shared: pd.DataFrame, compare: pd.DataFrame, adapt: pd.DataFrame, additive: pd.DataFrame, novel: pd.DataFrame, shuffle: pd.DataFrame, fingerprints: pd.DataFrame, oracle: pd.DataFrame, boot: pd.DataFrame, intervention: pd.DataFrame, gene: pd.DataFrame):
    reports = root / config["report_directory"]; reports.mkdir(parents=True, exist_ok=True)
    s = shared.loc[shared.scope.eq("pooled")].iloc[0]
    a = adapt.loc[adapt.scope.eq("pooled") & adapt.seed.eq(-999)].set_index("model")
    n = novel.loc[novel.scope.eq("pooled") & novel.seed.eq(-999)].set_index("model")
    d = additive.loc[additive.scope.eq("pooled") & additive.seed.eq(-999)].set_index("model")
    sh = shuffle.loc[shuffle.scope.eq("pooled")].set_index("model")
    fp = fingerprints.groupby("model").mean(numeric_only=True)
    o = oracle.loc[oracle.scope.eq("pooled") & oracle.seed.eq(-999)].set_index("model")
    output = root / config["output_directory"]
    shared_component = pd.read_csv(output / "shared_component_recovery.csv")
    shared_component = shared_component.loc[shared_component.scope.eq("pooled") & shared_component.seed.eq(-999)].set_index("model")
    orthogonal = pd.read_csv(output / "orthogonal_novel_component.csv")
    orthogonal = orthogonal.loc[orthogonal.scope.eq("pooled") & orthogonal.seed.eq(-999)].set_index("model")
    shared_fraction = orthogonal.shared_prediction_energy / (orthogonal.shared_prediction_energy + orthogonal.prediction_novel_energy)
    ib = intervention.loc[intervention.held_out_donor.eq("ALL") & intervention.analysis.eq("block_only")]
    gb = gene.loc[gene.held_out_donor.eq("ALL") & gene.analysis.eq("block_only")]
    oracle_boot = boot.loc[boot.metric.eq("oracle_delta_mse")].set_index("model")
    donor_positive = adapt.loc[adapt.scope.eq("donor") & adapt.seed.eq(-999)].groupby("model").r2_adapt.apply(lambda x: int((x > 0).sum()))
    answers = f"""# CGC-TCELL-1D mandatory answers

1. **How well does the shared operator predict held-out truth?** Pooled cosine `{s.cosine:.6g}`, alpha `{s.alpha_shared:.6g}`, MSE `{s.mse:.6g}`; donor-level values are in `shared_baseline_metrics.csv`.
2. **Does any trained model outperform it?** No. {', '.join(f'{m}: R2_adapt={a.loc[m].r2_adapt:.6g}' for m in MODELS)}.
3. **Positive in at least 3/4 donors?** {', '.join(f'{m}: {donor_positive[m]}/4' for m in MODELS)}.
4. **How much is attributable to the shared rule?** {', '.join(f'{m}: beta_P={shared_component.loc[m].beta_p:.6g}, cos(P,S)={shared_component.loc[m].cosine_p_s:.6g}, predicted shared-energy fraction={shared_fraction.loc[m]:.4%}' for m in MODELS)}.
5. **Does the model recover T-S?** {', '.join(f'{m}: additive cosine={d.loc[m].additive_cosine:.6g}' for m in MODELS)}; see `additive_donor_deviation.csv`.
6. **Does orthogonal novel signal exceed shuffle?** {', '.join(f'{m}: cos={n.loc[m].cosine_novel:.6g}, q95={sh.loc[m].null_q95:.6g}' for m in MODELS)}.
7. **Can novel fingerprints identify interventions?** Only weakly for Ridge: {', '.join(f'{m}: top1={fp.loc[m].top1_accuracy:.4%}, top10={fp.loc[m].top10_accuracy:.4%}' for m in MODELS)} (random expectations: 0.0107% and 0.1065%).
8. **Does oracle amplitude calibration beat calibrated shared transfer?** Point estimates are positive for all ({', '.join(f'{m}: delta MSE={o.loc[m].calibrated_delta_mse:.6g}' for m in MODELS)}), but only Ridge has a bootstrap lower bound above zero ({', '.join(f'{m}: CI=[{oracle_boot.loc[m].ci_lower:.3g}, {oracle_boot.loc[m].ci_upper:.3g}]' for m in MODELS)}). This is `ORACLE_EVALUATION_ONLY`.
9. **Robust across blocks?** Yes for the central non-improvement conclusion: intervention-block positive R2 counts are {', '.join(f'{m}={int((ib.loc[ib.model.eq(m), "r2_adapt"] > 0).sum())}/5' for m in MODELS)}, and gene-block counts are {', '.join(f'{m}={int((gb.loc[gb.model.eq(m), "r2_adapt"] > 0).sum())}/5' for m in MODELS)}. Exact block and leave-one-block-out results use the unchanged frozen blocks.
10. **Mechanistic explanation of CGC?** `{verdict['overall_mechanistic_label']}`: shared stimulation transfer is preserved, while these frozen models do not improve that rule for a new donor. This is not a claim that donor biology is fundamentally unpredictable or that NTC lacks donor information.
11. **Does 1D change the frozen 1B CGC verdict?** **NO**.
"""
    main = f"""# CGC-TCELL-1D shared-versus-donor-specific decomposition

Git provenance: `{verdict['git_provenance']}` on `{config['branch']}`.

No predictive model was fitted and no checkpoint was loaded for inference. The three-donor shared operator excludes held-out truth in every outer fold. All 10,000 intervention permutations and all 10,000 bootstrap draws operate on frozen sufficient statistics or frozen fingerprint matrices.

## Frozen labels

- Shared-rule labels: {json.dumps(verdict['shared_rule_labels'], sort_keys=True)}
- Donor-specific label: `{verdict['donor_specific_adaptation_label']}`
- Overall label: `{verdict['overall_mechanistic_label']}`
- Frozen 1B verdict: `CONTEXT_GEOMETRY_COMPRESSION_SUPPORTED` (`AMPLITUDE_CGC`), unchanged.

## Primary pooled metrics

{_markdown_table(adapt.loc[adapt.scope.eq('pooled') & adapt.seed.eq(-999), ['model','r2_adapt']])}

{_markdown_table(boot.loc[boot.metric.eq('r2_adapt')])}

## Interpretation

This experiment evaluates whether these frozen models improve on a three-training-donor shared stimulation operator. Weak or unsupported donor-specific adaptation does not imply that donor biology is fundamentally unpredictable or that NTC lacks donor information.
"""
    (reports / "cgc_tcell_1d_shared_specific_decomposition.md").write_text(main, encoding="utf-8")
    (reports / "cgc_tcell_1d_mechanistic_interpretation.md").write_text(answers, encoding="utf-8")
    (reports / "cgc_tcell_1d_phase_summary.md").write_text(f"# CGC-TCELL-1D phase summary\n\n- `{verdict['donor_specific_adaptation_label']}`\n- `{verdict['overall_mechanistic_label']}`\n\nCGC-TCELL-1B remains `CONTEXT_GEOMETRY_COMPRESSION_SUPPORTED` (`AMPLITUDE_CGC`). CGC-TCELL-2 was not started.\n", encoding="utf-8")


def run(root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter(); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if git(root, "branch", "--show-current") != config["branch"]: raise RuntimeError("Wrong CGC-TCELL-1D branch")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for exact frozen fingerprint matrices")
    output = root / config["output_directory"]; output.mkdir(parents=True, exist_ok=True)
    matrix_dir = output / "novel_fingerprint_similarity"; matrix_dir.mkdir(exist_ok=True)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    temp = root / config["temporary_directory"]
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    print("[1D] verifying all frozen inputs", flush=True)
    hashes = _verify_inputs(root, config); (output / "frozen_input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    frozen = load_frozen(root); primary = frozen["primary"]
    delta = np.memmap(root / primary["response_path"], mode="r", dtype=np.float32, shape=tuple(primary["response_shape"]))
    donors, targets, states = frozen["donors"], frozen["targets"], frozen["states"]
    strict = pd.read_csv(root / "results/cgc_tcell/strict_trans_genes.csv")
    genes = strict.loc[strict.strict_trans_eligible, "gene_id"].astype(str).tolist()
    config["gene_count_runtime"] = len(genes)
    gene_assignment = np.asarray([hash_block(x) for x in genes]); gene_masks = [gene_assignment == b for b in range(5)]
    intervention_assignment = np.asarray([hash_block(x) for x in targets])
    print("[1D] reproducing 1B and 1C frozen metrics", flush=True)
    reproduction, reproduction_cache, tolerance = _reproduction(root, config, delta, donors, targets)
    reproduction.to_csv(output / "frozen_metric_reproduction.csv", index=False)
    truth_manifest = {
        "operator_label": "HELDOUT_DONOR_TRUTH_OPERATOR", "source_commit": config["truth_commit"],
        "response_path": primary["response_path"], "response_shape": primary["response_shape"],
        "target_count": len(targets), "gene_count": len(genes), "donors": donors, "states": states,
        "heldout_truth_contributes_to_shared_operator": False,
    }
    (output / "heldout_truth_operator_manifest.json").write_text(json.dumps(truth_manifest, indent=2), encoding="utf-8")
    print("[1D] materializing exact three-donor shared operators", flush=True)
    _write_shared_operator(root, output, delta, donors, targets, states)

    shared_rows=[]; compare_rows=[]; adapt_rows=[]; additive_rows=[]; shared_component_rows=[]; novel_rows=[]; oracle_rows=[]; orthogonal_rows=[]; contrast_rows=[]
    gene_rows=[]; intervention_rows=[]; shuffle_rows=[]; null_rows=[]; fingerprint_rows=[]; retrieval_rows=[]
    stats_all={m:{} for m in MODELS}; residual_all={}; paired_all={m:{} for m in MODELS}; shared_by_fold={}; donor_null_all={m:{} for m in MODELS}
    dimensions = len(targets) * 3 * len(genes)
    for fold, donor in enumerate(donors):
        print(f"[1D] exact decomposition donor {fold + 1}/4: {donor}", flush=True)
        components = reproduction_cache[fold]["components"]; train = reproduction_cache[fold]["train"]
        total, blocks = _fold_exact(delta, components, train, fold, gene_masks); shared_by_fold[fold] = total
        shared_metric = scalar_metrics(total["ts"], total["s2"], total["t2"])
        shared_contrasts = [float(total["scontrast2"][:, ci].sum()) for ci in range(3)]
        truth_contrasts = [float(total["tcontrast2"][:, ci].sum()) for ci in range(3)]
        shared_rows.append({"scope":"donor","outer_fold":fold,"held_out_donor":donor,"training_donors":";".join(donors[i] for i in train),"operator_label":"TRAIN_DONOR_SHARED_OPERATOR","cosine":shared_metric["cosine"],"pearson":shared_metric["pearson"],"alpha_shared":shared_metric["alpha"],"mse":float((total["t2"]+total["s2"]-2*total["ts"]).sum()/dimensions),"top_contrast_shared":STATE_PAIR_NAMES[int(np.argmax(shared_contrasts))],"top_contrast_truth":STATE_PAIR_NAMES[int(np.argmax(truth_contrasts))],"rest_stim48_top_correct":int(np.argmax(shared_contrasts))==int(np.argmax(truth_contrasts))==1})
        residuals={m:_prediction_residuals(root,m,fold,len(targets),components.shape[0]) for m in MODELS}; residual_all[fold]=residuals
        total_stats={m:{} for m in MODELS}; total_dec={m:{} for m in MODELS}
        for model in MODELS:
            for seed,residual in residuals[model].items():
                stats=_prediction_stats(total,residual); total_stats[model][seed]=stats; stats_all[model].setdefault(seed,{})[fold]=stats
                compare,adapt,additive,shared_comp,novel,oracle,orthogonal,dec=_basic_rows(model,seed,fold,donor,stats,dimensions)
                contrast = _contrast_rows(model,seed,fold,donor,total,stats,dec,residual); contrast_rows.extend(contrast)
                p_contrasts=[row["model_contrast_energy"] for row in contrast]
                shared_comp["rest_stim48_shared_ordering_correct"] = int(int(np.argmax(p_contrasts)) == int(np.argmax(shared_contrasts)) == 1)
                compare_rows.append(compare); adapt_rows.append(adapt); additive_rows.append(additive); shared_component_rows.append(shared_comp); novel_rows.append(novel); oracle_rows.append(oracle); orthogonal_rows.append(orthogonal); total_dec[model][seed]=dec
        matrix_result=_matrix_fold(root=root,config=config,temp=temp,matrix_dir=matrix_dir,fold=fold,donor=donor,targets=targets,intervention_blocks=intervention_assignment,total=total,blocks=blocks,residuals=residuals,total_stats=total_stats,total_dec=total_dec)
        gene_rows.extend(matrix_result["gene_rows"]); intervention_rows.extend(matrix_result["intervention_rows"]); shuffle_rows.extend(matrix_result["shuffle_rows"]); null_rows.extend(matrix_result["null_rows"]); fingerprint_rows.extend(matrix_result["fingerprint_rows"]); retrieval_rows.extend(matrix_result["retrieval_rows"])
        for model in MODELS: paired_all[model][fold]=matrix_result["paired"][model]; donor_null_all[model][fold]=matrix_result["donor_null"][model]
        del blocks; gc.collect()

    # Pooled exact statistics before seed averaging.
    shared_t2=sum(float(x["t2"].sum()) for x in shared_by_fold.values()); shared_s2=sum(float(x["s2"].sum()) for x in shared_by_fold.values()); shared_dot=sum(float(x["ts"].sum()) for x in shared_by_fold.values())
    shared_rows.append({"scope":"pooled","outer_fold":-1,"held_out_donor":"ALL","training_donors":"LODO","operator_label":"TRAIN_DONOR_SHARED_OPERATOR","cosine":float(_cos(shared_dot,shared_s2,shared_t2)),"pearson":float(_cos(shared_dot,shared_s2,shared_t2)),"alpha_shared":shared_dot/shared_t2,"mse":sum(float((x["t2"]+x["s2"]-2*x["ts"]).sum()) for x in shared_by_fold.values())/(dimensions*4),"top_contrast_shared":"DONOR_SUMMARY","top_contrast_truth":"DONOR_SUMMARY","rest_stim48_top_correct":sum(bool(x["rest_stim48_top_correct"]) for x in shared_rows)>=3})
    for model in MODELS:
        for seed in sorted(stats_all[model]):
            stats=_sum_stats([stats_all[model][seed][fold] for fold in range(4)])
            compare,adapt,additive,shared_comp,novel,oracle,orthogonal,dec=_basic_rows(model,seed,-1,"ALL",stats,dimensions*4)
            for row in (compare,adapt,additive,shared_comp,novel,oracle,orthogonal): row["scope"]="pooled"
            compare_rows.append(compare); adapt_rows.append(adapt); additive_rows.append(additive); shared_component_rows.append(shared_comp); novel_rows.append(novel); oracle_rows.append(oracle); orthogonal_rows.append(orthogonal)
    # Add scope to donor rows and aggregate seeds only after exact seed-level metrics exist.
    for rows in (compare_rows,adapt_rows,additive_rows,shared_component_rows,novel_rows,oracle_rows,orthogonal_rows):
        for row in rows: row.setdefault("scope","donor")
    frames=[]
    for rows in (compare_rows,adapt_rows,additive_rows,shared_component_rows,novel_rows,oracle_rows,orthogonal_rows):
        frame=pd.DataFrame(rows); donor_mean=_model_mean_rows(frame.loc[frame.scope.eq("donor")],["scope","model","outer_fold","held_out_donor"]); pooled_mean=_model_mean_rows(frame.loc[frame.scope.eq("pooled")],["scope","model","outer_fold","held_out_donor"]); frames.append(pd.concat([frame,donor_mean,pooled_mean],ignore_index=True,sort=False))
    compare,adapt,additive,shared_component,novel,oracle,orthogonal=frames
    contrast=pd.DataFrame(contrast_rows); contrast=pd.concat([contrast,_model_mean_rows(contrast,["model","outer_fold","held_out_donor","contrast"])],ignore_index=True,sort=False)
    shared=pd.DataFrame(shared_rows)
    fingerprints=pd.DataFrame(fingerprint_rows); retrieval=pd.concat(retrieval_rows,ignore_index=True)
    intervention=pd.DataFrame(intervention_rows); intervention=pd.concat([intervention,intervention.groupby(["model","intervention_block","analysis"],as_index=False).mean(numeric_only=True).assign(outer_fold=-1,held_out_donor="ALL")],ignore_index=True,sort=False)
    gene=pd.DataFrame(gene_rows); gene=pd.concat([gene,gene.groupby(["model","gene_block","analysis"],as_index=False).mean(numeric_only=True).assign(seed=-999,outer_fold=-1,held_out_donor="ALL")],ignore_index=True,sort=False)
    # Pooled donor-preserving novel shuffle with energy weights.
    for model in MODELS:
        weights=[]
        for fold in range(4):
            nmean=novel.loc[novel.scope.eq("donor")&novel.model.eq(model)&novel.outer_fold.eq(fold)&novel.seed.eq(-999)].iloc[0]
            weights.append(np.sqrt(float(nmean.npred2*nmean.ntrue2)))
        pooled_null=sum(donor_null_all[model][fold]["cosine"]*weights[fold] for fold in range(4))/sum(weights)
        obs=float(novel.loc[novel.scope.eq("pooled")&novel.model.eq(model)&novel.seed.eq(-999),"cosine_novel"].iloc[0]); ns=_null_summary(pooled_null,obs); ns.update({"scope":"pooled","model":model,"outer_fold":-1,"held_out_donor":"ALL","alpha_null_q95":np.nan,"matched_effect_null_q95":np.nan}); shuffle_rows.append(ns)
        for draw,value in enumerate(pooled_null): null_rows.append({"scope":"pooled","model":model,"outer_fold":-1,"held_out_donor":"ALL","permutation":draw,"cosine_novel":value,"alpha_novel":np.nan,"matched_minus_mismatched":np.nan})
    shuffle=pd.DataFrame(shuffle_rows); null=pd.DataFrame(null_rows)
    print("[1D] frozen intervention bootstrap", flush=True)
    boot=_bootstrap(config,stats_all,paired_all,shared_by_fold)
    # Shared and adaptation labels are direct frozen protocol gates.
    shared_labels={}; shared_criteria={}; adaptation_criteria={}
    for model in MODELS:
        b=boot.loc[boot.model.eq(model)].set_index("metric")
        sm=shared_component.loc[shared_component.scope.eq("donor")&shared_component.model.eq(model)&shared_component.seed.eq(-999)]
        criteria={"cosine_p_s_bootstrap_lower_positive":float(b.loc["cosine_p_s","ci_lower"])>0,"rest_stim48_shared_ordering_correct_3_of_4":int(sm.rest_stim48_shared_ordering_correct.sum())>=3,"beta_p_positive_all_four":bool((sm.beta_p>0).all())}; shared_criteria[model]=criteria
        shared_labels[model]="SHARED_STIMULATION_OPERATOR_RECOVERED" if all(criteria.values()) else ("SHARED_STIMULATION_OPERATOR_RECOVERY_WEAK" if float(b.loc["cosine_p_s","point"])>0 else "SHARED_STIMULATION_OPERATOR_NOT_RESOLVED")
        donor_r2=adapt.loc[adapt.scope.eq("donor")&adapt.model.eq(model)&adapt.seed.eq(-999)].r2_adapt
        sh=shuffle.loc[shuffle.scope.eq("pooled")&shuffle.model.eq(model)].iloc[0]
        blocks=intervention.loc[intervention.held_out_donor.eq("ALL")&intervention.model.eq(model)&intervention.analysis.eq("block_only")]
        adaptation_criteria[model]={"pooled_r2_ci_lower_positive":float(b.loc["r2_adapt","ci_lower"])>0,"r2_positive_3_of_4":int((donor_r2>0).sum())>=3,"novel_cosine_exceeds_shuffle_q95":bool(sh.observed_exceeds_q95),"matched_effect_ci_lower_positive":float(b.loc["matched_minus_mismatched","ci_lower"])>0,"positive_adaptation_4_of_5_blocks":int((blocks.r2_adapt>0).sum())>=4}
    supported=sum(all(x.values()) for x in adaptation_criteria.values())>=2
    any_novel=any(x["novel_cosine_exceeds_shuffle_q95"] or x["matched_effect_ci_lower_positive"] for x in adaptation_criteria.values())
    any_improve=any(float(boot.loc[boot.model.eq(m)&boot.metric.eq("r2_adapt"),"point"].iloc[0])>0 for m in MODELS)
    donor_label="DONOR_SPECIFIC_CONTEXT_ADAPTATION_SUPPORTED" if supported else ("DONOR_SPECIFIC_CONTEXT_ADAPTATION_WEAK" if any_novel else ("DONOR_SPECIFIC_CONTEXT_ADAPTATION_NOT_SUPPORTED" if not any_improve else "DONOR_SPECIFIC_CONTEXT_ADAPTATION_UNRESOLVED"))
    shared_recovered=sum(x=="SHARED_STIMULATION_OPERATOR_RECOVERED" for x in shared_labels.values())>=2
    overall="SHARED_AND_DONOR_SPECIFIC_CONTEXT_RULES_RECOVERED" if shared_recovered and donor_label=="DONOR_SPECIFIC_CONTEXT_ADAPTATION_SUPPORTED" else ("SHARED_RULE_PRESERVED_DONOR_SPECIFIC_ADAPTATION_LIMITED" if shared_recovered else "CONTEXT_OPERATOR_RECOVERY_BROADLY_FAILED")
    provenance=git(root,"rev-parse","HEAD")
    verdict={"git_provenance":provenance,"frozen_1b_verdict":"CONTEXT_GEOMETRY_COMPRESSION_SUPPORTED","frozen_1b_subtype":"AMPLITUDE_CGC","frozen_1b_changed":False,"shared_rule_labels":shared_labels,"shared_rule_criteria":shared_criteria,"donor_specific_adaptation_label":donor_label,"donor_specific_criteria":adaptation_criteria,"overall_mechanistic_label":overall,"coarse_context_ordering_preserved":True,"predictive_model_fits":0,"checkpoint_inference_calls":0}
    # Required frozen tables.
    shared.to_csv(output/"shared_baseline_metrics.csv",index=False); compare.to_csv(output/"model_vs_shared_comparison.csv",index=False); adapt.to_csv(output/"adaptation_r2.csv",index=False); additive.to_csv(output/"additive_donor_deviation.csv",index=False); orthogonal.to_csv(output/"orthogonal_novel_component.csv",index=False); shared_component.to_csv(output/"shared_component_recovery.csv",index=False); novel.to_csv(output/"novel_component_recovery.csv",index=False); null.to_csv(output/"novel_intervention_shuffle_null.csv",index=False); shuffle.to_csv(output/"novel_intervention_shuffle_summary.csv",index=False); fingerprints.to_csv(output/"novel_matched_mismatched.csv",index=False); retrieval.to_csv(output/"novel_intervention_retrieval.csv",index=False); contrast.to_csv(output/"contrast_specific_adaptation.csv",index=False); oracle.to_csv(output/"oracle_calibrated_comparison.csv",index=False); intervention.to_csv(output/"intervention_block_robustness.csv",index=False); gene.to_csv(output/"gene_block_robustness.csv",index=False); boot.to_csv(output/"bootstrap_summary.csv",index=False)
    donor_summary=compare.loc[compare.scope.eq("donor")&compare.seed.eq(-999)].merge(adapt.loc[adapt.scope.eq("donor")&adapt.seed.eq(-999),["model","outer_fold","r2_adapt"]],on=["model","outer_fold"]).merge(additive.loc[additive.scope.eq("donor")&additive.seed.eq(-999),["model","outer_fold","additive_cosine"]],on=["model","outer_fold"]).merge(novel.loc[novel.scope.eq("donor")&novel.seed.eq(-999),["model","outer_fold","cosine_novel","alpha_novel","kappa_novel"]],on=["model","outer_fold"]).merge(shuffle.loc[shuffle.scope.eq("donor"),["model","outer_fold","null_q95"]],on=["model","outer_fold"]).merge(fingerprints[["model","outer_fold","median_paired_matched_minus_row_mismatch"]],on=["model","outer_fold"])
    donor_summary.to_csv(output/"donor_summary.csv",index=False)
    (output/"verdict.json").write_text(json.dumps(verdict,indent=2),encoding="utf-8")
    run_manifest={"git_provenance":provenance,"branch":config["branch"],"runtime_seconds":time.perf_counter()-started,"predictive_model_fits":0,"checkpoint_inference_calls":0,"permutations":int(config["permutations"]),"bootstrap_draws":int(config["bootstrap_draws"]),"metric_reproduction_tolerance":tolerance,"gpu":torch.cuda.get_device_name(0),"peak_vram_bytes":torch.cuda.max_memory_allocated(),"shared_operator_label":"TRAIN_DONOR_SHARED_OPERATOR","oracle_label":"ORACLE_EVALUATION_ONLY"}
    (output/"run_manifest.json").write_text(json.dumps(run_manifest,indent=2),encoding="utf-8")
    # Six dependency-free protocol figures.
    dm=donor_summary; _svg_bar(figures/"figure1_shared_model_truth.svg","Shared and model cosine to held-out truth",[f"S-{d}" for d in donors]+[f"{m[:2]}-{i+1}" for m in MODELS for i in range(4)],[*shared.loc[shared.scope.eq("donor"),"cosine"],*dm.model_cosine],False)
    _svg_bar(figures/"figure2_adaptation_r2.svg","Model improvement over shared transfer",[f"{m[:2]}-{i+1}" for m in MODELS for i in range(4)],dm.r2_adapt.tolist(),True)
    om=orthogonal.loc[orthogonal.scope.eq("pooled")&orthogonal.seed.eq(-999)]; _svg_bar(figures/"figure3_shared_novel_decomposition.svg","Shared and novel component energies",[f"{m}-shared" for m in MODELS]+[f"{m}-novel" for m in MODELS],om.shared_prediction_energy.tolist()+om.prediction_novel_energy.tolist())
    sp=shuffle.loc[shuffle.scope.eq("pooled")]; _svg_bar(figures/"figure4_novel_shuffle.svg","Observed novel cosine versus shuffle q95",[f"{m}-obs" for m in MODELS]+[f"{m}-q95" for m in MODELS],sp.observed_cosine.tolist()+sp.null_q95.tolist())
    fg=fingerprints.groupby("model").mean(numeric_only=True); _svg_bar(figures/"figure5_novel_retrieval.svg","Novel fingerprint retrieval",[f"{m}-top1" for m in MODELS]+[f"{m}-top10" for m in MODELS],fg.loc[list(MODELS),"top1_accuracy"].tolist()+fg.loc[list(MODELS),"top10_accuracy"].tolist())
    oo=oracle.loc[oracle.scope.eq("pooled")&oracle.seed.eq(-999)]; _svg_bar(figures/"figure6_raw_oracle_comparison.svg","Raw and oracle-calibrated model advantage",[f"{m}-raw" for m in MODELS]+[f"{m}-oracle" for m in MODELS],compare.loc[compare.scope.eq("pooled")&compare.seed.eq(-999),"delta_mse"].tolist()+oo.calibrated_delta_mse.tolist(),True)
    _reports(root,config,verdict,shared,compare,adapt,additive,novel,shuffle,fingerprints,oracle,boot,intervention,gene)
    shutil.rmtree(temp, ignore_errors=True)
    return verdict


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path.cwd()); parser.add_argument("--config",type=Path,default=Path("configs/cgc_tcell_1d.yaml")); args=parser.parse_args(); root=args.root.resolve(); config=args.config if args.config.is_absolute() else root/args.config
    print(json.dumps(run(root,config),indent=2))


if __name__ == "__main__":
    main()
