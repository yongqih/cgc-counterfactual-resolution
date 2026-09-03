"""Frozen, training-free CGC-TCELL-1C operator-alignment adjudication.

Every calculation consumes CGC-TCELL-0A truth and CGC-TCELL-1B predictions.
No checkpoint is loaded and no predictive model is fitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from igc_virtual_cell.cgc_tcell.core import hash_block
from igc_virtual_cell.cgc_tcell_1b.audit import load_frozen, sha256_file
from igc_virtual_cell.cgc_tcell_1b.benchmark import STATE_PAIRS, STATE_PAIR_NAMES, two_way


MODELS = ("Ridge", "Bilinear", "MLP")
MODEL_FILES = {
    "Ridge": "ridge_predictions.parquet",
    "Bilinear": "bilinear_predictions.parquet",
    "MLP": "mlp_predictions.parquet",
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _cos(dot: float | np.ndarray, x2: float | np.ndarray, y2: float | np.ndarray):
    return np.asarray(dot) / np.sqrt(np.maximum(np.asarray(x2) * np.asarray(y2), 1e-30))


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    return value


def _prediction_residuals(root: Path, model: str, fold: int, p: int, rank: int) -> dict[int, np.ndarray]:
    path = root / "results/cgc_tcell_1b" / MODEL_FILES[model]
    table = pq.read_table(path, filters=[("outer_fold", "=", fold)])
    seeds = np.asarray(table.column("seed"))
    values = table.column("residual_response_pca_coefficients").combine_chunks()
    flat = values.values.to_numpy(zero_copy_only=False).reshape(-1, rank)
    result = {}
    for seed in sorted(np.unique(seeds).tolist()):
        selected = seeds == seed
        array = np.asarray(flat[selected], np.float32).reshape(p, 3, rank)
        result[int(seed)] = array
    return result


def _input_hashes(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if git(root, "merge-base", "--is-ancestor", config["prediction_commit"], "HEAD") != "":
        pass
    frozen_paths = [
        root / "results/cgc_tcell/normalized_response_manifest.json",
        root / "results/cgc_tcell/strict_trans_genes.csv",
        root / "results/cgc_tcell/dataset_inventory.json",
        root / "results/cgc_tcell_1b/verdict.json",
        root / "results/cgc_tcell_1b/donor_summary.csv",
        root / "results/cgc_tcell_1b/bootstrap_summary.csv",
        root / "results/cgc_tcell_1b/model_checkpoint_manifest.json",
    ]
    frozen_paths += [root / "results/cgc_tcell_1b" / name for name in MODEL_FILES.values()]
    frozen_paths += sorted((root / "results/cgc_tcell_1b/checkpoints").glob("outer_*_response_basis.npz"))
    records = []
    for path in frozen_paths:
        if not path.exists():
            raise RuntimeError(f"CGC_1C_INVALID_FROZEN_INPUT: missing {path}")
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    expected = json.loads((root / "results/cgc_tcell_1b/model_checkpoint_manifest.json").read_text())
    checkpoint_hashes_ok = all(
        (root / row["path"]).exists() and sha256_file(root / row["path"]) == row["sha256"]
        for row in expected["checkpoints"]
    )
    truth_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", config["truth_commit"], "--", "results/cgc_tcell"],
        capture_output=True,
    ).returncode == 0
    prediction_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", config["prediction_commit"], "--", "results/cgc_tcell_1b", "results/reports/cgc_tcell_1b_model_benchmark.md", "results/reports/cgc_tcell_1b_phase_summary.md", "results/reports/cgc_tcell_1b_split_audit.md"],
        capture_output=True,
    ).returncode == 0
    if not checkpoint_hashes_ok or not truth_diff or not prediction_diff:
        raise RuntimeError("CGC_1C_INVALID_FROZEN_INPUT")
    return {
        "git_provenance": git(root, "rev-parse", "HEAD"),
        "truth_commit": config["truth_commit"],
        "prediction_commit": config["prediction_commit"],
        "truth_artifacts_unchanged": truth_diff,
        "prediction_artifacts_unchanged": prediction_diff,
        "checkpoint_hashes_match_1b_manifest": checkpoint_hashes_ok,
        "predictive_training_performed": False,
        "files": records,
    }


def _operator_sufficient(delta, components, train: list[int], held: int, gene_chunk: int):
    p, _, _, genes = delta.shape
    rank = components.shape[0]
    truth2 = np.zeros(p, np.float64)
    anchor2 = np.zeros(p, np.float64)
    dot = np.zeros(p, np.float64)
    truth_proj = np.zeros((p, 3, rank), np.float32)
    anchor_proj = np.zeros_like(truth_proj)
    state_mean_max = 0.0
    for start in range(0, genes, gene_chunk):
        stop = min(start + gene_chunk, genes)
        truth = two_way(np.asarray(delta[:, held, :, start:stop], np.float32))
        anchor = two_way(
            np.asarray(delta[:, train, :, start:stop], np.float32).mean(axis=1, dtype=np.float64).astype(np.float32)
        )
        truth2 += np.square(truth, dtype=np.float64).sum(axis=(1, 2))
        anchor2 += np.square(anchor, dtype=np.float64).sum(axis=(1, 2))
        dot += np.multiply(truth, anchor, dtype=np.float64).sum(axis=(1, 2))
        vb = components[:, start:stop]
        truth_proj += np.einsum("psg,rg->psr", truth, vb, optimize=True)
        anchor_proj += np.einsum("psg,rg->psr", anchor, vb, optimize=True)
        state_mean_max = max(state_mean_max, float(np.abs(truth.mean(axis=0)).max()))
    return {
        "truth2": truth2,
        "anchor2": anchor2,
        "dot": dot,
        "truth_proj": truth_proj,
        "anchor_proj": anchor_proj,
        "generic_template_max_abs_after_two_way": state_mean_max,
    }


def _predicted_sufficient(base: dict[str, np.ndarray], residual: np.ndarray, gram: np.ndarray | None = None):
    rgamma = two_way(residual)
    pred2 = base["anchor2"].copy()
    pred2 += 2 * np.einsum("psr,psr->p", base["anchor_proj"], rgamma)
    if gram is None:
        pred2 += np.square(rgamma, dtype=np.float64).sum(axis=(1, 2))
    else:
        pred2 += np.einsum("psr,rt,pst->p", rgamma, gram, rgamma, optimize=True)
    dot = base["dot"] + np.einsum("psr,psr->p", base["truth_proj"], rgamma)
    return rgamma, pred2, dot


def _metric_row(model, seed, fold, donor, truth2, pred2, dot):
    t2, p2, dp = map(float, (truth2.sum(), pred2.sum(), dot.sum()))
    alpha = dp / t2
    cosine = float(_cos(dp, t2, p2))
    kappa = p2 / t2
    epar = alpha * alpha
    eperp = kappa - epar
    return {
        "model": model, "seed": seed, "outer_fold": fold, "held_out_donor": donor,
        "truth2": t2, "pred2": p2, "dot": dp, "kappa_c": kappa, "alpha": alpha,
        "cosine": cosine, "pearson": cosine, "e_parallel": epar, "e_perpendicular": eperp,
        "f_parallel_pred": epar / kappa, "f_perpendicular_pred": 1 - epar / kappa,
        "energy_identity_error": abs(kappa - epar - eperp),
        "cosine_square_identity_error": abs(epar / kappa - cosine * cosine),
    }


def reproduce_gate(root: Path, config: dict[str, Any], delta, donors, targets):
    prior = pd.read_csv(root / "results/cgc_tcell_1b/donor_summary.csv")
    rows, cache = [], {}
    for fold, donor in enumerate(donors):
        train = [index for index in range(4) if index != fold]
        basis = np.load(root / f"results/cgc_tcell_1b/checkpoints/outer_{fold}_response_basis.npz")
        components = np.asarray(basis["components"], np.float32)
        base = _operator_sufficient(delta, components, train, fold, int(config["gene_chunk"]))
        cache[fold] = {"components": components, "base": base, "train": train, "residuals": {}}
        for model in MODELS:
            residuals = _prediction_residuals(root, model, fold, len(targets), components.shape[0])
            cache[fold]["residuals"][model] = residuals
            for seed, residual in residuals.items():
                _, pred2, dot = _predicted_sufficient(base, residual)
                row = _metric_row(model, seed, fold, donor, base["truth2"], pred2, dot)
                expected = prior.loc[
                    prior.model.eq(model) & prior.outer_fold.eq(fold) & prior.seed.eq(seed)
                ].iloc[0]
                for metric, prior_name in (("kappa_c", "kappa_c"), ("alpha", "alpha"), ("cosine", "operator_cosine"), ("pearson", "operator_pearson")):
                    row[f"frozen_1b_{metric}"] = float(expected[prior_name])
                    row[f"absolute_error_{metric}"] = abs(row[metric] - float(expected[prior_name]))
                rows.append(row)
    donor_frame = pd.DataFrame(rows)
    frozen_bootstrap = pd.read_csv(root / "results/cgc_tcell_1b/bootstrap_summary.csv")
    pooled_rows = []
    for model in MODELS:
        seed_metrics = []
        for seed, group in donor_frame.loc[donor_frame.model.eq(model)].groupby("seed"):
            t2, p2, dp = group[["truth2", "pred2", "dot"]].sum()
            seed_metrics.append(
                {"kappa_c": p2 / t2, "alpha": dp / t2, "cosine": float(_cos(dp, t2, p2)), "pearson": float(_cos(dp, t2, p2))}
            )
        values = pd.DataFrame(seed_metrics).mean()
        row = {"scope": "pooled", "model": model, "seed": -999, "outer_fold": -1, "held_out_donor": "ALL"}
        for metric in ("kappa_c", "alpha", "cosine", "pearson"):
            expected = float(frozen_bootstrap.loc[frozen_bootstrap.model.eq(model) & frozen_bootstrap.metric.eq(metric), "point"].iloc[0])
            row[metric] = float(values[metric]); row[f"frozen_1b_{metric}"] = expected; row[f"absolute_error_{metric}"] = abs(float(values[metric]) - expected)
        pooled_rows.append(row)
    donor_frame.insert(0, "scope", "donor")
    frame = pd.concat([donor_frame, pd.DataFrame(pooled_rows)], ignore_index=True, sort=False)
    tolerance = max(1e-8, float(frame.filter(like="absolute_error").to_numpy().max()) * 1.01)
    if tolerance > 5e-7 or (frame.filter(like="absolute_error").to_numpy() > tolerance).any():
        raise RuntimeError("CGC_1C_FROZEN_METRIC_REPRODUCTION_FAILED")
    frame["accepted_tolerance"] = tolerance
    frame["reproduced"] = frame.filter(like="absolute_error").max(axis=1) <= tolerance
    return frame, cache, tolerance


def _gpu_cross(left: np.ndarray, right: np.ndarray, path: Path, row_batch: int) -> np.memmap:
    p = left.shape[0]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(p, p))
    device = torch.device("cuda")
    right_t = torch.from_numpy(np.ascontiguousarray(right.T)).to(device)
    for start in range(0, p, row_batch):
        stop = min(start + row_batch, p)
        block = torch.from_numpy(np.ascontiguousarray(left[start:stop])).to(device)
        out[start:stop] = (block @ right_t).cpu().numpy()
    out.flush()
    del right_t, block
    torch.cuda.empty_cache()
    return out


def _add_low_rank(base: np.ndarray, left: np.ndarray, right: np.ndarray, path: Path, row_batch: int) -> np.memmap:
    p = left.shape[0]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(p, p))
    device = torch.device("cuda")
    right_t = torch.from_numpy(np.ascontiguousarray(right.T)).to(device)
    for start in range(0, p, row_batch):
        stop = min(start + row_batch, p)
        block = torch.from_numpy(np.ascontiguousarray(left[start:stop])).to(device)
        out[start:stop] = np.asarray(base[start:stop], np.float32) + (block @ right_t).cpu().numpy()
    out.flush()
    del right_t, block
    torch.cuda.empty_cache()
    return out


def _mean_similarity_add(raw: np.ndarray, pred2: np.ndarray, truth2: np.ndarray, mean_path: Path, divisor: int, first: bool, row_batch: int):
    mode = "w+" if first else "r+"
    out = np.lib.format.open_memmap(mean_path, mode=mode, dtype=np.float32, shape=raw.shape)
    for start in range(0, raw.shape[0], row_batch):
        stop = min(start + row_batch, raw.shape[0])
        denominator = np.sqrt(np.maximum(pred2[start:stop, None] * truth2[None, :], 1e-30))
        contribution = np.asarray(raw[start:stop], np.float64) / denominator
        if first:
            out[start:stop] = (contribution / divisor).astype(np.float32)
        else:
            out[start:stop] += (contribution / divisor).astype(np.float32)
    out.flush()
    return out


def _permutation_dots(raw: np.ndarray, p: int, draws: int, seed: int, allowed: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    columns = np.arange(p) if allowed is None else np.asarray(allowed)
    result = np.empty(draws, np.float64)
    for draw in range(draws):
        permuted = rng.permutation(columns)
        result[draw] = np.asarray(raw[permuted, columns], np.float64).sum()
    return result


def _similarity_summary(matrix: np.ndarray, targets: list[str], *, retain_rows: bool = True):
    p = matrix.shape[0]
    diagonal = np.asarray(matrix[np.arange(p), np.arange(p)], np.float64)
    row_median = np.empty(p, np.float32)
    ranks = np.empty(p, np.int32)
    mismatch_chunks = []
    for start in range(0, p, 128):
        stop = min(start + 128, p)
        block = np.asarray(matrix[start:stop], np.float32).copy()
        block[np.arange(stop - start), np.arange(start, stop)] = np.nan
        row_median[start:stop] = np.nanmedian(block, axis=1)
        ranks[start:stop] = 1 + np.sum(block > diagonal[start:stop, None], axis=1)
        mismatch_chunks.append(block.ravel())
    mismatch = np.concatenate(mismatch_chunks)
    mismatch = mismatch[np.isfinite(mismatch)]
    mismatch_median = float(np.median(mismatch))
    paired = diagonal - row_median
    summary = {
        "median_matched_cosine": float(np.median(diagonal)),
        "median_mismatched_cosine": mismatch_median,
        "median_paired_matched_minus_row_mismatch": float(np.median(paired)),
        "fraction_matched_above_global_mismatch_median": float(np.mean(diagonal > mismatch_median)),
        "top1_accuracy": float(np.mean(ranks <= 1)),
        "top5_accuracy": float(np.mean(ranks <= 5)),
        "top10_accuracy": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_true_match_rank": float(np.median(ranks)),
        "mean_percentile_rank": float(np.mean((p - ranks + 1) / p)),
        "median_percentile_rank": float(np.median((p - ranks + 1) / p)),
        "fraction_percentile_above_0_5": float(np.mean((p - ranks + 1) / p > .5)),
        "fraction_percentile_above_0_9": float(np.mean((p - ranks + 1) / p > .9)),
    }
    rows = None
    if retain_rows:
        rows = pd.DataFrame(
            {
                "perturbation_id": targets,
                "matched_cosine": diagonal,
                "row_mismatched_median": row_median,
                "matched_minus_row_mismatched": paired,
                "true_match_rank": ranks,
                "percentile_rank": (p - ranks + 1) / p,
            }
        )
    return summary, rows, paired, ranks


def _fingerprint_permutation(matrix, row_median, draws, seed):
    p = matrix.shape[0]
    rng = np.random.default_rng(seed)
    rows = np.arange(p)
    effects = np.empty(draws, np.float64)
    top1 = np.empty(draws, np.float64)
    top10 = np.empty(draws, np.float64)
    for draw in range(draws):
        perm = rng.permutation(p)
        selected = np.asarray(matrix[rows, perm], np.float64)
        effects[draw] = np.median(selected - row_median)
        # Under a label permutation the rank is uniform over columns; use the
        # exact observed row ordering through the selected similarity.
        rank = 1 + np.sum(np.asarray(matrix) > selected[:, None], axis=1)
        top1[draw] = np.mean(rank <= 1)
        top10[draw] = np.mean(rank <= 10)
    return effects, top1, top10


def _analyze_matrix(matrix, targets, draws, seed, retain_rows=True):
    summary, rows, paired, ranks = _similarity_summary(matrix, targets, retain_rows=retain_rows)
    p = len(targets)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws)
    for draw in range(draws):
        bootstrap[draw] = np.median(paired[rng.integers(0, p, p)])
    # Permuting truth labels gives the exact identity-null matched effect.
    null = np.empty(draws)
    matrix_rows = np.arange(p)
    row_median = np.asarray(matrix).copy()
    row_median[np.arange(p), np.arange(p)] = np.nan
    row_median = np.nanmedian(row_median, axis=1)
    rng = np.random.default_rng(seed + 1)
    for draw in range(draws):
        perm = rng.permutation(p)
        null[draw] = np.median(np.asarray(matrix[matrix_rows, perm]) - row_median)
    observed = summary["median_paired_matched_minus_row_mismatch"]
    summary.update(
        {
            "matched_minus_mismatched_ci_lower": float(np.quantile(bootstrap, .025)),
            "matched_minus_mismatched_ci_upper": float(np.quantile(bootstrap, .975)),
            "matched_permutation_p": float((1 + np.sum(null >= observed)) / (draws + 1)),
            "random_top1_expectation": 1 / p,
            "random_top5_expectation": 5 / p,
            "random_top10_expectation": 10 / p,
        }
    )
    return summary, rows


def _truth_ceiling(delta, donors, gene_blocks, config):
    p, _, _, genes = delta.shape
    partitions = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    stats: dict[str, dict[str, np.ndarray]] = {}
    donor_stats: dict[int, dict[str, np.ndarray]] = {}
    for left, right in partitions:
        stats[f"{donors[left[0]]}+{donors[left[1]]}|{donors[right[0]]}+{donors[right[1]]}"] = {
            key: np.zeros(p, np.float64) for key in ("left2", "right2", "dot")
        }
    for donor in range(4):
        donor_stats[donor] = {key: np.zeros(p, np.float64) for key in ("left2", "right2", "dot")}
    block_stats = {b: {name: {key: np.zeros(p, np.float64) for key in ("left2", "right2", "dot")} for name in stats} for b in range(5)}
    chunk = int(config["gene_chunk"])
    for start in range(0, genes, chunk):
        stop = min(start + chunk, genes)
        x = np.asarray(delta[:, :, :, start:stop], np.float32)
        for (left, right), (name, item) in zip(partitions, stats.items()):
            a = two_way(x[:, left].mean(axis=1, dtype=np.float64).astype(np.float32))
            b = two_way(x[:, right].mean(axis=1, dtype=np.float64).astype(np.float32))
            item["left2"] += np.square(a, dtype=np.float64).sum(axis=(1, 2))
            item["right2"] += np.square(b, dtype=np.float64).sum(axis=(1, 2))
            item["dot"] += np.multiply(a, b, dtype=np.float64).sum(axis=(1, 2))
            for block in range(5):
                mask = gene_blocks[start:stop] == block
                if mask.any():
                    bi = block_stats[block][name]
                    bi["left2"] += np.square(a[:, :, mask], dtype=np.float64).sum(axis=(1, 2))
                    bi["right2"] += np.square(b[:, :, mask], dtype=np.float64).sum(axis=(1, 2))
                    bi["dot"] += np.multiply(a[:, :, mask], b[:, :, mask], dtype=np.float64).sum(axis=(1, 2))
        for donor in range(4):
            other = [d for d in range(4) if d != donor]
            a = two_way(x[:, donor])
            b = two_way(x[:, other].mean(axis=1, dtype=np.float64).astype(np.float32))
            item = donor_stats[donor]
            item["left2"] += np.square(a, dtype=np.float64).sum(axis=(1, 2))
            item["right2"] += np.square(b, dtype=np.float64).sum(axis=(1, 2))
            item["dot"] += np.multiply(a, b, dtype=np.float64).sum(axis=(1, 2))
    rows = []
    for name, item in stats.items():
        rows.append({"ceiling_type": "balanced_2v2", "partition": name, "cosine": float(_cos(item["dot"].sum(), item["left2"].sum(), item["right2"].sum()))})
    for donor, item in donor_stats.items():
        rows.append({"ceiling_type": "donor_vs_other3", "partition": donors[donor], "cosine": float(_cos(item["dot"].sum(), item["left2"].sum(), item["right2"].sum()))})
    return pd.DataFrame(rows), stats, donor_stats, block_stats


def _fit_shared_axes(delta, train, rank, gene_chunk, seed):
    p, _, _, genes = delta.shape
    rows = p * len(train) * 3
    q = rank + 5
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    omega = torch.randn(genes, q, generator=generator, device=device)
    y = torch.empty((rows, q), device=device)
    cursor = 0
    # Gene chunks cannot be used for right multiplication without accumulating;
    # donor interaction tensors fit comfortably one donor at a time.
    for donor in train:
        gamma = two_way(np.asarray(delta[:, donor], np.float32))
        count = gamma.shape[0] * 3
        y[cursor:cursor+count] = torch.from_numpy(gamma.reshape(count, genes)).to(device) @ omega
        cursor += count
    qmat = torch.linalg.qr(y, mode="reduced").Q
    b = torch.zeros((q, genes), device=device)
    cursor = 0
    for donor in train:
        gamma = two_way(np.asarray(delta[:, donor], np.float32))
        flat = torch.from_numpy(gamma.reshape(-1, genes)).to(device)
        count = len(flat)
        b += qmat[cursor:cursor+count].T @ flat
        cursor += count
    _, _, vh = torch.linalg.svd(b, full_matrices=False)
    axes = vh[:rank].cpu().numpy().astype(np.float32)
    del y, qmat, b, vh, omega
    torch.cuda.empty_cache()
    return axes


def _shared_axis_rows(delta, cache, donors, config):
    rows=[]
    max_rank=max(map(int,config["shared_axis_ranks"]))
    for fold, donor in enumerate(donors):
        item=cache[fold]; axes=_fit_shared_axes(delta,item["train"],max_rank,int(config["gene_chunk"]),int(config["random_seed"])+fold)
        truth_proj=np.zeros((delta.shape[0],3,max_rank),np.float32)
        anchor_proj=np.zeros_like(truth_proj)
        for start in range(0,delta.shape[-1],int(config["gene_chunk"])):
            stop=min(start+int(config["gene_chunk"]),delta.shape[-1])
            truth=two_way(np.asarray(delta[:,fold,:,start:stop],np.float32))
            anchor=two_way(np.asarray(delta[:,item["train"],:,start:stop],np.float32).mean(axis=1,dtype=np.float64).astype(np.float32))
            truth_proj += np.einsum("psg,kg->psk",truth,axes[:,start:stop],optimize=True)
            anchor_proj += np.einsum("psg,kg->psk",anchor,axes[:,start:stop],optimize=True)
        for model,residuals in item["residuals"].items():
            v=item["components"]
            rv_axes=v@axes.T
            for seed,residual in residuals.items():
                rg=two_way(residual)
                pred_proj=anchor_proj+np.einsum("psr,rk->psk",rg,rv_axes,optimize=True)
                _,pred2,dot=_predicted_sufficient(item["base"],residual)
                for k in map(int,config["shared_axis_ranks"]):
                    t2=float(item["base"]["truth2"].sum()-np.square(truth_proj[:,:,:k],dtype=np.float64).sum())
                    p2=float(pred2.sum()-np.square(pred_proj[:,:,:k],dtype=np.float64).sum())
                    dp=float(dot.sum()-np.multiply(truth_proj[:,:,:k],pred_proj[:,:,:k],dtype=np.float64).sum())
                    rows.append({"model":model,"seed":seed,"outer_fold":fold,"held_out_donor":donor,"removed_rank":k,"residual_cosine":float(_cos(dp,t2,p2)),"residual_alpha":dp/t2,"truth_residual_energy":t2,"predicted_residual_energy":p2})
    return pd.DataFrame(rows)


def _svg(path: Path, title: str, panels: list[tuple[str, float, str]], *, reference: float | None = None):
    width,height=1000,520; margin=80; plot_w=width-2*margin; plot_h=height-150
    maximum=max([abs(v) for _,v,_ in panels]+([abs(reference)] if reference is not None else [0])+[1e-6])
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{title}</text>']
    bar_w=plot_w/max(len(panels),1)*.62
    for i,(label,value,color) in enumerate(panels):
        x=margin+(i+.5)*plot_w/len(panels)-bar_w/2; h=plot_h*abs(value)/maximum; y=height-90-h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="14">{value:.5g}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-62}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
    parts.append(f'<line x1="{margin}" y1="{height-90}" x2="{width-margin}" y2="{height-90}" stroke="#222"/>')
    parts.append('</svg>')
    path.write_text("\n".join(parts),encoding="utf-8")


def _block_base(truth_full, anchor_full, components, columns):
    truth = np.ascontiguousarray(truth_full[:, :, columns], dtype=np.float32)
    anchor = np.ascontiguousarray(anchor_full[:, :, columns], dtype=np.float32)
    vb = components[:, columns]
    return truth, anchor, {
        "truth2": np.square(truth, dtype=np.float64).sum(axis=(1, 2)),
        "anchor2": np.square(anchor, dtype=np.float64).sum(axis=(1, 2)),
        "dot": np.multiply(truth, anchor, dtype=np.float64).sum(axis=(1, 2)),
        "truth_proj": np.einsum("psg,rg->psr", truth, vb, optimize=True),
        "anchor_proj": np.einsum("psg,rg->psr", anchor, vb, optimize=True),
    }, vb @ vb.T


def _operator_base_from_contrasts(paths: list[Path], output: Path, p: int, row_batch: int):
    sources = [np.load(path, mmap_mode="r") for path in paths]
    out = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(p, p))
    for start in range(0, p, row_batch):
        stop = min(start + row_batch, p)
        out[start:stop] = sum(np.asarray(source[start:stop], np.float32) for source in sources) / 3.0
    out.flush()
    return out


def _block_permutation_dots(raw, assignment, draws, seed):
    unique = sorted(np.unique(assignment).tolist())
    result = np.empty((draws, len(unique)), np.float64)
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        for column, block in enumerate(unique):
            indices = np.flatnonzero(assignment == block)
            result[draw, column] = np.asarray(raw[rng.permutation(indices), indices], np.float64).sum()
    return result


def _null_summary(values: np.ndarray, observed: float):
    mean = float(np.mean(values)); std = float(np.std(values, ddof=1))
    return {
        "observed_cosine": observed,
        "null_mean": mean,
        "null_median": float(np.median(values)),
        "null_q95": float(np.quantile(values, .95)),
        "null_q99": float(np.quantile(values, .99)),
        "empirical_p": float((1 + np.sum(values >= observed)) / (len(values) + 1)),
        "z_score": np.nan if std <= 0 else (observed - mean) / std,
        "observed_exceeds_q95": observed > float(np.quantile(values, .95)),
    }


def _model_matrix_analysis(
    *, root, config, temp, matrix_dir, donor, fold, model, targets, base_operator,
    base_contrasts, base_sufficient, components, residuals, intervention_blocks,
    truth_contrast_proj, truth_contrast2,
):
    p=len(targets); draws=int(config["permutations"]); row_batch=int(config["matrix_row_batch"])
    mean_path=matrix_dir/f"{model}_{donor}.float32.npy"
    null_by_seed={}; stats_by_seed={}; raw_paths=[]
    contrast_mean_paths=[temp/f"meanS_{fold}_{model}_contrast_{i}.npy" for i in range(3)]
    block_seed_rows=[]; block_null_by_seed=[]
    seeds=sorted(residuals)
    for seed_index,seed in enumerate(seeds):
        residual=residuals[seed]; rg,pred2,dot=_predicted_sufficient(base_sufficient,residual)
        raw_path=temp/f"raw_{fold}_{model}_{seed}.npy"
        raw=_add_low_rank(base_operator,rg.reshape(p,-1),base_sufficient["truth_proj"].reshape(p,-1),raw_path,row_batch)
        meanS=_mean_similarity_add(raw,pred2,base_sufficient["truth2"],mean_path,len(seeds),seed_index==0,row_batch)
        null_dot=_permutation_dots(raw,p,draws,int(config["random_seed"])+fold*101)
        null_by_seed[seed]=null_dot/np.sqrt(float(pred2.sum()*base_sufficient["truth2"].sum()))
        stats_by_seed[seed]={"truth2":base_sufficient["truth2"],"pred2":pred2,"dot":dot,"rgamma":rg}
        block_dots=_block_permutation_dots(raw,intervention_blocks,draws,int(config["random_seed"])+fold*101+7)
        block_null_by_seed.append((seed, block_dots))
        for block in range(5):
            for analysis,keep in (("block_only",intervention_blocks==block),("leave_one_block_out",intervention_blocks!=block)):
                t2=float(base_sufficient["truth2"][keep].sum()); p2=float(pred2[keep].sum()); dp=float(dot[keep].sum())
                block_seed_rows.append({"model":model,"seed":seed,"outer_fold":fold,"held_out_donor":donor,"block":block,"analysis":analysis,"truth2":t2,"pred2":p2,"dot":dp,"observed_cosine":float(_cos(dp,t2,p2)),"alpha":dp/t2,"kappa_c":p2/t2,"f_parallel_pred":float(_cos(dp,t2,p2))**2})
        for ci,(left,right) in enumerate(STATE_PAIRS):
            rc=rg[:,left]-rg[:,right]
            rawc_path=temp/f"raw_{fold}_{model}_{seed}_contrast_{ci}.npy"
            rawc=_add_low_rank(base_contrasts[ci],rc,truth_contrast_proj[ci],rawc_path,row_batch)
            anchor_c2=base_sufficient["anchor_contrast2"][:,ci]
            anchor_cproj=base_sufficient["anchor_contrast_proj"][:,ci]
            gram=np.eye(rc.shape[1],dtype=np.float32)
            pred_c2=anchor_c2+2*np.einsum("pr,pr->p",anchor_cproj,rc)+np.einsum("pr,rs,ps->p",rc,gram,rc,optimize=True)
            cm=_mean_similarity_add(rawc,pred_c2,truth_contrast2[:,ci],contrast_mean_paths[ci],len(seeds),seed_index==0,row_batch)
            del rawc,cm
            Path(rawc_path).unlink()
        del raw,meanS
        raw_paths.append(raw_path)
    meanS=np.load(mean_path,mmap_mode="r")
    fp_summary,fp_rows=_analyze_matrix(meanS,targets,draws,int(config["random_seed"])+fold*1009+MODELS.index(model)*37,True)
    fp_summary.update({"model":model,"held_out_donor":donor,"outer_fold":fold})
    fp_rows.insert(0,"model",model); fp_rows.insert(1,"held_out_donor",donor); fp_rows.insert(2,"outer_fold",fold)
    donor_null=np.mean(np.stack(list(null_by_seed.values())),axis=0)
    donor_observed=float(np.mean([float(_cos(x["dot"].sum(),x["truth2"].sum(),x["pred2"].sum())) for x in stats_by_seed.values()]))
    shuffle=_null_summary(donor_null,donor_observed); shuffle.update({"scope":"donor","model":model,"held_out_donor":donor,"outer_fold":fold})
    contrast_rows=[]
    for ci,path in enumerate(contrast_mean_paths):
        matrix=np.load(path,mmap_mode="r")
        summary,_=_analyze_matrix(matrix,targets,draws,int(config["random_seed"])+fold*2027+ci*71+MODELS.index(model),False)
        summary.update({"model":model,"held_out_donor":donor,"outer_fold":fold,"contrast":STATE_PAIR_NAMES[ci]})
        contrast_rows.append(summary)
        del matrix
        Path(path).unlink()
    # Reuse the same within-block permutations across block-only and LOB summaries.
    block_frame=pd.DataFrame(block_seed_rows)
    intervention_rows=[]
    for block in range(5):
        for analysis in ("block_only","leave_one_block_out"):
            frame=block_frame.loc[block_frame.block.eq(block)&block_frame.analysis.eq(analysis)]
            observed=float(frame.observed_cosine.mean())
            seed_null_cos=[]
            for seed, block_dots in block_null_by_seed:
                seed_frame=frame.loc[frame.seed.eq(seed)].iloc[0]
                dot_null=block_dots[:,block] if analysis=="block_only" else block_dots.sum(axis=1)-block_dots[:,block]
                seed_null_cos.append(dot_null/np.sqrt(float(seed_frame.truth2*seed_frame.pred2)))
            ns=_null_summary(np.mean(np.stack(seed_null_cos),axis=0),observed)
            ns.update({"model":model,"held_out_donor":donor,"outer_fold":fold,"block":block,"analysis":analysis,"alpha":float(frame.alpha.mean()),"kappa_c":float(frame.kappa_c.mean()),"f_parallel_pred":float(frame.f_parallel_pred.mean())})
            intervention_rows.append(ns)
    for path in raw_paths: Path(path).unlink()
    return {
        "fingerprint_summary":fp_summary,"fingerprint_rows":fp_rows,"shuffle_summary":shuffle,
        "shuffle_null":donor_null,"shuffle_null_by_seed":null_by_seed,"stats_by_seed":stats_by_seed,"contrast_rows":contrast_rows,
        "intervention_rows":intervention_rows,
    }


def _full_matrix_stage(root,config,delta,cache,donors,targets,gene_blocks,intervention_blocks,ceiling,ceiling_stats,donor_ceiling_stats,block_ceiling_stats,temp,matrix_dir,output):
    p=len(targets); draws=int(config["permutations"]); rb=int(config["matrix_row_batch"])
    fingerprint_summaries=[]; fingerprint_rows=[]; shuffle_summaries=[]; shuffle_null_rows=[]
    contrast_rows=[]; gene_rows=[]; intervention_rows=[]; sufficient={m:{} for m in MODELS}; paired={m:{} for m in MODELS}
    pooled_null_parts={m:{} for m in MODELS}
    for fold,donor in enumerate(donors):
        print(f"[1C] full-matrix donor {fold + 1}/4: {donor}", flush=True)
        item=cache[fold]; components=item["components"]; train=item["train"]
        truth_full=two_way(np.asarray(delta[:,fold],np.float32))
        anchor_full=two_way(np.asarray(delta[:,train],np.float32).mean(axis=1,dtype=np.float64).astype(np.float32))
        contrast_total_paths=[temp/f"base_total_{fold}_contrast_{ci}.npy" for ci in range(3)]
        contrast_totals=[np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(p,p)) for path in contrast_total_paths]
        for matrix in contrast_totals: matrix[:]=0
        # Enrich the reproduction sufficient statistics with exact contrast terms.
        base=item["base"]
        base["anchor_contrast2"]=np.zeros((p,3),np.float64); base["anchor_contrast_proj"]=np.zeros((p,3,components.shape[0]),np.float32)
        truth_contrast2=np.zeros((p,3),np.float64); truth_contrast_proj=np.zeros((3,p,components.shape[0]),np.float32)
        for gene_block in range(5):
            print(f"[1C] donor {donor}: gene block {gene_block + 1}/5", flush=True)
            columns=gene_blocks==gene_block
            truth,anchor,block_base,gram=_block_base(truth_full,anchor_full,components,columns)
            block_contrast_paths=[]
            for ci,(left,right) in enumerate(STATE_PAIRS):
                tc=truth[:,left]-truth[:,right]; ac=anchor[:,left]-anchor[:,right]
                cpath=temp/f"base_{fold}_gene_{gene_block}_contrast_{ci}.npy"
                cm=_gpu_cross(ac,tc,cpath,rb); block_contrast_paths.append(cpath)
                for start in range(0,p,rb):
                    stop=min(start+rb,p); contrast_totals[ci][start:stop]+=np.asarray(cm[start:stop],np.float32)
                truth_contrast2[:,ci]+=np.square(tc,dtype=np.float64).sum(axis=1)
                base["anchor_contrast2"][:,ci]+=np.square(ac,dtype=np.float64).sum(axis=1)
                vb=components[:,columns]
                truth_contrast_proj[ci]+=tc@vb.T
                base["anchor_contrast_proj"][:,ci]+=ac@vb.T
                del cm
            op_path=temp/f"base_{fold}_gene_{gene_block}_operator.npy"
            opbase=_operator_base_from_contrasts(block_contrast_paths,op_path,p,rb)
            for model in MODELS:
                residuals=item["residuals"][model]; mean_path=temp/f"meanS_{fold}_{model}_gene_{gene_block}.npy"
                seed_null=[]; seed_metrics=[]
                for si,(seed,residual) in enumerate(sorted(residuals.items())):
                    rg,pred2,dot=_predicted_sufficient(block_base,residual,gram)
                    raw_path=temp/f"raw_{fold}_{model}_{seed}_gene_{gene_block}.npy"
                    raw=_add_low_rank(opbase,rg.reshape(p,-1),block_base["truth_proj"].reshape(p,-1),raw_path,rb)
                    ms=_mean_similarity_add(raw,pred2,block_base["truth2"],mean_path,len(residuals),si==0,rb)
                    nd=_permutation_dots(raw,p,draws,int(config["random_seed"])+fold*101+gene_block*13)
                    seed_null.append(nd/np.sqrt(float(pred2.sum()*block_base["truth2"].sum())))
                    seed_metrics.append(_metric_row(model,seed,fold,donor,block_base["truth2"],pred2,dot))
                    del raw,ms; Path(raw_path).unlink()
                ms=np.load(mean_path,mmap_mode="r")
                sm,_=_analyze_matrix(ms,targets,draws,int(config["random_seed"])+fold*3001+gene_block*97+MODELS.index(model),False)
                null=np.mean(np.stack(seed_null),axis=0); observed=float(np.mean([r["cosine"] for r in seed_metrics])); ns=_null_summary(null,observed)
                gene_rows.append({"model":model,"held_out_donor":donor,"outer_fold":fold,"gene_block":gene_block,**ns,"matched_minus_mismatched":sm["median_paired_matched_minus_row_mismatch"],"retrieval_mrr":sm["mean_reciprocal_rank"],"f_parallel_pred":float(np.mean([r["f_parallel_pred"] for r in seed_metrics]))})
                del ms; Path(mean_path).unlink()
            del opbase,truth,anchor
            Path(op_path).unlink()
            for path in block_contrast_paths: Path(path).unlink()
        for matrix in contrast_totals: matrix.flush()
        op_total_path=temp/f"base_total_{fold}_operator.npy"
        op_total=_operator_base_from_contrasts(contrast_total_paths,op_total_path,p,rb)
        for model in MODELS:
            print(f"[1C] donor {donor}: total/contrast matrices for {model}", flush=True)
            result=_model_matrix_analysis(root=root,config=config,temp=temp,matrix_dir=matrix_dir,donor=donor,fold=fold,model=model,targets=targets,base_operator=op_total,base_contrasts=contrast_totals,base_sufficient=base,components=components,residuals=item["residuals"][model],intervention_blocks=intervention_blocks,truth_contrast_proj=truth_contrast_proj,truth_contrast2=truth_contrast2)
            fingerprint_summaries.append(result["fingerprint_summary"]); fingerprint_rows.append(result["fingerprint_rows"]); shuffle_summaries.append(result["shuffle_summary"]); contrast_rows.extend(result["contrast_rows"]); intervention_rows.extend(result["intervention_rows"])
            paired[model][fold]=result["fingerprint_rows"]["matched_minus_row_mismatched"].to_numpy()
            for seed,stats in result["stats_by_seed"].items(): sufficient[model].setdefault(seed,{})[fold]=stats
            pooled_null_parts[model][fold]=result["shuffle_null"]
            pooled_null_parts[model][(fold,"by_seed")]=result["shuffle_null_by_seed"]
            for draw,value in enumerate(result["shuffle_null"]): shuffle_null_rows.append({"scope":"donor","model":model,"held_out_donor":donor,"permutation":draw,"cosine":value,"pearson":value})
        del op_total
        Path(op_total_path).unlink()
        del contrast_totals
        gc.collect()
        del truth_full,anchor_full
    # Pooled donor-preserving intervention shuffle.
    for model in MODELS:
        seed_null=[]
        seed_observed=[]
        for seed,donor_stats in sufficient[model].items():
            t2=sum(float(x["truth2"].sum()) for x in donor_stats.values()); p2=sum(float(x["pred2"].sum()) for x in donor_stats.values()); dp=sum(float(x["dot"].sum()) for x in donor_stats.values())
            seed_observed.append(float(_cos(dp,t2,p2)))
            null_dot=[]
            for fold in range(4):
                donor_null_cos=pooled_null_parts[model][(fold,"by_seed")][seed]
                donor_stat=donor_stats[fold]
                null_dot.append(donor_null_cos*np.sqrt(float(donor_stat["truth2"].sum()*donor_stat["pred2"].sum())))
            seed_null.append(sum(null_dot)/np.sqrt(t2*p2))
        pooled=np.mean(np.stack(seed_null),axis=0)
        obs=float(np.mean(seed_observed)); ns=_null_summary(pooled,obs); ns.update({"scope":"pooled","model":model,"held_out_donor":"ALL","outer_fold":-1}); shuffle_summaries.append(ns)
        for draw,value in enumerate(pooled): shuffle_null_rows.append({"scope":"pooled","model":model,"held_out_donor":"ALL","permutation":draw,"cosine":value,"pearson":value})
    fps=pd.DataFrame(fingerprint_summaries); fps.to_csv(output/"matched_mismatched_summary.csv",index=False)
    retrieval=pd.concat(fingerprint_rows,ignore_index=True); retrieval.to_csv(output/"intervention_retrieval.csv",index=False)
    pd.DataFrame(contrast_rows).to_csv(output/"contrast_specific_identity.csv",index=False)
    pd.DataFrame(shuffle_summaries).to_csv(output/"intervention_shuffle_summary.csv",index=False)
    pd.DataFrame(shuffle_null_rows).to_csv(output/"intervention_shuffle_null.csv",index=False)
    pd.DataFrame(gene_rows).to_csv(output/"gene_block_robustness.csv",index=False)
    intervention_frame=pd.DataFrame(intervention_rows)
    for index,row in intervention_frame.iterrows():
        keep=intervention_blocks==int(row.block)
        if row.analysis=="leave_one_block_out": keep=~keep
        ceilings=[]
        for stats in ceiling_stats.values(): ceilings.append(float(_cos(stats["dot"][keep].sum(),stats["left2"][keep].sum(),stats["right2"][keep].sum())))
        intervention_frame.loc[index,"truth_ceiling"]=np.mean(ceilings)
        intervention_frame.loc[index,"normalized_recovery"]=row.observed_cosine/np.mean(ceilings)
    intervention_frame.to_csv(output/"intervention_block_robustness.csv",index=False)
    return {"fingerprint":fps,"retrieval":retrieval,"shuffle":pd.DataFrame(shuffle_summaries),"contrast":pd.DataFrame(contrast_rows),"gene":pd.DataFrame(gene_rows),"intervention":pd.DataFrame(intervention_rows),"sufficient":sufficient,"paired":paired,"ceiling_stats":ceiling_stats,"donor_ceiling_stats":donor_ceiling_stats}


def _bootstrap(config,matrix_result,models):
    draws=int(config["bootstrap_draws"])
    first_model=next(iter(matrix_result["sufficient"].values()))
    first_seed=next(iter(first_model.values()))
    first_donor=next(iter(first_seed.values()))
    p=len(first_donor["truth2"])
    rng=np.random.default_rng(int(config["random_seed"])+9001)
    model_values={m:{k:np.empty(draws) for k in ("kappa_c","alpha","cosine","e_parallel","e_perpendicular","f_parallel_pred","matched_minus_mismatched","truth_ceiling","normalized_recovery","ceiling_gap")} for m in models}
    ceiling_stats=matrix_result["ceiling_stats"]
    for draw in range(draws):
        idx=rng.integers(0,p,p)
        ceilings=[]
        for item in ceiling_stats.values(): ceilings.append(float(_cos(item["dot"][idx].sum(),item["left2"][idx].sum(),item["right2"][idx].sum())))
        ceiling=float(np.mean(ceilings))
        for model in models:
            per={k:[] for k in ("kappa_c","alpha","cosine","e_parallel","e_perpendicular","f_parallel_pred")}
            for seed,donors in matrix_result["sufficient"][model].items():
                t2=sum(float(x["truth2"][idx].sum()) for x in donors.values()); p2=sum(float(x["pred2"][idx].sum()) for x in donors.values()); dp=sum(float(x["dot"][idx].sum()) for x in donors.values())
                k=p2/t2; a=dp/t2; c=float(_cos(dp,t2,p2));
                for key,value in (("kappa_c",k),("alpha",a),("cosine",c),("e_parallel",a*a),("e_perpendicular",k-a*a),("f_parallel_pred",c*c)): per[key].append(value)
            for key in per: model_values[model][key][draw]=np.mean(per[key])
            effects=np.concatenate([matrix_result["paired"][model][fold][idx] for fold in range(4)])
            model_values[model]["matched_minus_mismatched"][draw]=np.median(effects)
            model_values[model]["truth_ceiling"][draw]=ceiling
            model_values[model]["normalized_recovery"][draw]=model_values[model]["cosine"][draw]/ceiling
            model_values[model]["ceiling_gap"][draw]=ceiling-model_values[model]["cosine"][draw]
    rows=[]
    for model,metrics in model_values.items():
        for metric,values in metrics.items(): rows.append({"model":model,"metric":metric,"point":float(np.mean(values)),"ci_lower":float(np.quantile(values,.025)),"ci_upper":float(np.quantile(values,.975)),"draws":draws})
    return pd.DataFrame(rows)


def _finalize(root,config,output,figures,reproduction,energy,ceiling,shared,matrix_result,tolerance,started):
    bootstrap=_bootstrap(config,matrix_result,MODELS); bootstrap.to_csv(output/"bootstrap_summary.csv",index=False)
    ceiling_primary=float(ceiling.loc[ceiling.ceiling_type.eq("balanced_2v2"),"cosine"].mean())
    ceiling_boot=bootstrap.loc[bootstrap.model.eq("Ridge")&bootstrap.metric.eq("truth_ceiling")].iloc[0]
    ceiling = pd.concat(
        [
            ceiling,
            pd.DataFrame(
                [
                    {"ceiling_type":"balanced_2v2_aggregate","partition":"mean","cosine":ceiling_primary,"ci_lower":ceiling_boot.ci_lower,"ci_upper":ceiling_boot.ci_upper,"bootstrap_draws":int(config["bootstrap_draws"])},
                    {"ceiling_type":"balanced_2v2_aggregate","partition":"median","cosine":float(ceiling.loc[ceiling.ceiling_type.eq("balanced_2v2"),"cosine"].median()),"ci_lower":np.nan,"ci_upper":np.nan,"bootstrap_draws":int(config["bootstrap_draws"])},
                ]
            ),
        ],
        ignore_index=True,
    )
    ceiling.to_csv(output/"truth_noise_ceiling.csv",index=False)
    normalized=[]; donor_rows=[]
    prior_order=pd.read_csv(root/"results/cgc_tcell_1b/contrast_ordering.csv")
    for model in MODELS:
        b=bootstrap.loc[bootstrap.model.eq(model)].set_index("metric")
        normalized.append({"model":model,"scope":"pooled","truth_ceiling":ceiling_primary,"observed_cosine":float(b.loc["cosine","point"]),"normalized_recovery":float(b.loc["normalized_recovery","point"]),"normalized_recovery_ci_lower":float(b.loc["normalized_recovery","ci_lower"]),"normalized_recovery_ci_upper":float(b.loc["normalized_recovery","ci_upper"]),"ceiling_gap":float(b.loc["ceiling_gap","point"]),"ceiling_gap_ci_lower":float(b.loc["ceiling_gap","ci_lower"]),"ceiling_gap_ci_upper":float(b.loc["ceiling_gap","ci_upper"])})
        for fold,donor in enumerate(load_frozen(root)["donors"]):
            e=energy.loc[energy.model.eq(model)&energy.outer_fold.eq(fold)].mean(numeric_only=True)
            sh=matrix_result["shuffle"].loc[(matrix_result["shuffle"].model==model)&(matrix_result["shuffle"].outer_fold==fold)].iloc[0]
            fp=matrix_result["fingerprint"].loc[(matrix_result["fingerprint"].model==model)&(matrix_result["fingerprint"].outer_fold==fold)].iloc[0]
            ret=matrix_result["retrieval"].loc[(matrix_result["retrieval"].model==model)&(matrix_result["retrieval"].outer_fold==fold)]
            dc=matrix_result["donor_ceiling_stats"][fold]; c=float(_cos(dc["dot"].sum(),dc["left2"].sum(),dc["right2"].sum()))
            donor_rows.append({"model":model,"held_out_donor":donor,"outer_fold":fold,"observed_cosine":e.cosine,"shuffle_null_q95":sh.null_q95,"shuffle_empirical_p":sh.empirical_p,"truth_ceiling_donor_vs_other3":c,"normalized_recovery":e.cosine/c,"matched_minus_mismatched":fp.median_paired_matched_minus_row_mismatch,"top1_retrieval":float((ret.true_match_rank<=1).mean()),"top10_retrieval":float((ret.true_match_rank<=10).mean()),"f_parallel_pred":e.f_parallel_pred,"top_contrast_correct":bool(prior_order.loc[prior_order.model.eq(model)&prior_order.outer_fold.eq(fold),"top_contrast_correct"].mean()>.5)})
    normalized_frame=pd.DataFrame(normalized); normalized_frame.to_csv(output/"normalized_operator_recovery.csv",index=False)
    donor=pd.DataFrame(donor_rows); donor.to_csv(output/"donor_summary.csv",index=False)
    pd.DataFrame([{"status":"NOISE_CORRECTED_ALIGNMENT_NOT_IDENTIFIABLE","reason":"Balanced 2-vs-2 truth halves are not repeated measurements of the identical latent donor operator, and model predictions depend on overlapping training donors; attenuation-correction independence assumptions are not satisfied."}]).to_csv(output/"noise_corrected_alignment.csv",index=False)
    pooled_shuffle=matrix_result["shuffle"].loc[matrix_result["shuffle"].scope.eq("pooled")].set_index("model")
    model_pass={}
    for model in MODELS:
        fp=matrix_result["fingerprint"].loc[matrix_result["fingerprint"].model.eq(model)]
        donors_positive=int((fp.median_paired_matched_minus_row_mismatch>0).sum())
        b=bootstrap.loc[bootstrap.model.eq(model)].set_index("metric")
        retrieval_top1=float((matrix_result["retrieval"].loc[matrix_result["retrieval"].model.eq(model),"true_match_rank"]<=1).mean())
        model_pass[model]={"shuffle":bool(pooled_shuffle.loc[model,"observed_exceeds_q95"] and pooled_shuffle.loc[model,"empirical_p"]<.05),"matched_ci":bool(b.loc["matched_minus_mismatched","ci_lower"]>0),"donors_positive":donors_positive>=3,"retrieval":retrieval_top1>1/9386}
    full=sum(all(v.values()) for v in model_pass.values())>=2; shuffled=sum(v["shuffle"] for v in model_pass.values())>=2
    fingerprint_model_mean=matrix_result["fingerprint"].groupby("model")[["median_paired_matched_minus_row_mismatch","top1_accuracy","top10_accuracy"]].mean()
    # Operationalize the protocol's qualitative WEAK clause transparently:
    # a sub-0.01 cosine-unit matched effect or <1% top-1 / <5% top-10
    # absolute retrieval is small even when enriched over a 1/P null.
    weak_absolute=bool(
        (fingerprint_model_mean["median_paired_matched_minus_row_mismatch"]<.01).any()
        or (fingerprint_model_mean["top1_accuracy"]<.01).any()
        or (fingerprint_model_mean["top10_accuracy"]<.05).any()
    )
    signal="INTERVENTION_SPECIFIC_CONTEXT_OPERATOR_SIGNAL_PRESENT" if full and not weak_absolute else ("INTERVENTION_SPECIFIC_CONTEXT_OPERATOR_SIGNAL_WEAK" if shuffled else "INTERVENTION_SPECIFIC_CONTEXT_OPERATOR_SIGNAL_NOT_RESOLVED")
    orthogonal_models=0
    for model in MODELS:
        if int((donor.loc[donor.model.eq(model),"f_parallel_pred"]<.10).sum())>=3: orthogonal_models+=1
    geometry="PREDICTED_CONTEXT_ENERGY_MOSTLY_TRUTH_ORTHOGONAL" if orthogonal_models>=2 else "PREDICTED_CONTEXT_ENERGY_SUBSTANTIALLY_TRUTH_ALIGNED"
    coarse_ok=bool(donor.top_contrast_correct.all() and signal in {"INTERVENTION_SPECIFIC_CONTEXT_OPERATOR_SIGNAL_WEAK","INTERVENTION_SPECIFIC_CONTEXT_OPERATOR_SIGNAL_NOT_RESOLVED"} and (donor.f_parallel_pred<.10).all())
    coarse="COARSE_CONTEXT_ORDERING_PRESERVED_SPECIFIC_OPERATOR_LOST" if coarse_ok else "COARSE_AND_INTERVENTION_SPECIFIC_CONTEXT_STRUCTURE_PRESENT"
    verdict={"git_provenance":git(root,"rev-parse","HEAD"),"frozen_1b_verdict":"CONTEXT_GEOMETRY_COMPRESSION_SUPPORTED","frozen_1b_subtype":"AMPLITUDE_CGC","frozen_1b_changed":False,"intervention_specific_signal_label":signal,"geometry_label":geometry,"coarse_vs_specific_label":coarse,"per_model_criteria":model_pass,"weak_absolute_effect_guard":weak_absolute,"weak_guard_operationalization":{"matched_effect_below":.01,"top1_absolute_below":.01,"top10_absolute_below":.05},"fingerprint_model_means":fingerprint_model_mean.to_dict(orient="index")}
    (output/"verdict.json").write_text(json.dumps(_safe_json(verdict),indent=2))
    run_manifest={"git_provenance":git(root,"rev-parse","HEAD"),"branch":config["branch"],"runtime_seconds":time.perf_counter()-started,"predictive_model_fits":0,"checkpoint_loads":0,"permutations":config["permutations"],"bootstrap_draws":config["bootstrap_draws"],"metric_reproduction_tolerance":tolerance,"generic_template_result":"GENERIC_STIMULATION_TEMPLATE_ALREADY_REMOVED_BY_TWO_WAY_CENTERING","noise_corrected_alignment":"NOISE_CORRECTED_ALIGNMENT_NOT_IDENTIFIABLE","gpu":torch.cuda.get_device_name(0),"peak_vram_bytes":torch.cuda.max_memory_allocated()}
    (output/"run_manifest.json").write_text(json.dumps(_safe_json(run_manifest),indent=2))
    model_energy=energy.groupby("model")[["e_parallel","e_perpendicular","f_parallel_pred"]].mean()
    _svg(figures/"figure_1_energy_decomposition.svg","Energy decomposition",[(f"{m} parallel",model_energy.loc[m,"e_parallel"],"#2b6cb0") for m in MODELS]+[(f"{m} orthogonal",model_energy.loc[m,"e_perpendicular"],"#d97706") for m in MODELS])
    _svg(figures/"figure_2_intervention_shuffle.svg","Observed cosine vs intervention-shuffle q95",sum(([(f"{m} observed",pooled_shuffle.loc[m,"observed_cosine"],"#2b6cb0"),(f"{m} null q95",pooled_shuffle.loc[m,"null_q95"],"#9ca3af")] for m in MODELS),[]))
    fpmean=matrix_result["fingerprint"].groupby("model")[["median_matched_cosine","median_mismatched_cosine"]].mean()
    _svg(figures/"figure_3_matched_mismatched.svg","Matched vs mismatched fingerprints",sum(([(f"{m} matched",fpmean.loc[m,"median_matched_cosine"],"#2b6cb0"),(f"{m} mismatch",fpmean.loc[m,"median_mismatched_cosine"],"#9ca3af")] for m in MODELS),[]))
    _svg(figures/"figure_4_truth_ceiling.svg","Truth ceiling vs model and shuffle",[("truth 2v2",ceiling_primary,"#15803d")]+[(f"{m} model",pooled_shuffle.loc[m,"observed_cosine"],"#2b6cb0") for m in MODELS]+[(f"{m} shuffle",pooled_shuffle.loc[m,"null_q95"],"#9ca3af") for m in MODELS])
    top10=donor.groupby("model").top10_retrieval.mean()
    _svg(figures/"figure_5_intervention_retrieval.svg","Top-10 intervention retrieval",[(m,top10[m],"#7c3aed") for m in MODELS]+[("random",10/9386,"#9ca3af")])
    _svg(figures/"figure_6_coarse_vs_specific.svg","Coarse contrast vs specific operator",[(f"{m} top contrast",donor.loc[donor.model.eq(m),"top_contrast_correct"].mean(),"#15803d") for m in MODELS]+[(f"{m} fingerprint cos",matrix_result["fingerprint"].loc[matrix_result["fingerprint"].model.eq(m),"median_matched_cosine"].mean(),"#d97706") for m in MODELS])
    _reports_1c(root,config,verdict,energy,matrix_result,ceiling,normalized_frame,donor,shared,bootstrap)


def _md(frame: pd.DataFrame) -> str:
    cols=list(frame.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]+["---:"]*(len(cols)-1))+" |"]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for v in row:
            if isinstance(v,(float,np.floating)): values.append("NA" if np.isnan(v) else f"{v:.6g}")
            else: values.append(str(v))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def _reports_1c(root,config,verdict,energy,matrix_result,ceiling,normalized,donor,shared,bootstrap):
    reports=root/config["report_directory"]; pooled=matrix_result["shuffle"].loc[matrix_result["shuffle"].scope.eq("pooled"),["model","observed_cosine","null_mean","null_q95","null_q99","empirical_p","z_score"]]
    e=energy.groupby("model",as_index=False)[["kappa_c","alpha","cosine","e_parallel","e_perpendicular","f_parallel_pred"]].mean()
    fp=matrix_result["fingerprint"].groupby("model",as_index=False)[["median_matched_cosine","median_mismatched_cosine","median_paired_matched_minus_row_mismatch","top1_accuracy","top10_accuracy","mean_reciprocal_rank","median_true_match_rank"]].mean()
    text=f"""# CGC-TCELL-1C operator-alignment adjudication

Git provenance: `{verdict['git_provenance']}` on `{config['branch']}`. No model was trained, retrained, tuned, or checkpoint-loaded.

## Energy decomposition

{_md(e)}

## Intervention-identity shuffle

{_md(pooled)}

## Matched fingerprints and retrieval

{_md(fp)}

## Truth ceiling and normalized recovery

{_md(ceiling)}

{_md(normalized)}

## Four-donor adjudication

{_md(donor)}

## Shared-axis residualization

{_md(shared.groupby(['model','removed_rank'],as_index=False)[['residual_cosine','residual_alpha']].mean())}

Interpretation labels:

- `{verdict['intervention_specific_signal_label']}`
- `{verdict['geometry_label']}`
- `{verdict['coarse_vs_specific_label']}`

CGC-TCELL-1C does not change the frozen 1B verdict: **NO**. The preregistered 1B relational criterion detected statistically positive coarse ordering, whereas the post-frozen 1C adjudication tested the stronger question of intervention-identity-specific operator recovery.
"""
    (reports/"cgc_tcell_1c_operator_alignment.md").write_text(text,encoding="utf-8")
    answers=f"""# CGC-TCELL-1C mechanistic interpretation

Git provenance: `{verdict['git_provenance']}`.

1. Parallel predicted energy: see `f_parallel_pred` and `e_parallel` in the frozen energy table.
2. Orthogonal predicted energy: see `e_perpendicular`; label `{verdict['geometry_label']}`.
3. Shuffle separation: see pooled empirical p and q95 in the operator report.
4. Intervention identification: top-1/top-10 and MRR are reported against exact random expectations.
5. Matched versus mismatched: paired effect, bootstrap CI, and permutation p are reported.
6. Truth reproducibility: all three balanced 2-vs-2 ceilings and normalized recovery are reported.
7. Donor consistency: all primary quantities are listed for all four held-out donors.
8. Robustness: frozen five-gene-block and five-intervention-block results are in the required CSVs.
9. Shared axes: K=1,3,5,10 residualized results are reported without selecting K.
10. Coarse versus specific: `{verdict['coarse_vs_specific_label']}`.
11. Does 1C alter 1B? **NO**.

Noise-corrected alignment is `NOISE_CORRECTED_ALIGNMENT_NOT_IDENTIFIABLE`; the attenuation-correction independence assumptions are not satisfied.
"""
    (reports/"cgc_tcell_1c_mechanistic_interpretation.md").write_text(answers,encoding="utf-8")
    (reports/"cgc_tcell_1c_phase_summary.md").write_text(f"# CGC-TCELL-1C phase summary\n\nFrozen 1B verdict: `CONTEXT_GEOMETRY_COMPRESSION_SUPPORTED` (`AMPLITUDE_CGC`), unchanged.\n\nFinal mechanistic labels:\n\n- `{verdict['intervention_specific_signal_label']}`\n- `{verdict['geometry_label']}`\n- `{verdict['coarse_vs_specific_label']}`\n\nNo predictive training, retraining, tuning, or architecture rescue was performed.\n",encoding="utf-8")


def run(root: Path, config_path: Path):
    started=time.perf_counter(); config=yaml.safe_load(config_path.read_text())
    if git(root,"branch","--show-current") != config["branch"]: raise RuntimeError("Wrong CGC-TCELL-1C branch")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for frozen matrix adjudication")
    output=root/config["output_directory"]; output.mkdir(parents=True,exist_ok=True)
    matrix_dir=output/"fingerprint_similarity_matrix"; matrix_dir.mkdir(exist_ok=True)
    figures=output/"figures"; figures.mkdir(exist_ok=True)
    temp=root/config["temporary_directory"]
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    print("[1C] hashing and verifying frozen inputs", flush=True)
    hashes=_input_hashes(root,config); (output/"frozen_input_hashes.json").write_text(json.dumps(hashes,indent=2))
    frozen=load_frozen(root); primary=frozen["primary"]
    delta=np.memmap(root/primary["response_path"],mode="r",dtype=np.float32,shape=tuple(primary["response_shape"]))
    donors,targets=frozen["donors"],frozen["targets"]
    strict=pd.read_csv(root/"results/cgc_tcell/strict_trans_genes.csv")
    genes=strict.loc[strict.strict_trans_eligible,"gene_id"].astype(str).tolist()
    gene_blocks=np.asarray([hash_block(g) for g in genes]); intervention_blocks=np.asarray([hash_block(p) for p in targets])
    print("[1C] reproducing frozen 1B metrics", flush=True)
    reproduction,cache,tolerance=reproduce_gate(root,config,delta,donors,targets)
    reproduction.to_csv(output/"metric_reproduction.csv",index=False)
    energy=reproduction.loc[reproduction.scope.eq("donor"),["model","seed","outer_fold","held_out_donor","kappa_c","alpha","cosine","e_parallel","e_perpendicular","f_parallel_pred","f_perpendicular_pred","energy_identity_error","cosine_square_identity_error"]].copy()
    energy.to_csv(output/"parallel_orthogonal_energy.csv",index=False)
    energy[["model","seed","outer_fold","held_out_donor","f_parallel_pred","f_perpendicular_pred"]].to_csv(output/"predicted_energy_fraction.csv",index=False)
    print("[1C] computing truth noise ceilings", flush=True)
    ceiling,ceiling_stats,donor_ceiling_stats,block_ceiling_stats=_truth_ceiling(delta,donors,gene_blocks,config)
    ceiling.to_csv(output/"truth_noise_ceiling.csv",index=False)
    print("[1C] fitting evaluation-only training-donor shared axes", flush=True)
    shared=_shared_axis_rows(delta,cache,donors,config); shared.to_csv(output/"shared_axis_residualization.csv",index=False)
    # Remaining full-matrix analyses are implemented in the second stage so
    # that the mandatory reproduction gate always finishes and freezes first.
    print("[1C] starting exact full fingerprint matrices", flush=True)
    matrix_result=_full_matrix_stage(root,config,delta,cache,donors,targets,gene_blocks,intervention_blocks,ceiling,ceiling_stats,donor_ceiling_stats,block_ceiling_stats,temp,matrix_dir,output)
    print("[1C] bootstrapping and writing frozen reports", flush=True)
    _finalize(root,config,output,figures,reproduction,energy,ceiling,shared,matrix_result,tolerance,started)
    from igc_virtual_cell.cgc_tcell_1c.figures import generate as generate_figures
    generate_figures(root)
    shutil.rmtree(temp)
    return json.loads((output/"verdict.json").read_text())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path.cwd()); parser.add_argument("--config",type=Path,default=Path("configs/cgc_tcell_1c.yaml")); args=parser.parse_args()
    root=args.root.resolve(); config=args.config if args.config.is_absolute() else root/args.config
    print(json.dumps(run(root,config),indent=2))


if __name__ == "__main__":
    main()
