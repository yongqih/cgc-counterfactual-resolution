"""Resumable strict-OOF Level-2 RNA-only pilot for GSE207049."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.extmath import randomized_svd

from igc_virtual_cell.level2_lea import pairing_table


SEED = 207049
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
PCA_COMPONENTS = [10, 25, 50, 100]
MODEL_NAMES = ["null", "ridge", "pca_ridge", "mlp"]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32, copy=False)


def dual_ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    x_mean, x_scale = standardize_fit(x_train)
    xtr = standardize_apply(x_train, x_mean, x_scale).astype(np.float64)
    xte = standardize_apply(x_test, x_mean, x_scale).astype(np.float64)
    y_mean = y_train.mean(axis=0, dtype=np.float64)
    yc = y_train.astype(np.float64) - y_mean
    gram = xtr @ xtr.T
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(len(xtr), dtype=np.float64), yc
    )
    return (xte @ xtr.T @ coefficients + y_mean).astype(np.float32)


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.square(y_true, dtype=np.float64)))
    numerator = float(np.sum(np.square(y_true - y_pred, dtype=np.float64)))
    return 1.0 - numerator / denominator if denominator > 0 else float("nan")


def tune_ridge(
    x: np.ndarray, delta: np.ndarray, strata: np.ndarray, seed: int
) -> tuple[float, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    totals = {alpha: [0.0, 0.0] for alpha in ALPHAS}
    for train, valid in splitter.split(x, strata):
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        y_valid = delta[valid] - mean_delta
        denominator = float(np.sum(np.square(y_valid, dtype=np.float64)))
        for alpha in ALPHAS:
            prediction = dual_ridge_predict(x[train], y_train, x[valid], alpha)
            totals[alpha][0] += float(np.sum(np.square(y_valid - prediction, dtype=np.float64)))
            totals[alpha][1] += denominator
    rows = [
        {"alpha": alpha, "inner_pooled_R2": 1 - sse / denominator}
        for alpha, (sse, denominator) in totals.items()
    ]
    table = pd.DataFrame(rows)
    best = float(table.sort_values(["inner_pooled_R2", "alpha"], ascending=[False, True]).iloc[0]["alpha"])
    return best, table


def pca_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    mean, scale = standardize_fit(x_train)
    xtr = standardize_apply(x_train, mean, scale)
    xte = standardize_apply(x_test, mean, scale)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    z_train = pca.fit_transform(xtr).astype(np.float32)
    z_test = pca.transform(xte).astype(np.float32)
    return dual_ridge_predict(z_train, y_train, z_test, alpha)


def tune_pca_ridge(
    x: np.ndarray, delta: np.ndarray, strata: np.ndarray, seed: int
) -> tuple[int, float, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    totals = {(k, alpha): [0.0, 0.0] for k in PCA_COMPONENTS for alpha in ALPHAS}
    for split_number, (train, valid) in enumerate(splitter.split(x, strata)):
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        y_valid = delta[valid] - mean_delta
        denominator = float(np.sum(np.square(y_valid, dtype=np.float64)))
        x_mean, x_scale = standardize_fit(x[train])
        xtr = standardize_apply(x[train], x_mean, x_scale)
        xva = standardize_apply(x[valid], x_mean, x_scale)
        max_k = min(max(PCA_COMPONENTS), len(train) - 1)
        pca = PCA(n_components=max_k, svd_solver="randomized", random_state=seed + split_number)
        z_train_full = pca.fit_transform(xtr).astype(np.float32)
        z_valid_full = pca.transform(xva).astype(np.float32)
        for k in PCA_COMPONENTS:
            if k > max_k:
                continue
            for alpha in ALPHAS:
                prediction = dual_ridge_predict(
                    z_train_full[:, :k], y_train, z_valid_full[:, :k], alpha
                )
                totals[(k, alpha)][0] += float(
                    np.sum(np.square(y_valid - prediction, dtype=np.float64))
                )
                totals[(k, alpha)][1] += denominator
    rows = [
        {"n_components": k, "alpha": alpha, "inner_pooled_R2": 1 - sse / denominator}
        for (k, alpha), (sse, denominator) in totals.items()
        if denominator > 0
    ]
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["inner_pooled_R2", "n_components", "alpha"],
        ascending=[False, True, True],
    ).iloc[0]
    return int(best["n_components"]), float(best["alpha"]), table


class RNAResidualMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def fit_mlp_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = RNAResidualMLP(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
    xv = torch.from_numpy(x_valid).to(device)
    yv = torch.from_numpy(y_valid).to(device)
    best_loss = float("inf")
    best_epoch = 1
    patience = 0
    history = []
    for epoch in range(1, 201):
        model.train()
        train_sum = 0.0
        train_n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(xb)
            train_n += len(xb)
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(xv), yv))
        history.append({"epoch": epoch, "train_MSE": train_sum / train_n, "valid_MSE": valid_loss})
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history


def train_mlp_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = RNAResidualMLP(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test), 64):
            predictions.append(model(torch.from_numpy(x_test[start:start + 64]).to(device)).cpu().numpy())
    result = np.concatenate(predictions).astype(np.float32)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mlp_predict(
    x_train: np.ndarray,
    delta_train: np.ndarray,
    x_test: np.ndarray,
    strata: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    inner_train, valid = next(splitter.split(x_train, strata))
    inner_mean_delta = delta_train[inner_train].mean(axis=0)
    y_inner = delta_train[inner_train] - inner_mean_delta
    y_valid = delta_train[valid] - inner_mean_delta
    x_mean, x_scale = standardize_fit(x_train[inner_train])
    x_inner = standardize_apply(x_train[inner_train], x_mean, x_scale)
    x_valid = standardize_apply(x_train[valid], x_mean, x_scale)
    best_epoch, history = fit_mlp_epochs(
        x_inner, y_inner.astype(np.float32), x_valid, y_valid.astype(np.float32), device, seed
    )
    outer_mean_delta = delta_train.mean(axis=0)
    y_outer = (delta_train - outer_mean_delta).astype(np.float32)
    outer_x_mean, outer_x_scale = standardize_fit(x_train)
    x_outer = standardize_apply(x_train, outer_x_mean, outer_x_scale)
    x_test_scaled = standardize_apply(x_test, outer_x_mean, outer_x_scale)
    prediction = train_mlp_fixed_epochs(
        x_outer, y_outer, x_test_scaled, best_epoch, device, seed + 1000
    )
    return prediction, best_epoch, pd.DataFrame(history)


def load_paired_data(matrix_path: Path, metadata_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path, sep="\t")
    samples, pairing = pairing_table(metadata)
    paired_lines = sorted(pairing.loc[pairing["paired"], "line"])
    selected = samples.loc[samples["line"].isin(paired_lines), "raw_file_name"].tolist()
    dtype = {name: np.float32 for name in selected}
    expression = pd.read_csv(
        matrix_path, sep="\t", usecols=["GeneID", *selected], dtype=dtype
    )
    genes = expression.pop("GeneID").astype(str).tolist()
    baseline = []
    treated = []
    rows = []
    indexed_pairing = pairing.set_index("line")
    for line in paired_lines:
        group = samples.loc[samples["line"] == line]
        etoh = group.loc[group["treatment"] == "ETOH", "raw_file_name"].tolist()
        dex = group.loc[group["treatment"] == "DEX", "raw_file_name"].tolist()
        baseline.append(expression[etoh].mean(axis=1).to_numpy(dtype=np.float32))
        treated.append(expression[dex].mean(axis=1).to_numpy(dtype=np.float32))
        row = indexed_pairing.loc[line]
        rows.append({
            "line": line, "ancestry": row["pop2"], "population": row["pop"],
            "individual_id": row["1000_genomes_id1"], "ETOH_replicates": len(etoh),
            "DEX_replicates": len(dex),
        })
    return genes, np.asarray(baseline), np.asarray(treated), pd.DataFrame(rows)


def vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    pearson_den = np.sqrt((yt * yt).sum(axis=1) * (yp * yp).sum(axis=1))
    pearson = np.divide((yt * yp).sum(axis=1), pearson_den, out=np.full(len(yt), np.nan), where=pearson_den > 0)
    cosine_den = np.sqrt((y_true * y_true).sum(axis=1) * (y_pred * y_pred).sum(axis=1))
    cosine = np.divide((y_true * y_pred).sum(axis=1), cosine_den, out=np.full(len(yt), np.nan), where=cosine_den > 0)
    return pearson, cosine


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, dex_true: np.ndarray, dex_pred: np.ndarray) -> dict[str, float]:
    pearson, cosine = vector_metrics(y_true, y_pred)
    finite_pearson = pearson[np.isfinite(pearson)]
    finite_cosine = cosine[np.isfinite(cosine)]
    per_gene_den = np.sum(np.square(y_true, dtype=np.float64), axis=0)
    per_gene_r2 = 1 - np.divide(
        np.sum(np.square(y_true - y_pred, dtype=np.float64), axis=0),
        per_gene_den, out=np.full(y_true.shape[1], np.nan), where=per_gene_den > 0,
    )
    state_pearson, _ = vector_metrics(dex_true, dex_pred)
    return {
        "pooled_OOF_residual_R2": pooled_r2(y_true, y_pred),
        "median_per_gene_R2": float(np.nanmedian(per_gene_r2)),
        "mean_per_gene_R2": float(np.nanmean(per_gene_r2)),
        "fraction_genes_R2_gt_0": float(np.nanmean(per_gene_r2 > 0)),
        "fraction_genes_R2_gt_0.05": float(np.nanmean(per_gene_r2 > 0.05)),
        "fraction_genes_R2_gt_0.1": float(np.nanmean(per_gene_r2 > 0.1)),
        "median_response_vector_Pearson": float(np.median(finite_pearson)) if len(finite_pearson) else float("nan"),
        "median_response_vector_cosine": float(np.median(finite_cosine)) if len(finite_cosine) else float("nan"),
        "variance_retention_kappa": float(np.var(y_pred) / np.var(y_true)),
        "median_absolute_state_Pearson": float(np.nanmedian(state_pearson)),
        "absolute_state_pooled_R2": float(1 - np.sum((dex_true - dex_pred) ** 2) / np.sum((dex_true - dex_true.mean(axis=0)) ** 2)),
    }


def geometry(y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    all_true = []
    all_pred = []
    for fold in sorted(set(folds)):
        indices = np.flatnonzero(folds == fold)
        true_dist = pdist(y_true[indices], metric="euclidean")
        pred_dist = pdist(y_pred[indices], metric="euclidean")
        rho = float(spearmanr(true_dist, pred_dist).statistic) if np.std(pred_dist) > 0 else float("nan")
        rows.append({"outer_fold": fold, "n_LCLs": len(indices), "pair_count": len(true_dist), "Spearman": rho})
        all_true.append(true_dist)
        all_pred.append(pred_dist)
    pooled = float(spearmanr(np.concatenate(all_true), np.concatenate(all_pred)).statistic) if np.std(np.concatenate(all_pred)) > 0 else float("nan")
    rows.append({"outer_fold": "pooled_within_fold", "n_LCLs": len(y_true), "pair_count": sum(map(len, all_true)), "Spearman": pooled})
    return pooled, pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def bootstrap_metrics(y_true: np.ndarray, predictions: dict[str, np.ndarray], folds: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 9000)
    n = len(y_true)
    n_genes = y_true.shape[1]
    true_dist = squareform(pdist(y_true))
    pred_dist = {name: squareform(pdist(value)) for name, value in predictions.items()}
    true_row_sums = y_true.sum(axis=1, dtype=np.float64)
    true_row_sumsq = np.square(y_true, dtype=np.float64).sum(axis=1)
    cached = {}
    for name, prediction in predictions.items():
        line_pearson, _ = vector_metrics(y_true, prediction)
        cached[name] = {
            "line_pearson": line_pearson,
            "row_sse": np.square(y_true - prediction, dtype=np.float64).sum(axis=1),
            "pred_row_sums": prediction.sum(axis=1, dtype=np.float64),
            "pred_row_sumsq": np.square(prediction, dtype=np.float64).sum(axis=1),
        }
    rows = []
    for draw in range(1000):
        indices = rng.integers(0, n, n)
        valid_pairs = np.triu(folds[indices, None] == folds[indices][None, :], 1)
        distinct = indices[:, None] != indices[None, :]
        valid_pairs &= distinct
        for name, prediction in predictions.items():
            model_cache = cached[name]
            denominator = float(true_row_sumsq[indices].sum())
            residual_r2 = 1 - float(model_cache["row_sse"][indices].sum()) / denominator
            pred_mean = float(model_cache["pred_row_sums"][indices].sum()) / (n * n_genes)
            pred_var = float(model_cache["pred_row_sumsq"][indices].sum()) / (n * n_genes) - pred_mean**2
            true_mean = float(true_row_sums[indices].sum()) / (n * n_genes)
            true_var = float(true_row_sumsq[indices].sum()) / (n * n_genes) - true_mean**2
            sampled_pearson = model_cache["line_pearson"][indices]
            finite_pearson = sampled_pearson[np.isfinite(sampled_pearson)]
            geometry_rho = float(spearmanr(
                true_dist[np.ix_(indices, indices)][valid_pairs],
                pred_dist[name][np.ix_(indices, indices)][valid_pairs],
            ).statistic) if pred_var > 0 else float("nan")
            rows.append({
                "draw": draw, "model": name,
                "pooled_OOF_residual_R2": residual_r2,
                "median_response_vector_Pearson": float(np.median(finite_pearson)) if len(finite_pearson) else float("nan"),
                "variance_retention_kappa": pred_var / true_var,
                "geometry_Spearman": geometry_rho,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if json.loads(args.integrity.read_text())["verdict"] != "DATA_INTEGRITY_PASS":
        raise SystemExit("DATA_INTEGRITY_FAIL: modeling prohibited")
    if not json.loads(args.truth.read_text())["truth_gate_pass"]:
        raise SystemExit("TRUTH_GATE_FAIL: modeling prohibited")

    started = time.perf_counter()
    genes, baseline, treated, units = load_paired_data(args.matrix, args.metadata)
    delta = treated - baseline
    strata = units["ancestry"].to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(outer.split(baseline, strata))
    fold_ids = np.full(len(units), -1, dtype=int)
    for fold, (_, test) in enumerate(splits):
        fold_ids[test] = fold
    if np.any(fold_ids < 0):
        raise AssertionError("Incomplete OOF assignment")
    units["outer_fold"] = fold_ids
    units.to_csv(args.output / "PAIRING_AUDIT.csv", index=False)

    predictions = {name: np.zeros_like(delta, dtype=np.float32) for name in MODEL_NAMES}
    targets = np.zeros_like(delta, dtype=np.float32)
    shared_means = np.zeros_like(delta, dtype=np.float32)
    tuning_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train, test) in enumerate(splits):
        checkpoint = checkpoint_dir / f"fold_{fold}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint, allow_pickle=False)
            if np.array_equal(saved["test_indices"], test):
                targets[test] = saved["targets"]
                shared_means[test] = saved["shared_means"]
                for name in MODEL_NAMES:
                    predictions[name][test] = saved[name]
                tuning_rows.append(pd.DataFrame([{
                    "outer_fold": fold, "model": "resumed_fold_summary",
                    "ridge_alpha": float(saved["ridge_alpha"]),
                    "pca_components": int(saved["pca_components"]),
                    "pca_alpha": float(saved["pca_alpha"]),
                    "best_epoch": int(saved["mlp_best_epoch"]),
                }]))
                print(f"fold {fold}: resumed", flush=True)
                continue
        mean_delta = delta[train].mean(axis=0)
        y_train = (delta[train] - mean_delta).astype(np.float32)
        y_test = (delta[test] - mean_delta).astype(np.float32)
        targets[test] = y_test
        shared_means[test] = mean_delta
        predictions["null"][test] = 0.0

        ridge_alpha, ridge_tuning = tune_ridge(
            baseline[train], delta[train], strata[train], SEED + fold
        )
        ridge_tuning["outer_fold"] = fold
        ridge_tuning["model"] = "ridge"
        tuning_rows.append(ridge_tuning)
        predictions["ridge"][test] = dual_ridge_predict(
            baseline[train], y_train, baseline[test], ridge_alpha
        )

        pca_k, pca_alpha, pca_tuning = tune_pca_ridge(
            baseline[train], delta[train], strata[train], SEED + 100 + fold
        )
        pca_tuning["outer_fold"] = fold
        pca_tuning["model"] = "pca_ridge"
        tuning_rows.append(pca_tuning)
        predictions["pca_ridge"][test] = pca_ridge_predict(
            baseline[train], y_train, baseline[test], pca_k, pca_alpha, SEED + fold
        )

        mlp_prediction, best_epoch, history = mlp_predict(
            baseline[train], delta[train], baseline[test], strata[train], device, SEED + fold
        )
        predictions["mlp"][test] = mlp_prediction
        history["outer_fold"] = fold
        history["model"] = "mlp"
        history.to_csv(checkpoint_dir / f"fold_{fold}_mlp_history.csv", index=False)
        tuning_rows.append(pd.DataFrame([{
            "outer_fold": fold, "model": "mlp", "best_epoch": best_epoch,
            "device": str(device), "ridge_alpha": ridge_alpha,
            "pca_components": pca_k, "pca_alpha": pca_alpha,
        }]))

        np.savez_compressed(
            checkpoint, test_indices=test, targets=y_test,
            shared_means=np.broadcast_to(mean_delta, y_test.shape),
            ridge_alpha=np.asarray(ridge_alpha), pca_components=np.asarray(pca_k),
            pca_alpha=np.asarray(pca_alpha), mlp_best_epoch=np.asarray(best_epoch),
            **{name: predictions[name][test] for name in MODEL_NAMES},
        )
        print(
            f"fold {fold}: ridge alpha={ridge_alpha:g}; pca={pca_k}, alpha={pca_alpha:g}; mlp epochs={best_epoch}",
            flush=True,
        )

    if tuning_rows:
        pd.concat(tuning_rows, ignore_index=True, sort=False).to_csv(
            args.output / "MODEL_INNER_TUNING.csv", index=False
        )
    pc_rows = []
    for fold, (train, test) in enumerate(splits):
        mean_delta = delta[train].mean(axis=0)
        y_train = (delta[train] - mean_delta).astype(np.float32)
        _, _, vt = randomized_svd(
            y_train, n_components=10, random_state=SEED + 500 + fold
        )
        for model_name in MODEL_NAMES:
            z_true = targets[test] @ vt.T
            z_pred = predictions[model_name][test] @ vt.T
            for component in range(10):
                denominator = float(np.sum(z_true[:, component] ** 2))
                pc_rows.append({
                    "outer_fold": fold, "model": model_name, "component": component + 1,
                    "coordinate_R2": float(1 - np.sum((z_true[:, component] - z_pred[:, component]) ** 2) / denominator),
                    "true_variance": float(np.var(z_true[:, component])),
                    "predicted_variance": float(np.var(z_pred[:, component])),
                })
    pd.DataFrame(pc_rows).to_csv(args.output / "RESPONSE_PC_RESULTS.csv", index=False)
    np.savez_compressed(
        args.output / "LEVEL2_OOF_FROZEN.npz", targets=targets, baseline=baseline,
        treated=treated, shared_means=shared_means, folds=fold_ids,
        **{name: value for name, value in predictions.items()},
    )

    model_rows = []
    per_gene_rows = []
    per_lcl_rows = []
    geometry_tables = []
    for name in MODEL_NAMES:
        dex_prediction = baseline + shared_means + predictions[name]
        summary = metric_summary(targets, predictions[name], treated, dex_prediction)
        geometry_rho, geometry_table = geometry(targets, predictions[name], fold_ids)
        summary.update({"model": name, "geometry_Spearman": geometry_rho})
        model_rows.append(summary)
        geometry_table["model"] = name
        geometry_tables.append(geometry_table)

        gene_den = np.sum(np.square(targets, dtype=np.float64), axis=0)
        gene_r2 = 1 - np.divide(
            np.sum(np.square(targets - predictions[name], dtype=np.float64), axis=0),
            gene_den, out=np.full(len(genes), np.nan), where=gene_den > 0,
        )
        per_gene_rows.append(pd.DataFrame({"gene_id": genes, "model": name, "OOF_R2": gene_r2}))
        pearson, cosine = vector_metrics(targets, predictions[name])
        per_lcl_rows.append(pd.DataFrame({
            "line": units["line"], "outer_fold": fold_ids, "model": name,
            "response_vector_Pearson": pearson, "response_vector_cosine": cosine,
        }))

    model_results = pd.DataFrame(model_rows)
    model_results.to_csv(args.output / "MODEL_RESULTS.csv", index=False)
    pd.concat(per_gene_rows, ignore_index=True).to_csv(args.output / "PER_GENE_R2.csv", index=False)
    pd.concat(per_lcl_rows, ignore_index=True).to_csv(args.output / "PER_LCL_METRICS.csv", index=False)
    pd.concat(geometry_tables, ignore_index=True).to_csv(args.output / "GEOMETRY_RESULTS.csv", index=False)

    bootstrap = bootstrap_metrics(targets, predictions, fold_ids)
    bootstrap.to_csv(args.output / "BOOTSTRAP_FROZEN_OOF.csv", index=False)
    ci_rows = []
    for model, group in bootstrap.groupby("model"):
        for metric in ["pooled_OOF_residual_R2", "median_response_vector_Pearson", "variance_retention_kappa", "geometry_Spearman"]:
            low, high = group[metric].quantile([0.025, 0.975])
            ci_rows.append({"model": model, "metric": metric, "CI_low": low, "CI_high": high})
    pd.DataFrame(ci_rows).to_csv(args.output / "BOOTSTRAP_CI.csv", index=False)

    figure_dir = args.output / "figures"
    figure_dir.mkdir(exist_ok=True)
    for metric, filename, ylabel in [
        ("pooled_OOF_residual_R2", "model_residual_r2.png", "Pooled OOF residual R²"),
        ("variance_retention_kappa", "variance_retention.png", "Variance retention κ"),
        ("geometry_Spearman", "geometry_recovery.png", "Within-fold geometry Spearman"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(model_results["model"], model_results[metric], color="#3B82F6")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)

    runtime = time.perf_counter() - started
    best_nonnull = model_results.loc[model_results["model"] != "null"].sort_values(
        "pooled_OOF_residual_R2", ascending=False
    ).iloc[0]
    report = f"""# LEVEL2-LEA Minimal Pilot Report

## Scope

- Biological prediction units: {len(units)} paired LCLs; genes: {len(genes):,}.
- Strict five-fold LCL-disjoint OOF prediction, ancestry-stratified.
- Predictor: baseline ETOH RNA only.
- Target: DEX-minus-ETOH response minus the outer-training mean response.
- Models: zero-residual null, dual Ridge, fold-local PCA+Ridge, and two-hidden-layer MLP.
- Device for MLP: `{device}`; run time: {runtime / 60:.1f} minutes.
- Pretrained transcriptomic models were not run.

## Primary pilot results

{markdown_table(model_results)}

The best non-null pilot model was `{best_nonnull['model']}` with pooled OOF
residual R² **{best_nonnull['pooled_OOF_residual_R2']:.4f}**, variance retention
**{best_nonnull['variance_retention_kappa']:.4f}**, and within-fold geometry
Spearman **{best_nonnull['geometry_Spearman']:.4f}**.

## Interpretation boundary

This is the preregistered minimal pilot, not a Level-3 biological-equivalence
claim. It uses the authors' post-SVA `voom_resid` representation. The Truth Gate
and data-integrity reports must be read with this report. No genotype, ancestry,
DEX-derived feature, or response-eQTL label was supplied to a predictor.
"""
    (args.output / "LEVEL2_PILOT_REPORT.md").write_text(report, encoding="utf-8")
    print(model_results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
