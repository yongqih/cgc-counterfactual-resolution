"""Fold-local response-program recovery for all frozen pilot/full models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x0 = np.asarray(x, dtype=np.float64).ravel()
    y0 = np.asarray(y, dtype=np.float64).ravel()
    x0 -= x0.mean()
    y0 -= y0.mean()
    denominator = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
    return float(np.dot(x0, y0) / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-oof", type=Path, required=True)
    parser.add_argument("--full-oof", type=Path, required=True)
    parser.add_argument("--truth-pc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = np.load(args.pilot_oof, allow_pickle=False)
    full = np.load(args.full_oof, allow_pickle=False)
    baseline = pilot["baseline"]
    delta = pilot["treated"] - baseline
    targets = pilot["targets"]
    folds = pilot["folds"].astype(int)
    predictions = {
        "null": pilot["null"],
        "ridge": pilot["ridge"],
        "pca_ridge": pilot["pca_ridge"],
        "pilot_mlp": pilot["mlp"],
        "rbf_kernel_ridge": full["rbf_kernel_ridge"],
        "hist_gradient_boosting": full["hist_gradient_boosting"],
        "deep_residual_mlp": full["deep_residual_mlp"],
    }
    truth_reliability = pd.read_csv(args.truth_pc).set_index("component")["replicate_correlation"]

    detailed_rows = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        if not np.allclose(targets[test], delta[test] - mean_delta, atol=2e-6):
            raise AssertionError(f"Target mismatch in fold {fold}")
        pca = PCA(n_components=16, svd_solver="randomized", random_state=207049 + fold)
        pca.fit(y_train)
        true_coordinates = pca.transform(targets[test])
        for model, prediction in predictions.items():
            pred_coordinates = pca.transform(prediction[test])
            for local_index, sample_index in enumerate(test):
                for component in range(16):
                    detailed_rows.append({
                        "outer_fold": fold,
                        "sample_index": sample_index,
                        "model": model,
                        "component": component + 1,
                        "true_coordinate": true_coordinates[local_index, component],
                        "predicted_coordinate": pred_coordinates[local_index, component],
                        "training_response_variance_ratio": pca.explained_variance_ratio_[component],
                    })
    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(args.output / "RESPONSE_PROGRAM_COORDINATES.csv", index=False)

    rows = []
    for (model, component), group in detailed.groupby(["model", "component"]):
        true = group["true_coordinate"].to_numpy()
        pred = group["predicted_coordinate"].to_numpy()
        denominator = float(np.square(true).sum())
        true_variance = float(np.var(true))
        pred_variance = float(np.var(pred))
        rows.append({
            "row_type": "coordinate",
            "model": model,
            "K": component,
            "coordinate": component,
            "OOF_R2": 1 - float(np.square(true - pred).sum()) / denominator,
            "Pearson": correlation(true, pred),
            "sign_agreement": float(np.mean(np.sign(true) == np.sign(pred))),
            "predicted_to_true_variance": pred_variance / true_variance if true_variance else float("nan"),
            "truth_replicate_reliability": float(truth_reliability.get(component, np.nan)),
        })
    coordinate = pd.DataFrame(rows)
    summary_rows = []
    for model, model_group in detailed.groupby("model"):
        for k in [4, 8, 16]:
            group = model_group.loc[model_group["component"] <= k]
            true = group["true_coordinate"].to_numpy()
            pred = group["predicted_coordinate"].to_numpy()
            denominator = float(np.square(true).sum())
            summary_rows.append({
                "row_type": "variance_weighted_K",
                "model": model,
                "K": k,
                "coordinate": np.nan,
                "OOF_R2": 1 - float(np.square(true - pred).sum()) / denominator,
                "Pearson": correlation(true, pred),
                "sign_agreement": float(np.mean(np.sign(true) == np.sign(pred))),
                "predicted_to_true_variance": float(np.var(pred) / np.var(true)),
                "truth_replicate_reliability": np.nan,
            })
    results = pd.concat([coordinate, pd.DataFrame(summary_rows)], ignore_index=True)
    results.to_csv(args.output / "RESPONSE_PROGRAM_RESULTS.csv", index=False)
    print(results.loc[results["row_type"] == "variance_weighted_K"].to_string(index=False))


if __name__ == "__main__":
    main()
