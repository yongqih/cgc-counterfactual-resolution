"""Project frozen permutation/mismatch predictions into fold-local response programs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-oof", type=Path, required=True)
    parser.add_argument("--pilot-controls", type=Path, required=True)
    parser.add_argument("--full-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = np.load(args.pilot_oof, allow_pickle=False)
    pilot_controls = np.load(args.pilot_controls, allow_pickle=False)
    full = np.load(args.full_oof, allow_pickle=False)
    delta = pilot["treated"] - pilot["baseline"]
    targets = pilot["targets"]
    folds = pilot["folds"].astype(int)
    predictions = {}
    for control in ["response_permutation", "baseline_response_mismatch"]:
        for model in ["ridge", "pca_ridge", "mlp"]:
            predictions[(control, "pilot_mlp" if model == "mlp" else model)] = pilot_controls[f"{control}__{model}"]
        for model in ["rbf_kernel_ridge", "hist_gradient_boosting", "deep_residual_mlp"]:
            predictions[(control, model)] = full[f"{control}__{model}"]

    stored: dict[tuple[str, str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        mean_delta = delta[train].mean(axis=0)
        y_train = delta[train] - mean_delta
        pca = PCA(n_components=16, svd_solver="randomized", random_state=207049 + fold)
        pca.fit(y_train)
        true = pca.transform(targets[test])
        for (control, model), prediction in predictions.items():
            pred = pca.transform(prediction[test])
            for component in range(16):
                stored.setdefault((control, model, component + 1), []).append(
                    (true[:, component], pred[:, component])
                )
    rows = []
    for (control, model, component), values in stored.items():
        true = np.concatenate([item[0] for item in values])
        pred = np.concatenate([item[1] for item in values])
        rows.append({
            "row_type": "coordinate", "control": control, "model": model,
            "K": component, "coordinate": component,
            "OOF_R2": 1 - np.square(true - pred).sum() / np.square(true).sum(),
        })
    coordinate = pd.DataFrame(rows)
    summaries = []
    for (control, model), group in coordinate.groupby(["control", "model"]):
        for k in [4, 8, 16]:
            values = []
            for component in range(1, k + 1):
                component_values = stored[(control, model, component)]
                true = np.concatenate([item[0] for item in component_values])
                pred = np.concatenate([item[1] for item in component_values])
                values.append((true, pred))
            true = np.concatenate([item[0] for item in values])
            pred = np.concatenate([item[1] for item in values])
            summaries.append({
                "row_type": "variance_weighted_K", "control": control,
                "model": model, "K": k, "coordinate": np.nan,
                "OOF_R2": 1 - np.square(true - pred).sum() / np.square(true).sum(),
            })
    results = pd.concat([coordinate, pd.DataFrame(summaries)], ignore_index=True)
    results.to_csv(args.output / "RESPONSE_PROGRAM_NEGATIVE_CONTROLS.csv", index=False)
    print(results.loc[results["row_type"] == "variance_weighted_K"].to_string(index=False))


if __name__ == "__main__":
    main()
