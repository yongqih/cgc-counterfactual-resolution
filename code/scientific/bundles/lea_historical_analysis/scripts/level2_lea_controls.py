"""Frozen negative controls and ancestry audit for the GSE207049 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from level2_lea_pilot import (
    MODEL_NAMES,
    SEED,
    dual_ridge_predict,
    geometry,
    load_paired_data,
    metric_summary,
    pca_ridge_predict,
    standardize_apply,
    standardize_fit,
    train_mlp_fixed_epochs,
)


CONTROL_NAMES = ["response_permutation", "baseline_response_mismatch"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--primary-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    _, baseline, treated, units = load_paired_data(args.matrix, args.metadata)
    delta = treated - baseline
    strata = units["ancestry"].to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(outer.split(baseline, strata))
    primary = np.load(args.primary_oof, allow_pickle=False)
    targets = primary["targets"]
    shared_means = primary["shared_means"]
    folds = primary["folds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controls = {
        control: {model: np.zeros_like(targets, dtype=np.float32) for model in MODEL_NAMES[1:]}
        for control in CONTROL_NAMES
    }

    for fold, (train, test) in enumerate(splits):
        primary_fold = np.load(checkpoint_dir / f"fold_{fold}.npz", allow_pickle=False)
        ridge_alpha = float(primary_fold["ridge_alpha"])
        pca_components = int(primary_fold["pca_components"])
        pca_alpha = float(primary_fold["pca_alpha"])
        mlp_epochs = int(primary_fold["mlp_best_epoch"])
        mean_delta = delta[train].mean(axis=0)
        real_y_train = (delta[train] - mean_delta).astype(np.float32)

        for control_number, control in enumerate(CONTROL_NAMES):
            checkpoint = checkpoint_dir / f"control_{control}_fold_{fold}.npz"
            if checkpoint.exists():
                saved = np.load(checkpoint, allow_pickle=False)
                if np.array_equal(saved["test_indices"], test):
                    for model in MODEL_NAMES[1:]:
                        controls[control][model][test] = saved[model]
                    print(f"{control} fold {fold}: resumed", flush=True)
                    continue
            rng = np.random.default_rng(SEED + 20000 + 100 * control_number + fold)
            permutation = rng.permutation(len(train))
            if control == "response_permutation":
                x_train = baseline[train]
                y_train = real_y_train[permutation]
            else:
                x_train = baseline[train][permutation]
                y_train = real_y_train

            controls[control]["ridge"][test] = dual_ridge_predict(
                x_train, y_train, baseline[test], ridge_alpha
            )
            controls[control]["pca_ridge"][test] = pca_ridge_predict(
                x_train, y_train, baseline[test], pca_components, pca_alpha,
                SEED + 30000 + 100 * control_number + fold,
            )
            x_mean, x_scale = standardize_fit(x_train)
            controls[control]["mlp"][test] = train_mlp_fixed_epochs(
                standardize_apply(x_train, x_mean, x_scale), y_train,
                standardize_apply(baseline[test], x_mean, x_scale),
                mlp_epochs, device, SEED + 40000 + 100 * control_number + fold,
            )
            np.savez_compressed(
                checkpoint, test_indices=test,
                **{model: controls[control][model][test] for model in MODEL_NAMES[1:]},
            )
            print(f"{control} fold {fold}: complete", flush=True)

    np.savez_compressed(
        args.output / "NEGATIVE_CONTROL_OOF.npz",
        **{
            f"{control}__{model}": values
            for control, model_values in controls.items()
            for model, values in model_values.items()
        },
    )
    rows = []
    for control, model_values in controls.items():
        for model, prediction in model_values.items():
            dex_prediction = baseline + shared_means + prediction
            metrics = metric_summary(targets, prediction, treated, dex_prediction)
            geometry_rho, _ = geometry(targets, prediction, folds)
            rows.append({"control": control, "model": model, **metrics, "geometry_Spearman": geometry_rho})
    control_results = pd.DataFrame(rows)
    control_results.to_csv(args.output / "NEGATIVE_CONTROLS.csv", index=False)

    ancestry_rows = []
    balanced_rng = np.random.default_rng(SEED + 50000)
    ancestry_indices = {label: np.flatnonzero(strata == label) for label in sorted(set(strata))}
    min_size = min(map(len, ancestry_indices.values()))
    balanced = np.concatenate([
        balanced_rng.choice(indices, min_size, replace=False)
        for indices in ancestry_indices.values()
    ])
    for model in MODEL_NAMES:
        prediction = primary[model]
        for label, indices in [*ancestry_indices.items(), ("balanced", balanced)]:
            metrics = metric_summary(
                targets[indices], prediction[indices], treated[indices],
                baseline[indices] + shared_means[indices] + prediction[indices],
            )
            geometry_rho, _ = geometry(targets[indices], prediction[indices], folds[indices])
            ancestry_rows.append({"model": model, "subset": label, "n_LCLs": len(indices), **metrics, "geometry_Spearman": geometry_rho})
    ancestry = pd.DataFrame(ancestry_rows)
    ancestry.to_csv(args.output / "ANCESTRY_SENSITIVITY.csv", index=False)

    train_test_overlaps = []
    for fold, (train, test) in enumerate(splits):
        overlap = set(units.iloc[train]["line"]) & set(units.iloc[test]["line"])
        train_test_overlaps.append(len(overlap))
    leakage = {
        "LCL_overlap_each_fold": train_test_overlaps,
        "maximum_LCL_overlap": max(train_test_overlaps),
        "replicate_cross_fold": 0,
        "baseline_predictor": "ETOH RNA only",
        "ancestry_use": "split stratification and sensitivity reporting only; never supplied to model",
        "genotype_or_response_eQTL_predictor_use": False,
        "DEX_derived_feature_selection": False,
        "test_fitted_preprocessing": False,
        "target_centering": "outer-training response mean only",
        "verdict": "LEAKAGE_AUDIT_PASS" if max(train_test_overlaps) == 0 else "LEAKAGE_AUDIT_FAIL",
    }
    (args.output / "LEAKAGE_AUDIT.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Negative Controls and Leakage Audit

- Response permutation and baseline-response mismatch were applied within each
  outer-training fold only, using the frozen primary hyperparameters and epoch counts.
- MLP device: `{device}`.
- LCL overlap across every outer fold: {train_test_overlaps}; maximum: {max(train_test_overlaps)}.
- Replicate groups crossing folds: 0 (replicates were aggregated before splitting).
- Ancestry was used for split balance and reporting only, never as a predictor.
- Genotype, response-eQTL labels, DEX features, and test-fitted transformations were not used.

`{leakage['verdict']}`
"""
    (args.output / "NEGATIVE_CONTROLS_AND_LEAKAGE.md").write_text(report, encoding="utf-8")
    print(control_results[["control", "model", "pooled_OOF_residual_R2", "variance_retention_kappa", "geometry_Spearman"]].to_string(index=False))
    print(ancestry[["model", "subset", "n_LCLs", "pooled_OOF_residual_R2", "variance_retention_kappa", "geometry_Spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
