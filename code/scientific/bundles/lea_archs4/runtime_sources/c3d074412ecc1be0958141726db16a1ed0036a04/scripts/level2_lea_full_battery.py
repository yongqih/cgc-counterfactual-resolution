"""Run the frozen transparent full model battery without rerunning the pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from igc_virtual_cell.level2_full import (
    BoostingSetting,
    DeepSetting,
    fit_boosting_program_decoder,
    fit_deep_outer,
    pooled_r2,
    rbf_kernel_ridge_predict,
    tune_boosting,
    tune_deep,
    tune_rbf_kernel,
)
from igc_virtual_cell.level2_lea import pairing_table


PILOT_NAMES = {"null": "null", "ridge": "ridge", "pca_ridge": "pca_ridge", "pilot_mlp": "mlp"}
NEW_NAMES = ["rbf_kernel_ridge", "hist_gradient_boosting", "deep_residual_mlp"]


def vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered_true = y_true - y_true.mean(axis=1, keepdims=True)
    centered_pred = y_pred - y_pred.mean(axis=1, keepdims=True)
    pearson_denominator = np.sqrt(
        np.square(centered_true).sum(axis=1) * np.square(centered_pred).sum(axis=1)
    )
    pearson = np.divide(
        (centered_true * centered_pred).sum(axis=1), pearson_denominator,
        out=np.full(len(y_true), np.nan), where=pearson_denominator > 0,
    )
    cosine_denominator = np.sqrt(
        np.square(y_true).sum(axis=1) * np.square(y_pred).sum(axis=1)
    )
    cosine = np.divide(
        (y_true * y_pred).sum(axis=1), cosine_denominator,
        out=np.full(len(y_true), np.nan), where=cosine_denominator > 0,
    )
    return pearson, cosine


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: np.ndarray,
    treated: np.ndarray,
    shared_mean: np.ndarray,
) -> dict[str, float]:
    pearson, cosine = vector_metrics(y_true, y_pred)
    finite_pearson = pearson[np.isfinite(pearson)]
    finite_cosine = cosine[np.isfinite(cosine)]
    gene_denominator = np.square(y_true, dtype=np.float64).sum(axis=0)
    gene_r2 = 1 - np.divide(
        np.square(y_true - y_pred, dtype=np.float64).sum(axis=0),
        gene_denominator, out=np.full(y_true.shape[1], np.nan), where=gene_denominator > 0,
    )
    kappa = float(np.var(y_pred) / np.var(y_true))
    predicted_treated = baseline + shared_mean + y_pred
    state_pearson, _ = vector_metrics(treated, predicted_treated)
    return {
        "pooled_OOF_residual_R2": pooled_r2(y_true, y_pred),
        "median_per_gene_R2": float(np.nanmedian(gene_r2)),
        "mean_per_gene_R2": float(np.nanmean(gene_r2)),
        "fraction_genes_R2_gt_0": float(np.nanmean(gene_r2 > 0)),
        "fraction_genes_R2_gt_0.05": float(np.nanmean(gene_r2 > 0.05)),
        "fraction_genes_R2_gt_0.1": float(np.nanmean(gene_r2 > 0.1)),
        "median_per_LCL_Pearson": float(np.median(finite_pearson)) if len(finite_pearson) else float("nan"),
        "median_per_LCL_cosine": float(np.median(finite_cosine)) if len(finite_cosine) else float("nan"),
        "variance_retention_kappa": kappa,
        "amplitude_retention_sqrt_kappa": float(np.sqrt(max(kappa, 0))),
        "median_absolute_state_Pearson": float(np.nanmedian(state_pearson)),
        "absolute_state_pooled_R2": float(
            1 - np.square(treated - predicted_treated, dtype=np.float64).sum()
            / np.square(treated - treated.mean(axis=0), dtype=np.float64).sum()
        ),
    }


def bootstrap_summary(
    y_true: np.ndarray, predictions: dict[str, np.ndarray], draws: int = 1000
) -> pd.DataFrame:
    rng = np.random.default_rng(217049)
    n, genes = y_true.shape
    true_sums = y_true.sum(axis=1, dtype=np.float64)
    true_sumsq = np.square(y_true, dtype=np.float64).sum(axis=1)
    cached = {}
    for name, prediction in predictions.items():
        pearson, _ = vector_metrics(y_true, prediction)
        cached[name] = {
            "sse": np.square(y_true - prediction, dtype=np.float64).sum(axis=1),
            "sum": prediction.sum(axis=1, dtype=np.float64),
            "sumsq": np.square(prediction, dtype=np.float64).sum(axis=1),
            "pearson": pearson,
        }
    rows = []
    for draw in range(draws):
        indices = rng.integers(0, n, n)
        true_mean = float(true_sums[indices].sum()) / (n * genes)
        true_var = float(true_sumsq[indices].sum()) / (n * genes) - true_mean**2
        denominator = float(true_sumsq[indices].sum())
        for name, values in cached.items():
            pred_mean = float(values["sum"][indices].sum()) / (n * genes)
            pred_var = float(values["sumsq"][indices].sum()) / (n * genes) - pred_mean**2
            line_values = values["pearson"][indices]
            finite = line_values[np.isfinite(line_values)]
            rows.append({
                "draw": draw,
                "model": name,
                "pooled_OOF_residual_R2": 1 - float(values["sse"][indices].sum()) / denominator,
                "variance_retention_kappa": pred_var / true_var,
                "median_per_LCL_Pearson": float(np.median(finite)) if len(finite) else float("nan"),
            })
    return pd.DataFrame(rows)


def paired_units(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, sep="\t")
    _, pairing = pairing_table(metadata)
    return pairing.loc[pairing["paired"]].sort_values("line").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-oof", type=Path, required=True)
    parser.add_argument("--pilot-checkpoints", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "full_checkpoints"
    checkpoints.mkdir(exist_ok=True)

    pilot = np.load(args.pilot_oof, allow_pickle=False)
    baseline = pilot["baseline"].astype(np.float32)
    treated = pilot["treated"].astype(np.float32)
    delta = treated - baseline
    targets = pilot["targets"].astype(np.float32)
    shared_means = pilot["shared_means"].astype(np.float32)
    folds = pilot["folds"].astype(int)
    units = paired_units(args.metadata)
    if len(units) != len(baseline):
        raise AssertionError("Paired-unit count does not match frozen OOF arrays")
    units["outer_fold"] = folds
    units.to_csv(args.output / "FULL_BATTERY_UNITS.csv", index=False)
    for fold in range(5):
        pilot_fold = np.load(args.pilot_checkpoints / f"fold_{fold}.npz", allow_pickle=False)
        if not np.array_equal(np.flatnonzero(folds == fold), pilot_fold["test_indices"]):
            raise AssertionError(f"Frozen fold {fold} differs from pilot checkpoint")

    strata = units["pop2"].to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel_config = config["kernel_ridge"]
    boost_settings = [BoostingSetting(**item) for item in config["boosting"]["candidate_settings"]]
    deep_settings = [DeepSetting(**item) for item in config["deep_residual_mlp"]["candidate_settings"]]
    deep_seeds = config["deep_residual_mlp"]["final_seeds"]

    predictions = {name: np.zeros_like(targets) for name in NEW_NAMES}
    controls = {
        control: {name: np.zeros_like(targets) for name in NEW_NAMES}
        for control in ["response_permutation", "baseline_response_mismatch"]
    }
    deep_seed_predictions = np.zeros((len(deep_seeds), *targets.shape), dtype=np.float32)
    selection_rows = []
    tuning_tables = []
    started = time.perf_counter()

    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        checkpoint = checkpoints / f"full_fold_{fold}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint, allow_pickle=False)
            if not np.array_equal(saved["test_indices"], test):
                raise AssertionError(f"Full checkpoint fold {fold} index mismatch")
            for name in NEW_NAMES:
                predictions[name][test] = saved[name]
                for control in controls:
                    controls[control][name][test] = saved[f"{control}__{name}"]
            deep_seed_predictions[:, test] = saved["deep_seed_predictions"]
            selection_rows.append(json.loads(str(saved["selection_json"])))
            print(f"full fold {fold}: resumed", flush=True)
            continue

        mean_delta = delta[train].mean(axis=0)
        y_train = (delta[train] - mean_delta).astype(np.float32)
        if not np.allclose(targets[test], delta[test] - mean_delta, atol=2e-6):
            raise AssertionError(f"Frozen target centering mismatch in fold {fold}")

        gamma_multiplier, kernel_alpha, kernel_tuning = tune_rbf_kernel(
            baseline[train], delta[train], strata[train],
            kernel_config["gamma_multipliers_of_training_median_squared_distance"],
            kernel_config["alphas"], config["seed"] + fold,
        )
        kernel_tuning["outer_fold"] = fold
        kernel_tuning["model"] = "rbf_kernel_ridge"
        tuning_tables.append(kernel_tuning)
        predictions["rbf_kernel_ridge"][test], fitted_gamma = rbf_kernel_ridge_predict(
            baseline[train], y_train, baseline[test], gamma_multiplier, kernel_alpha
        )

        boost_index, boost_tuning = tune_boosting(
            baseline[train], delta[train], strata[train], boost_settings,
            config["boosting"]["baseline_pca_components"],
            config["boosting"]["response_pca_components"],
            config["seed"] + 100 + fold,
        )
        boost_tuning["outer_fold"] = fold
        boost_tuning["model"] = "hist_gradient_boosting"
        tuning_tables.append(boost_tuning)
        predictions["hist_gradient_boosting"][test] = fit_boosting_program_decoder(
            baseline[train], y_train, baseline[test], boost_settings[boost_index],
            config["boosting"]["baseline_pca_components"],
            config["boosting"]["response_pca_components"],
            config["seed"] + 200 + fold,
        )

        deep_index, deep_epochs, deep_history = tune_deep(
            baseline[train], delta[train], strata[train], deep_settings,
            config["deep_residual_mlp"]["baseline_pca_components"],
            device, config["seed"] + 300 + fold,
        )
        deep_history["outer_fold"] = fold
        deep_history["model"] = "deep_residual_mlp"
        deep_history.to_csv(checkpoints / f"deep_tuning_history_fold_{fold}.csv", index=False)
        deep_prediction, seed_prediction = fit_deep_outer(
            baseline[train], y_train, baseline[test], deep_settings[deep_index],
            config["deep_residual_mlp"]["baseline_pca_components"], deep_epochs,
            deep_seeds, device, fold,
        )
        predictions["deep_residual_mlp"][test] = deep_prediction
        deep_seed_predictions[:, test] = seed_prediction

        for control_index, control in enumerate(controls):
            rng = np.random.default_rng(config["seed"] + 10000 + control_index * 100 + fold)
            permutation = rng.permutation(len(train))
            if control == "response_permutation":
                control_x = baseline[train]
                control_y = y_train[permutation]
            else:
                control_x = baseline[train][permutation]
                control_y = y_train
            controls[control]["rbf_kernel_ridge"][test], _ = rbf_kernel_ridge_predict(
                control_x, control_y, baseline[test], gamma_multiplier, kernel_alpha
            )
            controls[control]["hist_gradient_boosting"][test] = fit_boosting_program_decoder(
                control_x, control_y, baseline[test], boost_settings[boost_index],
                config["boosting"]["baseline_pca_components"],
                config["boosting"]["response_pca_components"],
                config["seed"] + 20000 + control_index * 100 + fold,
            )
            controls[control]["deep_residual_mlp"][test], _ = fit_deep_outer(
                control_x, control_y, baseline[test], deep_settings[deep_index],
                config["deep_residual_mlp"]["baseline_pca_components"], deep_epochs,
                [deep_seeds[0]], device, fold + 20 * (control_index + 1),
            )

        selection = {
            "outer_fold": fold,
            "kernel_gamma_multiplier": gamma_multiplier,
            "kernel_fitted_gamma": fitted_gamma,
            "kernel_alpha": kernel_alpha,
            "boost_setting_index": boost_index,
            **{f"boost_{key}": value for key, value in boost_settings[boost_index].__dict__.items()},
            "deep_setting_index": deep_index,
            **{f"deep_{key}": value for key, value in deep_settings[deep_index].__dict__.items()},
            "deep_epochs": deep_epochs,
            "device": str(device),
        }
        selection_rows.append(selection)
        save_items = {
            "test_indices": test,
            "deep_seed_predictions": seed_prediction,
            "selection_json": np.asarray(json.dumps(selection)),
            **{name: predictions[name][test] for name in NEW_NAMES},
        }
        for control, values in controls.items():
            for name in NEW_NAMES:
                save_items[f"{control}__{name}"] = values[name][test]
        np.savez_compressed(checkpoint, **save_items)
        print(
            f"full fold {fold}: kernel(g={gamma_multiplier:g},a={kernel_alpha:g}); "
            f"boost={boost_index}; deep={deep_index},epochs={deep_epochs}", flush=True,
        )

    if tuning_tables:
        pd.concat(tuning_tables, ignore_index=True, sort=False).to_csv(
            args.output / "FULL_MODEL_INNER_TUNING.csv", index=False
        )
    pd.DataFrame(selection_rows).sort_values("outer_fold").to_csv(
        args.output / "FULL_MODEL_SELECTED_HYPERPARAMETERS.csv", index=False
    )
    np.savez_compressed(
        args.output / "FULL_BATTERY_OOF.npz",
        targets=targets, folds=folds, deep_seed_predictions=deep_seed_predictions,
        **predictions,
        **{
            f"{control}__{name}": value
            for control, model_values in controls.items()
            for name, value in model_values.items()
        },
    )

    all_predictions = {name: pilot[key] for name, key in PILOT_NAMES.items()}
    all_predictions.update(predictions)
    result_rows = []
    gene_rows = []
    lcl_rows = []
    for name, prediction in all_predictions.items():
        result_rows.append({"model": name, **summarize(
            targets, prediction, baseline, treated, shared_means
        )})
        denominator = np.square(targets, dtype=np.float64).sum(axis=0)
        r2 = 1 - np.divide(
            np.square(targets - prediction, dtype=np.float64).sum(axis=0), denominator,
            out=np.full(targets.shape[1], np.nan), where=denominator > 0,
        )
        gene_rows.append(pd.DataFrame({"gene_row": np.arange(len(r2)), "model": name, "OOF_R2": r2}))
        pearson, cosine = vector_metrics(targets, prediction)
        lcl_rows.append(pd.DataFrame({
            "line": units["line"], "outer_fold": folds, "model": name,
            "Pearson": pearson, "cosine": cosine,
        }))
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "FULL_MODEL_BATTERY_RESULTS.csv", index=False)
    results[[
        "model", "pooled_OOF_residual_R2", "variance_retention_kappa",
        "amplitude_retention_sqrt_kappa",
    ]].to_csv(args.output / "VARIANCE_COMPRESSION_RESULTS.csv", index=False)
    pd.concat(gene_rows, ignore_index=True).to_csv(args.output / "FULL_PER_GENE_R2.csv", index=False)
    pd.concat(lcl_rows, ignore_index=True).to_csv(args.output / "FULL_PER_LCL_METRICS.csv", index=False)

    bootstrap = bootstrap_summary(targets, all_predictions)
    bootstrap.to_csv(args.output / "FULL_MODEL_BOOTSTRAP.csv", index=False)
    ci_rows = []
    for (model_name, metric), values in bootstrap.melt(
        id_vars=["draw", "model"], var_name="metric", value_name="value"
    ).groupby(["model", "metric"]):
        ci_rows.append({
            "model": model_name, "metric": metric,
            "CI_low": values["value"].quantile(0.025),
            "CI_high": values["value"].quantile(0.975),
        })
    pd.DataFrame(ci_rows).to_csv(args.output / "FULL_MODEL_BOOTSTRAP_CI.csv", index=False)

    control_rows = []
    for control, model_values in controls.items():
        for name, prediction in model_values.items():
            control_rows.append({"control": control, "model": name, **summarize(
                targets, prediction, baseline, treated, shared_means
            )})
    pd.DataFrame(control_rows).to_csv(args.output / "FULL_MODEL_NEGATIVE_CONTROLS.csv", index=False)

    seed_rows = []
    for seed_index, seed in enumerate(deep_seeds):
        seed_rows.append({
            "seed": seed,
            **summarize(
                targets, deep_seed_predictions[seed_index], baseline, treated, shared_means
            ),
        })
    pd.DataFrame(seed_rows).to_csv(args.output / "DEEP_MLP_SEED_VARIABILITY.csv", index=False)
    print(results.to_string(index=False), flush=True)
    print(f"runtime_minutes={(time.perf_counter() - started) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()
