from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from igc_virtual_cell.level2_full import (
    BoostingSetting,
    DeepSetting,
    fit_boosting_program_decoder,
    fit_deep_outer,
    rbf_kernel_ridge_predict,
)
from scripts.level2_lea_pilot import (
    dual_ridge_predict,
    pca_ridge_predict,
    standardize_apply,
    standardize_fit,
    train_mlp_fixed_epochs,
)


ALL_MODELS = (
    "ridge",
    "pca_ridge",
    "pilot_mlp",
    "rbf_kernel_ridge",
    "hist_gradient_boosting",
    "deep_residual_mlp",
)
REAL_RESULT_FILE = "results/level2_lea/FULL_MODEL_BATTERY_RESULTS.csv"
PILOT_SELECTION_FILE = "results/level2_lea/MODEL_SELECTED_HYPERPARAMETERS.csv"
FULL_SELECTION_FILE = "results/level2_lea/FULL_MODEL_SELECTED_HYPERPARAMETERS.csv"
UNIT_FILE = "results/level2_lea/FULL_BATTERY_UNITS.csv"
TRUTH_FILE = "results/level2_lea/TRUTH_RELIABILITY.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_config(root: Path) -> dict[str, Any]:
    return json.loads((root / "configs/lea_semisynthetic_control_frozen.json").read_text(encoding="utf-8"))


def source_npz(source_root: Path) -> Path:
    return source_root / "results/level2_lea/LEVEL2_OOF_FROZEN.npz"


def validate_sources(root: Path, source_root: Path) -> dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    config = load_config(root)
    required = [
        root / REAL_RESULT_FILE,
        root / PILOT_SELECTION_FILE,
        root / FULL_SELECTION_FILE,
        root / UNIT_FILE,
        root / TRUTH_FILE,
        source_npz(source_root),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"LEA_FROZEN_PIPELINE_MISMATCH: missing {missing}")
    authority_head = git_value(root, "rev-parse", config["authority_branch"])
    if authority_head != config["authority_head"]:
        raise RuntimeError(
            f"LEA_FROZEN_PIPELINE_MISMATCH: authority {authority_head} != {config['authority_head']}"
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            config["real_result_commit"],
            config["authority_head"],
        ],
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError("LEA_FROZEN_PIPELINE_MISMATCH: result commit not in authority")

    real = pd.read_csv(root / REAL_RESULT_FILE).set_index("model")
    checks = []
    for model, expected in config["real_full_gene_oof_r2"].items():
        observed = float(real.loc[model, "pooled_OOF_residual_R2"])
        checks.append(
            {
                "model": model,
                "observed": observed,
                "expected": float(expected),
                "absolute_difference": abs(observed - float(expected)),
                "passed": abs(observed - float(expected)) <= 1e-12,
            }
        )
    check_frame = pd.DataFrame(checks)
    if not bool(check_frame["passed"].all()):
        raise RuntimeError("LEA_FROZEN_PIPELINE_MISMATCH: real R2 mismatch")

    frozen = np.load(source_npz(source_root), allow_pickle=False)
    baseline = np.asarray(frozen["baseline"], dtype=np.float32)
    folds = np.asarray(frozen["folds"], dtype=np.int64)
    targets = np.asarray(frozen["targets"], dtype=np.float32)
    units = pd.read_csv(root / UNIT_FILE)
    if baseline.shape != (342, 10_157) or targets.shape != baseline.shape:
        raise RuntimeError(f"LEA_FROZEN_PIPELINE_MISMATCH: array shapes {baseline.shape}/{targets.shape}")
    if not np.array_equal(folds, units["outer_fold"].to_numpy(dtype=np.int64)):
        raise RuntimeError("LEA_FROZEN_PIPELINE_MISMATCH: fold vector mismatch")
    if tuple(np.bincount(folds, minlength=5)) != (69, 69, 68, 68, 68):
        raise RuntimeError("LEA_FROZEN_PIPELINE_MISMATCH: fold counts mismatch")
    truth = json.loads((root / TRUTH_FILE).read_text(encoding="utf-8"))
    reliability = float(truth["global_residual_replicate_correlation"])
    if abs(reliability - 0.7442542885406368) > 1e-12:
        raise RuntimeError("LEA_FROZEN_PIPELINE_MISMATCH: truth reliability mismatch")
    return {
        "config": config,
        "source_npz": source_npz(source_root),
        "source_npz_sha256": sha256(source_npz(source_root)),
        "baseline": baseline,
        "folds": folds,
        "real_checks": check_frame,
        "authority_head": authority_head,
        "real_reliability": reliability,
        "real_reliability_ci": truth["bootstrap_95_CI"],
        "pilot_selection": pd.read_csv(root / PILOT_SELECTION_FILE).sort_values("outer_fold"),
        "full_selection": pd.read_csv(root / FULL_SELECTION_FILE).sort_values("outer_fold"),
    }


def write_input_reconciliation(root: Path, source_root: Path) -> Path:
    audit = validate_sources(root, source_root)
    config = audit["config"]
    checks = audit["real_checks"]
    table = [
        "| model | observed real OOF R2 | frozen expected | absolute difference | pass |",
        "|:--|--:|--:|--:|:--:|",
    ]
    for row in checks.itertuples(index=False):
        table.append(
            f"| {row.model} | {row.observed:.15f} | {row.expected:.15f} | "
            f"{row.absolute_difference:.3e} | {row.passed} |"
        )
    fold_counts = tuple(map(int, np.bincount(audit["folds"], minlength=5)))
    text = f"""# Lea semi-synthetic input reconciliation

Status: **LEA_FROZEN_PIPELINE_MATCH**

- Authority branch: `{config['authority_branch']}`.
- Authority HEAD: `{audit['authority_head']}`.
- Frozen result commit: `{config['real_result_commit']}`.
- Frozen OOF source: `{audit['source_npz']}`.
- Frozen OOF SHA-256: `{audit['source_npz_sha256']}`.
- Baseline shape: `{audit['baseline'].shape}`.
- Biological-LCL outer-fold counts: `{fold_counts}`.
- Real replicate reliability: `{audit['real_reliability']:.15f}`; 95% CI `{audit['real_reliability_ci']}`.
- Six-model numerical tolerance: `1e-12`.

## Frozen real-result reconciliation

{chr(10).join(table)}

All six values, the exact 342 by 10,157 baseline matrix, and the five frozen LCL-disjoint folds passed. No synthetic outcome was generated or inspected during this gate.
"""
    out = root / "results/lea_semisynthetic_identifiability_control"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "SEMISYNTHETIC_INPUT_RECONCILIATION.md"
    path.write_text(text, encoding="utf-8")
    return path


def _orthonormal_columns(rng: np.random.Generator, rows: int, rank: int) -> np.ndarray:
    raw = rng.normal(size=(rows, rank))
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs[None, :]).astype(np.float32)


@dataclass(frozen=True)
class SeedComponents:
    z_x: np.ndarray
    z_h: np.ndarray
    w_y: np.ndarray


def generate_seed_components(
    baseline: np.ndarray, seed: int, rank: int
) -> SeedComponents:
    mean, scale = standardize_fit(baseline)
    standardized = standardize_apply(baseline, mean, scale)
    w_x = _orthonormal_columns(np.random.default_rng(seed + 100_000), baseline.shape[1], rank)
    z_x = standardized @ w_x
    z_x = ((z_x - z_x.mean(axis=0)) / z_x.std(axis=0)).astype(np.float32)
    z_h = np.random.default_rng(seed + 200_000).normal(size=(len(baseline), rank)).astype(np.float32)
    w_y = _orthonormal_columns(
        np.random.default_rng(seed + 300_000), baseline.shape[1], rank
    ).T
    return SeedComponents(z_x=z_x, z_h=z_h, w_y=w_y)


def generate_observed_replicates(
    components: SeedComponents,
    lambda_value: float,
    seed: int,
    lambda_index: int,
    target_reliability: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    z = (
        math.sqrt(lambda_value) * components.z_x
        + math.sqrt(1.0 - lambda_value) * components.z_h
    )
    signal = (z @ components.w_y).astype(np.float32)
    signal_variance = float(np.var(signal, dtype=np.float64))
    noise_variance = signal_variance * (1.0 - target_reliability) / target_reliability
    sigma = math.sqrt(noise_variance)
    shape = signal.shape
    noise_a = np.random.default_rng(seed + 400_000 + 10 * lambda_index).normal(
        0.0, sigma, size=shape
    ).astype(np.float32)
    noise_b = np.random.default_rng(seed + 400_001 + 10 * lambda_index).normal(
        0.0, sigma, size=shape
    ).astype(np.float32)
    replicate_a = signal + noise_a
    replicate_b = signal + noise_b
    a = replicate_a.ravel().astype(np.float64)
    b = replicate_b.ravel().astype(np.float64)
    reliability = float(np.corrcoef(a, b)[0, 1])
    return replicate_a, replicate_b, {
        "signal_variance": signal_variance,
        "noise_variance": noise_variance,
        "realized_reliability": reliability,
    }


def scheduled_models(config: dict[str, Any], lambda_value: float) -> list[str]:
    values = set(map(float, config["tier_2_lambdas"]))
    if float(lambda_value) in values:
        return list(config["tier_2_models"])
    return list(config["tier_1_models"])


def _selection(audit: dict[str, Any], fold: int) -> tuple[pd.Series, pd.Series]:
    pilot = audit["pilot_selection"].set_index("outer_fold").loc[fold]
    full = audit["full_selection"].set_index("outer_fold").loc[fold]
    return pilot, full


def fit_frozen_model(
    model: str,
    baseline: np.ndarray,
    response: np.ndarray,
    folds: np.ndarray,
    fold: int,
    audit: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.flatnonzero(folds != fold)
    test = np.flatnonzero(folds == fold)
    mean_response = response[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    y_train = (response[train] - mean_response).astype(np.float32)
    y_test = (response[test] - mean_response).astype(np.float32)
    pilot, full = _selection(audit, fold)
    if model == "ridge":
        prediction = dual_ridge_predict(
            baseline[train], y_train, baseline[test], float(pilot["ridge_alpha"])
        )
    elif model == "pca_ridge":
        prediction = pca_ridge_predict(
            baseline[train],
            y_train,
            baseline[test],
            int(pilot["pca_components"]),
            float(pilot["pca_ridge_alpha"]),
            207_049 + fold,
        )
    elif model == "pilot_mlp":
        mean, scale = standardize_fit(baseline[train])
        x_train = standardize_apply(baseline[train], mean, scale)
        x_test = standardize_apply(baseline[test], mean, scale)
        prediction = train_mlp_fixed_epochs(
            x_train,
            y_train,
            x_test,
            int(pilot["mlp_inner_selected_epoch"]),
            device,
            207_049 + fold + 1_000,
        )
    elif model == "rbf_kernel_ridge":
        prediction, _ = rbf_kernel_ridge_predict(
            baseline[train],
            y_train,
            baseline[test],
            float(full["kernel_gamma_multiplier"]),
            float(full["kernel_alpha"]),
        )
    elif model == "hist_gradient_boosting":
        setting = BoostingSetting(
            learning_rate=float(full["boost_learning_rate"]),
            max_iter=int(full["boost_max_iter"]),
            max_leaf_nodes=int(full["boost_max_leaf_nodes"]),
            l2_regularization=float(full["boost_l2_regularization"]),
        )
        prediction = fit_boosting_program_decoder(
            baseline[train],
            y_train,
            baseline[test],
            setting,
            baseline_components=100,
            response_components=16,
            seed=207_049 + 200 + fold,
        )
    elif model == "deep_residual_mlp":
        setting = DeepSetting(
            learning_rate=float(full["deep_learning_rate"]),
            weight_decay=float(full["deep_weight_decay"]),
        )
        prediction, _ = fit_deep_outer(
            baseline[train],
            y_train,
            baseline[test],
            setting,
            components=128,
            epochs=int(full["deep_epochs"]),
            seeds=[207_049, 207_050, 207_051],
            device=device,
            fold=fold,
        )
    else:
        raise ValueError(f"unknown frozen model {model}")
    return prediction.astype(np.float32), y_test, test


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _lambda_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def benchmark_runtime(root: Path, source_root: Path) -> Path:
    audit = validate_sources(root, source_root)
    config = audit["config"]
    rng = np.random.default_rng(991_207_049)
    response = rng.normal(size=audit["baseline"].shape).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timings = []
    for model in ALL_MODELS:
        start = time.perf_counter()
        prediction, target, _ = fit_frozen_model(
            model, audit["baseline"], response, audit["folds"], 0, audit, device
        )
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"response-free benchmark failed for {model}")
        timings.append({"model": model, "seconds_per_fold": time.perf_counter() - start})
        del prediction, target
    frame = pd.DataFrame(timings)
    fit_counts = {
        model: (
            5 * len(config["simulation_seeds"]) * len(config["tier_1_lambdas"])
            if model in config["tier_1_models"]
            else 5 * len(config["simulation_seeds"]) * len(config["tier_2_lambdas"])
        )
        for model in ALL_MODELS
    }
    frame["planned_fold_fits"] = frame["model"].map(fit_counts)
    frame["projected_seconds"] = frame["seconds_per_fold"] * frame["planned_fold_fits"]
    out = root / "results/lea_semisynthetic_identifiability_control"
    frame.to_csv(out / "SEMISYNTHETIC_RUNTIME_BENCHMARK.csv", index=False)
    text = "# Response-free runtime benchmark\n\n"
    text += f"- Device: `{device}`.\n"
    text += "- Target: independent Gaussian matched-shape matrix; no scientific R2 was computed or inspected.\n"
    text += f"- Planned fold fits: `{int(frame['planned_fold_fits'].sum())}`.\n"
    text += f"- Linear projected runtime: `{frame['projected_seconds'].sum()/60:.1f}` minutes.\n\n"
    text += "| model | seconds per fold | planned fold fits | projected seconds |\n"
    text += "|:--|--:|--:|--:|\n"
    for row in frame.itertuples(index=False):
        text += (
            f"| {row.model} | {row.seconds_per_fold:.3f} | {row.planned_fold_fits} | "
            f"{row.projected_seconds:.1f} |\n"
        )
    path = out / "SEMISYNTHETIC_RUNTIME_ESTIMATE.md"
    path.write_text(text, encoding="utf-8")
    return path


def _cache_path(out: Path, model: str, seed: int, lambda_value: float, fold: int) -> Path:
    return out / "_cache" / f"seed_{seed}" / f"lambda_{_lambda_tag(lambda_value)}" / model / f"fold_{fold}.json"


def _valid_cache(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if any(payload.get(key) != value for key, value in identity.items()):
        return None
    required = ("sse", "denominator", "test_count", "runtime_seconds")
    if not all(key in payload and np.isfinite(float(payload[key])) for key in required):
        return None
    return payload


def run_formal(root: Path, source_root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit = validate_sources(root, source_root)
    config = audit["config"]
    out = root / "results/lea_semisynthetic_identifiability_control"
    reconciliation = out / "SEMISYNTHETIC_INPUT_RECONCILIATION.md"
    if not reconciliation.is_file() or "LEA_FROZEN_PIPELINE_MATCH" not in reconciliation.read_text(encoding="utf-8"):
        raise RuntimeError("LEA_FROZEN_PIPELINE_MISMATCH: reconciliation gate absent")
    config_sha = sha256(root / "configs/lea_semisynthetic_control_frozen.json")
    source_sha = audit["source_npz_sha256"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = audit["baseline"]
    folds = audit["folds"]
    reliability_rows = []
    result_rows = []
    total = sum(
        len(scheduled_models(config, float(value)))
        for value in config["lambda_grid"]
    ) * len(config["simulation_seeds"])
    completed = 0
    started = time.perf_counter()
    for seed in config["simulation_seeds"]:
        components = generate_seed_components(baseline, int(seed), int(config["latent_rank"]))
        for lambda_index, lambda_value_raw in enumerate(config["lambda_grid"]):
            lambda_value = float(lambda_value_raw)
            replicate_a, replicate_b, calibration = generate_observed_replicates(
                components,
                lambda_value,
                int(seed),
                lambda_index,
                float(config["target_replicate_reliability"]),
            )
            error = abs(
                calibration["realized_reliability"]
                - float(config["target_replicate_reliability"])
            )
            reliability_rows.append(
                {
                    "seed": seed,
                    "lambda": lambda_value,
                    "latent_rank": config["latent_rank"],
                    **calibration,
                    "target_reliability": config["target_replicate_reliability"],
                    "absolute_error": error,
                    "noise_calibration_pass": error <= config["reliability_absolute_tolerance"],
                }
            )
            if error > config["reliability_absolute_tolerance"]:
                raise RuntimeError(
                    f"NOISE_CALIBRATION_FAIL seed={seed} lambda={lambda_value} reliability={calibration['realized_reliability']}"
                )
            del replicate_b
            for model in scheduled_models(config, lambda_value):
                fold_payloads = []
                model_started = time.perf_counter()
                for fold in range(5):
                    identity = {
                        "config_sha256": config_sha,
                        "source_npz_sha256": source_sha,
                        "model": model,
                        "seed": int(seed),
                        "lambda": lambda_value,
                        "fold": fold,
                        "latent_rank": int(config["latent_rank"]),
                    }
                    cache_path = _cache_path(out, model, int(seed), lambda_value, fold)
                    payload = _valid_cache(cache_path, identity)
                    if payload is None:
                        fold_started = time.perf_counter()
                        prediction, target, test = fit_frozen_model(
                            model, baseline, replicate_a, folds, fold, audit, device
                        )
                        payload = {
                            **identity,
                            "sse": float(np.square(target - prediction, dtype=np.float64).sum()),
                            "denominator": float(np.square(target, dtype=np.float64).sum()),
                            "test_count": int(len(test)),
                            "gene_count": int(target.shape[1]),
                            "runtime_seconds": time.perf_counter() - fold_started,
                            "device": str(device),
                        }
                        _atomic_json(cache_path, payload)
                    fold_payloads.append(payload)
                sse = sum(float(item["sse"]) for item in fold_payloads)
                denominator = sum(float(item["denominator"]) for item in fold_payloads)
                result_rows.append(
                    {
                        "model": model,
                        "lambda": lambda_value,
                        "seed": seed,
                        "latent_rank": config["latent_rank"],
                        "pooled_full_gene_oof_r2": 1.0 - sse / denominator,
                        "sse": sse,
                        "denominator": denominator,
                        "lcl_count": sum(int(item["test_count"]) for item in fold_payloads),
                        "gene_count": int(fold_payloads[0]["gene_count"]),
                        "fold_runtime_seconds": sum(float(item["runtime_seconds"]) for item in fold_payloads),
                        "wall_seconds_this_pass": time.perf_counter() - model_started,
                        "device": str(device),
                        "tier": "TIER_1_FULL_SWEEP" if model in config["tier_1_models"] else "TIER_2_ENDPOINT_CONFIRMATION",
                    }
                )
                completed += 1
                print(
                    f"semi-synthetic {completed}/{total} seed={seed} lambda={lambda_value:g} "
                    f"model={model} elapsed_min={(time.perf_counter()-started)/60:.1f}",
                    flush=True,
                )
            del replicate_a
    reliability = pd.DataFrame(reliability_rows)
    results = pd.DataFrame(result_rows)
    reliability.to_csv(out / "SEMISYNTHETIC_RELIABILITY.csv", index=False)
    results.to_csv(out / "SEMISYNTHETIC_FULL_GENE_R2.csv", index=False)
    return finalize(root, audit, reliability, results, device, time.perf_counter() - started)


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, lambda_value), group in results.groupby(["model", "lambda"], sort=False):
        values = group["pooled_full_gene_oof_r2"]
        rows.append(
            {
                "model": model,
                "lambda": lambda_value,
                "seed_count": len(group),
                "median_r2": float(values.median()),
                "mean_r2": float(values.mean()),
                "q25_r2": float(values.quantile(0.25)),
                "q75_r2": float(values.quantile(0.75)),
                "min_r2": float(values.min()),
                "max_r2": float(values.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "lambda"], kind="stable")


def adjudicate(results: pd.DataFrame, config: dict[str, Any]) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    sweep = _summary(results)
    lambda_zero = results[results["lambda"] == 0].pivot(index="seed", columns="model", values="pooled_full_gene_oof_r2")
    median_seedwise_best = float(lambda_zero.max(axis=1).median())
    rows = []
    qualifying = []
    any_positive = False
    for model in ALL_MODELS:
        model_sweep = sweep[sweep["model"] == model].set_index("lambda")
        zero = float(model_sweep.loc[0.0, "median_r2"])
        one = float(model_sweep.loc[1.0, "median_r2"])
        any_positive |= one > zero
        paired = results[(results["model"] == model) & (results["lambda"].isin([0.0, 1.0]))].pivot(
            index="seed", columns="lambda", values="pooled_full_gene_oof_r2"
        )
        differences = paired[1.0] - paired[0.0]
        full_curve = model_sweep if len(model_sweep) == len(config["lambda_grid"]) else None
        rho = (
            float(spearmanr(full_curve.index.to_numpy(dtype=float), full_curve["median_r2"]).statistic)
            if full_curve is not None
            else float("nan")
        )
        positive_gate = one >= config["adjudication"]["positive_control_min_model_median_r2"]
        separation_gate = (
            float(differences.median())
            >= config["adjudication"]["clear_separation_min_paired_median_difference"]
            and bool((differences > 0).all())
        )
        monotonicity_gate = bool(np.isfinite(rho) and rho >= config["adjudication"]["monotonicity_min_spearman"])
        model_qualifies = model in config["tier_1_models"] and positive_gate and separation_gate and monotonicity_gate
        if model_qualifies:
            qualifying.append(model)
        rows.append(
            {
                "model": model,
                "lambda0_median_r2": zero,
                "lambda1_median_r2": one,
                "paired_endpoint_median_difference": float(differences.median()),
                "all_10_endpoint_differences_positive": bool((differences > 0).all()),
                "full_curve_spearman": rho,
                "lambda1_r2_gate": positive_gate,
                "endpoint_separation_gate": separation_gate,
                "monotonicity_gate": monotonicity_gate,
                "model_positive_control_qualifies": model_qualifies,
            }
        )
    negative_gate = median_seedwise_best <= config["adjudication"]["negative_control_max_median_seedwise_best_r2"]
    if negative_gate and qualifying:
        verdict = "PIPELINE_POSITIVE_CONTROL_PASS"
    elif any_positive:
        verdict = "PIPELINE_POSITIVE_CONTROL_PARTIAL"
    else:
        verdict = "PIPELINE_POSITIVE_CONTROL_FAIL"
    diagnostics = {
        "negative_control_median_seedwise_best_r2": median_seedwise_best,
        "negative_control_gate": negative_gate,
        "qualifying_models": qualifying,
        "any_model_lambda1_median_above_lambda0": any_positive,
    }
    return verdict, pd.DataFrame(rows), diagnostics


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _panel(ax: Any, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def draw_figure(
    out: Path,
    reliability: pd.DataFrame,
    results: pd.DataFrame,
    sweep: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _style()
    colors = {
        "ridge": "#0072B2",
        "rbf_kernel_ridge": "#D55E00",
        "deep_residual_mlp": "#009E73",
    }
    labels = {
        "ridge": "Ridge",
        "rbf_kernel_ridge": "RBF Ridge",
        "deep_residual_mlp": "deep residual MLP",
    }
    fig = plt.figure(figsize=(10.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(0.9, 1.1, 1.25), height_ratios=(0.88, 1.12))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_c = fig.add_subplot(grid[1, :2])
    ax_d = fig.add_subplot(grid[1, 2])
    source_rows: list[dict[str, Any]] = []

    ax_a.set_axis_off()
    boxes = {
        "real baseline X": (0.08, 0.74),
        "baseline latent zX": (0.08, 0.43),
        "hidden latent zH": (0.58, 0.43),
        "gene response Γ": (0.33, 0.08),
    }
    for text, (x, y) in boxes.items():
        box = FancyBboxPatch((x, y), 0.34, 0.13, boxstyle="round,pad=0.02", facecolor="#F3F4F6", edgecolor="#4B5563", lw=0.8)
        ax_a.add_patch(box)
        ax_a.text(x + 0.17, y + 0.065, text, ha="center", va="center", fontsize=7.2)
    arrows = [((0.25, 0.74), (0.25, 0.56)), ((0.25, 0.43), (0.43, 0.21)), ((0.75, 0.43), (0.57, 0.21))]
    for start, end in arrows:
        ax_a.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#4B5563"))
    ax_a.text(0.5, 0.34, "λ mixture", ha="center", va="center", fontsize=8, color="#7A3E00")
    ax_a.text(0.5, 0.01, "+ matched replicate noise", ha="center", va="bottom", fontsize=6.8, color="#555555")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_title("Baseline-encoded response control")
    source_rows.append({"panel": "a", "series": "construction", "x": np.nan, "estimate": np.nan, "lower": np.nan, "upper": np.nan, "notes": "real X -> zX; zX/zH lambda mixture -> gene response plus replicate noise"})

    for lambda_value, group in reliability.groupby("lambda"):
        ax_b.scatter(np.full(len(group), lambda_value), group["realized_reliability"], s=14, color="#4C78A8", alpha=0.65, edgecolor="none")
        for row in group.itertuples(index=False):
            source_rows.append({"panel": "b", "series": f"seed={row.seed}", "x": lambda_value, "estimate": row.realized_reliability, "lower": np.nan, "upper": np.nan, "notes": "flattened replicate Pearson"})
    target = float(config["target_replicate_reliability"])
    tolerance = float(config["reliability_absolute_tolerance"])
    ax_b.axhspan(target - tolerance, target + tolerance, color="#D9D9D9", alpha=0.55, lw=0)
    ax_b.axhline(target, color="#333333", lw=0.9)
    ax_b.set_xticks(config["lambda_grid"])
    ax_b.set_xlabel("baseline-dependent latent fraction, λ")
    ax_b.set_ylabel("replicate reliability")
    ax_b.set_title("Measurement ceiling is matched for every synthetic world")

    rng = np.random.default_rng(207_049)
    offscale_models: list[tuple[str, float, float]] = []
    for model in config["tier_1_models"]:
        raw = results[results["model"] == model]
        summary = sweep[sweep["model"] == model]
        if float(raw["pooled_full_gene_oof_r2"].min()) < -0.10:
            offscale_models.append(
                (
                    labels[model],
                    float(summary["median_r2"].min()),
                    float(summary["median_r2"].max()),
                )
            )
        else:
            jitter = rng.normal(0, 0.006, size=len(raw))
            ax_c.scatter(raw["lambda"] + jitter, raw["pooled_full_gene_oof_r2"], s=9, alpha=0.18, color=colors[model], edgecolor="none")
            ax_c.plot(summary["lambda"], summary["median_r2"], marker="o", ms=4, lw=1.6, color=colors[model], label=labels[model])
            ax_c.fill_between(summary["lambda"], summary["q25_r2"], summary["q75_r2"], color=colors[model], alpha=0.12, lw=0)
        for row in summary.to_dict(orient="records"):
            source_rows.append({"panel": "c", "series": model, "x": row["lambda"], "estimate": row["median_r2"], "lower": row["q25_r2"], "upper": row["q75_r2"], "notes": "median and interquartile range across 10 seeds"})
    real_reference = float(config["real_full_gene_oof_r2"]["rbf_kernel_ridge"])
    ax_c.axhline(real_reference, color="#555555", lw=0.8, ls="--", label="real Lea RBF reference")
    ax_c.axhline(0, color="#999999", lw=0.7)
    ax_c.set_xticks(config["lambda_grid"])
    ax_c.set_xlabel("baseline-dependent latent fraction, λ")
    ax_c.set_ylabel("pooled full-gene OOF R²")
    ax_c.set_ylim(-0.015, 0.052)
    if offscale_models:
        offscale_text = "; ".join(
            f"{name} off-scale: median R² {minimum:.2f} to {maximum:.2f}"
            for name, minimum, maximum in offscale_models
        )
        ax_c.text(
            0.02,
            0.95,
            offscale_text,
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#555555",
        )
    ax_c.set_title("Frozen pipelines detect but weakly recover encoded response structure")
    ax_c.legend(frameon=False, ncol=2, loc="best")

    endpoint_model = config["figure_endpoint_model"]
    endpoints = results[(results["model"] == endpoint_model) & (results["lambda"].isin([0.0, 1.0]))]
    groups = [
        np.asarray([real_reference]),
        endpoints[endpoints["lambda"] == 0.0]["pooled_full_gene_oof_r2"].to_numpy(),
        endpoints[endpoints["lambda"] == 1.0]["pooled_full_gene_oof_r2"].to_numpy(),
    ]
    names = ["real Lea\nRBF", "synthetic\nλ=0", "synthetic\nλ=1"]
    box = ax_d.boxplot(groups, positions=[0, 1, 2], widths=0.58, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "lw": 1.2})
    for patch, color in zip(box["boxes"], ["#BDBDBD", "#9ECAE1", "#FC9272"], strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("#555555")
        patch.set_linewidth(0.8)
    for position, values in enumerate(groups):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax_d.scatter(position + jitter, values, s=13, color="#333333", alpha=0.65, zorder=3)
        for index, value in enumerate(values):
            source_rows.append({"panel": "d", "series": names[position].replace("\n", " "), "x": index, "estimate": value, "lower": np.nan, "upper": np.nan, "notes": "real reference or seed-level synthetic endpoint"})
    ax_d.axhline(0, color="#999999", lw=0.7)
    ax_d.set_xticks([0, 1, 2], names)
    ax_d.set_ylabel("pooled full-gene OOF R²")
    ax_d.set_title("Real result is a reference, not an inverted calibration")

    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "abcd", strict=True):
        _panel(ax, label)
        if ax is not ax_a:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out")
    fig.savefig(out / "Figure_Lea_Semisynthetic_Control.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "Figure_Lea_Semisynthetic_Control.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    source = pd.DataFrame(source_rows)
    source.to_csv(out / "Figure_Lea_Semisynthetic_Control_source_data.csv", index=False)
    return source


def finalize(
    root: Path,
    audit: dict[str, Any],
    reliability: pd.DataFrame,
    results: pd.DataFrame,
    device: torch.device,
    runtime_seconds: float,
) -> dict[str, Any]:
    config = audit["config"]
    out = root / "results/lea_semisynthetic_identifiability_control"
    sweep = _summary(results)
    verdict, model_summary, diagnostics = adjudicate(results, config)
    sweep.to_csv(out / "SEMISYNTHETIC_SIGNAL_SWEEP.csv", index=False)
    model_summary.to_csv(out / "SEMISYNTHETIC_MODEL_SUMMARY.csv", index=False)
    source = draw_figure(out, reliability, results, sweep, config)
    rows = [
        "# Lea semi-synthetic identifiability positive control — final",
        "",
        f"**{verdict}**",
        "",
        "## Scope and integrity",
        "",
        "This `SEMI_SYNTHETIC_BASELINE_ENCODED_RESPONSE_CONTROL` used the exact 342-LCL, 10,157-row frozen ETOH baseline matrix, biological-LCL-disjoint folds, model implementations, selected hyperparameters, and full-gene OOF residual R2 metric. Only the response was replaced. No synthetic-outcome tuning was performed.",
        "",
        f"- Frozen OOF source SHA-256: `{audit['source_npz_sha256']}`.",
        f"- Device: `{device}`; formal runtime: `{runtime_seconds/60:.1f}` minutes.",
        f"- Reliability gates passed: `{bool(reliability['noise_calibration_pass'].all())}` ({len(reliability)}/{len(reliability)} worlds).",
        f"- Realized reliability range: `{reliability.realized_reliability.min():.6f}` to `{reliability.realized_reliability.max():.6f}` around target `0.7443`.",
        "",
        "## Endpoint and monotonicity adjudication",
        "",
        f"- Median seedwise best-model R2 at lambda zero: `{diagnostics['negative_control_median_seedwise_best_r2']:.6f}` (gate <=0.02: `{diagnostics['negative_control_gate']}`).",
        f"- Qualifying positive-control models: `{diagnostics['qualifying_models']}`.",
        "",
        "| model | lambda=0 median R2 | lambda=1 median R2 | paired median gain | all gains positive | full-curve Spearman | qualifies |",
        "|:--|--:|--:|--:|:--:|--:|:--:|",
    ]
    for row in model_summary.itertuples(index=False):
        rho = "not full sweep" if not np.isfinite(row.full_curve_spearman) else f"{row.full_curve_spearman:.6f}"
        rows.append(
            f"| {row.model} | {row.lambda0_median_r2:.6f} | {row.lambda1_median_r2:.6f} | "
            f"{row.paired_endpoint_median_difference:.6f} | {row.all_10_endpoint_differences_positive} | "
            f"{rho} | {row.model_positive_control_qualifies} |"
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if verdict == "PIPELINE_POSITIVE_CONTROL_PASS":
        rows.append("The same held-out pipeline that recovered almost none of the real individual-specific full-gene response readily recovered a matched-dimensional response when transferable response structure was deliberately encoded in baseline transcriptomic state. The near-zero real-data result is therefore not an intrinsic consequence of sample size, output dimensionality, or the evaluation pipeline alone.")
    elif verdict == "PIPELINE_POSITIVE_CONTROL_PARTIAL":
        rows.append("The frozen pipeline distinguished baseline-encoded from baseline-independent synthetic worlds, but did not satisfy every predeclared magnitude and monotonicity gate. The control supports partial pipeline sensitivity and does not fully close a power-based interpretation of the real near-zero result. Because the rank-4 signal was embedded through random directions in the 10,157-dimensional baseline space with only 342 samples, this outcome conflates response identifiability with high-dimensional sample efficiency; it does not falsify baseline informativeness, but it is not a clean positive control for the stronger claim that fully baseline-encoded response is readily recoverable by the frozen battery.")
    else:
        rows.append("The frozen pipeline failed to recover the deliberately baseline-encoded synthetic response strongly enough to validate sensitivity. The Lea near-zero real-data result must therefore be weakened because pipeline insensitivity remains a viable explanation.")
    rows.extend(
        [
            "",
            "`lambda` is a simulation-generating parameter only. The real-data R2 is shown as a reference and is not inverted into an estimate of biological information content. This control is not an information-theoretic proof or upper bound.",
            "",
            "## Provenance",
            "",
            f"- Authority HEAD: `{config['authority_head']}`.",
            f"- Frozen real-result commit: `{config['real_result_commit']}`.",
            f"- Analysis implementation commit at execution: `{git_value(root, 'rev-parse', 'HEAD')}`.",
            f"- Figure source rows: `{len(source)}`.",
            "",
            "## Stop",
            "",
            "No generator rescue, rank expansion, CCA, mutual-information analysis, new architecture, or new dataset is authorized.",
            "",
            verdict,
        ]
    )
    (out / "SEMISYNTHETIC_FINAL.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"verdict": verdict, "diagnostics": diagnostics, "runtime_minutes": runtime_seconds / 60}
