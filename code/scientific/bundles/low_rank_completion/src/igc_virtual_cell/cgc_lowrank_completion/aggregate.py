from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .design import ENTRIES, OUTER_FOLDS, frozen_masks
from .runner import _cache_paths, _valid_cache, sha256


MODELS = ("MATCHED_AFFINE_RIDGE", "ADDITIVE_MAIN_EFFECT", "LOW_RANK_INTERACTION")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _load_cache_rows(source_root: Path, kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for fold in range(OUTER_FOLDS):
        if not _valid_cache(source_root, kind, fold):
            raise RuntimeError(f"LOW_RANK_MISSING_OR_CORRUPT_{kind.upper()}_FOLD:{fold}")
        archive, manifest = _cache_paths(source_root, kind, fold)
        with np.load(archive, allow_pickle=False) as values:
            fold_prediction = json.loads(str(values["prediction_rows_json"]))
            fold_diagnostics = json.loads(str(values["diagnostic_rows_json"]))
            fold_fits = json.loads(str(values["fit_rows_json"]))
            fold_selected = json.loads(str(values["selected_rows_json"]))
        for row in fold_prediction:
            row["coefficient_cache"] = str(archive)
            row["coefficient_cache_sha256"] = json.loads(manifest.read_text(encoding="utf-8"))["archive_sha256"]
        prediction.extend(fold_prediction)
        diagnostics.extend(fold_diagnostics)
        fits.extend(fold_fits)
        for row in fold_selected:
            row["outer_fold"] = fold
            selected.append(row)
    return pd.DataFrame(prediction), pd.DataFrame(diagnostics), pd.DataFrame(fits), pd.DataFrame(selected)


def aggregate_synthetic(root: Path, source_root: Path) -> tuple[pd.DataFrame, bool]:
    root = root.resolve()
    source_root = source_root.resolve()
    output = root / "results/cgc_lowrank_interaction_completion/LOW_RANK_SYNTHETIC_POSITIVE_CONTROL.csv"
    prediction, _, _, selected = _load_cache_rows(source_root, "synthetic")
    if len(prediction) != ENTRIES * 2 or prediction.target_index.nunique() != ENTRIES:
        raise RuntimeError("LOW_RANK_SYNTHETIC_OOF_COVERAGE_FAIL")
    rows: list[dict[str, object]] = []
    estimates: dict[str, float] = {}
    for model, data in prediction.groupby("model", sort=False):
        context_g = 1.0 - data.context_residual_cross.sum() / data.context_truth_cross.sum()
        full_g = 1.0 - data.full_residual_cross.sum() / data.full_truth_cross.sum()
        estimates[str(model)] = float(context_g)
        rows.append(
            {
                "scope": "pooled",
                "model": model,
                "context_specific_g": context_g,
                "full_response_g": full_g,
                "known_rank": 4,
                "selected_rank_mode": int(selected["rank"].mode().iloc[0]),
                "selected_rank4_fraction": float((selected["rank"] == 4).mean()),
                "selected_final_convergence_fraction": float(selected.converged.mean()),
                "entries_oof": prediction.target_index.nunique(),
            }
        )
    gain = estimates["LOW_RANK_INTERACTION"] - estimates["ADDITIVE_MAIN_EFFECT"]
    convergence = float(selected.converged.mean())
    passed = (
        estimates["LOW_RANK_INTERACTION"] >= 0.80
        and gain >= 0.50
        and prediction.target_index.nunique() == ENTRIES
        and convergence >= 0.95
    )
    for row in rows:
        row["lowrank_minus_additive"] = gain
        row["gate_passed"] = passed
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result, bool(passed)


def _bootstrap_results(
    prediction: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    draws: dict[tuple[str, str], np.ndarray] = {}
    ordered = prediction.sort_values(["model", "target_index"])
    for model in MODELS:
        data = ordered[ordered.model == model]
        if data.target_index.tolist() != list(range(ENTRIES)):
            raise RuntimeError(f"LOW_RANK_REAL_OOF_AXIS_FAIL:{model}")
        for endpoint, truth_column, residual_column in (
            ("CONTEXT_SPECIFIC", "context_truth_cross", "context_residual_cross"),
            ("FULL_RESPONSE", "full_truth_cross", "full_residual_cross"),
        ):
            truth = data[truth_column].to_numpy(np.float64)
            after = data[residual_column].to_numpy(np.float64)
            estimate = 1.0 - after.sum(dtype=np.float64) / truth.sum(dtype=np.float64)
            boot = 1.0 - (weights @ after) / (weights @ truth)
            draws[(model, endpoint)] = boot
            rows.append(
                {
                    "comparison": "MODEL_ESTIMATE",
                    "model": model,
                    "reference_model": "",
                    "endpoint": endpoint,
                    "estimate": estimate,
                    "lower_95": np.quantile(boot, 0.025),
                    "upper_95": np.quantile(boot, 0.975),
                    "bootstrap_draws": len(boot),
                }
            )
    for model, reference, label in (
        ("LOW_RANK_INTERACTION", "MATCHED_AFFINE_RIDGE", "LOWRANK_MINUS_AFFINE"),
        ("LOW_RANK_INTERACTION", "ADDITIVE_MAIN_EFFECT", "INTERACTION_INCREMENT"),
    ):
        for endpoint in ("CONTEXT_SPECIFIC", "FULL_RESPONSE"):
            difference = draws[(model, endpoint)] - draws[(reference, endpoint)]
            estimate = next(
                row["estimate"]
                for row in rows
                if row["model"] == model and row["endpoint"] == endpoint
            ) - next(
                row["estimate"]
                for row in rows
                if row["model"] == reference and row["endpoint"] == endpoint
            )
            rows.append(
                {
                    "comparison": label,
                    "model": model,
                    "reference_model": reference,
                    "endpoint": endpoint,
                    "estimate": estimate,
                    "lower_95": np.quantile(difference, 0.025),
                    "upper_95": np.quantile(difference, 0.975),
                    "bootstrap_draws": len(difference),
                }
            )
    return pd.DataFrame(rows), draws


def _context_results(prediction: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, context), data in prediction.groupby(["model", "context_index"]):
        rows.append(
            {
                "model": model,
                "context_index": context,
                "entries": len(data),
                "context_specific_g": 1.0 - data.context_residual_cross.sum() / data.context_truth_cross.sum(),
                "full_response_g": 1.0 - data.full_residual_cross.sum() / data.full_truth_cross.sum(),
            }
        )
    result = pd.DataFrame(rows)
    wide = result.pivot(index="context_index", columns="model", values="context_specific_g")
    gain = (
        wide["LOW_RANK_INTERACTION"] - wide["MATCHED_AFFINE_RIDGE"]
    ).rename("lowrank_minus_affine_context_g")
    result = result.merge(gain, on="context_index", how="left")
    return result


def _diagnostic_results(prediction: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    same = diagnostics[diagnostics.component == "context_specific_prediction"].copy()
    for (model, plate), data in same.groupby(["model", "plate"]):
        truth = data.truth_norm.sum()
        pred = data.predicted_norm.sum()
        dot = data.truth_dot_prediction.sum()
        rows.append(
            {
                "model": model,
                "component": "context_specific_prediction",
                "replicate_scope": f"plate{6 if int(plate)==0 else 14}",
                "truth_aligned_amplitude": dot / truth,
                "cosine_alignment": dot / np.sqrt(truth * pred) if truth > 0 and pred > 0 else np.nan,
                "scale_ratio": np.sqrt(pred / truth) if truth > 0 and pred >= 0 else np.nan,
                "predicted_interaction_energy_fraction": np.nan,
                "predicted_interaction_cross_covariance_fraction": np.nan,
            }
        )
    for model, data in prediction.groupby("model"):
        truth = data.context_truth_cross.sum()
        pred = data.context_predicted_cross.sum()
        dot = data.context_truth_dot_prediction_symmetric.sum()
        rows.append(
            {
                "model": model,
                "component": "context_specific_prediction",
                "replicate_scope": "cross_plate",
                "truth_aligned_amplitude": dot / truth,
                "cosine_alignment": dot / np.sqrt(truth * pred) if truth > 0 and pred > 0 else np.nan,
                "scale_ratio": np.sqrt(pred / truth) if truth > 0 and pred >= 0 else np.nan,
                "predicted_interaction_energy_fraction": np.nan,
                "predicted_interaction_cross_covariance_fraction": np.nan,
            }
        )
    interaction = diagnostics[diagnostics.component == "cp_interaction_only"]
    lowrank_prediction_energy = prediction.loc[
        prediction.model == "LOW_RANK_INTERACTION", "context_predicted_cross"
    ].sum()
    # This frozen cache quantity is a Plate6 x Plate14 bilinear product.  It is
    # signed cross-replicate covariance, not a non-negative same-plate energy.
    interaction_cross_covariance = interaction.predicted_norm.sum()
    rows.append(
        {
            "model": "LOW_RANK_INTERACTION",
            "component": "cp_interaction_only",
            "replicate_scope": "cross_plate",
            "truth_aligned_amplitude": np.nan,
            "cosine_alignment": np.nan,
            "scale_ratio": np.nan,
            "predicted_interaction_energy_fraction": np.nan,
            "predicted_interaction_cross_covariance_fraction": interaction_cross_covariance
            / lowrank_prediction_energy,
        }
    )
    return pd.DataFrame(rows)


def aggregate_real(root: Path, source_root: Path) -> dict[str, object]:
    root = root.resolve()
    source_root = source_root.resolve()
    out = root / "results/cgc_lowrank_interaction_completion"
    synthetic, synthetic_passed = aggregate_synthetic(root, source_root)
    prediction, diagnostics, fits, selected = _load_cache_rows(source_root, "real")
    if len(prediction) != ENTRIES * len(MODELS) or prediction.target_index.nunique() != ENTRIES:
        raise RuntimeError("LOW_RANK_REAL_OOF_COVERAGE_FAIL")
    split = json.loads(
        (root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    prediction["context_id"] = prediction.context_index.map(dict(enumerate(split["contexts"])))
    prediction["intervention_id"] = prediction.intervention_index.map(dict(enumerate(split["interventions"])))
    prediction.to_parquet(out / "LOW_RANK_PREDICTIONS.parquet", index=False)
    fits.to_csv(out / "LOW_RANK_HYPERPARAMETER_SELECTION.csv", index=False)
    cp_rows = diagnostics.component == "cp_interaction_only"
    diagnostics["predicted_interaction_cross_covariance_fraction"] = np.where(
        cp_rows, diagnostics["predicted_interaction_energy_fraction"], np.nan
    )
    diagnostics.loc[cp_rows, "predicted_interaction_energy_fraction"] = np.nan
    diagnostics.to_csv(out / "LOW_RANK_INTERACTION_DIAGNOSTICS_BY_ENTRY.csv", index=False)
    weights = np.load(
        source_root / "data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy",
        mmap_mode="r",
    ).astype(np.float64)
    recovery, draws = _bootstrap_results(prediction, weights)
    recovery.to_csv(out / "LOW_RANK_RECOVERY_RESULTS.csv", index=False)
    context = _context_results(prediction)
    context["context_id"] = context.context_index.map(dict(enumerate(split["contexts"])))
    context.to_csv(out / "LOW_RANK_CONTEXT_RESULTS.csv", index=False)
    diagnostic_summary = _diagnostic_results(prediction, diagnostics)
    diagnostic_summary.to_csv(out / "LOW_RANK_INTERACTION_DIAGNOSTICS.csv", index=False)

    selected_final = selected.copy()
    convergence = float(selected_final.converged.mean())
    low = recovery[
        (recovery.comparison == "MODEL_ESTIMATE")
        & (recovery.model == "LOW_RANK_INTERACTION")
        & (recovery.endpoint == "CONTEXT_SPECIFIC")
    ].iloc[0]
    contrast = recovery[
        (recovery.comparison == "LOWRANK_MINUS_AFFINE")
        & (recovery.endpoint == "CONTEXT_SPECIFIC")
    ].iloc[0]
    invariance = json.loads((out / "LOW_RANK_OUTCOME_INVARIANCE.json").read_text(encoding="utf-8"))
    valid = bool(synthetic_passed and invariance["passed"] and convergence >= 0.95)
    if not valid:
        verdict = "LOW_RANK_INTERACTION_COMPLETION_INVALID"
    elif contrast.lower_95 > 0 and contrast.estimate >= 0.25 and low.estimate >= 0.50:
        verdict = "LOW_RANK_INTERACTION_COMPLETION_CLOSES_GAP"
    elif contrast.lower_95 > 0 and contrast.estimate >= 0.05:
        verdict = "LOW_RANK_INTERACTION_COMPLETION_PARTIAL"
    else:
        verdict = "LOW_RANK_INTERACTION_COMPLETION_NO_RESCUE"
    result = {
        "verdict": verdict,
        "valid": valid,
        "synthetic_passed": synthetic_passed,
        "selected_final_convergence_fraction": convergence,
        "recovery": recovery,
        "context": context,
        "diagnostics": diagnostic_summary,
        "selected": selected_final,
        "prediction": prediction,
        "git_commit_at_aggregation": _git(root, "rev-parse", "HEAD"),
    }
    return result
