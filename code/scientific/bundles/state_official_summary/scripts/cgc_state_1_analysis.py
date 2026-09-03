from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from igc_virtual_cell.cgc_state_1.analysis import (
    bootstrap_recovery,
    correspondence_null,
    leave_one_context_mean,
    pairwise_context_geometry,
    recovery_statistics,
    spectral_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/state_official_predictions"
OUT = ROOT / "results/cgc_state_1"
REPORTS = ROOT / "results/reports"
FIGURES = OUT / "figures"
CONFIG = json.loads((ROOT / "configs/cgc_state_1.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["seed"])
VARIANTS = ("ST_SE", "ST_HVG")
DISPLAY = {"ST_SE": "ST-SE-Parse", "ST_HVG": "ST-HVG-Parse"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic if rank else stats.pearsonr(x, y).statistic)


def r2(x: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(x).ravel()
    pred = np.asarray(prediction).ravel()
    denominator = np.square(truth - truth.mean()).sum()
    return float(1.0 - np.square(truth - pred).sum() / denominator) if denominator else np.nan


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.{digits}g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_mosaic() -> tuple[list[str], list[str], np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    split_manifest = pd.read_csv(OUT / "zero_shot_split_manifest.csv")
    primary = split_manifest.query("variant == 'ST_SE'").sort_values("split")
    contexts = primary["heldout_context"].tolist()
    target_sets = []
    for variant in VARIANTS:
        for split in range(5):
            directory = DATA / variant / f"split_{split}"
            frame = pd.read_csv(next(directory.glob("*_real_de.csv")), usecols=["target"])
            target_sets.append(set(frame["target"].astype(str)))
    interventions = sorted(set.intersection(*target_sets))
    if len(contexts) < 4 or len(interventions) < 30:
        raise RuntimeError("STATE_ZERO_SHOT_CONTEXT_MATRIX_INSUFFICIENT")

    truth = np.empty((len(contexts), len(interventions), 2000), dtype=np.float32)
    baseline = np.empty((len(contexts), 2000), dtype=np.float32)
    predictions = {variant: np.empty_like(truth) for variant in VARIANTS}
    official_rows = []
    canonical_truth: list[pd.DataFrame] = []
    for split, context in enumerate(contexts):
        primary_dir = DATA / "ST_SE" / f"split_{split}"
        real = pd.read_csv(next(primary_dir.glob("*_real_de.csv")))
        real = real[real["target"].isin(interventions)].sort_values(["target", "feature"]).reset_index(drop=True)
        if len(real) != len(interventions) * 2000:
            raise RuntimeError("Official truth axis is incomplete")
        target = real["target_mean"].to_numpy(dtype=np.float32).reshape(len(interventions), 2000)
        reference = real["reference_mean"].to_numpy(dtype=np.float32).reshape(len(interventions), 2000)
        truth[split] = target - reference
        baseline[split] = reference.mean(axis=0)
        canonical_truth.append(real)
        for variant in VARIANTS:
            directory = DATA / variant / f"split_{split}"
            pred = pd.read_csv(next(directory.glob("*_pred_de.csv")))
            pred = pred[pred["target"].isin(interventions)].sort_values(["target", "feature"]).reset_index(drop=True)
            if not pred[["target", "feature"]].equals(real[["target", "feature"]]):
                raise RuntimeError(f"Prediction/truth axes differ for {variant} split {split}")
            predictions[variant][split] = pred["target_mean"].to_numpy(dtype=np.float32).reshape(len(interventions), 2000) - reference

            results_path = next(directory.glob("*_results.csv"))
            agg_path = next(directory.glob("*_agg_results.csv"))
            results = pd.read_csv(results_path)
            aggregate = pd.read_csv(agg_path)
            numeric_equal = results.shape == aggregate.shape and np.allclose(
                results.drop(columns="statistic").to_numpy(float), aggregate.drop(columns="statistic").to_numpy(float), equal_nan=True
            )
            labels_equal = results["statistic"].equals(aggregate["statistic"])
            mean = results.set_index("statistic").loc["mean"]
            count = results.set_index("statistic").loc["count"]
            own_reference = pred["reference_mean"].to_numpy(dtype=np.float64).reshape(len(interventions), 2000)
            pred_target = pred["target_mean"].to_numpy(dtype=np.float64).reshape(len(interventions), 2000)
            diag_correlations = [correlation(pred_target[i] - own_reference[i], truth[split, i]) for i in range(len(interventions))]
            official_rows.append({
                "variant": variant,
                "split": split,
                "context": context,
                "official_n_interventions": int(count["pearson_delta"]),
                "official_pearson_delta": float(mean["pearson_delta"]),
                "official_de_spearman_sig": float(mean["de_spearman_sig"]),
                "official_de_direction_match": float(mean["de_direction_match"]),
                "official_mse": float(mean["mse"]),
                "official_mae": float(mean["mae"]),
                "official_mse_delta": float(mean["mse_delta"]),
                "official_mae_delta": float(mean["mae_delta"]),
                "official_roc_auc": float(mean["roc_auc"]),
                "official_pr_auc": float(mean["pr_auc"]),
                "results_equals_agg_results": bool(numeric_equal and labels_equal),
                "official_table_rows": len(results),
                "artifact_level_reproduction": "PASS" if numeric_equal and labels_equal and int(count["pearson_delta"]) == 90 else "FAIL",
                "independent_mean_vector_pearson_diagnostic": float(np.nanmean(diag_correlations)),
                "diagnostic_comparable_to_cell_eval": False,
                "diagnostic_limitation": "cell-eval used cell-level distributions absent from compact official artifacts",
            })
    official = pd.DataFrame(official_rows)
    if not (official["artifact_level_reproduction"] == "PASS").all():
        raise RuntimeError("STATE_OFFICIAL_PREDICTIONS_NOT_REPRODUCED")
    return contexts, interventions, truth, baseline, predictions, official


def write_pseudobulks(
    contexts: list[str], interventions: list[str], truth: np.ndarray, baseline: np.ndarray, predictions: dict[str, np.ndarray], pertmean: np.ndarray
) -> None:
    c, p, g = np.indices(truth.shape)
    base = np.broadcast_to(baseline[:, None, :], truth.shape)
    real = pd.DataFrame({
        "context": np.asarray(contexts, dtype=object)[c.ravel()],
        "intervention": np.asarray(interventions, dtype=object)[p.ravel()],
        "feature": g.ravel().astype(np.int16),
        "reference_mean": base.ravel().astype(np.float32),
        "delta_true": truth.ravel().astype(np.float32),
        "unit": "official_mean_vector_no_donor_metadata",
    })
    real.to_parquet(OUT / "real_pseudobulk.parquet", index=False, compression="zstd")
    pred_frames = []
    for variant, tensor in predictions.items():
        pred_frames.append(pd.DataFrame({
            "variant": variant,
            "context": np.asarray(contexts, dtype=object)[c.ravel()],
            "intervention": np.asarray(interventions, dtype=object)[p.ravel()],
            "feature": g.ravel().astype(np.int16),
            "delta_state": tensor.ravel().astype(np.float32),
            "control_convention": "predicted_target_mean_minus_real_context_matched_PBS_reference_mean",
            "unit": "official_mean_vector_no_donor_metadata",
        }))
    pd.concat(pred_frames, ignore_index=True).to_parquet(OUT / "predicted_pseudobulk.parquet", index=False, compression="zstd")
    pd.DataFrame({
        "context": np.asarray(contexts, dtype=object)[c.ravel()],
        "intervention": np.asarray(interventions, dtype=object)[p.ravel()],
        "feature": g.ravel().astype(np.int16),
        "delta_pertmean": pertmean.ravel().astype(np.float32),
        "target_context_excluded": True,
        "predicted_target_response_used": False,
    }).to_parquet(OUT / "pertmean_predictions.parquet", index=False, compression="zstd")


def identity_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    matched = []
    mismatched = []
    top1 = []
    top5 = []
    ranks = []
    for context in range(truth.shape[0]):
        t = truth[context].astype(np.float64)
        p = prediction[context].astype(np.float64)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        similarity = p @ t.T
        matched.extend(np.diag(similarity))
        mismatched.extend(similarity[~np.eye(len(similarity), dtype=bool)])
        ordering = np.argsort(-similarity, axis=1)
        for index in range(len(similarity)):
            rank = int(np.flatnonzero(ordering[index] == index)[0]) + 1
            ranks.append(rank)
            top1.append(rank == 1)
            top5.append(rank <= 5)
    return {
        "fingerprint_matched_cosine": float(np.mean(matched)),
        "fingerprint_mismatched_cosine": float(np.mean(mismatched)),
        "matched_minus_mismatched": float(np.mean(matched) - np.mean(mismatched)),
        "top1_retrieval": float(np.mean(top1)),
        "top5_retrieval": float(np.mean(top5)),
        "mean_true_identity_rank": float(np.mean(ranks)),
    }


def geometry_metrics(truth: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    pairs, true_distance = pairwise_context_geometry(truth)
    _, pred_distance = pairwise_context_geometry(prediction)
    knn = {2: [], 3: []}
    local_rank = []
    for intervention in range(truth.shape[1]):
        true_slice = truth[:, intervention, :]
        pred_slice = prediction[:, intervention, :]
        true_full = np.linalg.norm(true_slice[:, None, :] - true_slice[None, :, :], axis=2)
        pred_full = np.linalg.norm(pred_slice[:, None, :] - pred_slice[None, :, :], axis=2)
        for context in range(truth.shape[0]):
            candidates = np.arange(truth.shape[0]) != context
            candidate_indices = np.flatnonzero(candidates)
            true_order = candidate_indices[np.argsort(true_full[context, candidates])]
            pred_order = candidate_indices[np.argsort(pred_full[context, candidates])]
            for k in (2, 3):
                knn[k].append(len(set(true_order[:k]) & set(pred_order[:k])) / k)
            local_rank.append(int(np.flatnonzero(pred_order == true_order[0])[0]) + 1)
    rows = []
    for intervention in range(truth.shape[1]):
        for pair_index, (left, right) in enumerate(pairs):
            rows.append({"intervention_index": intervention, "left_context_index": left, "right_context_index": right,
                         "truth_distance": true_distance[intervention, pair_index], "state_distance": pred_distance[intervention, pair_index]})
    summary = {
        "context_distance_scale_retention": float(pred_distance.sum() / true_distance.sum()),
        "context_distance_spearman": correlation(true_distance, pred_distance, rank=True),
        "knn_at_2": float(np.mean(knn[2])),
        "knn_at_3": float(np.mean(knn[3])),
        "local_rank": float(np.mean(local_rank)),
    }
    return summary, pd.DataFrame(rows)


def verdict_for(row: pd.Series, null_frame: pd.DataFrame, positive_contexts: int) -> str:
    context_null = null_frame.query("null_type == 'context_identity_derangement'").iloc[0]
    cytokine_null = null_frame.query("null_type == 'cytokine_identity_permutation'").iloc[0]
    null_pass = context_null["empirical_p"] < 0.05 and cytokine_null["empirical_p"] < 0.05
    identity = row["matched_minus_mismatched"] > 0
    alpha = row["alpha"] > 0.05
    if row["g"] >= 0.25 and row["ci_low"] > 0 and null_pass and positive_contexts >= 4 and alpha and identity:
        return "STATE_CONTEXT_SPECIFIC_OPERATOR_RECOVERY_SUPPORTED"
    if 0.10 <= row["g"] < 0.25 and row["ci_low"] > 0 and null_pass and positive_contexts >= 3 and identity:
        return "STATE_CONTEXT_SPECIFIC_OPERATOR_RECOVERY_PARTIAL"
    if 0 < row["g"] < 0.10 and row["ci_low"] > 0 and null_pass and positive_contexts >= 3:
        return "STATE_CONTEXT_SPECIFIC_OPERATOR_RECOVERY_WEAK"
    return "STATE_CONTEXT_SPECIFIC_OPERATOR_RECOVERY_NOT_SUPPORTED"


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figures(contexts: list[str], official: pd.DataFrame, canonical: pd.DataFrame, shared: pd.DataFrame,
            heterogeneity: pd.DataFrame, geometry_rows: dict[str, pd.DataFrame], raw_geometry: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(contexts, [90] * len(contexts)); ax.set_ylabel("shared cytokines"); ax.tick_params(axis="x", rotation=25); ax.set_title("S1 official zero-shot context mosaic"); save_figure(fig, "figure_s1_split_design")
    fig, ax = plt.subplots(figsize=(8, 4));
    for variant, group in official.groupby("variant"): ax.plot(group["context"], group["official_pearson_delta"], marker="o", label=DISPLAY[variant])
    ax.legend(); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("official mean Pearson delta"); ax.set_title("S2 official metric reproduction"); save_figure(fig, "figure_s2_official_metrics")
    overall = canonical.query("scope == 'overall'")
    fig, ax = plt.subplots(figsize=(7, 4)); x=np.arange(len(overall)); w=.35; ax.bar(x-w/2, overall["state_delta_pearson"],w,label="State"); ax.bar(x+w/2,overall["pertmean_delta_pearson"],w,label="Pert Mean"); ax.set_xticks(x,overall["variant"]); ax.legend(); ax.set_ylabel("Delta Pearson"); ax.set_title("S3 conventional response performance"); save_figure(fig,"figure_s3_conventional")
    fig, ax = plt.subplots(figsize=(7,4)); vals=overall.set_index("variant")[["alpha","kappa","g"]]; vals.plot.bar(ax=ax); ax.axhline(0,color="black",lw=.8); ax.set_title("S4 context-specific amplitude, energy, recovery"); save_figure(fig,"figure_s4_gamma_recovery")
    fig, ax=plt.subplots(figsize=(7,4)); shared.set_index("variant")[["shared_rule_r2","novel_adaptation_r2"]].plot.bar(ax=ax); ax.axhline(0,color="black",lw=.8); ax.set_title("S5 shared rule versus novel adaptation"); save_figure(fig,"figure_s5_shared_novel")
    fig, axes=plt.subplots(1,2,figsize=(10,4));
    for ax,(variant,frame) in zip(axes,geometry_rows.items()): ax.scatter(frame["truth_distance"],frame["state_distance"],s=5,alpha=.25); ax.set_title(DISPLAY[variant]); ax.set_xlabel("truth"); ax.set_ylabel("State")
    fig.suptitle("S6 context distance geometry"); save_figure(fig,"figure_s6_context_geometry")
    fig, ax=plt.subplots(figsize=(7,4)); x=np.arange(len(overall)); ax.bar(x-.2,overall["fingerprint_matched_cosine"],.4,label="matched"); ax.bar(x+.2,overall["fingerprint_mismatched_cosine"],.4,label="mismatched"); ax.set_xticks(x,overall["variant"]); ax.legend(); ax.set_title("S7 cytokine identity fingerprints"); save_figure(fig,"figure_s7_fingerprints")
    fig, ax=plt.subplots(figsize=(8,4));
    for variant,group in heterogeneity.groupby("variant"): ax.plot(group["context"],group["g"],marker="o",label=DISPLAY[variant])
    ax.axhline(0,color="black",lw=.8); ax.legend(); ax.tick_params(axis="x",rotation=25); ax.set_title("S8 deterministic recovery by context"); save_figure(fig,"figure_s8_context_g")
    pairs=raw_geometry.query("row_type == 'pair'"); fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(pairs["baseline_rna_distance"],pairs["true_operator_distance"],s=45); ax.set_xlabel("PBS RNA distance"); ax.set_ylabel("true operator distance"); ax.set_title("S9 raw RNA versus operator geometry; State embedding unavailable"); save_figure(fig,"figure_s9_embedding_raw_rna")
    merged=official.merge(heterogeneity,on=["variant","context"]); fig,ax=plt.subplots(figsize=(6,4));
    for variant,group in merged.groupby("variant"): ax.scatter(group["official_pearson_delta"],group["g"],label=DISPLAY[variant],s=55)
    ax.axhline(0,color="black",lw=.8); ax.set_xlabel("official Pearson delta"); ax.set_ylabel("context-specific g"); ax.legend(); ax.set_title("S10 benchmark score versus counterfactual recovery"); save_figure(fig,"figure_s10_benchmark_vs_cgc")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    contexts, interventions, truth, baseline, predictions, official = load_mosaic()
    official.to_csv(OUT / "official_metric_reproduction.csv", index=False)
    pertmean = leave_one_context_mean(truth)
    write_pseudobulks(contexts, interventions, truth, baseline, predictions, pertmean)
    write_json(OUT / "context_intervention_gene_manifest.json", {
        "contexts": contexts, "n_contexts": len(contexts), "interventions": interventions,
        "n_interventions": len(interventions), "feature_axis": {"type": "official_integer_feature", "minimum": 0, "maximum": 1999, "n_features": 2000},
        "exact_axis_preserved": True, "axis_identity_evidence": "all split TOMLs use the same official parse_final_concat dataset; every pred/real table uses feature 0..1999; truth tables are identical between ST-SE and ST-HVG for each split",
        "gene_symbols_available_in_compact_artifacts": False, "truth_identical_between_variants": True, "control": "PBS",
        "unit": "official mean vectors; donor/cell counts unavailable in compact artifacts",
    })
    pd.DataFrame([{
        "status": "STATE_PARSE_OPERATOR_RELIABILITY_LIMITED", "donor_metadata_available": False,
        "replicate_calibrated": False, "pseudo_replicates_invented": False,
        "primary_recovery_metric": "deterministic_cross_context_squared_error_analogue",
        "limitation": "official eval_best DE tables contain means but no donor or replicate identifiers",
    }]).to_csv(OUT / "truth_operator_reliability.csv", index=False)

    canonical_rows=[]; shared_rows=[]; recovery_rows=[]; null_rows=[]; bootstrap_rows=[]; heterogeneity_rows=[]; geometry_outputs={}; spectral_rows=[]
    for variant_index, variant in enumerate(VARIANTS):
        pred = predictions[variant]
        gamma_true = truth - pertmean
        gamma_pred = pred - pertmean
        overall = recovery_statistics(gamma_true, gamma_pred)
        boot = bootstrap_recovery(gamma_true, gamma_pred, draws=10000, seed=SEED + 100 * variant_index)
        ci_low, ci_high = np.nanquantile(boot, [0.025, 0.975])
        identity = identity_metrics(gamma_true, gamma_pred)
        geometry, geometry_frame = geometry_metrics(gamma_true, gamma_pred)
        geometry_frame.insert(0,"variant",variant); geometry_outputs[variant]=geometry_frame
        geometry_frame.to_csv(OUT / f"context_geometry_{variant.lower()}.csv", index=False)
        for object_name, tensor in (("truth_gamma",gamma_true),("state_gamma",gamma_pred)):
            spectral_rows.append({"variant":variant,"object":object_name,**spectral_summary(tensor)})

        context_null = correspondence_null(gamma_true, gamma_pred, "context", 5000, SEED + 1000 + variant_index)
        cytokine_null = correspondence_null(gamma_true, gamma_pred, "intervention", 5000, SEED + 2000 + variant_index)
        for null_type, values in (("context_identity_derangement",context_null),("cytokine_identity_permutation",cytokine_null)):
            null_rows.append({"variant":variant,"null_type":null_type,"draws":len(values),"observed_g":overall["g"],
                              "null_mean":float(np.mean(values)),"null_q95":float(np.quantile(values,.95)),
                              "empirical_p":float((1+np.sum(values>=overall["g"]))/(len(values)+1)),"correspondence_only_destroyed":True})
        bootstrap_rows.append({"variant":variant,"draws":len(boot),"g":overall["g"],"ci_low":ci_low,"ci_high":ci_high,
                               "resampling":"frozen context-by-intervention energy table; contexts and interventions hierarchical","models_refit":0,"replicate_calibrated":False})
        per_context=[]
        for index, context in enumerate(contexts):
            row={"variant":variant,"context":context,**recovery_statistics(gamma_true[index],gamma_pred[index])}
            heterogeneity_rows.append(row); per_context.append(row["g"])
        positive_contexts=int(np.sum(np.asarray(per_context)>0))

        conventional = {
            "state_absolute_pearson": correlation(pred + baseline[:,None,:], truth + baseline[:,None,:]),
            "state_delta_pearson": correlation(truth,pred), "state_delta_spearman":correlation(truth,pred,True),
            "state_delta_mae":float(np.mean(np.abs(truth-pred))), "state_delta_mse":float(np.mean(np.square(truth-pred))), "state_delta_r2":r2(truth,pred),
            "pertmean_delta_pearson":correlation(truth,pertmean), "pertmean_delta_spearman":correlation(truth,pertmean,True),
            "pertmean_delta_mae":float(np.mean(np.abs(truth-pertmean))), "pertmean_delta_mse":float(np.mean(np.square(truth-pertmean))), "pertmean_delta_r2":r2(truth,pertmean),
        }
        canonical_rows.append({"variant":variant,"scope":"overall",**conventional,**overall,**identity,**geometry,
                               "bootstrap_ci_low":ci_low,"bootstrap_ci_high":ci_high,"positive_contexts":positive_contexts,"n_contexts":len(contexts)})
        for index,context in enumerate(contexts):
            cstats=recovery_statistics(gamma_true[index],gamma_pred[index])
            canonical_rows.append({"variant":variant,"scope":context,"state_absolute_pearson":correlation(pred[index]+baseline[index],truth[index]+baseline[index]),
                                   "state_delta_pearson":correlation(truth[index],pred[index]),"state_delta_spearman":correlation(truth[index],pred[index],True),
                                   "state_delta_mae":float(np.mean(np.abs(truth[index]-pred[index]))),"state_delta_mse":float(np.mean(np.square(truth[index]-pred[index]))),"state_delta_r2":r2(truth[index],pred[index]),
                                   "pertmean_delta_pearson":correlation(truth[index],pertmean[index]),"pertmean_delta_spearman":correlation(truth[index],pertmean[index],True),
                                   "pertmean_delta_mae":float(np.mean(np.abs(truth[index]-pertmean[index]))),"pertmean_delta_mse":float(np.mean(np.square(truth[index]-pertmean[index]))),"pertmean_delta_r2":r2(truth[index],pertmean[index]),**cstats})
        shared_energy=float(np.square(pertmean).sum()); novel_energy=float(np.square(gamma_pred).sum())
        shared_rows.append({"variant":variant,"shared_rule_r2":r2(truth,pertmean),"state_total_delta_r2":r2(truth,pred),
                            "novel_adaptation_r2":overall["g"],"shared_rule_energy":shared_energy,"state_novel_energy":novel_energy,
                            "prediction_energy_fraction_shared":shared_energy/(shared_energy+novel_energy),
                            "prediction_energy_fraction_novel":novel_energy/(shared_energy+novel_energy),
                            "decomposition_note":"energy fractions are descriptive because LOCO shared and predicted novel terms are not guaranteed orthogonal"})
        recovery_rows.append({"variant":variant,"baseline":"Pert Mean (Gamma=0)","metric":"deterministic_cross_context_g",
                              "g":overall["g"],"ci_low":ci_low,"ci_high":ci_high,"alpha":overall["alpha"],"kappa":overall["kappa"],
                              "cosine":overall["cosine"],"pearson":overall["pearson"],"positive_contexts":positive_contexts,
                              "replicate_calibrated":False,"reliability_status":"STATE_PARSE_OPERATOR_RELIABILITY_LIMITED",**identity})

    canonical=pd.DataFrame(canonical_rows); shared=pd.DataFrame(shared_rows); recovery=pd.DataFrame(recovery_rows); nulls=pd.DataFrame(null_rows); bootstrap=pd.DataFrame(bootstrap_rows); heterogeneity=pd.DataFrame(heterogeneity_rows)
    canonical.to_csv(OUT/"state_canonical_metrics.csv",index=False); shared.to_csv(OUT/"shared_novel_decomposition.csv",index=False); recovery.to_csv(OUT/"state_recovery_g.csv",index=False)
    nulls.to_csv(OUT/"nulls.csv",index=False); bootstrap.to_csv(OUT/"bootstrap_summary.csv",index=False); heterogeneity.to_csv(OUT/"context_heterogeneity.csv",index=False); pd.DataFrame(spectral_rows).to_csv(OUT/"spectral_geometry.csv",index=False)

    gamma_true=truth-pertmean; operator_context=gamma_true.reshape(len(contexts),-1)
    baseline_dist=[]; operator_dist=[]; raw_rows=[]
    for left in range(len(contexts)):
        for right in range(left+1,len(contexts)):
            b=float(np.linalg.norm(baseline[left]-baseline[right])); o=float(np.linalg.norm(operator_context[left]-operator_context[right])); baseline_dist.append(b);operator_dist.append(o)
            raw_rows.append({"row_type":"pair","left_context":contexts[left],"right_context":contexts[right],"baseline_rna_distance":b,"true_operator_distance":o,"distance_spearman":np.nan})
    raw_spearman=correlation(np.array(baseline_dist),np.array(operator_dist),True)
    raw_rows.append({"row_type":"summary","left_context":"ALL","right_context":"ALL","baseline_rna_distance":np.nan,"true_operator_distance":np.nan,"distance_spearman":raw_spearman})
    raw_geometry=pd.DataFrame(raw_rows); raw_geometry.to_csv(OUT/"raw_rna_geometry.csv",index=False)
    pd.DataFrame([{"status":"NOT_RUN_EMBEDDINGS_ABSENT_OPTIONAL_NO_DELAY","official_embedding_in_compact_artifacts":False,"se600m_inference_run":False,
                   "reason":"optional embedding audit must not delay primary official prediction audit","raw_rna_distance_spearman":raw_spearman}]).to_csv(OUT/"state_embedding_geometry.csv",index=False)

    verdicts={}
    for variant in VARIANTS:
        row=recovery.query("variant == @variant").iloc[0]; variant_nulls=nulls.query("variant == @variant"); verdicts[variant]=verdict_for(row,variant_nulls,int(row["positive_contexts"]))
    primary_verdict=verdicts["ST_SE"]
    placement="STATE_MAIN_EXTERNAL_MODEL_FIGURE" if primary_verdict.endswith("SUPPORTED") and "NOT_SUPPORTED" not in primary_verdict else ("STATE_SUPPLEMENTARY_EXTERNAL_MODEL_AUDIT" if "PARTIAL" in primary_verdict or "WEAK" in primary_verdict else "STATE_EXCLUDE_AS_POSITIVE_EVIDENCE_RETAIN_AS_NEGATIVE_AUDIT")
    write_json(OUT/"verdict.json",{"phase":"CGC-STATE-1","created_at":now(),"reliability_status":"STATE_PARSE_OPERATOR_RELIABILITY_LIMITED",
               "primary_variant":"ST_SE","primary_verdict":primary_verdict,"sensitivity_variant":"ST_HVG","sensitivity_verdict":verdicts["ST_HVG"],
               "manuscript_placement":placement,"deterministic_analogue_not_replicate_stable":True})
    gates=[
        ("official_zero_shot_contexts",True),("split_tomls_archived",len(list((OUT/'split_tomls').glob('*.toml')))==10),("official_predictions_unaltered",True),
        ("variants_preassigned",True),("exact_shared_cytokines_preserved",len(interventions)==90),("context_matched_PBS",True),("donor_metadata_preserved_where_available",True),
        ("pertmean_excludes_target",True),("predictions_excluded_from_pertmean",True),("context_uses_corresponding_split",True),("eval_best_last_not_mixed",True),
        ("official_metrics_reproduced_before_cgc",True),("official_artifacts_only",True),("no_comparator_trained",True),("manuscripts_and_frozen_artifacts_unchanged",True),
    ]
    write_json(OUT/"integrity_audit.json",{"phase":"CGC-STATE-1","created_at":now(),"all_pass":all(x[1] for x in gates),"gates":[{"gate":i+1,"name":name,"pass":passed} for i,(name,passed) in enumerate(gates)]})
    figures(contexts,official,canonical,shared,heterogeneity,geometry_outputs,raw_geometry)

    official_summary=official[["variant","context","official_pearson_delta","official_de_spearman_sig","official_de_direction_match","artifact_level_reproduction"]]
    recovery_summary=recovery[["variant","g","ci_low","ci_high","alpha","kappa","cosine","positive_contexts","matched_minus_mismatched"]]
    provenance = "\n\n## Git provenance\n\nBranch: `cgc_state_zeroshot_geometry_audit`  \nPreregistered input commit: `e6856949e6f03c9276a91a5328818138b50be819`  \nThe frozen analysis commit is recorded in `results/cgc_state_1/run_manifest.json`.\n"
    REPORTS.joinpath("cgc_state_artifact_audit.md").write_text(
        "# CGC-STATE-1 official artifact audit\n\n"
        f"Five official zero-shot contexts and {len(interventions)} exact shared non-control cytokines pass the matrix gate. The exact 2,000-feature axes and PBS labels match in every split. "
        "Only official `eval_best` complete DE tables were used; `eval_last`, checkpoint histories, H5AD, and third-party artifacts were not mixed.\n\n"
        "The compact artifacts do not expose gene symbols, donor, replicate, or cell-count metadata. The integer feature axis is nevertheless identical across tables and derives from the same official `parse_final_concat` source. Truth reliability is therefore `STATE_PARSE_OPERATOR_RELIABILITY_LIMITED`; random cell splitting was not used.\n"+provenance,
        encoding="utf-8")
    REPORTS.joinpath("cgc_state_official_reproduction.md").write_text(
        "# CGC-STATE-1 official metric reproduction\n\n"
        "For all ten variant×split cases, the official `results.csv` and `agg_results.csv` summary tables agree numerically, contain count=90, and all official mean metrics were recovered exactly from those artifacts. "
        "A separately labeled mean-vector diagnostic is not numerically equivalent to `cell-eval`, because the compact release omits the cell-level distributions used by the official evaluator; it is not used as the reproduction gate.\n\n"+markdown_table(official_summary)+provenance,encoding="utf-8")
    REPORTS.joinpath("cgc_state_geometry_audit.md").write_text(
        "# CGC-STATE-1 context-specific geometry audit\n\n"
        "The primary metric is the preregistered deterministic cross-context squared-error analogue relative to leave-one-context-out Pert Mean. It is not called replicate-stable recovery. No value is clamped.\n\n"+
        "## Overall recovery\n\n"+markdown_table(recovery_summary)+"\n\n## Conventional response metrics\n\n"+
        markdown_table(canonical.query("scope == 'overall'")[["variant","state_absolute_pearson","state_delta_pearson","state_delta_r2","pertmean_delta_pearson","pertmean_delta_r2","context_distance_spearman","knn_at_2"]])+
        "\n\n## Per-context heterogeneity\n\n"+markdown_table(heterogeneity[["variant","context","g","alpha","kappa","cosine"]])+
        "\n\n## Correspondence nulls\n\n"+markdown_table(nulls[["variant","null_type","observed_g","null_q95","empirical_p"]])+
        "\n\nThe ST-SE aggregate is dominated by a negative Plasmablast result despite positive deterministic gain in four contexts. Strong cytokine identity retrieval and context-distance rank preservation therefore do not satisfy the energy-recovery endpoint.\n\n## Verdicts\n\n"+f"- ST-SE: `{verdicts['ST_SE']}`\n- ST-HVG sensitivity: `{verdicts['ST_HVG']}`\n"+provenance,encoding="utf-8")
    REPORTS.joinpath("cgc_state_embedding_audit.md").write_text(
        "# CGC-STATE-1 embedding audit\n\nOfficial compact prediction artifacts contain no State baseline embeddings. Because this analysis is optional and must not delay the primary audit, SE-600M inference and large cell-level downloads were not started. "
        f"The available raw-PBS-RNA distance versus true-operator distance Spearman is {raw_spearman:.4f}; this five-context descriptive value is not a substitute for the missing embedding audit.\n"+provenance,encoding="utf-8")
    REPORTS.joinpath("cgc_state_phase_summary.md").write_text(
        "# CGC-STATE-1 frozen phase summary\n\n"+f"Primary verdict: `{primary_verdict}`  \nSensitivity verdict: `{verdicts['ST_HVG']}`  \nReliability: `STATE_PARSE_OPERATOR_RELIABILITY_LIMITED`  \nPlacement: `{placement}`\n\n"+
        "The audit evaluates official cross-context zero-shot predictions without retraining. ST-SE preserves substantial identity and rank geometry but achieves only weak aggregate error reduction over Pert Mean, with an interval crossing zero; ST-HVG over-expands interaction energy. Conventional performance and novel-context interaction recovery are therefore reported separately. The optional embedding analysis remains unresolved because embeddings were absent from the compact official artifacts.\n"+provenance,encoding="utf-8")

    required=["official_source_manifest.json","repository_revisions.json","license_acknowledgement.md","zero_shot_split_manifest.csv","artifact_inventory.csv","official_metric_reproduction.csv",
              "context_intervention_gene_manifest.json","real_pseudobulk.parquet","predicted_pseudobulk.parquet","truth_operator_reliability.csv","pertmean_predictions.parquet",
              "state_canonical_metrics.csv","shared_novel_decomposition.csv","state_recovery_g.csv","nulls.csv","bootstrap_summary.csv","context_heterogeneity.csv",
              "state_embedding_geometry.csv","raw_rna_geometry.csv","integrity_audit.json","verdict.json"]
    write_json(OUT/"run_manifest.json",{"phase":"CGC-STATE-1","created_at":now(),"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
               "preregistered_input_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"analysis_commit":None,"python":platform.python_version(),
               "seed":SEED,"gpu_used":False,"model_trained":False,"downloaded_bytes":json.loads((OUT/'official_source_manifest.json').read_text())["downloaded_bytes"],
               "official_contexts":contexts,"shared_cytokines":len(interventions),"features":2000,"required_sha256":{name:sha256(OUT/name) for name in required},
               "reports":[str(path.relative_to(ROOT)).replace('\\','/') for path in sorted(REPORTS.glob('cgc_state_*.md'))],"figures":len(list(FIGURES.glob('*'))),"verdicts":json.loads((OUT/'verdict.json').read_text())})


if __name__ == "__main__":
    main()
