from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.modeling import (
    _core_metrics,
    build_output_tables,
    fit_oof_platform,
    load_platform_data,
    primary_inference,
    run_primary_nulls,
)
from igc_virtual_cell.crc_pdo_personalized.reporting import make_figures, write_qa


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _null_summary(draws: pd.DataFrame) -> pd.DataFrame:
    observed = draws.query("null_family == 'observed_Ridge'").iloc[0]
    metrics = ["PG_macro", "g_func", "reversal_balanced_accuracy_minus_0.5"]
    records: list[dict[str, Any]] = []
    for family, family_frame in draws.groupby("null_family", sort=False):
        for metric in metrics:
            values = family_frame[metric].to_numpy(float)
            is_permutation = family in {"baseline_RNA_shuffle", "patient_response_profile_shuffle"}
            records.append(
                {
                    "null_family": family,
                    "metric": metric,
                    "observed": float(observed[metric]),
                    "null_draws": len(values),
                    "null_mean": float(np.mean(values)),
                    "null_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "null_95th_percentile": float(np.quantile(values, 0.95)),
                    "observed_minus_null_mean": float(observed[metric] - np.mean(values)),
                    "one_sided_empirical_p": (
                        float((1 + np.sum(values >= observed[metric])) / (len(values) + 1))
                        if is_permutation
                        else np.nan
                    ),
                    "held_patient_sealed": True,
                    "refit_predictions": is_permutation,
                }
            )
    return pd.DataFrame.from_records(records)


def _verdict(inference: pd.DataFrame, null_summary: pd.DataFrame) -> str:
    positive = inference.set_index("metric")["positive_after_joint_correction"].astype(bool)
    refit = null_summary[null_summary["null_family"].isin(["baseline_RNA_shuffle", "patient_response_profile_shuffle"])]
    p_table = refit.pivot(index="metric", columns="null_family", values="one_sided_empirical_p")
    null_specific = (p_table <= 0.05).all(axis=1)
    confirmed = positive & null_specific.reindex(positive.index).fillna(False)
    if confirmed.all():
        return "PATIENT_SPECIFIC_DRUG_PREFERENCES_RECOVERABLE"
    if confirmed.any():
        return "PATIENT_SPECIFIC_DRUG_PREFERENCES_PARTIALLY_RECOVERABLE"
    estimates = inference.set_index("metric")["estimate"]
    permutation_exceedance = (p_table > 0.05).all(axis=1)
    if (not positive.any()) and (
        ((estimates.loc[["PG_macro", "g_func"]] <= 0).all())
        or permutation_exceedance.reindex(["PG_macro", "g_func"]).fillna(False).all()
    ):
        return "POPULATION_DRUG_RANKING_DOMINATES_PERSONALIZATION"
    return "PERSONALIZED_APPLICATION_INCONCLUSIVE"


def _patient_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gain = tables["PERSONALIZATION_GAIN.csv"]
    recovery = tables["FUNCTIONAL_PERSONALIZATION_RECOVERY.csv"].query("row_type == 'patient'")
    topk = tables["TOPK_RECOVERY.csv"]
    reversals = tables["PREFERENCE_REVERSALS.csv"]
    records: list[dict[str, Any]] = []
    for keys, frame in gain.groupby(["platform", "estimator", "patient_id"], sort=False):
        platform, estimator, patient = keys
        rec = recovery.query("platform == @platform and estimator == @estimator and patient_id == @patient").iloc[0]
        rev = reversals.query("platform == @platform and estimator == @estimator and patient_id == @patient")
        actual = rev["true_reversal"].to_numpy(bool)
        predicted = rev["predicted_reversal"].to_numpy(bool)
        tp = int(np.sum(actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        fp = int(np.sum(~actual & predicted))
        tn = int(np.sum(~actual & ~predicted))
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        row: dict[str, Any] = {
            **dict(zip(gain.columns, frame.iloc[0].tolist(), strict=True)),
            "g_func_i": float(rec["g_func"]),
            "true_reversal_count": int(actual.sum()),
            "reversal_prevalence": float(actual.mean()),
            "reversal_recall": tp / (tp + fn) if tp + fn else np.nan,
            "reversal_precision": tp / (tp + fp) if tp + fp else np.nan,
            "reversal_balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else np.nan,
            "reversal_mcc": (tp * tn - fp * fn) / denominator if denominator else np.nan,
        }
        patient_topk = topk.query("platform == @platform and estimator == @estimator and patient_id == @patient")
        for _, top_row in patient_topk.iterrows():
            k = int(top_row["k"])
            row[f"top{k}_model_overlap_fraction"] = float(top_row["model_overlap_fraction"])
            row[f"top{k}_population_overlap_fraction"] = float(top_row["population_overlap_fraction"])
        records.append(row)
    return pd.DataFrame.from_records(records)


def _report(
    results_dir: Path,
    tables: dict[str, pd.DataFrame],
    verdict: str,
    prediction_implementation_commit: str,
    finalization_commit: str,
) -> None:
    gain = tables["PERSONALIZATION_GAIN.csv"]
    primary_gain = gain.query("platform == 'RNAseq' and estimator == 'Ridge'")
    inference = tables["PRIMARY_APPLICATION_INFERENCE.csv"].set_index("metric")
    recovery = tables["FUNCTIONAL_PERSONALIZATION_RECOVERY.csv"]
    reversals = tables["PREFERENCE_REVERSALS.csv"].query("platform == 'RNAseq' and estimator == 'Ridge'")
    near = tables["NEAR_TIE_REVERSAL_ANALYSIS.csv"].query("platform == 'RNAseq' and estimator == 'Ridge'")
    topk = tables["TOPK_RECOVERY.csv"].query("platform == 'RNAseq' and estimator == 'Ridge'")
    nulls = tables["NULL_CONTROLS.csv"]
    patient = tables["PATIENT_LEVEL_SUMMARY.csv"].query("platform == 'RNAseq' and estimator == 'Ridge'")
    actual = reversals["true_reversal"].to_numpy(bool)
    predicted = reversals["predicted_reversal"].to_numpy(bool)
    tp, fn = int(np.sum(actual & predicted)), int(np.sum(actual & ~predicted))
    fp, tn = int(np.sum(~actual & predicted)), int(np.sum(~actual & ~predicted))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else np.nan
    ba = 0.5 * (recall + tn / (tn + fp))
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else np.nan
    primary_g = float(recovery.query("platform == 'RNAseq' and estimator == 'Ridge' and row_type == 'patient_balanced_macro'")["g_func"].iloc[0])
    secondary = gain.query("platform == 'HTA2.0' and estimator == 'Ridge'")
    top_lines = []
    for k, frame in topk.groupby("k"):
        top_lines.append(
            f"- Top-{int(k)} mean overlap: model {frame['model_overlap_fraction'].mean():.3f}; population {frame['population_overlap_fraction'].mean():.3f}."
        )
    near_lines = []
    for _, row in near.iterrows():
        near_lines.append(
            f"- {row['population_margin_bin']}: {int(row['true_reversals'])}/{int(row['valid_pairs'])} true reversals; recall {row['reversal_recall']:.3f}; balanced accuracy {row['balanced_accuracy']:.3f}."
        )
    null_lines = []
    for family in ("baseline_RNA_shuffle", "patient_response_profile_shuffle"):
        frame = nulls[nulls["null_family"].eq(family)].set_index("metric")
        null_lines.append(
            f"- {family}: PG p={frame.loc['PG_macro', 'one_sided_empirical_p']:.4f}; g_func p={frame.loc['g_func', 'one_sided_empirical_p']:.4f}; reversal p={frame.loc['reversal_balanced_accuracy_minus_0.5', 'one_sided_empirical_p']:.4f}."
        )
    lines = [
        "# CRC PDO personalized ex vivo drug-prioritization application",
        "",
        f"**Final scientific verdict: `{verdict}`**",
        "",
        "## Scope and frozen design",
        "",
        "Kryeziu et al. 2026 provides 213 PDOs from 102 metastatic colorectal cancer patients. Data S5 contains author DSS for 211 PDOs. The outcome-blind audit froze 24 harmonized, completely measured drugs before modeling. The primary RNA-seq intersection contains 91 PDOs from 52 patients; the secondary HTA2.0 intersection contains 118 PDOs from 50 patients. Multiple PDOs are averaged within patient after requiring both expression and complete DSS.",
        "",
        "The primary estimator is multioutput Ridge; PCA20+Ridge is secondary. Evaluation is strict leave-one-patient-out. Every outer fold recomputes the expression filter, scaling, nested five-fold alpha selection, patient-balanced drug means, and predictions from training patients only. The endpoint is ex vivo DSS, not patient treatment response.",
        "",
        "## Primary RNA-seq results",
        "",
        f"- Population Kendall tau-b: {primary_gain['kendall_population'].mean():.4f}; personalized Ridge: {primary_gain['kendall_model'].mean():.4f}.",
        f"- Population Spearman rho: {primary_gain['spearman_population'].mean():.4f}; personalized Ridge: {primary_gain['spearman_model'].mean():.4f}.",
    ]
    for metric in ["PG_macro", "g_func", "reversal_balanced_accuracy_minus_0.5"]:
        row = inference.loc[metric]
        lines.append(
            f"- {metric}: {row['estimate']:.6f}, simultaneous 95% CI [{row['simultaneous_95_ci_lower']:.6f}, {row['simultaneous_95_ci_upper']:.6f}], one-sided max-T adjusted p={row['one_sided_maxT_adjusted_p']:.4f}."
        )
    lines.extend(
        [
            f"- Functional personalization recovery g_func: {primary_g:.6f}.",
            f"- True patient-specific reversals: {int(actual.sum())}/{len(actual)} ({actual.mean():.3%}); recall {recall:.3f}; precision {precision:.3f}; balanced accuracy {ba:.3f}; MCC {mcc:.3f}.",
            f"- Patient heterogeneity: PG_i > 0 in {(patient['PG_i'] > 0).sum()}/{len(patient)} patients; g_func_i > 0 in {(patient['g_func_i'] > 0).sum()}/{len(patient)} patients.",
            "",
            "## Prespecified margin and top-k analyses",
            "",
            *near_lines,
            *top_lines,
            "",
            "## Held-patient correspondence nulls",
            "",
            *null_lines,
            "- Population-only and baseline-free drug-identity models are identical by design to the held-fold training-patient drug mean.",
            "",
            "## Secondary platform and estimator checks",
            "",
            f"- HTA2.0 Ridge: population Kendall {secondary['kendall_population'].mean():.4f}; model Kendall {secondary['kendall_model'].mean():.4f}; PG_macro {secondary['PG_i'].mean():.6f}.",
            "- PCA20+Ridge is reported as a prespecified secondary estimator in all output tables; it is not a rescue or a second confirmatory family.",
            "",
            "## Interpretation and claim boundary",
            "",
        ]
    )
    if verdict == "PATIENT_SPECIFIC_DRUG_PREFERENCES_RECOVERABLE":
        lines.append("Baseline RNA robustly recovered patient-specific ex vivo drug-preference information beyond population-average drug effects across all three corrected primary endpoints.")
    elif verdict == "PATIENT_SPECIFIC_DRUG_PREFERENCES_PARTIALLY_RECOVERABLE":
        lines.append("Baseline RNA reproducibly recovered only part of the patient-specific ex vivo drug-preference information; the unresolved primary endpoints bound the application claim.")
    elif verdict == "POPULATION_DRUG_RANKING_DOMINATES_PERSONALIZATION":
        lines.append("The frozen personalized models did not materially exceed the held-fold population drug ranking. In this cohort and representation, most patient-specific ex vivo drug-preference information remains compressed relative to population drug effects.")
    else:
        lines.append("The primary endpoints are not jointly decisive: the frozen analysis does not establish robust patient-specific ex vivo drug-prioritization beyond population drug effects.")
    lines.extend(
        [
            "",
            "These results concern patient-derived organoid DSS only. They do not establish therapeutic efficacy or constitute patient treatment advice.",
            "",
            "## Provenance and QA",
            "",
            "- Panel/feasibility freeze: `a534a99`.",
            f"- Prespecified prediction implementation freeze: `{prediction_implementation_commit}`.",
            f"- Final inference implementation: `{finalization_commit}` (permutation t statistics are re-studentized within every shared sign-flip draw).",
            "- No raw reads or GSE294511_RAW.tar were downloaded.",
            "- All held-patient overlaps are zero; every required prediction is finite.",
            "- Primary inference uses 10,000 patient bootstrap draws and 10,000 shared sign-flip max-T draws; both training-side shuffle nulls use 1,000 complete LOPO refit draws.",
            "",
        ]
    )
    (results_dir / "CRC_PDO_PERSONALIZED_APPLICATION_FINAL.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    results_dir = root / "results" / "crc_pdo_personalized_drug_application"
    processed_dir = root / "data" / "crc_pdo_personalized_application" / "processed"
    config = json.loads((root / "configs" / "crc_pdo_personalized_application.json").read_text())
    finalization_commit = _git(root, "rev-parse", "HEAD")
    prediction_implementation_commit = "4e6600c62103ab5c3b6a6a26d8aa096661ea3801"

    if args.postprocess_only:
        table_names = [
            "HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv",
            "HELD_PATIENT_DRUG_RANKINGS.csv",
            "POPULATION_REFERENCE_RANKINGS.csv",
            "PERSONALIZATION_GAIN.csv",
            "FUNCTIONAL_PERSONALIZATION_RECOVERY.csv",
            "PREFERENCE_REVERSALS.csv",
            "NEAR_TIE_REVERSAL_ANALYSIS.csv",
            "TOPK_RECOVERY.csv",
            "MODEL_FOLD_MANIFEST.csv",
            "PATIENT_LEVEL_SUMMARY.csv",
        ]
        tables = {name: pd.read_csv(results_dir / name) for name in table_names}
        null_draws = pd.read_csv(results_dir / "NULL_CONTROL_DRAWS.csv")
        primary_frame = tables["HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv"].query(
            "platform == 'RNAseq' and estimator == 'Ridge'"
        )
        patient_order = primary_frame["patient_id"].drop_duplicates().tolist()
        drug_order = primary_frame["drug_name"].drop_duplicates().tolist()
        def matrix(column: str) -> np.ndarray:
            return primary_frame.pivot(index="patient_id", columns="drug_name", values=column).loc[
                patient_order, drug_order
            ].to_numpy(float)
        truth = matrix("true_DSS")
        personalized = matrix("predicted_DSS")
        population = matrix("population_DSS")
    else:
        platform_results = []
        fold_tables = []
        primary_caches = None
        for platform in ("RNAseq", "HTA2.0"):
            data = load_platform_data(processed_dir, platform)
            predictions, folds, caches = fit_oof_platform(data, config)
            platform_results.append((data, predictions))
            fold_tables.append(folds)
            if platform == "RNAseq":
                primary_caches = caches

        tables = build_output_tables(platform_results, config)
        tables["MODEL_FOLD_MANIFEST.csv"] = pd.concat(fold_tables, ignore_index=True)
        primary_data, primary_predictions = platform_results[0]
        null_draws = run_primary_nulls(
            primary_data,
            primary_caches,
            primary_predictions["Ridge"],
            primary_predictions["Population"],
            config,
        )
        tables["PATIENT_LEVEL_SUMMARY.csv"] = _patient_summary(tables)
        truth = primary_data.y
        personalized = primary_predictions["Ridge"]
        population = primary_predictions["Population"]

    tables["NULL_CONTROLS.csv"] = _null_summary(null_draws)
    inference, _ = primary_inference(
        truth,
        personalized,
        population,
        config,
    )
    tables["PRIMARY_APPLICATION_INFERENCE.csv"] = inference
    for name in [
        "CRC_PDO_PATIENT_MANIFEST.csv",
        "CRC_PDO_DRUG_COVERAGE.csv",
        "CRC_PDO_EXPRESSION_COVERAGE.csv",
        "PATIENT_PDO_EXPRESSION_DRUG_INTERSECTION.csv",
        "PRIMARY_DRUG_PANEL_MANIFEST.csv",
    ]:
        tables[name] = pd.read_csv(results_dir / name)
    frozen_audit_tables = {
        "CRC_PDO_PATIENT_MANIFEST.csv",
        "CRC_PDO_DRUG_COVERAGE.csv",
        "CRC_PDO_EXPRESSION_COVERAGE.csv",
        "PATIENT_PDO_EXPRESSION_DRUG_INTERSECTION.csv",
        "PRIMARY_DRUG_PANEL_MANIFEST.csv",
    }
    for name, table in tables.items():
        if name not in frozen_audit_tables:
            table.to_csv(results_dir / name, index=False)
    null_draws.to_csv(results_dir / "NULL_CONTROL_DRAWS.csv", index=False)

    verdict = _verdict(inference, tables["NULL_CONTROLS.csv"])
    _report(
        results_dir,
        tables,
        verdict,
        prediction_implementation_commit,
        finalization_commit,
    )
    make_figures(results_dir, tables)
    source_manifest = json.loads((root / "data" / "crc_pdo_personalized_application" / "source_manifest.json").read_text())
    qa = write_qa(
        results_dir,
        source_manifest=source_manifest,
        panel_freeze_commit="a534a99",
        prediction_implementation_commit=prediction_implementation_commit,
        finalization_commit=finalization_commit,
        verdict=verdict,
        tables=tables,
    )
    if not qa["qa_pass"]:
        raise RuntimeError("Final QA failed")
    print(json.dumps({"verdict": verdict, "qa_pass": True, "results_dir": str(results_dir)}, indent=2))


if __name__ == "__main__":
    main()
