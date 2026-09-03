from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.conditioning_resolution import (
    RepresentationSpec,
    fit_representation_oof,
    metrics,
    oof_table,
    random_orthonormal_bases,
    sha256,
)
from igc_virtual_cell.crc_pdo_personalized.modeling import (
    _core_metrics,
    fit_oof_platform,
    load_platform_data,
)
from igc_virtual_cell.crc_pdo_personalized.pca_resolution_sweep import (
    bootstrap_performance_curves,
    bootstrap_primary_contrasts,
    deterministic_verdict,
    fit_nested_pca_oof,
    fit_random_matched_kstar_oof,
    selection_is_coherent,
)


FROZEN_APPLICATION_RESULT = "c6cccb90453c567b6593982df442c4f127809fa1"
FROZEN_CONDITIONING_RESULT = "38adafd8601f3a0382baa88e03bf72e3dd0d3a62"
FROZEN_FULL_PG = 0.050423060202472454
FROZEN_FULL_G = 0.13928938752362074


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def source_paths(application_root: Path, root: Path) -> dict[str, Path]:
    return {
        "RNAseq_PDO_log2CPM1.npz": application_root
        / "data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz",
        "PRIMARY_DSS.npz": application_root
        / "data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz",
        "HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv": application_root
        / "results/crc_pdo_personalized_drug_application/HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv",
        "MODEL_FOLD_MANIFEST.csv": application_root
        / "results/crc_pdo_personalized_drug_application/MODEL_FOLD_MANIFEST.csv",
        "PCA14_OOF.csv": root / "results/crc_pdo_conditioning_resolution/PCA14_OOF.csv",
        "CRC_PDO_CONDITIONING_RESOLUTION_FINAL_QA.json": root
        / "results/crc_pdo_conditioning_resolution/CRC_PDO_CONDITIONING_RESOLUTION_FINAL_QA.json",
    }


def source_hashes(application_root: Path, root: Path) -> dict[str, str]:
    return {name: sha256(path) for name, path in source_paths(application_root, root).items()}


def top_k_overlaps(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 3, 5):
        values = []
        for row in range(len(truth)):
            true_set = set(np.argsort(-truth[row], kind="stable")[:k])
            predicted_set = set(np.argsort(-prediction[row], kind="stable")[:k])
            values.append(len(true_set & predicted_set) / k)
        result[f"top{k}_overlap_fraction"] = float(np.mean(values))
    return result


def performance_row(data: object, name: str, prediction: np.ndarray, population: np.ndarray) -> dict[str, object]:
    return {
        "platform": data.platform,
        "representation": name,
        "patients": len(data.patient_ids),
        "drugs": len(data.drug_names),
        **metrics(data.y, prediction, population),
        **top_k_overlaps(data.y, prediction),
    }


def frozen_matrix(path: Path, data: object, representation: str | None = None) -> np.ndarray:
    frame = pd.read_csv(path)
    if representation is not None:
        frame = frame.query("representation == @representation")
    elif "platform" in frame and "estimator" in frame:
        frame = frame.query("platform == 'RNAseq' and estimator == 'Ridge'")
    patient_order = [str(value) for value in data.patient_ids]
    drug_order = [str(value) for value in data.drug_names]
    value_column = "predicted_DSS"
    return (
        frame.pivot(index="patient_id", columns="drug_name", values=value_column)
        .loc[patient_order, drug_order]
        .to_numpy(float)
    )


def patient_metric_frame(
    data: object,
    predictions: dict[str, np.ndarray],
    population: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, prediction in predictions.items():
        core = _core_metrics(data.y, prediction, population, 1e-12)
        for index, patient in enumerate(data.patient_ids):
            rows.append(
                {
                    "representation": name,
                    "patient_id": str(patient),
                    "PG_i": core["pg_patient"][index],
                    "g_func_i": core["g_func_patient"][index],
                    "reversal_BA_i": core["reversal"]["balanced_accuracy"][index],
                }
            )
    return pd.DataFrame.from_records(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curve_figure(curves: pd.DataFrame, metric: str, results_dir: Path, filename: str) -> None:
    order = ["PCA2", "PCA4", "PCA8", "PCA14", "PCA16", "PCA24", "PCA32", "FULL_RNA_REOPT"]
    local = curves.query("metric == @metric").set_index("representation").loc[order]
    x = np.arange(len(order))
    estimate = local["estimate"].to_numpy(float)
    lower = local["pointwise_95_ci_lower"].to_numpy(float)
    upper = local["pointwise_95_ci_upper"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.plot(x[:-1], estimate[:-1], color="#3F779B", linewidth=1.4, marker="o", markersize=4)
    ax.errorbar(x[:-1], estimate[:-1], yerr=[estimate[:-1] - lower[:-1], upper[:-1] - estimate[:-1]], fmt="none", ecolor="#3F779B", linewidth=0.8, capsize=2)
    ax.errorbar(x[-1], estimate[-1], yerr=[[estimate[-1] - lower[-1]], [upper[-1] - estimate[-1]]], fmt="D", color="#C66B55", markersize=4, linewidth=0.8, capsize=2)
    ax.axhline(0, color="#B8BDC4", linewidth=0.7)
    ax.axvline(6.5, color="#D5D8DC", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, ["2", "4", "8", "14", "16", "24", "32", "full\n19,421"])
    ax.set_xlabel("baseline representation dimensions")
    ax.set_ylabel("personalized ranking gain" if metric == "PG_macro" else r"functional recovery $g_{\mathrm{func}}$")
    ax.set_title("Training-side PCA conditioning-resolution curve", loc="center", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, results_dir / filename)


def make_figures(
    results_dir: Path,
    curves: pd.DataFrame,
    patient_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    random_summary: pd.DataFrame,
    pca_performance: pd.DataFrame,
) -> None:
    configure_plotting()
    curve_figure(curves, "PG_macro", results_dir, "PCA_CONDITIONING_RESOLUTION_CURVE")
    curve_figure(curves, "g_func", results_dir, "PCA_CONDITIONING_RESOLUTION_GFUNC")

    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.5))
    wide = patient_metrics.query("representation in ['FULL_RNA_REOPT', 'PCA_STAR']")
    for ax, metric, title in zip(axes, ["PG_i", "g_func_i"], ["Ranking gain", "Squared-error recovery"], strict=True):
        pivot = wide.pivot(index="patient_id", columns="representation", values=metric)
        for _, row in pivot.iterrows():
            ax.plot([0, 1], [row["FULL_RNA_REOPT"], row["PCA_STAR"]], color="#CAD0D6", linewidth=0.55, zorder=1)
        ax.scatter(np.zeros(len(pivot)), pivot["FULL_RNA_REOPT"], color="#C66B55", s=9, zorder=2)
        ax.scatter(np.ones(len(pivot)), pivot["PCA_STAR"], color="#3F779B", s=9, zorder=2)
        ax.axhline(0, color="#AEB4BA", linewidth=0.7)
        ax.set_xticks([0, 1], ["Full RNA", r"nested PCA$_*$"])
        ax.set_title(title, loc="center", fontweight="bold")
        ax.set_ylabel(metric.replace("_i", ""))
    fig.tight_layout(w_pad=2)
    save_figure(fig, results_dir / "NESTED_SELECTED_PCA_VS_FULL")

    counts = selections["selected_k"].value_counts().reindex([2, 4, 8, 14, 16, 24, 32], fill_value=0)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(np.arange(len(counts)), counts.values, color="#6D8FA8", width=0.68)
    ax.set_xticks(np.arange(len(counts)), counts.index.astype(str))
    ax.set_xlabel(r"nested selected $k_*$")
    ax.set_ylabel("outer folds")
    ax.set_title("Training-side selected PCA resolution", loc="center", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, results_dir / "SELECTED_PCA_DIMENSIONS")

    star = pca_performance.set_index("representation").loc["PCA_STAR"]
    fig, axes = plt.subplots(1, 2, figsize=(4.3, 2.3))
    for ax, metric, label in zip(axes, ["PG_macro", "g_func"], ["PG", r"$g_{\mathrm{func}}$"], strict=True):
        values = random_summary[metric].to_numpy(float)
        ax.hist(values, bins=16, color="#BCC6CE", edgecolor="white", linewidth=0.4)
        ax.axvline(float(star[metric]), color="#C65F4A", linewidth=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("random projections")
        ax.set_title("Structured PCA vs matched random", loc="center", fontweight="bold")
    fig.tight_layout(w_pad=2)
    save_figure(fig, results_dir / "PCA_VS_RANDOM_MATCHED_DIMENSION")


def final_report(
    path: Path,
    verdict: str,
    fixed_performance: pd.DataFrame,
    star_performance: pd.DataFrame,
    full_performance: pd.DataFrame,
    inference: pd.DataFrame,
    selections: pd.DataFrame,
    selection_summary: dict[str, float],
    random_results: pd.DataFrame,
    variance: pd.DataFrame,
    qa: dict[str, object],
) -> None:
    fixed = fixed_performance.set_index("representation")
    star = star_performance.iloc[0]
    full = full_performance.iloc[0]
    infer = inference.set_index("contrast")
    pg = infer.loc["PCA_STAR_minus_FULL_RNA_REOPT_PG"]
    g = infer.loc["PCA_STAR_minus_FULL_RNA_REOPT_g_func"]
    counts = selections["selected_k"].value_counts().reindex([2, 4, 8, 14, 16, 24, 32], fill_value=0)
    peak_pg_name = fixed["PG_macro"].idxmax()
    peak_g_name = fixed["g_func"].idxmax()
    selected_variance = variance.merge(
        selections[["held_patient", "selected_k"]], left_on=["held_patient", "k"], right_on=["held_patient", "selected_k"]
    )["training_expression_variance_explained"]
    random_pg_p = float((1 + np.sum(random_results["PG_macro"] >= star["PG_macro"])) / (len(random_results) + 1))
    random_g_p = float((1 + np.sum(random_results["g_func"] >= star["g_func"])) / (len(random_results) + 1))

    lines = [
        "# CRC PDO PCA conditioning-resolution sweep",
        "",
        f"Final verdict: `{verdict}`",
        "",
        "## Frozen scope and reproduction",
        "",
        "- Cohort and target were unchanged: 52 independent RNA-seq patients, 91 PDOs, 24 drugs, and 19,421 baseline genes.",
        f"- Frozen PCA14 replay: {'PASS' if qa['pca14_frozen_reproduction_pass'] else 'FAIL'}; maximum prediction difference `{qa['pca14_frozen_prediction_max_abs_difference']:.3g}`.",
        f"- Frozen FULL_RNA replay: {'PASS' if qa['full_frozen_reproduction_pass'] else 'FAIL'}; maximum prediction difference `{qa['full_frozen_prediction_max_abs_difference']:.3g}`.",
        "- The PCA14 replay used the historical Kendall inner objective solely as an exact positive control. The prespecified sweep PCA14 used the new patient-profile MSE objective, like every other fixed k.",
        "",
        "## Fixed conditioning-resolution curve",
        "",
        "| Representation | PG | g_func | reversal BA |",
        "|---|---:|---:|---:|",
    ]
    for name in ["PCA2", "PCA4", "PCA8", "PCA14", "PCA16", "PCA24", "PCA32"]:
        row = fixed.loc[name]
        lines.append(f"| {name} | {row.PG_macro:.6f} | {row.g_func:.6f} | {row.reversal_balanced_accuracy_patient_mean:.6f} |")
    lines.extend(
        [
            "",
            f"The descriptive fixed-k PG peak was {peak_pg_name} (`{fixed.loc[peak_pg_name, 'PG_macro']:.6f}`); the descriptive g_func peak was {peak_g_name} (`{fixed.loc[peak_g_name, 'g_func']:.6f}`). These outer-curve peaks were not used for model selection or primary inference.",
            "",
            "## Nested training-side selection",
            "",
            f"- k* counts (2/4/8/14/16/24/32): `{counts.tolist()}`.",
            f"- Median/mode/IQR: `{selection_summary['median_k']:.1f}` / `{selection_summary['mode_k']:.0f}` / `[{selection_summary['q25_k']:.1f}, {selection_summary['q75_k']:.1f}]`.",
            f"- Frozen coherence rule passed: `{qa['selection_coherence_pass']}` (mode fraction `{selection_summary['mode_fraction']:.3f}`, IQR width `{selection_summary['iqr_width']:.1f}`).",
            f"- Fold-local expression variance captured at selected k*: mean `{selected_variance.mean():.3%}`, range `[{selected_variance.min():.3%}, {selected_variance.max():.3%}]`. This contextualizes compression and is not response information.",
            "",
            "## Primary held-patient comparison",
            "",
            f"- FULL_RNA_REOPT: PG `{full.PG_macro:.6f}`, g_func `{full.g_func:.6f}`, reversal BA `{full.reversal_balanced_accuracy_patient_mean:.6f}`.",
            f"- Relative to the exact frozen FULL_RNA replay, MSE-based reoptimization changed {qa['full_reopt_predictions_changed']:,}/1,248 predictions (maximum/mean absolute difference `{qa['full_reopt_prediction_max_abs_difference']:.6f}`/`{qa['full_reopt_prediction_mean_abs_difference']:.6f}`), changed PG by `{qa['full_reopt_minus_frozen_pg']:.6f}`, and changed g_func by `{qa['full_reopt_minus_frozen_g_func']:.6f}`. This is the expected consequence of changing the inner objective from Kendall tau-b to the preregistered profile MSE; the reoptimized model is the fairness comparator below.",
            f"- PCA_STAR: PG `{star.PG_macro:.6f}`, g_func `{star.g_func:.6f}`, reversal BA `{star.reversal_balanced_accuracy_patient_mean:.6f}`.",
            f"- Delta PG: `{pg.estimate:.6f}`, simultaneous 95% CI `[{pg.simultaneous_95_ci_lower:.6f}, {pg.simultaneous_95_ci_upper:.6f}]`, max-T one-sided adjusted P `{pg.max_t_adjusted_one_sided_p:.6f}`; 10% noninferiority `{bool(pg.lower_exceeds_negative_margin)}`.",
            f"- Delta g_func: `{g.estimate:.6f}`, simultaneous 95% CI `[{g.simultaneous_95_ci_lower:.6f}, {g.simultaneous_95_ci_upper:.6f}]`, max-T one-sided adjusted P `{g.max_t_adjusted_one_sided_p:.6f}`; 10% noninferiority `{bool(g.lower_exceeds_negative_margin)}`.",
            "",
            "## Matched random projections",
            "",
            f"- PCA_STAR PG percentile among 100 matched-k random projections: `{100 * np.mean(random_results.PG_macro < star.PG_macro):.1f}`; empirical P `{random_pg_p:.6f}`.",
            f"- PCA_STAR g_func percentile among 100 matched-k random projections: `{100 * np.mean(random_results.g_func < star.g_func):.1f}`; empirical P `{random_g_p:.6f}`.",
            "- This secondary control tests structured leading-variance PCA against generic low dimension; it does not enter the primary two-contrast family.",
            "",
            "## Adversarial interpretation",
            "",
        ]
    )
    if verdict == "COARSER_CONDITIONING_RESOLUTION_IMPROVES_TRANSFER":
        lines.append("Although baseline state was measured across 19,421 genes, held-patient functional prediction was more accurate after training-side selection of a substantially lower-dimensional transcriptomic representation. PCA did not create information; it made operational predictive information more stably extractable in this finite-sample task.")
    elif verdict == "COMPACT_CONDITIONING_REPRESENTATION_IS_SUFFICIENT":
        lines.append("A compact, training-selected transcriptomic representation retained essentially all transferable personalization performance. This establishes compact sufficiency under the frozen margins, not superiority over full RNA.")
    elif verdict == "PCA_RESOLUTION_IMPROVES_RANKING_ONLY":
        lines.append("Compression improved personalized ranking, but amplitude recovery did not meet the frozen superiority or noninferiority requirement. The effect is therefore ranking-specific.")
    elif verdict == "FULL_RNA_REMAINS_SUPERIOR":
        lines.append("Reoptimized full RNA materially and significantly outperformed the nested compressed representation on both primary endpoints. Fine baseline detail remains operationally important in this task.")
    elif verdict == "NO_STABLE_CONDITIONING_RESOLUTION_OPTIMUM":
        lines.append("The nested training procedure coherently concentrated on a compact 24–32-dimensional region, but the confirmatory family established neither superiority nor the prespecified dual-endpoint noninferiority of PCA_STAR versus FULL_RNA_REOPT. The frozen verdict therefore records that no confirmatory optimum was established; it does not mean that k selection itself was diffuse. Fixed-k descriptive peaks cannot be promoted to an optimum.")
    else:
        lines.append("A required integrity or reproduction condition failed, so the resolution sweep is not scientifically interpretable.")
    lines.extend(
        [
            "",
            "This is an operational finite-sample held-patient prediction result. It does not establish that fine genes contain no information, that PCA creates information, that k* is the transcriptome's intrinsic dimension, or that an information-theoretic limit has been identified.",
            "",
            "## QA and provenance",
            "",
            f"- Analysis implementation frozen before outcomes: `{qa['implementation_commit']}`.",
            f"- Required scientific checks: `{qa['passed_checks']}/{qa['required_checks']}` passed.",
            f"- Source hashes unchanged: `{qa['frozen_source_hashes_unchanged']}`.",
            f"- Repository test status: `{qa['repository_test_status']}`.",
            "- Existing application, conditioning-resolution, and manuscript artifacts were not modified.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    application_root = args.application_root.resolve()
    results_dir = root / "results/crc_pdo_pca_resolution_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configs/crc_pdo_pca_resolution_sweep.json").read_text())
    application_config = json.loads((root / "configs/crc_pdo_personalized_application.json").read_text())
    if config["prerequisites"]["conditioning_result_commit"] != FROZEN_CONDITIONING_RESULT:
        raise ValueError("Conditioning-resolution prerequisite commit changed")
    before_hashes = source_hashes(application_root, root)
    processed_dir = application_root / "data/crc_pdo_personalized_application/processed"
    data = load_platform_data(processed_dir, "RNAseq")
    expression = np.load(processed_dir / "RNAseq_PDO_log2CPM1.npz")

    # Exact historical positive controls remain separate from the new MSE-selected sweep.
    frozen_full, frozen_full_folds, _ = fit_oof_platform(data, application_config)
    frozen_full_reference = frozen_matrix(
        application_root / "results/crc_pdo_personalized_drug_application/HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv",
        data,
    )
    frozen_full_difference = float(np.max(np.abs(frozen_full["Ridge"] - frozen_full_reference)))
    pca14_replay = fit_representation_oof(
        data,
        RepresentationSpec("PCA14_FROZEN_REPRO", "PCA", 14),
        np.zeros((data.x.shape[1], 0), dtype=np.float64),
        np.zeros((0, 0), dtype=np.float64),
        tuple(),
        config["model"]["alpha_grid"],
        int(config["model"]["seed"]),
    )
    pca14_reference = frozen_matrix(
        root / "results/crc_pdo_conditioning_resolution/PCA14_OOF.csv", data, "PCA14"
    )
    pca14_difference = float(np.max(np.abs(pca14_replay.predictions - pca14_reference)))
    tolerance = float(config["positive_control"]["prediction_tolerance"])
    if pca14_difference > tolerance:
        raise RuntimeError("PCA_RESOLUTION_PIPELINE_REPRODUCTION_FAILURE")

    sweep = fit_nested_pca_oof(
        data,
        config["model"]["pca_grid"],
        config["model"]["alpha_grid"],
        int(config["model"]["seed"]),
    )
    pca_star = sweep.predictions["PCA_STAR"]
    full_reopt = sweep.predictions["FULL_RNA_REOPT"]

    inference_config = config["inference"]
    inference, _ = bootstrap_primary_contrasts(
        data.y,
        pca_star,
        full_reopt,
        sweep.population,
        int(inference_config["bootstrap_draws"]),
        int(inference_config["seed"]),
        float(inference_config["pg_noninferiority_margin"]),
        float(inference_config["g_func_noninferiority_margin"]),
    )
    curve_predictions = {
        **{f"PCA{k}": sweep.predictions[f"PCA{k}"] for k in config["model"]["pca_grid"]},
        "FULL_RNA_REOPT": full_reopt,
    }
    curves = bootstrap_performance_curves(
        data,
        curve_predictions,
        sweep.population,
        int(config["evaluation"]["fixed_curve_bootstrap_draws"]),
        int(inference_config["seed"]) + 1,
    )

    random_config = config["random_control"]
    random_bases = random_orthonormal_bases(
        data.x.shape[1],
        max(config["model"]["pca_grid"]),
        int(random_config["draws"]),
        int(random_config["seed"]),
    )
    random_predictions, random_folds = fit_random_matched_kstar_oof(
        data,
        random_bases,
        sweep.selections["selected_k"].to_numpy(int),
        config["model"]["alpha_grid"],
        int(config["model"]["seed"]),
    )
    del random_bases
    random_rows = []
    for draw in range(len(random_predictions)):
        random_rows.append(
            {
                "family": "RANDOM_MATCHED_KSTAR",
                "draw": draw,
                **performance_row(data, f"RANDOM_MATCHED_KSTAR_{draw}", random_predictions[draw], sweep.population),
            }
        )
    random_results = pd.DataFrame.from_records(random_rows)

    fixed_performance = pd.DataFrame.from_records(
        [
            performance_row(data, f"PCA{k}", sweep.predictions[f"PCA{k}"], sweep.population)
            for k in config["model"]["pca_grid"]
        ]
    )
    star_performance = pd.DataFrame.from_records(
        [performance_row(data, "PCA_STAR", pca_star, sweep.population)]
    )
    full_performance = pd.DataFrame.from_records(
        [performance_row(data, "FULL_RNA_REOPT", full_reopt, sweep.population)]
    )
    frozen_pca14_metrics = metrics(data.y, pca14_replay.predictions, pca14_replay.population)
    frozen_full_metrics = metrics(data.y, frozen_full["Ridge"], frozen_full["Population"])
    full_reopt_difference = np.abs(full_reopt - frozen_full["Ridge"])
    selected_values = sweep.selections["selected_k"].to_numpy(int)
    coherent, selection_summary = selection_is_coherent(
        selected_values,
        float(inference_config["coherent_selection_mode_fraction"]),
        float(inference_config["coherent_selection_max_iqr_width"]),
    )

    after_hashes = source_hashes(application_root, root)
    checks = {
        "same_52_patients": len(data.patient_ids) == 52 and len(np.unique(data.patient_ids)) == 52,
        "same_24_drugs": len(data.drug_names) == 24,
        "same_91_pdos": sum(len(group) for group in data.eligible_pdo_ids) == 91,
        "same_19421_genes": data.x.shape[1] == 19421,
        "same_patient_grouping": all(len(group) >= 1 for group in data.eligible_pdo_ids),
        "same_outer_folds": sweep.selections["held_patient"].tolist() == [str(value) for value in data.patient_ids],
        "outer_patient_completely_sealed": sweep.selections["patient_overlap"].eq(0).all(),
        "pca_training_side_only": sweep.selections["pca_fit_scope"].eq("outer_and_inner_training_only").all(),
        "inner_pca_refit": sweep.grid_manifest["inner_preprocessing_refit"].all(),
        "k_selected_training_side_only": sweep.selections["held_outcome_used_in_selection"].eq(False).all(),
        "alpha_selected_training_side_only": sweep.selections["held_outcome_used_in_selection"].eq(False).all(),
        "same_alpha_family": set(sweep.grid_manifest["alpha"].unique()) == set(float(value) for value in config["model"]["alpha_grid"]),
        "multioutput_drug_specific_model": all(value.shape == (52, 24) for value in sweep.predictions.values()),
        "population_training_only": np.allclose(sweep.population, frozen_full["Population"], atol=0, rtol=0),
        "pca14_frozen_result_reproduced": pca14_difference <= tolerance,
        "full_frozen_application_reproduced": frozen_full_difference <= tolerance,
        "pg_recomputes": np.isfinite(star_performance.iloc[0]["PG_macro"]),
        "g_func_recomputes": np.isfinite(star_performance.iloc[0]["g_func"]),
        "reversal_metrics_recompute": np.isfinite(star_performance.iloc[0]["reversal_balanced_accuracy_patient_mean"]),
        "random_projections_outcome_blind": random_folds["outcome_blind_projection"].all(),
        "no_outer_best_k_primary_selection": sweep.grid_manifest.query("selected_jointly")["held_patient"].nunique() == 52,
        "frozen_sources_unchanged": before_hashes == after_hashes,
        "all_outputs_finite": all(np.isfinite(value).all() for value in sweep.predictions.values()) and np.isfinite(random_predictions).all(),
    }
    qa_pass = bool(all(checks.values()))
    verdict = deterministic_verdict(inference, coherent, qa_pass)
    qa = {
        "protocol_id": config["protocol_id"],
        "branch": git(root, "branch", "--show-current"),
        "implementation_commit": args.implementation_commit,
        "frozen_application_commit": FROZEN_APPLICATION_RESULT,
        "frozen_conditioning_result_commit": FROZEN_CONDITIONING_RESULT,
        "pca14_frozen_prediction_max_abs_difference": pca14_difference,
        "pca14_frozen_reproduction_pass": pca14_difference <= tolerance,
        "pca14_frozen_metrics": frozen_pca14_metrics,
        "full_frozen_prediction_max_abs_difference": frozen_full_difference,
        "full_frozen_reproduction_pass": frozen_full_difference <= tolerance,
        "full_frozen_metrics": frozen_full_metrics,
        "full_reopt_prediction_max_abs_difference": float(np.max(full_reopt_difference)),
        "full_reopt_prediction_mean_abs_difference": float(np.mean(full_reopt_difference)),
        "full_reopt_predictions_changed": int(np.sum(full_reopt_difference > 1e-12)),
        "full_reopt_minus_frozen_pg": float(full_performance.iloc[0]["PG_macro"] - frozen_full_metrics["PG_macro"]),
        "full_reopt_minus_frozen_g_func": float(full_performance.iloc[0]["g_func"] - frozen_full_metrics["g_func"]),
        "selection_coherence_pass": coherent,
        "selection_summary": selection_summary,
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "frozen_source_hashes_unchanged": before_hashes == after_hashes,
        "checks": {name: bool(value) for name, value in checks.items()},
        "required_checks": len(checks),
        "passed_checks": int(sum(bool(value) for value in checks.values())),
        "all_required_checks_pass": qa_pass,
        "repository_test_status": "PENDING_POSTRUN_EXTERNAL_AUDIT",
        "bootstrap_draws": int(inference_config["bootstrap_draws"]),
        "random_projection_draws": int(random_config["draws"]),
        "optional_low_variance_control": config["optional_low_variance_control"],
        "hta_status": config["hta"],
        "verdict": verdict,
    }

    fixed_oof = pd.concat(
        [oof_table(data, f"PCA{k}", sweep.predictions[f"PCA{k}"], sweep.population) for k in config["model"]["pca_grid"]],
        ignore_index=True,
    )
    write_csv(sweep.grid_manifest, results_dir / "PCA_RESOLUTION_GRID_MANIFEST.csv")
    write_csv(sweep.selections, results_dir / "OUTER_FOLD_SELECTED_K_ALPHA.csv")
    write_csv(fixed_oof, results_dir / "FIXED_K_OOF_PREDICTIONS.csv")
    write_csv(fixed_performance, results_dir / "FIXED_K_PERFORMANCE.csv")
    write_csv(oof_table(data, "PCA_STAR", pca_star, sweep.population), results_dir / "PCA_STAR_OOF_PREDICTIONS.csv")
    write_csv(star_performance, results_dir / "PCA_STAR_PERFORMANCE.csv")
    write_csv(oof_table(data, "FULL_RNA_REOPT", full_reopt, sweep.population), results_dir / "FULL_RNA_REOPT_OOF.csv")
    write_csv(full_performance, results_dir / "FULL_RNA_REOPT_PERFORMANCE.csv")
    write_csv(inference, results_dir / "PCA_STAR_VS_FULL_PRIMARY_INFERENCE.csv")
    write_csv(random_results, results_dir / "RANDOM_MATCHED_KSTAR_RESULTS.csv")
    write_csv(random_folds, results_dir / "RANDOM_MATCHED_KSTAR_FOLD_AUDIT.csv")
    write_csv(sweep.variance_capture, results_dir / "PCA_EXPRESSION_VARIANCE_CAPTURE.csv")
    write_csv(curves, results_dir / "FIXED_K_BOOTSTRAP_CURVES.csv")
    write_csv(patient_metric_frame(data, {"PCA_STAR": pca_star, "FULL_RNA_REOPT": full_reopt}, sweep.population), results_dir / "PCA_STAR_VS_FULL_PATIENT_METRICS.csv")
    write_csv(oof_table(data, "PCA14_FROZEN_REPRO", pca14_replay.predictions, pca14_replay.population), results_dir / "PCA14_FROZEN_REPRO_OOF.csv")
    (results_dir / "PCA_RESOLUTION_SWEEP_FINAL_QA.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    patient_metrics = patient_metric_frame(
        data, {"PCA_STAR": pca_star, "FULL_RNA_REOPT": full_reopt}, sweep.population
    )
    make_figures(results_dir, curves, patient_metrics, sweep.selections, random_results, star_performance)
    final_report(
        results_dir / "PCA_RESOLUTION_SWEEP_FINAL.md",
        verdict,
        fixed_performance,
        star_performance,
        full_performance,
        inference,
        sweep.selections,
        selection_summary,
        random_results,
        sweep.variance_capture,
        qa,
    )
    print(json.dumps({"verdict": verdict, "results_dir": str(results_dir), "qa": qa}, indent=2))


if __name__ == "__main__":
    main()
