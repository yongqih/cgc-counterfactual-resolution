from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.conditioning_resolution import (
    metrics,
    oof_table,
    random_orthonormal_bases,
    sha256,
)
from igc_virtual_cell.crc_pdo_personalized.dense_resolution_plateau import (
    adjacent_gain_table,
    bootstrap_performance,
    deterministic_verdict,
    full_increment_table,
    gain_and_slope_tables,
    performance_interval_frame,
    plateau_band,
    retention_frame,
    saturation_thresholds,
)
from igc_virtual_cell.crc_pdo_personalized.modeling import load_platform_data
from igc_virtual_cell.crc_pdo_personalized.pca_resolution_sweep import (
    fit_nested_pca_oof,
    fit_random_matched_kstar_oof,
)


PRIOR_RESULT_COMMIT = "1bb0e95b39c2181638d732097f8f868689f32c87"
PRIOR_LEDGER_COMMIT = "624e6f239459418f8d6a8792ff41d2dcc1c2c61c"
NUMERICAL_TOLERANCE = 5e-13


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def top_k_overlaps(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 3, 5):
        overlaps = []
        for row in range(len(truth)):
            truth_set = set(np.argsort(-truth[row], kind="stable")[:k])
            prediction_set = set(np.argsort(-prediction[row], kind="stable")[:k])
            overlaps.append(len(truth_set & prediction_set) / k)
        result[f"top{k}_overlap_fraction"] = float(np.mean(overlaps))
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


def matrix_from_oof(path: Path, data: object, representation: str | None = None) -> np.ndarray:
    frame = pd.read_csv(path)
    if representation is not None:
        frame = frame.query("representation == @representation")
    patients = [str(value) for value in data.patient_ids]
    drugs = [str(value) for value in data.drug_names]
    return (
        frame.pivot(index="patient_id", columns="drug_name", values="predicted_DSS")
        .loc[patients, drugs]
        .to_numpy(float)
    )


def source_paths(root: Path, application_root: Path, manuscript_path: Path | None) -> dict[str, Path]:
    paths = {
        "RNAseq_PDO_log2CPM1.npz": application_root
        / "data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz",
        "PRIMARY_DSS.npz": application_root
        / "data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz",
        "PRIOR_FIXED_K_OOF_PREDICTIONS.csv": root
        / "results/crc_pdo_pca_resolution_sweep/FIXED_K_OOF_PREDICTIONS.csv",
        "PRIOR_PCA_STAR_OOF_PREDICTIONS.csv": root
        / "results/crc_pdo_pca_resolution_sweep/PCA_STAR_OOF_PREDICTIONS.csv",
        "PRIOR_FULL_RNA_REOPT_OOF.csv": root
        / "results/crc_pdo_pca_resolution_sweep/FULL_RNA_REOPT_OOF.csv",
        "PRIOR_FINAL_QA.json": root
        / "results/crc_pdo_pca_resolution_sweep/PCA_RESOLUTION_SWEEP_FINAL_QA.json",
    }
    if manuscript_path is not None and manuscript_path.exists():
        paths["MANUSCRIPT_DOCX"] = manuscript_path
    return paths


def source_hashes(root: Path, application_root: Path, manuscript_path: Path | None) -> dict[str, str]:
    return {
        name: sha256(path)
        for name, path in source_paths(root, application_root, manuscript_path).items()
    }


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


def dense_curve(
    publication: pd.DataFrame,
    metric: str,
    results_dir: Path,
    filename: str,
) -> None:
    local = publication.query("representation_family in ['fixed_PCA', 'full_RNA']").copy()
    fixed = local.query("representation_family == 'fixed_PCA'").sort_values("dimension")
    full = local.query("representation_family == 'full_RNA'").iloc[0]
    y = fixed[metric].to_numpy(float)
    lower = fixed[f"{metric}_ci_lower"].to_numpy(float)
    upper = fixed[f"{metric}_ci_upper"].to_numpy(float)
    x = np.arange(len(fixed))
    fig, ax = plt.subplots(figsize=(5.2, 2.55))
    ax.plot(x, y, marker="o", markersize=3.7, linewidth=1.35, color="#3E789D")
    ax.errorbar(x, y, yerr=[y - lower, upper - y], fmt="none", ecolor="#3E789D", linewidth=0.7, capsize=1.7)
    full_x = len(x) + 1.2
    ax.errorbar(
        [full_x],
        [full[metric]],
        yerr=[[full[metric] - full[f"{metric}_ci_lower"]], [full[f"{metric}_ci_upper"] - full[metric]]],
        fmt="D",
        markersize=4,
        color="#C56A53",
        capsize=2,
        linewidth=0.8,
    )
    ax.axvline(len(x) + 0.35, color="#D4D8DC", linestyle="--", linewidth=0.7)
    ax.axhline(0, color="#B3B9BE", linewidth=0.7)
    labels = fixed["dimension"].astype(int).astype(str).tolist() + ["full\n19,421"]
    ax.set_xticks([*x, full_x], labels, rotation=45, ha="right")
    ax.set_xlabel("baseline representation dimensions")
    ax.set_ylabel("personalized ranking gain" if metric == "PG_macro" else r"functional recovery $g_{\mathrm{func}}$")
    ax.set_title("Dense conditioning-resolution curve", loc="center", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, results_dir / filename)


def make_figures(
    results_dir: Path,
    publication: pd.DataFrame,
    retention: pd.DataFrame,
    selections: pd.DataFrame,
    random_results: pd.DataFrame,
    marginal: pd.DataFrame,
) -> None:
    configure_plotting()
    dense_curve(publication, "PG_macro", results_dir, "DENSE_PCA_RESOLUTION_CURVE")
    dense_curve(publication, "g_func", results_dir, "DENSE_PCA_RESOLUTION_GFUNC")

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    for metric, label, color in [
        ("PG_macro", "PG retention", "#3E789D"),
        ("g_func", r"$g_{\mathrm{func}}$ retention", "#54A095"),
    ]:
        local = retention.query("metric == @metric").sort_values("dimension")
        ax.plot(local.dimension, local.retention, marker="o", markersize=3.5, linewidth=1.25, label=label, color=color)
    ax.axhline(0.90, color="#AEB4BA", linewidth=0.8, linestyle="--")
    ax.axhline(0.95, color="#777F87", linewidth=0.8, linestyle=":")
    ax.text(32.5, 0.90, "90%", fontsize=7, va="center")
    ax.text(32.5, 0.95, "95%", fontsize=7, va="center")
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("retention vs full RNA")
    ax.set_title("Operational information saturation", loc="center", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, results_dir / "NORMALIZED_INFORMATION_SATURATION")

    grid = sorted(selections.selected_k.unique())
    counts = selections.selected_k.value_counts().reindex(grid, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.0, 2.35))
    ax.bar(np.arange(len(grid)), counts.values, color="#7293AA", width=0.72)
    ax.set_xticks(np.arange(len(grid)), [str(value) for value in grid], rotation=45, ha="right")
    ax.set_xlabel(r"nested selected $k_*$")
    ax.set_ylabel("outer folds")
    ax.set_title("Training-selected resolution band", loc="center", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, results_dir / "TRAINING_SELECTED_RESOLUTION_BAND")

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35))
    for ax, metric, ylabel in zip(axes, ["PG_macro", "g_func"], ["PG", r"$g_{\mathrm{func}}$"], strict=True):
        positions = []
        values = []
        observed = []
        for position, k in enumerate([16, 24, 32]):
            local = random_results.query("control_k == @k")
            positions.append(position)
            values.append(local[metric].to_numpy(float))
            observed.append(float(local[f"pca_observed_{metric}"].iloc[0]))
        parts = ax.violinplot(values, positions=positions, widths=0.7, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#BEC8D0")
            body.set_edgecolor("none")
            body.set_alpha(0.9)
        ax.scatter(positions, observed, color="#C65F4A", s=20, zorder=3)
        ax.set_xticks(positions, ["16", "24", "32"])
        ax.set_xlabel("matched dimensions")
        ax.set_ylabel(ylabel)
        ax.set_title("Structured PCA vs random", loc="center", fontweight="bold")
    fig.tight_layout(w_pad=2)
    save_figure(fig, results_dir / "PCA_VS_MATCHED_RANDOM_AT_16_24_32")

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35))
    for ax, metric, ylabel, color in zip(
        axes,
        ["PG_macro", "g_func"],
        [r"adjacent $\Delta$PG", r"adjacent $\Delta g_{\mathrm{func}}$"],
        ["#3E789D", "#54A095"],
        strict=True,
    ):
        local = marginal.query("metric == @metric")
        y = local.estimate.to_numpy(float)
        lower = local.pointwise_95_ci_lower.to_numpy(float)
        upper = local.pointwise_95_ci_upper.to_numpy(float)
        x = np.arange(len(local))
        ax.errorbar(x, y, yerr=[y - lower, upper - y], fmt="o", markersize=3.5, color=color, linewidth=0.8, capsize=2)
        ax.axhline(0, color="#AEB4BA", linewidth=0.7)
        ax.set_xticks(x, [f"{a}→{b}" for a, b in zip(local.from_k, local.to_k, strict=True)], rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Local marginal gain", loc="center", fontweight="bold")
    fig.tight_layout(w_pad=2)
    save_figure(fig, results_dir / "MARGINAL_GAIN_BY_ADDED_DIMENSION")

    fixed = publication.query("representation_family == 'fixed_PCA'").sort_values("dimension")
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35))
    for ax, metric, ylabel in zip(axes, ["PG_macro", "g_func"], ["PG", r"$g_{\mathrm{func}}$"], strict=True):
        ax.plot(fixed.explained_baseline_variance, fixed[metric], marker="o", markersize=3.5, linewidth=1.2, color="#3E789D")
        ax.set_xlabel("training expression variance explained")
        ax.set_ylabel(ylabel)
        ax.set_title("Performance vs expression variance", loc="center", fontweight="bold")
    fig.tight_layout(w_pad=2)
    save_figure(fig, results_dir / "PREDICTIVE_PERFORMANCE_VS_EXPRESSION_VARIANCE")


def final_report(
    path: Path,
    verdict: str,
    performance: pd.DataFrame,
    thresholds: pd.DataFrame,
    gains: pd.DataFrame,
    marginal: pd.DataFrame,
    increments: pd.DataFrame,
    selections: pd.DataFrame,
    random_results: pd.DataFrame,
    plateau: pd.DataFrame,
    qa: dict[str, object],
) -> None:
    fixed = performance.query("representation_family == 'fixed_PCA'").set_index("dimension")
    full = performance.query("representation == 'FULL_RNA_REOPT'").iloc[0]
    star = performance.query("representation == 'PCA_STAR_DENSE'").iloc[0]
    threshold_index = thresholds.set_index(["retention_threshold", "metric"])
    counts = selections.selected_k.value_counts().sort_index()
    lines = [
        "# Dense CRC PDO conditioning-resolution saturation audit",
        "",
        f"Final verdict: `{verdict}`",
        "",
        "## Frozen task and reproduction",
        "",
        "- The frozen 52-patient, 91-PDO, 24-drug, 19,421-gene task and multi-output Ridge residual formulation were unchanged.",
        f"- FULL_RNA_REOPT reproduction maximum difference: `{qa['full_rna_reopt_max_abs_difference']:.3g}`.",
        f"- PCA16/PCA24/PCA32 reproduction maximum differences: `{qa['anchor_reproduction_max_abs_difference']['PCA16']:.3g}` / `{qa['anchor_reproduction_max_abs_difference']['PCA24']:.3g}` / `{qa['anchor_reproduction_max_abs_difference']['PCA32']:.3g}`.",
        "",
        "## Dense fixed-k performance",
        "",
        "| k | PG | g_func | reversal BA | expression variance |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, row in fixed.iterrows():
        lines.append(f"| {int(k)} | {row.PG_macro:.6f} | {row.g_func:.6f} | {row.reversal_balanced_accuracy_patient_mean:.6f} | {row.explained_baseline_variance:.3%} |")
    lines.extend(
        [
            "",
            f"FULL_RNA_REOPT: PG `{full.PG_macro:.6f}`, g_func `{full.g_func:.6f}`, reversal BA `{full.reversal_balanced_accuracy_patient_mean:.6f}`.",
            f"PCA_STAR_DENSE: PG `{star.PG_macro:.6f}`, g_func `{star.g_func:.6f}`, reversal BA `{star.reversal_balanced_accuracy_patient_mean:.6f}`.",
            "",
            "## Operational saturation thresholds",
            "",
        ]
    )
    for threshold in [0.90, 0.95]:
        for metric in ["PG", "g_func", "joint"]:
            row = threshold_index.loc[(threshold, metric)]
            lines.append(
                f"- {int(threshold * 100)}% {metric}: observed k `{row.observed_k:g}`; bootstrap median/IQR/90% range `{row.bootstrap_median_k:g}` / `[{row.bootstrap_q25_k:g},{row.bootstrap_q75_k:g}]` / `[{row.bootstrap_90_range_lower_k:g},{row.bootstrap_90_range_upper_k:g}]`; defined draws `{int(row.bootstrap_defined_draws)}/10000`; representation-dimension compression `{row.representation_dimension_compression_ratio:.1f}×`."
            )
    lines.extend(
        [
            "",
            "Bootstrap replicates with non-positive FULL_RNA metric denominators were excluded only for the affected retention ratio and are counted in `SATURATION_THRESHOLDS.csv`; no undefined ratio was silently imputed.",
            "",
            "## Early and late gain",
            "",
        ]
    )
    for metric in ["PG_macro", "g_func"]:
        local = gains.query("metric == @metric").set_index("quantity")
        lines.append(
            f"- {metric}: PCA2→PCA16 `{local.loc['gain_PCA2_to_PCA16', 'estimate']:.6f}`; PCA16→PCA32 `{local.loc['gain_PCA16_to_PCA32', 'estimate']:.6f}`; early capture of PCA2→FULL improvement `{local.loc['early_capture_fraction_of_PCA2_to_FULL', 'estimate']:.2%}`; early/late slopes `{local.loc['beta_early_per_dimension', 'estimate']:.6g}` / `{local.loc['beta_late_per_dimension', 'estimate']:.6g}` per dimension."
        )
    lines.extend(["", "Adjacent late-region gains and paired pointwise bootstrap intervals:", ""])
    for (start, end), local in marginal.groupby(["from_k", "to_k"], sort=True):
        pg = local.query("metric == 'PG_macro'").iloc[0]
        g = local.query("metric == 'g_func'").iloc[0]
        lines.append(f"- {int(start)}→{int(end)}: ΔPG `{pg.estimate:.6f}` `[{pg.pointwise_95_ci_lower:.6f},{pg.pointwise_95_ci_upper:.6f}]`; Δg `{g.estimate:.6f}` `[{g.pointwise_95_ci_lower:.6f},{g.pointwise_95_ci_upper:.6f}]`.")
    plateau_members = plateau.query("in_plateau_band").dimension.astype(int).tolist()
    lines.extend(
        [
            "",
            "## Plateau, full-RNA increment, and structure control",
            "",
            f"- Descriptive plateau band: `{plateau_members}`; lowest/highest/count `{min(plateau_members) if plateau_members else 'NA'}` / `{max(plateau_members) if plateau_members else 'NA'}` / `{len(plateau_members)}`.",
            f"- Dense nested k counts: `{counts.to_dict()}`; compact-region fraction `{qa['dense_selection_compact_fraction']:.3%}`.",
        ]
    )
    for k in [16, 20, 24, 28, 32]:
        local = increments.query("k == @k").set_index("metric")
        lines.append(f"- FULL−PCA{k}: PG `{local.loc['PG_macro','estimate']:.6f}` `[{local.loc['PG_macro','pointwise_95_ci_lower']:.6f},{local.loc['PG_macro','pointwise_95_ci_upper']:.6f}]`; g `{local.loc['g_func','estimate']:.6f}` `[{local.loc['g_func','pointwise_95_ci_lower']:.6f},{local.loc['g_func','pointwise_95_ci_upper']:.6f}]`.")
    for k in [16, 24, 32]:
        local = random_results.query("control_k == @k").iloc[0]
        lines.append(f"- PCA{k} vs random{k}: PG percentile `{local.pca_PG_percentile:.1f}`, empirical P `{local.pca_PG_empirical_p:.6f}`; g percentile `{local.pca_g_func_percentile:.1f}`, empirical P `{local.pca_g_func_empirical_p:.6f}`.")
    lines.extend(["", "## Scientific interpretation", ""])
    if verdict == "PREDICTIVE_INFORMATION_SATURATION_SUPPORTED":
        lines.append("Held-patient prediction rose strongly across the first tens of dominant transcriptomic coordinates and then entered a broad high-performance region. Nominal 19,421-gene measurement resolution therefore substantially exceeded the effective representation scale at which transferable predictive information was operationally recovered.")
    elif verdict == "COMPACT_PREDICTIVE_STRUCTURE_SUPPORTED_WITHOUT_CLEAR_SATURATION":
        lines.append("Compact dominant transcriptomic structure was predictive and far exceeded matched random projections, but the dense curve did not satisfy the frozen broad-plateau definition.")
    elif verdict == "PREDICTIVE_PERFORMANCE_CONTINUES_TO_SCALE_WITH_DIMENSION":
        lines.append("Predictive performance continued to gain materially through the dense region, and the complete gene-level representation retained a material advantage.")
    else:
        lines.append("The frozen reproduction, structured-control, and curve-shape requirements did not establish a reproducible conditioning-resolution pattern.")
    lines.extend(
        [
            "",
            "PCA coordinates remain deterministic functions of full baseline RNA. This result does not show that a 16–32-analyte assay is sufficient, that fine genes contain no information, or that an intrinsic/information-theoretic transcriptome dimension has been identified.",
            "",
            "## QA and provenance",
            "",
            f"- Implementation frozen before dense outcomes: `{qa['implementation_commit']}`.",
            f"- Scientific QA: `{qa['passed_checks']}/{qa['required_checks']}` checks passed.",
            f"- Prior hashes and manuscript unchanged: `{qa['prior_hashes_unchanged']}`.",
            f"- Test status: `{qa['repository_test_status']}`.",
            "- The manuscript was not modified.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--manuscript", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    application_root = args.application_root.resolve()
    manuscript_path = args.manuscript.resolve() if args.manuscript is not None else None
    results_dir = root / "results/crc_pdo_dense_resolution_plateau"
    results_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configs/crc_pdo_dense_resolution_plateau.json").read_text())
    if config["prerequisite"]["scientific_result_commit"] != PRIOR_RESULT_COMMIT:
        raise ValueError("Prior result authority changed")
    if config["prerequisite"]["ledger_commit"] != PRIOR_LEDGER_COMMIT:
        raise ValueError("Prior ledger authority changed")
    hashes_before = source_hashes(root, application_root, manuscript_path)

    processed = application_root / "data/crc_pdo_personalized_application/processed"
    data = load_platform_data(processed, "RNAseq")
    dense_grid = [int(value) for value in config["model"]["dense_pca_grid"]]
    alpha_grid = [float(value) for value in config["model"]["alpha_grid"]]
    seed = int(config["model"]["seed"])
    sweep = fit_nested_pca_oof(data, dense_grid, alpha_grid, seed)

    prior_result_dir = root / "results/crc_pdo_pca_resolution_sweep"
    prior_full = matrix_from_oof(prior_result_dir / "FULL_RNA_REOPT_OOF.csv", data)
    full_difference = float(np.max(np.abs(sweep.predictions["FULL_RNA_REOPT"] - prior_full)))
    anchor_differences: dict[str, float] = {}
    for name in ["PCA16", "PCA24", "PCA32"]:
        prior = matrix_from_oof(prior_result_dir / "FIXED_K_OOF_PREDICTIONS.csv", data, name)
        anchor_differences[name] = float(np.max(np.abs(sweep.predictions[name] - prior)))
    if full_difference > NUMERICAL_TOLERANCE or max(anchor_differences.values()) > NUMERICAL_TOLERANCE:
        raise RuntimeError("DENSE_RESOLUTION_PRIOR_REPRODUCTION_FAILURE")

    fixed_names = [f"PCA{k}" for k in dense_grid]
    all_predictions = {
        **{name: sweep.predictions[name] for name in fixed_names},
        "PCA_STAR_DENSE": sweep.predictions["PCA_STAR"],
        "FULL_RNA_REOPT": sweep.predictions["FULL_RNA_REOPT"],
    }
    performance = pd.DataFrame.from_records(
        [performance_row(data, name, prediction, sweep.population) for name, prediction in all_predictions.items()]
    )
    performance["dimension"] = performance.representation.map(
        {**{f"PCA{k}": k for k in dense_grid}, "PCA_STAR_DENSE": float(np.median(sweep.selections.selected_k)), "FULL_RNA_REOPT": data.x.shape[1]}
    )
    performance["representation_family"] = performance.representation.map(
        {**{f"PCA{k}": "fixed_PCA" for k in dense_grid}, "PCA_STAR_DENSE": "nested_PCA", "FULL_RNA_REOPT": "full_RNA"}
    )

    bootstrap_config = config["bootstrap"]
    bootstrap = bootstrap_performance(
        data,
        all_predictions,
        sweep.population,
        int(bootstrap_config["draws"]),
        int(bootstrap_config["seed"]),
    )
    intervals = performance_interval_frame(bootstrap)
    retention, bootstrap_retention = retention_frame(bootstrap, fixed_names, "FULL_RNA_REOPT")
    observed_retention = (
        retention.pivot(index="dimension", columns="metric", values="retention")
        .loc[dense_grid, ["PG_macro", "g_func"]]
        .to_numpy(float)
    )
    threshold_table = saturation_thresholds(
        np.asarray(dense_grid),
        observed_retention,
        bootstrap_retention,
        config["saturation"]["retention_thresholds"],
        data.x.shape[1],
    )
    gain_table = gain_and_slope_tables(
        bootstrap, np.asarray(dense_grid), fixed_names, "FULL_RNA_REOPT"
    )
    marginal_table = adjacent_gain_table(
        bootstrap,
        np.asarray(dense_grid),
        fixed_names,
        float(config["saturation"]["practical_pg_gain"]),
        float(config["saturation"]["practical_g_func_gain"]),
    )
    increment_table = full_increment_table(
        bootstrap, fixed_names, [16, 20, 24, 28, 32], "FULL_RNA_REOPT"
    )

    random_config = config["random_control"]
    bases = random_orthonormal_bases(
        data.x.shape[1],
        32,
        int(random_config["draws"]),
        int(random_config["seed"]),
    )
    random_rows: list[dict[str, object]] = []
    random_fold_frames: list[pd.DataFrame] = []
    random_passes = 0
    performance_index = performance.set_index("representation")
    for k in random_config["dimensions"]:
        predictions, folds = fit_random_matched_kstar_oof(
            data,
            bases,
            np.full(len(data.patient_ids), int(k), dtype=int),
            alpha_grid,
            seed,
        )
        folds.insert(0, "control_k", int(k))
        random_fold_frames.append(folds)
        draw_metrics = [performance_row(data, f"RANDOM{k}_{draw}", predictions[draw], sweep.population) for draw in range(len(predictions))]
        draw_frame = pd.DataFrame.from_records(draw_metrics)
        pca_pg = float(performance_index.loc[f"PCA{k}", "PG_macro"])
        pca_g = float(performance_index.loc[f"PCA{k}", "g_func"])
        pg_percentile = float(100 * np.mean(draw_frame.PG_macro < pca_pg))
        g_percentile = float(100 * np.mean(draw_frame.g_func < pca_g))
        pg_p = float((1 + np.sum(draw_frame.PG_macro >= pca_pg)) / (len(draw_frame) + 1))
        g_p = float((1 + np.sum(draw_frame.g_func >= pca_g)) / (len(draw_frame) + 1))
        random_passes += int(pca_pg > np.quantile(draw_frame.PG_macro, 0.95) and pca_g > np.quantile(draw_frame.g_func, 0.95))
        for draw, row in draw_frame.iterrows():
            random_rows.append(
                {
                    "control_k": int(k),
                    "draw": int(draw),
                    **{column: row[column] for column in row.index if column not in {"platform", "representation", "patients", "drugs"}},
                    "pca_observed_PG_macro": pca_pg,
                    "pca_observed_g_func": pca_g,
                    "pca_PG_percentile": pg_percentile,
                    "pca_g_func_percentile": g_percentile,
                    "pca_PG_empirical_p": pg_p,
                    "pca_g_func_empirical_p": g_p,
                }
            )
    del bases
    random_results = pd.DataFrame.from_records(random_rows)
    random_fold_audit = pd.concat(random_fold_frames, ignore_index=True)

    full_row = performance_index.loc["FULL_RNA_REOPT"]
    star_row = performance_index.loc["PCA_STAR_DENSE"]
    reference_pg = float(max(full_row.PG_macro, star_row.PG_macro))
    reference_g = float(max(full_row.g_func, star_row.g_func))
    saturation_config = config["saturation"]
    plateau = plateau_band(
        performance,
        reference_pg,
        reference_g,
        float(saturation_config["practical_pg_gain"]),
        float(saturation_config["practical_g_func_gain"]),
    )
    plateau_members = plateau.query("in_plateau_band").dimension.to_numpy(int)
    span = int(np.ptp(plateau_members)) if len(plateau_members) else 0

    gain_index = gain_table.set_index(["metric", "quantity"])
    early_gain_pass = bool(
        gain_index.loc[("PG_macro", "gain_PCA2_to_PCA16"), "estimate"] >= saturation_config["practical_pg_gain"]
        and gain_index.loc[("g_func", "gain_PCA2_to_PCA16"), "estimate"] >= saturation_config["practical_g_func_gain"]
    )
    late_total_pass = bool(
        gain_index.loc[("PG_macro", "gain_PCA16_to_PCA32"), "estimate"] < saturation_config["practical_pg_gain"]
        and gain_index.loc[("g_func", "gain_PCA16_to_PCA32"), "estimate"] < saturation_config["practical_g_func_gain"]
    )
    no_material_adjacent = bool(~marginal_table.positive_gain_reaches_practical_scale.any())
    broad_plateau_pass = bool(
        len(plateau_members) >= saturation_config["broad_plateau_min_grid_points"]
        and span >= saturation_config["broad_plateau_min_dimension_span"]
        and late_total_pass
        and no_material_adjacent
    )
    compact_fraction = float(
        sweep.selections.selected_k.between(
            saturation_config["dense_selection_region"][0], saturation_config["dense_selection_region"][1]
        ).mean()
    )
    dense_selection_pass = compact_fraction >= saturation_config["dense_selection_compact_fraction"]
    joint_k90 = threshold_table.query("retention_threshold == 0.90 and metric == 'joint'").iloc[0].observed_k
    joint_k90_by_32 = bool(np.isfinite(joint_k90) and joint_k90 <= 32)
    random_structure_pass = random_passes >= 2
    full_minus_32 = increment_table.query("k == 32").set_index("metric")
    continued_scaling_pass = bool(
        gain_index.loc[("PG_macro", "gain_PCA16_to_PCA32"), "estimate"] >= saturation_config["practical_pg_gain"]
        and gain_index.loc[("g_func", "gain_PCA16_to_PCA32"), "estimate"] >= saturation_config["practical_g_func_gain"]
        and full_minus_32.loc["PG_macro", "estimate"] >= saturation_config["practical_pg_gain"]
        and full_minus_32.loc["g_func", "estimate"] >= saturation_config["practical_g_func_gain"]
    )

    variance = sweep.variance_capture.rename(
        columns={"training_expression_variance_explained": "explained_baseline_variance"}
    )
    variance_summary = (
        variance.groupby("k", as_index=False).explained_baseline_variance
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"k": "dimension", "mean": "mean_explained_baseline_variance", "min": "min_explained_baseline_variance", "max": "max_explained_baseline_variance"})
    )
    selected_variance = variance.merge(
        sweep.selections[["held_patient", "selected_k"]],
        left_on=["held_patient", "k"],
        right_on=["held_patient", "selected_k"],
    ).explained_baseline_variance
    interval_wide = intervals.pivot(index="representation", columns="metric", values=["estimate", "pointwise_95_ci_lower", "pointwise_95_ci_upper"])
    interval_wide.columns = [f"{metric}_{stat}" for stat, metric in interval_wide.columns]
    interval_wide = interval_wide.reset_index()
    publication = performance.merge(interval_wide.drop(columns=["PG_macro_estimate", "g_func_estimate"]), on="representation")
    publication = publication.rename(
        columns={
            "PG_macro_pointwise_95_ci_lower": "PG_macro_ci_lower",
            "PG_macro_pointwise_95_ci_upper": "PG_macro_ci_upper",
            "g_func_pointwise_95_ci_lower": "g_func_ci_lower",
            "g_func_pointwise_95_ci_upper": "g_func_ci_upper",
        }
    )
    publication = publication.merge(
        variance_summary[["dimension", "mean_explained_baseline_variance"]].rename(columns={"mean_explained_baseline_variance": "explained_baseline_variance"}),
        on="dimension",
        how="left",
    )
    publication.loc[publication.representation.eq("PCA_STAR_DENSE"), "explained_baseline_variance"] = float(selected_variance.mean())
    publication.loc[publication.representation.eq("FULL_RNA_REOPT"), "explained_baseline_variance"] = 1.0
    retention_wide = retention.pivot(index="dimension", columns="metric", values="retention").reset_index().rename(columns={"PG_macro": "retention_PG", "g_func": "retention_g"})
    publication = publication.merge(retention_wide, on="dimension", how="left")
    publication.loc[publication.representation.eq("PCA_STAR_DENSE"), "retention_PG"] = star_row.PG_macro / full_row.PG_macro
    publication.loc[publication.representation.eq("PCA_STAR_DENSE"), "retention_g"] = star_row.g_func / full_row.g_func
    publication.loc[publication.representation.eq("FULL_RNA_REOPT"), ["retention_PG", "retention_g"]] = 1.0
    counts = sweep.selections.selected_k.value_counts()
    publication["selected_by_inner_CV_count"] = publication.apply(
        lambda row: int(counts.get(int(row.dimension), 0)) if row.representation_family == "fixed_PCA" else 52 if row.representation == "PCA_STAR_DENSE" else 0,
        axis=1,
    )
    publication["claim_boundary"] = config["claim_boundary"]

    hashes_after = source_hashes(root, application_root, manuscript_path)
    prior_star_hash_preserved = hashes_before["PRIOR_PCA_STAR_OOF_PREDICTIONS.csv"] == hashes_after["PRIOR_PCA_STAR_OOF_PREDICTIONS.csv"]
    manuscript_unchanged = (
        "MANUSCRIPT_DOCX" not in hashes_before
        or hashes_before["MANUSCRIPT_DOCX"] == hashes_after["MANUSCRIPT_DOCX"]
    )
    prior_population = matrix_population(
        prior_result_dir / "FULL_RNA_REOPT_OOF.csv", data
    )
    population_max_abs_difference = float(
        np.max(np.abs(sweep.population - prior_population))
    )
    checks = {
        "same_52_patients": len(data.patient_ids) == 52,
        "same_91_pdos": sum(len(group) for group in data.eligible_pdo_ids) == 91,
        "same_24_drugs": len(data.drug_names) == 24,
        "same_19421_genes": data.x.shape[1] == 19421,
        "same_patient_disjoint_folds": sweep.selections.patient_overlap.eq(0).all(),
        "same_patient_specific_multioutput_model": all(value.shape == (52, 24) for value in sweep.predictions.values()),
        "same_population_residual_target": population_max_abs_difference <= NUMERICAL_TOLERANCE,
        "same_ridge_alpha_grid": set(sweep.grid_manifest.alpha.unique()) == set(alpha_grid),
        "all_pca_transforms_training_side_only": sweep.selections.pca_fit_scope.eq("outer_and_inner_training_only").all(),
        "inner_pca_refit": sweep.grid_manifest.inner_preprocessing_refit.all(),
        "no_outer_outcome_selects_k": sweep.selections.held_outcome_used_in_selection.eq(False).all(),
        "full_rna_reopt_reproduced": full_difference <= NUMERICAL_TOLERANCE,
        "pca16_24_32_reproduced": max(anchor_differences.values()) <= NUMERICAL_TOLERANCE,
        "prior_pca_star_preserved": prior_star_hash_preserved,
        "dense_grid_completed_before_interpretation": set(performance.query("representation_family == 'fixed_PCA'").dimension.astype(int)) == set(dense_grid),
        "random_projections_outcome_blind": random_fold_audit.outcome_blind_projection.all(),
        "pg_recomputable": np.isfinite(performance.PG_macro).all(),
        "g_func_recomputable": np.isfinite(performance.g_func).all(),
        "reversal_ba_recomputable": np.isfinite(performance.reversal_balanced_accuracy_patient_mean).all(),
        "prior_project_hashes_unchanged": hashes_before == hashes_after,
        "manuscript_unchanged": manuscript_unchanged,
        "all_outputs_finite_except_documented_thresholds": np.isfinite(performance.select_dtypes(include=[np.number])).all().all(),
    }
    qa_pass = bool(all(checks.values()))
    verdict = deterministic_verdict(
        early_gain_pass,
        joint_k90_by_32,
        broad_plateau_pass,
        dense_selection_pass,
        random_structure_pass,
        continued_scaling_pass,
        qa_pass,
    )
    qa = {
        "protocol_id": config["protocol_id"],
        "branch": git(root, "branch", "--show-current"),
        "implementation_commit": args.implementation_commit,
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "prior_ledger_commit": PRIOR_LEDGER_COMMIT,
        "full_rna_reopt_max_abs_difference": full_difference,
        "population_reference_max_abs_difference": population_max_abs_difference,
        "anchor_reproduction_max_abs_difference": anchor_differences,
        "prior_pca_star_hash_preserved": prior_star_hash_preserved,
        "dense_selection_compact_fraction": compact_fraction,
        "plateau_dimensions": plateau_members.tolist(),
        "plateau_span": span,
        "verdict_gates": {
            "strong_early_gain": early_gain_pass,
            "joint_k90_by_32": joint_k90_by_32,
            "broad_plateau": broad_plateau_pass,
            "dense_selection_compact": dense_selection_pass,
            "random_structure": random_structure_pass,
            "continued_scaling": continued_scaling_pass,
            "late_total_below_practical_scale": late_total_pass,
            "no_material_adjacent_late_gain": no_material_adjacent,
        },
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "prior_hashes_unchanged": hashes_before == hashes_after,
        "checks": {name: bool(value) for name, value in checks.items()},
        "required_checks": len(checks),
        "passed_checks": int(sum(bool(value) for value in checks.values())),
        "all_required_checks_pass": qa_pass,
        "repository_test_status": "PENDING_POSTRUN_EXTERNAL_AUDIT",
        "bootstrap_draws": int(bootstrap_config["draws"]),
        "random_draws_per_dimension": int(random_config["draws"]),
        "optional_tail_pc_control": config["optional_tail_pc_control"],
        "verdict": verdict,
    }

    dense_oof = pd.concat(
        [oof_table(data, name, sweep.predictions[name], sweep.population) for name in fixed_names],
        ignore_index=True,
    )
    grid_manifest = sweep.grid_manifest.copy()
    grid_manifest["grid_frozen_before_outcomes"] = True
    grid_manifest["experimental_variable"] = "baseline_representation_dimension"
    write_csv(grid_manifest, results_dir / "DENSE_RESOLUTION_GRID_MANIFEST.csv")
    write_csv(dense_oof, results_dir / "FIXED_K_DENSE_OOF_PREDICTIONS.csv")
    write_csv(performance.query("representation_family == 'fixed_PCA'"), results_dir / "FIXED_K_DENSE_PERFORMANCE.csv")
    write_csv(sweep.selections.rename(columns={"selected_k": "selected_k_dense"}), results_dir / "PCA_STAR_DENSE_SELECTION.csv")
    write_csv(performance.query("representation == 'PCA_STAR_DENSE'"), results_dir / "PCA_STAR_DENSE_PERFORMANCE.csv")
    write_csv(oof_table(data, "PCA_STAR_DENSE", sweep.predictions["PCA_STAR"], sweep.population), results_dir / "PCA_STAR_DENSE_OOF_PREDICTIONS.csv")
    write_csv(oof_table(data, "FULL_RNA_REOPT", sweep.predictions["FULL_RNA_REOPT"], sweep.population), results_dir / "FULL_RNA_REOPT_OOF.csv")
    write_csv(retention, results_dir / "PERFORMANCE_RETENTION.csv")
    write_csv(threshold_table, results_dir / "SATURATION_THRESHOLDS.csv")
    write_csv(gain_table, results_dir / "EARLY_VS_LATE_GAIN.csv")
    write_csv(marginal_table, results_dir / "DENSE_MARGINAL_GAINS.csv")
    write_csv(increment_table, results_dir / "FULL_RNA_INCREMENT.csv")
    write_csv(random_results, results_dir / "RANDOM_CONTROL_16_24_32.csv")
    write_csv(random_fold_audit, results_dir / "RANDOM_CONTROL_FOLD_AUDIT.csv")
    write_csv(variance, results_dir / "PCA_VARIANCE_EXPLAINED.csv")
    write_csv(plateau, results_dir / "PLATEAU_BAND.csv")
    write_csv(publication, results_dir / "CONDITIONING_RESOLUTION_PUBLICATION_SOURCE.csv")
    (results_dir / "DENSE_CONDITIONING_RESOLUTION_FINAL_QA.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    make_figures(results_dir, publication, retention, sweep.selections, random_results, marginal_table)
    final_report(
        results_dir / "DENSE_CONDITIONING_RESOLUTION_FINAL.md",
        verdict,
        publication,
        threshold_table,
        gain_table,
        marginal_table,
        increment_table,
        sweep.selections,
        random_results,
        plateau,
        qa,
    )
    print(json.dumps({"verdict": verdict, "results_dir": str(results_dir), "qa": qa}, indent=2))


def matrix_population(path: Path, data: object) -> np.ndarray:
    frame = pd.read_csv(path)
    patients = [str(value) for value in data.patient_ids]
    drugs = [str(value) for value in data.drug_names]
    return (
        frame.pivot(index="patient_id", columns="drug_name", values="population_DSS")
        .loc[patients, drugs]
        .to_numpy(float)
    )


if __name__ == "__main__":
    main()
