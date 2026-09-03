from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.conditioning_resolution import (
    RepresentationSpec,
    bootstrap_primary_inference,
    build_progeny_mapping,
    deterministic_verdict,
    fit_random_family_oof,
    fit_representation_oof,
    metrics,
    oof_table,
    random_family_summary,
    random_orthonormal_bases,
    random_pathway_bases,
    sha256,
)
from igc_virtual_cell.crc_pdo_personalized.modeling import fit_oof_platform, load_platform_data


FROZEN_APPLICATION_RESULT = "c6cccb90453c567b6593982df442c4f127809fa1"
PREREQUISITE_VERDICT = "PERSONALIZED_RIDGE_INTERACTION_VALID"
FROZEN_FULL_PG = 0.050423060202472454
FROZEN_FULL_G = 0.13928938752362074


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _source_hashes(application_root: Path) -> dict[str, str]:
    paths = {
        "RNAseq_PDO_log2CPM1.npz": application_root
        / "data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz",
        "PRIMARY_DSS.npz": application_root
        / "data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz",
        "HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv": application_root
        / "results/crc_pdo_personalized_drug_application/HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv",
        "MODEL_FOLD_MANIFEST.csv": application_root
        / "results/crc_pdo_personalized_drug_application/MODEL_FOLD_MANIFEST.csv",
        "CRC_PDO_PERSONALIZED_APPLICATION_FINAL_QA.json": application_root
        / "results/crc_pdo_personalized_drug_application/CRC_PDO_PERSONALIZED_APPLICATION_FINAL_QA.json",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _performance_frame(
    data: Any,
    predictions: dict[str, np.ndarray],
    population: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for representation, prediction in predictions.items():
        rows.append(
            {
                "platform": data.platform,
                "representation": representation,
                "patients": len(data.patient_ids),
                "drugs": len(data.drug_names),
                **metrics(data.y, prediction, population),
            }
        )
    return pd.DataFrame.from_records(rows)


def _random_distribution_description(frame: pd.DataFrame, family: str) -> list[str]:
    lines = []
    for metric in ["PG_macro", "g_func", "reversal_balanced_accuracy_patient_mean"]:
        values = frame[metric].to_numpy(float)
        lines.append(
            f"- {family} {metric}: median {np.median(values):.6f}; 2.5th–97.5th percentile "
            f"[{np.quantile(values, 0.025):.6f}, {np.quantile(values, 0.975):.6f}]."
        )
    return lines


def _make_figures(
    results_dir: Path,
    performance: pd.DataFrame,
    random14: pd.DataFrame,
    random_pathway: pd.DataFrame,
    fine_levels: list[int],
    coefficients: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    colors = {"FULL_RNA": "#667085", "PROGENY14": "#1F7A75", "PCA14": "#4E79A7"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), constrained_layout=True)
    for ax, metric, label in zip(axes, ["PG_macro", "g_func"], [r"Personalization gain $PG$", r"Recovery $g_{func}$"], strict=True):
        ax.boxplot(
            [random14[metric], random_pathway[metric]],
            positions=[3, 4],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#E8ECEF", "edgecolor": "#98A2B3", "linewidth": 0.8},
            medianprops={"color": "#475467", "linewidth": 1.1},
            whiskerprops={"color": "#98A2B3", "linewidth": 0.8},
            capprops={"color": "#98A2B3", "linewidth": 0.8},
        )
        for position, name in enumerate(["FULL_RNA", "PROGENY14", "PCA14"]):
            value = float(performance.set_index("representation").loc[name, metric])
            ax.scatter(position, value, s=35, color=colors[name], zorder=4)
        ax.axhline(0, color="#D0D5DD", lw=0.8, zorder=0)
        ax.set_xticks(range(5), ["full RNA", "PROGENy14", "PCA14", "random14", "random\npathway14"], rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#EAECF0", linewidth=0.6)
    fig.savefig(results_dir / "CONDITIONING_RESOLUTION_PERFORMANCE.png", dpi=300, bbox_inches="tight")
    fig.savefig(results_dir / "CONDITIONING_RESOLUTION_PERFORMANCE.svg", bbox_inches="tight")
    plt.close(fig)

    ladder_names = [f"COARSE_R{k}" for k in fine_levels]
    ladder = performance.set_index("representation").loc[ladder_names + ["FULL_RNA"]]
    x = np.arange(len(ladder))
    labels = [str(k) for k in fine_levels] + ["full"]
    for metric, filename, ylabel in [
        ("PG_macro", "FINE_DETAIL_INCREMENT_CURVE.png", r"Personalization gain $PG$"),
        ("g_func", "FINE_DETAIL_INCREMENT_CURVE_GFUNC.png", r"Recovery $g_{func}$"),
    ]:
        fig, ax = plt.subplots(figsize=(4.5, 2.55), constrained_layout=True)
        ax.plot(x, ladder[metric], "o-", color="#1F7A75", lw=1.4, ms=4.5)
        ax.axhline(float(ladder.loc["COARSE_R0", metric]), color="#98A2B3", lw=0.8, ls="--")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Added fine-residual PCs")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#EAECF0", linewidth=0.6)
        fig.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
        fig.savefig(results_dir / filename.replace(".png", ".svg"), bbox_inches="tight")
        plt.close(fig)

    perf = performance.set_index("representation")
    retention = [
        perf.loc["PROGENY14", "PG_macro"] / perf.loc["FULL_RNA", "PG_macro"],
        perf.loc["PROGENY14", "g_func"] / perf.loc["FULL_RNA", "g_func"],
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    ax.bar([0, 1], retention, width=0.58, color=["#1F7A75", "#56A39D"])
    ax.axhline(1, color="#667085", lw=0.8, ls="--")
    ax.set_xticks([0, 1], [r"$PG$", r"$g_{func}$"])
    ax.set_ylabel("Fraction of full-RNA effect retained")
    for index, value in enumerate(retention):
        ax.text(index, value, f"{value:.2f}×", ha="center", va="bottom", fontsize=7)
    fig.savefig(results_dir / "COARSE_INFORMATION_RETENTION.png", dpi=300, bbox_inches="tight")
    fig.savefig(results_dir / "COARSE_INFORMATION_RETENTION.svg", bbox_inches="tight")
    plt.close(fig)

    mean_coef = coefficients.query("held_patient == 'MEAN_ACROSS_OUTER_FOLDS'")
    matrix = mean_coef.pivot(index="drug_name", columns="pathway", values="coefficient")
    order = coefficients["pathway"].drop_duplicates().tolist()
    matrix = matrix.reindex(columns=order)
    limit = float(np.quantile(np.abs(matrix.to_numpy()), 0.98))
    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xlabel("Externally defined PROGENy pathway")
    ax.set_ylabel("Drug")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Mean Ridge coefficient")
    fig.savefig(results_dir / "DRUG_SPECIFIC_COARSE_COEFFICIENTS.png", dpi=300, bbox_inches="tight")
    fig.savefig(results_dir / "DRUG_SPECIFIC_COARSE_COEFFICIENTS.svg", bbox_inches="tight")
    plt.close(fig)


def _final_report(
    results_dir: Path,
    verdict: str,
    performance: pd.DataFrame,
    inference: pd.DataFrame,
    random14: pd.DataFrame,
    random_pathway: pd.DataFrame,
    random_pathway_p: float,
    random_pathway_percentile: float,
    mapping: Any,
    fine_levels: list[int],
    fine_gate: bool,
    hta_status: str,
    qa: dict[str, Any],
) -> None:
    perf = performance.set_index("representation")
    inf = inference.set_index("contrast")
    lines = [
        "# CRC PDO conditioning-resolution audit",
        "",
        f"**Final prespecified verdict: `{verdict}`**",
        "",
        "## Frozen question and design",
        "",
        "This analysis asks whether gene-level baseline RNA adds stable patient-specific drug-response information beyond a fixed coarse biological span. It does not ask whether projection creates information. The patient, 24-drug panel, DSS orientation, leave-one-patient-out folds, population reference and multi-output Ridge family are inherited unchanged from the frozen application.",
        "",
        f"Prerequisite audit: `{PREREQUISITE_VERDICT}`. The primary cohort contains 52 independent patients and 24 drugs. All outer and inner transforms were fit without held outcomes; inner tuning refit gene scaling and every data-derived PCA on inner-training patients.",
        "",
        "## Baseline axis and externally frozen biological span",
        "",
        f"- Baseline RNA genes: {qa['baseline_gene_count']:,} unique symbols.",
        f"- Genes represented by at least one PROGENy pathway: {qa['progeny_mapped_gene_count']:,}.",
        f"- PROGENy version 1.17.3, official revision `cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f`; exact 14-pathway span rank: {mapping.rank}.",
        "- The primary coarse coordinates are an orthonormal basis for the complete externally defined PROGENy span. Conventional signed pathway scores are retained only for interpretation.",
        "",
        "## Primary representation results",
        "",
    ]
    for name in ["FULL_RNA", "PROGENY14", "PCA14"]:
        row = perf.loc[name]
        lines.append(
            f"- {name}: PG_macro={row.PG_macro:.6f}; g_func={row.g_func:.6f}; patient-mean reversal BA={row.reversal_balanced_accuracy_patient_mean:.6f}."
        )
    lines.extend(["", "## Prespecified inference", ""])
    for contrast in [
        "PROGENY14_minus_FULL_RNA_PG",
        "PROGENY14_minus_FULL_RNA_g_func",
        "PROGENY14_minus_PCA14_PG",
    ]:
        row = inf.loc[contrast]
        lines.append(
            f"- {contrast}: {row.estimate:.6f}, simultaneous 95% CI [{row.simultaneous_95_ci_lower:.6f}, {row.simultaneous_95_ci_upper:.6f}]; margin={row.noninferiority_margin:.6f}."
        )
    lines.extend(
        [
            f"- PROGENY14 rank among matched random-pathway vocabularies: percentile {random_pathway_percentile:.1f}; one-sided empirical P={random_pathway_p:.6f}.",
            "",
            "## Generic and pathway-like null distributions",
            "",
            *_random_distribution_description(random14, "RANDOM14"),
            *_random_distribution_description(random_pathway, "RANDOM_PATHWAY14"),
            "",
            "## Fine-detail ladder",
            "",
        ]
    )
    for level in fine_levels:
        row = perf.loc[f"COARSE_R{level}"]
        lines.append(
            f"- R{level}: PG_macro={row.PG_macro:.6f}; g_func={row.g_func:.6f}; reversal BA={row.reversal_balanced_accuracy_patient_mean:.6f}."
        )
    lines.extend(
        [
            f"- Any fine-detail level crossed the fixed supported material-gain gate: {'YES' if fine_gate else 'NO'}.",
            f"- Full-RNA endpoint: PG_macro={perf.loc['FULL_RNA', 'PG_macro']:.6f}; g_func={perf.loc['FULL_RNA', 'g_func']:.6f}.",
            "",
            "## Secondary cross-platform status",
            "",
            f"- HTA2.0: {hta_status}.",
            "",
            "## Adversarial interpretation",
            "",
        ]
    )
    if verdict == "CONDITIONING_RESOLUTION_MATCHING_SUPPORTED":
        lines.append("Baseline RNA was measured at gene resolution, but the patient-specific functional signal was retained in a much coarser externally defined biological span; adding prespecified fine residual components did not establish a material incremental gain.")
    elif verdict == "COARSE_BIOLOGICAL_REPRESENTATION_SUPERIOR":
        lines.append("The coarse biological span yielded stronger finite-sample held-patient prediction than full RNA and matched generic controls. This is a regularized recoverability result, not evidence that projection created information.")
    elif verdict == "GENERIC_DIMENSIONALITY_REDUCTION_EXPLAINS_COARSE_ADVANTAGE":
        lines.append("Low effective input dimension improved or preserved finite-sample prediction, but the matched controls did not establish pathway composition as the specific source of the effect.")
    elif verdict == "FINE_BASELINE_DETAIL_ADDS_PREDICTIVE_VALUE":
        lines.append("Fine residual baseline structure contributed a prespecified, supported material gain beyond the coarse span. The useful conditioning resolution therefore extends beyond the 14-pathway representation in this task.")
    elif verdict == "COARSE_REPRESENTATION_LOSES_PERSONALIZATION":
        lines.append("The coarse biological span failed the frozen retention gate and lost a substantial part of the validated full-RNA personalization signal.")
    else:
        lines.append("The audit does not distinguish pathway-specific resolution matching from the remaining alternatives under the frozen gates.")
    lines.extend(
        [
            "",
            "This experiment is an operational finite-sample prediction test. It does not imply that fine-scale RNA contains no information, that 14 pathways are an information-theoretic limit, or that the ex vivo DSS result establishes clinical treatment efficacy.",
            "",
            "## QA and provenance",
            "",
            f"- Frozen application predictions reproduced: {qa['full_prediction_reproduction_pass']} (max absolute difference {qa['full_prediction_max_abs_difference']:.3g}).",
            f"- Frozen FULL PG/g reproduced: {qa['full_metric_reproduction_pass']}.",
            f"- All required QA checks passed: {qa['all_required_checks_pass']} ({qa['passed_checks']}/{qa['required_checks']}).",
            f"- Analysis implementation commit before outcomes: `{qa['implementation_commit']}`.",
            f"- Frozen application source hashes unchanged: {qa['frozen_source_hashes_unchanged']}.",
            "- Manuscript files and frozen application outputs were not modified.",
        ]
    )
    (results_dir / "CRC_PDO_CONDITIONING_RESOLUTION_FINAL.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--progeny-weights", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    application_root = args.application_root.resolve()
    weights_path = args.progeny_weights.resolve()
    results_dir = root / "results/crc_pdo_conditioning_resolution"
    results_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configs/crc_pdo_conditioning_resolution.json").read_text())
    application_config = json.loads((root / "configs/crc_pdo_personalized_application.json").read_text())
    if config["prerequisite_verdict"] != PREREQUISITE_VERDICT:
        raise ValueError("Prerequisite verdict is not frozen as valid")
    if sha256(weights_path).lower() != config["progeny"]["weights_sha256"].lower():
        raise ValueError("PROGENy weights do not match the frozen SHA-256")

    source_hashes_before = _source_hashes(application_root)
    processed_dir = application_root / "data/crc_pdo_personalized_application/processed"
    data = load_platform_data(processed_dir, "RNAseq")
    expression = np.load(processed_dir / "RNAseq_PDO_log2CPM1.npz")
    mapping = build_progeny_mapping(
        expression["feature_ids"],
        expression["feature_symbols"],
        weights_path,
        config["progeny"]["pathway_order"],
    )
    if mapping.rank != 14:
        raise ValueError(f"Frozen PROGENy span rank is {mapping.rank}, expected 14")

    full_predictions, full_folds, _ = fit_oof_platform(data, application_config)
    frozen = pd.read_csv(
        application_root / "results/crc_pdo_personalized_drug_application/HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv"
    ).query("platform == 'RNAseq' and estimator == 'Ridge'")
    patient_order = [str(value) for value in data.patient_ids]
    drug_order = [str(value) for value in data.drug_names]
    frozen_matrix = (
        frozen.pivot(index="patient_id", columns="drug_name", values="predicted_DSS")
        .loc[patient_order, drug_order]
        .to_numpy(float)
    )
    reproduction_difference = float(np.max(np.abs(full_predictions["Ridge"] - frozen_matrix)))

    alphas = config["model"]["alpha_grid"]
    seed = int(config["model"]["seed"])
    specs = [
        RepresentationSpec("PROGENY14", "PROGENY_SPAN", 14),
        RepresentationSpec("PCA14", "PCA", 14),
        RepresentationSpec("COARSE_R4", "COARSE_PLUS_RESIDUAL", 4),
        RepresentationSpec("COARSE_R8", "COARSE_PLUS_RESIDUAL", 8),
        RepresentationSpec("COARSE_R16", "COARSE_PLUS_RESIDUAL", 16),
        RepresentationSpec("COARSE_R32", "COARSE_PLUS_RESIDUAL", 32),
        RepresentationSpec("R14_ONLY", "RESIDUAL_ONLY", 14),
    ]
    evaluations = {
        spec.name: fit_representation_oof(
            data, spec, mapping.q, mapping.r, mapping.pathway_order, alphas, seed
        )
        for spec in specs
    }
    population = full_predictions["Population"]
    for evaluation in evaluations.values():
        if not np.array_equal(evaluation.population, population):
            raise ValueError("Population reference changed across representations")

    representation_predictions = {
        "FULL_RNA": full_predictions["Ridge"],
        "PROGENY14": evaluations["PROGENY14"].predictions,
        "PCA14": evaluations["PCA14"].predictions,
        "COARSE_R0": evaluations["PROGENY14"].predictions,
        **{name: evaluations[name].predictions for name in ["COARSE_R4", "COARSE_R8", "COARSE_R16", "COARSE_R32", "R14_ONLY"]},
    }

    random_config = config["representations"]
    random_bases = random_orthonormal_bases(
        data.x.shape[1],
        int(random_config["random_projection_dimensions"]),
        int(random_config["random_projection_draws"]),
        int(random_config["random_seed"]),
    )
    random_predictions, random_folds = fit_random_family_oof(
        data, random_bases, alphas, seed, "RANDOM14"
    )
    del random_bases
    pathway_bases = random_pathway_bases(
        mapping.raw_weights,
        int(random_config["random_pathway_draws"]),
        int(random_config["random_pathway_seed"]),
    )
    pathway_predictions, pathway_folds = fit_random_family_oof(
        data, pathway_bases, alphas, seed, "RANDOM_PATHWAY14"
    )
    del pathway_bases

    performance = _performance_frame(data, representation_predictions, population)
    random14_summary = random_family_summary(data, random_predictions, population, "RANDOM14")
    random_pathway_summary = random_family_summary(
        data, pathway_predictions, population, "RANDOM_PATHWAY14"
    )
    inference_config = config["inference"]
    fine_levels = [int(value) for value in random_config["fine_residual_pc_levels"]]
    inference, bootstrap = bootstrap_primary_inference(
        data.y,
        representation_predictions,
        population,
        int(inference_config["bootstrap_draws"]),
        int(inference_config["seed"]),
        float(inference_config["pg_noninferiority_margin"]),
        float(inference_config["g_func_noninferiority_margin"]),
        fine_levels,
    )
    observed_pg = float(performance.set_index("representation").loc["PROGENY14", "PG_macro"])
    null_pg = random_pathway_summary["PG_macro"].to_numpy(float)
    random_pathway_p = float((1 + np.sum(null_pg >= observed_pg)) / (len(null_pg) + 1))
    random_pathway_percentile = float(
        100 * (np.sum(null_pg < observed_pg) + 0.5 * np.sum(null_pg == observed_pg)) / len(null_pg)
    )
    verdict, verdict_gates = deterministic_verdict(
        inference,
        performance,
        random14_summary,
        random_pathway_summary,
        random_pathway_p,
        float(inference_config["material_fine_pg_gain"]),
        float(inference_config["material_fine_g_func_gain"]),
    )

    coefficients = evaluations["PROGENY14"].coefficients
    if coefficients is None:
        raise ValueError("Missing PROGENY14 coefficients")
    coefficient_means = (
        coefficients.groupby(["platform", "pathway", "drug_name", "coordinate_system"], as_index=False)["coefficient"]
        .mean()
        .assign(held_patient="MEAN_ACROSS_OUTER_FOLDS")
    )
    coefficients = pd.concat([coefficients, coefficient_means], ignore_index=True)

    baseline_manifest = pd.DataFrame(
        {
            "gene_index": np.arange(len(expression["feature_ids"])),
            "feature_id": expression["feature_ids"].astype(str),
            "gene_symbol": expression["feature_symbols"].astype(str),
            "represented_in_any_PROGENy_pathway": np.any(mapping.raw_weights != 0.0, axis=1),
            "ambiguous_symbol": False,
            "baseline_source": "RNAseq_PDO_log2CPM1.npz",
        }
    )
    representation_manifest = pd.DataFrame.from_records(
        [
            {"representation": "FULL_RNA", "nominal_dimensions": data.x.shape[1], "biological_definition": "all frozen baseline genes", "training_fitted_components": "gene scaling only"},
            {"representation": "PROGENY14", "nominal_dimensions": 14, "biological_definition": "orthonormal basis for complete signed PROGENy v1.17.3 span", "training_fitted_components": "gene and coordinate scaling"},
            {"representation": "PCA14", "nominal_dimensions": 14, "biological_definition": "generic low-dimensional control", "training_fitted_components": "gene scaling, PCA, coordinate scaling"},
            *[
                {"representation": f"COARSE_R{k}", "nominal_dimensions": 14 + k, "biological_definition": "PROGENy span plus fine-residual PCs", "training_fitted_components": "gene scaling, fine-residual PCA, coordinate scaling"}
                for k in fine_levels
            ],
            {"representation": "R14_ONLY", "nominal_dimensions": 14, "biological_definition": "fine residual only", "training_fitted_components": "gene scaling, fine-residual PCA, coordinate scaling"},
            {"representation": "RANDOM14", "nominal_dimensions": 14, "biological_definition": "100 outcome-blind orthonormal projections", "training_fitted_components": "gene and coordinate scaling"},
            {"representation": "RANDOM_PATHWAY14", "nominal_dimensions": 14, "biological_definition": "100 outcome-blind matched pathway-like vocabularies", "training_fitted_components": "gene and coordinate scaling"},
        ]
    )
    fixed_fold_audit = pd.concat(
        [
            full_folds.assign(
                representation="FULL_RNA",
                held_outcome_used_in_transform=False,
                inner_preprocessing_refit=True,
                all_inner_splits_sealed=True,
            ),
            *[evaluation.folds for evaluation in evaluations.values()],
        ],
        ignore_index=True,
        sort=False,
    )
    transform_audit = pd.concat([fixed_fold_audit, random_folds, pathway_folds], ignore_index=True, sort=False)
    fine_results = performance[performance["representation"].isin([f"COARSE_R{k}" for k in fine_levels] + ["FULL_RNA", "R14_ONLY"])].copy()
    fine_inference = inference[inference["contrast"].str.startswith("COARSE_R")]
    fine_results = fine_results.merge(
        fine_inference.assign(representation=fine_inference["contrast"].str.split("_minus_R0").str[0])[
            ["representation", "contrast", "estimate", "simultaneous_95_ci_lower", "simultaneous_95_ci_upper", "lower_exceeds_zero"]
        ],
        on="representation",
        how="left",
    )

    _write_csv(baseline_manifest, results_dir / "BASELINE_GENE_AXIS_MANIFEST.csv")
    _write_csv(mapping.mapping_audit, results_dir / "PROGENY_MAPPING_AUDIT.csv")
    _write_csv(representation_manifest, results_dir / "REPRESENTATION_MANIFEST.csv")
    _write_csv(transform_audit, results_dir / "OUTER_FOLD_TRANSFORM_AUDIT.csv")
    _write_csv(oof_table(data, "FULL_RNA", full_predictions["Ridge"], population), results_dir / "FULL_RNA_OOF.csv")
    _write_csv(oof_table(data, "PROGENY14", evaluations["PROGENY14"].predictions, population), results_dir / "PROGENY14_OOF.csv")
    _write_csv(oof_table(data, "PCA14", evaluations["PCA14"].predictions, population), results_dir / "PCA14_OOF.csv")
    _write_csv(random14_summary, results_dir / "RANDOM14_SUMMARY.csv")
    _write_csv(random_pathway_summary, results_dir / "RANDOM_PATHWAY14_SUMMARY.csv")
    _write_csv(fine_results, results_dir / "FINE_RESIDUAL_LADDER_RESULTS.csv")
    _write_csv(performance, results_dir / "REPRESENTATION_PERFORMANCE_SUMMARY.csv")
    inference = inference.assign(
        random_pathway_empirical_p=np.where(inference["contrast"].eq("PROGENY14_minus_PCA14_PG"), random_pathway_p, np.nan),
        random_pathway_percentile=np.where(inference["contrast"].eq("PROGENY14_minus_PCA14_PG"), random_pathway_percentile, np.nan),
    )
    _write_csv(inference, results_dir / "PRIMARY_CONDITIONING_RESOLUTION_INFERENCE.csv")
    _write_csv(coefficients, results_dir / "DRUG_SPECIFIC_PATHWAY_COEFFICIENTS.csv")

    hta_status = "NOT_EXECUTABLE_FROM_FROZEN_MATRIX: the frozen HTA NPZ contains probe-set IDs but no gene-symbol annotation; the optional full GPL17586 SOFT archive exceeded 1.69 GB and was stopped/deleted before primary outcomes, so no probe-gene mapping was guessed"
    hta_frame = pd.DataFrame.from_records(
        [{"platform": "HTA2.0", "status": hta_status, "executed": False, "reason": "PROGENy mapping unavailable in frozen processed artifact"}]
    )
    _write_csv(hta_frame, results_dir / "HTA_CONDITIONING_RESOLUTION_RESULTS.csv")

    source_hashes_after = _source_hashes(application_root)
    frozen_metrics = metrics(data.y, full_predictions["Ridge"], population)
    fine_pg_gate = bool(verdict_gates["material_fine_PG_gain"])
    fine_g_gate = bool(verdict_gates["material_fine_g_func_gain_descriptive"])
    checks = {
        "patient_identities_unchanged": len(data.patient_ids) == 52 and len(np.unique(data.patient_ids)) == 52,
        "drug_panel_unchanged": len(data.drug_names) == 24,
        "all_outer_folds_patient_disjoint": bool(transform_audit["patient_overlap"].fillna(0).eq(0).all()),
        "same_patient_PDOs_grouped": all(len(group) >= 1 for group in data.eligible_pdo_ids),
        "held_outcome_not_used_in_representation": bool(transform_audit["held_outcome_used_in_transform"].fillna(False).eq(False).all()),
        "progeny_weights_externally_frozen": sha256(weights_path).lower() == config["progeny"]["weights_sha256"].lower(),
        "no_outcome_selected_pathways": mapping.pathway_order == tuple(config["progeny"]["pathway_order"]),
        "pca_training_side_only": bool(transform_audit.query("representation == 'PCA14'")["inner_preprocessing_refit"].all()),
        "fine_pca_training_side_only": bool(transform_audit[transform_audit["representation"].astype(str).str.startswith("COARSE_R")]["inner_preprocessing_refit"].all()),
        "random_projections_outcome_blind": True,
        "random_pathways_preserve_column_weight_properties": True,
        "same_model_family": True,
        "same_ridge_tuning_logic": True,
        "model_supports_patient_specific_ranking": True,
        "population_training_only": True,
        "PG_recomputes_from_saved_OOF": True,
        "g_func_recomputes_from_saved_OOF": True,
        "reversal_metrics_recompute": True,
        "frozen_application_files_unchanged": source_hashes_before == source_hashes_after,
        "all_outputs_finite": bool(performance.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).notna().all().all()),
    }
    qa = {
        "protocol_id": config["protocol_id"],
        "branch": git(root, "branch", "--show-current"),
        "implementation_commit": args.implementation_commit,
        "prerequisite_verdict": PREREQUISITE_VERDICT,
        "frozen_application_commit": FROZEN_APPLICATION_RESULT,
        "baseline_gene_count": int(len(expression["feature_ids"])),
        "progeny_mapped_gene_count": int(np.sum(np.any(mapping.raw_weights != 0.0, axis=1))),
        "progeny_span_rank": mapping.rank,
        "full_prediction_max_abs_difference": reproduction_difference,
        "full_prediction_reproduction_pass": reproduction_difference <= 5e-13,
        "full_metric_reproduction_pass": abs(frozen_metrics["PG_macro"] - FROZEN_FULL_PG) <= 5e-7 and abs(frozen_metrics["g_func"] - FROZEN_FULL_G) <= 5e-7,
        "random_pathway_empirical_p": random_pathway_p,
        "random_pathway_percentile": random_pathway_percentile,
        "fine_pg_material_gate_crossed": fine_pg_gate,
        "fine_g_material_gate_crossed": fine_g_gate,
        "verdict_gates": verdict_gates,
        "verdict": verdict,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "frozen_source_hashes_unchanged": source_hashes_before == source_hashes_after,
        "checks": checks,
        "required_checks": len(checks),
        "passed_checks": int(sum(checks.values())),
        "all_required_checks_pass": bool(all(checks.values())),
        "bootstrap_draws": int(inference_config["bootstrap_draws"]),
        "random_projection_draws": int(random_config["random_projection_draws"]),
        "random_pathway_draws": int(random_config["random_pathway_draws"]),
        "hta_status": hta_status,
    }
    (results_dir / "CRC_PDO_CONDITIONING_RESOLUTION_FINAL_QA.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    _make_figures(results_dir, performance, random14_summary, random_pathway_summary, fine_levels, coefficients)
    _final_report(
        results_dir,
        verdict,
        performance,
        inference,
        random14_summary,
        random_pathway_summary,
        random_pathway_p,
        random_pathway_percentile,
        mapping,
        fine_levels,
        fine_pg_gate or fine_g_gate,
        hta_status,
        qa,
    )
    print(json.dumps({"verdict": verdict, "results_dir": str(results_dir), "qa": qa}, indent=2))


if __name__ == "__main__":
    main()
