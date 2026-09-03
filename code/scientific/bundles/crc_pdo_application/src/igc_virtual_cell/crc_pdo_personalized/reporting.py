from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#25282D",
    "slate": "#667085",
    "teal": "#2A7F8E",
    "coral": "#C8785C",
    "blue": "#4E79A7",
    "light": "#DDE5E8",
}


def _style() -> None:
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
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(results_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    _style()
    gain = tables["PERSONALIZATION_GAIN.csv"]
    primary = gain.query("platform == 'RNAseq' and estimator == 'Ridge'").copy()

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.scatter(
        primary["kendall_population"],
        primary["kendall_model"],
        c=np.where(primary["PG_i"] >= 0, PALETTE["teal"], PALETTE["coral"]),
        s=23,
        linewidths=0.35,
        edgecolors="white",
        alpha=0.9,
    )
    limits = [
        min(primary["kendall_population"].min(), primary["kendall_model"].min()) - 0.03,
        max(primary["kendall_population"].max(), primary["kendall_model"].max()) + 0.03,
    ]
    ax.plot(limits, limits, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.set(xlim=limits, ylim=limits, xlabel="Population Kendall tau-b", ylabel="Personalized Kendall tau-b")
    ax.set_title("Held-patient ex vivo drug ranking", loc="center", fontweight="bold")
    _save(fig, results_dir / "PERSONALIZED_VS_POPULATION_RANKING.png")

    ordered = primary.sort_values("PG_i").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.25, 2.25))
    colors = np.where(ordered["PG_i"] >= 0, PALETTE["teal"], PALETTE["coral"])
    ax.vlines(np.arange(len(ordered)), 0, ordered["PG_i"], color=colors, lw=1.0)
    ax.scatter(np.arange(len(ordered)), ordered["PG_i"], color=colors, s=12, zorder=3)
    ax.axhline(0, color=PALETTE["ink"], lw=0.65)
    ax.set(xlabel="Held patients (sorted)", ylabel=r"Personalization gain, $PG_i$")
    ax.set_title("Baseline RNA helps some patients and harms others", loc="center", fontweight="bold")
    ax.set_xticks([])
    _save(fig, results_dir / "PERSONALIZATION_GAIN_BY_PATIENT.png")

    reversals = tables["PREFERENCE_REVERSALS.csv"].query(
        "platform == 'RNAseq' and estimator == 'Ridge'"
    )
    actual = reversals["true_reversal"].to_numpy(bool)
    predicted = reversals["predicted_reversal"].to_numpy(bool)
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    values = [actual.mean(), tp / (tp + fn), tp / (tp + fp), 0.5 * (tp / (tp + fn) + tn / (tn + fp))]
    labels = ["True\nprevalence", "Recall", "Precision", "Balanced\naccuracy"]
    fig, ax = plt.subplots(figsize=(3.7, 2.4))
    ax.bar(np.arange(4), values, color=[PALETTE["slate"], PALETTE["teal"], PALETTE["blue"], PALETTE["coral"]], width=0.62)
    ax.axhline(0.5, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.set(ylim=(0, 1), ylabel="Fraction", xticks=np.arange(4), xticklabels=labels)
    ax.set_title("Patient-specific preference reversals", loc="center", fontweight="bold")
    _save(fig, results_dir / "PATIENT_SPECIFIC_PREFERENCE_REVERSALS.png")

    near = tables["NEAR_TIE_REVERSAL_ANALYSIS.csv"].query(
        "platform == 'RNAseq' and estimator == 'Ridge'"
    ).set_index("population_margin_bin").loc[["low", "middle", "high"]]
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    x = np.arange(3)
    ax.plot(x, near["reversal_recall"], marker="o", color=PALETTE["teal"], lw=1.2, label="Recall")
    ax.plot(x, near["balanced_accuracy"], marker="o", color=PALETTE["coral"], lw=1.2, label="Balanced accuracy")
    ax.axhline(0.5, color=PALETTE["slate"], lw=0.7, ls="--")
    ax.set(xticks=x, xticklabels=["Low", "Middle", "High"], ylim=(0, 1), xlabel="Training-population margin", ylabel="Reversal recovery")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Recovery across prespecified margin tertiles", loc="center", fontweight="bold")
    _save(fig, results_dir / "REVERSAL_RECOVERY_BY_POPULATION_MARGIN.png")

    inference = tables["PRIMARY_APPLICATION_INFERENCE.csv"]
    labels = [r"$PG_{macro}$", r"$g_{func}$", "Reversal BA - 0.5"]
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    y = np.arange(3)[::-1]
    estimates = inference["estimate"].to_numpy()
    low = inference["simultaneous_95_ci_lower"].to_numpy()
    high = inference["simultaneous_95_ci_upper"].to_numpy()
    ax.errorbar(estimates, y, xerr=[estimates - low, high - estimates], fmt="o", color=PALETTE["teal"], ecolor=PALETTE["slate"], capsize=2, lw=0.9)
    ax.axvline(0, color=PALETTE["ink"], lw=0.65)
    ax.set(yticks=y, yticklabels=labels, xlabel="Estimate (simultaneous 95% interval)")
    ax.set_title("Functional personalization recovery", loc="center", fontweight="bold")
    _save(fig, results_dir / "FUNCTIONAL_PERSONALIZATION_RECOVERY.png")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_qa(
    results_dir: Path,
    *,
    source_manifest: dict[str, Any],
    panel_freeze_commit: str,
    prediction_implementation_commit: str,
    finalization_commit: str,
    verdict: str,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    required = [
        "CRC_PDO_PATIENT_MANIFEST.csv",
        "CRC_PDO_DRUG_COVERAGE.csv",
        "CRC_PDO_EXPRESSION_COVERAGE.csv",
        "PATIENT_PDO_EXPRESSION_DRUG_INTERSECTION.csv",
        "PRIMARY_DRUG_PANEL_MANIFEST.csv",
        "HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv",
        "HELD_PATIENT_DRUG_RANKINGS.csv",
        "POPULATION_REFERENCE_RANKINGS.csv",
        "PERSONALIZATION_GAIN.csv",
        "FUNCTIONAL_PERSONALIZATION_RECOVERY.csv",
        "PREFERENCE_REVERSALS.csv",
        "NEAR_TIE_REVERSAL_ANALYSIS.csv",
        "TOPK_RECOVERY.csv",
        "NULL_CONTROLS.csv",
        "PRIMARY_APPLICATION_INFERENCE.csv",
        "MODEL_FOLD_MANIFEST.csv",
        "PATIENT_LEVEL_SUMMARY.csv",
        "PERSONALIZED_VS_POPULATION_RANKING.png",
        "PERSONALIZATION_GAIN_BY_PATIENT.png",
        "PATIENT_SPECIFIC_PREFERENCE_REVERSALS.png",
        "REVERSAL_RECOVERY_BY_POPULATION_MARGIN.png",
        "FUNCTIONAL_PERSONALIZATION_RECOVERY.png",
        "CRC_PDO_PERSONALIZED_APPLICATION_FINAL.md",
    ]
    fold_manifest = tables["MODEL_FOLD_MANIFEST.csv"]
    predictions = tables["HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv"]
    panel = tables["PRIMARY_DRUG_PANEL_MANIFEST.csv"]
    forbidden_terms = ("pathway", "clinical efficacy", "treatment recommendation")
    text_to_scan = (results_dir / "CRC_PDO_PERSONALIZED_APPLICATION_FINAL.md").read_text(encoding="utf-8").lower()
    qa = {
        "protocol_id": "CRC_PDO_PERSONALIZED_APPLICATION_V1",
        "verdict": verdict,
        "panel_freeze_commit": panel_freeze_commit,
        "prediction_implementation_commit": prediction_implementation_commit,
        "finalization_commit": finalization_commit,
        "all_required_files_present": all((results_dir / name).exists() for name in required),
        "required_files": required,
        "patient_overlap_max": int(fold_manifest["patient_overlap"].max()),
        "all_predictions_finite": bool(np.isfinite(predictions["predicted_DSS"]).all()),
        "frozen_drug_count": int(panel["drug_name"].nunique()),
        "frozen_panel_missingness_max": float(panel["missingness"].max()),
        "rna_patients": int(fold_manifest.query("platform == 'RNAseq'")["held_patient"].nunique()),
        "array_patients": int(fold_manifest.query("platform == 'HTA2.0'")["held_patient"].nunique()),
        "raw_sequencing_downloaded": False,
        "largest_deliberately_skipped_file_bytes": 4582666240,
        "forbidden_claim_terms_absent": {term: term not in text_to_scan for term in forbidden_terms},
        "source_manifest_sha256": hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest(),
        "output_sha256": {name: sha256_file(results_dir / name) for name in required if (results_dir / name).exists()},
    }
    qa["qa_pass"] = bool(
        qa["all_required_files_present"]
        and qa["patient_overlap_max"] == 0
        and qa["all_predictions_finite"]
        and qa["frozen_drug_count"] == 24
        and qa["frozen_panel_missingness_max"] == 0
        and qa["rna_patients"] == 52
        and qa["array_patients"] == 50
        and all(qa["forbidden_claim_terms_absent"].values())
    )
    (results_dir / "CRC_PDO_PERSONALIZED_APPLICATION_FINAL_QA.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    return qa
