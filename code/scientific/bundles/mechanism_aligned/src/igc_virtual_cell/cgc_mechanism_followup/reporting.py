"""Deterministic, reporting-only finalization of frozen BIO-2/MULTI-2 results.

This module is deliberately unable to fit models, regenerate nulls, recalibrate
power, or rerun the bridge.  It reads the frozen result tables and writes only
descriptive summaries, reconciliation records, figures, source data, and prose.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import OUT, ROOT, sha256


GROUP_KEYS = (
    "response_target",
    "panel",
    "modality",
    "model",
    "comparator",
    "estimator",
)
GROUP_AXES = ("moa_broad", "moa_fine", "target_family", "exact_target")
PANEL_ORDER = (
    "WIRING",
    "ACTIVITY",
    "PHOSPHO",
    "TOTAL_PROTEIN",
    "DEPENDENCY_ORACLE",
    "M5",
)
LEVEL_ORDER = ("BROAD_ACTION", "MOA", "TARGET_FAMILY", "EXACT_TARGET")
ESTIMATOR_ORDER = ("ridge", "bilinear_ridge")

BIO2_WIDTH_RATIOS = (2.15, 1.0)
MULTI2_WIDTH_RATIOS = (2.25, 1.0)
BRIDGE_WIDTH_RATIOS = (2.1, 1.0)

BLUE = "#236192"
TEAL = "#2A9D8F"
ORANGE = "#D57A2A"
RED = "#B94C4C"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
NORMAL_975 = 1.959963984540054

NULL_P_SEMANTICS = (
    "one-sided +1-corrected hierarchical-bootstrap tail probability "
    "Pr_boot(observed-minus-null <= 0) from one frozen feature-null OOF refit; "
    "not a frequency over repeated null refits"
)

REQUIRED_INPUTS = (
    "BIO2_HIERARCHY_RESULTS.csv",
    "BIO2_RUN_MANIFEST.json",
    "BIO2_MULTI2_BRIDGE.csv",
    "BIO2_MULTI2_BRIDGE_RESULT.json",
    "bio2_multi2_bridge_null.parquet",
    "MULTI2_INCREMENTAL_RECOVERY.csv",
    "MULTI2_PER_INTERVENTION_GAINS.csv",
    "MULTI2_ALIGNMENT_NULLS.csv",
    "MULTI2_POWER_CURVES.csv",
    "MULTI2_DETECTION_LIMITS.csv",
    "MULTI2_RUN_MANIFEST.json",
)

REPORTING_OUTPUTS = (
    "MULTI2_GROUP_SUMMARIES.csv",
    "CGC_MECHANISM_NUMERICAL_RECONCILIATION.csv",
    "CGC_MECHANISM_FOLLOWUP_FINAL.md",
    "figures/CGC_BIO2_FINAL.png",
    "figures/CGC_BIO2_FINAL.svg",
    "figures/CGC_MULTI2_FINAL.png",
    "figures/CGC_MULTI2_FINAL.svg",
    "figures/CGC_BRIDGE_FINAL.png",
    "figures/CGC_BRIDGE_FINAL.svg",
    "figure_source_data/CGC_BIO2_FINAL_SOURCE_DATA.csv",
    "figure_source_data/CGC_MULTI2_FINAL_SOURCE_DATA.csv",
    "figure_source_data/CGC_BRIDGE_FINAL_SOURCE_DATA.csv",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def _tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return list(dict.fromkeys(token.strip() for token in str(value).split(";") if token.strip()))


def _group_memberships(gains: pd.DataFrame) -> pd.DataFrame:
    """Expand only the four frozen annotation fields into exhaustive memberships."""

    _require_columns(
        gains,
        (*GROUP_KEYS, "intervention_axis", "intervention_id",
         "delta_g_aligned_minus_rna", "paired_se", "ci_low", "ci_high",
         "positive_context_fraction", "moa_broad", "moa_fine",
         "hgnc_target_family_ids", "hgnc_target_family_names", "exact_targets"),
        "MULTI2_PER_INTERVENTION_GAINS.csv",
    )
    rows: list[dict[str, object]] = []
    for record in gains.to_dict("records"):
        base = {key: record[key] for key in GROUP_KEYS}
        base.update(
            intervention_axis=int(record["intervention_axis"]),
            intervention_id=str(record["intervention_id"]),
            delta_g_aligned_minus_rna=float(record["delta_g_aligned_minus_rna"]),
            paired_se=float(record["paired_se"]),
            ci_low=float(record["ci_low"]),
            ci_high=float(record["ci_high"]),
            positive_context_fraction=float(record["positive_context_fraction"]),
        )
        for axis in ("moa_broad", "moa_fine"):
            label = str(record[axis]).strip()
            if not label:
                label = "UNANNOTATED"
            rows.append({**base, "group_axis": axis, "group_id": label, "group_label": label})

        family_ids = _tokens(record["hgnc_target_family_ids"])
        family_names = _tokens(record["hgnc_target_family_names"])
        if len(family_ids) != len(family_names):
            raise RuntimeError(
                f"target-family ID/name mismatch for {record['intervention_id']}: "
                f"{len(family_ids)} IDs vs {len(family_names)} names"
            )
        if not family_ids:
            rows.append(
                {**base, "group_axis": "target_family", "group_id": "UNANNOTATED",
                 "group_label": "UNANNOTATED"}
            )
        else:
            for family_id, family_name in zip(family_ids, family_names, strict=True):
                rows.append(
                    {**base, "group_axis": "target_family", "group_id": family_id,
                     "group_label": family_name}
                )

        targets = _tokens(record["exact_targets"])
        if not targets:
            rows.append(
                {**base, "group_axis": "exact_target", "group_id": "UNANNOTATED",
                 "group_label": "UNANNOTATED"}
            )
        else:
            for target in targets:
                rows.append(
                    {**base, "group_axis": "exact_target", "group_id": target,
                     "group_label": target}
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no MULTI2 group memberships were generated")
    return result


def _inverse_variance_random_effects(
    effects: np.ndarray, standard_errors: np.ndarray
) -> dict[str, float]:
    """DerSimonian-Laird inverse-variance random-effects summary.

    This is a reporting-only aggregation of frozen per-intervention effects and
    paired standard errors. It does not refit any prediction model.
    """

    effects = np.asarray(effects, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    valid = np.isfinite(effects) & np.isfinite(standard_errors) & (standard_errors > 0)
    if not valid.all():
        raise RuntimeError("random-effects group contains a non-finite or non-positive standard error")
    if not len(effects):
        raise RuntimeError("random-effects group is empty")
    if len(effects) == 1:
        # A single intervention has no between-intervention degrees of freedom.
        # Preserve its frozen effect separately in build_group_summaries, but do
        # not manufacture a random-effects estimate or zero heterogeneity.
        return {
            "random_effect_mean": np.nan,
            "random_effect_se": np.nan,
            "random_effect_ci_low": np.nan,
            "random_effect_ci_high": np.nan,
            "tau2_dl": np.nan,
            "q_heterogeneity": np.nan,
            "i2_percent": np.nan,
            "inverse_variance_weight_sum": np.nan,
        }
    variances = standard_errors ** 2
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (effects - fixed_mean) ** 2))
    df = max(len(effects) - 1, 0)
    c = float(np.sum(fixed_weights) - np.sum(fixed_weights ** 2) / np.sum(fixed_weights))
    tau2 = float(max(0.0, (q - df) / c)) if df and c > 0 else 0.0
    random_weights = 1.0 / (variances + tau2)
    mean = float(np.sum(random_weights * effects) / np.sum(random_weights))
    se = float(np.sqrt(1.0 / np.sum(random_weights)))
    i2 = float(max(0.0, (q - df) / q) * 100.0) if q > 0 and df else 0.0
    return {
        "random_effect_mean": mean,
        "random_effect_se": se,
        "random_effect_ci_low": mean - NORMAL_975 * se,
        "random_effect_ci_high": mean + NORMAL_975 * se,
        "tau2_dl": tau2,
        "q_heterogeneity": q,
        "i2_percent": i2,
        "inverse_variance_weight_sum": float(np.sum(random_weights)),
    }


def build_group_summaries(gains: pd.DataFrame) -> pd.DataFrame:
    """Summarize all frozen groups, estimating random effects only when possible."""

    memberships = _group_memberships(gains)
    keys = [*GROUP_KEYS, "group_axis", "group_id", "group_label"]
    rows: list[dict[str, object]] = []
    for key, group in memberships.groupby(keys, sort=True, dropna=False):
        values = group["delta_g_aligned_minus_rna"].to_numpy(float)
        n_interventions = int(group["intervention_axis"].nunique())
        if n_interventions != len(group):
            raise RuntimeError("a group contains duplicate intervention memberships")
        singleton = n_interventions == 1
        random_effects = _inverse_variance_random_effects(
            values, group["paired_se"].to_numpy(float)
        )
        single_row = group.iloc[0] if singleton else None
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                **random_effects,
                "n_interventions": n_interventions,
                "heterogeneity_estimable": not singleton,
                "single_intervention_effect": (
                    float(single_row["delta_g_aligned_minus_rna"]) if singleton else np.nan
                ),
                "single_intervention_se": (
                    float(single_row["paired_se"]) if singleton else np.nan
                ),
                "single_intervention_ci_low": (
                    float(single_row["ci_low"]) if singleton else np.nan
                ),
                "single_intervention_ci_high": (
                    float(single_row["ci_high"]) if singleton else np.nan
                ),
                "mean_delta_g": float(np.mean(values)),
                "median_delta_g": float(np.median(values)),
                "q25_delta_g": float(np.quantile(values, 0.25)),
                "q75_delta_g": float(np.quantile(values, 0.75)),
                "min_delta_g": float(np.min(values)),
                "max_delta_g": float(np.max(values)),
                "positive_fraction": float(np.mean(values > 0.0)),
                "ci_positive_fraction": float(np.mean(group["ci_low"].to_numpy(float) > 0.0)),
                "mean_positive_context_fraction": float(
                    np.mean(group["positive_context_fraction"].to_numpy(float))
                ),
                "summary_scope": "ALL_FROZEN_MEMBERS_NO_SELECTION",
                "random_effect_method": (
                    "NOT_APPLICABLE_SINGLE_INTERVENTION"
                    if singleton else "DER_SIMONIAN_LAIRD_INVERSE_VARIANCE"
                ),
                "inference_status": (
                    "NOT_ESTIMABLE_SINGLE_INTERVENTION"
                    if singleton else "PREREGISTERED_HETEROGENEITY_SUMMARY_NO_SELECTION"
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)
    return result


def _incremental_claim_boundary(row: pd.Series) -> str:
    if row.response_target != "PRIMARY_RESIDUAL":
        return "Secondary-Gamma confirmation axis; it does not modify the primary-residual verdict."
    aligned_models = {"M4_RNA_ALIGNED", "M_ORACLE_RNA_DEPENDENCY", "M5_RNA_WIRING_ACTIVITY"}
    if row.model in aligned_models and row.comparator == "M0_RNA":
        return "Primary aligned-versus-RNA recovery contrast; no lower confidence bound exceeds zero."
    if row.comparator == "M3_RNA_GENERIC":
        return "Aligned-versus-generic diagnostic; it measures avoidance of generic-feature degradation, not recovery over RNA."
    if row.model == "M3_RNA_GENERIC" and row.comparator == "M0_RNA":
        return "Generic-versus-RNA diagnostic; it quantifies generic-feature degradation."
    if row.model == "M2_ALIGNED" and row.comparator == "M1_GENERIC":
        return "Feature-only aligned-versus-generic diagnostic without baseline RNA; not a primary remedy test."
    raise RuntimeError(
        f"unrecognized incremental contrast for claim boundary: "
        f"{row.response_target}|{row.model}|{row.comparator}"
    )


def _record(**values: object) -> dict[str, object]:
    columns = (
        "analysis", "record_type", "record_id", "metric", "response_target",
        "panel", "modality", "model", "comparator", "estimator", "level",
        "representation", "direction", "null_type", "target_delta_g",
        "estimate", "estimate_text", "observed_estimate", "reference_estimate",
        "ci_low", "ci_high", "p_value", "p_value_semantics", "power", "bias",
        "false_positive_rate", "ci_coverage", "n", "n_semantics", "status",
        "source_file", "source_row_key", "claim_boundary",
    )
    return {column: values.get(column, np.nan) for column in columns}


def build_reconciliation(
    bio: pd.DataFrame,
    incremental: pd.DataFrame,
    nulls: pd.DataFrame,
    power: pd.DataFrame,
    limits: pd.DataFrame,
    bridge: dict[str, object],
) -> pd.DataFrame:
    """Normalize every frozen aggregate numerical row into one audit table."""

    rows: list[dict[str, object]] = []
    for index, row in bio.reset_index(drop=True).iterrows():
        key = f"{row.level}|{row.representation}|{row.direction}"
        rows.append(_record(
            analysis="BIO2", record_type="HIERARCHY_CONDITIONAL", record_id=f"BIO2-C-{index:03d}",
            metric="conditional_effect", level=row.level, representation=row.representation,
            direction=row.direction, estimate=row.conditional_effect, ci_low=row.ci_low,
            ci_high=row.ci_high, p_value=row.qap_p_positive,
            p_value_semantics="frozen conditional QAP positive-tail p+",
            n=row.independent_clusters, n_semantics="independent annotation clusters",
            status=row.decision_status, source_file="BIO2_HIERARCHY_RESULTS.csv",
            source_row_key=key,
            claim_boundary="Highest robust attribution is NONE; exact target is LIMITED_POWER.",
        ))
        rows.append(_record(
            analysis="BIO2", record_type="HIERARCHY_MATCHED", record_id=f"BIO2-M-{index:03d}",
            metric="matched_effect", level=row.level, representation=row.representation,
            direction=row.direction, estimate=row.matched_effect, ci_low=row.matched_ci_low,
            ci_high=row.matched_ci_high, n=row.matched_positive_pairs,
            n_semantics="matched positive pairs", status=row.decision_status,
            source_file="BIO2_HIERARCHY_RESULTS.csv", source_row_key=key,
            claim_boundary="Matched estimates are confirmation axes and do not upgrade robustness.",
        ))

    for index, row in incremental.reset_index(drop=True).iterrows():
        key = "|".join(str(row[column]) for column in (
            "response_target", "panel", "model", "comparator", "estimator"
        ))
        rows.append(_record(
            analysis="MULTI2", record_type="INCREMENTAL_RECOVERY",
            record_id=f"MULTI2-I-{index:03d}", metric="delta_g",
            response_target=row.response_target, panel=row.panel, modality=row.modality,
            model=row.model, comparator=row.comparator, estimator=row.estimator,
            estimate=row.delta_g, ci_low=row.ci_low, ci_high=row.ci_high,
            n=row.bootstrap_draws, n_semantics="frozen-table hierarchical bootstrap draws",
            status=row.status, source_file="MULTI2_INCREMENTAL_RECOVERY.csv",
            source_row_key=key,
            claim_boundary=_incremental_claim_boundary(row),
        ))

    for index, row in nulls.reset_index(drop=True).iterrows():
        key = "|".join(str(row[column]) for column in (
            "response_target", "panel", "null_type", "estimator"
        ))
        rows.append(_record(
            analysis="MULTI2", record_type="FEATURE_NULL_CONTRAST",
            record_id=f"MULTI2-N-{index:03d}", metric="observed_minus_feature_null_delta_g",
            response_target=row.response_target, panel=row.panel, modality=row.modality,
            estimator=row.estimator, null_type=row.null_type,
            estimate=row.contrast_delta_g, observed_estimate=row.observed_delta_g,
            reference_estimate=row.null_delta_g, ci_low=row.contrast_ci_low,
            ci_high=row.contrast_ci_high, p_value=row.null_p,
            p_value_semantics=NULL_P_SEMANTICS, n=row.table_draws,
            n_semantics=f"hierarchical-bootstrap table draws; OOF refits={int(row.oof_refits)}",
            status=row.status, source_file="MULTI2_ALIGNMENT_NULLS.csv",
            source_row_key=key,
            claim_boundary="A contrast against a feature null cannot rescue a non-positive aligned-versus-RNA result.",
        ))

    for index, row in power.reset_index(drop=True).iterrows():
        key = f"{row.response_target}|{row.panel}|{row.target_delta_g}"
        rows.append(_record(
            analysis="MULTI2", record_type="POWER_CURVE", record_id=f"MULTI2-P-{index:03d}",
            metric="fold_local_detection_power", response_target=row.response_target,
            panel=row.panel, modality=row.modality, target_delta_g=row.target_delta_g,
            estimate=row.estimate, power=row.power, bias=row.bias,
            false_positive_rate=row.false_positive_rate, ci_coverage=row.ci_coverage,
            n=row.replicates, n_semantics="synthetic replicates at this frozen effect level",
            status=row.status, source_file="MULTI2_POWER_CURVES.csv", source_row_key=key,
            claim_boundary="Non-recovery supports absence only at or above the panel-specific MDE80.",
        ))

    for index, row in limits.reset_index(drop=True).iterrows():
        key = f"{row.response_target}|{row.panel}|{row.estimator}"
        numeric_mde = pd.to_numeric(pd.Series([row.mde80]), errors="coerce").iloc[0]
        rows.append(_record(
            analysis="MULTI2", record_type="DETECTION_LIMIT", record_id=f"MULTI2-D-{index:03d}",
            metric="MDE80", response_target=row.response_target, panel=row.panel,
            modality=row.modality, estimator=row.estimator,
            estimate=numeric_mde, estimate_text=str(row.mde80), n=row.replicates,
            n_semantics="synthetic replicates per frozen effect level", status=row.status,
            source_file="MULTI2_DETECTION_LIMITS.csv", source_row_key=key,
            claim_boundary="MDE is a resolution limit, not evidence for a smaller absent effect.",
        ))

    rows.append(_record(
        analysis="BIO2_MULTI2_BRIDGE", record_type="FROZEN_SECONDARY_BRIDGE",
        record_id="BRIDGE-001", metric="spearman_rho",
        estimate=float(bridge["spearman_rho"]),
        p_value=float(bridge["blocked_permutation_p_two_sided"]),
        p_value_semantics="two-sided structure-preserving permutation p",
        n=int(bridge["n_complete_interventions"]), n_semantics="complete intervention pairs",
        status="FROZEN_SECONDARY_NULL", source_file="BIO2_MULTI2_BRIDGE_RESULT.json",
        source_row_key="spearman_rho|blocked_permutation_p_two_sided",
        claim_boundary="The null bridge does not erase either primary analysis and does not support preferential rescue.",
    ))
    rows.append(_record(
        analysis="BIO2_MULTI2_BRIDGE", record_type="FROZEN_SECONDARY_BRIDGE",
        record_id="BRIDGE-002", metric="positive_direction_p",
        estimate=float(bridge["blocked_permutation_p_positive"]),
        n=int(bridge["draws"]), n_semantics="structure-preserving permutation draws",
        status="FROZEN_SECONDARY_NULL", source_file="BIO2_MULTI2_BRIDGE_RESULT.json",
        source_row_key="blocked_permutation_p_positive",
        claim_boundary="No positive BIO2-to-MULTI2 association was established.",
    ))
    return pd.DataFrame(rows)


def _style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "cgc-mechanism-followup-final",
    })


def _panel_label(ax: plt.Axes, label: str) -> None:
    if len(label) != 1 or not label.islower():
        raise ValueError("panel labels must be one lowercase letter")
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.0, fontweight="bold", color="black")


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight",
                metadata={"Software": "CGC reporting-only finalizer"})
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def _errorbar_forest(ax: plt.Axes, frame: pd.DataFrame, labels: list[str], colors: list[str]) -> None:
    y = np.arange(len(frame))[::-1]
    x = frame["estimate"].to_numpy(float)
    low = frame["ci_low"].to_numpy(float)
    high = frame["ci_high"].to_numpy(float)
    xerr = np.vstack([x - low, high - x])
    for i in range(len(frame)):
        ax.errorbar(x[i], y[i], xerr=xerr[:, i:i + 1], fmt="o", markersize=3.2,
                    color=colors[i], ecolor=colors[i], elinewidth=0.8, capsize=1.7)
    ax.axvline(0, color="#333333", lw=0.65, ls="--", zorder=0)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color="#ECECEC", lw=0.55)


def bio2_source_data(bio: pd.DataFrame) -> pd.DataFrame:
    pooled = bio.query("representation == 'Gene' and direction == 'pooled' and level in @LEVEL_ORDER").copy()
    pooled["figure_panel"] = "a"
    pooled["figure_metric"] = "primary_gene_pooled_hierarchy"
    representation = bio.query("direction == 'pooled' and level in @LEVEL_ORDER").copy()
    representation["figure_panel"] = "b"
    representation["figure_metric"] = "representation_confirmation"
    directions = bio.query(
        "representation == 'Gene' and direction in ['plate6_to_plate14','plate14_to_plate6'] and level in @LEVEL_ORDER"
    ).copy()
    directions["figure_panel"] = "c"
    directions["figure_metric"] = "cross_plate_direction"
    return pd.concat([pooled, representation, directions], ignore_index=True, sort=False)


def render_bio2(bio: pd.DataFrame, figure_dir: Path, source_dir: Path) -> None:
    source = bio2_source_data(bio)
    source_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_dir / "CGC_BIO2_FINAL_SOURCE_DATA.csv", index=False)
    _style()
    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=BIO2_WIDTH_RATIOS, height_ratios=(1.0, 1.0))
    axa = fig.add_subplot(outer[:, 0])
    axb = fig.add_subplot(outer[0, 1])
    axc = fig.add_subplot(outer[1, 1])

    a = source[source.figure_panel.eq("a")].set_index("level").loc[list(LEVEL_ORDER)].reset_index()
    af = a.rename(columns={"conditional_effect": "estimate"})
    labels = ["broad action", "MOA", "target family", "exact target"]
    colors = [BLUE, BLUE, BLUE, ORANGE]
    _errorbar_forest(axa, af, labels, colors)
    axa.set_xlabel("conditional cross-plate organization effect")
    axa.set_title("Frozen gene-level hierarchy", loc="left", fontweight="bold")
    axa.text(0.98, 0.98, "highest robust level: none\nexact target: limited power (3 clusters)",
             transform=axa.transAxes, ha="right", va="top", color="#333333",
             bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    _panel_label(axa, "a")

    b = source[source.figure_panel.eq("b")].copy()
    matrix = b.pivot(index="level", columns="representation", values="conditional_effect")
    matrix = matrix.reindex(index=LEVEL_ORDER, columns=["Gene", "PROGENy", "CollecTRI"])
    vmax = float(np.nanmax(np.abs(matrix.to_numpy(float))))
    image = axb.imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axb.set_xticks(np.arange(3), ["gene", "PROGENy", "CollecTRI"], rotation=25, ha="right")
    axb.set_yticks(np.arange(4), ["broad", "MOA", "family", "target"])
    axb.set_title("Pooled confirmation axes", loc="left", fontweight="bold")
    cb = fig.colorbar(image, ax=axb, fraction=0.05, pad=0.03)
    cb.set_label("effect")
    _panel_label(axb, "b")

    c = source[source.figure_panel.eq("c")].copy()
    markers = {"plate6_to_plate14": "o", "plate14_to_plate6": "s"}
    colors_d = {"plate6_to_plate14": TEAL, "plate14_to_plate6": RED}
    x = np.arange(len(LEVEL_ORDER))
    for direction in markers:
        local = c[c.direction.eq(direction)].set_index("level").reindex(LEVEL_ORDER)
        axc.plot(x, local.conditional_effect, marker=markers[direction], ms=3.2, lw=0.8,
                 color=colors_d[direction], label=direction.replace("plate", "P").replace("_to_", "→"))
    axc.axhline(0, color="#333333", lw=0.65, ls="--")
    axc.set_xticks(x, ["broad", "MOA", "family", "target"], rotation=25, ha="right")
    axc.set_ylabel("directional effect")
    axc.set_title("Cross-plate direction", loc="left", fontweight="bold")
    axc.legend(ncol=1, loc="best")
    _panel_label(axc, "c")
    _save(fig, figure_dir / "CGC_BIO2_FINAL")


def _aligned_vs_rna(frame: pd.DataFrame) -> pd.DataFrame:
    models = {"M4_RNA_ALIGNED", "M_ORACLE_RNA_DEPENDENCY", "M5_RNA_WIRING_ACTIVITY"}
    return frame[
        frame.response_target.eq("PRIMARY_RESIDUAL")
        & frame.model.isin(models)
        & frame.comparator.eq("M0_RNA")
    ].copy()


def multi2_source_data(
    incremental: pd.DataFrame, nulls: pd.DataFrame, power: pd.DataFrame
) -> pd.DataFrame:
    aligned = _aligned_vs_rna(incremental)
    aligned["figure_panel"] = "a"
    aligned["figure_metric"] = "aligned_minus_rna"
    ordinary = incremental[
        incremental.response_target.eq("PRIMARY_RESIDUAL")
        & (
            incremental.comparator.eq("M3_RNA_GENERIC")
            | (incremental.model.eq("M3_RNA_GENERIC") & incremental.comparator.eq("M0_RNA"))
        )
    ].copy()
    ordinary["figure_panel"] = "b"
    ordinary["figure_metric"] = np.where(
        ordinary.comparator.eq("M3_RNA_GENERIC"), "aligned_minus_generic", "generic_minus_rna"
    )
    p = power.copy()
    p["figure_panel"] = "c"
    p["figure_metric"] = "fold_local_power"
    n = nulls.copy()
    n["figure_panel"] = "d"
    n["figure_metric"] = "feature_null_contrast"
    return pd.concat([aligned, ordinary, p, n], ignore_index=True, sort=False)


def feature_null_heatmap_data(
    nulls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return p-value and explicit non-executable masks for the frozen null grid."""

    null_order = [
        "DRUG_TARGET_MAPPING_PERMUTATION", "PATHWAY_MAPPING_PERMUTATION",
        "CONTEXT_MODALITY_WITHIN_LINEAGE", "RANDOM_GENE_SET_MATCHED",
    ]
    row_order = [
        f"{panel} · {est.replace('_ridge', '')}"
        for panel in PANEL_ORDER for est in ESTIMATOR_ORDER
    ]
    complete = nulls[nulls.status.eq("COMPLETE_ACTUAL_FEATURE_NULL_OOF_REFIT")].copy()
    complete["row"] = (
        complete.panel.astype(str) + " · "
        + complete.estimator.astype(str).str.replace("_ridge", "", regex=False)
    )
    values = complete.pivot(index="row", columns="null_type", values="null_p").reindex(
        index=row_order, columns=null_order
    )
    fail_closed = {
        (str(row.panel), str(row.null_type))
        for row in nulls[~nulls.status.eq("COMPLETE_ACTUAL_FEATURE_NULL_OOF_REFIT")].itertuples()
    }
    non_executable = pd.DataFrame(False, index=row_order, columns=null_order)
    for row_label in row_order:
        panel = row_label.split(" · ", 1)[0]
        for null_type in null_order:
            if (panel, null_type) in fail_closed:
                non_executable.loc[row_label, null_type] = True
    unexplained = values.isna() & ~non_executable
    if unexplained.any().any():
        cells = [f"{row}|{column}" for row, column in zip(*np.where(unexplained.to_numpy()))]
        raise RuntimeError(f"unexplained missing feature-null heatmap cells: {cells}")
    if int(non_executable.to_numpy().sum()) != 2 * len(fail_closed):
        raise RuntimeError("fail-closed feature-null families did not expand to both estimators")
    return values, non_executable


def render_multi2(
    incremental: pd.DataFrame,
    nulls: pd.DataFrame,
    power: pd.DataFrame,
    figure_dir: Path,
    source_dir: Path,
) -> None:
    source = multi2_source_data(incremental, nulls, power)
    source_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_dir / "CGC_MULTI2_FINAL_SOURCE_DATA.csv", index=False)
    _style()
    fig = plt.figure(figsize=(7.5, 6.4), constrained_layout=True)
    outer = fig.add_gridspec(3, 2, width_ratios=MULTI2_WIDTH_RATIOS,
                            height_ratios=(1.05, 1.0, 1.0))
    axa = fig.add_subplot(outer[:, 0])
    axb = fig.add_subplot(outer[0, 1])
    axc = fig.add_subplot(outer[1, 1])
    axd = fig.add_subplot(outer[2, 1])

    a = source[source.figure_panel.eq("a")].copy()
    a["panel_rank"] = a.panel.map({name: i for i, name in enumerate(PANEL_ORDER)})
    a["estimator_rank"] = a.estimator.map({name: i for i, name in enumerate(ESTIMATOR_ORDER)})
    a = a.sort_values(["panel_rank", "estimator_rank"], kind="stable")
    af = a.copy()
    af["estimate"] = af["delta_g"]
    labels = [f"{row.panel.replace('_', ' ').lower()} / {'bilinear' if row.estimator == 'bilinear_ridge' else 'ridge'}"
              for row in a.itertuples()]
    colors = [TEAL if estimator == "bilinear_ridge" else BLUE for estimator in a.estimator]
    _errorbar_forest(axa, af, labels, colors)
    axa.set_xlabel("incremental recovery, aligned model − RNA")
    axa.set_title("Primary residual: no aligned model beats RNA", loc="left", fontweight="bold")
    axa.text(0.02, 0.02, "0/12 lower confidence bounds > 0", transform=axa.transAxes,
             ha="left", va="bottom", color="#333333")
    _panel_label(axa, "a")

    b = source[source.figure_panel.eq("b")].copy()
    b["panel_rank"] = b.panel.map({name: i for i, name in enumerate(PANEL_ORDER)})
    b["estimator_rank"] = b.estimator.map({name: i for i, name in enumerate(ESTIMATOR_ORDER)})
    b = b.sort_values(["panel_rank", "estimator_rank", "figure_metric"], kind="stable")
    paired_keys = list(b[["panel", "estimator"]].drop_duplicates().itertuples(index=False, name=None))
    y = np.arange(len(paired_keys))[::-1]
    ylabels: list[str] = []
    for yi, (panel, estimator) in zip(y, paired_keys, strict=True):
        pair = b[b.panel.eq(panel) & b.estimator.eq(estimator)].set_index("figure_metric")
        x_aligned = float(pair.loc["aligned_minus_generic", "delta_g"])
        x_generic = float(pair.loc["generic_minus_rna", "delta_g"])
        axb.plot([x_generic, x_aligned], [yi, yi], color=LIGHT_GREY, lw=0.65, zorder=0)
        axb.scatter(x_aligned, yi, s=16, color=TEAL, marker="o", zorder=2)
        axb.scatter(x_generic, yi, s=16, color=RED, marker="s", zorder=2)
        ylabels.append(
            f"{panel.replace('_', ' ').lower()} / "
            f"{'bilinear' if estimator == 'bilinear_ridge' else 'ridge'}"
        )
    axb.axvline(0, color="#333333", lw=0.65, ls="--")
    axb.set_xlabel("incremental recovery")
    axb.set_yticks(y, ylabels, fontsize=5.2)
    axb.set_title("Alignment avoids generic degradation", loc="left", fontweight="bold")
    axb.scatter([], [], s=16, color=TEAL, marker="o", label="aligned − generic")
    axb.scatter([], [], s=16, color=RED, marker="s", label="generic − RNA")
    axb.legend(loc="lower right")
    _panel_label(axb, "b")

    c = source[source.figure_panel.eq("c")].copy()
    palette = dict(zip(PANEL_ORDER, [BLUE, TEAL, ORANGE, RED, GREY, "#7A5195"], strict=True))
    for panel in PANEL_ORDER:
        local = c[c.panel.eq(panel)].sort_values("target_delta_g")
        axc.plot(local.target_delta_g, local.power, marker="o", ms=2.6, lw=0.8,
                 color=palette[panel], label=panel.replace("_", " ").lower())
    axc.axhline(0.8, color="#333333", lw=0.55, ls="--")
    axc.set(xlabel="injected Δg", ylabel="detection power", ylim=(-0.03, 1.04))
    axc.set_title("Fold-local power curves", loc="left", fontweight="bold")
    axc.legend(ncol=2, loc="upper left", handlelength=1.3, columnspacing=0.7)
    _panel_label(axc, "c")

    d = source[source.figure_panel.eq("d")].copy()
    matrix, non_executable = feature_null_heatmap_data(d)
    row_order = matrix.index.tolist()
    cmap = matplotlib.colormaps["Blues"].with_extremes(bad="#C8C8C8")
    plotted = -np.log10(matrix.to_numpy(float))
    plotted[non_executable.to_numpy()] = np.nan
    image = axd.imshow(np.ma.masked_invalid(plotted), cmap=cmap, vmin=0, vmax=2,
                       aspect="auto")
    axd.set_xticks(np.arange(4), ["target", "pathway", "context", "random"], rotation=25, ha="right")
    axd.set_yticks(
        np.arange(len(row_order)),
        [label.lower().replace("_", " ") for label in row_order],
        fontsize=4.8,
    )
    axd.set_title("One-refit bootstrap-null tails", loc="left", fontweight="bold")
    for row_index, column_index in zip(*np.where(non_executable.to_numpy())):
        axd.text(column_index, row_index, "NE", ha="center", va="center",
                 fontsize=4.4, color="#4D4D4D", fontweight="bold")
    cb = fig.colorbar(image, ax=axd, fraction=0.05, pad=0.03)
    cb.set_label("−log10 p")
    _panel_label(axd, "d")
    _save(fig, figure_dir / "CGC_MULTI2_FINAL")


def bridge_source_data(joined: pd.DataFrame, null_draws: pd.DataFrame) -> pd.DataFrame:
    points = joined[
        np.isfinite(pd.to_numeric(joined.organization_score, errors="coerce"))
        & np.isfinite(pd.to_numeric(joined.delta_g_aligned_minus_rna, errors="coerce"))
    ].copy()
    points["figure_panel"] = "a"
    points["figure_metric"] = "intervention_pair"
    draws = null_draws.copy()
    draws["figure_panel"] = "b"
    draws["figure_metric"] = "structure_preserving_null_rho"
    return pd.concat([points, draws], ignore_index=True, sort=False)


def render_bridge(
    joined: pd.DataFrame,
    null_draws: pd.DataFrame,
    result: dict[str, object],
    figure_dir: Path,
    source_dir: Path,
) -> None:
    source = bridge_source_data(joined, null_draws)
    source_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_dir / "CGC_BRIDGE_FINAL_SOURCE_DATA.csv", index=False)
    _style()
    fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=BRIDGE_WIDTH_RATIOS)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])

    points = source[source.figure_panel.eq("a")]
    axa.scatter(points.organization_score, points.delta_g_aligned_minus_rna,
                s=18, facecolor=BLUE, edgecolor="white", linewidth=0.35, alpha=0.9)
    axa.axhline(0, color="#333333", lw=0.55, ls="--")
    axa.axvline(0, color="#333333", lw=0.55, ls="--")
    axa.set(xlabel="BIO2 intervention organization score",
            ylabel="MULTI2 M5 aligned − RNA recovery")
    axa.set_title("No preferential rescue of organized residuals", loc="left", fontweight="bold")
    axa.text(0.03, 0.04,
             f"Spearman ρ = {float(result['spearman_rho']):.3f}\n"
             f"two-sided p = {float(result['blocked_permutation_p_two_sided']):.3f}",
             transform=axa.transAxes, ha="left", va="bottom")
    _panel_label(axa, "a")

    draws = source[source.figure_panel.eq("b")].rho.to_numpy(float)
    axb.hist(draws[np.isfinite(draws)], bins=28, color=LIGHT_GREY, edgecolor="white", linewidth=0.35)
    axb.axvline(float(result["spearman_rho"]), color=RED, lw=1.2)
    axb.set(xlabel="permuted Spearman ρ", ylabel="draws")
    axb.set_title("Frozen structure-preserving null", loc="left", fontweight="bold")
    _panel_label(axb, "b")
    _save(fig, figure_dir / "CGC_BRIDGE_FINAL")


def _git_last_commit(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(relative)],
            cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "UNAVAILABLE"


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _manifest_path(path: Path) -> str:
    """Serialize repository files relatively and external outputs absolutely."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.6f}"


def _final_report(
    bio: pd.DataFrame,
    incremental: pd.DataFrame,
    nulls: pd.DataFrame,
    limits: pd.DataFrame,
    bridge: dict[str, object],
    groups: pd.DataFrame,
    input_paths: dict[str, Path],
) -> str:
    gene = bio.query("representation == 'Gene'").copy()
    hierarchy_lines = []
    for level in (*LEVEL_ORDER, "GENOTYPE_DRUG", "GENOTYPE_DRUG_DIRECT_SENSITIVITY"):
        expected_direction = "pooled" if level in LEVEL_ORDER else "residual_cross_plate"
        local = gene[gene.level.eq(level) & gene.direction.eq(expected_direction)]
        if local.empty:
            continue
        row = local.iloc[0]
        hierarchy_lines.append(
            f"| {level} | {_fmt(float(row.conditional_effect))} | "
            f"[{_fmt(float(row.ci_low))}, {_fmt(float(row.ci_high))}] | "
            f"{int(row.independent_clusters)} | {row.decision_status} |"
        )

    aligned = _aligned_vs_rna(incremental)
    aligned_positive = int((aligned.ci_low > 0).sum())
    ordinary_aligned = incremental[
        incremental.response_target.eq("PRIMARY_RESIDUAL")
        & incremental.comparator.eq("M3_RNA_GENERIC")
    ]
    ordinary_generic = incremental[
        incremental.response_target.eq("PRIMARY_RESIDUAL")
        & incremental.model.eq("M3_RNA_GENERIC")
        & incremental.comparator.eq("M0_RNA")
    ]
    aligned_over_generic = int((ordinary_aligned.ci_low > 0).sum())
    generic_degraded = int((ordinary_generic.ci_high < 0).sum())
    complete_nulls = nulls[nulls.status.eq("COMPLETE_ACTUAL_FEATURE_NULL_OOF_REFIT")]
    null_positive = complete_nulls[complete_nulls.contrast_ci_low > 0]
    null_note = "None."
    if len(null_positive):
        row = null_positive.iloc[0]
        null_note = (
            f"{len(null_positive)}/{len(complete_nulls)} contrast intervals were above zero: "
            f"{row.panel} / {row.null_type} / {row.estimator}, contrast "
            f"{float(row.contrast_delta_g):.6f}. Its observed aligned-minus-RNA gain was "
            f"{float(row.observed_delta_g):.6f}, so it does not establish residual recovery."
        )
    mde_text = ", ".join(f"{row.panel}={row.mde80}" for row in limits.itertuples())
    heterogeneity_estimable = int(groups["heterogeneity_estimable"].sum())
    singleton_groups = int((~groups["heterogeneity_estimable"]).sum())
    provenance_lines = [
        f"| `{name}` | `{sha256(path)}` | `{_git_last_commit(path)}` |"
        for name, path in input_paths.items()
    ]
    return f"""# CGC mechanism follow-up — frozen final report

This is a deterministic reporting-only integration of frozen BIO-2, MULTI-2,
power, feature-null, and bridge artifacts. No model, null feature map, power
calibration, or bridge analysis was rerun.

## Integrated conclusion

BIO-2 identifies no robust biological attribution rung in the frozen hierarchy.
The exact-target rung remains `LIMITED_POWER`, not evidence of absence. MULTI-2
shows that mechanism alignment often avoids the severe degradation produced by
generic context features, but no aligned model establishes positive primary-
residual recovery over RNA. The available aligned modalities therefore do not
confirm a context-specific residual remedy. The frozen secondary bridge is null
and does not support preferential rescue of interventions with stronger BIO-2
organization.

## BIO-2

`HIGHEST_ROBUST_ATTRIBUTION_LEVEL=NONE`

| level | pooled gene effect | 95% cluster interval | clusters | status |
|---|---:|---:|---:|---|
{chr(10).join(hierarchy_lines)}

The exact-target design has three independent clusters and is explicitly capped
at `LIMITED_POWER`. Directional and matched confirmation axes do not upgrade any
level to robust attribution.

## MULTI-2

- Primary aligned-versus-RNA contrasts with lower confidence bound above zero:
  **{aligned_positive}/{len(aligned)}**.
- Ordinary-modality aligned-versus-generic contrasts with lower confidence bound
  above zero: **{aligned_over_generic}/{len(ordinary_aligned)}**.
- Ordinary generic-versus-RNA contrasts with upper confidence bound below zero:
  **{generic_degraded}/{len(ordinary_generic)}**.

Thus, aligned features beat generic features mainly by avoiding generic-feature
degradation; they never beat RNA under the frozen primary recovery criterion.
`MULTI2_RESIDUAL_RECOVERY_NOT_CONFIRMED`.

### Feature-null semantics

Each executable correspondence-null row used exactly one frozen feature-null OOF
refit. `null_p` is the {NULL_P_SEMANTICS}. It is not a permutation frequency over
many model refits. There are {len(complete_nulls)} executable estimator contrasts
and {len(nulls) - len(complete_nulls)} explicit fail-closed rows. {null_note}

### Detection limits

Frozen MDE80 values: {mde_text}. These are resolution limits. A null real-data
result supports absence only at or above its panel-specific MDE80; smaller effects
remain unresolved.

### Preregistered biological heterogeneity summaries

`MULTI2_GROUP_SUMMARIES.csv` contains {len(groups)} rows across all frozen result
strata and all four permitted annotation axes: `moa_broad`, `moa_fine`, target
family, and exact target. The {heterogeneity_estimable} groups containing at least
two interventions include the preregistered inverse-variance random-effects
summary (DerSimonian-Laird tau-squared). The {singleton_groups} singleton groups
retain their frozen single-intervention effect, standard error, and interval in
explicit `single_intervention_*` fields; their random-effects and heterogeneity
fields are NA with status `NOT_ESTIMABLE_SINGLE_INTERVENTION`. All groups also
include n, unweighted descriptive distribution summaries, positive fraction, and
CI-positive fraction. These exhaustive, unselected group summaries receive no
multiplicity-corrected groupwise inference and do not modify the global conclusion.

## Frozen BIO-2 × MULTI-2 bridge

- Complete intervention pairs: **{int(bridge['n_complete_interventions'])}**
- Spearman rho: **{float(bridge['spearman_rho']):.6f}**
- Structure-preserving two-sided p: **{float(bridge['blocked_permutation_p_two_sided']):.6f}**
- Positive-direction p: **{float(bridge['blocked_permutation_p_positive']):.6f}**

This secondary null result means that interventions with stronger residual
organization were not preferentially rescued by the available aligned modalities.
It does not erase either primary analysis.

## Frozen-input reconciliation and provenance

| frozen input | SHA-256 | last source commit |
|---|---|---|
{chr(10).join(provenance_lines)}

`CGC_MECHANISM_NUMERICAL_RECONCILIATION.csv` gives row-level traceability for all
BIO-2 hierarchy estimates, MULTI-2 aggregate contrasts, power points, detection
limits, executable and fail-closed null rows, and bridge statistics.
`CGC_MECHANISM_REPORTING_MANIFEST.json` freezes the reporting-code commit and
SHA-256 hashes for all reporting inputs and outputs.

## Figure policy

All three final figures are quantitative-only, use an NBT-style restrained visual
system, lowercase panel labels, editable SVG plus 450-dpi PNG, unequal panel sizes,
and exact CSV source data. No figure contains a conceptual mechanism cartoon or an
unfrozen statistic.
"""


def run_reporting(input_dir: Path = OUT, output_dir: Path | None = None) -> dict[str, object]:
    """Create reporting outputs while proving frozen inputs remain byte-identical."""

    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else Path(output_dir)
    input_paths = {name: input_dir / name for name in REQUIRED_INPUTS}
    missing = [name for name, path in input_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen reporting inputs: {missing}")
    hashes_before = {name: sha256(path) for name, path in input_paths.items()}

    bio = pd.read_csv(input_paths["BIO2_HIERARCHY_RESULTS.csv"])
    incremental = pd.read_csv(input_paths["MULTI2_INCREMENTAL_RECOVERY.csv"])
    gains = pd.read_csv(input_paths["MULTI2_PER_INTERVENTION_GAINS.csv"], keep_default_na=False)
    nulls = pd.read_csv(input_paths["MULTI2_ALIGNMENT_NULLS.csv"])
    power = pd.read_csv(input_paths["MULTI2_POWER_CURVES.csv"])
    limits = pd.read_csv(input_paths["MULTI2_DETECTION_LIMITS.csv"], dtype={"mde80": str})
    bridge_joined = pd.read_csv(input_paths["BIO2_MULTI2_BRIDGE.csv"])
    bridge = json.loads(input_paths["BIO2_MULTI2_BRIDGE_RESULT.json"].read_text(encoding="utf-8"))
    bridge_null = pd.read_parquet(input_paths["bio2_multi2_bridge_null.parquet"])

    if json.loads(input_paths["BIO2_RUN_MANIFEST.json"].read_text())["highest_robust_attribution_level"] != "NONE":
        raise RuntimeError("BIO2 frozen verdict mismatch")
    if not np.isclose(float(bridge["spearman_rho"]), -0.15773775825792447, rtol=0, atol=1e-15):
        raise RuntimeError("frozen bridge rho mismatch")
    if not np.isclose(float(bridge["blocked_permutation_p_two_sided"]), 0.5692430756924307,
                      rtol=0, atol=1e-15):
        raise RuntimeError("frozen bridge p-value mismatch")
    if len(incremental) != 72 or len(gains) != 804 or len(nulls) != 43 or len(limits) != 6:
        raise RuntimeError("frozen MULTI2 aggregate row-count mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = build_group_summaries(gains)
    groups.to_csv(output_dir / "MULTI2_GROUP_SUMMARIES.csv", index=False)
    reconciliation = build_reconciliation(bio, incremental, nulls, power, limits, bridge)
    reconciliation.to_csv(output_dir / "CGC_MECHANISM_NUMERICAL_RECONCILIATION.csv", index=False)

    figure_dir = output_dir / "figures"
    source_dir = output_dir / "figure_source_data"
    render_bio2(bio, figure_dir, source_dir)
    render_multi2(incremental, nulls, power, figure_dir, source_dir)
    render_bridge(bridge_joined, bridge_null, bridge, figure_dir, source_dir)

    report = _final_report(bio, incremental, nulls, limits, bridge, groups, input_paths)
    (output_dir / "CGC_MECHANISM_FOLLOWUP_FINAL.md").write_text(report, encoding="utf-8")

    output_paths = {name: output_dir / name for name in REPORTING_OUTPUTS}
    missing_outputs = [name for name, path in output_paths.items() if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f"missing reporting outputs: {missing_outputs}")
    reporting_manifest = {
        "status": "CGC_MECHANISM_REPORTING_COMPLETE",
        "reporting_code_commit": _git_value("rev-parse", "HEAD"),
        "reporting_branch": _git_value("branch", "--show-current"),
        "scientific_analysis_rerun": False,
        "frozen_inputs": {
            name: {"path": _manifest_path(path), "sha256": hashes_before[name]}
            for name, path in input_paths.items()
        },
        "reporting_outputs": {
            name: {"path": _manifest_path(path), "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
    }
    (output_dir / "CGC_MECHANISM_REPORTING_MANIFEST.json").write_text(
        json.dumps(reporting_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hashes_after = {name: sha256(path) for name, path in input_paths.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("a frozen reporting input changed during finalization")
    return {
        "status": "CGC_MECHANISM_REPORTING_COMPLETE",
        "group_summary_rows": len(groups),
        "reconciliation_rows": len(reconciliation),
        "figures": 3,
        "reporting_manifest": "CGC_MECHANISM_REPORTING_MANIFEST.json",
        "frozen_inputs_unchanged": True,
    }


__all__ = [
    "BIO2_WIDTH_RATIOS", "BRIDGE_WIDTH_RATIOS", "MULTI2_WIDTH_RATIOS",
    "NULL_P_SEMANTICS", "build_group_summaries", "build_reconciliation",
    "feature_null_heatmap_data", "run_reporting",
]
