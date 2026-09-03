from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cgc_resolution_poc_v2"


def figures() -> None:
    comparisons = pd.read_csv(OUT / "RESOLUTION2_PAIRED_COMPARISONS.csv")
    points = comparisons[comparisons.family == "point_estimate"].set_index("contrast").estimate
    budgets = ("m49_k92", "m40_k4")
    colors = {"gene": "#7f8c8d", "pathway": "#2878b5", "random": "#d58b36"}
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(2)
    ax.bar(x - 0.18, [points[f"{b}_g_gene"] for b in budgets], 0.36, label="Full gene", color=colors["gene"])
    ax.bar(x + 0.18, [points[f"{b}_g_pathway"] for b in budgets], 0.36, label="5-pathway pooled", color=colors["pathway"])
    ax.set_xticks(x, ["m=49, k=92", "m=40, k=4"])
    ax.set_ylabel("Replicate-stable recovery g")
    ax.set_title("RES2-A — Same prediction, different output resolution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "RES2_A_GENE_VS_PATHWAY.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - 0.18, [points[f"{b}_g_pathway"] for b in budgets], 0.36, label="Canonical pathways", color=colors["pathway"])
    ax.bar(x + 0.18, [points[f"{b}_g_matched_random_median"] for b in budgets], 0.36, label="Nearest reliability-control median", color=colors["random"])
    ax.set_xticks(x, ["m=49, k=92", "m=40, k=4"])
    ax.set_ylabel("Replicate-stable recovery g")
    ax.set_title("RES2-B — Biological coordinates versus random projections")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "RES2_B_REAL_VS_RANDOM.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.08, 0.5, "RNA + intervention", ha="center", va="center", fontsize=15,
            bbox=dict(boxstyle="round,pad=0.5", fc="#e8f1f8", ec="#2878b5"))
    ax.annotate("", xy=(0.30, 0.5), xytext=(0.18, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    for x0, label, sub, color in (
        (0.42, "Pathways", "tested: partial", "#2878b5"),
        (0.65, "Programs", "not executable", "#8e6c8a"),
        (0.87, "Identifiable genes", "future empirical gate", "#5b8c5a"),
    ):
        ax.text(x0, 0.5, f"{label}\n{sub}", ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=color, lw=2))
    ax.text(0.5, 0.88, "Match counterfactual resolution to identifiable biological information",
            ha="center", va="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "RES2_C_RESOLUTION_CONCEPT.png", dpi=180)
    plt.close(fig)


def report() -> None:
    comparisons = pd.read_csv(OUT / "RESOLUTION2_PAIRED_COMPARISONS.csv")
    matching = pd.read_csv(OUT / "RESOLUTION2_RELIABILITY_MATCHING.csv")
    primary = comparisons[comparisons.family == "PATHWAY_PRIMARY_2BUDGET_X_2CONTRAST"].set_index("contrast")
    points = comparisons[comparisons.family == "point_estimate"].set_index("contrast").estimate
    match_summary = matching.groupby(["m", "k", "pathway"]).agg(
        exact_pool=("exact_pool_size", "first"),
        selected_exact=("exact_match", "sum"),
        median_relative_variance_error=("relative_variance_error", "median"),
        median_fisher_reliability_error=("fisher_reliability_error", "median"),
    ).reset_index()
    lines = [
        "# CGC-RESOLUTION-2 final report",
        "",
        "## Git provenance",
        "",
        "- Branch: `codex/cgc_counterfactual_resolution_v2`",
        "- Parent: `e47b76bd267545c45c19d4a08dd6dfc787d306e7`",
        "- Protocol freeze: `dfffb8ab0121b8546d437da64c3fa1e14454c44b`",
        "- Replay integrity result: `adecedc20a7a770cce8f205769cb513ed2623ddf`",
        "- Pathway result freeze: `92844ad`",
        "- Paired inference freeze: `16b48ee`",
        "",
        "## Frozen replay and scope",
        "",
        "The deterministic M2 replay passed before biological-coordinate outcomes were inspected. The reproduced gene-level recoveries were exactly `0.11068053088808294` at `m=49,k=92` and within `2.22e-16` of `0.07718711664021238` at `m=40,k=4`; the `1e-10` gate, repeat determinism, split/provenance, and adversarial hidden-target leakage checks all passed. No estimator, hyperparameter, split, budget, or prediction was changed.",
        "",
        "## Pathway result",
        "",
        f"At the primary budget, pooled five-pathway recovery was `{points['m49_k92_g_pathway']:.9f}` versus full-gene `{points['m49_k92_g_gene']:.9f}` and matched-random median `{points['m49_k92_g_matched_random_median']:.9f}`. At the secondary budget the corresponding values were `{points['m40_k4_g_pathway']:.9f}`, `{points['m40_k4_g_gene']:.9f}`, and `{points['m40_k4_g_matched_random_median']:.9f}`.",
        "",
        "The frozen 10,000-draw paired hierarchical bootstrap with the four-member two-budget max-T family gave:",
        "",
        "| contrast | estimate | simultaneous 95% CI | passes >0 |",
        "|---|---:|---:|:---:|",
    ]
    for name, row in primary.iterrows():
        lines.append(f"| `{name}` | {row.estimate:.6f} | [{row.simultaneous_lower_95:.6f}, {row.simultaneous_upper_95:.6f}] | {bool(row.corrected_positive)} |")
    lines += [
        "",
        "Thus canonical pathway projection improves recovery over the full transcriptome at both budgets, but the primary pathway-minus-random simultaneous interval crosses zero. The preregistered pathway hard gate is not met.",
        "",
        "## Reliability-matching audit",
        "",
        "None of the 5,000 structured candidates per pathway met both exact thresholds at either budget. The frozen fallback therefore retained the nearest 500 without relaxing thresholds. This is a material limitation: the controls preserve loading structure and training-side gene bins, but their realized truth variance/reliability remains imperfectly matched.",
        "",
        "| budget | pathway | exact pool | median relative variance error | median Fisher-z reliability error |",
        "|---|---|---:|---:|---:|",
    ]
    for row in match_summary.itertuples(index=False):
        lines.append(f"| m={row.m},k={row.k} | {row.pathway} | {row.exact_pool} | {row.median_relative_variance_error:.3f} | {row.median_fisher_reliability_error:.3f} |")
    lines += [
        "",
        "## Program arm feasibility",
        "",
        "The program arm was not approximated. Its frozen literal definition requires 41,850 episode-local rank-8 PCA fits (K=4 is the prefix), each excluding the two hidden target rows, plus 5,000 independent 25,695×K Gaussian-QR candidates per episode and K. That is 418,500,000 candidate subspaces and 64,514,598,750,000 Gaussian values before matching. No mathematically exact, tested shortcut preserving episode-local exclusion and the Haar/null specification was available. Global PCA, grouped bases, fewer candidates, and approximate random projections were forbidden.",
        "",
        "`PROGRAM_RESOLUTION_POC_NOT_EXECUTABLE`",
        "",
        "## Verdict",
        "",
        "`PATHWAY_RESOLUTION_POC_NOT_SUPPORTED`",
        "",
        "`COUNTERFACTUAL_RESOLUTION_PRINCIPLE_PARTIALLY_SUPPORTED`",
        "",
        "The supported statement is narrow: under the exact same frozen RNA-only predictions, externally defined pathway coordinates show more replicate-stable recovery than the full 25,695-gene vector. Correspondence-specific biological superiority over the preregistered random-projection control did not survive joint correction, so this is not evidence that pathway-level Virtual Cells solve CGC. The already frozen Lea PC1-4 observation remains orthogonal qualitative support and was not numerically combined.",
        "",
        "The experiment stops here. No new pathway, ontology, K, estimator, architecture, or identifiable-gene head was added.",
    ]
    (OUT / "COUNTERFACTUAL_RESOLUTION_POC_V2_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    figures()
    report()
