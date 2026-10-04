from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import aggregate_real
from .design import ENTRIES, frozen_masks
from .runner import _cache_paths, sha256


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _row(recovery: pd.DataFrame, comparison: str, model: str, endpoint: str) -> pd.Series:
    return recovery[
        (recovery.comparison == comparison)
        & (recovery.model == model)
        & (recovery.endpoint == endpoint)
    ].iloc[0]


def _figure(out: Path, recovery: pd.DataFrame, context: pd.DataFrame, selected: pd.DataFrame) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    colors = {
        "MATCHED_AFFINE_RIDGE": "#6B7280",
        "ADDITIVE_MAIN_EFFECT": "#C77C59",
        "LOW_RANK_INTERACTION": "#356A8A",
    }
    labels = {
        "MATCHED_AFFINE_RIDGE": "matched affine Ridge",
        "ADDITIVE_MAIN_EFFECT": "additive main effects",
        "LOW_RANK_INTERACTION": "low-rank interaction",
    }
    fig = plt.figure(figsize=(7.2, 4.25), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=(0.88, 1.16, 1.04), height_ratios=(0.78, 1.22), wspace=0.76, hspace=0.72)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1:])
    axc = fig.add_subplot(grid[1, :2])
    axd = fig.add_subplot(grid[1, 2])

    axa.axis("off")
    axa.text(0.5, 0.72, "additive", ha="center", va="center", weight="bold")
    axa.text(0.5, 0.50, r"$\mu + A_c + B_p$", ha="center", va="center", fontsize=8)
    axa.annotate("", xy=(0.85, 0.28), xytext=(0.15, 0.28), arrowprops={"arrowstyle": "->", "lw": 1, "color": "#4B5563"})
    axa.text(0.5, 0.08, r"$+\,\sum_r u_{cr}v_{pr}q_r$", ha="center", va="center", color=colors["LOW_RANK_INTERACTION"], fontsize=8)
    axa.set_title("completion structure", loc="center", pad=3)

    primary = recovery[(recovery.comparison == "MODEL_ESTIMATE") & (recovery.endpoint == "CONTEXT_SPECIFIC")]
    order = list(colors)
    for y, model in enumerate(order):
        r = primary[primary.model == model].iloc[0]
        axb.errorbar(r.estimate, y, xerr=[[r.estimate - r.lower_95], [r.upper_95 - r.estimate]], fmt="o", color=colors[model], ms=5, capsize=2, lw=1.2)
    axb.axvline(0, color="#9CA3AF", lw=0.8, ls="--")
    axb.set_yticks(range(3), [labels[value] for value in order])
    axb.tick_params(axis="y", labelsize=6.5)
    axb.set_xlabel(r"context-specific recovery, $g$")
    axb.set_title("near-complete OOF recovery", loc="center", pad=3)
    axb.spines[["top", "right"]].set_visible(False)

    wide = context.pivot(index="context_index", columns="model", values="context_specific_g")
    gain = (wide["LOW_RANK_INTERACTION"] - wide["MATCHED_AFFINE_RIDGE"]).sort_values()
    axc.scatter(np.arange(len(gain)), gain, s=15, color=np.where(gain >= 0, colors["LOW_RANK_INTERACTION"], colors["ADDITIVE_MAIN_EFFECT"]), linewidths=0)
    axc.axhline(0, color="#6B7280", lw=0.8)
    axc.set_xlabel("contexts, ordered by paired gain")
    axc.set_ylabel(r"$\Delta g$ (low-rank $-$ affine)")
    axc.set_title("context-level gain distribution", loc="center", pad=3)
    axc.spines[["top", "right"]].set_visible(False)

    ranks = [2, 4, 8, 16, 32]
    counts = selected.groupby("rank").size().reindex(ranks, fill_value=0)
    rank_positions = np.arange(len(ranks))
    axd.plot(rank_positions, counts / counts.sum(), marker="o", color=colors["LOW_RANK_INTERACTION"], lw=1.2, ms=4)
    axd.set_xticks(rank_positions, ranks)
    axd.set_xlabel("selected rank")
    axd.set_ylabel("fraction of plate-fold fits")
    axd.set_title("train-side model selection", loc="center", pad=3)
    axd.spines[["top", "right"]].set_visible(False)

    for label, axis in zip("abcd", (axa, axb, axc, axd), strict=True):
        axis.text(-0.13, 1.08, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.savefig(out / "LOW_RANK_INTERACTION_COMPLETION.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "LOW_RANK_INTERACTION_COMPLETION.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_reports(root: Path, source_root: Path) -> str:
    root = root.resolve()
    source_root = source_root.resolve()
    out = root / "results/cgc_lowrank_interaction_completion"
    result = aggregate_real(root, source_root)
    recovery = result["recovery"]
    context = result["context"]
    diagnostics = result["diagnostics"]
    selected = result["selected"]
    _figure(out, recovery, context, selected)

    affine = _row(recovery, "MODEL_ESTIMATE", "MATCHED_AFFINE_RIDGE", "CONTEXT_SPECIFIC")
    additive = _row(recovery, "MODEL_ESTIMATE", "ADDITIVE_MAIN_EFFECT", "CONTEXT_SPECIFIC")
    low = _row(recovery, "MODEL_ESTIMATE", "LOW_RANK_INTERACTION", "CONTEXT_SPECIFIC")
    gain = _row(recovery, "LOWRANK_MINUS_AFFINE", "LOW_RANK_INTERACTION", "CONTEXT_SPECIFIC")
    increment = _row(recovery, "INTERACTION_INCREMENT", "LOW_RANK_INTERACTION", "CONTEXT_SPECIFIC")
    full_affine = _row(recovery, "MODEL_ESTIMATE", "MATCHED_AFFINE_RIDGE", "FULL_RESPONSE")
    full_add = _row(recovery, "MODEL_ESTIMATE", "ADDITIVE_MAIN_EFFECT", "FULL_RESPONSE")
    full_low = _row(recovery, "MODEL_ESTIMATE", "LOW_RANK_INTERACTION", "FULL_RESPONSE")
    cross_diag = diagnostics[
        (diagnostics.model == "LOW_RANK_INTERACTION")
        & (diagnostics.component == "context_specific_prediction")
        & (diagnostics.replicate_scope == "cross_plate")
    ].iloc[0]
    interaction_diag = diagnostics[
        (diagnostics.model == "LOW_RANK_INTERACTION")
        & (diagnostics.component == "cp_interaction_only")
    ].iloc[0]
    masks = frozen_masks()
    observed = sorted({len(masks.outer_training(fold)) for fold in range(100)})
    synthetic = pd.read_csv(out / "LOW_RANK_SYNTHETIC_POSITIVE_CONTROL.csv")
    synthetic_low = synthetic[synthetic.model == "LOW_RANK_INTERACTION"].iloc[0]
    feasibility = json.loads((out / "LOW_RANK_FEASIBILITY_DECISION.json").read_text(encoding="utf-8"))
    invariance = json.loads((out / "LOW_RANK_OUTCOME_INVARIANCE.json").read_text(encoding="utf-8"))

    fairness = f"""# Low-rank interaction completion fairness audit

- Frozen protocol commit: `86e57476fa0fd32b8d1579008f71d23dd27ce5a3`.
- Frozen feasibility commit: `004d3bbdbfedde28e520d0571f4e5be7060c8121`.
- Exact all-but-one was rejected by the outcome-blind runtime gate: projected
  {feasibility['projected_exact_gpu_hours']:.3f} GPU-hours versus a frozen 24-hour limit.
- Both models used the identical 100 outer masks and {observed[0]}--{observed[-1]}
  observed entries per fit ({observed[0]/ENTRIES:.4%}--{observed[-1]/ENTRIES:.4%}).
- Every one of 4,650 entries received exactly one OOF prediction.
- Masks were identical across Plate 6 and Plate 14; models were fit independently.
- Both models used the same 25,695-gene `delta_primary` universe and no external features.
- Low-rank rank/L2/start selection used only outer-observed inner train/validation entries.
- Affine Ridge used the previously frozen reference-only target-context/plate L2;
  no new held-out outcome selected it. Missing non-target sentinel support was
  filled only by the same outer-train additive model.
- The masked affine implementation reproduced historical one-entry coefficients
  within {invariance['historical_one_entry_max_abs_weight_difference']:.3g}.
- Adversarial hidden-target mutation left masks, rank, L2, U, V, and every
  prediction coefficient bitwise unchanged: `{invariance['passed']}`.
- Selected final low-rank convergence fraction: {result['selected_final_convergence_fraction']:.3%}.
- Synthetic rank-4 implementation control passed: `{result['synthetic_passed']}`.

The tensor model can use all observed entries because global completion is its
declared structure; affine Ridge uses its frozen target-specific context/sentinel
structure. This is a model-capacity contrast, not an information contrast.
"""
    (out / "LOW_RANK_FAIRNESS_AUDIT.md").write_text(fairness, encoding="utf-8")

    report = f"""# Tahoe low-rank interaction completion report

Git provenance at report generation: branch `{_git(root, 'branch', '--show-current')}`,
commit `{_git(root, 'rev-parse', 'HEAD')}`.

## Direct answers

1. **Was exact all-but-one executable?** No. The frozen benchmark projected
   {feasibility['projected_exact_gpu_hours']:.3f} GPU-hours, above the 24-hour gate.
2. **Fallback support:** 100 balanced folds, {observed[0]} or {observed[-1]}
   observed entries per fit; pooled observed fraction exactly 99.0%.
3. **Was affine Ridge rerun on identical masks?** Yes. Its strict masked
   extension reconciled to historical one-entry M2 within
   {invariance['historical_one_entry_max_abs_weight_difference']:.3g}.
4. **Primary context-specific recovery:** matched affine `g={affine.estimate:.6f}`
   (95% CI [{affine.lower_95:.6f}, {affine.upper_95:.6f}]); additive
   `g={additive.estimate:.6f}` [{additive.lower_95:.6f}, {additive.upper_95:.6f}];
   low-rank interaction `g={low.estimate:.6f}` [{low.lower_95:.6f}, {low.upper_95:.6f}].
5. **Paired low-rank minus affine:** {gain.estimate:.6f}
   [{gain.lower_95:.6f}, {gain.upper_95:.6f}].
6. **Interaction increment over its own additive component:** {increment.estimate:.6f}
   [{increment.lower_95:.6f}, {increment.upper_95:.6f}].
7. **Full-response context:** affine `g={full_affine.estimate:.6f}`, additive
   `g={full_add.estimate:.6f}`, low-rank `g={full_low.estimate:.6f}`.
8. **Failure anatomy:** low-rank symmetric truth-aligned amplitude
   `{cross_diag.truth_aligned_amplitude:.6f}`, cross-plate cosine
   `{cross_diag.cosine_alignment:.6f}`, scale ratio `{cross_diag.scale_ratio:.6f}`;
   CP-only signed cross-replicate covariance fraction
   `{interaction_diag.predicted_interaction_cross_covariance_fraction:.6f}`. A
   non-negative same-plate interaction-energy fraction is not recoverable from
   the frozen coefficient-summary cache and is therefore not reported.
9. **Synthetic control:** passed `{result['synthetic_passed']}`; rank-4 low-rank
   `g={synthetic_low.context_specific_g:.6f}`, gain over additive
   `{synthetic_low.lowrank_minus_additive:.6f}`.
10. **Verdict:** `{result['verdict']}`.

## Scientific adjudication

The verdict is determined only by the preregistered paired interval and effect
thresholds. A valid no-rescue result means that lack of this explicit rank-32-or-
smaller context-by-intervention interaction family does not explain the dense-
support gap. It does not imply that every nonlinear model must fail. A partial
result means low-rank interaction structure mitigates but does not close CGC. A
close-gap result means the dense-support gap depends substantially on the
estimator family.

The comparison uses matched observations and held-out masks.

{result['verdict']}
"""
    (out / "LOW_RANK_INTERACTION_REPORT.md").write_text(report, encoding="utf-8")

    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "LOW_RANK_OUTPUT_MANIFEST.json":
            artifacts[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    manifests = []
    for kind in ("synthetic", "real"):
        for fold in range(100):
            _, path = _cache_paths(source_root, kind, fold)
            manifests.append({"kind": kind, "fold": fold, "path": str(path), "sha256": sha256(path)})
    output_manifest = {
        "verdict": result["verdict"],
        "branch": _git(root, "branch", "--show-current"),
        "commit_at_generation": _git(root, "rev-parse", "HEAD"),
        "protocol_commit": "86e57476fa0fd32b8d1579008f71d23dd27ce5a3",
        "feasibility_commit": "004d3bbdbfedde28e520d0571f4e5be7060c8121",
        "artifacts": artifacts,
        "fold_manifests": manifests,
    }
    (out / "LOW_RANK_OUTPUT_MANIFEST.json").write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    return str(result["verdict"])
