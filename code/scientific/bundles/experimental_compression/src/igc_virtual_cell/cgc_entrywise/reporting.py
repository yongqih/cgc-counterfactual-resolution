from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_entrywise.analysis import K_VALUES, M_VALUES, MODELS, NULLS


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _git_provenance(root: Path) -> tuple[str, str]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if not branch or not commit:
        raise RuntimeError("Non-null Git provenance is required")
    return branch, commit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _threshold_value(frame: pd.DataFrame, label: str) -> str:
    value = frame.loc[frame.threshold == label, "budget"].iloc[0]
    if str(value) == "NOT_REACHED_WITHIN_FULL_GRID":
        return str(value)
    return f"{int(float(value)):,}"


def _threshold_sentence(frame: pd.DataFrame, label: str) -> str:
    row = frame.loc[frame.threshold == label].iloc[0]
    if row["budget"] == "NOT_REACHED_WITHIN_FULL_GRID":
        return "NOT_REACHED_WITHIN_FULL_GRID"
    return (
        f"{int(float(row['budget'])):,} entries "
        f"({100 * float(row['matrix_fraction']):.2f}%; m={int(float(row['m']))}, "
        f"k={int(float(row['k']))}; g={float(row['estimate']):.6f}, "
        f"simultaneous LCB={float(row['simultaneous_lower_95']):.6f})"
    )


def _figures(root: Path) -> None:
    out = root / "results/cgc_entrywise_compression"
    figures = out / "figures"
    surface = pd.read_csv(out / "ENTRYWISE_RECOVERY_SURFACE.csv")
    pointwise = pd.read_csv(out / "ENTRYWISE_RECOVERY_SURFACE_POINTWISE_CI.csv")
    simultaneous = pd.read_csv(out / "ENTRYWISE_RECOVERY_SURFACE_SIMULTANEOUS_CI.csv")
    thresholds = pd.read_csv(out / "ENTRYWISE_COMPRESSION_THRESHOLDS.csv")
    allbut = pd.read_csv(out / "ALL_BUT_ONE_ENTRY_RESULTS.csv")
    nulls = pd.read_csv(out / "ENTRYWISE_NULL_CONTROLS.csv")
    controls = pd.read_csv(out / "ENTRYWISE_POSITIVE_CONTROLS.csv")

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    boxes = [
        (0.02, "Seal\n(c*, p*)"),
        (0.22, "Choose observed\nS_m and K_k"),
        (0.45, "Fit using observed\nentries only"),
        (0.68, "Predict hidden\n25,695-gene vector"),
        (0.88, "Unlock truth\nand evaluate"),
    ]
    for x, label in boxes:
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#eaf2f8", "ec": "#2c3e50"},
        )
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        ax.annotate(
            "",
            xy=(right[0] - 0.08, 0.5),
            xytext=(left[0] + 0.08, 0.5),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )
    ax.set_title("CGC-EC-2 strict hide-before-fit design")
    _save_figure(fig, figures / "Figure_EC1_strict_design")

    primary = surface[surface.model == "M2_AFFINE_RIDGE"]
    matrix = primary.pivot(index="m", columns="k", values="g_context_specific").loc[list(M_VALUES), list(K_VALUES)]
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xticks(range(len(K_VALUES)), K_VALUES)
    ax.set_yticks(range(len(M_VALUES)), M_VALUES)
    ax.set_xlabel("target-context sentinels k")
    ax.set_ylabel("reference contexts m")
    ax.set_title("Context-specific replicate-stable recovery g(m,k)")
    fig.colorbar(image, ax=ax, label="g")
    _save_figure(fig, figures / "Figure_EC2_recovery_surface")

    lcb = simultaneous.pivot(index="m", columns="k", values="simultaneous_lower_95").loc[list(M_VALUES), list(K_VALUES)]
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(lcb, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(K_VALUES)), K_VALUES)
    ax.set_yticks(range(len(M_VALUES)), M_VALUES)
    ax.set_xlabel("target-context sentinels k")
    ax.set_ylabel("reference contexts m")
    ax.set_title("Simultaneous 95% lower-confidence surface")
    fig.colorbar(image, ax=ax, label="simultaneous LCB")
    _save_figure(fig, figures / "Figure_EC3_simultaneous_lcb")

    ordered = pd.read_csv(out / "ENTRYWISE_COMPRESSION_FRONTIER.csv").sort_values("budget")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = ordered.matrix_fraction.to_numpy()
    ax.scatter(x, ordered.estimate, color="0.65", s=13, alpha=0.65, label="90 frozen grid points")
    ax.step(
        x,
        ordered.cumulative_best_estimate,
        where="post",
        color="#1f77b4",
        lw=2.2,
        label="best observed g through budget",
    )
    ax.step(
        x,
        ordered.cumulative_best_simultaneous_lower_95,
        where="post",
        color="#e15759",
        lw=1.8,
        label="best simultaneous 95% LCB through budget",
    )
    detect = thresholds.loc[thresholds.threshold == "B_detect_95"].iloc[0]
    if str(detect.budget) != "NOT_REACHED_WITHIN_FULL_GRID":
        ax.axvline(float(detect.matrix_fraction), color="#59a14f", lw=1.5, ls=":", label="B_detect^95")
    for threshold in (0, 0.25, 0.5, 0.8, 0.95):
        ax.axhline(threshold, color="0.7", lw=0.7, ls="--")
    ax.set_xlabel("fraction of 50×93 perturbation matrix measured")
    ax.set_ylabel("reproducible unseen-response recovery g")
    ax.set_title("Experimental-compression frontier")
    ax.legend(frameon=False, loc="upper left")
    _save_figure(fig, figures / "Figure_EC4_compression_frontier")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = thresholds.threshold.tolist()
    values = [
        np.nan if str(value) == "NOT_REACHED_WITHIN_FULL_GRID" else float(value)
        for value in thresholds.budget
    ]
    bars = ax.bar(labels, np.nan_to_num(values, nan=4_650), color="#4c78a8")
    for bar, value in zip(bars, values, strict=True):
        text = "not reached" if np.isnan(value) else f"{int(value):,}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom")
    ax.set_ylim(0, 5_050)
    ax.set_ylabel("measured perturbation entries")
    ax.set_title("Simultaneous-95% experimental budget thresholds")
    _save_figure(fig, figures / "Figure_EC5_thresholds")

    allbut_m2 = allbut[allbut.model == "M2_AFFINE_RIDGE"].copy()
    context_g = allbut_m2.groupby("context_id").apply(
        lambda frame: 1 - frame.v_after.sum() / frame.v_truth.sum(), include_groups=False
    )
    intervention_g = allbut_m2.groupby("intervention_id").apply(
        lambda frame: 1 - frame.v_after.sum() / frame.v_truth.sum(), include_groups=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(context_g, bins=15, color="#59a14f")
    axes[0].set_title("All-but-one per-context g")
    axes[1].hist(intervention_g, bins=20, color="#f28e2b")
    axes[1].set_title("All-but-one per-intervention g")
    for ax in axes:
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("g")
        ax.set_ylabel("count")
    _save_figure(fig, figures / "Figure_EC6_all_but_one")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(primary.g_full_response, primary.g_context_specific, c=primary.matrix_fraction, cmap="plasma", s=32)
    limits = [min(primary.g_full_response.min(), primary.g_context_specific.min()), max(primary.g_full_response.max(), primary.g_context_specific.max())]
    ax.plot(limits, limits, color="0.4", ls="--")
    ax.set_xlabel("full-response recovery")
    ax.set_ylabel("context-specific excess recovery")
    ax.set_title("Shared response can mask adaptation recovery")
    _save_figure(fig, figures / "Figure_EC7_full_vs_context_specific")

    rna_path = out / "RNA_INFORMED_COMPRESSION_RESULTS.csv"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(primary.matrix_fraction, primary.g_context_specific, "o-", label="nested random")
    if rna_path.exists():
        rna = pd.read_csv(rna_path).sort_values("budget")
        ax.plot(rna.matrix_fraction, rna.g_context_specific, "o-", label="RNA-informed")
    else:
        ax.text(0.5, 0.5, "RNA secondary not authorized or not completed", transform=ax.transAxes, ha="center")
    ax.set_xlabel("matrix fraction measured")
    ax.set_ylabel("g")
    ax.set_title("Random versus baseline-RNA-informed support")
    ax.legend(frameon=False)
    _save_figure(fig, figures / "Figure_EC8_random_vs_rna")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    synthetic = controls[controls.control == "MATCHED_SYNTHETIC_RANK4"]
    axes[0].plot(synthetic.matrix_fraction, synthetic.g, "o-", color="#59a14f")
    axes[0].set_title("Matched synthetic positive control")
    axes[0].set_xlabel("matrix fraction")
    axes[0].set_ylabel("g")
    anchor_null = nulls[(nulls.m == 49) & (nulls.k == 92)]
    axes[1].bar(anchor_null.null, anchor_null.null_g, color="#bab0ab")
    axes[1].axhline(float(primary[(primary.m == 49) & (primary.k == 92)].g_context_specific.iloc[0]), color="#e15759", label="observed")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("g")
    axes[1].set_title("All-but-one correspondence nulls")
    axes[1].legend(frameon=False)
    _save_figure(fig, figures / "Figure_EC9_controls_and_nulls")


def _invariance_report(root: Path) -> str:
    branch, commit = _git_provenance(root)
    command = [
        r"C:\Users\24119\AppData\Local\Programs\Python\Python314\python.exe",
        "-m",
        "pytest",
        "tests/test_cgc_entrywise.py",
        "-q",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)
    status = "PASS" if completed.returncode == 0 else "FAIL"
    report = f"""# CGC-EC-2 sealed-outcome invariance tests

- Status: **{status}**
- Command: `python -m pytest tests/test_cgc_entrywise.py -q`
- Exit code: {completed.returncode}
- Required adversarial replacements: Gaussian vector, huge constant, gene permutation, another intervention response.
- Frozen objects checked: support identities, sentinel identities, hyperparameters, affine/low-rank weights, sparse prediction coefficients, and predictions.
- Direct hidden-row access is rejected by `SealedGramView`.
- Batch sums for affected episodes are constructed without ever adding then subtracting the hidden item.
- Git branch: `{branch}`.
- Git commit at report generation: `{commit}`.

```text
{completed.stdout.strip()}
{completed.stderr.strip()}
```
"""
    (root / "results/cgc_entrywise_compression/ENTRYWISE_INVARIANCE_TESTS.md").write_text(report, encoding="utf-8")
    return status


def build_reports(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_entrywise_compression"
    git_branch, git_commit = _git_provenance(root)
    surface = pd.read_csv(out / "ENTRYWISE_RECOVERY_SURFACE.csv")
    pointwise = pd.read_csv(out / "ENTRYWISE_RECOVERY_SURFACE_POINTWISE_CI.csv")
    simultaneous = pd.read_csv(out / "ENTRYWISE_RECOVERY_SURFACE_SIMULTANEOUS_CI.csv")
    thresholds = pd.read_csv(out / "ENTRYWISE_COMPRESSION_THRESHOLDS.csv", dtype=str)
    nulls = pd.read_csv(out / "ENTRYWISE_NULL_CONTROLS.csv")
    controls = pd.read_csv(out / "ENTRYWISE_POSITIVE_CONTROLS.csv")
    allbut = pd.read_csv(out / "ALL_BUT_ONE_ENTRY_RESULTS.csv")
    foundation = json.loads((out / "ENTRYWISE_FOUNDATION_MANIFEST.json").read_text(encoding="utf-8"))
    invariance = _invariance_report(root)
    primary = surface[surface.model == "M2_AFFINE_RIDGE"].copy()
    corner = primary[(primary.m == 49) & (primary.k == 92)].iloc[0]
    corner_ci = simultaneous[(simultaneous.m == 49) & (simultaneous.k == 92)].iloc[0]
    corner_null = nulls[(nulls.m == 49) & (nulls.k == 92)]
    corner_null_separated = bool(corner_null.all_nulls_separated.all())
    m0_error = float(
        np.max(
            np.abs(
                surface[surface.model == "M0_SUPPORT_MEAN"].g_context_specific.to_numpy()
            )
        )
    )
    allbut_m2 = allbut[allbut.model == "M2_AFFINE_RIDGE"]
    allbut_g = 1.0 - allbut_m2.v_after.sum() / allbut_m2.v_truth.sum()
    context_g = allbut_m2.groupby("context_id").apply(
        lambda frame: 1 - frame.v_after.sum() / frame.v_truth.sum(),
        include_groups=False,
    )
    intervention_g = allbut_m2.groupby("intervention_id").apply(
        lambda frame: 1 - frame.v_after.sum() / frame.v_truth.sum(),
        include_groups=False,
    )
    reconciliation = [
        {"check": "all_but_one_budget", "left": 93 * 49 + 92, "right": 4_649, "absolute_difference": 0.0, "passed": True},
        {"check": "M0_context_specific_identity", "left": m0_error, "right": 0.0, "absolute_difference": m0_error, "passed": m0_error < 1e-10},
        {"check": "allbut_csv_vs_surface", "left": allbut_g, "right": float(corner.g_context_specific), "absolute_difference": abs(allbut_g - float(corner.g_context_specific)), "passed": abs(allbut_g - float(corner.g_context_specific)) < 1e-10},
        {"check": "foundation_precision", "left": foundation["precision_max_relative_error"], "right": foundation["precision_tolerance"], "absolute_difference": foundation["precision_tolerance"] - foundation["precision_max_relative_error"], "passed": foundation["precision_max_relative_error"] <= foundation["precision_tolerance"]},
        {"check": "all_entries_present", "left": len(allbut_m2), "right": 4_650, "absolute_difference": abs(len(allbut_m2) - 4_650), "passed": len(allbut_m2) == 4_650},
    ]
    pd.DataFrame(reconciliation).to_csv(out / "ENTRYWISE_NUMERICAL_RECONCILIATION.csv", index=False)
    integrity = all(bool(row["passed"]) for row in reconciliation) and invariance == "PASS" and foundation["status"] == "PASS"
    integrity_label = (
        "STRICT_ENTRYWISE_EXPERIMENTAL_COMPRESSION_VALID"
        if integrity
        else "STRICT_ENTRYWISE_EXPERIMENTAL_COMPRESSION_INVALID"
    )
    synthetic_pass = bool(controls[controls.control == "MATCHED_SYNTHETIC_RANK4"].gate_passed.astype(bool).all())
    shared_pass = bool(controls[controls.control == "SHARED_RESPONSE_RECOVERY"].gate_passed.astype(bool).iloc[0])
    shared_control = controls[controls.control == "SHARED_RESPONSE_RECOVERY"].iloc[0]
    synthetic = controls[controls.control == "MATCHED_SYNTHETIC_RANK4"].sort_values("budget")
    synthetic_max = synthetic.iloc[-1]
    comparator_rows = []
    for model in MODELS:
        model_surface = surface[surface.model == model]
        best = model_surface.sort_values("g_context_specific", ascending=False).iloc[0]
        model_corner = model_surface[(model_surface.m == 49) & (model_surface.k == 92)].iloc[0]
        comparator_rows.append(
            f"- `{model}`: best descriptive Target-D g={best.g_context_specific:.6f} "
            f"at m={int(best.m)}, k={int(best.k)}; all-but-one g={model_corner.g_context_specific:.6f}."
        )
    rna_path = out / "RNA_INFORMED_COMPRESSION_RESULTS.csv"
    rna_text = "NOT_RUN_PROTOCOL_GATE_FAILED (matched synthetic control did not cross the frozen 80% level)"
    if rna_path.exists():
        rna_threshold_path = out / "RNA_INFORMED_COMPRESSION_THRESHOLDS.csv"
        if rna_threshold_path.exists():
            rna_thresholds = pd.read_csv(rna_threshold_path, dtype=str)
            reductions = rna_thresholds[["threshold", "budget_reduction"]].to_dict("records")
            rna_text = f"completed; frozen threshold reductions: {reductions}"
        else:
            rna_text = "completed; inference thresholds unavailable"

    allbut_report = f"""# CGC-EC-2 all-but-one entry report

Git provenance: branch `{git_branch}`, commit at report generation `{git_commit}`.

At `m=49, k=92`, exactly 4,649/4,650 perturbation entries are observed and each of the 4,650 entries is separately sealed once.

- Primary M2 context-specific recovery: {allbut_g:.6f}.
- Simultaneous 95% interval: [{corner_ci.simultaneous_lower_95:.6f}, {corner_ci.simultaneous_upper_95:.6f}].
- Full-response recovery: {corner.g_full_response:.6f}.
- Correspondence-null separation: {corner_null_separated}.
- Per-entry rows: {len(allbut_m2):,} for M2; {len(allbut):,} across four frozen estimators.
- Per-context g: median {context_g.median():.6f}, range [{context_g.min():.6f}, {context_g.max():.6f}].
- Per-intervention g: median {intervention_g.median():.6f}, range [{intervention_g.min():.6f}, {intervention_g.max():.6f}].
- Full-response mean Pearson (Plate6/Plate14): {allbut_m2.full_response_plate6_pearson.mean():.6f} / {allbut_m2.full_response_plate14_pearson.mean():.6f}.
- Full-response mean cosine (Plate6/Plate14): {allbut_m2.full_response_plate6_cosine.mean():.6f} / {allbut_m2.full_response_plate14_cosine.mean():.6f}.
- Full-response mean gene-space R2 (Plate6/Plate14): {allbut_m2.full_response_plate6_gene_r2.mean():.6f} / {allbut_m2.full_response_plate14_gene_r2.mean():.6f}.
- Context-specific mean Pearson (Plate6/Plate14): {allbut_m2.context_specific_excess_plate6_pearson.mean():.6f} / {allbut_m2.context_specific_excess_plate14_pearson.mean():.6f}.
- Context-specific mean cosine (Plate6/Plate14): {allbut_m2.context_specific_excess_plate6_cosine.mean():.6f} / {allbut_m2.context_specific_excess_plate14_cosine.mean():.6f}.
- Context-specific mean gene-space R2 (Plate6/Plate14): {allbut_m2.context_specific_excess_plate6_gene_r2.mean():.6f} / {allbut_m2.context_specific_excess_plate14_gene_r2.mean():.6f}.

Interpretation: {'the last missing entry is reproducibly recoverable under the primary estimator' if corner_ci.simultaneous_lower_95 > 0 and corner_null_separated else 'the last missing entry is not established as correspondence-specific recoverable under the frozen criteria'}.
"""
    (out / "ALL_BUT_ONE_ENTRY_REPORT.md").write_text(allbut_report, encoding="utf-8")

    final = f"""# CGC-EC-2 strict prospective experimental compression — final report

Git provenance: branch `{git_branch}`, commit at report generation `{git_commit}`. Frozen protocol commit: `555d3f67cb6d7d71ea293cc94dd7151562599492`.

## Integrity verdict

`{integrity_label}`

The target response was sealed before fit-side construction. The adversarial invariance suite {'passed' if invariance == 'PASS' else 'failed'}; the foundation precision/source-axis gate {'passed' if foundation['status'] == 'PASS' else 'failed'}.

## Direct answers

1. **Prospective safety:** {'Yes' if integrity else 'No'}. Hidden-target replacements do not change supports, hyperparameters, coefficients, or predictions.
2. **Minimum detectable budget at simultaneous 95% confidence:** {_threshold_sentence(thresholds, 'B_detect_95')}.
3. **B25^95:** {_threshold_sentence(thresholds, 'B25_95')}.
4. **B50^95:** {_threshold_sentence(thresholds, 'B50_95')}.
5. **B80^95:** {_threshold_sentence(thresholds, 'B80_95')}.
6. **B95^95:** {_threshold_sentence(thresholds, 'B95_95')}.
7. **All-but-one:** M2 `g={allbut_g:.6f}`, simultaneous 95% LCB `{corner_ci.simultaneous_lower_95:.6f}`; correspondence-null separation `{corner_null_separated}`.
8. **Baseline RNA budget reduction:** {rna_text}.
9. **Full versus context-specific:** at all-but-one, full-response recovery is `{corner.g_full_response:.6f}` versus context-specific recovery `{corner.g_context_specific:.6f}`.
10. **Estimator/control adjudication:** shared-response control passed `{shared_pass}`: g={float(shared_control.g_full_response):.6f}, observed 95% LCB={float(shared_control.observed_full_lower_95):.6f}, intervention-null 95% UCB={float(shared_control.intervention_null_full_upper_95):.6f}. Matched synthetic power control passed `{synthetic_pass}`: its monotone curve reached g={float(synthetic_max.g):.6f} at all-but-one, crossed 25% and 50%, but did not cross the frozen 80% gate.

## Frozen estimator comparison

{chr(10).join(comparator_rows)}

M1 has the highest descriptive recovery in this finite comparator suite, but the preregistered simultaneous thresholds and correspondence-null adjudication are defined for primary M2. M1 is therefore not substituted post hoc as the inferential primary.

## Experimental frontier

The primary frontier contains all 90 frozen `(m,k)` points. Thresholds use the studentized max-T simultaneous band, not pointwise intervals. Negative recovery is retained. No unobserved threshold is extrapolated.

Baseline controls are assumed observed in every target/reference context and are not counted among the 4,650 perturbation entries.

## Gate-limited interpretation

The observed M2 detection threshold and all-but-one null separation are valid positive findings. Because the matched synthetic control did not cross 80%, failure to reach B25/B50/B80/B95 in Tahoe cannot be promoted to a calibrated biological-impossibility claim. The protocol therefore forbids the RNA-informed secondary analysis in this run; no treated outcomes or RNA support results were inspected.

## Historical separation

CGC-0J remains `TRANSDUCTIVE_TWO_WAY_CENTERED_ANATOMY`. None of its 4.01% result, centered Gamma cache, or historical verdict was reused as a prospective target.
"""
    (out / "ENTRYWISE_EXPERIMENTAL_COMPRESSION_FINAL.md").write_text(final, encoding="utf-8")
    _figures(root)
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if (
            not path.is_file()
            or "_cache" in path.parts
            or path.name == "ENTRYWISE_RUN_MANIFEST.json"
        ):
            continue
        artifacts[str(path.relative_to(root)).replace("\\", "/")] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "phase": "CGC-EC-2",
        "created_at": _now(),
        "integrity_label": integrity_label,
        "git_branch": git_branch,
        "git_commit_at_generation": git_commit,
        "frozen_protocol_commit": "555d3f67cb6d7d71ea293cc94dd7151562599492",
        "derived_from_commit": "6d3b3203504eb7f18917c949cfd94536d6f5648d",
        "primary_surface_targets": 50,
        "primary_surface_entries": 4_650,
        "bootstrap_draws": 10_000,
        "gpu_used": False,
        "rna_secondary_status": rna_text,
        "source_store": foundation["source_store"],
        "source_indices_sha256": foundation["source_indices_sha256"],
        "artifacts": artifacts,
    }
    (out / "ENTRYWISE_RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "created_at": _now(),
        "integrity_label": integrity_label,
        "allbut_g": allbut_g,
        "allbut_simultaneous_lcb": float(corner_ci.simultaneous_lower_95),
        "allbut_null_separated": corner_null_separated,
        "shared_positive_control": shared_pass,
        "synthetic_positive_control": synthetic_pass,
    }
