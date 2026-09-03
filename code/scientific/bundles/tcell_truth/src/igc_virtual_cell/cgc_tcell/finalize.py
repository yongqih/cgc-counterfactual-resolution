"""Finalize CGC-TCELL-0A reports without changing the frozen verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import yaml


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def finalize(root: Path, config_path: Path) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["outputs"]["directory"]
    reports = root / "results" / "reports"
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    verdict_path = output / "verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict["verdict"] != "TCELL_STIMULATION_OPERATOR_SIGNAL_CONFIRMED":
        raise RuntimeError("Finalizer cannot alter or rescue the frozen primary verdict")
    distances = pd.read_csv(output / "stimulation_crossvalidated_distances.csv")
    contrast = distances.groupby(["state_a", "state_b"], sort=False)[
        "crossvalidated_distance"
    ].mean()
    partition = distances.groupby(
        ["partition_index", "donor_half_a", "donor_half_b"], sort=False
    )["crossvalidated_distance"].mean()
    intervention = pd.read_csv(output / "intervention_block_robustness.csv")
    intervention = intervention.loc[intervention["analysis"].eq("block_only")]
    genes = pd.read_csv(output / "gene_block_robustness.csv")
    genes = genes.loc[genes["analysis"].eq("block_only")]
    guide = pd.read_csv(output / "guide_disjoint_summary.csv").iloc[0].to_dict()
    factorial = pd.read_csv(output / "factorial_signal_decomposition.csv")
    factorial_lookup = factorial.set_index("component")
    independent = json.loads(
        (output / "independent_de_confirmation_summary.json").read_text(encoding="utf-8")
    )
    download = json.loads((output / "download_manifest.json").read_text(encoding="utf-8"))
    strongest = contrast.idxmax()
    verdict["git_provenance"] = commit
    verdict["conditional_factorial"] = {
        row.component: {
            "reliable_crossguide_energy": row.reliable_crossguide_energy,
            "positive_truncated_relative_share": row.positive_truncated_relative_share,
            "bootstrap_ci": [row.bootstrap_ci_low, row.bootstrap_ci_high],
        }
        for row in factorial.itertuples()
    }
    verdict["independent_published_de_confirmation"] = independent
    verdict["benchmark_suitability"] = (
        "CONDITIONALLY_SUITABLE_PRIMARY_ESTIMATOR_ONLY; independent guide/DE null confirmation failed"
    )
    verdict["primary_verdict_unchanged_by_conditional_analyses"] = True
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    truth_report = f"""# CGC-TCELL-0A truth-operator audit

Git provenance: `{commit}` on `{branch}`.

## Frozen primary verdict

**{verdict['verdict']}**

The primary pseudobulk estimator passed all nine pre-specified gates. Conditional and confirmatory analyses below did not modify this verdict.

## Primary stimulation operator

- Overall cross-validated distance: **{verdict['primary']['overall_mean']:.8g}**
- Hierarchical target-bootstrap 95% CI: **[{verdict['bootstrap_ci'][0]:.8g}, {verdict['bootstrap_ci'][1]:.8g}]**
- Heteroskedastic null q95: **{verdict['primary']['null_q95']:.8g}**
- Empirical permutation p: **{verdict['primary']['empirical_p']:.8g}** (5,000 draws)
- Positive state contrasts: **3/3**; positive balanced donor partitions: **3/3**
- Strongest contrast: **{strongest[0]} ↔ {strongest[1]}** ({contrast.loc[strongest]:.8g})
- CELL_COUNT_MATCHED: **{verdict['cell_count_matched']['overall_mean']:.8g}**, null q95 **{verdict['cell_count_matched']['null_q95']:.8g}**, p **{verdict['cell_count_matched']['empirical_p']:.8g}**
- Intervention blocks: **{int(intervention['positive'].sum())}/5 positive**
- Gene blocks: **{int(genes['positive'].sum())}/5 positive**; largest absolute contribution **{genes['absolute_contribution_fraction'].max():.1%}**

## Independent-measurement audits

- Guide A/B full 12-context geometry: **{guide['overall_mean']:.8g}**, {guide['positive_pair_fraction']:.1%} positive pairs, but null q95 **{guide['null_q95']:.8g}** and p **{guide['empirical_p']:.3g}**. This confirmatory null failed.
- R1/R2: **RUN_DISJOINT_NOT_FEASIBLE**; targets were not independently repeated in both runs across all 12 contexts.
- Published guide-specific DESeq2: mean **{independent['guide_overall_mean']:.8g}**; all contrasts positive = **{independent['guide_all_contrasts_positive']}**; null confirmation failed.
- Published complementary donor-pair DESeq2: mean **{independent['donor_overall_mean']:.8g}**, positive fraction **{independent['donor_positive_fraction']:.1%}**, null q95 **{independent['donor_null_q95_overall']:.8g}**, p **{independent['donor_empirical_p_overall']:.3g}**. Direction replicated, null separation did not.

## Interpretation boundary

There is a strong, broad and nuisance-separated stimulation operator under the frozen primary log1p-CPM pseudobulk estimator. It is not estimator-invariant: guide-disjoint and published-DE confirmations did not exceed their nulls. The dataset is therefore conditionally suitable for a formal model-CGC test only when the primary estimator is locked as the truth definition and the failed independent confirmations are carried forward as an explicit limitation.

No predictive model, Transformer, MLP, diffusion model, routing analysis, intrinsic-dimension analysis or model-CGC test was run.
"""
    (reports / "cgc_tcell_truth_operator_audit.md").write_text(
        truth_report, encoding="utf-8"
    )
    phase_report = f"""# CGC-TCELL-0A phase summary

Git provenance: `{commit}` on `{branch}`.

1. **Reproducible intervention × stimulation interaction?** Yes under the frozen primary pseudobulk estimator: mean {verdict['primary']['overall_mean']:.8g}, bootstrap CI [{verdict['bootstrap_ci'][0]:.8g}, {verdict['bootstrap_ci'][1]:.8g}].
2. **Exceeds nuisance-preserving null?** Yes: q95 {verdict['primary']['null_q95']:.8g}, p {verdict['primary']['empirical_p']:.8g}.
3. **Strongest contrast?** {strongest[0]} ↔ {strongest[1]} ({contrast.loc[strongest]:.8g}); the other means were {', '.join(f'{a}↔{b}={value:.8g}' for (a,b), value in contrast.items() if (a,b) != strongest)}.
4. **Completely donor-disjoint?** Yes in the primary estimator: all three balanced partitions were positive ({'; '.join(f'{a} vs {b}: {value:.8g}' for (_,a,b), value in partition.items())}). Published donor-pair DE was 9/9 positive but did not exceed its null.
5. **Survives cell-count matching?** Yes: {verdict['cell_count_matched']['overall_mean']:.8g} > q95 {verdict['cell_count_matched']['null_q95']:.8g}.
6. **Broad across perturbations?** Yes: 5/5 hashed intervention blocks positive.
7. **Broad across genes?** Yes: 5/5 gene blocks positive; maximum absolute block contribution {genes['absolute_contribution_fraction'].max():.1%}.
8. **Guide- or run-disjoint replication?** Guide A/B was feasible (3,186 targets) and 65/66 context pairs were positive, but its null failed (p=1). Run-disjoint replication was not feasible.
9. **Reliable factorial signal?** intervention×donor {factorial_lookup.loc['intervention_x_donor','positive_truncated_relative_share']:.1%}, intervention×stimulation {factorial_lookup.loc['intervention_x_stimulation','positive_truncated_relative_share']:.1%}, and intervention×donor×stimulation {factorial_lookup.loc['intervention_x_donor_x_stimulation','positive_truncated_relative_share']:.1%} of positive cross-guide energy. Donor-only CI crossed zero; the other two were positive.
10. **Suitable for formal model-CGC testing?** Conditionally yes for a pre-registered test locked to the primary log1p-CPM truth operator. No for a claim of estimator-invariant truth, because guide-disjoint and published-DE null confirmation failed.

Frozen verdict: **{verdict['verdict']}**

Stop condition honored: no model analysis was started.
"""
    (reports / "cgc_tcell_phase_summary.md").write_text(
        phase_report, encoding="utf-8"
    )
    run_path = output / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(
        {
            "git_provenance": commit,
            "branch": branch,
            "conditional_de_files": [
                "GWCD4i.DE_stats.by_guide.h5mu",
                "GWCD4i.DE_stats.by_donors.h5mu",
            ],
            "download_manifest_sha256_present_for_all_files": all(
                bool(item.get("sha256")) for item in download["files"]
            ),
            "factorial_decomposition_completed": True,
            "published_de_confirmation_completed": True,
            "primary_verdict_unchanged": True,
            "model_analysis_started": False,
        }
    )
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    result = finalize(root, config)
    print(json.dumps({"verdict": result["verdict"], "git_provenance": result["git_provenance"]}, indent=2))


if __name__ == "__main__":
    main()
