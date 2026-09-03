"""Frozen, model-free CGC-TCELL-0A stimulation-operator analysis."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from igc_virtual_cell.cgc_tcell.core import (
    crossmeasurement_context_distances,
    crossmeasurement_permutation_null,
    donor_disjoint_distances,
    donor_partitions,
    hash_block,
    hierarchical_target_bootstrap,
    permutation_null_from_crossgrams,
    stratified_permutation_indices,
)
from igc_virtual_cell.phase1.targeted_tensorized import _windows_memory_bytes


def _git(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _open_response(root: Path, artifact: dict[str, Any]) -> np.memmap:
    return np.memmap(
        root / artifact["response_path"],
        mode="r",
        dtype=np.float32,
        shape=tuple(artifact["response_shape"]),
    )


def _center_targets(delta: np.ndarray, path: Path) -> np.memmap:
    centered = np.memmap(path, mode="w+", dtype=np.float32, shape=delta.shape)
    for donor in range(4):
        for state in range(3):
            values = np.asarray(delta[:, donor, state], dtype=np.float32)
            centered[:, donor, state] = values - values.mean(
                axis=0, keepdims=True, dtype=np.float64
            ).astype(np.float32)
    centered.flush()
    return centered


def _summary_from_distances(frame: pd.DataFrame) -> dict[str, float | int]:
    partition_means = frame.groupby("partition_index")["crossvalidated_distance"].mean()
    contrast_means = frame.groupby(["state_a", "state_b"], sort=False)[
        "crossvalidated_distance"
    ].mean()
    return {
        "overall_mean": float(frame["crossvalidated_distance"].mean()),
        "overall_median": float(frame["crossvalidated_distance"].median()),
        "positive_fraction_9": float(frame["positive"].mean()),
        "positive_donor_partitions": int((partition_means > 0).sum()),
        "positive_state_contrasts": int((contrast_means > 0).sum()),
    }


def _run_null(
    delta: np.ndarray,
    nuisance: np.ndarray,
    *,
    draws: int,
    bins: int,
    seed: int,
    batch_draws: int,
    work: Path,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    permutation_path = work / f"{prefix}_permutations.int32.mmap"
    centered_path = work / f"{prefix}_centered.float32.mmap"
    permutations = stratified_permutation_indices(
        nuisance.transpose(1, 2, 0).reshape(12, delta.shape[0]),
        draws=draws,
        bins=bins,
        seed=seed,
        output_path=permutation_path,
    )
    centered = _center_targets(delta, centered_path)
    values = permutation_null_from_crossgrams(
        centered,
        permutations,
        batch_draws=batch_draws,
        device="cuda",
    )
    columns = []
    state_pairs = list(combinations(range(3), 2))
    for partition in range(3):
        for left, right in state_pairs:
            columns.append(f"partition_{partition}__contrast_{left}_{right}")
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "permutation", np.arange(draws))
    frame["overall_mean"] = values.mean(axis=1)
    frame["positive_fraction_9"] = (values > 0).mean(axis=1)
    summary = {
        "null_q95": float(np.quantile(frame["overall_mean"], 0.95)),
        "null_mean": float(frame["overall_mean"].mean()),
        "null_sd": float(frame["overall_mean"].std(ddof=1)),
    }
    del centered, permutations
    return frame, summary


def _intervention_blocks(
    delta: np.ndarray,
    targets: list[str],
    *,
    donors: list[str],
    states: list[str],
) -> pd.DataFrame:
    assignment = np.asarray([hash_block(target) for target in targets])
    rows = []
    for block in range(5):
        keep = assignment == block
        distances, _ = donor_disjoint_distances(
            delta[keep], donors=donors, states=states
        )
        rows.append(
            {
                "analysis": "block_only",
                "block": block,
                "targets": int(keep.sum()),
                "overall_mean": float(distances["crossvalidated_distance"].mean()),
                "positive": bool(distances["crossvalidated_distance"].mean() > 0),
            }
        )
        distances, _ = donor_disjoint_distances(
            delta[~keep], donors=donors, states=states
        )
        rows.append(
            {
                "analysis": "leave_one_block_out",
                "block": block,
                "targets": int((~keep).sum()),
                "overall_mean": float(distances["crossvalidated_distance"].mean()),
                "positive": bool(distances["crossvalidated_distance"].mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def _gene_blocks(
    delta: np.ndarray,
    strict: pd.DataFrame,
    *,
    donors: list[str],
    states: list[str],
) -> pd.DataFrame:
    genes = strict.loc[strict["strict_trans_eligible"]].copy()
    labels = genes["gene_id"].astype(str).map(hash_block).to_numpy()
    block_rows = []
    raw_contributions = []
    for block in range(5):
        keep = labels == block
        distances, _ = donor_disjoint_distances(
            delta[:, :, :, keep], donors=donors, states=states
        )
        mean = float(distances["crossvalidated_distance"].mean())
        contribution = mean * float(keep.mean())
        raw_contributions.append(contribution)
        block_rows.append(
            {
                "analysis": "block_only",
                "block": block,
                "genes": int(keep.sum()),
                "overall_mean": mean,
                "weighted_contribution": contribution,
                "positive": mean > 0,
            }
        )
        distances, _ = donor_disjoint_distances(
            delta[:, :, :, ~keep], donors=donors, states=states
        )
        block_rows.append(
            {
                "analysis": "leave_one_block_out",
                "block": block,
                "genes": int((~keep).sum()),
                "overall_mean": float(distances["crossvalidated_distance"].mean()),
                "weighted_contribution": np.nan,
                "positive": bool(distances["crossvalidated_distance"].mean() > 0),
            }
        )
    denominator = float(np.abs(raw_contributions).sum())
    for row in block_rows:
        if row["analysis"] == "block_only":
            row["absolute_contribution_fraction"] = (
                abs(row["weighted_contribution"]) / denominator if denominator else np.nan
            )
        else:
            row["absolute_contribution_fraction"] = np.nan
    return pd.DataFrame(block_rows)


def _partition_manifest(donors: list[str], states: list[str]) -> pd.DataFrame:
    rows = []
    for index, (left, right) in enumerate(donor_partitions(donors)):
        rows.append(
            {
                "partition_index": index,
                "donor_half_a": "|".join(donors[item] for item in left),
                "donor_half_b": "|".join(donors[item] for item in right),
                "states": "|".join(states),
            }
        )
    return pd.DataFrame(rows)


def _cohort_analysis(
    root: Path,
    artifact: dict[str, Any],
    *,
    cohort: str,
    donors: list[str],
    states: list[str],
    null_draws: int,
    null_bins: int,
    seed: int,
    batch_draws: int,
    work: Path,
) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame, dict[str, float | int]]:
    delta = _open_response(root, artifact)
    distances, contributions = donor_disjoint_distances(
        delta, donors=donors, states=states
    )
    distances.insert(0, "cohort", cohort)
    nuisance = np.load(root / artifact["nuisance_path"])
    null, null_summary = _run_null(
        delta,
        nuisance,
        draws=null_draws,
        bins=null_bins,
        seed=seed,
        batch_draws=batch_draws,
        work=work,
        prefix=cohort,
    )
    observed = _summary_from_distances(distances)
    observed.update(null_summary)
    observed["empirical_p"] = float(
        (1 + np.count_nonzero(null["overall_mean"] >= observed["overall_mean"]))
        / (len(null) + 1)
    )
    observed["exceeds_null_q95"] = bool(
        observed["overall_mean"] > observed["null_q95"]
    )
    return distances, contributions, null, observed


def _guide_disjoint_analysis(
    root: Path,
    output: Path,
    work: Path,
    config: dict[str, Any],
    donors: list[str],
    states: list[str],
) -> dict[str, Any]:
    manifest_path = output / "guide_disjoint_manifest.json"
    if not manifest_path.exists():
        return {"status": "GUIDE_DISJOINT_FULL_CONTEXT_NOT_FEASIBLE"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "GUIDE_DISJOINT_FULL_CONTEXT_MATERIALIZED":
        return manifest
    shape = tuple(manifest["response_shape"])
    left = np.memmap(root / manifest["response_a_path"], mode="r", dtype=np.float32, shape=shape)
    right = np.memmap(root / manifest["response_b_path"], mode="r", dtype=np.float32, shape=shape)
    left_context = left.reshape(shape[0], 12, shape[-1])
    right_context = right.reshape(shape[0], 12, shape[-1])
    observed = crossmeasurement_context_distances(left_context, right_context)
    labels = [f"{donor}|{state}" for donor in donors for state in states]
    observed["context_a_label"] = observed["context_a"].map(dict(enumerate(labels)))
    observed["context_b_label"] = observed["context_b"].map(dict(enumerate(labels)))
    observed.to_csv(output / "guide_disjoint_distances.csv", index=False)
    nuisance = np.load(root / manifest["nuisance_path"]).transpose(1, 2, 0).reshape(12, shape[0])
    maps = stratified_permutation_indices(
        nuisance,
        draws=int(config["null"]["permutations"]),
        bins=int(config["null"]["cell_count_bins"]),
        seed=int(config["random_seed"]) + 3,
        output_path=work / "guide_disjoint_permutations.int32.mmap",
    )
    null_values = crossmeasurement_permutation_null(
        left_context,
        right_context,
        maps,
        batch_draws=int(config["null"]["gpu_batch_size"]),
        device="cuda",
    )
    null = pd.DataFrame(null_values)
    null.insert(0, "permutation", np.arange(len(null)))
    null["overall_mean"] = null_values.mean(axis=1)
    null.to_csv(output / "guide_disjoint_null.csv", index=False)
    overall = float(observed["crossvalidated_distance"].mean())
    q95 = float(np.quantile(null["overall_mean"], 0.95))
    p_value = float(
        (1 + np.count_nonzero(null["overall_mean"] >= overall)) / (len(null) + 1)
    )
    return {
        "status": "GUIDE_DISJOINT_FULL_CONTEXT_COMPLETED",
        "targets": shape[0],
        "contexts": 12,
        "context_pairs": len(observed),
        "overall_mean": overall,
        "positive_pair_fraction": float(observed["positive"].mean()),
        "null_q95": q95,
        "empirical_p": p_value,
        "exceeds_null_q95": overall > q95,
    }


def run_analysis(root: Path, config_path: Path) -> dict[str, Any]:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["outputs"]["directory"]
    processed = root / config["storage"]["processed_directory"]
    reports = root / "results" / "reports"
    work = processed / "null_work"
    work.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    response_manifest = json.loads(
        (output / "normalized_response_manifest.json").read_text(encoding="utf-8")
    )
    inventory = json.loads((output / "dataset_inventory.json").read_text(encoding="utf-8"))
    donors = list(map(str, inventory["donors"]))
    states = list(map(str, config["frozen_states"]))
    _partition_manifest(donors, states).to_csv(
        output / "donor_partition_manifest.csv", index=False
    )
    primary_artifact = response_manifest["artifacts"]["primary"]
    primary_distances, contributions, primary_null, primary_summary = _cohort_analysis(
        root,
        primary_artifact,
        cohort="PRIMARY",
        donors=donors,
        states=states,
        null_draws=int(config["null"]["permutations"]),
        null_bins=int(config["null"]["cell_count_bins"]),
        seed=int(config["random_seed"]),
        batch_draws=int(config["null"]["gpu_batch_size"]),
        work=work,
    )
    primary_distances.to_csv(output / "stimulation_crossvalidated_distances.csv", index=False)
    bootstrap = hierarchical_target_bootstrap(
        contributions,
        states=states,
        draws=int(config["bootstrap"]["draws"]),
        seed=int(config["random_seed"]) + 1,
    )
    bootstrap.to_csv(output / "stimulation_bootstrap.csv", index=False)
    primary_null.to_csv(output / "stimulation_null.csv", index=False)

    primary_delta = _open_response(root, primary_artifact)
    targets = list(map(str, primary_artifact["target_ids"]))
    intervention_blocks = _intervention_blocks(
        primary_delta, targets, donors=donors, states=states
    )
    intervention_blocks.to_csv(output / "intervention_block_robustness.csv", index=False)
    strict = pd.read_csv(output / "strict_trans_genes.csv")
    gene_blocks = _gene_blocks(primary_delta, strict, donors=donors, states=states)
    gene_blocks.to_csv(output / "gene_block_robustness.csv", index=False)

    matched_artifact = response_manifest["artifacts"]["cell_count_matched"]
    matched_distances, _, matched_null, matched_summary = _cohort_analysis(
        root,
        matched_artifact,
        cohort="CELL_COUNT_MATCHED",
        donors=donors,
        states=states,
        null_draws=int(config["null"]["permutations"]),
        null_bins=int(config["null"]["cell_count_bins"]),
        seed=int(config["random_seed"]) + 2,
        batch_draws=int(config["null"]["gpu_batch_size"]),
        work=work,
    )
    pd.DataFrame(
        [
            {"cohort": "CELL_COUNT_MATCHED", **matched_summary},
            {
                "cohort": "CELL_COUNT_MATCHED",
                "null_draws": len(matched_null),
                "targets": matched_artifact["targets"],
            },
        ]
    ).to_csv(output / "cell_count_matched_summary.csv", index=False)

    guide_summary = _guide_disjoint_analysis(
        root, output, work, config, donors, states
    )
    guide_feasible = guide_summary["status"] == "GUIDE_DISJOINT_FULL_CONTEXT_COMPLETED"
    pd.DataFrame([guide_summary]).to_csv(output / "guide_disjoint_summary.csv", index=False)
    run_feasible = int(inventory["run_disjoint_complete_targets_pre_qc"]) >= int(
        config.get("run_disjoint", {}).get("minimum_complete_targets", 300)
    )
    pd.DataFrame(
        [
            {
                "status": "RUN_DISJOINT_PENDING_MATERIALIZATION"
                if run_feasible
                else "RUN_DISJOINT_NOT_FEASIBLE",
                "complete_targets_pre_qc": inventory[
                    "run_disjoint_complete_targets_pre_qc"
                ],
                "frozen_threshold": config.get("run_disjoint", {}).get(
                    "minimum_complete_targets", 300
                ),
            }
        ]
    ).to_csv(output / "run_disjoint_summary.csv", index=False)

    overall_bootstrap = bootstrap.loc[bootstrap["statistic"].eq("overall_mean")].iloc[0]
    primary_block_only = intervention_blocks.loc[
        intervention_blocks["analysis"].eq("block_only")
    ]
    gene_block_only = gene_blocks.loc[gene_blocks["analysis"].eq("block_only")]
    gates = {
        "overall_mean_positive": primary_summary["overall_mean"] > 0,
        "bootstrap_ci_low_positive": float(overall_bootstrap["ci_low"]) > 0,
        "observed_exceeds_null_q95": bool(primary_summary["exceeds_null_q95"]),
        "empirical_p_below_0_05": float(primary_summary["empirical_p"])
        < float(config["decision"]["empirical_p_max_exclusive"]),
        "all_three_state_contrasts_positive": primary_summary[
            "positive_state_contrasts"
        ]
        == 3,
        "all_three_partitions_positive": primary_summary[
            "positive_donor_partitions"
        ]
        == 3,
        "cell_count_matched_positive_and_exceeds_q95": matched_summary["overall_mean"]
        > 0
        and bool(matched_summary["exceeds_null_q95"]),
        "four_of_five_intervention_blocks_positive": int(
            primary_block_only["positive"].sum()
        )
        >= int(config["decision"]["positive_intervention_blocks_min"]),
        "no_gene_block_over_half_absolute_contribution": float(
            gene_block_only["absolute_contribution_fraction"].max()
        )
        <= float(config["decision"]["maximum_single_gene_block_fraction"]),
    }
    if not gates["observed_exceeds_null_q95"]:
        verdict = "TCELL_STIMULATION_OPERATOR_SIGNAL_NOT_RESOLVED"
    elif all(gates.values()):
        verdict = "TCELL_STIMULATION_OPERATOR_SIGNAL_CONFIRMED"
    else:
        verdict = "TCELL_STIMULATION_OPERATOR_SIGNAL_PARTIAL"

    if verdict == "TCELL_STIMULATION_OPERATOR_SIGNAL_CONFIRMED":
        factorial_status = "REQUIRES_GUIDE_OR_RUN_DISJOINT_NOISE_ESTIMATE"
        de_status = "AUTHORIZED_AFTER_PRIMARY_CONFIRMATION_NOT_YET_DOWNLOADED"
    else:
        factorial_status = "NOT_RUN_PRIMARY_SIGNAL_NOT_CONFIRMED"
        de_status = "NOT_AUTHORIZED_PRIMARY_SIGNAL_NOT_CONFIRMED"
    pd.DataFrame([{"status": factorial_status}]).to_csv(
        output / "factorial_signal_decomposition.csv", index=False
    )
    pd.DataFrame([{"status": de_status}]).to_csv(
        output / "independent_de_confirmation.csv", index=False
    )
    verdict_payload = {
        "git_provenance": _git(root),
        "verdict": verdict,
        "gates": gates,
        "primary": primary_summary,
        "bootstrap_ci": [float(overall_bootstrap["ci_low"]), float(overall_bootstrap["ci_high"])],
        "cell_count_matched": matched_summary,
        "guide_disjoint_feasible_pre_qc": guide_feasible,
        "guide_disjoint": guide_summary,
        "run_disjoint_feasible_pre_qc": run_feasible,
    }
    (output / "verdict.json").write_text(
        json.dumps(verdict_payload, indent=2) + "\n", encoding="utf-8"
    )
    strongest = (
        primary_distances.groupby(["state_a", "state_b"], sort=False)[
            "crossvalidated_distance"
        ]
        .mean()
        .idxmax()
    )
    report = f"""# CGC-TCELL-0A truth-operator audit

- Verdict: **{verdict}**
- Overall cross-validated distance: **{primary_summary['overall_mean']:.8g}**
- Hierarchical bootstrap 95% CI: **[{float(overall_bootstrap['ci_low']):.8g}, {float(overall_bootstrap['ci_high']):.8g}]**
- Heteroskedastic null q95: **{primary_summary['null_q95']:.8g}**
- Empirical permutation p: **{primary_summary['empirical_p']:.6g}**
- Positive donor partitions: **{primary_summary['positive_donor_partitions']}/3**
- Positive stimulation contrasts: **{primary_summary['positive_state_contrasts']}/3**
- Strongest contrast: **{strongest[0]} ↔ {strongest[1]}**
- Cell-count-matched observed/null q95: **{matched_summary['overall_mean']:.8g} / {matched_summary['null_q95']:.8g}**
- Positive intervention blocks: **{int(primary_block_only['positive'].sum())}/5**
- Largest absolute gene-block contribution: **{float(gene_block_only['absolute_contribution_fraction'].max()):.3f}**

No predictive model was trained. No cell-level data or published DE estimates were used.
"""
    (reports / "cgc_tcell_truth_operator_audit.md").write_text(report, encoding="utf-8")
    (reports / "cgc_tcell_factorial_decomposition.md").write_text(
        f"# CGC-TCELL-0A factorial decomposition\n\nStatus: **{factorial_status}**\n",
        encoding="utf-8",
    )
    (reports / "cgc_tcell_phase_summary.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "git_provenance": _git(root),
        "branch": subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "model_training": False,
        "cell_level_data_used": False,
        "published_de_used": False,
        "null_permutations_primary": int(config["null"]["permutations"]),
        "null_permutations_cell_count_matched": int(config["null"]["permutations"]),
        "bootstrap_draws": int(config["bootstrap"]["draws"]),
        "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "peak_rss_bytes": int(_windows_memory_bytes()[1]),
        "runtime_seconds": time.perf_counter() - started,
        "verdict": verdict,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return verdict_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    print(json.dumps(run_analysis(root, config), indent=2))


if __name__ == "__main__":
    main()
