"""Measurement-disjoint ETOH-A to response-B eligibility and distance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from igc_virtual_cell.level2_lea import pairing_table, truth_complete_lines


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x0 = np.asarray(x, dtype=np.float64).ravel()
    y0 = np.asarray(y, dtype=np.float64).ravel()
    x0 -= x0.mean()
    y0 -= y0.mean()
    denominator = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
    return float(np.dot(x0, y0) / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    coupling = config["measurement_coupling"]
    metadata = pd.read_csv(args.metadata, sep="\t")
    samples, pairing = pairing_table(metadata)
    complete = truth_complete_lines(samples)
    pairing = pairing.copy()
    pairing["measurement_disjoint_eligible"] = pairing["line"].isin(complete)
    pairing["ETOH_replicate_count"] = pairing["ETOH"]
    pairing["DEX_replicate_count"] = pairing["DEX"]
    pairing["eligibility_reason"] = np.where(
        pairing["measurement_disjoint_eligible"],
        "two matched independent ETOH and DEX versions",
        "fewer than two matched independent versions on at least one treatment side",
    )
    pairing.to_csv(args.output / "CROSS_REPLICATE_ELIGIBILITY.csv", index=False)

    names = sorted({name for values in complete.values() for name in values})
    dtype = {name: np.float32 for name in names}
    expression = pd.read_csv(
        args.matrix, sep="\t", usecols=["GeneID", *names], dtype=dtype
    ).set_index("GeneID")
    lines = sorted(complete)
    baseline_a = []
    response_a = []
    response_b = []
    for line in lines:
        etoh_a, dex_a, etoh_b, dex_b = complete[line]
        baseline_a.append(expression[etoh_a].to_numpy())
        response_a.append(expression[dex_a].to_numpy() - expression[etoh_a].to_numpy())
        response_b.append(expression[dex_b].to_numpy() - expression[etoh_b].to_numpy())
    baseline_a = np.asarray(baseline_a, dtype=np.float64)
    response_a = np.asarray(response_a, dtype=np.float64)
    response_b = np.asarray(response_b, dtype=np.float64)
    response_a -= response_a.mean(axis=0, keepdims=True)
    response_b -= response_b.mean(axis=0, keepdims=True)

    mean = baseline_a.mean(axis=0)
    scale = baseline_a.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (baseline_a - mean) / scale
    components = min(16, len(lines) - 1)
    baseline_scores = PCA(
        n_components=components, svd_solver="randomized", random_state=config["seed"]
    ).fit_transform(standardized)
    baseline_distance = squareform(pdist(baseline_scores, metric="euclidean"))
    response_distance = squareform(pdist(response_b, metric="euclidean"))
    upper = np.triu_indices(len(lines), 1)
    distance_spearman = float(spearmanr(
        baseline_distance[upper], response_distance[upper]
    ).statistic)

    np.fill_diagonal(baseline_distance, np.inf)
    nearest = np.argmin(baseline_distance, axis=1)
    near_response = response_distance[np.arange(len(lines)), nearest]
    rng = np.random.default_rng(config["seed"] + 60000)
    random_medians = np.empty(coupling["random_draws"])
    for draw in range(coupling["random_draws"]):
        comparison = np.empty(len(lines), dtype=int)
        for index in range(len(lines)):
            choice = rng.integers(0, len(lines) - 1)
            comparison[index] = choice + (choice >= index)
        random_medians[draw] = np.median(
            response_distance[np.arange(len(lines)), comparison]
        )
    near_median = float(np.median(near_response))
    random_median = float(np.median(random_medians))
    ratio = near_median / random_median
    lower_tail_p = float((1 + np.sum(random_medians <= near_median)) / (len(random_medians) + 1))
    replicate_reliability = correlation(response_a, response_b)
    enough_for_modeling = len(lines) >= coupling["minimum_complete_LCLs_for_decoupled_modeling"]
    coupling_not_sufficient = bool(
        replicate_reliability > 0 and distance_spearman <= 0.20
    )

    pairs = pd.DataFrame({
        "line": lines,
        "nearest_line_by_ETOH_A": [lines[index] for index in nearest],
        "baseline_A_distance": baseline_distance[np.arange(len(lines)), nearest],
        "response_B_distance": near_response,
    })
    pairs.to_csv(args.output / "MEASUREMENT_DECOUPLED_NEAREST_PAIRS.csv", index=False)
    audit = {
        "complete_measurement_disjoint_LCLs": len(lines),
        "minimum_for_modeling": coupling["minimum_complete_LCLs_for_decoupled_modeling"],
        "decoupled_modeling_allowed": enough_for_modeling,
        "diagnostic_only": not enough_for_modeling,
        "response_A_B_replicate_correlation": replicate_reliability,
        "baseline_A_response_B_distance_Spearman": distance_spearman,
        "nearest_pair_median_response_B_distance": near_median,
        "random_pair_median_response_B_distance": random_median,
        "nearest_to_random_response_distance_ratio": ratio,
        "nearest_pair_lower_tail_permutation_p": lower_tail_p,
        "coupling_not_sufficient_to_explain_primary_failure": coupling_not_sufficient,
        "verdict": "MEASUREMENT_DECOUPLED_DIAGNOSTIC_COMPLETE",
    }
    (args.output / "MEASUREMENT_COUPLING_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Measurement Coupling Audit

## Eligibility

- Fully measurement-disjoint LCLs with two matched ETOH and DEX versions: **{len(lines)}**.
- Frozen minimum for decoupled model training: {coupling['minimum_complete_LCLs_for_decoupled_modeling']}.
- Decoupled ML allowed: **{enough_for_modeling}**.

The cohort is below the frozen modeling minimum, so no low-powered decoupled
Ridge/MLP result is presented as primary evidence. The analysis is restricted to
a measurement-decoupled diagnostic: ETOH replicate A defines baseline geometry,
while DEX(B)-ETOH(B) defines response geometry.

## Diagnostic

- Independent response A/B replicate correlation: {replicate_reliability:.4f}.
- Baseline-A vs response-B pair-distance Spearman: {distance_spearman:.4f}.
- Baseline-nearest-pair median response-B distance: {near_median:.4f}.
- Random-pair median response-B distance: {random_median:.4f}.
- Nearest/random response-distance ratio: {ratio:.4f}.
- One-sided random-pair lower-tail p-value: {lower_tail_p:.4g}.
- Frozen coupling-not-sufficient diagnostic gate: **{coupling_not_sufficient}**.

This diagnostic does not replace the primary OOF residual R2 and is not a
Level-3 RNA-equivalence test.

`MEASUREMENT_DECOUPLED_DIAGNOSTIC_COMPLETE`
"""
    (args.output / "MEASUREMENT_COUPLING_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
