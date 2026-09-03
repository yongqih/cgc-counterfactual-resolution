"""Frozen replicate and genetic-support Truth Gate for LEVEL2-LEA-RNA-CF-0A."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from igc_virtual_cell.level2_lea import pairing_table, truth_complete_lines


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x0 = np.asarray(x, dtype=np.float64).ravel()
    y0 = np.asarray(y, dtype=np.float64).ravel()
    x0 -= x0.mean()
    y0 -= y0.mean()
    denominator = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
    return float(np.dot(x0, y0) / denominator) if denominator > 0 else float("nan")


def row_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=1, keepdims=True)
    y0 = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x0 * x0).sum(axis=1) * (y0 * y0).sum(axis=1))
    return np.divide(
        (x0 * y0).sum(axis=1), denominator,
        out=np.full(len(x), np.nan), where=denominator > 0,
    )


def column_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return row_correlations(x.T, y.T)


def load_selected_expression(path: Path, sample_names: list[str]) -> tuple[list[str], pd.DataFrame]:
    usecols = ["GeneID", *sample_names]
    dtype = {name: np.float32 for name in sample_names}
    frame = pd.read_csv(path, sep="\t", usecols=usecols, dtype=dtype)
    genes = frame.pop("GeneID").astype(str).tolist()
    return genes, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--lfsr", type=Path, required=True)
    parser.add_argument("--posterior-means", type=Path, required=True)
    parser.add_argument("--sharing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, sep="\t")
    samples, _ = pairing_table(metadata)
    complete = truth_complete_lines(samples)
    truth_names = sorted({name for names in complete.values() for name in names})
    genes, expression = load_selected_expression(args.matrix, truth_names)

    lines = sorted(complete)
    delta_a = []
    delta_b = []
    labels = []
    for line in lines:
        etoh_a, dex_a, etoh_b, dex_b = complete[line]
        delta_a.append(expression[dex_a].to_numpy() - expression[etoh_a].to_numpy())
        delta_b.append(expression[dex_b].to_numpy() - expression[etoh_b].to_numpy())
        labels.append((etoh_a, dex_a, etoh_b, dex_b))
    delta_a = np.asarray(delta_a, dtype=np.float64)
    delta_b = np.asarray(delta_b, dtype=np.float64)
    residual_a = delta_a - delta_a.mean(axis=0, keepdims=True)
    residual_b = delta_b - delta_b.mean(axis=0, keepdims=True)

    observed = correlation(residual_a, residual_b)
    per_line = row_correlations(residual_a, residual_b)
    per_gene = column_correlations(residual_a, residual_b)
    covariance = float(np.mean(residual_a * residual_b))
    global_reliability = float(
        2 * covariance / (np.var(residual_a) + np.var(residual_b))
    )

    rng = np.random.default_rng(207049)
    bootstrap = np.empty(1000)
    null = np.empty(1000)
    for draw in range(1000):
        indices = rng.integers(0, len(lines), len(lines))
        bootstrap[draw] = correlation(residual_a[indices], residual_b[indices])
        null[draw] = correlation(residual_a, residual_b[rng.permutation(len(lines))])
    bootstrap_ci = np.quantile(bootstrap, [0.025, 0.975])
    permutation_p = float((1 + np.sum(null >= observed)) / (len(null) + 1))

    _, _, vt = np.linalg.svd(residual_a, full_matrices=False)
    n_components = min(10, len(lines) - 1)
    basis = vt[:n_components]
    scores_a = residual_a @ basis.T
    scores_b = residual_b @ basis.T
    pc_rows = []
    for component in range(n_components):
        pc_rows.append({
            "component": component + 1,
            "replicate_correlation": correlation(scores_a[:, component], scores_b[:, component]),
            "version_a_variance_fraction": float(
                np.var(scores_a[:, component]) / np.var(residual_a, axis=0).sum()
            ),
            "version_b_to_a_variance_ratio": float(
                np.var(scores_b[:, component]) / np.var(scores_a[:, component])
            ),
        })
    pd.DataFrame(pc_rows).to_csv(args.output / "TRUTH_RESPONSE_PC_RELIABILITY.csv", index=False)

    gene_table = pd.DataFrame({"gene_id": genes, "replicate_correlation": per_gene})
    gene_table.to_csv(args.output / "TRUTH_PER_GENE_RELIABILITY.csv", index=False)
    line_table = pd.DataFrame({
        "line": lines,
        "replicate_response_vector_correlation": per_line,
        "ETOH_version_a": [item[0] for item in labels],
        "DEX_version_a": [item[1] for item in labels],
        "ETOH_version_b": [item[2] for item in labels],
        "DEX_version_b": [item[3] for item in labels],
    })
    line_table.to_csv(args.output / "TRUTH_REPLICATE_LCLS.csv", index=False)

    lfsr = pd.read_csv(args.lfsr, sep="\t")
    posterior = pd.read_csv(args.posterior_means, sep="\t")
    sharing = pd.read_csv(args.sharing, sep="\t")
    if len(lfsr) != len(posterior):
        raise AssertionError("Published LFSR and posterior-mean rows are not aligned")
    sharing_indices = sharing["id"].astype(int).to_numpy() - 1
    if sharing_indices.min() < 0 or sharing_indices.max() >= len(lfsr):
        raise AssertionError("Sharing-table id falls outside the complete eQTL result tables")
    if not np.allclose(
        sharing[lfsr.columns].to_numpy(),
        lfsr.iloc[sharing_indices].to_numpy(),
        equal_nan=True,
    ):
        raise AssertionError("Sharing-table id linkage does not reproduce published LFSR values")
    beta_dex = posterior["DEX"].to_numpy()
    beta_etoh = posterior["ETOH"].to_numpy()
    sign_change = np.sign(beta_dex) != np.sign(beta_etoh)
    twofold = (np.abs(beta_dex) >= 2 * np.abs(beta_etoh)) | (
        np.abs(beta_etoh) >= 2 * np.abs(beta_dex)
    )
    supported = (
        ((lfsr["DEX"] < 0.10) | (lfsr["ETOH"] < 0.10))
        & (sign_change | twofold)
    )
    supported_indices = np.flatnonzero(supported)
    annotations = sharing.set_index(sharing_indices)[["gene", "SNP"]]
    genetic = pd.DataFrame({
        "published_row_index_0based": supported_indices,
        "published_row_index_1based": supported_indices + 1,
    })
    genetic = genetic.join(annotations, on="published_row_index_0based")
    genetic = genetic.drop(columns="published_row_index_0based")
    genetic["DEX_lfsr"] = lfsr.iloc[supported_indices]["DEX"].to_numpy()
    genetic["ETOH_lfsr"] = lfsr.iloc[supported_indices]["ETOH"].to_numpy()
    genetic["DEX_posterior_mean"] = beta_dex[supported_indices]
    genetic["ETOH_posterior_mean"] = beta_etoh[supported_indices]
    genetic["sign_change"] = sign_change[supported_indices]
    genetic["twofold_or_more"] = twofold[supported_indices]
    genetic.to_csv(args.output / "TRUTH_DEX_ETOH_GENETIC_SUPPORT.csv", index=False)

    replicate_gate = bool(bootstrap_ci[0] > 0)
    genetic_gate = bool(len(genetic) > 0)
    passed = replicate_gate and genetic_gate
    verdict = (
        "INDIVIDUAL_RESPONSE_RESIDUAL_RELIABILITY_ESTABLISHED"
        if passed else "INDIVIDUAL_RESPONSE_RESIDUAL_RELIABILITY_NOT_ESTABLISHED"
    )
    audit = {
        "experiment": "LEVEL2-LEA-RNA-CF-0A",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit_at_audit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "replicate_complete_LCLs": len(lines),
        "genes": len(genes),
        "global_residual_replicate_correlation": observed,
        "global_reliability_coefficient": global_reliability,
        "bootstrap_95_CI": bootstrap_ci.tolist(),
        "permutation_p_value": permutation_p,
        "median_per_LCL_response_vector_correlation": float(np.nanmedian(per_line)),
        "median_per_gene_replicate_correlation": float(np.nanmedian(per_gene)),
        "fraction_genes_positive_replicate_correlation": float(np.nanmean(per_gene > 0)),
        "published_DEX_ETOH_context_dependent_eQTL_count": int(len(genetic)),
        "replicate_gate": replicate_gate,
        "genetic_support_gate": genetic_gate,
        "truth_gate_pass": passed,
        "verdict": verdict,
    }
    (args.output / "TRUTH_RELIABILITY.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Truth Reliability Report

## Verdict

`{verdict}`

## Replicate evidence

- Replicate-complete biological LCLs: {len(lines)}.
- Genes retained in the authors' original row structure: {len(genes):,}.
- Flattened individual-residual replicate correlation: **{observed:.4f}**.
- LCL-bootstrap 95% CI (1,000 frozen draws): **[{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]**.
- LCL-label permutation p-value (1,000 frozen draws): **{permutation_p:.4g}**.
- Global covariance reliability coefficient: {global_reliability:.4f}.
- Median per-LCL response-vector correlation: {np.nanmedian(per_line):.4f}.
- Median per-gene replicate correlation: {np.nanmedian(per_gene):.4f}; fraction positive: {np.nanmean(per_gene > 0):.3f}.
- Replicate gate (bootstrap lower bound > 0): **{replicate_gate}**.

## Independent genetic support

- Published DEX/ETOH context-dependent eQTL rows meeting the frozen rule:
  LFSR < 0.10 in DEX or ETOH and posterior effects differ by sign or at least twofold: **{len(genetic):,}**.
- Genetic support is used only for truth validation and never as a predictor.
- Genetic-support gate: **{genetic_gate}**.

## Interpretation boundary

This gate establishes whether observed individual-specific DEX residuals contain
reproducible signal; it does not establish RNA-only predictability. The analysis
uses the authors' post-SVA `voom_resid` matrix and carries the preprocessing
caveat recorded in `GSE207049_DATA_INTEGRITY_REPORT.md`.

`{verdict}`
"""
    (args.output / "TRUTH_RELIABILITY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
