"""Replicate truth and context-support scaling for CGC-SUPPORT-0C."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_tahoe_0c.extraction import write_json


TRUTH_PERMUTATIONS = 5_000
TRUTH_BOOTSTRAPS = 10_000
TRUTH_PERMUTATION_SEED = 202608201
TRUTH_BOOTSTRAP_SEED = 202608202


def load_core(root: Path) -> tuple[np.ndarray, dict[str, list[str]]]:
    manifest = json.loads(
        (root / "results/cgc_tahoe_0c/core_tensor_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    delta = np.load(root / manifest["tensor_path"]).astype(np.float64)
    axes = json.loads((root / manifest["axes_path"]).read_text(encoding="utf-8"))
    if delta.shape != (2, 50, 93, manifest["readout_gene_count"]):
        raise RuntimeError("TAHOE_CORE_EXTRACTION_INVALID")
    return delta, axes


def interaction_tensor(delta: np.ndarray) -> np.ndarray:
    return (
        delta
        - delta.mean(axis=1, keepdims=True)
        - delta.mean(axis=2, keepdims=True)
        + delta.mean(axis=(1, 2), keepdims=True)
    )


def _bootstrap_signal(
    condition_signal: np.ndarray, draws: int, seed: int
) -> np.ndarray:
    contexts, interventions = condition_signal.shape
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        batch = stop - start
        sampled_contexts = rng.integers(0, contexts, size=(batch, contexts))
        sampled_interventions = rng.integers(
            0, interventions, size=(batch, contexts, interventions)
        )
        expanded_contexts = np.broadcast_to(
            sampled_contexts[:, :, None], sampled_interventions.shape
        )
        values[start:stop] = condition_signal[
            expanded_contexts, sampled_interventions
        ].mean(axis=(1, 2))
    return values


def run_truth_gate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    delta, axes = load_core(root)
    gamma = interaction_tensor(delta)
    plate6, plate14 = gamma
    dimensions = plate6.size
    dot = float(np.sum(plate6 * plate14, dtype=np.float64))
    norm6 = float(np.sum(plate6 * plate6, dtype=np.float64))
    norm14 = float(np.sum(plate14 * plate14, dtype=np.float64))
    cosine = dot / np.sqrt(norm6 * norm14)
    pearson = float(np.corrcoef(plate6.ravel(), plate14.ravel())[0, 1])
    signal_energy = dot / dimensions
    observed_energy = (norm6 + norm14) / (2 * dimensions)
    signal_fraction = signal_energy / observed_energy

    condition_signal = np.mean(plate6 * plate14, axis=2)
    context_signal = condition_signal.mean(axis=1)
    bootstrap = _bootstrap_signal(
        condition_signal, TRUTH_BOOTSTRAPS, TRUTH_BOOTSTRAP_SEED
    )
    bootstrap_lower, bootstrap_upper = np.quantile(bootstrap, [0.025, 0.975])

    cross_gram = np.einsum(
        "cpg,cqg->cpq", plate6, plate14, optimize=True
    ) / plate6.shape[2]
    rng = np.random.default_rng(TRUTH_PERMUTATION_SEED)
    null_signal = np.empty(TRUTH_PERMUTATIONS, dtype=np.float64)
    row_index = np.arange(plate6.shape[1])
    for draw in range(TRUTH_PERMUTATIONS):
        total = 0.0
        for context in range(plate6.shape[0]):
            permutation = rng.permutation(plate6.shape[1])
            total += cross_gram[context, row_index, permutation].sum()
        null_signal[draw] = total / (plate6.shape[0] * plate6.shape[1])
    null_cosine = null_signal * dimensions / np.sqrt(norm6 * norm14)
    null_pearson = null_cosine.copy()
    null_q95 = float(np.quantile(null_signal, 0.95))
    empirical_p = float((1 + np.count_nonzero(null_signal >= signal_energy)) / (1 + len(null_signal)))

    reliability_rows = [
        {
            "scope": "global",
            "context_id": "ALL",
            "context_name": "ALL",
            "operator_cosine": cosine,
            "pearson": pearson,
            "signal_energy": signal_energy,
            "observed_energy": observed_energy,
            "signal_fraction": signal_fraction,
            "positive_signal": signal_energy > 0,
            "bootstrap_lower_95": float(bootstrap_lower),
            "bootstrap_upper_95": float(bootstrap_upper),
            "shuffle_q95": null_q95,
            "empirical_p": empirical_p,
        }
    ]
    names = (
        pd.read_csv(root / "results/cgc_tahoe_0b/cell_line_manifest.csv")
        .set_index("cell_line_id")["cell_name"]
        .to_dict()
    )
    for index, context in enumerate(axes["contexts"]):
        a = plate6[index].ravel()
        b = plate14[index].ravel()
        reliability_rows.append(
            {
                "scope": "context",
                "context_id": context,
                "context_name": names.get(context, context),
                "operator_cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                "pearson": float(np.corrcoef(a, b)[0, 1]),
                "signal_energy": float(context_signal[index]),
                "observed_energy": float((np.mean(a * a) + np.mean(b * b)) / 2),
                "signal_fraction": float(
                    context_signal[index] / ((np.mean(a * a) + np.mean(b * b)) / 2)
                ),
                "positive_signal": bool(context_signal[index] > 0),
                "bootstrap_lower_95": np.nan,
                "bootstrap_upper_95": np.nan,
                "shuffle_q95": np.nan,
                "empirical_p": np.nan,
            }
        )
    pd.DataFrame(reliability_rows).to_csv(
        result_dir / "truth_replicate_reliability.csv", index=False
    )
    pd.DataFrame(
        {
            "permutation": np.arange(TRUTH_PERMUTATIONS),
            "seed": TRUTH_PERMUTATION_SEED,
            "operator_cosine": null_cosine,
            "pearson": null_pearson,
            "signal_energy": null_signal,
        }
    ).to_csv(result_dir / "truth_shuffle_null.csv", index=False)

    positive_contexts = int(np.count_nonzero(context_signal > 0))
    gates = {
        "signal_energy_positive": bool(signal_energy > 0),
        "bootstrap_lower_positive": bool(bootstrap_lower > 0),
        "observed_above_shuffle_q95": bool(signal_energy > null_q95),
        "empirical_p_below_0_05": bool(empirical_p < 0.05),
        "at_least_40_positive_contexts": bool(positive_contexts >= 40),
    }
    passed = all(gates.values())
    label = (
        "TAHOE_REPLICATE_STABLE_CONTEXT_OPERATOR_CONFIRMED"
        if passed
        else "TAHOE_REPLICATE_STABLE_CONTEXT_OPERATOR_NOT_CONFIRMED"
    )
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "truth_label": label,
        "continue_support_scaling": passed,
        "readout_genes": len(axes["genes"]),
        "operator_cosine": cosine,
        "pearson": pearson,
        "signal_energy": signal_energy,
        "observed_energy": observed_energy,
        "signal_fraction": signal_fraction,
        "bootstrap_lower_95": float(bootstrap_lower),
        "bootstrap_upper_95": float(bootstrap_upper),
        "shuffle_q95": null_q95,
        "empirical_p": empirical_p,
        "positive_contexts": positive_contexts,
        "total_contexts": len(axes["contexts"]),
        "gates": gates,
        "permutations": TRUTH_PERMUTATIONS,
        "bootstrap_draws": TRUTH_BOOTSTRAPS,
    }
    write_json(result_dir / "truth_gate.json", result)
    verdict = {
        "phase": "CGC-SUPPORT-0C",
        "status": "truth_gate_passed" if passed else "stopped_at_truth_gate",
        "truth_label": label,
        "support_scaling_run": False,
    }
    write_json(result_dir / "verdict.json", verdict)
    return result
