"""Replicate-calibrated full-transcriptome Tahoe truth operator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr


UNIVERSES = (
    "G_ALL_MAPPED",
    "G_PRIMARY",
    "G_BROAD",
    "G_STRICT",
    "G_PRIMARY_NO_MITO",
    "G_PRIMARY_NO_MITO_RIBO",
)
GENE_CHUNK = 512
PERMUTATIONS = 5_000
BOOTSTRAPS = 10_000
PERMUTATION_SEED = 2_602
BOOTSTRAP_SEED = 2_603


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def interaction(values: np.ndarray) -> np.ndarray:
    """Two-way center context and intervention axes independently per plate/gene."""

    return (
        values
        - values.mean(axis=1, keepdims=True, dtype=np.float64)
        - values.mean(axis=2, keepdims=True, dtype=np.float64)
        + values.mean(axis=(1, 2), keepdims=True, dtype=np.float64)
    )


@dataclass
class _Accumulator:
    gene_count: int = 0
    dot: float = 0.0
    norm6: float = 0.0
    norm14: float = 0.0
    sum6: float = 0.0
    sum14: float = 0.0
    context_dot: np.ndarray | None = None
    context_norm6: np.ndarray | None = None
    context_norm14: np.ndarray | None = None
    context_sum6: np.ndarray | None = None
    context_sum14: np.ndarray | None = None
    condition_cross: np.ndarray | None = None
    cross_gram: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.context_dot = np.zeros(50, dtype=np.float64)
        self.context_norm6 = np.zeros(50, dtype=np.float64)
        self.context_norm14 = np.zeros(50, dtype=np.float64)
        self.context_sum6 = np.zeros(50, dtype=np.float64)
        self.context_sum14 = np.zeros(50, dtype=np.float64)
        self.condition_cross = np.zeros((50, 93), dtype=np.float64)
        self.cross_gram = np.zeros((50, 93, 93), dtype=np.float64)

    def update(self, gamma: np.ndarray) -> None:
        a, b = gamma
        self.gene_count += gamma.shape[-1]
        self.dot += float(np.sum(a * b, dtype=np.float64))
        self.norm6 += float(np.sum(a * a, dtype=np.float64))
        self.norm14 += float(np.sum(b * b, dtype=np.float64))
        self.sum6 += float(np.sum(a, dtype=np.float64))
        self.sum14 += float(np.sum(b, dtype=np.float64))
        self.context_dot += np.sum(a * b, axis=(1, 2), dtype=np.float64)
        self.context_norm6 += np.sum(a * a, axis=(1, 2), dtype=np.float64)
        self.context_norm14 += np.sum(b * b, axis=(1, 2), dtype=np.float64)
        self.context_sum6 += np.sum(a, axis=(1, 2), dtype=np.float64)
        self.context_sum14 += np.sum(b, axis=(1, 2), dtype=np.float64)
        self.condition_cross += np.sum(a * b, axis=2, dtype=np.float64)
        self.cross_gram += np.einsum("cpg,cqg->cpq", a, b, optimize=True)


def _bootstrap(condition_signal: np.ndarray) -> np.ndarray:
    contexts, interventions = condition_signal.shape
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 250):
        stop = min(BOOTSTRAPS, start + 250)
        batch = stop - start
        sampled_contexts = rng.integers(0, contexts, size=(batch, contexts))
        sampled_interventions = rng.integers(0, interventions, size=(batch, contexts, interventions))
        expanded = np.broadcast_to(sampled_contexts[:, :, None], sampled_interventions.shape)
        values[start:stop] = condition_signal[expanded, sampled_interventions].mean(axis=(1, 2))
    return values


def _permutation_null(cross_gram: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(PERMUTATION_SEED)
    values = np.empty(PERMUTATIONS, dtype=np.float64)
    row = np.arange(93)[None, None, :]
    context = np.arange(50)[None, :, None]
    for start in range(0, PERMUTATIONS, 100):
        stop = min(PERMUTATIONS, start + 100)
        permutations = np.argsort(rng.random((stop - start, 50, 93)), axis=2)
        values[start:stop] = cross_gram[context, row, permutations].mean(axis=(1, 2))
    return values


def run_truth(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    response = zarr.open_group(out / "response_tensors.zarr", mode="r")["delta_primary"]
    masks = {
        universe: np.load(out / f"gene_indices_{universe.lower()}.npy")
        for universe in UNIVERSES
    }
    mask_flags = {
        universe: np.zeros(response.shape[-1], dtype=bool) for universe in UNIVERSES
    }
    for universe in UNIVERSES:
        mask_flags[universe][masks[universe]] = True
    accumulators = {universe: _Accumulator() for universe in UNIVERSES}
    for start in range(0, response.shape[-1], GENE_CHUNK):
        stop = min(response.shape[-1], start + GENE_CHUNK)
        block = np.asarray(response[:, :, :, start:stop], dtype=np.float64)
        for universe in UNIVERSES:
            local = mask_flags[universe][start:stop]
            if local.any():
                accumulators[universe].update(interaction(block[..., local]))
        print(f"truth genes {stop}/{response.shape[-1]}", flush=True)

    reliability_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    context_names = (
        pd.read_csv(root / "results/cgc_tahoe_0b/cell_line_manifest.csv")
        .sort_values("cell_line_id")
        .reset_index(drop=True)
    )
    for universe, acc in accumulators.items():
        if acc.gene_count != len(masks[universe]):
            raise RuntimeError(f"{universe}: gene accounting mismatch")
        dimensions = 50 * 93 * acc.gene_count
        mean6 = acc.sum6 / dimensions
        mean14 = acc.sum14 / dimensions
        covariance = acc.dot - dimensions * mean6 * mean14
        variance6 = acc.norm6 - dimensions * mean6 * mean6
        variance14 = acc.norm14 - dimensions * mean14 * mean14
        cosine = acc.dot / np.sqrt(acc.norm6 * acc.norm14)
        pearson = covariance / np.sqrt(variance6 * variance14)
        signal = acc.dot / dimensions
        observed = (acc.norm6 + acc.norm14) / (2 * dimensions)
        condition_signal = acc.condition_cross / acc.gene_count
        context_signal = condition_signal.mean(axis=1)
        bootstrap = _bootstrap(condition_signal)
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        null = _permutation_null(acc.cross_gram / acc.gene_count)
        null_q95 = float(np.quantile(null, 0.95))
        empirical_p = float((1 + np.count_nonzero(null >= signal)) / (1 + len(null)))
        positive = int(np.count_nonzero(context_signal > 0))
        gates = {
            "signal_energy_positive": bool(signal > 0),
            "bootstrap_lower_positive": bool(ci_low > 0),
            "observed_above_shuffle_q95": bool(signal > null_q95),
            "empirical_p_below_0_05": bool(empirical_p < 0.05),
            "at_least_40_positive_contexts": bool(positive >= 40),
        }
        summaries[universe] = {
            "gene_count": acc.gene_count,
            "cosine": float(cosine),
            "pearson": float(pearson),
            "signal_energy": float(signal),
            "observed_energy": float(observed),
            "signal_fraction": float(signal / observed),
            "bootstrap_lower_95": float(ci_low),
            "bootstrap_upper_95": float(ci_high),
            "shuffle_q95": null_q95,
            "empirical_p": empirical_p,
            "positive_contexts": positive,
            "gates": gates,
            "passed": all(gates.values()),
        }
        reliability_rows.append({"universe": universe, "scope": "global", "context_id": "ALL", **summaries[universe]})
        for context in range(50):
            n = 93 * acc.gene_count
            m6 = acc.context_sum6[context] / n
            m14 = acc.context_sum14[context] / n
            cov = acc.context_dot[context] - n * m6 * m14
            v6 = acc.context_norm6[context] - n * m6 * m6
            v14 = acc.context_norm14[context] - n * m14 * m14
            reliability_rows.append(
                {
                    "universe": universe,
                    "scope": "context",
                    "context_id": str(context_names.iloc[context]["cell_line_id"]),
                    "context_name": str(context_names.iloc[context]["cell_name"]),
                    "gene_count": acc.gene_count,
                    "cosine": float(acc.context_dot[context] / np.sqrt(acc.context_norm6[context] * acc.context_norm14[context])),
                    "pearson": float(cov / np.sqrt(v6 * v14)),
                    "signal_energy": float(context_signal[context]),
                    "observed_energy": float((acc.context_norm6[context] + acc.context_norm14[context]) / (2 * n)),
                    "signal_fraction": float(context_signal[context] / ((acc.context_norm6[context] + acc.context_norm14[context]) / (2 * n))),
                    "positive_context": bool(context_signal[context] > 0),
                }
            )
        null_rows.extend(
            {
                "universe": universe,
                "permutation": index,
                "seed": PERMUTATION_SEED,
                "signal_energy": float(value),
            }
            for index, value in enumerate(null)
        )
        bootstrap_rows.append(
            {
                "analysis": "truth_signal_energy",
                "universe": universe,
                "draws": BOOTSTRAPS,
                "seed": BOOTSTRAP_SEED,
                "estimate": float(signal),
                "lower_95": float(ci_low),
                "upper_95": float(ci_high),
            }
        )
    pd.DataFrame(reliability_rows).to_csv(out / "truth_reliability.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out / "truth_null.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "bootstrap_summary.csv", index=False)
    primary = summaries["G_PRIMARY"]
    result = {
        "created_at": _now(),
        "primary_universe": "G_PRIMARY",
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "hierarchical_bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "universes": summaries,
        "primary_truth_passed": primary["passed"],
        "truth_label": "TAHOE_FULL_TRANSCRIPTOME_OPERATOR_RELIABLE" if primary["passed"] else "TAHOE_FULL_TRANSCRIPTOME_OPERATOR_NOT_RELIABLE",
        "support_scaling_authorized": bool(primary["passed"]),
    }
    _write_json(out / "truth_gate.json", result)
    report = f"""# CGC-0I full-transcriptome truth gate

- Primary universe: G_PRIMARY ({primary['gene_count']:,} genes).
- Cross-plate cosine: {primary['cosine']:.6f}; Pearson: {primary['pearson']:.6f}.
- Signal energy: {primary['signal_energy']:.8g}; observed energy: {primary['observed_energy']:.8g}; signal fraction: {primary['signal_fraction']:.4f}.
- Hierarchical bootstrap 95% CI: [{primary['bootstrap_lower_95']:.8g}, {primary['bootstrap_upper_95']:.8g}].
- Intervention-identity null q95: {primary['shuffle_q95']:.8g}; empirical p: {primary['empirical_p']:.6g}.
- Positive contexts: {primary['positive_contexts']}/50.
- Gate: **{'PASS' if primary['passed'] else 'FAIL'}** — `{result['truth_label']}`.

Plate 6 and Plate 14 were kept separate throughout; metrics use the replicate-paired cross-plate operator.
"""
    (root / "results/reports/cgc_tahoe_0i_fulltranscriptome_truth.md").write_text(report, encoding="utf-8")
    return result

