from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_entrywise.analysis import GENE_COUNT
from igc_virtual_cell.cgc_entrywise.estimators import fit_batches, sentinel_matrix
from igc_virtual_cell.cgc_entrywise.foundation import load_foundation
from igc_virtual_cell.cgc_entrywise.inference import BOOTSTRAPS, BOOTSTRAP_SEED, _hierarchical_weights
from igc_virtual_cell.cgc_entrywise.metrics import excess_geometry, matched_context_gram, matched_residual_cross, metric_arrays


SYNTHETIC_SEED = 202_608_224
SYNTHETIC_GENES = 512
SYNTHETIC_RANK = 4
ANCHORS = ((1, 0), (4, 4), (8, 8), (16, 16), (32, 32), (49, 92))


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _load_split(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )


def _support_and_sentinel(split: dict[str, Any], target: int) -> tuple[np.ndarray, np.ndarray]:
    contexts = {value: index for index, value in enumerate(split["contexts"])}
    interventions = {value: index for index, value in enumerate(split["interventions"])}
    context = split["contexts"][target]
    support = np.asarray(
        [
            [contexts[value] for value in split["context_support_orders"][context][str(sequence)]["order"]]
            for sequence in range(8)
        ],
        dtype=np.int64,
    )
    sentinel = np.asarray(
        [
            [interventions[value] for value in split["sentinel_orders"][context][str(sequence)]["full_order"]]
            for sequence in range(8)
        ],
        dtype=np.int64,
    )
    return support, sentinel


def _observed_scales(root: Path) -> dict[str, float]:
    same6, same14, cross, _, _ = load_foundation(root)
    diagonal_cross = np.diag(cross).astype(np.float64)
    diagonal_observed = (np.diag(same6).astype(np.float64) + np.diag(same14).astype(np.float64)) / 2
    full_signal = float(diagonal_cross.mean() / GENE_COUNT)
    observed = float(diagonal_observed.mean() / GENE_COUNT)
    utility = np.load(
        root / "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz",
        allow_pickle=False,
    )
    context_signal = float(np.asarray(utility["vtruth"])[-1].mean())
    return {
        "full_signal_per_gene": full_signal,
        "observed_energy_per_gene": observed,
        "context_signal_per_gene": context_signal,
        "noise_energy_per_gene": max(observed - full_signal, 0.0),
    }


def _synthetic_matrix(scales: dict[str, float]) -> np.ndarray:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    shared = rng.normal(size=(93, SYNTHETIC_GENES))
    shared -= shared.mean(axis=1, keepdims=True)
    z = rng.normal(size=(50, SYNTHETIC_RANK))
    z -= z.mean(axis=0, keepdims=True)
    loadings = rng.normal(size=(93, SYNTHETIC_RANK, SYNTHETIC_GENES))
    context = np.einsum("cr,prg->cpg", z, loadings, optimize=True)
    context -= context.mean(axis=0, keepdims=True)
    shared_target = max(scales["full_signal_per_gene"] - scales["context_signal_per_gene"], 1e-8)
    context_target = max(scales["context_signal_per_gene"], 1e-8)
    shared *= np.sqrt(shared_target / np.mean(shared * shared))
    context *= np.sqrt(context_target / np.mean(context * context))
    signal = shared[None, :, :] + context
    noise_scale = np.sqrt(max(scales["noise_energy_per_gene"], 1e-8))
    return np.stack(
        (
            signal + rng.normal(scale=noise_scale, size=signal.shape),
            signal + rng.normal(scale=noise_scale, size=signal.shape),
        )
    ).astype(np.float64)


def _gram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = values.reshape(2, 4_650, values.shape[-1])
    return flat[0] @ flat[0].T, flat[1] @ flat[1].T, flat[0] @ flat[1].T


def _real_parameters(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "results/cgc_entrywise_compression/ENTRYWISE_HYPERPARAMETERS.csv")
    return frame.set_index(["target_context_index", "plate", "m", "k"])


def _m0_full_response_control(root: Path) -> dict[str, Any]:
    """Evaluate M0 against a true intervention-shuffled full-response null.

    The context-specific null table cannot be reused here: its residual is
    defined after subtracting the correctly matched intervention support mean.
    A full-response intervention null must instead replace that support mean
    with the support mean of the frozen deranged intervention identity.
    """

    split = _load_split(root)
    _, _, cross, _, _ = load_foundation(root)
    cross4 = cross.reshape(50, 93, 50, 93)
    utility = np.load(
        root / "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz",
        allow_pickle=False,
    )
    full_truth = np.asarray(utility["full_truth"], dtype=np.float64)
    observed_residual = np.asarray(utility["vafter"], dtype=np.float64)[0, -1, 0]
    null_residual = np.zeros((50, 93), dtype=np.float64)
    intervention_index = {
        value: index for index, value in enumerate(split["interventions"])
    }
    for target, context in enumerate(split["contexts"]):
        sources = np.asarray([value for value in range(50) if value != target])
        for sequence in range(8):
            mapping = split["null_maps"][context][str(sequence)]["intervention_identity"]
            mapped = np.asarray(
                [intervention_index[mapping[value]] for value in split["interventions"]],
                dtype=np.int64,
            )
            for intervention, null_intervention in enumerate(mapped):
                truth = cross4[target, intervention, target, intervention]
                truth6_to_null14 = cross4[
                    target, intervention, sources, null_intervention
                ].mean(dtype=np.float64)
                null6_to_truth14 = cross4[
                    sources, null_intervention, target, intervention
                ].mean(dtype=np.float64)
                null_cross = cross4[:, null_intervention, :, null_intervention][
                    np.ix_(sources, sources)
                ].mean(dtype=np.float64)
                null_residual[target, intervention] += (
                    truth - truth6_to_null14 - null6_to_truth14 + null_cross
                ) / (8 * GENE_COUNT)

    observed_estimate = 1.0 - observed_residual.sum() / full_truth.sum()
    null_estimate = 1.0 - null_residual.sum() / full_truth.sum()
    observed_draws = np.empty(BOOTSTRAPS, dtype=np.float64)
    null_draws = np.empty(BOOTSTRAPS, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAPS, 25):
        stop = min(BOOTSTRAPS, start + 25)
        weights = _hierarchical_weights(rng, stop - start).astype(np.float64)
        denominator = weights @ full_truth.ravel()
        observed_draws[start:stop] = 1.0 - weights @ observed_residual.ravel() / denominator
        null_draws[start:stop] = 1.0 - weights @ null_residual.ravel() / denominator
    observed_low, observed_high = np.quantile(observed_draws, [0.025, 0.975])
    null_low, null_high = np.quantile(null_draws, [0.025, 0.975])
    return {
        "control": "SHARED_RESPONSE_RECOVERY",
        "m": 49,
        "k": 0,
        "g_full_response": float(observed_estimate),
        "observed_full_lower_95": float(observed_low),
        "observed_full_upper_95": float(observed_high),
        "intervention_null_g_full_response": float(null_estimate),
        "intervention_null_full_lower_95": float(null_low),
        "intervention_null_full_upper_95": float(null_high),
        "gate_rule": "observed_lower_95_gt_zero_and_gt_null_upper_95",
        "gate_passed": bool(observed_low > 0 and observed_low > null_high),
    }


def run_synthetic(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    split = _load_split(root)
    scales = _observed_scales(root)
    values = _synthetic_matrix(scales)
    same6, same14, cross = _gram(values)
    same4 = (same6.reshape(50, 93, 50, 93), same14.reshape(50, 93, 50, 93))
    cross4 = cross.reshape(50, 93, 50, 93)
    matched_same = tuple(matched_context_gram(value) for value in same4)
    matched_cross = matched_context_gram(cross4)
    parameters = _real_parameters(root)
    rows: list[dict[str, Any]] = []
    for m, k in ANCHORS:
        truth_sum = 0.0
        residual_sum = 0.0
        episode_count = 0
        for target in range(50):
            support_orders, sentinel_orders = _support_and_sentinel(split, target)
            for sequence in range(8):
                sequence_count = 1 if (m == 49 and k == 92) else 8
                if sequence >= sequence_count:
                    continue
                sources = support_orders[sequence, :m]
                truth = np.diag(excess_geometry(cross4, np.zeros(4_650), target, sources).gram)
                batches = []
                for plate, label in enumerate(("plate6", "plate14")):
                    ridge = float(parameters.loc[(target, label, m, k), "ridge_lambda"])
                    batches.append(
                        fit_batches(
                            matched_same[plate],
                            same4[plate],
                            target,
                            sources,
                            sentinel_orders[sequence],
                            k,
                            ridge,
                            0,
                            (target + 1) % 50,
                            np.roll(np.arange(93), -1),
                        )["M2"]
                    )
                residual = matched_residual_cross(
                    matched_cross, target, sources, batches[0], batches[1]
                )
                truth_sum += float(truth.sum() / SYNTHETIC_GENES)
                residual_sum += float(residual.sum() / SYNTHETIC_GENES)
                episode_count += 93
        rows.append(
            {
                "control": "MATCHED_SYNTHETIC_RANK4",
                "m": m,
                "k": k,
                "budget": 93 * m + k,
                "matrix_fraction": (93 * m + k) / 4_650,
                "g": 1.0 - residual_sum / truth_sum,
                "truth_sum": truth_sum,
                "residual_sum": residual_sum,
                "episodes": episode_count,
                "genes": SYNTHETIC_GENES,
                "seed": SYNTHETIC_SEED,
                **scales,
            }
        )
        print(f"synthetic positive control m={m} k={k}", flush=True)
    values_g = np.asarray([row["g"] for row in rows])
    crossed = [bool(np.any(values_g >= threshold)) for threshold in (0.25, 0.5, 0.8)]
    monotone = bool(np.all(np.diff(values_g) >= 0))
    passed = bool(monotone and all(crossed))
    for row in rows:
        row.update(
            {
                "monotone_anchor_curve": monotone,
                "crossed_25": crossed[0],
                "crossed_50": crossed[1],
                "crossed_80": crossed[2],
                "gate_passed": passed,
            }
        )
    return rows


def run_oracle(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    same6, same14, cross, sums, _ = load_foundation(root)
    same4 = (same6.reshape(50, 93, 50, 93), same14.reshape(50, 93, 50, 93))
    cross4 = cross.reshape(50, 93, 50, 93)
    rows: list[dict[str, Any]] = []
    for target in range(50):
        sources = np.asarray([value for value in range(50) if value != target])
        geometry6 = excess_geometry(same4[0], sums[0], target, sources)
        geometry14 = excess_geometry(same4[1], sums[1], target, sources)
        cross_geometry = excess_geometry(cross4, sums[0], target, sources)
        norm6 = np.diag(geometry6.gram)
        norm14 = np.diag(geometry14.gram)
        dot = np.diag(cross_geometry.gram)
        for direction in ("plate6_to_plate14", "plate14_to_plate6"):
            if direction == "plate6_to_plate14":
                truth_norm, prediction_norm = norm14, norm6
                truth_sum, prediction_sum = geometry14.excess_sum, geometry6.excess_sum
            else:
                truth_norm, prediction_norm = norm6, norm14
                truth_sum, prediction_sum = geometry6.excess_sum, geometry14.excess_sum
            metrics = metric_arrays(
                truth_norm,
                prediction_norm,
                dot,
                truth_sum,
                prediction_sum,
                GENE_COUNT,
            )
            for intervention in range(93):
                rows.append(
                    {
                        "control": "REPLICATE_ORACLE_CEILING",
                        "direction": direction,
                        "context_index": target,
                        "intervention_index": intervention,
                        **{
                            name: float(metrics[intervention, index])
                            for index, name in enumerate(
                                ("pearson", "cosine", "gene_r2", "mse", "variance_retention")
                            )
                        },
                    }
                )
    return rows


def run_positive_controls(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_entrywise_compression"
    control_a = _m0_full_response_control(root)
    synthetic = run_synthetic(root)
    oracle = run_oracle(root)
    rows = [control_a, *synthetic]
    pd.DataFrame(rows).to_csv(out / "ENTRYWISE_POSITIVE_CONTROLS.csv", index=False)
    pd.DataFrame(oracle).to_csv(out / "ENTRYWISE_REPLICATE_ORACLE.csv", index=False)
    result = {
        "created_at": _now(),
        "positive_control_a_passed": control_a["gate_passed"],
        "positive_control_b_passed": bool(synthetic[0]["gate_passed"]),
        "oracle_label": "REPLICATE_ORACLE_CEILING",
        "oracle_deployment_eligible": False,
    }
    return result
