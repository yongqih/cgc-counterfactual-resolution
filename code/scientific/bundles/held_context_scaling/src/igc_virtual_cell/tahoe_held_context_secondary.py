"""Secondary domain-blocked and synthetic controls for Tahoe scaling.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.tahoe_held_context_scaling import (
    DESIGN_SEED,
    M_GRID,
    FrozenInputs,
    ResponseGrams,
    _stable_seed,
    evaluate_episode,
    fit_episode,
    load_frozen_inputs,
    load_response_grams,
    output_dir,
    prepare_design,
    run_episodes,
)


DOMAIN_LADDERS = 10
MIN_DOMAIN_SIZE = 3
CONTROL_GENES = 128
CONTROL_SEED = 202608284


_DOMAIN_STATE: tuple[FrozenInputs, ResponseGrams] | None = None


def _initialize_domain_worker(frozen: FrozenInputs, grams: ResponseGrams) -> None:
    global _DOMAIN_STATE
    _DOMAIN_STATE = frozen, grams


def _domain_target(target: int) -> list[dict[str, Any]]:
    if _DOMAIN_STATE is None:
        raise RuntimeError("Domain worker is not initialized")
    frozen, grams = _DOMAIN_STATE
    target_lineage = frozen.lineages[target]
    allowed = np.asarray(
        [index for index, lineage in enumerate(frozen.lineages) if lineage != target_lineage],
        dtype=np.int32,
    )
    maximal_m = len(allowed)
    m_values = [m for m in M_GRID if m < maximal_m]
    if maximal_m not in m_values:
        m_values.append(maximal_m)
    permutations = [
        np.random.default_rng(
            _stable_seed(DESIGN_SEED, "domain", target_lineage, frozen.contexts[target], ladder)
        ).permutation(allowed)
        for ladder in range(DOMAIN_LADDERS)
    ]
    rows: list[dict[str, Any]] = []
    for m in m_values:
        ladder_ids: list[int | str] = ["domain_max"] if m == maximal_m else list(range(DOMAIN_LADDERS))
        for ladder in ladder_ids:
            support = allowed if ladder == "domain_max" else permutations[int(ladder)][:m]
            weights6, weights14, parameters = fit_episode(
                frozen, grams, target, support, "rbf_ridge"
            )
            values = evaluate_episode(grams, target, support, weights6, weights14, allowed)
            reference = float(values["reference"].sum())
            error = float(values["model_error"].sum())
            fixed = float(values["fixed_reference"].sum())
            rows.append(
                {
                    "target_index": target,
                    "target_context": frozen.contexts[target],
                    "heldout_domain": target_lineage,
                    "heldout_domain_size": sum(value == target_lineage for value in frozen.lineages),
                    "allowed_training_contexts": maximal_m,
                    "m": m,
                    "ladder_id": ladder,
                    "model": "rbf_ridge",
                    "reference_cross_energy": reference,
                    "model_error_cross_energy": error,
                    "fixed_reference_cross_energy": fixed,
                    "g": 1 - error / reference if reference > 0 else np.nan,
                    "g_fixed": 1 - error / fixed if fixed > 0 else np.nan,
                    "alpha": parameters["alpha"],
                    "gamma_factor": parameters["gamma_factor"],
                    "inner_objective": parameters["inner_objective"],
                    "all_target_domain_responses_sealed": True,
                    "target_treated_outcomes_used_in_fit": False,
                    "domain_definition_uses_outcomes": False,
                }
            )
    return rows


def run_domain_blocked(
    root: Path,
    frozen: FrozenInputs,
    grams: ResponseGrams,
    workers: int = 6,
) -> pd.DataFrame:
    counts = pd.Series(frozen.lineages).value_counts()
    adequate = set(counts[counts >= MIN_DOMAIN_SIZE].index)
    targets = [index for index, lineage in enumerate(frozen.lineages) if lineage in adequate]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_domain_worker,
        initargs=(frozen, grams),
    ) as executor:
        futures = {executor.submit(_domain_target, target): target for target in targets}
        for position, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"domain-blocked targets completed {position}/{len(targets)}", flush=True)
    raw = pd.DataFrame(rows).sort_values(
        ["heldout_domain", "target_index", "m", "ladder_id"], key=lambda col: col.astype(str)
    )
    raw.to_csv(output_dir(root) / "TAHOE_DOMAIN_BLOCKED_SCALING_RAW.csv", index=False)
    summary_rows = []
    for (domain, m), group in raw.groupby(["heldout_domain", "m"], sort=True):
        summary_rows.append(
            {
                "heldout_domain": domain,
                "m": int(m),
                "target_contexts": group["target_context"].nunique(),
                "ladders_per_nonmax_target": DOMAIN_LADDERS,
                "reference_cross_energy": group["reference_cross_energy"].sum(),
                "model_error_cross_energy": group["model_error_cross_energy"].sum(),
                "pooled_g": 1
                - group["model_error_cross_energy"].sum()
                / group["reference_cross_energy"].sum(),
                "median_episode_g": group["g"].median(),
                "fraction_episode_g_positive": float((group["g"] > 0).mean()),
                "domain_outcomes_in_training": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir(root) / "TAHOE_DOMAIN_BLOCKED_SCALING.csv", index=False)
    return summary


def _grams_from_tensor(values: np.ndarray) -> ResponseGrams:
    same6 = np.einsum("cpg,dpg->pcd", values[0], values[0], optimize=True)
    same14 = np.einsum("cpg,dpg->pcd", values[1], values[1], optimize=True)
    cross = np.einsum("cpg,dpg->pcd", values[0], values[1], optimize=True)
    return ResponseGrams(
        same6=same6,
        same14=same14,
        cross=cross,
        gene_count=values.shape[-1],
        intervention_count=values.shape[2],
    )


def synthetic_grams(
    frozen: FrozenInputs,
    observed: ResponseGrams,
    positive: bool,
) -> ResponseGrams:
    rng = np.random.default_rng(CONTROL_SEED + int(not positive))
    mean_kernel = (frozen.baseline_kernels[0] + frozen.baseline_kernels[1]) / 2
    eigenvalues, eigenvectors = np.linalg.eigh((mean_kernel + mean_kernel.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    baseline_latent = eigenvectors[:, order[:6]] * np.sqrt(
        np.maximum(eigenvalues[order[:6]], 0.0)
    )
    baseline_latent = (baseline_latent - baseline_latent.mean(axis=0)) / np.maximum(
        baseline_latent.std(axis=0), 1e-12
    )
    if positive:
        context_latent = baseline_latent
    else:
        context_latent = rng.normal(size=baseline_latent.shape)
        context_latent = (context_latent - context_latent.mean(axis=0)) / context_latent.std(axis=0)
    loadings = rng.normal(size=(93, context_latent.shape[1], CONTROL_GENES))
    true_context = np.einsum("cd,pdg->cpg", context_latent, loadings, optimize=True)
    true_context /= math.sqrt(context_latent.shape[1])
    shared = rng.normal(size=(93, CONTROL_GENES))
    noise_sd = 0.75
    values = np.empty((2, 50, 93, CONTROL_GENES), dtype=np.float64)
    for plate in range(2):
        values[plate] = shared[None] + true_context + noise_sd * rng.normal(size=true_context.shape)
    observed_scale = math.sqrt(
        max(float(np.trace(observed.total_same6 + observed.total_same14)) / (2 * 50 * 93 * observed.gene_count), 1e-12)
    )
    values *= observed_scale / max(float(values.std()), 1e-12)
    return _grams_from_tensor(values)


def run_controls(
    root: Path,
    frozen: FrozenInputs,
    observed: ResponseGrams,
    workers: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, ladders = prepare_design(root, frozen)
    synthetic_frozen = FrozenInputs(
        contexts=frozen.contexts,
        interventions=frozen.interventions,
        gene_indices=np.arange(CONTROL_GENES, dtype=np.int32),
        baselines=frozen.baselines,
        baseline_kernels=frozen.baseline_kernels,
        lineages=frozen.lineages,
        response_store=frozen.response_store,
        source_root=frozen.source_root,
    )
    outputs = []
    for label, positive in (("POSITIVE_CONTROL", True), ("NULL_CONTROL", False)):
        grams = synthetic_grams(frozen, observed, positive)
        raw, _ = run_episodes(
            root,
            synthetic_frozen,
            grams,
            ladders,
            targets=range(50),
            m_grid=M_GRID,
            models=("rbf_ridge",),
            workers=workers,
            output_prefix=f"TAHOE_HELD_CONTEXT_SCALING_{label}_RAW",
        )
        rows = []
        for m, group in raw.groupby("m", sort=True):
            rows.append(
                {
                    "control": label.lower(),
                    "m": int(m),
                    "pooled_g": 1
                    - group["model_error_cross_energy"].sum()
                    / group["reference_cross_energy"].sum(),
                    "median_episode_g": group["g"].median(),
                    "fraction_episode_g_positive": float((group["g"] > 0).mean()),
                    "contexts": group["target_context"].nunique(),
                    "ladders": 1 if int(m) == 49 else 20,
                    "synthetic_response_genes": CONTROL_GENES,
                    "real_interventions": 93,
                    "real_baseline_geometry": True,
                    "two_independent_synthetic_replicates": True,
                    "construction_tuned_to_plateau": False,
                }
            )
        summary = pd.DataFrame(rows)
        path = output_dir(root) / f"TAHOE_HELD_CONTEXT_SCALING_{label}.csv"
        summary.to_csv(path, index=False)
        outputs.append(summary)
    return outputs[0], outputs[1]


def run_secondary(root: Path, source_root: Path, command: str) -> None:
    frozen = load_frozen_inputs(root, source_root)
    cache = output_dir(root) / "_cache/TAHOE_PRIMARY_RESPONSE_GRAMS.npz"
    grams = load_response_grams(cache)
    if command in {"domain", "all"}:
        run_domain_blocked(root, frozen, grams)
    if command in {"controls", "all"}:
        run_controls(root, frozen, grams)

