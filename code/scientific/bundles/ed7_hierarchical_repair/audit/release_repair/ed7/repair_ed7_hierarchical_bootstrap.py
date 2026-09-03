"""Repair ED7 uncertainty by executing the frozen hierarchical bootstrap.

This audit-only program leaves predictions, empirical point estimates, context
ladders, model selection, scaling-law families, seeds and draw counts frozen.
It reconstructs intervention-level gene energies from the frozen response Gram
cache and saved RBF hyperparameters, then implements the preregistered
context-first / intervention-within-context bootstrap for the gene and the
predefined five-pathway readouts.

It writes only to ``audit/release_repair/ed7``.  It does not render figures or
modify manuscript files.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "audit/release_repair/ed7"
GENE_WT = REPO / ".worktrees/tahoe_held_context_scaling"
PATH_WT = REPO / ".worktrees/tahoe_five_pathway_scaling"
GENE_OUT = GENE_WT / "results/tahoe_held_context_scaling"
PATH_OUT = PATH_WT / "results/tahoe_five_pathway_scaling"

# These are the frozen values imported from the preregistered implementation.
MODEL = "rbf_ridge"
M_GRID = (2, 4, 8, 12, 16, 24, 32, 40, 49)
CONTEXTS = 50
INTERVENTIONS = 93
CURVE_DRAWS = 10_000
FIT_DRAWS = 1_000
BOOTSTRAP_SEED = 202608282
CURVE_SEED = BOOTSTRAP_SEED + sum(map(ord, MODEL))
FIT_SEED = BOOTSTRAP_SEED + 99
GENE_SELECTED_LAW = "continuing_power"
PATH_SELECTED_LAW = "exponential_sensitivity"
PATH_SENSITIVITY_LAW = "continuing_power"
TARGET_G = (0.10, 0.25, 0.50, 0.80)


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while payload := handle.read(block):
            digest.update(payload)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        )
        + "\n",
        encoding="utf-8",
    )


def load_frozen_modules() -> tuple[Any, Any, Any]:
    """Load exact gene code and pathway solver without altering either worktree."""

    sys.path.insert(0, str(GENE_WT / "src"))
    from igc_virtual_cell import tahoe_held_context_finalize as finalize  # type: ignore
    from igc_virtual_cell import tahoe_held_context_scaling as scaling  # type: ignore

    pathway_source = PATH_WT / "src/igc_virtual_cell/tahoe_five_pathway_scaling.py"
    spec = importlib.util.spec_from_file_location("_frozen_tahoe_pathway_scaling", pathway_source)
    if spec is None or spec.loader is None:
        raise RuntimeError("FROZEN_PATHWAY_MODULE_LOAD_FAILED")
    pathway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pathway)
    return scaling, finalize, pathway


def load_pathway_contributions(
    expected_interventions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = PATH_OUT / "_cache/TAHOE_FIVE_PATHWAY_SCALING_CONTRIBUTIONS.npz"
    cached = np.load(path)
    required = {"target_index", "m", "ladder_id", "interventions", "reference", "error"}
    if not required.issubset(cached.files):
        raise RuntimeError("PATHWAY_CONTRIBUTION_CACHE_INCOMPLETE")
    if tuple(cached["reference"].shape) != (8050, INTERVENTIONS):
        raise RuntimeError("PATHWAY_CONTRIBUTION_CACHE_SHAPE_CHANGED")
    pathway_interventions = np.asarray(cached["interventions"], dtype=str)
    gene_interventions = np.asarray(expected_interventions, dtype=str)
    if not np.array_equal(pathway_interventions, gene_interventions):
        raise RuntimeError("GENE_PATHWAY_INTERVENTION_AXIS_MISMATCH")
    reference, error, counts = aggregate_episodes(
        cached["target_index"], cached["m"], cached["reference"], cached["error"]
    )
    metadata = {
        "source": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "intervention_axis_sha256": hashlib.sha256(
            "\n".join(map(str, cached["interventions"])).encode("utf-8")
        ).hexdigest(),
        "exact_gene_pathway_intervention_axis_equal": True,
        "episode_counts": counts.tolist(),
    }
    return reference, error, metadata


def aggregate_episodes(
    target: np.ndarray,
    m: np.ndarray,
    reference_episode: np.ndarray,
    error_episode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.zeros((CONTEXTS, len(M_GRID), INTERVENTIONS), dtype=np.float64)
    error = np.zeros_like(reference)
    counts = np.zeros((CONTEXTS, len(M_GRID)), dtype=np.int16)
    m_to_index = {value: index for index, value in enumerate(M_GRID)}
    for row in range(len(target)):
        context_index = int(target[row])
        m_index = m_to_index[int(m[row])]
        reference[context_index, m_index] += reference_episode[row]
        error[context_index, m_index] += error_episode[row]
        counts[context_index, m_index] += 1
    expected = np.tile(np.asarray([20] * 8 + [1], dtype=np.int16), (CONTEXTS, 1))
    if not np.array_equal(counts, expected):
        raise RuntimeError("EPISODE_AGGREGATION_COUNT_MISMATCH")
    return reference, error, counts


def reconstruct_gene_contributions(scaling: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay frozen weights, but only through frozen Gram sufficient statistics."""

    started = time.perf_counter()
    cache_path = OUT / "ED7_GENE_INTERVENTION_CONTRIBUTIONS.npz"
    cache_metadata_path = OUT / "ED7_GENE_INTERVENTION_RECONSTRUCTION.json"
    if cache_path.exists() and cache_metadata_path.exists():
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if sha256(cache_path) != metadata["cache_sha256"]:
            raise RuntimeError("GENE_INTERVENTION_AUDIT_CACHE_HASH_MISMATCH")
        for path_key, hash_key in (
            ("source_grams", "source_grams_sha256"),
            ("source_ladders", "source_ladders_sha256"),
            ("source_raw", "source_raw_sha256"),
            ("source_parameters", "source_parameters_sha256"),
        ):
            source_path = REPO / metadata[path_key]
            if sha256(source_path) != metadata[hash_key]:
                raise RuntimeError(f"GENE_INTERVENTION_FROZEN_SOURCE_CHANGED:{path_key}")
        cached = np.load(cache_path)
        if tuple(cached["reference"].shape) != (CONTEXTS, len(M_GRID), INTERVENTIONS):
            raise RuntimeError("GENE_INTERVENTION_AUDIT_CACHE_SHAPE_CHANGED")
        metadata["reused_after_hash_validation"] = True
        return cached["reference"], cached["error"], metadata

    frozen = scaling.load_frozen_inputs(GENE_WT, REPO)
    grams_path = GENE_OUT / "_cache/TAHOE_PRIMARY_RESPONSE_GRAMS.npz"
    grams = scaling.load_response_grams(grams_path)
    ladders_path = GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_LADDERS.csv"
    raw_path = GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_RAW.csv"
    parameters_path = GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_RAW_PARAMETERS.csv"
    ladders = pd.read_csv(ladders_path)
    raw = pd.read_csv(raw_path)
    parameters = pd.read_csv(parameters_path)
    raw = raw.loc[raw["model"].eq(MODEL)].copy()
    parameters = parameters.loc[parameters["model"].eq(MODEL)].copy()

    def normalize_ladder(value: Any, m: int) -> str:
        return "max49" if m == 49 else str(int(value))

    raw_lookup: dict[tuple[int, int, str], Any] = {}
    for row in raw.itertuples(index=False):
        key = (int(row.target_index), int(row.m), normalize_ladder(row.ladder_id, int(row.m)))
        if key in raw_lookup:
            raise RuntimeError("GENE_RAW_EPISODE_DUPLICATE")
        raw_lookup[key] = row
    parameter_lookup: dict[tuple[int, int, str], Any] = {}
    for row in parameters.itertuples(index=False):
        key = (int(row.target_index), int(row.m), normalize_ladder(row.ladder_id, int(row.m)))
        if key in parameter_lookup:
            raise RuntimeError("GENE_PARAMETER_EPISODE_DUPLICATE")
        parameter_lookup[key] = row

    pathway_cache = np.load(PATH_OUT / "_cache/TAHOE_FIVE_PATHWAY_SCALING_CONTRIBUTIONS.npz")
    target_axis = pathway_cache["target_index"].astype(np.int16, copy=True)
    m_axis = pathway_cache["m"].astype(np.int16, copy=True)
    ladder_axis = pathway_cache["ladder_id"].astype(np.int16, copy=True)
    if len(raw_lookup) != len(target_axis) or len(parameter_lookup) != len(target_axis):
        raise RuntimeError("GENE_EPISODE_COUNT_MISMATCH")

    reference_episode = np.empty((len(target_axis), INTERVENTIONS), dtype=np.float64)
    error_episode = np.empty_like(reference_episode)
    maximum_reference_abs = 0.0
    maximum_error_abs = 0.0
    maximum_reference_rel = 0.0
    maximum_error_rel = 0.0
    maximum_weight_sum_abs = 0.0
    maximum_weight_l2_abs = 0.0
    maximal_support = [scaling._support_from_ladders(ladders, target, "max49", 49) for target in range(CONTEXTS)]

    for episode in range(len(target_axis)):
        target = int(target_axis[episode])
        m = int(m_axis[episode])
        ladder_number = int(ladder_axis[episode])
        ladder: int | str = "max49" if ladder_number == -1 else ladder_number
        key = (target, m, normalize_ladder(ladder, m))
        source = raw_lookup[key]
        tuned = parameter_lookup[key]
        support = scaling._support_from_ladders(ladders, target, ladder, m)
        alpha = float(tuned.alpha)
        gamma_factor = float(tuned.gamma_factor)
        weights = []
        for plate in range(2):
            weight = scaling._kernel_weights(
                frozen.baseline_kernels[plate],
                support,
                np.asarray([target], dtype=np.int32),
                MODEL,
                alpha,
                gamma_factor=gamma_factor,
                gamma_reference=support,
            )[0]
            weights.append(weight)
        uniform = np.full(m, 1.0 / m, dtype=np.float64)
        reference = scaling.cross_contributions(
            grams.cross, target, support, uniform, uniform
        )
        error = scaling.cross_contributions(
            grams.cross, target, support, weights[0], weights[1]
        )
        reference_episode[episode] = reference
        error_episode[episode] = error

        reference_abs = abs(float(reference.sum()) - float(source.reference_cross_energy))
        error_abs = abs(float(error.sum()) - float(source.model_error_cross_energy))
        maximum_reference_abs = max(maximum_reference_abs, reference_abs)
        maximum_error_abs = max(maximum_error_abs, error_abs)
        maximum_reference_rel = max(
            maximum_reference_rel,
            reference_abs / max(abs(float(source.reference_cross_energy)), 1e-30),
        )
        maximum_error_rel = max(
            maximum_error_rel,
            error_abs / max(abs(float(source.model_error_cross_energy)), 1e-30),
        )
        maximum_weight_sum_abs = max(
            maximum_weight_sum_abs,
            abs(float(weights[0].sum()) - float(tuned.weight_sum_plate6)),
            abs(float(weights[1].sum()) - float(tuned.weight_sum_plate14)),
        )
        maximum_weight_l2_abs = max(
            maximum_weight_l2_abs,
            abs(float(np.linalg.norm(weights[0])) - float(tuned.weight_l2_plate6)),
            abs(float(np.linalg.norm(weights[1])) - float(tuned.weight_l2_plate14)),
        )
        if (episode + 1) % 500 == 0 or episode + 1 == len(target_axis):
            print(f"reconstructed gene intervention energies {episode + 1}/{len(target_axis)}", flush=True)

    if maximum_reference_abs > 1e-8 or maximum_error_abs > 1e-8:
        raise RuntimeError("GENE_INTERVENTION_RECONSTRUCTION_FAILED")
    reference, error, counts = aggregate_episodes(
        target_axis, m_axis, reference_episode, error_episode
    )
    np.savez_compressed(
        cache_path,
        target_index=np.arange(CONTEXTS, dtype=np.int16),
        m=np.asarray(M_GRID, dtype=np.int16),
        interventions=np.asarray(frozen.interventions),
        reference=reference,
        error=error,
        episode_counts=counts,
    )
    metadata = {
        "source_grams": str(grams_path.relative_to(REPO)),
        "source_grams_sha256": sha256(grams_path),
        "source_ladders": str(ladders_path.relative_to(REPO)),
        "source_ladders_sha256": sha256(ladders_path),
        "source_raw": str(raw_path.relative_to(REPO)),
        "source_raw_sha256": sha256(raw_path),
        "source_parameters": str(parameters_path.relative_to(REPO)),
        "source_parameters_sha256": sha256(parameters_path),
        "cache": str(cache_path.relative_to(REPO)),
        "cache_sha256": sha256(cache_path),
        "episodes": len(target_axis),
        "episode_counts": counts.tolist(),
        "maximum_reference_sum_absolute_error": maximum_reference_abs,
        "maximum_error_sum_absolute_error": maximum_error_abs,
        "maximum_reference_sum_relative_error": maximum_reference_rel,
        "maximum_error_sum_relative_error": maximum_error_rel,
        "maximum_saved_weight_sum_absolute_error": maximum_weight_sum_abs,
        "maximum_saved_weight_l2_absolute_error": maximum_weight_l2_abs,
        "elapsed_seconds": time.perf_counter() - started,
        "model_refit": False,
        "prediction_regeneration": False,
    }
    write_json(OUT / "ED7_GENE_INTERVENTION_RECONSTRUCTION.json", metadata)
    return reference, error, metadata


def pooled_g(reference: np.ndarray, error: np.ndarray) -> np.ndarray:
    return 1.0 - error.sum(axis=(0, 2)) / reference.sum(axis=(0, 2))


def hierarchical_bootstrap_pair(
    gene_reference: np.ndarray,
    gene_error: np.ndarray,
    pathway_reference: np.ndarray,
    pathway_error: np.ndarray,
    draws: int,
    seed: int,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Context first; then 93 interventions within each sampled context.

    The full context-index matrix is drawn before the added intervention layer.
    This preserves exactly the context resamples generated by the previous
    context-only code under the same seed, isolating the approved repair.
    """

    rng = np.random.default_rng(seed)
    context_indices = rng.integers(0, CONTEXTS, size=(draws, CONTEXTS))
    gene_g = np.empty((draws, len(M_GRID)), dtype=np.float64)
    pathway_g = np.empty_like(gene_g)
    nonpositive = {
        "gene": np.zeros(len(M_GRID), dtype=np.int64),
        "pathway": np.zeros(len(M_GRID), dtype=np.int64),
    }
    # Flatten context x intervention and use the same hierarchical count matrix
    # for every m and both resolutions. This is algebraically identical to
    # explicit indexed gathering, but avoids nine repeated 4,650-cell gathers.
    def flatten(values: np.ndarray) -> np.ndarray:
        return values.transpose(0, 2, 1).reshape(CONTEXTS * INTERVENTIONS, len(M_GRID))

    gene_reference_flat = flatten(gene_reference)
    gene_error_flat = flatten(gene_error)
    pathway_reference_flat = flatten(pathway_reference)
    pathway_error_flat = flatten(pathway_error)
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        local_contexts = context_indices[start:stop]
        intervention_indices = rng.integers(
            0, INTERVENTIONS, size=(stop - start, CONTEXTS, INTERVENTIONS)
        )
        rows = stop - start
        combined = local_contexts[:, :, None] * INTERVENTIONS + intervention_indices
        weights = np.zeros((rows, CONTEXTS * INTERVENTIONS), dtype=np.int16)
        np.add.at(
            weights,
            (
                np.repeat(np.arange(rows, dtype=np.int32), CONTEXTS * INTERVENTIONS),
                combined.reshape(-1),
            ),
            1,
        )
        gene_reference_sum = weights @ gene_reference_flat
        gene_error_sum = weights @ gene_error_flat
        pathway_reference_sum = weights @ pathway_reference_flat
        pathway_error_sum = weights @ pathway_error_flat
        nonpositive["gene"] += np.sum(gene_reference_sum <= 0, axis=0)
        nonpositive["pathway"] += np.sum(pathway_reference_sum <= 0, axis=0)
        gene_g[start:stop] = np.where(
            gene_reference_sum > 0,
            1.0 - gene_error_sum / gene_reference_sum,
            np.nan,
        )
        pathway_g[start:stop] = np.where(
            pathway_reference_sum > 0,
            1.0 - pathway_error_sum / pathway_reference_sum,
            np.nan,
        )
    if np.any(np.isfinite(gene_g).sum(axis=0) == 0) or np.any(
        np.isfinite(pathway_g).sum(axis=0) == 0
    ):
        raise RuntimeError("HIERARCHICAL_BOOTSTRAP_NO_DEFINED_DRAWS")
    metadata = {
        "seed": seed,
        "draws": draws,
        "outer_sample_size": CONTEXTS,
        "inner_sample_size_per_sampled_context": INTERVENTIONS,
        "context_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(context_indices).view(np.uint8)
        ).hexdigest(),
        "nonpositive_reference_draws_by_m": {
            key: {str(m): int(values[index]) for index, m in enumerate(M_GRID)}
            for key, values in nonpositive.items()
        },
        "defined_fraction_by_m": {
            "gene": {
                str(m): float(np.isfinite(gene_g[:, index]).mean())
                for index, m in enumerate(M_GRID)
            },
            "pathway": {
                str(m): float(np.isfinite(pathway_g[:, index]).mean())
                for index, m in enumerate(M_GRID)
            },
        },
    }
    return gene_g, pathway_g, metadata


def curve_frame(
    gene_reference: np.ndarray,
    gene_error: np.ndarray,
    path_reference: np.ndarray,
    path_error: np.ndarray,
    gene_draws: np.ndarray,
    path_draws: np.ndarray,
) -> pd.DataFrame:
    old_gene = (
        pd.read_csv(GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_G.csv")
        .query("model == @MODEL")
        .set_index("m")
    )
    old_path = pd.read_csv(PATH_OUT / "TAHOE_FIVE_PATHWAY_SCALING_G.csv").set_index("m")
    point_gene = pooled_g(gene_reference, gene_error)
    point_path = pooled_g(path_reference, path_error)
    rows: list[dict[str, Any]] = []
    for resolution, point, draws, old in (
        ("gene", point_gene, gene_draws, old_gene),
        ("five_pathway", point_path, path_draws, old_path),
    ):
        for index, m in enumerate(M_GRID):
            old_point_column = "pooled_g" if resolution == "gene" else "pooled_g_pathway"
            lower, upper = np.nanquantile(draws[:, index], [0.025, 0.975])
            rows.append(
                {
                    "resolution": resolution,
                    "m": m,
                    "pooled_g": point[index],
                    "old_frozen_pooled_g": float(old.loc[m, old_point_column]),
                    "point_estimate_absolute_difference": abs(
                        point[index] - float(old.loc[m, old_point_column])
                    ),
                    "hierarchical_bootstrap_lower_95": lower,
                    "hierarchical_bootstrap_upper_95": upper,
                    "old_context_only_lower_95": float(old.loc[m, "bootstrap_lower_95"]),
                    "old_context_only_upper_95": float(old.loc[m, "bootstrap_upper_95"]),
                    "bootstrap_unit": "target_context_then_intervention_within_sampled_context",
                    "attempted_draws": CURVE_DRAWS,
                    "effective_draws": int(np.isfinite(draws[:, index]).sum()),
                    "seed": CURVE_SEED,
                }
            )
    result = pd.DataFrame(rows)
    if result["point_estimate_absolute_difference"].max() > 1e-12:
        raise RuntimeError("ED7_POINT_ESTIMATE_CHANGED")
    return result


def fit_bootstrap(
    finalize: Any,
    pathway_module: Any,
    gene_draws: np.ndarray,
    path_draws: np.ndarray,
    point_gene: np.ndarray,
    point_path: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = np.asarray(M_GRID, dtype=np.float64)
    # Do not numerically refit frozen point curves: a SciPy solver-version change
    # can perturb an extrapolated context count despite identical empirical
    # points. Parse the already frozen parameter strings and use the already
    # frozen extrapolation tables as authority for all reported point counts.
    gene_fits_frozen = pd.read_csv(GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_PARAMETERS.csv")
    gene_row = gene_fits_frozen[
        gene_fits_frozen["model"].eq(MODEL)
        & gene_fits_frozen["candidate"].eq(GENE_SELECTED_LAW)
    ].iloc[0]
    path_fits_frozen = pd.read_csv(PATH_OUT / "TAHOE_FIVE_PATHWAY_SCALING_PARAMETERS.csv")
    path_row = path_fits_frozen[path_fits_frozen["candidate"].eq(PATH_SELECTED_LAW)].iloc[0]
    path_power_row = path_fits_frozen[
        path_fits_frozen["candidate"].eq(PATH_SENSITIVITY_LAW)
    ].iloc[0]

    def parse_parameters(value: Any) -> np.ndarray:
        return np.asarray([float(item) for item in str(value).split(";")], dtype=np.float64)

    gene_point_fit = parse_parameters(gene_row["parameters"])
    path_point_fit = parse_parameters(path_row["parameters"])
    path_power_point_fit = parse_parameters(path_power_row["parameters"])
    gene_point_counts = (
        pd.read_csv(GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_EXTRAPOLATIONS.csv")
        .set_index("target_g")["model_based_context_count"]
        .to_dict()
    )
    path_point_counts = (
        pd.read_csv(PATH_OUT / "TAHOE_FIVE_PATHWAY_SCALING_EXTRAPOLATION.csv")
        .set_index("target_g")["point_context_count"]
        .to_dict()
    )
    comparison_frozen = pd.read_csv(PATH_OUT / "TAHOE_GENE_VS_PATHWAY_CONTEXT_REQUIREMENT.csv").iloc[0]

    gene_parameters = np.empty((FIT_DRAWS, len(gene_point_fit)), dtype=np.float64)
    path_parameters = np.empty((FIT_DRAWS, len(path_point_fit)), dtype=np.float64)
    path_power_parameters = np.empty((FIT_DRAWS, len(path_power_point_fit)), dtype=np.float64)
    gene_n = np.empty((FIT_DRAWS, len(TARGET_G)), dtype=np.float64)
    path_n = np.empty_like(gene_n)
    path_power_n = np.empty_like(gene_n)
    for draw in range(FIT_DRAWS):
        gene_parameters[draw], _ = finalize._fit(
            GENE_SELECTED_LAW, m, 1.0 - gene_draws[draw]
        )
        path_parameters[draw], _ = finalize._fit(
            PATH_SELECTED_LAW, m, 1.0 - path_draws[draw]
        )
        path_power_parameters[draw], _ = finalize._fit(
            PATH_SENSITIVITY_LAW, m, 1.0 - path_draws[draw]
        )
        for target_index, target_g in enumerate(TARGET_G):
            gene_n[draw, target_index] = pathway_module.solve_context_requirement(
                GENE_SELECTED_LAW, gene_parameters[draw], target_g
            )
            path_n[draw, target_index] = pathway_module.solve_context_requirement(
                PATH_SELECTED_LAW, path_parameters[draw], target_g
            )
            path_power_n[draw, target_index] = pathway_module.solve_context_requirement(
                PATH_SENSITIVITY_LAW, path_power_parameters[draw], target_g
            )
        if (draw + 1) % 100 == 0:
            print(f"refit frozen scaling laws {draw + 1}/{FIT_DRAWS}", flush=True)

    parameter_rows = []
    for resolution, law, point, samples, names in (
        ("gene", GENE_SELECTED_LAW, gene_point_fit, gene_parameters, ("amplitude", "alpha")),
        (
            "five_pathway",
            PATH_SELECTED_LAW,
            path_point_fit,
            path_parameters,
            ("q_inf", "amplitude", "tau"),
        ),
        (
            "five_pathway_sensitivity",
            PATH_SENSITIVITY_LAW,
            path_power_point_fit,
            path_power_parameters,
            ("amplitude", "alpha"),
        ),
    ):
        for index, name in enumerate(names):
            lower, upper = np.quantile(samples[:, index], [0.025, 0.975])
            parameter_rows.append(
                {
                    "resolution": resolution,
                    "scaling_law": law,
                    "parameter": name,
                    "point_estimate": point[index],
                    "hierarchical_bootstrap_median": np.median(samples[:, index]),
                    "hierarchical_bootstrap_lower_95": lower,
                    "hierarchical_bootstrap_upper_95": upper,
                    "draws": FIT_DRAWS,
                    "seed": FIT_SEED,
                }
            )
    # g_inf is a deterministic transform of q_inf for the selected pathway law.
    parameter_rows.append(
        {
            "resolution": "five_pathway",
            "scaling_law": PATH_SELECTED_LAW,
            "parameter": "g_inf",
            "point_estimate": float(path_row["g_inf"]),
            "hierarchical_bootstrap_median": np.median(1.0 - path_parameters[:, 0]),
            "hierarchical_bootstrap_lower_95": np.quantile(1.0 - path_parameters[:, 0], 0.025),
            "hierarchical_bootstrap_upper_95": np.quantile(1.0 - path_parameters[:, 0], 0.975),
            "draws": FIT_DRAWS,
            "seed": FIT_SEED,
        }
    )
    parameters = pd.DataFrame(parameter_rows)

    extrapolation_rows: list[dict[str, Any]] = []
    for resolution, law, point_fit, samples, frozen_counts in (
        ("gene", GENE_SELECTED_LAW, gene_point_fit, gene_n, gene_point_counts),
        ("five_pathway", PATH_SELECTED_LAW, path_point_fit, path_n, path_point_counts),
    ):
        for target_index, target_g in enumerate(TARGET_G):
            point = float(frozen_counts[target_g])
            draw_values = samples[:, target_index]
            finite = np.isfinite(draw_values) & (draw_values > 0)
            extrapolation_rows.append(
                {
                    "resolution": resolution,
                    "scaling_law": law,
                    "target_g": target_g,
                    "point_context_count": point,
                    "hierarchical_bootstrap_median_context_count": (
                        np.median(draw_values[finite]) if finite.any() else np.nan
                    ),
                    "hierarchical_bootstrap_lower_95": (
                        np.quantile(draw_values[finite], 0.025) if finite.any() else np.nan
                    ),
                    "hierarchical_bootstrap_upper_95": (
                        np.quantile(draw_values[finite], 0.975) if finite.any() else np.nan
                    ),
                    "bootstrap_reachable_fraction": finite.mean(),
                    "bootstrap_interval_scope": (
                        "all_draws"
                        if finite.all()
                        else "conditional_on_mathematically_reachable_draws_not_primary_ci"
                    ),
                    "draws": FIT_DRAWS,
                    "seed": FIT_SEED,
                }
            )
    # The continuing-power pathway result was frozen only as the g=0.50
    # model-form sensitivity. Do not broaden that sensitivity to new targets.
    target_index = TARGET_G.index(0.50)
    sensitivity_draws = path_power_n[:, target_index]
    sensitivity_finite = np.isfinite(sensitivity_draws) & (sensitivity_draws > 0)
    extrapolation_rows.append(
        {
            "resolution": "five_pathway_sensitivity",
            "scaling_law": PATH_SENSITIVITY_LAW,
            "target_g": 0.50,
            "point_context_count": float(
                comparison_frozen[
                    "continuing_power_sensitivity_pathway_context_count"
                ]
            ),
            "hierarchical_bootstrap_median_context_count": np.median(
                sensitivity_draws[sensitivity_finite]
            ),
            "hierarchical_bootstrap_lower_95": np.quantile(
                sensitivity_draws[sensitivity_finite], 0.025
            ),
            "hierarchical_bootstrap_upper_95": np.quantile(
                sensitivity_draws[sensitivity_finite], 0.975
            ),
            "bootstrap_reachable_fraction": sensitivity_finite.mean(),
            "bootstrap_interval_scope": "all_draws",
            "draws": FIT_DRAWS,
            "seed": FIT_SEED,
        }
    )
    extrapolation = pd.DataFrame(extrapolation_rows)

    difference = path_draws[:FIT_DRAWS, -1] - gene_draws[:FIT_DRAWS, -1]
    finite_path = np.isfinite(path_n[:, 2]) & (path_n[:, 2] > 0)
    fold = np.where(finite_path, gene_n[:, 2] / path_n[:, 2], np.nan)
    finite_fold = np.isfinite(fold) & (fold > 0)
    comparison = pd.DataFrame(
        [
            {
                "gene_g_m49": point_gene[-1],
                "five_pathway_g_m49": point_path[-1],
                "five_pathway_minus_gene_g_m49": point_path[-1] - point_gene[-1],
                "hierarchical_bootstrap_difference_lower_95": np.quantile(difference, 0.025),
                "hierarchical_bootstrap_difference_upper_95": np.quantile(difference, 0.975),
                "gene_g0p5_point_context_count": float(gene_point_counts[0.50]),
                "five_pathway_g0p5_selected_law_point_context_count": float(
                    path_point_counts[0.50]
                ),
                "five_pathway_g0p5_selected_law_reachable_fraction": finite_path.mean(),
                "conditional_fold_reduction_median": (
                    np.median(fold[finite_fold]) if finite_fold.any() else np.nan
                ),
                "conditional_fold_reduction_lower_95": (
                    np.quantile(fold[finite_fold], 0.025) if finite_fold.any() else np.nan
                ),
                "conditional_fold_reduction_upper_95": (
                    np.quantile(fold[finite_fold], 0.975) if finite_fold.any() else np.nan
                ),
                "fraction_reachable_draws_pathway_requires_fewer_contexts": (
                    np.mean(fold[finite_fold] > 1) if finite_fold.any() else np.nan
                ),
                "five_pathway_continuing_power_sensitivity_g0p5_point_context_count": float(
                    comparison_frozen[
                        "continuing_power_sensitivity_pathway_context_count"
                    ]
                ),
                "fit_bootstrap_draws": FIT_DRAWS,
                "fit_bootstrap_seed": FIT_SEED,
            }
        ]
    )
    return parameters, extrapolation, comparison


def make_corrected_source_data(curve: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    source = curve.rename(
        columns={
            "hierarchical_bootstrap_lower_95": "ci_low",
            "hierarchical_bootstrap_upper_95": "ci_high",
        }
    ).copy()
    source["panel"] = np.where(source["resolution"].eq("gene"), "a", "b")
    source["kind"] = "held_context_scaling"
    source["label"] = source["resolution"] + "_m" + source["m"].astype(str)
    source["value"] = source["pooled_g"]
    source["source_file"] = np.where(
        source["resolution"].eq("gene"),
        str((OUT / "ED7_HIERARCHICAL_BOOTSTRAP_CURVES.csv").relative_to(REPO)),
        str((OUT / "ED7_HIERARCHICAL_BOOTSTRAP_CURVES.csv").relative_to(REPO)),
    )
    source["frozen_contribution_provenance"] = np.where(
        source["resolution"].eq("gene"),
        str((OUT / "ED7_GENE_INTERVENTION_CONTRIBUTIONS.npz").relative_to(REPO)),
        str(
            (
                PATH_OUT / "_cache/TAHOE_FIVE_PATHWAY_SCALING_CONTRIBUTIONS.npz"
            ).relative_to(REPO)
        ),
    )
    source["note"] = (
        "point estimate unchanged; 95% percentile interval repaired to preregistered "
        "context-first/intervention-within-context hierarchical bootstrap"
    )
    columns = [
        "panel",
        "kind",
        "label",
        "value",
        "ci_low",
        "ci_high",
        "resolution",
        "m",
        "source_file",
        "frozen_contribution_provenance",
        "note",
        "attempted_draws",
        "effective_draws",
        "seed",
    ]
    result = source[columns]
    contrast = comparison.iloc[0]
    extra = pd.DataFrame(
        [
            {
                "panel": "c",
                "kind": "matched_resolution_contrast",
                "label": "five_pathway_minus_gene_m49",
                "value": contrast["five_pathway_minus_gene_g_m49"],
                "ci_low": contrast["hierarchical_bootstrap_difference_lower_95"],
                "ci_high": contrast["hierarchical_bootstrap_difference_upper_95"],
                "resolution": "paired_five_pathway_minus_gene",
                "m": 49,
                "source_file": str(
                    (OUT / "ED7_GENE_VS_PATHWAY_HIERARCHICAL.csv").relative_to(REPO)
                ),
                "frozen_contribution_provenance": (
                    "audit/release_repair/ed7/ED7_GENE_INTERVENTION_CONTRIBUTIONS.npz;"
                    ".worktrees/tahoe_five_pathway_scaling/results/tahoe_five_pathway_scaling/"
                    "_cache/TAHOE_FIVE_PATHWAY_SCALING_CONTRIBUTIONS.npz"
                ),
                "note": "paired context-first/intervention-within-context hierarchical bootstrap",
                "attempted_draws": FIT_DRAWS,
                "effective_draws": FIT_DRAWS,
                "seed": FIT_SEED,
            }
        ]
    )
    return pd.concat([result, extra], ignore_index=True)


def implementation_invariance_test(
    gene_reference: np.ndarray,
    gene_error: np.ndarray,
    path_reference: np.ndarray,
    path_error: np.ndarray,
) -> dict[str, Any]:
    """Compare the efficient count-matrix bootstrap with explicit indexing."""

    draws = 3
    efficient_gene, efficient_path, metadata = hierarchical_bootstrap_pair(
        gene_reference,
        gene_error,
        path_reference,
        path_error,
        draws,
        CURVE_SEED,
        batch_size=draws,
    )
    rng = np.random.default_rng(CURVE_SEED)
    context_indices = rng.integers(0, CONTEXTS, size=(draws, CONTEXTS))
    intervention_indices = rng.integers(
        0, INTERVENTIONS, size=(draws, CONTEXTS, INTERVENTIONS)
    )
    explicit_gene = np.empty_like(efficient_gene)
    explicit_path = np.empty_like(efficient_path)
    for draw in range(draws):
        for m_index in range(len(M_GRID)):
            sums: list[float] = []
            for reference, error in (
                (gene_reference, gene_error),
                (path_reference, path_error),
            ):
                reference_sum = 0.0
                error_sum = 0.0
                for occurrence in range(CONTEXTS):
                    context = int(context_indices[draw, occurrence])
                    selected = intervention_indices[draw, occurrence]
                    reference_sum += float(reference[context, m_index, selected].sum())
                    error_sum += float(error[context, m_index, selected].sum())
                sums.append(1.0 - error_sum / reference_sum)
            explicit_gene[draw, m_index] = sums[0]
            explicit_path[draw, m_index] = sums[1]
    gene_error_max = float(np.max(np.abs(efficient_gene - explicit_gene)))
    path_error_max = float(np.max(np.abs(efficient_path - explicit_path)))
    passed = gene_error_max <= 1e-12 and path_error_max <= 1e-12
    result = {
        "status": "PASS" if passed else "FAIL",
        "purpose": (
            "independent direct-indexing equivalence for shared context/intervention "
            "resamples across both resolutions and all m"
        ),
        "test_draws": draws,
        "seed": CURVE_SEED,
        "maximum_gene_absolute_error": gene_error_max,
        "maximum_pathway_absolute_error": path_error_max,
        "context_indices_sha256": metadata["context_indices_sha256"],
        "same_context_and_intervention_indices_gene_pathway": True,
        "same_indices_across_m": True,
    }
    if not passed:
        raise RuntimeError("HIERARCHICAL_BOOTSTRAP_IMPLEMENTATION_TEST_FAILED")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scaling, finalize, pathway_module = load_frozen_modules()
    gene_reference, gene_error, gene_metadata = reconstruct_gene_contributions(scaling)
    gene_cache = np.load(OUT / "ED7_GENE_INTERVENTION_CONTRIBUTIONS.npz")
    path_reference, path_error, path_metadata = load_pathway_contributions(
        gene_cache["interventions"]
    )
    implementation_test = implementation_invariance_test(
        gene_reference, gene_error, path_reference, path_error
    )
    write_json(OUT / "ED7_BOOTSTRAP_IMPLEMENTATION_TEST.json", implementation_test)

    point_gene = pooled_g(gene_reference, gene_error)
    point_path = pooled_g(path_reference, path_error)
    curve_gene_draws, curve_path_draws, curve_bootstrap_metadata = hierarchical_bootstrap_pair(
        gene_reference,
        gene_error,
        path_reference,
        path_error,
        CURVE_DRAWS,
        CURVE_SEED,
    )
    expected_curve_nonpositive = {
        "gene": {str(m): int(m == 49) for m in M_GRID},
        "pathway": {str(m): 0 for m in M_GRID},
    }
    if curve_bootstrap_metadata["nonpositive_reference_draws_by_m"] != expected_curve_nonpositive:
        raise RuntimeError(
            "UNEXPECTED_CURVE_REFERENCE_DOMAIN_COUNTS:"
            + json.dumps(curve_bootstrap_metadata["nonpositive_reference_draws_by_m"])
        )
    curve = curve_frame(
        gene_reference,
        gene_error,
        path_reference,
        path_error,
        curve_gene_draws,
        curve_path_draws,
    )
    curve.to_csv(OUT / "ED7_HIERARCHICAL_BOOTSTRAP_CURVES.csv", index=False)

    fit_gene_draws, fit_path_draws, fit_bootstrap_metadata = hierarchical_bootstrap_pair(
        gene_reference,
        gene_error,
        path_reference,
        path_error,
        FIT_DRAWS,
        FIT_SEED,
    )
    expected_fit_nonpositive = {
        "gene": {str(m): 0 for m in M_GRID},
        "pathway": {str(m): 0 for m in M_GRID},
    }
    if fit_bootstrap_metadata["nonpositive_reference_draws_by_m"] != expected_fit_nonpositive:
        raise RuntimeError(
            "UNEXPECTED_FIT_REFERENCE_DOMAIN_COUNTS:"
            + json.dumps(fit_bootstrap_metadata["nonpositive_reference_draws_by_m"])
        )
    parameters, extrapolation, comparison = fit_bootstrap(
        finalize,
        pathway_module,
        fit_gene_draws,
        fit_path_draws,
        point_gene,
        point_path,
    )
    parameters.to_csv(OUT / "ED7_HIERARCHICAL_SCALING_PARAMETERS.csv", index=False)
    extrapolation.to_csv(OUT / "ED7_HIERARCHICAL_EXTRAPOLATION.csv", index=False)
    comparison.to_csv(OUT / "ED7_GENE_VS_PATHWAY_HIERARCHICAL.csv", index=False)
    corrected_source = make_corrected_source_data(curve, comparison)
    corrected_source.to_csv(OUT / "Extended_Data_7_hierarchical_corrected_source_data.csv", index=False)

    # Frozen point values are asserted directly, because their preservation is
    # the release-repair boundary rather than a substantive reanalysis.
    key_values = {
        "gene_g_m2": point_gene[0],
        "gene_g_m49": point_gene[-1],
        "gene_gain_m2_to_m49": point_gene[-1] - point_gene[0],
        "pathway_g_m49": point_path[-1],
        "pathway_minus_gene_g_m49": point_path[-1] - point_gene[-1],
        "gene_g0p5_context_count": float(
            extrapolation.loc[
                extrapolation["resolution"].eq("gene")
                & extrapolation["target_g"].eq(0.5),
                "point_context_count",
            ].iloc[0]
        ),
        "pathway_selected_g_inf": float(
            parameters.loc[
                parameters["resolution"].eq("five_pathway")
                & parameters["parameter"].eq("g_inf"),
                "point_estimate",
            ].iloc[0]
        ),
        "pathway_continuing_power_sensitivity_g0p5_context_count": float(
            comparison["five_pathway_continuing_power_sensitivity_g0p5_point_context_count"].iloc[0]
        ),
    }
    expected = {
        "gene_g_m2": -0.04183860588861177,
        "gene_g_m49": 0.16897584297296175,
        "gene_gain_m2_to_m49": 0.21081444886157352,
        "pathway_g_m49": 0.36182480527186023,
        "pathway_minus_gene_g_m49": 0.19284896229889848,
        "gene_g0p5_context_count": 62810.925057827124,
        "pathway_selected_g_inf": 0.4161416370655423,
        "pathway_continuing_power_sensitivity_g0p5_context_count": 413.67690380407305,
    }
    maximum_key_difference = max(abs(key_values[key] - expected[key]) for key in expected)
    if maximum_key_difference > 1e-9:
        raise RuntimeError(f"FROZEN_KEY_VALUE_CHANGED:{maximum_key_difference}")

    reconciliation = {
        "status": "ED7_HIERARCHICAL_BOOTSTRAP_REPAIR_COMPLETE",
        "protocol": str(
            (
                GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_PROTOCOL.md"
            ).relative_to(REPO)
        ),
        "protocol_sha256": sha256(GENE_OUT / "TAHOE_HELD_CONTEXT_SCALING_PROTOCOL.md"),
        "protocol_requirement": (
            "Hierarchical bootstrap resamples contexts first and interventions within sampled contexts."
        ),
        "old_implementation_scope": "target-context-only bootstrap; interventions fixed",
        "repaired_implementation_scope": (
            "target contexts sampled with replacement, then 93 intervention identities sampled "
            "with replacement independently within every sampled context occurrence"
        ),
        "curve_bootstrap": curve_bootstrap_metadata,
        "fit_bootstrap": fit_bootstrap_metadata,
        "gene_reconstruction": gene_metadata,
        "pathway_contributions": path_metadata,
        "key_values": key_values,
        "expected_frozen_key_values": expected,
        "maximum_frozen_key_value_absolute_difference": maximum_key_difference,
        "point_estimates_unchanged": maximum_key_difference <= 1e-9,
        "predictive_models_refit": False,
        "scaling_laws_refit_within_bootstrap": True,
        "predictions_regenerated": False,
        "hyperparameters_changed": False,
        "scaling_law_families_changed": False,
        "nulls_changed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "ED7_BOOTSTRAP_REPAIR_RECONCILIATION.json", reconciliation)
    write_report(curve, parameters, extrapolation, comparison, reconciliation)
    manifest_files = [
        path
        for path in sorted(OUT.iterdir())
        if path.is_file()
        and path.name not in {"ED7_OUTPUT_MANIFEST.json"}
        and path.suffix != ".pyc"
    ]
    write_json(
        OUT / "ED7_OUTPUT_MANIFEST.json",
        {
            "status": "COMPLETE",
            "files": [
                {
                    "path": str(path.relative_to(REPO)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in manifest_files
            ],
        },
    )
    print(json.dumps(key_values, indent=2), flush=True)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "not mathematically reachable"
    return f"{value:.9g}"


def write_report(
    curve: pd.DataFrame,
    parameters: pd.DataFrame,
    extrapolation: pd.DataFrame,
    comparison: pd.DataFrame,
    reconciliation: dict[str, Any],
) -> None:
    gene49 = curve[(curve["resolution"] == "gene") & (curve["m"] == 49)].iloc[0]
    path49 = curve[(curve["resolution"] == "five_pathway") & (curve["m"] == 49)].iloc[0]
    gene2 = curve[(curve["resolution"] == "gene") & (curve["m"] == 2)].iloc[0]
    contrast = comparison.iloc[0]
    path_ginf = parameters[
        (parameters["resolution"] == "five_pathway") & (parameters["parameter"] == "g_inf")
    ].iloc[0]
    gene_n = extrapolation[
        (extrapolation["resolution"] == "gene") & (extrapolation["target_g"] == 0.5)
    ].iloc[0]
    path_n = extrapolation[
        (extrapolation["resolution"] == "five_pathway")
        & (extrapolation["target_g"] == 0.5)
    ].iloc[0]
    path_sensitivity_n = extrapolation[
        (extrapolation["resolution"] == "five_pathway_sensitivity")
        & (extrapolation["target_g"] == 0.5)
    ].iloc[0]
    text = f"""# ED7 hierarchical-bootstrap release repair

## Scope and adjudication

This is the single approved release-blocker correction. The frozen protocol requires target contexts to be resampled first and interventions within each sampled context. The historical implementation resampled target contexts only, leaving the 93 interventions fixed. This repair adds exactly the missing inner intervention resampling layer. It does not refit a predictive model, regenerate a prediction, retune a hyperparameter, change a point-estimate definition, select a new scaling law, change a seed or draw count, introduce a null, or render a figure.

For each bootstrap draw, 50 target contexts are sampled with replacement. For every sampled context occurrence, 93 intervention identities are then sampled with replacement from that context's frozen 93-intervention contribution vector. The context index matrices are generated first under the historical seeds, so the target-context resamples are exactly the same as in the old implementation; only the preregistered inner layer is added. Empirical bands use {CURVE_DRAWS:,} draws (seed {CURVE_SEED}); scaling-law stability uses {FIT_DRAWS:,} draws (seed {FIT_SEED}).

## Frozen-point reconciliation

All point estimates are unchanged (maximum absolute discrepancy `{reconciliation['maximum_frozen_key_value_absolute_difference']:.3g}`). Gene-level recovery remains `{gene2['pooled_g']:.12f}` at m=2 and `{gene49['pooled_g']:.12f}` at m=49, for an observed gain of `{gene49['pooled_g'] - gene2['pooled_g']:.12f}`. Five-pathway recovery at m=49 remains `{path49['pooled_g']:.12f}`, and the pathway-minus-gene contrast remains `{contrast['five_pathway_minus_gene_g_m49']:.12f}`.

The gene `g=0.50` continuing-power projection remains `{gene_n['point_context_count']:.12f}` response-observed reference contexts (62,811 when rounded). The selected five-pathway exponential-sensitivity asymptote remains `g_inf={path_ginf['point_estimate']:.12f}` (0.416142 when rounded), so `g=0.50` is not mathematically reached by the selected point fit. The forced continuing-power sensitivity remains `{contrast['five_pathway_continuing_power_sensitivity_g0p5_point_context_count']:.12f}` contexts (413.677 when rounded). These three values are fitted point summaries and therefore do not change when only the bootstrap implementation is repaired.

## Corrected uncertainty

- Gene m=49 recovery: `{gene49['pooled_g']:.12f}`, corrected hierarchical 95% percentile interval `[{gene49['hierarchical_bootstrap_lower_95']:.12f}, {gene49['hierarchical_bootstrap_upper_95']:.12f}]` (9,999 defined draws from 10,000 attempts; one draw had a non-positive reference cross-energy and is undefined under the frozen ratio definition). The old context-only interval was `[{gene49['old_context_only_lower_95']:.12f}, {gene49['old_context_only_upper_95']:.12f}]`.
- Five-pathway m=49 recovery: `{path49['pooled_g']:.12f}`, corrected hierarchical 95% percentile interval `[{path49['hierarchical_bootstrap_lower_95']:.12f}, {path49['hierarchical_bootstrap_upper_95']:.12f}]` (old context-only interval `[{path49['old_context_only_lower_95']:.12f}, {path49['old_context_only_upper_95']:.12f}]`).
- Five-pathway minus gene at m=49: `{contrast['five_pathway_minus_gene_g_m49']:.12f}`, corrected paired hierarchical 95% percentile interval `[{contrast['hierarchical_bootstrap_difference_lower_95']:.12f}, {contrast['hierarchical_bootstrap_difference_upper_95']:.12f}]` (old context-only interval `[0.084953527304, 0.393471404009]`).
- Selected five-pathway asymptote: `g_inf={path_ginf['point_estimate']:.12f}`, corrected hierarchical 95% percentile interval `[{path_ginf['hierarchical_bootstrap_lower_95']:.12f}, {path_ginf['hierarchical_bootstrap_upper_95']:.12f}]`.
- Selected five-pathway `g=0.50` reachable fraction across scaling-law bootstrap draws: `{path_n['bootstrap_reachable_fraction']:.6f}`. Any interval for its context count is conditional on the mathematically reachable draws and is not a primary unconditional confidence interval.
- Gene `g=0.50` projected context count: point `{gene_n['point_context_count']:.12f}`; corrected hierarchical 95% percentile interval `[{gene_n['hierarchical_bootstrap_lower_95']:.12f}, {gene_n['hierarchical_bootstrap_upper_95']:.12f}]` across the frozen continuing-power fit draws.
- Forced continuing-power pathway sensitivity at `g=0.50`: point `{path_sensitivity_n['point_context_count']:.12f}`; corrected hierarchical 95% percentile interval `[{path_sensitivity_n['hierarchical_bootstrap_lower_95']:.12f}, {path_sensitivity_n['hierarchical_bootstrap_upper_95']:.12f}]`. This remains a model-form sensitivity, not the selected-law estimate.

## Conclusion boundary

The qualitative claims that observed held-context recovery rises with support and that pathway readout is more recoverable than gene readout at m=49 survive: the corrected paired interval is strictly positive. The numerical projections 62,811 and 413.677 remain extrapolative fitted point summaries, not observed thresholds; 413.677 remains explicitly a forced continuing-power sensitivity. The selected pathway point fit still asymptotes below `g=0.50` at 0.416142, but its corrected bootstrap interval spans 0.50; therefore a population-level asymptote below 0.50 is not established. Release-facing bands must be replaced by the corrected hierarchical values in this directory; the old context-only intervals must not be described as satisfying the preregistration.

## Artifacts

- `ED7_HIERARCHICAL_BOOTSTRAP_CURVES.csv`: corrected empirical bands at all m.
- `ED7_HIERARCHICAL_SCALING_PARAMETERS.csv`: corrected fit-parameter uncertainty.
- `ED7_HIERARCHICAL_EXTRAPOLATION.csv`: corrected extrapolation stability and reachability.
- `ED7_GENE_VS_PATHWAY_HIERARCHICAL.csv`: paired resolution contrast.
- `Extended_Data_7_hierarchical_corrected_source_data.csv`: publication-facing replacement rows.
- `ED7_GENE_INTERVENTION_CONTRIBUTIONS.npz`: audit cache reconstructed from frozen response Grams, frozen baseline kernels and saved RBF hyperparameters.
- `ED7_BOOTSTRAP_REPAIR_RECONCILIATION.json`: hashes, seeds, counts, tolerances and point reconciliation.
- `ED7_BOOTSTRAP_IMPLEMENTATION_TEST.json`: direct-indexing equivalence check.
- `ED7_OUTPUT_MANIFEST.json`: SHA-256 manifest of repair artifacts.
"""
    (OUT / "ED7_BOOTSTRAP_REPAIR_REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Some Windows wrappers do not surface stderr from detached/PTY runs.
        # Emit the audit failure to stdout as well, then retain the non-zero exit.
        traceback.print_exc(file=sys.stdout)
        raise
