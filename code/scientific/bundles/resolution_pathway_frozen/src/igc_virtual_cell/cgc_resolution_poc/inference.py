from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_resolution_poc.pathway import PATHWAYS, recovery
from igc_virtual_cell.cgc_resolution_poc.replay import BUDGETS, CONTEXTS, INTERVENTIONS


BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20_260_822


def hierarchical_weights(draws: int = BOOTSTRAPS, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Frozen context-then-intervention resampling counts on the 50 x 93 table."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, CONTEXTS * INTERVENTIONS), dtype=np.int16)
    for draw in range(draws):
        contexts = rng.integers(0, CONTEXTS, size=CONTEXTS)
        interventions = rng.integers(0, INTERVENTIONS, size=(CONTEXTS, INTERVENTIONS))
        flat = (contexts[:, None] * INTERVENTIONS + interventions).ravel()
        weights[draw] = np.bincount(flat, minlength=CONTEXTS * INTERVENTIONS)
    return weights


def _weighted_sums(weights: np.ndarray, values: np.ndarray, block: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[:2] == (CONTEXTS, INTERVENTIONS):
        matrix = values.reshape(CONTEXTS * INTERVENTIONS, -1)
    elif values.shape[1:3] == (CONTEXTS, INTERVENTIONS):
        matrix = np.moveaxis(values, 0, -1).reshape(CONTEXTS * INTERVENTIONS, -1)
    else:
        raise ValueError(f"Unexpected utility axes: {values.shape}")
    output = np.empty((len(weights), matrix.shape[1]), dtype=np.float64)
    for start in range(0, len(weights), block):
        stop = min(start + block, len(weights))
        output[start:stop] = weights[start:stop].astype(np.float64) @ matrix
    return output


def bootstrap_g(weights: np.ndarray, vtruth: np.ndarray, vafter: np.ndarray) -> np.ndarray:
    truth = _weighted_sums(weights, vtruth)
    after = _weighted_sums(weights, vafter)
    # Coordinates trailing a context x intervention table are pooled by the
    # frozen ratio-of-sums. A leading axis, by contrast, identifies separate
    # matched-null candidates and must remain separate.
    if vtruth.shape[:2] == (CONTEXTS, INTERVENTIONS):
        truth = truth.sum(axis=1, keepdims=True)
        after = after.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1.0 - after / truth
    if not np.isfinite(result).all():
        raise RuntimeError("RESOLUTION2_BOOTSTRAP_DENOMINATOR_INVALID")
    return result


def simultaneous_rows(
    names: list[str], observed: np.ndarray, draws: np.ndarray, family: str
) -> list[dict[str, Any]]:
    if draws.shape != (BOOTSTRAPS, len(names)):
        raise ValueError("Unexpected bootstrap family shape")
    standard_error = draws.std(axis=0, ddof=1)
    if np.any(~np.isfinite(standard_error)) or np.any(standard_error <= 0):
        raise RuntimeError("RESOLUTION2_BOOTSTRAP_STANDARD_ERROR_INVALID")
    centered = (draws - observed[None, :]) / standard_error[None, :]
    critical = float(np.quantile(np.max(np.abs(centered), axis=1), 0.95))
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        low = float(observed[index] - critical * standard_error[index])
        high = float(observed[index] + critical * standard_error[index])
        rows.append(
            {
                "family": family,
                "contrast": name,
                "estimate": float(observed[index]),
                "bootstrap_se": float(standard_error[index]),
                "simultaneous_critical_95": critical,
                "simultaneous_lower_95": low,
                "simultaneous_upper_95": high,
                "corrected_positive": bool(low > 0),
                "bootstrap_draws": BOOTSTRAPS,
            }
        )
    return rows


def _gene_utility(source_root: Path, m: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    path = source_root / "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz"
    with np.load(path, allow_pickle=False) as archive:
        models = list(map(str, archive["models"]))
        ms = archive["m_values"].astype(int).tolist()
        ks = archive["k_values"].astype(int).tolist()
        model = models.index("M2_AFFINE_RIDGE")
        return (
            np.asarray(archive["vtruth"][ms.index(m)], dtype=np.float64),
            np.asarray(archive["vafter"][model, ms.index(m), ks.index(k)], dtype=np.float64),
        )


def write_program_nonexecutability(output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for m, k, sequences in BUDGETS:
        episodes = CONTEXTS * INTERVENTIONS * sequences
        accessible_per_plate = m * INTERVENTIONS + k
        for rank in (4, 8):
            rows.append(
                {
                    "m": m,
                    "k": k,
                    "program_k": rank,
                    "status": "PROGRAM_RESOLUTION_POC_NOT_EXECUTABLE",
                    "episode_local_pca_fits": episodes,
                    "accessible_training_rows_both_plates": 2 * accessible_per_plate,
                    "reason": "No mathematically exact tested shortcut for the frozen episode-local thin SVD was available; global or grouped PCA is forbidden.",
                }
            )
    pd.DataFrame(rows).to_csv(output / "RESOLUTION2_PROGRAM_RESULTS.csv", index=False)
    null_rows: list[dict[str, Any]] = []
    for rank in (4, 8):
        episodes = sum(CONTEXTS * INTERVENTIONS * sequences for _, _, sequences in BUDGETS)
        candidates = episodes * 5_000
        null_rows.append(
            {
                "program_k": rank,
                "status": "PROGRAM_RESOLUTION_POC_NOT_EXECUTABLE",
                "eligible_episodes": episodes,
                "candidate_subspaces": candidates,
                "retained_subspaces": episodes * 500,
                "gaussian_values_required": candidates * 25_695 * rank,
                "reason": "Literal frozen candidate generation was computationally intractable and no exact Haar-preserving contraction shortcut was validated.",
            }
        )
    pd.DataFrame(null_rows).to_csv(output / "RESOLUTION2_PROGRAM_RANDOM_NULL.csv", index=False)


def run_inference(root: Path, source_root: Path) -> dict[str, Any]:
    output = root / "results/cgc_resolution_poc_v2"
    cache = root / "data/cgc_resolution2_replay/pathway_null/pathway_frozen_utilities.npz"
    weights_path = root / "data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy"
    if weights_path.exists():
        weights = np.load(weights_path)
    else:
        weights = hierarchical_weights()
        np.save(weights_path, weights)
    rows: list[dict[str, Any]] = []
    primary_names: list[str] = []
    primary_observed: list[float] = []
    primary_draws: list[np.ndarray] = []
    secondary_names: list[str] = []
    secondary_observed: list[float] = []
    secondary_draws: list[np.ndarray] = []
    with np.load(cache, allow_pickle=False) as archive:
        for m, k, _ in BUDGETS:
            real_vt = np.asarray(archive[f"real_vtruth_m{m}_k{k}"], dtype=np.float64)
            real_va = np.asarray(archive[f"real_vafter_m{m}_k{k}"], dtype=np.float64)
            null_vt = np.asarray(archive[f"null_vtruth_m{m}_k{k}"], dtype=np.float64)
            null_va = np.asarray(archive[f"null_vafter_m{m}_k{k}"], dtype=np.float64)
            gene_vt, gene_va = _gene_utility(source_root, m, k)
            real_point = recovery(real_vt, real_va)
            gene_point = recovery(gene_vt, gene_va)
            null_point_each = 1.0 - null_va.reshape(500, -1).sum(axis=1) / null_vt.reshape(500, -1).sum(axis=1)
            null_point = float(np.median(null_point_each))
            real_draw = bootstrap_g(weights, real_vt, real_va).ravel()
            gene_draw = bootstrap_g(weights, gene_vt, gene_va).ravel()
            null_draw = np.median(bootstrap_g(weights, null_vt, null_va), axis=1)
            for suffix, estimate, draw in (
                ("pathway_minus_gene", real_point - gene_point, real_draw - gene_draw),
                ("pathway_minus_matched_random", real_point - null_point, real_draw - null_draw),
            ):
                primary_names.append(f"m{m}_k{k}_{suffix}")
                primary_observed.append(estimate)
                primary_draws.append(draw)
            rows.append({"family": "point_estimate", "contrast": f"m{m}_k{k}_g_pathway", "estimate": real_point})
            rows.append({"family": "point_estimate", "contrast": f"m{m}_k{k}_g_gene", "estimate": gene_point})
            rows.append({"family": "point_estimate", "contrast": f"m{m}_k{k}_g_matched_random_median", "estimate": null_point})
            for pathway in PATHWAYS:
                index = PATHWAYS.index(pathway)
                pvt, pva = real_vt[..., index], real_va[..., index]
                nvt = np.asarray(archive[f"null_vtruth_{pathway}_m{m}_k{k}"], dtype=np.float64)
                nva = np.asarray(archive[f"null_vafter_{pathway}_m{m}_k{k}"], dtype=np.float64)
                ppoint = recovery(pvt, pva)
                npoint = float(np.median(1.0 - nva.reshape(500, -1).sum(axis=1) / nvt.reshape(500, -1).sum(axis=1)))
                pdraw = bootstrap_g(weights, pvt, pva).ravel()
                ndraw = np.median(bootstrap_g(weights, nvt, nva), axis=1)
                secondary_names.append(f"m{m}_k{k}_{pathway}_minus_matched_random")
                secondary_observed.append(ppoint - npoint)
                secondary_draws.append(pdraw - ndraw)
    rows.extend(simultaneous_rows(primary_names, np.asarray(primary_observed), np.column_stack(primary_draws), "PATHWAY_PRIMARY_2BUDGET_X_2CONTRAST"))
    rows.extend(simultaneous_rows(secondary_names, np.asarray(secondary_observed), np.column_stack(secondary_draws), "PATHWAY_SECONDARY_2BUDGET_X_5PATHWAY"))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "RESOLUTION2_PAIRED_COMPARISONS.csv", index=False)
    write_program_nonexecutability(output)
    primary = frame[frame["family"] == "PATHWAY_PRIMARY_2BUDGET_X_2CONTRAST"]
    primary_gate = bool(primary[primary["contrast"].str.startswith("m49_k92")]["corrected_positive"].all())
    return {"primary_gate": primary_gate, "primary": primary.to_dict("records")}
