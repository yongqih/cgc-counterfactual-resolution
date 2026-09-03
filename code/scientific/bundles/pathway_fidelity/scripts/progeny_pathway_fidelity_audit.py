#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_resolution_poc.pathway import PATHWAYS, load_pathway_weights
from igc_virtual_cell.cgc_resolution_poc.replay import CONTEXTS, GENE_COUNT, INTERVENTIONS


AUTHORITY_COMMIT = "46b25eec4a97e3001cc43262980af8f69c533478"
M, K = 49, 92
BLOCK_GENES = 512
BOOTSTRAPS = 10_000
EXPECTED_HASHES = {
    "weights": "f5c4faa94fe03b2d088b72743da175380dfcf2edbfacd0077cc25eb19d2ca7e9",
    "metadata": "f690be9e4a57bcf1fc2c70d1e458ea901950c5c42cc84c6d1fb7e728094c4960",
    "indices": "efea7718149514c763fbbad86d2337fde3377bf6c977365f0046576a85b94343",
    "truth": "005603c5a01c085762c9eecb266de6302f95356ca5eb4026f8c0380f57793f73",
    "episode": "b0e410e9c0f5d0d9f0a4d3fbfd02cf247d171cd203c480083dc6fa666b68d928",
    "utility": "bd3b4f5097b4f158f58da102849d52080bb9ae7b7a4c4325f37bf574fd1a7c08",
    "bootstrap": "f0a9697175fe2d4b8bc800e4352d961ed349f9e9a33df138c793be24f982c9cf",
    "split": "da3fc0f3422c12e2a1ec9fcdc6447cd137836a963290a262a82451430df1198d",
}
FULL_TABLE_ATOL = 5e-7
FROZEN_PATHWAY_DENOMINATOR_ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orthogonal_basis(weight_matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(weight_matrix, dtype=np.float64)
    if matrix.shape != (GENE_COUNT, len(PATHWAYS)) or not np.isfinite(matrix).all():
        raise RuntimeError("PROGENY_SPAN_MATRIX_INVALID")
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    relative_cutoff = max(matrix.shape) * np.finfo(np.float64).eps
    absolute_tolerance = float(singular_values[0] * relative_cutoff)
    rank = int(np.sum(singular_values > absolute_tolerance))
    q = np.asarray(u[:, :rank], dtype=np.float64)
    gram_eigenvalues = np.linalg.eigvalsh(matrix.T @ matrix)
    positive = gram_eigenvalues[gram_eigenvalues > absolute_tolerance**2]
    condition = float(positive.max() / positive.min()) if len(positive) == len(PATHWAYS) else None
    span = {
        "matrix_shape": list(matrix.shape),
        "singular_values": singular_values.tolist(),
        "rank_tolerance_absolute": absolute_tolerance,
        "pseudoinverse_rcond": float(relative_cutoff),
        "numerical_rank": rank,
        "gram_eigenvalues": gram_eigenvalues.tolist(),
        "condition_number_WtW": condition,
        "condition_number_status": "finite_full_rank" if condition is not None else "infinite_rank_deficient",
        "rank_direction_discarded": bool(rank < len(PATHWAYS)),
        "orthonormality_max_abs_error": float(np.max(np.abs(q.T @ q - np.eye(rank)))),
        "weight_span_reconstruction_relative_error": float(
            np.linalg.norm(matrix - q @ (q.T @ matrix)) / np.linalg.norm(matrix)
        ),
        "projector_implementation": "implicit Q @ Q.T from compact SVD; no GxG matrix materialized",
    }
    return q, span


def all_but_one_deltas(block: np.ndarray) -> np.ndarray:
    values = np.asarray(block, dtype=np.float64)
    if values.shape[1] != CONTEXTS:
        raise ValueError("context axis mismatch")
    return (CONTEXTS / (CONTEXTS - 1.0)) * (values - values.mean(axis=1, keepdims=True))


def weighted_ratios(weights: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    counts = np.asarray(weights)
    num = np.asarray(numerator, dtype=np.float64).reshape(-1)
    den = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if counts.shape != (BOOTSTRAPS, CONTEXTS * INTERVENTIONS):
        raise RuntimeError("PROGENY_BOOTSTRAP_WEIGHT_SHAPE_INVALID")
    output = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 200):
        stop = min(start + 200, BOOTSTRAPS)
        block = counts[start:stop].astype(np.float64)
        draw_denominator = block @ den
        if np.any(~np.isfinite(draw_denominator)) or np.any(draw_denominator <= 0):
            raise RuntimeError("PROGENY_BOOTSTRAP_DENOMINATOR_INVALID")
        output[start:stop] = (block @ num) / draw_denominator
    return output


def validate_inputs(root: Path, truth_root: Path) -> dict[str, Path]:
    paths = {
        "weights": truth_root / "data/cgc_bio1_official/frozen_program_weights.parquet",
        "metadata": truth_root / "results/cgc_tahoe_0i/gene_metadata_frozen.csv",
        "indices": truth_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy",
        "truth": truth_root / "data/cgc_resolution2_replay/truth_g_primary_float32.npy",
        "episode": truth_root / "data/cgc_resolution2_replay/m49_k92_frozen_weights.npz",
        "utility": truth_root / "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz",
        "bootstrap": truth_root / "data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy",
        "split": root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json",
    }
    for name, path in paths.items():
        if not path.exists() or sha256(path) != EXPECTED_HASHES[name]:
            raise RuntimeError(f"PROGENY_FIDELITY_INPUT_HASH_MISMATCH:{name}")
    return paths


def support_table(weights: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    union = np.any(weights != 0, axis=0)
    for pathway, vector in zip(PATHWAYS, weights, strict=True):
        nonzero = vector != 0
        rows.append({
            "scope": "pathway",
            "pathway": pathway,
            "nonzero_weight_genes": int(nonzero.sum()),
            "positive_weight_genes": int((vector > 0).sum()),
            "negative_weight_genes": int((vector < 0).sum()),
            "gene_universe": GENE_COUNT,
            "nonzero_fraction": float(nonzero.mean()),
        })
    rows.append({
        "scope": "five_pathway_union",
        "pathway": "UNION",
        "nonzero_weight_genes": int(union.sum()),
        "positive_weight_genes": np.nan,
        "negative_weight_genes": np.nan,
        "gene_universe": GENE_COUNT,
        "nonzero_fraction": float(union.mean()),
    })
    return pd.DataFrame(rows)


def compute(root: Path, truth_root: Path) -> dict[str, Any]:
    output = root / "results/cgc_progeny_pathway_fidelity"
    output.mkdir(parents=True, exist_ok=True)
    paths = validate_inputs(root, truth_root)
    weights, _ = load_pathway_weights(truth_root)
    support = support_table(weights)
    support.to_csv(output / "PROGENY_PATHWAY_GENE_SUPPORT.csv", index=False)
    q, span = orthogonal_basis(weights.T)

    with np.load(paths["episode"], allow_pickle=False) as archive:
        sources = np.asarray(archive["sources"], dtype=np.int64)
    all_but_one = bool(
        sources.shape == (1, CONTEXTS, CONTEXTS - 1)
        and all(set(map(int, sources[0, target])) == (set(range(CONTEXTS)) - {target}) for target in range(CONTEXTS))
    )
    if not all_but_one:
        raise RuntimeError("PROGENY_PRIMARY_EPISODE_NOT_ALL_BUT_ONE")

    truth = np.load(paths["truth"], mmap_mode="r")
    if truth.shape != (2, CONTEXTS, INTERVENTIONS, GENE_COUNT) or truth.dtype != np.float32:
        raise RuntimeError("PROGENY_TRUTH_CACHE_AXIS_INVALID")
    full_cross = np.zeros((CONTEXTS, INTERVENTIONS), dtype=np.float64)
    full_energy = np.zeros((2, CONTEXTS, INTERVENTIONS), dtype=np.float64)
    coordinates = np.zeros((2, CONTEXTS, INTERVENTIONS, span["numerical_rank"]), dtype=np.float64)
    for start in range(0, GENE_COUNT, BLOCK_GENES):
        stop = min(start + BLOCK_GENES, GENE_COUNT)
        delta = all_but_one_deltas(np.asarray(truth[..., start:stop], dtype=np.float64))
        full_cross += np.einsum("cig,cig->ci", delta[0], delta[1], optimize=True)
        for plate in range(2):
            full_energy[plate] += np.einsum("cig,cig->ci", delta[plate], delta[plate], optimize=True)
        coordinates += np.einsum("pcig,gr->pcir", delta, q[start:stop], optimize=True)
        print(f"PROGENY fidelity genes={stop}/{GENE_COUNT}", flush=True)

    path_cross = np.einsum("cir,cir->ci", coordinates[0], coordinates[1], optimize=True)
    path_energy = np.einsum("pcir,pcir->pci", coordinates, coordinates, optimize=True)
    residual_cross = full_cross - path_cross
    with np.load(paths["utility"], allow_pickle=False) as archive:
        m_values = archive["m_values"].astype(int).tolist()
        frozen_vtruth = np.asarray(archive["vtruth"][m_values.index(M)], dtype=np.float64)
    direct_vtruth = full_cross / GENE_COUNT
    full_table_max_abs_error = float(np.max(np.abs(direct_vtruth - frozen_vtruth)))
    if full_table_max_abs_error > FULL_TABLE_ATOL:
        raise RuntimeError(f"PROGENY_FULL_TRUTH_REPLAY_MISMATCH:{full_table_max_abs_error}")

    score_transform = q.T @ weights.T
    frozen_scores = np.einsum("pcir,rh->pcih", coordinates, score_transform, optimize=True)
    nonorthogonal_denominator = float(np.sum(frozen_scores[0] * frozen_scores[1], dtype=np.float64))
    frozen_result = pd.read_csv(root / "results/cgc_resolution_poc_v2/RESOLUTION2_PATHWAY_RESULTS.csv")
    expected_denominator = float(frozen_result.loc[
        frozen_result["m"].eq(M) & frozen_result["k"].eq(K) & frozen_result["pathway"].eq("POOLED"),
        "truth_denominator",
    ].iloc[0])
    if abs(nonorthogonal_denominator - expected_denominator) > FROZEN_PATHWAY_DENOMINATOR_ATOL:
        raise RuntimeError("PROGENY_FROZEN_PATHWAY_DENOMINATOR_MISMATCH")

    bootstrap_weights = np.load(paths["bootstrap"], mmap_mode="r")
    if bootstrap_weights.dtype != np.int16 or not np.all(bootstrap_weights.sum(axis=1) == CONTEXTS * INTERVENTIONS):
        raise RuntimeError("PROGENY_BOOTSTRAP_WEIGHTS_INVALID")
    bootstrap = weighted_ratios(bootstrap_weights, path_cross, full_cross)
    point_full = float(full_cross.sum(dtype=np.float64))
    point_path = float(path_cross.sum(dtype=np.float64))
    fidelity = point_path / point_full
    oracle_rows: list[dict[str, Any]] = [{
        "analysis_type": "primary_cross_replicate",
        "replicate": "Plate6_x_Plate14",
        "full_signal": point_full,
        "pathway_span_signal": point_path,
        "residual_signal": float(residual_cross.sum(dtype=np.float64)),
        "fidelity_fraction": fidelity,
        "bootstrap_se": float(bootstrap.std(ddof=1)),
        "bootstrap_percentile_lower_95": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_percentile_upper_95": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_draws": BOOTSTRAPS,
        "squared_error_fidelity": np.nan,
        "cosine_D_vs_projected_D": np.nan,
        "variance_signal_retained": np.nan,
    }]
    for plate, label in enumerate(("Plate6", "Plate14")):
        full = float(full_energy[plate].sum(dtype=np.float64))
        projected = float(path_energy[plate].sum(dtype=np.float64))
        retained = projected / full
        oracle_rows.append({
            "analysis_type": "secondary_within_replicate",
            "replicate": label,
            "full_signal": full,
            "pathway_span_signal": projected,
            "residual_signal": full - projected,
            "fidelity_fraction": np.nan,
            "bootstrap_se": np.nan,
            "bootstrap_percentile_lower_95": np.nan,
            "bootstrap_percentile_upper_95": np.nan,
            "bootstrap_draws": np.nan,
            "squared_error_fidelity": 1.0 - (full - projected) / full,
            "cosine_D_vs_projected_D": float(np.sqrt(retained)),
            "variance_signal_retained": retained,
        })
    pd.DataFrame(oracle_rows).to_csv(output / "PROGENY_PATHWAY_ORACLE_FIDELITY.csv", index=False)

    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    source = pd.DataFrame({
        "target_context_index": np.repeat(np.arange(CONTEXTS), INTERVENTIONS),
        "target_context": np.repeat(np.asarray(split["contexts"], dtype=str), INTERVENTIONS),
        "intervention_index": np.tile(np.arange(INTERVENTIONS), CONTEXTS),
        "intervention": np.tile(np.asarray(split["interventions"], dtype=str), CONTEXTS),
        "full_cross_signal_raw": full_cross.ravel(),
        "pathway_span_cross_signal_raw": path_cross.ravel(),
        "residual_cross_signal_raw": residual_cross.ravel(),
        "full_cross_signal_per_gene": direct_vtruth.ravel(),
        "pathway_span_cross_signal_per_gene": (path_cross / GENE_COUNT).ravel(),
    })
    source.to_csv(output / "PROGENY_PATHWAY_FIDELITY_SOURCE_DATA.csv", index=False)

    span.update({
        "authority_commit": AUTHORITY_COMMIT,
        "pathways": list(PATHWAYS),
        "gene_universe": GENE_COUNT,
        "episode": {"m": M, "k": K, "sequence_count": 1, "all_but_one_sources_verified": all_but_one},
        "input_sha256": EXPECTED_HASHES,
        "full_vtruth_table_max_abs_error": full_table_max_abs_error,
        "full_vtruth_validation_tolerance": FULL_TABLE_ATOL,
        "frozen_nonorthogonal_pathway_denominator_expected": expected_denominator,
        "frozen_nonorthogonal_pathway_denominator_recomputed": nonorthogonal_denominator,
        "frozen_nonorthogonal_pathway_denominator_abs_error": abs(nonorthogonal_denominator - expected_denominator),
        "frozen_pathway_denominator_validation_tolerance": FROZEN_PATHWAY_DENOMINATOR_ATOL,
        "projection_precedes_cross_plate_aggregation": True,
        "random_projection_control_run": False,
    })
    (output / "PROGENY_PATHWAY_SPAN_AUDIT.json").write_text(
        json.dumps(span, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "fidelity_fraction": fidelity,
        "bootstrap_lower_95": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_upper_95": float(np.quantile(bootstrap, 0.975)),
        "numerical_rank": span["numerical_rank"],
        "union_nonzero_genes": int(support.iloc[-1]["nonzero_weight_genes"]),
        "full_table_max_abs_error": full_table_max_abs_error,
        "pathway_denominator_abs_error": abs(nonorthogonal_denominator - expected_denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen five-pathway PROGENy oracle-fidelity audit")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--truth-root", type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell"),
    )
    args = parser.parse_args()
    if not args.formal:
        parser.error("--formal is required")
    print(json.dumps(compute(args.root.resolve(), args.truth_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
