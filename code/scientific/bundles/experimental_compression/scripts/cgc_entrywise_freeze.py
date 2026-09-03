from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


M_VALUES = [1, 2, 4, 8, 16, 24, 32, 40, 49]
K_VALUES = [0, 1, 2, 4, 8, 16, 32, 64, 80, 92]
SUPPORT_SEQUENCES = list(range(8))
SENTINEL_BASE_SEED = 202_608_222
NULL_BASE_SEED = 202_608_223
SYNTHETIC_SEED = 202_608_224
BOOTSTRAP_SEED = 202_608_225


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _seed(base: int, label: str, sequence: int) -> int:
    token = f"{base}|{label}|{sequence}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _permutation(values: list[str], seed: int) -> list[str]:
    # Local import keeps protocol generation independent of the scientific runtime.
    import random

    result = list(values)
    random.Random(seed).shuffle(result)
    return result


def _derangement(values: list[str], seed: int) -> dict[str, str]:
    order = _permutation(values, seed)
    shifted = order[1:] + order[:1]
    mapping = dict(zip(order, shifted, strict=True))
    if any(key == value for key, value in mapping.items()):
        raise RuntimeError("Derangement construction failed")
    return mapping


def _load_supports(root: Path, contexts: list[str]) -> tuple[dict[str, Any], list[int]]:
    path = root / "results/cgc_tahoe_0c/support_sequences.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    selected: dict[str, Any] = {}
    seeds: set[int] = set()
    for context in contexts:
        context_rows = [
            row for row in rows
            if row["heldout_context"] == context and int(row["sequence"]) in SUPPORT_SEQUENCES
        ]
        by_sequence: dict[str, Any] = {}
        for sequence in SUPPORT_SEQUENCES:
            subset = sorted(
                (row for row in context_rows if int(row["sequence"]) == sequence),
                key=lambda row: int(row["rank"]),
            )
            if len(subset) != 49:
                raise RuntimeError(f"{context} sequence {sequence}: expected 49 supports")
            order = [row["support_context"] for row in subset]
            if context in order or set(order) != set(contexts) - {context}:
                raise RuntimeError("Frozen context support order changed")
            seed = int(subset[0]["sequence_seed"])
            seeds.add(seed)
            by_sequence[str(sequence)] = {"seed": seed, "order": order}
        selected[context] = by_sequence
    return selected, sorted(seeds)


def freeze(root: Path, source_root: Path) -> None:
    root = root.resolve()
    source_root = source_root.resolve()
    out = root / "results/cgc_entrywise_compression"
    axes = json.loads((root / "data/tahoe100m_plate6_14_core/axes.json").read_text(encoding="utf-8"))
    contexts = list(axes["contexts"])
    interventions = list(axes["interventions"])
    if len(contexts) != 50 or len(interventions) != 93:
        raise RuntimeError("Frozen Tahoe core axes changed")

    universe = json.loads((root / "results/cgc_tahoe_0i/gene_universe_hashes.json").read_text(encoding="utf-8"))["G_PRIMARY"]
    response_manifest_path = root / "results/cgc_tahoe_0i/response_manifest.json"
    response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    if response_manifest["gene_universes"]["G_PRIMARY"] != 25_695:
        raise RuntimeError("G_PRIMARY changed")
    source_store = source_root / "results/cgc_tahoe_0i/response_tensors.zarr"
    source_indices = source_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy"
    if not source_store.exists() or not source_indices.exists():
        raise FileNotFoundError("Audited CGC-0I response store is not available")

    information = {
        "phase": "CGC-EC-2",
        "frozen_at": _now(),
        "derived_from_commit": "6d3b3203504eb7f18917c949cfd94536d6f5648d",
        "upstream_full_transcriptome_commit": "32bffc563c6f994e0741e498b367211eedbb1bbf",
        "upstream_full_transcriptome_frozen_tip": "c96ee83bd59e3974bc42cd1cfc2aecf7b7dd9e70",
        "source_root_runtime": str(source_root),
        "response_store_logical_path": "results/cgc_tahoe_0i/response_tensors.zarr",
        "response_array": "delta_primary",
        "response_shape_full": [2, 50, 93, 62_710],
        "selected_shape": [2, 50, 93, 25_695],
        "plates": ["plate6", "plate14"],
        "contexts": contexts,
        "interventions": interventions,
        "gene_universe": "G_PRIMARY",
        "gene_count": int(universe["gene_count"]),
        "gene_sha256_newline_symbols": universe["sha256_newline_gene_symbols"],
        "gene_indices_logical_path": universe["indices_path"],
        "gene_indices_sha256": _sha256(source_indices),
        "response_manifest_sha256": _sha256(response_manifest_path),
        "response_definition": response_manifest["primary"],
        "baseline_controls_assumed_observed_all_contexts": True,
        "baseline_rna_logical_path": "data/tahoe100m_control_state/control_wells_log1p_cpm_float32.npy",
        "baseline_rna_axes": "data/tahoe100m_control_state/axes.json",
        "raw_h5ad_reprocessed": False,
        "large_data_copied_to_worktree": False,
        "historical_0j_role": "TRANSDUCTIVE_TWO_WAY_CENTERED_ANATOMY",
    }
    _write_json(out / "ENTRYWISE_INFORMATION_SET.json", information)

    supports, support_seed_values = _load_supports(root, contexts)
    sentinel_orders: dict[str, Any] = {}
    null_maps: dict[str, Any] = {}
    for context in contexts:
        sentinel_orders[context] = {}
        null_maps[context] = {}
        for sequence in SUPPORT_SEQUENCES:
            sentinel_seed = _seed(SENTINEL_BASE_SEED, context, sequence)
            sentinel_orders[context][str(sequence)] = {
                "seed": sentinel_seed,
                "full_order": _permutation(interventions, sentinel_seed),
                "target_rule": "filter target intervention, then take first k",
            }
            null_maps[context][str(sequence)] = {
                "intervention_identity": _derangement(
                    interventions, _seed(NULL_BASE_SEED, f"intervention|{context}", sequence)
                ),
                "sentinel_identity": _derangement(
                    interventions, _seed(NULL_BASE_SEED, f"sentinel|{context}", sequence)
                ),
            }
    context_null_maps = {
        str(sequence): _derangement(
            contexts, _seed(NULL_BASE_SEED, "context", sequence)
        )
        for sequence in SUPPORT_SEQUENCES
    }

    fold_rows = list(csv.DictReader((root / "results/cgc_tahoe_0c/intervention_folds.csv").open(encoding="utf-8")))
    intervention_folds = {row["intervention_id"]: int(row["fold"]) for row in fold_rows}
    if set(intervention_folds) != set(interventions) or set(intervention_folds.values()) != set(range(5)):
        raise RuntimeError("Frozen intervention folds changed")

    split = {
        "phase": "CGC-EC-2",
        "frozen_at": _now(),
        "contexts": contexts,
        "interventions": interventions,
        "m_grid": M_VALUES,
        "k_grid": K_VALUES,
        "budget_formula": "93*m+k",
        "full_budget": 4_650,
        "all_but_one_budget": 4_649,
        "support_sequence_ids": SUPPORT_SEQUENCES,
        "support_source": "first eight outcome-independent CGC-SUPPORT-0C sequences",
        "support_source_base_seed": 202_608_203,
        "support_sequence_seed_values": support_seed_values,
        "context_support_orders": supports,
        "sentinel_base_seed": SENTINEL_BASE_SEED,
        "sentinel_orders": sentinel_orders,
        "inner_intervention_folds": intervention_folds,
        "inner_fold_assignment": "sha256_first8_little_endian_mod_5; outcome_balancing=False",
        "null_base_seed": NULL_BASE_SEED,
        "null_maps": null_maps,
        "context_identity_null_maps": context_null_maps,
        "reference_context_null_rule": "cyclically shift fitted weights by one position in the frozen support order; m=1 is not applicable",
        "synthetic_seed": SYNTHETIC_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": 10_000,
        "thresholds": [0.0, 0.25, 0.50, 0.80, 0.95],
        "rna_order_materialization": "deferred until primary integrity and positive-control gates pass; rank-32 baseline-only SVD Euclidean nearest contexts",
        "all_but_one_seed_collapse": True,
    }
    _write_json(out / "ENTRYWISE_SPLIT_MANIFEST.json", split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell"),
    )
    args = parser.parse_args()
    freeze(args.root, args.source_root)


if __name__ == "__main__":
    main()
