from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from igc_virtual_cell.cgc_entrywise.analysis import _null_maps, _orders
from igc_virtual_cell.cgc_entrywise.estimators import fit_batches
from igc_virtual_cell.cgc_entrywise.foundation import load_foundation
from igc_virtual_cell.cgc_entrywise.metrics import excess_geometry, matched_context_gram, matched_residual_cross


GENE_COUNT = 25_695
CONTEXTS = 50
INTERVENTIONS = 93
PLATES = ("plate6", "plate14")
BUDGETS = ((49, 92, 1), (40, 4, 8))
EXPECTED_G = {(49, 92): 0.11068053088808294, (40, 4): 0.07718711664021238}
TOLERANCE = 1e-10
SOURCE_HASHES = {
    "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json": "da3fc0f3422c12e2a1ec9fcdc6447cd137836a963290a262a82451430df1198d",
    "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz": "bd3b4f5097b4f158f58da102849d52080bb9ae7b7a4c4325f37bf574fd1a7c08",
    "data/cgc_entrywise_cache/same_plate6_float32.npy": "90681a0429339581a329536036b9353cc1761feb76a34b5cb5c64df34b28281f",
    "data/cgc_entrywise_cache/same_plate14_float32.npy": "9c1703a17855545f0dad3ba3d34c60c5187780612051bace36545ecd204567fd",
    "data/cgc_entrywise_cache/cross_plate6_plate14_float32.npy": "cece3ec0e2066bb46e9bc8bcdd11cb6666e486da3362358a15addaa1927909a3",
    "data/cgc_entrywise_cache/sample_gene_sums_float64.npy": "1f266acbc9ef8d84a63fe9166dc32988e14b882df6356c127f143e1b71655ee1",
}
TARGET_CACHE_AGGREGATE = "688cb5b0054f6f824981f8566bf72622b9bc991a425fd231c0fb8fcfe6d12d5d"
GENE_INDICES_SHA256 = "efea7718149514c763fbbad86d2337fde3377bf6c977365f0046576a85b94343"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _target_cache_aggregate(cache_root: Path) -> str:
    lines = [f"{path.name}:{sha256(path)}" for path in sorted(cache_root.glob("target_*.npz"))]
    if len(lines) != CONTEXTS:
        raise RuntimeError("RESOLUTION2_TARGET_CACHE_COUNT_MISMATCH")
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def validate_sources(source_root: Path, truth_root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = source_root / relative
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"RESOLUTION2_SOURCE_HASH_MISMATCH:{relative}")
        checks[relative] = observed
    cache_root = source_root / "results/cgc_entrywise_compression/_cache"
    aggregate = _target_cache_aggregate(cache_root)
    if aggregate != TARGET_CACHE_AGGREGATE:
        raise RuntimeError("RESOLUTION2_TARGET_CACHE_HASH_MISMATCH")
    checks["target_cache_aggregate"] = aggregate
    indices = truth_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy"
    if sha256(indices) != GENE_INDICES_SHA256:
        raise RuntimeError("RESOLUTION2_GENE_INDICES_HASH_MISMATCH")
    gene_indices = np.load(indices)
    if gene_indices.shape != (GENE_COUNT,) or not np.all(np.diff(gene_indices) > 0):
        raise RuntimeError("RESOLUTION2_GENE_AXIS_MISMATCH")
    store = zarr.open_group(truth_root / "results/cgc_tahoe_0i/response_tensors.zarr", mode="r")
    response = store["delta_primary"]
    if response.shape != (2, CONTEXTS, INTERVENTIONS, 62_710) or response.dtype != np.float32:
        raise RuntimeError("RESOLUTION2_TRUTH_ARRAY_MISMATCH")
    split = json.loads(
        (source_root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    if list(store.attrs["contexts"]) != split["contexts"] or list(store.attrs["interventions"]) != split["interventions"]:
        raise RuntimeError("RESOLUTION2_TRUTH_SPLIT_AXIS_MISMATCH")
    checks["gene_indices"] = GENE_INDICES_SHA256
    checks["truth_shape"] = list(response.shape)
    return checks


@dataclass(frozen=True)
class FrozenParameters:
    ridge: float
    rank: int


class ReplayContext:
    def __init__(self, analysis_root: Path, source_root: Path):
        self.analysis_root = analysis_root.resolve()
        self.source_root = source_root.resolve()
        self.split = json.loads(
            (self.source_root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        same6, same14, cross, sums, _ = load_foundation(self.source_root)
        self.same4 = (same6.reshape(CONTEXTS, INTERVENTIONS, CONTEXTS, INTERVENTIONS),
                      same14.reshape(CONTEXTS, INTERVENTIONS, CONTEXTS, INTERVENTIONS))
        self.cross4 = cross.reshape(CONTEXTS, INTERVENTIONS, CONTEXTS, INTERVENTIONS)
        self.sums = np.asarray(sums)
        self.matched_same = tuple(matched_context_gram(value) for value in self.same4)
        self.matched_cross = matched_context_gram(self.cross4)
        self.parameters = self._load_parameters()

    def _load_parameters(self) -> dict[tuple[int, int, int, int], FrozenParameters]:
        result: dict[tuple[int, int, int, int], FrozenParameters] = {}
        cache_root = self.source_root / "results/cgc_entrywise_compression/_cache"
        for target, path in enumerate(sorted(cache_root.glob("target_*.npz"))):
            with np.load(path, allow_pickle=False) as archive:
                rows = json.loads(str(archive["parameter_rows_json"]))
            for row in rows:
                m, k = int(row["m"]), int(row["k"])
                if (m, k) not in {(49, 92), (40, 4)}:
                    continue
                if int(row["target_context_index"]) != target or bool(row["target_outcome_used"]):
                    raise RuntimeError("RESOLUTION2_FROZEN_PARAMETER_LEAKAGE")
                plate = PLATES.index(str(row["plate"]))
                key = (target, plate, m, k)
                if key in result:
                    raise RuntimeError("RESOLUTION2_DUPLICATE_FROZEN_PARAMETER")
                result[key] = FrozenParameters(float(row["ridge_lambda"]), int(row["rank"]))
        if len(result) != CONTEXTS * 2 * 2:
            raise RuntimeError("RESOLUTION2_FROZEN_PARAMETER_COUNT_MISMATCH")
        return result

    def replay_weights(
        self,
        target: int,
        m: int,
        k: int,
        sequence: int,
        episodes: np.ndarray | None = None,
        same_override: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        support_orders, sentinel_orders = _orders(self.split, target, self.analysis_root, "random")
        sources = np.asarray(support_orders[sequence, :m], dtype=np.int64)
        context_null, _, sentinel_null = _null_maps(self.split, target, sequence)
        same_values = self.same4 if same_override is None else same_override
        matched = self.matched_same if same_override is None else tuple(matched_context_gram(x) for x in same_override)
        weights: list[np.ndarray] = []
        for plate in range(2):
            params = self.parameters[(target, plate, m, k)]
            batch = fit_batches(
                matched[plate],
                same_values[plate],
                target,
                sources,
                sentinel_orders[sequence],
                k,
                params.ridge,
                params.rank,
                context_null,
                sentinel_null,
                episodes=episodes,
            )
            weights.append(np.asarray(batch["M2"], dtype=np.float64))
        return sources, (weights[0], weights[1])

    def metric_replay(self) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, np.ndarray]]]:
        utility_path = self.source_root / "results/cgc_entrywise_compression/_cache/frozen_utility_table.npz"
        with np.load(utility_path, allow_pickle=False) as utility:
            models = list(map(str, utility["models"]))
            m_values = utility["m_values"].astype(int).tolist()
            k_values = utility["k_values"].astype(int).tolist()
            model_index = models.index("M2_AFFINE_RIDGE")
            frozen = {
                (m, k): (
                    np.asarray(utility["vtruth"][m_values.index(m)], dtype=np.float64),
                    np.asarray(utility["vafter"][model_index, m_values.index(m), k_values.index(k)], dtype=np.float64),
                )
                for m, k, _ in BUDGETS
            }
        rows: list[dict[str, Any]] = []
        replayed: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for m, k, sequence_count in BUDGETS:
            truth = np.zeros((CONTEXTS, INTERVENTIONS), dtype=np.float64)
            after = np.zeros_like(truth)
            for target in range(CONTEXTS):
                for sequence in range(sequence_count):
                    sources, weights = self.replay_weights(target, m, k, sequence)
                    geometry = excess_geometry(self.cross4, self.sums[0], target, sources)
                    truth[target] += np.diag(geometry.gram) / GENE_COUNT / sequence_count
                    after[target] += matched_residual_cross(
                        self.matched_cross, target, sources, weights[0], weights[1]
                    ) / GENE_COUNT / sequence_count
            expected_truth, expected_after = frozen[(m, k)]
            g = 1.0 - float(after.sum(dtype=np.float64)) / float(truth.sum(dtype=np.float64))
            expected_g = EXPECTED_G[(m, k)]
            rows.extend(
                [
                    {"check": f"gene_metric_m{m}_k{k}", "observed": g, "expected": expected_g,
                     "absolute_difference": abs(g - expected_g), "tolerance": TOLERANCE,
                     "passed": abs(g - expected_g) <= TOLERANCE},
                    {"check": f"vtruth_table_m{m}_k{k}", "observed": float(np.max(np.abs(truth - expected_truth))),
                     "expected": 0.0, "absolute_difference": float(np.max(np.abs(truth - expected_truth))),
                     "tolerance": TOLERANCE, "passed": bool(np.allclose(truth, expected_truth, atol=TOLERANCE, rtol=0))},
                    {"check": f"vafter_table_m{m}_k{k}", "observed": float(np.max(np.abs(after - expected_after))),
                     "expected": 0.0, "absolute_difference": float(np.max(np.abs(after - expected_after))),
                     "tolerance": TOLERANCE, "passed": bool(np.allclose(after, expected_after, atol=TOLERANCE, rtol=0))},
                ]
            )
            replayed[(m, k)] = {"vtruth": truth, "vafter": after}
        return rows, replayed


def prepare_truth_cache(truth_root: Path, replay_root: Path) -> Path:
    replay_root.mkdir(parents=True, exist_ok=True)
    output = replay_root / "truth_g_primary_float32.npy"
    expected_shape = (2, CONTEXTS, INTERVENTIONS, GENE_COUNT)
    if output.exists():
        existing = np.load(output, mmap_mode="r")
        if existing.shape != expected_shape or existing.dtype != np.float32:
            raise RuntimeError("RESOLUTION2_TRUTH_CACHE_MISMATCH")
        return output
    gene_indices = np.load(truth_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy").astype(np.int64)
    source = zarr.open_group(truth_root / "results/cgc_tahoe_0i/response_tensors.zarr", mode="r")["delta_primary"]
    target = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=expected_shape)
    write = 0
    for start in range(0, source.shape[-1], 1_024):
        stop = min(start + 1_024, source.shape[-1])
        left = int(np.searchsorted(gene_indices, start, side="left"))
        right = int(np.searchsorted(gene_indices, stop, side="left"))
        if right == left:
            continue
        local = gene_indices[left:right] - start
        target[..., write : write + right - left] = np.asarray(source[..., start:stop], dtype=np.float32)[..., local]
        write += right - left
    target.flush()
    if write != GENE_COUNT or not np.isfinite(np.load(output, mmap_mode="r")).all():
        raise RuntimeError("RESOLUTION2_TRUTH_CACHE_BUILD_FAIL")
    return output


def _prediction_path(replay_root: Path, m: int, k: int) -> Path:
    return replay_root / f"m{m}_k{k}_predictions_float32.npy"


def materialize_predictions(context: ReplayContext, truth_path: Path, replay_root: Path) -> dict[str, Any]:
    values = np.load(truth_path, mmap_mode="r")
    replay_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"files": {}, "weights": {}}
    for m, k, sequence_count in BUDGETS:
        path = _prediction_path(replay_root, m, k)
        shape = (2, sequence_count, CONTEXTS, INTERVENTIONS, GENE_COUNT)
        if path.exists():
            predictions = np.lib.format.open_memmap(path, mode="r+", dtype=np.float32, shape=shape)
        else:
            predictions = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        weights_all = np.empty((2, sequence_count, CONTEXTS, INTERVENTIONS, m), dtype=np.float64)
        sources_all = np.empty((sequence_count, CONTEXTS, m), dtype=np.int16)
        for target in range(CONTEXTS):
            for sequence in range(sequence_count):
                sources, weights = context.replay_weights(target, m, k, sequence)
                sources_all[sequence, target] = sources
                uniform = np.full(m, 1.0 / m, dtype=np.float64)
                for plate in range(2):
                    weights_all[plate, sequence, target] = weights[plate]
                    delta_weights = weights[plate] - uniform
                    for start in range(0, GENE_COUNT, 2_048):
                        stop = min(start + 2_048, GENE_COUNT)
                        block = np.asarray(values[plate, sources, :, start:stop], dtype=np.float64)
                        projected = np.einsum("ps,spg->pg", delta_weights, block, optimize=True)
                        predictions[plate, sequence, target, :, start:stop] = projected.astype(np.float32)
            predictions.flush()
            print(f"RESOLUTION2 materialize m={m} k={k} target={target + 1}/{CONTEXTS}", flush=True)
        weights_path = replay_root / f"m{m}_k{k}_frozen_weights.npz"
        np.savez_compressed(weights_path, weights=weights_all, sources=sources_all)
        manifest["files"][path.name] = {"shape": list(shape), "dtype": "float32", "size_bytes": path.stat().st_size}
        manifest["weights"][weights_path.name] = {"sha256": sha256(weights_path), "size_bytes": weights_path.stat().st_size}
    return manifest


def deterministic_and_leakage_checks(
    context: ReplayContext, truth_path: Path, replay_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = np.load(truth_path, mmap_mode="r")
    target, hidden, m, k, sequence = 0, 0, 40, 4, 0
    episodes = np.asarray([hidden], dtype=np.int64)
    sources_a, weights_a = context.replay_weights(target, m, k, sequence, episodes=episodes)
    sources_b, weights_b = context.replay_weights(target, m, k, sequence, episodes=episodes)
    weights_equal = np.array_equal(weights_a[0], weights_b[0], equal_nan=True) and np.array_equal(
        weights_a[1], weights_b[1], equal_nan=True
    )
    uniform = np.full(m, 1.0 / m, dtype=np.float64)
    vectors = []
    for weights in (weights_a, weights_b):
        plate_vectors = []
        for plate in range(2):
            block = np.asarray(values[plate, sources_a, hidden], dtype=np.float64)
            plate_vectors.append(((weights[plate][hidden] - uniform) @ block).astype(np.float32))
        vectors.append(plate_vectors)
    vector_equal = all(np.array_equal(vectors[0][plate], vectors[1][plate]) for plate in range(2))
    rows.append({"check": "deterministic_weights_bitwise", "observed": int(weights_equal), "expected": 1,
                 "absolute_difference": int(not weights_equal), "tolerance": 0, "passed": weights_equal})
    rows.append({"check": "deterministic_vectors_bitwise", "observed": int(vector_equal), "expected": 1,
                 "absolute_difference": int(not vector_equal), "tolerance": 0, "passed": vector_equal})

    adversarial = []
    for plate in range(2):
        changed = np.asarray(context.same4[plate], dtype=np.float64).copy()
        changed[target, hidden, :, :] = 1e30
        changed[:, :, target, hidden] = -1e30
        adversarial.append(changed)
    _, weights_adv = context.replay_weights(target, m, k, sequence, episodes=episodes,
                                             same_override=(adversarial[0], adversarial[1]))
    leakage_weights_equal = np.array_equal(weights_a[0], weights_adv[0], equal_nan=True) and np.array_equal(
        weights_a[1], weights_adv[1], equal_nan=True
    )
    rows.append({"check": "hidden_target_adversarial_weight_invariance", "observed": int(leakage_weights_equal),
                 "expected": 1, "absolute_difference": int(not leakage_weights_equal), "tolerance": 0,
                 "passed": leakage_weights_equal})
    stored = np.load(_prediction_path(replay_root, m, k), mmap_mode="r")[:, sequence, target, hidden]
    stored_equal = all(np.array_equal(stored[plate], vectors[0][plate]) for plate in range(2))
    rows.append({"check": "stored_vector_matches_replay", "observed": int(stored_equal), "expected": 1,
                 "absolute_difference": int(not stored_equal), "tolerance": 0, "passed": stored_equal})
    return rows


def write_output_manifest(path: Path, payload: dict[str, Any], replay_root: Path) -> dict[str, Any]:
    complete = dict(payload)
    complete["prediction_files"] = {}
    for m, k, _ in BUDGETS:
        prediction = _prediction_path(replay_root, m, k)
        complete["prediction_files"][prediction.name] = {
            "path": str(prediction.resolve()),
            "size_bytes": prediction.stat().st_size,
            "sha256": sha256(prediction),
        }
    truth = replay_root / "truth_g_primary_float32.npy"
    complete["truth_cache"] = {"path": str(truth.resolve()), "size_bytes": truth.stat().st_size,
                               "sha256": sha256(truth)}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return complete
