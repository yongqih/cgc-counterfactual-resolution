from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .affine import (
    historical_one_entry_reconciliation,
    load_frozen_affine_parameters,
    matched_affine_prediction_weights,
)
from .core import CONTEXTS, ENTRIES, INTERVENTIONS, prediction_weights
from .design import SYNTHETIC_SEED, frozen_masks
from .evaluation import evaluate_fold
from .fitting import select_and_refit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cache_paths(source_root: Path, kind: str, fold: int) -> tuple[Path, Path]:
    cache = source_root / "data/cgc_lowrank_completion" / kind
    return cache / f"fold_{fold:03d}.npz", cache / f"fold_{fold:03d}.COMPLETE.json"


def _valid_cache(source_root: Path, kind: str, fold: int) -> bool:
    archive, manifest = _cache_paths(source_root, kind, fold)
    if not archive.exists() or not manifest.exists():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "COMPLETE"
        and int(payload.get("outer_fold", -1)) == fold
        and payload.get("archive_sha256") == sha256(archive)
    )


def synthetic_grams(source_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    cache = source_root / "data/cgc_lowrank_completion/synthetic"
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / "same_plate6_float32.npy", cache / "same_plate14_float32.npy", cache / "cross_float32.npy"]
    meta_path = cache / "synthetic_manifest.json"
    if all(path.exists() for path in paths) and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("seed") != SYNTHETIC_SEED or meta.get("known_rank") != 4:
            raise RuntimeError("LOW_RANK_SYNTHETIC_CACHE_MISMATCH")
        return tuple(np.load(path, mmap_mode="r") for path in paths) + (meta,)  # type: ignore[return-value]

    rng = np.random.default_rng(SYNTHETIC_SEED)
    genes = 512
    rank = 4
    intervention_main = rng.normal(0.0, 0.8, size=(INTERVENTIONS, genes))
    u = rng.normal(size=(CONTEXTS, rank))
    v = rng.normal(size=(INTERVENTIONS, rank))
    u -= u.mean(axis=0, keepdims=True)
    v -= v.mean(axis=0, keepdims=True)
    q = rng.normal(0.0, 1.0 / np.sqrt(rank), size=(rank, genes))
    interaction = np.einsum("cr,pr,rg->cpg", u, v, q, optimize=True)
    interaction_scale = float(np.sqrt(np.mean(interaction**2)))
    target_reliability = 0.90
    noise_sd = interaction_scale * np.sqrt((1.0 - target_reliability) / target_reliability)
    shared = intervention_main[None, :, :] + interaction
    plate6 = shared + rng.normal(0.0, noise_sd, size=shared.shape)
    plate14 = shared + rng.normal(0.0, noise_sd, size=shared.shape)
    y6 = plate6.reshape(ENTRIES, genes).astype(np.float32)
    y14 = plate14.reshape(ENTRIES, genes).astype(np.float32)
    grams = (
        np.asarray(y6 @ y6.T, dtype=np.float32),
        np.asarray(y14 @ y14.T, dtype=np.float32),
        np.asarray(y6 @ y14.T, dtype=np.float32),
    )
    for path, gram in zip(paths, grams, strict=True):
        np.save(path, gram)
    meta = {
        "seed": SYNTHETIC_SEED,
        "known_rank": rank,
        "genes": genes,
        "target_interaction_reliability": target_reliability,
        "interaction_rms": interaction_scale,
        "noise_sd": noise_sd,
        "source_outcome_used_for_tuning": False,
        "hashes": {path.name: sha256(path) for path in paths},
    }
    _write_json_atomic(meta_path, meta)
    return grams + (meta,)


def load_real_grams(source_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = source_root / "data/cgc_entrywise_cache"
    return (
        np.load(cache / "same_plate6_float32.npy", mmap_mode="r"),
        np.load(cache / "same_plate14_float32.npy", mmap_mode="r"),
        np.load(cache / "cross_plate6_plate14_float32.npy", mmap_mode="r"),
    )


def run_fold(
    root: Path,
    source_root: Path,
    fold: int,
    *,
    kind: str,
    device: str = "cuda",
    overwrite: bool = False,
) -> Path:
    root = root.resolve()
    source_root = source_root.resolve()
    if kind not in {"real", "synthetic"}:
        raise ValueError("kind must be real or synthetic")
    archive_path, manifest_path = _cache_paths(source_root, kind, fold)
    if not overwrite and _valid_cache(source_root, kind, fold):
        return archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    masks = frozen_masks()
    targets = masks.outer_targets(fold)
    train = masks.outer_training(fold)
    inner_validation = masks.inner_validation(fold)
    inner_train = masks.inner_training(fold)
    if kind == "real":
        same6, same14, cross = load_real_grams(source_root)
    else:
        same6, same14, cross, _ = synthetic_grams(source_root)
    same = (same6, same14)
    fit_rows: list[dict[str, object]] = []
    additive: list[np.ndarray] = []
    lowrank: list[np.ndarray] = []
    selected_rows: list[dict[str, object]] = []
    started = perf_counter()
    for plate in range(2):
        selected = select_and_refit(
            same[plate],
            train,
            inner_train,
            inner_validation,
            plate=plate,
            fold=fold,
            device=device,
        )
        weights = prediction_weights(train, targets, selected.fit)
        additive.append(weights.additive)
        lowrank.append(weights.full)
        fit_rows.extend(selected.rows)
        selected_rows.append(
            {
                "plate": plate,
                "rank": selected.fit.rank,
                "ridge": selected.fit.ridge,
                "seed": selected.fit.seed,
                "converged": selected.fit.converged,
                "steps": selected.fit.steps,
                "objective_final": selected.fit.objective_final,
                "peak_memory_bytes": selected.fit.peak_memory_bytes,
            }
        )
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "ADDITIVE_MAIN_EFFECT": (additive[0], additive[1]),
        "LOW_RANK_INTERACTION": (lowrank[0], lowrank[1]),
    }
    affine: list[np.ndarray] = []
    if kind == "real":
        parameters = load_frozen_affine_parameters(source_root)
        for plate in range(2):
            affine.append(
                matched_affine_prediction_weights(
                    same[plate], train, targets, parameters.ridge[plate]
                )
            )
        models["MATCHED_AFFINE_RIDGE"] = (affine[0], affine[1])
    evaluated = evaluate_fold(same6, same14, cross, train, targets, models, fold, device=device)
    arrays: dict[str, object] = {
        "train": train,
        "targets": targets,
        "additive6": additive[0].astype(np.float32),
        "additive14": additive[1].astype(np.float32),
        "lowrank6": lowrank[0].astype(np.float32),
        "lowrank14": lowrank[1].astype(np.float32),
        "fit_rows_json": np.asarray(json.dumps(fit_rows)),
        "selected_rows_json": np.asarray(json.dumps(selected_rows)),
        "prediction_rows_json": np.asarray(json.dumps(evaluated.prediction_rows)),
        "diagnostic_rows_json": np.asarray(json.dumps(evaluated.diagnostic_rows)),
    }
    if affine:
        arrays["affine6"] = affine[0].astype(np.float32)
        arrays["affine14"] = affine[1].astype(np.float32)
    temporary = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, archive_path)
    payload = {
        "status": "COMPLETE",
        "kind": kind,
        "outer_fold": fold,
        "outer_targets": len(targets),
        "outer_observed": len(train),
        "outer_observed_fraction": len(train) / ENTRIES,
        "inner_validation": len(inner_validation),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "selected": selected_rows,
        "runtime_seconds": perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "target_outcome_used": False,
    }
    _write_json_atomic(manifest_path, payload)
    return archive_path


def run_range(
    root: Path,
    source_root: Path,
    start_fold: int,
    stop_fold: int,
    *,
    kind: str,
    device: str = "cuda",
) -> None:
    for fold in range(start_fold, stop_fold):
        started = perf_counter()
        path = run_fold(root, source_root, fold, kind=kind, device=device)
        print(
            f"LOW_RANK {kind} fold={fold:03d} complete path={path} wall={perf_counter()-started:.3f}s",
            flush=True,
        )


def run_invariance(root: Path, source_root: Path, fold: int = 0, device: str = "cuda") -> Path:
    root = root.resolve()
    source_root = source_root.resolve()
    out = root / "results/cgc_lowrank_interaction_completion"
    masks = frozen_masks()
    targets = masks.outer_targets(fold)
    train = masks.outer_training(fold)
    inner_validation = masks.inner_validation(fold)
    inner_train = masks.inner_training(fold)
    same6, _, _ = load_real_grams(source_root)
    selected_a = select_and_refit(
        same6, train, inner_train, inner_validation, plate=0, fold=fold, device=device
    )
    changed = np.asarray(same6, dtype=np.float32).copy()
    changed[targets, :] = np.float32(1e20)
    changed[:, targets] = np.float32(-1e20)
    selected_b = select_and_refit(
        changed, train, inner_train, inner_validation, plate=0, fold=fold, device=device
    )
    weights_a = prediction_weights(train, targets, selected_a.fit)
    weights_b = prediction_weights(train, targets, selected_b.fit)
    parameters = load_frozen_affine_parameters(source_root)
    affine_a = matched_affine_prediction_weights(same6, train, targets, parameters.ridge[0])
    affine_b = matched_affine_prediction_weights(changed, train, targets, parameters.ridge[0])
    benchmark_target = int(targets[0])
    reconcile = historical_one_entry_reconciliation(
        source_root, same6, benchmark_target, 0, parameters
    )
    checks = {
        "outer_fold": fold,
        "outer_targets": len(targets),
        "selected_rank_identical": selected_a.fit.rank == selected_b.fit.rank,
        "selected_ridge_identical": selected_a.fit.ridge == selected_b.fit.ridge,
        "factors_u_bitwise": np.array_equal(selected_a.fit.u, selected_b.fit.u),
        "factors_v_bitwise": np.array_equal(selected_a.fit.v, selected_b.fit.v),
        "additive_coefficients_bitwise": np.array_equal(weights_a.additive, weights_b.additive),
        "interaction_coefficients_bitwise": np.array_equal(weights_a.interaction, weights_b.interaction),
        "full_coefficients_bitwise": np.array_equal(weights_a.full, weights_b.full),
        "affine_coefficients_bitwise": np.array_equal(affine_a, affine_b),
        "historical_one_entry_max_abs_weight_difference": reconcile,
        "historical_one_entry_reconciled": reconcile <= 1e-8,
    }
    checks["passed"] = all(
        bool(value)
        for key, value in checks.items()
        if key not in {"outer_fold", "outer_targets", "historical_one_entry_max_abs_weight_difference"}
    )
    path = out / "LOW_RANK_OUTCOME_INVARIANCE.json"
    _write_json_atomic(path, checks)
    if not checks["passed"]:
        raise RuntimeError("LOW_RANK_OUTCOME_INVARIANCE_FAIL")
    return path
