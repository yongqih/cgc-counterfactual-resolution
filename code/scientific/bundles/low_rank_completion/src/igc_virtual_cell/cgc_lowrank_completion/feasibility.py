from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .design import benchmark_targets, frozen_masks
from .fitting import select_and_refit


EXPECTED_HASHES = {
    "same_plate6_float32.npy": "90681a0429339581a329536036b9353cc1761feb76a34b5cb5c64df34b28281f",
    "same_plate14_float32.npy": "9c1703a17855545f0dad3ba3d34c60c5187780612051bace36545ecd204567fd",
    "cross_plate6_plate14_float32.npy": "cece3ec0e2066bb46e9bc8bcdd11cb6666e486da3362358a15addaa1927909a3",
    "sample_gene_sums_float64.npy": "1f266acbc9ef8d84a63fe9166dc32988e14b882df6356c127f143e1b71655ee1",
    "truth_g_primary_float32.npy": "005603c5a01c085762c9eecb266de6302f95356ca5eb4026f8c0380f57793f73",
    "m49_k92_predictions_float32.npy": "a1e99f8a949b8b205d1b360dd9e6919229e6b41969e98fdc57139f6983f798d6",
    "hierarchical_bootstrap_weights_int16.npy": "f0a9697175fe2d4b8bc800e4352d961ed349f9e9a33df138c793be24f982c9cf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def run_feasibility(root: Path, source_root: Path, device: str = "cuda") -> Path:
    root = root.resolve()
    source_root = source_root.resolve()
    out = root / "results/cgc_lowrank_interaction_completion"
    out.mkdir(parents=True, exist_ok=True)
    cache = source_root / "data/cgc_entrywise_cache"
    replay = source_root / "data/cgc_resolution2_replay"
    paths = {
        "same_plate6_float32.npy": cache / "same_plate6_float32.npy",
        "same_plate14_float32.npy": cache / "same_plate14_float32.npy",
        "cross_plate6_plate14_float32.npy": cache / "cross_plate6_plate14_float32.npy",
        "sample_gene_sums_float64.npy": cache / "sample_gene_sums_float64.npy",
        "truth_g_primary_float32.npy": replay / "truth_g_primary_float32.npy",
        "m49_k92_predictions_float32.npy": replay / "m49_k92_predictions_float32.npy",
        "hierarchical_bootstrap_weights_int16.npy": replay / "hierarchical_bootstrap_weights_int16.npy",
    }
    hash_rows = []
    for name, path in paths.items():
        observed = _sha256(path)
        expected = EXPECTED_HASHES[name]
        hash_rows.append({"artifact": str(path), "sha256": observed, "expected_sha256": expected, "passed": observed == expected})
    if not all(row["passed"] for row in hash_rows):
        raise RuntimeError("LOW_RANK_SOURCE_HASH_FAIL")
    split = json.loads(
        (root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    targets = benchmark_targets(split["contexts"], split["interventions"])
    masks = frozen_masks()
    grams = [np.load(paths["same_plate6_float32.npy"], mmap_mode="r"), np.load(paths["same_plate14_float32.npy"], mmap_mode="r")]
    rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    benchmark_started = perf_counter()
    for target in map(int, targets):
        fold = int(masks.outer_fold[target])
        outer_train = np.asarray([index for index in range(4_650) if index != target], dtype=np.int64)
        inner_validation = masks.inner_validation(fold)
        inner_validation = inner_validation[inner_validation != target]
        keep = np.ones(4_650, dtype=bool)
        keep[target] = False
        keep[inner_validation] = False
        inner_train = np.flatnonzero(keep).astype(np.int64)
        for plate in range(2):
            started = perf_counter()
            selected = select_and_refit(
                grams[plate], outer_train, inner_train, inner_validation, plate=plate, fold=fold, device=device
            )
            seconds = perf_counter() - started
            rows.append(
                {
                    "target_index": target,
                    "context_id": split["contexts"][target // 93],
                    "intervention_id": split["interventions"][target % 93],
                    "plate": plate,
                    "wall_seconds": seconds,
                    "selected_rank": selected.fit.rank,
                    "selected_ridge": selected.fit.ridge,
                    "selected_final_converged": selected.fit.converged,
                    "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in selected.rows),
                    "outer_target_scored": False,
                }
            )
            fit_rows.extend(selected.rows)
    elapsed = perf_counter() - benchmark_started
    benchmark = pd.DataFrame(rows)
    fits = pd.DataFrame(fit_rows)
    benchmark.to_csv(out / "LOW_RANK_FEASIBILITY_BENCHMARK.csv", index=False)
    fits.to_csv(out / "LOW_RANK_FEASIBILITY_FITS.csv", index=False)
    per_target_both_plates = float(benchmark.groupby("target_index").wall_seconds.sum().mean())
    projected_hours = per_target_both_plates * 4_650 / 3600.0
    peak = int(benchmark.peak_memory_bytes.max())
    convergence = float(fits.converged.mean())
    exact = projected_hours <= 24.0 and peak <= int(10.5 * 1024**3) and convergence >= 0.95
    design = "EXACT_ALL_BUT_ONE" if exact else "BALANCED_100_FOLD_ENTRYWISE_CROSSFIT"
    report = f"""# Low-rank interaction completion execution feasibility

Frozen protocol commit: `{_git(root, 'rev-parse', 'HEAD')}`.
Generated: {datetime.now(timezone.utc).astimezone().isoformat()}.

The benchmark executed the full frozen rank-by-L2 grid, both deterministic
starts, train-side selection, and final refit for four SHA256-selected target
episodes on both independent plates. Outer target outcomes were never scored.

- Representative target episodes: {len(targets)}.
- Timed plate-specific target fits: {len(benchmark)}.
- Total benchmark wall time: {elapsed:.3f} seconds.
- Mean wall time per target across both plates: {per_target_both_plates:.3f} seconds.
- Projected exact 4,650-target runtime: {projected_hours:.3f} GPU-hours.
- Peak allocated model VRAM: {peak / 1024**3:.3f} GiB.
- Candidate/final convergence fraction: {convergence:.3%}.
- Frozen exact gate: <=24 GPU-hours, <=10.5 GiB, >=95% convergence.
- Gate result: {'PASS' if exact else 'FAIL'}.
- Frozen execution design: `{design}`.

Because the gate is outcome-blind, this decision cannot favor either model.
The fallback, when selected, predicts every entry exactly once at pooled 99.0%
observed support and reruns affine Ridge on the identical outer masks.

## Input integrity

All seven frozen inputs matched their preregistered SHA256 values. Detailed
hashes are stored in `LOW_RANK_FEASIBILITY_INPUT_HASHES.csv`.
"""
    (out / "LOW_RANK_EXECUTION_FEASIBILITY.md").write_text(report, encoding="utf-8")
    pd.DataFrame(hash_rows).to_csv(out / "LOW_RANK_FEASIBILITY_INPUT_HASHES.csv", index=False)
    decision = {
        "protocol_commit": _git(root, "rev-parse", "HEAD"),
        "exact_all_but_one_executable": exact,
        "selected_design": design,
        "projected_exact_gpu_hours": projected_hours,
        "peak_memory_bytes": peak,
        "convergence_fraction": convergence,
        "benchmark_target_indices": list(map(int, targets)),
        "outer_targets_inspected": False,
    }
    (out / "LOW_RANK_FEASIBILITY_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return out / "LOW_RANK_EXECUTION_FEASIBILITY.md"
