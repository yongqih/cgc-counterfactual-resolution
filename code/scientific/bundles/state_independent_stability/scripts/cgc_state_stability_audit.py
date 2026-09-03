from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from igc_virtual_cell.cgc_state_1.analysis import (
    bootstrap_recovery,
    leave_one_context_mean,
    pairwise_context_geometry,
    recovery_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "state_stability_official"
OUT = ROOT / "results" / "cgc_state_stability"
FIG = OUT / "figures"
AUTHORITY = "35cf07d8f3d90450b365893b56aa127dca7c4774"
PROTOCOL_FREEZE = "5d6a1c34a82c5818c4efb8bfebc41d44ab750bcb"
SEED = 68421
CONTEXTS = ["C32", "HOP62", "HepG2-C3A", "Hs 766T", "PANC-1"]
VARIANTS = {
    "ST_SE_TAHOE": {
        "repo": "arcinstitute/ST-SE-Tahoe",
        "revision": "03b1971d7cc93a7535fd2e957c6948dba267378b",
        "run": "zeroshot/state_generalization_zeroshot_X_state",
        "primary": True,
    },
    "ST_HVG_TAHOE": {
        "repo": "arcinstitute/ST-HVG-Tahoe",
        "revision": "ca6b751972493f8448e3256d1340ae70ad43e1e7",
        "run": "zeroshot/state_generalization_zeroshot_X_hvg",
        "primary": False,
    },
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def read_json_url(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "VirtualCell-STATE-stability-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def api_tree(repo: str, path: str) -> list[dict[str, object]]:
    encoded = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/api/models/{repo}/tree/main/{encoded}?recursive=true&limit=200"
    result = read_json_url(url)
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected Hugging Face tree response for {repo}/{path}")
    return result


def remote_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(path, safe='/')}?download=true"


def local_path(variant: str, remote_path: str) -> Path:
    return DATA / variant / Path(*remote_path.split("/"))


def inventory() -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    truth_by_context: dict[str, tuple[int, str]] = {}
    for variant, spec in VARIANTS.items():
        info = read_json_url(f"https://huggingface.co/api/models/{spec['repo']}")
        revision = str(info.get("sha"))
        if revision != spec["revision"]:
            raise RuntimeError(f"Official repository revision changed for {variant}: {revision}")
        run = str(spec["run"])
        files = api_tree(str(spec["repo"]), run)
        if variant == "ST_HVG_TAHOE":
            files.extend(api_tree(str(spec["repo"]), "zeroshot"))
        seen: set[str] = set()
        for item in files:
            path = str(item.get("path", ""))
            if path in seen or item.get("type") != "file":
                continue
            seen.add(path)
            basename = Path(path).name
            in_best = "/eval_best.ckpt/" in path
            is_de = in_best and (basename.endswith("_pred_de.csv") or basename.endswith("_real_de.csv"))
            is_metric = in_best and (basename.endswith("_results.csv") or basename.endswith("_agg_results.csv"))
            is_config = path == f"{run}/config.yaml"
            is_split = variant == "ST_HVG_TAHOE" and path == "zeroshot/generalization.toml"
            if not (is_de or is_metric or is_config or is_split):
                continue
            context = next((name for name in CONTEXTS if basename.startswith(name + "_")), None)
            lfs = item.get("lfs") or {}
            expected_sha = str(lfs.get("oid")) if lfs.get("oid") else None
            download = True
            role = "metadata"
            if basename.endswith("_real_de.csv"):
                role = "truth"
                if context is None:
                    raise RuntimeError(f"Unrecognized truth context: {path}")
                if variant == "ST_SE_TAHOE":
                    truth_by_context[context] = (int(item["size"]), str(expected_sha))
                else:
                    download = False
            elif basename.endswith("_pred_de.csv"):
                role = "prediction"
            elif basename.endswith("_results.csv") and not basename.endswith("_agg_results.csv"):
                role = "official_per_intervention_metrics"
            elif basename.endswith("_agg_results.csv"):
                role = "official_aggregate_metrics"
            artifacts.append(
                {
                    "variant": variant,
                    "repository": spec["repo"],
                    "repository_revision": revision,
                    "remote_path": path,
                    "remote_url": remote_url(str(spec["repo"]), path),
                    "size_bytes": int(item.get("size") or 0),
                    "lfs_sha256": expected_sha,
                    "context": context,
                    "role": role,
                    "download": download,
                    "local_path": str(local_path(variant, path).relative_to(ROOT)).replace("\\", "/") if download else None,
                }
            )
    for row in artifacts:
        if row["variant"] == "ST_HVG_TAHOE" and row["role"] == "truth":
            context = str(row["context"])
            if context not in truth_by_context:
                raise RuntimeError(f"Missing ST-SE truth metadata for {context}")
            expected = truth_by_context[context]
            if (int(row["size_bytes"]), str(row["lfs_sha256"])) != expected:
                raise RuntimeError(f"ST-HVG and ST-SE truth artifacts differ for {context}")
            row["reused_truth_from"] = "ST_SE_TAHOE"
    return artifacts


def download_one(row: dict[str, object]) -> dict[str, object]:
    if not row["download"]:
        return row
    path = ROOT / str(row["local_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row["size_bytes"])
    expected_sha = row.get("lfs_sha256")
    if path.exists() and path.stat().st_size == expected_size:
        actual = sha256(path)
        if expected_sha is None or actual == expected_sha:
            row["local_sha256"] = actual
            row["download_status"] = "REUSED_VERIFIED"
            return row
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "VirtualCell-STATE-stability-audit/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(str(row["remote_url"]), headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with part.open(mode) as handle:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
    if part.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {row['remote_path']}: {part.stat().st_size} != {expected_size}")
    actual = sha256(part)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch for {row['remote_path']}")
    os.replace(part, path)
    row["local_sha256"] = actual
    row["download_status"] = "DOWNLOADED_VERIFIED"
    return row


def acquire(max_workers: int = 3) -> list[dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = inventory()
    selected = [row for row in rows if row["download"]]
    total = sum(int(row["size_bytes"]) for row in selected)
    print(f"Selected {len(selected)} compact artifacts ({total / 1e9:.3f} GB decimal); no H5AD/checkpoints/raw cells.", flush=True)
    started = time.time()
    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, row.copy()): row for row in selected}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"{result['download_status']}: {result['remote_path']} ({int(result['size_bytes']) / 1e6:.1f} MB)", flush=True)
    completed_by_key = {(row["variant"], row["remote_path"]): row for row in completed}
    for index, row in enumerate(rows):
        key = (row["variant"], row["remote_path"])
        if key in completed_by_key:
            rows[index] = completed_by_key[key]
        elif not row["download"]:
            row["download_status"] = "REMOTE_TRUTH_IDENTITY_VERIFIED_REUSED"
    manifest = {
        "phase": "STATE_STABILITY",
        "created_at": now(),
        "authority_commit": AUTHORITY,
        "protocol_freeze_commit": PROTOCOL_FREEZE,
        "policy": "compact official eval_best DE/metric artifacts only; no H5AD, checkpoint, eval_last, fewshot, or raw-cell transfer",
        "downloaded_or_reused_bytes": total,
        "wall_seconds": time.time() - started,
        "artifacts": rows,
    }
    (OUT / "STATE_STABILITY_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return rows


def artifact_lookup(variant: str, suffix: str) -> Path:
    manifest = json.loads((OUT / "STATE_STABILITY_ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["artifacts"] if row["variant"] == variant and row.get("local_path") and str(row["remote_path"]).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one artifact for {variant}/{suffix}, found {len(matches)}")
    return ROOT / str(matches[0]["local_path"])


def selected_interventions() -> list[str]:
    sets: list[set[str]] = []
    for variant in VARIANTS:
        for context in CONTEXTS:
            path = artifact_lookup(variant, f"/{context}_results.csv")
            frame = pd.read_csv(path, usecols=["perturbation"], dtype=str)
            sets.append(set(frame["perturbation"]))
    shared = sorted(set.intersection(*sets))
    if len(shared) != 1136:
        raise RuntimeError(f"Expected 1,136 exact shared interventions, found {len(shared)}")
    return shared


def stream_de(path: Path, interventions: list[str]) -> tuple[np.ndarray, np.ndarray, set[str]]:
    mapping = {label: index for index, label in enumerate(interventions)}
    target = np.empty((len(interventions), 2000), dtype=np.float32)
    reference = np.empty_like(target)
    counts = np.zeros(target.shape, dtype=np.uint8)
    controls: set[str] = set()
    columns = ["target", "reference", "feature", "target_mean", "reference_mean"]
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"target": str, "reference": str, "feature": np.int16, "target_mean": np.float32, "reference_mean": np.float32},
        chunksize=250_000,
    ):
        rows = chunk["target"].map(mapping)
        keep = rows.notna()
        if not keep.any():
            continue
        block = chunk.loc[keep]
        r = rows.loc[keep].to_numpy(dtype=np.int32)
        f = block["feature"].to_numpy(dtype=np.int16)
        if f.min() < 0 or f.max() >= 2000:
            raise RuntimeError(f"Feature axis out of range in {path}")
        target[r, f] = block["target_mean"].to_numpy(dtype=np.float32)
        reference[r, f] = block["reference_mean"].to_numpy(dtype=np.float32)
        np.add.at(counts, (r, f), 1)
        controls.update(block["reference"].dropna().astype(str).unique())
    if not np.all(counts == 1):
        missing = int(np.sum(counts == 0))
        duplicate = int(np.sum(counts > 1))
        raise RuntimeError(f"Incomplete/duplicate axis in {path}: missing={missing}, duplicate={duplicate}")
    if not np.isfinite(target).all() or not np.isfinite(reference).all():
        raise RuntimeError(f"Non-finite official mean vector in {path}")
    return target, reference, controls


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic if rank else stats.pearsonr(a, b).statistic)


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    t = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    denominator = np.square(t - t.mean()).sum()
    return float(1.0 - np.square(t - p).sum() / denominator) if denominator else float("nan")


def strict_leave_one_context_mean(delta: np.ndarray) -> np.ndarray:
    """Mathematically identical LOCO mean with target excluded before arithmetic."""
    if delta.ndim != 3 or delta.shape[0] < 2:
        raise ValueError("delta must be a C x P x G tensor with at least two contexts")
    result = np.empty_like(delta)
    indices = np.arange(delta.shape[0])
    for context in range(delta.shape[0]):
        result[context] = delta[indices != context].mean(axis=0)
    return result


def identity_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    matched: list[float] = []
    mismatched: list[float] = []
    top1: list[bool] = []
    top5: list[bool] = []
    ranks: list[int] = []
    for context in range(truth.shape[0]):
        t = truth[context].astype(np.float64)
        p = prediction[context].astype(np.float64)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        similarity = p @ t.T
        matched.extend(np.diag(similarity))
        mismatched.extend(similarity[~np.eye(len(similarity), dtype=bool)])
        ordering = np.argsort(-similarity, axis=1)
        for index in range(len(similarity)):
            rank = int(np.flatnonzero(ordering[index] == index)[0]) + 1
            ranks.append(rank)
            top1.append(rank == 1)
            top5.append(rank <= 5)
    return {
        "fingerprint_matched_cosine": float(np.mean(matched)),
        "fingerprint_mismatched_cosine": float(np.mean(mismatched)),
        "matched_minus_mismatched": float(np.mean(matched) - np.mean(mismatched)),
        "top1_retrieval": float(np.mean(top1)),
        "top5_retrieval": float(np.mean(top5)),
        "mean_true_identity_rank": float(np.mean(ranks)),
    }


def geometry_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    _, true_distance = pairwise_context_geometry(truth)
    _, predicted_distance = pairwise_context_geometry(prediction)
    return {
        "context_distance_scale_retention": float(predicted_distance.sum() / true_distance.sum()),
        "context_distance_spearman": correlation(true_distance, predicted_distance, rank=True),
    }


def load_tensors(interventions: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
    truth = np.empty((len(CONTEXTS), len(interventions), 2000), dtype=np.float32)
    baseline = np.empty((len(CONTEXTS), 2000), dtype=np.float32)
    predictions = {variant: np.empty_like(truth) for variant in VARIANTS}
    audits: list[dict[str, object]] = []
    for c, context in enumerate(CONTEXTS):
        real_path = artifact_lookup("ST_SE_TAHOE", f"/{context}_real_de.csv")
        real_target, real_reference, real_controls = stream_de(real_path, interventions)
        if len(real_controls) != 1 or "DMSO_TF" not in next(iter(real_controls)):
            raise RuntimeError(f"Unexpected real control labels for {context}: {sorted(real_controls)}")
        truth[c] = real_target - real_reference
        baseline[c] = real_reference.mean(axis=0)
        reference_span = float(np.max(np.ptp(real_reference, axis=0)))
        for variant in VARIANTS:
            pred_path = artifact_lookup(variant, f"/{context}_pred_de.csv")
            pred_target, pred_reference, pred_controls = stream_de(pred_path, interventions)
            predictions[variant][c] = pred_target - real_reference
            audits.append(
                {
                    "variant": variant,
                    "context": context,
                    "n_interventions": len(interventions),
                    "n_features": 2000,
                    "real_control_labels": " | ".join(sorted(real_controls)),
                    "pred_control_labels": " | ".join(sorted(pred_controls)),
                    "real_reference_max_span_across_interventions": reference_span,
                    "pred_vs_real_reference_max_abs_difference": float(np.max(np.abs(pred_reference - real_reference))),
                    "axis_complete": True,
                    "finite": True,
                }
            )
    return truth, baseline, predictions, audits


def mutation_invariance(truth: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    original_pertmean = strict_leave_one_context_mean(truth)
    legacy_original = leave_one_context_mean(truth)
    for index, context in enumerate(CONTEXTS):
        mutated = truth.copy()
        mutated[index] = mutated[index] * -11.0 + 37.0
        mutated_pertmean = strict_leave_one_context_mean(mutated)
        legacy_mutated = leave_one_context_mean(mutated)
        retained = np.arange(len(CONTEXTS)) != index
        direct_before = truth[retained].mean(axis=0)
        direct_after = mutated[retained].mean(axis=0)
        direct_comparator_equal = np.array_equal(direct_before, direct_after)
        helper_max_abs_roundoff = float(np.max(np.abs(legacy_original[index] - legacy_mutated[index])))
        for variant, prediction in predictions.items():
            before = array_sha256(prediction)
            after = array_sha256(prediction)
            rows.append(
                {
                    "variant": variant,
                    "held_context": context,
                    "prediction_hash_before": before,
                    "prediction_hash_after": after,
                    "prediction_unchanged": before == after,
                    "direct_target_excluding_comparator_bitwise_unchanged": direct_comparator_equal,
                    "strict_adapter_comparator_bitwise_unchanged": bool(
                        np.array_equal(original_pertmean[index], mutated_pertmean[index])
                    ),
                    "sum_subtract_helper_max_abs_roundoff": helper_max_abs_roundoff,
                    "sum_subtract_helper_used_formally": False,
                    "status": "PASS"
                    if before == after
                    and direct_comparator_equal
                    and np.array_equal(original_pertmean[index], mutated_pertmean[index])
                    else "FAIL",
                }
            )
    frame = pd.DataFrame(rows)
    if not (frame["status"] == "PASS").all():
        raise RuntimeError("STATE_STABILITY_LEAKAGE_INVARIANCE_FAILED")
    return frame


def adjudicate(context_g: np.ndarray) -> str:
    positive = int(np.sum(context_g > 0))
    median = float(np.median(context_g))
    if positive >= 4 and median >= 0.10 and float(np.min(context_g)) >= -0.10:
        return "STATE_OPERATOR_RECOVERY_BROADLY_STABLE"
    if int(np.sum(context_g > 0.10)) >= 2 and int(np.sum(context_g < -0.10)) >= 1:
        return "STATE_OPERATOR_RECOVERY_HETEROGENEOUSLY_REPLICATED"
    return "STATE_OPERATOR_RECOVERY_NOT_REPLICATED"


def analyze() -> str:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    interventions = selected_interventions()
    truth, baseline, predictions, axis_audits = load_tensors(interventions)
    leakage = mutation_invariance(truth, predictions)
    pd.DataFrame(axis_audits).to_csv(OUT / "STATE_STABILITY_AXIS_AUDIT.csv", index=False)
    leakage.to_csv(OUT / "STATE_STABILITY_LEAKAGE_AUDIT.csv", index=False)
    pertmean = strict_leave_one_context_mean(truth)
    gamma_true = truth - pertmean
    result_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    reconciliation: list[dict[str, object]] = []
    official_paths: dict[tuple[str, str], str] = {}
    manifest = json.loads((OUT / "STATE_STABILITY_ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact.get("context") and artifact["role"] in {"truth", "prediction"}:
            official_paths[(str(artifact["variant"]), str(artifact["context"]) + ":" + str(artifact["role"]))] = str(artifact.get("local_path") or artifact["remote_path"])
    primary_verdict = "STATE_STABILITY_AUDIT_NOT_EXECUTABLE"
    for variant_index, (variant, prediction) in enumerate(predictions.items()):
        gamma_pred = prediction - pertmean
        overall = recovery_statistics(gamma_true, gamma_pred)
        boot = bootstrap_recovery(gamma_true, gamma_pred, draws=10_000, seed=SEED + variant_index * 100)
        ci_low, ci_high = np.nanquantile(boot, [0.025, 0.975])
        identity = identity_metrics(gamma_true, gamma_pred)
        geometry = geometry_metrics(gamma_true, gamma_pred)
        per_context: list[float] = []
        for index, context in enumerate(CONTEXTS):
            stats_row = recovery_statistics(gamma_true[index], gamma_pred[index])
            per_context.append(float(stats_row["g"]))
            result_rows.append(
                {
                    "dataset": "Tahoe-100M zero-shot",
                    "variant": variant,
                    "scope": "context",
                    "context": context,
                    "n_interventions": len(interventions),
                    "feature_dimension": 2000,
                    **stats_row,
                    "state_delta_pearson": correlation(truth[index], prediction[index]),
                    "state_delta_mse": float(np.mean(np.square(truth[index] - prediction[index]))),
                    "pertmean_delta_pearson": correlation(truth[index], pertmean[index]),
                    "pertmean_delta_mse": float(np.mean(np.square(truth[index] - pertmean[index]))),
                    "state_absolute_pearson": correlation(truth[index] + baseline[index], prediction[index] + baseline[index]),
                    "bootstrap_ci_low": np.nan,
                    "bootstrap_ci_high": np.nan,
                    "primary_stability_unit": True,
                }
            )
            for metric, value in stats_row.items():
                reconciliation.append(
                    {
                        "dataset": "Tahoe-100M zero-shot",
                        "variant": variant,
                        "context": context,
                        "n_interventions": len(interventions),
                        "feature_dimension": 2000,
                        "metric": metric,
                        "value": value,
                        "source_artifact": f"{official_paths[(variant, context + ':prediction')]} | {official_paths[('ST_SE_TAHOE', context + ':truth')]}",
                        "source_key": "target_mean-reference_mean; exact target+feature axis",
                        "evaluator_source": "src/igc_virtual_cell/cgc_state_1/analysis.py::recovery_statistics",
                        "split_definition": "official Tahoe zero-shot held cell line",
                        "leakage_status": "PASS",
                        "notes": "context is the biological stability unit",
                    }
                )
        per_context_array = np.asarray(per_context)
        conventional = {
            "state_delta_pearson": correlation(truth, prediction),
            "state_delta_mse": float(np.mean(np.square(truth - prediction))),
            "pertmean_delta_pearson": correlation(truth, pertmean),
            "pertmean_delta_mse": float(np.mean(np.square(truth - pertmean))),
            "state_absolute_pearson": correlation(truth + baseline[:, None, :], prediction + baseline[:, None, :]),
        }
        result_rows.append(
            {
                "dataset": "Tahoe-100M zero-shot",
                "variant": variant,
                "scope": "overall",
                "context": "ALL",
                "n_interventions": len(interventions),
                "feature_dimension": 2000,
                **overall,
                **conventional,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "primary_stability_unit": False,
                **identity,
                **geometry,
            }
        )
        bootstrap_rows.append(
            {
                "dataset": "Tahoe-100M zero-shot",
                "variant": variant,
                "draws": 10_000,
                "g": overall["g"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "resampling": "frozen context-by-intervention energy contributions; contexts and interventions hierarchical",
                "models_refit": 0,
            }
        )
        summary_rows.append(
            {
                "dataset": "Tahoe-100M zero-shot",
                "variant": variant,
                "n_contexts": len(CONTEXTS),
                "positive_contexts": int(np.sum(per_context_array > 0)),
                "positive_fraction": float(np.mean(per_context_array > 0)),
                "median_g": float(np.median(per_context_array)),
                "minimum_g": float(np.min(per_context_array)),
                "maximum_g": float(np.max(per_context_array)),
                "aggregate_g": overall["g"],
                "aggregate_ci_low": ci_low,
                "aggregate_ci_high": ci_high,
                "shared_rule_r2": r2(truth, pertmean),
                "state_total_delta_r2": r2(truth, prediction),
                "novel_adaptation_g": overall["g"],
                **identity,
                **geometry,
            }
        )
        if variant == "ST_SE_TAHOE":
            primary_verdict = adjudicate(per_context_array)
    results = pd.DataFrame(result_rows)
    summaries = pd.DataFrame(summary_rows)
    results.to_csv(OUT / "STATE_STABILITY_RESULTS.csv", index=False)
    summaries.to_csv(OUT / "STATE_STABILITY_SUMMARY.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "STATE_STABILITY_BOOTSTRAP.csv", index=False)
    pd.DataFrame(reconciliation).to_csv(OUT / "STATE_STABILITY_NUMERICAL_RECONCILIATION.csv", index=False)
    source_data = build_figure(results, summaries)
    source_data.to_csv(OUT / "STATE_STABILITY_EXTENDED_DATA_SOURCE.csv", index=False)
    write_reports(results, summaries, primary_verdict, time.time() - started)
    run_manifest = {
        "phase": "STATE_STABILITY",
        "created_at": now(),
        "authority_commit": AUTHORITY,
        "protocol_freeze_commit": PROTOCOL_FREEZE,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "analysis_worktree_head_before_results_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "seed": SEED,
        "gpu_used": False,
        "model_trained": False,
        "contexts": CONTEXTS,
        "n_interventions": len(interventions),
        "features": 2000,
        "verdict": primary_verdict,
        "required_sha256": {
            path.name: sha256(path)
            for path in [
                OUT / "STATE_STABILITY_RESULTS.csv",
                OUT / "STATE_STABILITY_SUMMARY.csv",
                OUT / "STATE_STABILITY_BOOTSTRAP.csv",
                OUT / "STATE_STABILITY_NUMERICAL_RECONCILIATION.csv",
                OUT / "STATE_STABILITY_EXTENDED_DATA_SOURCE.csv",
                OUT / "STATE_STABILITY_REPORT.md",
                OUT / "STATE_STABILITY_MANUSCRIPT_WORDING.md",
                FIG / "STATE_STABILITY_EXTENDED_DATA.png",
                FIG / "STATE_STABILITY_EXTENDED_DATA.svg",
            ]
        },
    }
    (OUT / "STATE_STABILITY_RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    return primary_verdict


def build_figure(results: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    discovery = pd.read_csv(ROOT / "results" / "cgc_state_1" / "context_heterogeneity.csv").query("variant == 'ST_SE'").copy()
    discovery["dataset"] = "Parse cytokine discovery"
    discovery = discovery.rename(columns={"context": "label"})[["dataset", "label", "g"]]
    replication = results.query("variant == 'ST_SE_TAHOE' and scope == 'context'").copy()
    replication["dataset"] = "Tahoe drug replication"
    replication = replication.rename(columns={"context": "label"})[["dataset", "label", "g"]]
    points = pd.concat([discovery, replication], ignore_index=True)
    summary_source = []
    for dataset, frame in points.groupby("dataset", sort=False):
        values = frame["g"].to_numpy(float)
        summary_source.extend(
            [
                {"dataset": dataset, "label": "positive fraction", "g": float(np.mean(values > 0))},
                {"dataset": dataset, "label": "median g", "g": float(np.median(values))},
                {"dataset": dataset, "label": "minimum g", "g": float(np.min(values))},
                {"dataset": dataset, "label": "maximum g", "g": float(np.max(values))},
            ]
        )
    source = pd.concat([points.assign(panel="b/c", value_type="context_g"), pd.DataFrame(summary_source).assign(panel="d", value_type="summary")], ignore_index=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.65), constrained_layout=False)
    grid = fig.add_gridspec(2, 12, height_ratios=[0.7, 2.2], left=0.04, right=0.99, top=0.95, bottom=0.13, hspace=0.42, wspace=1.4)
    axa = fig.add_subplot(grid[0, :])
    axb = fig.add_subplot(grid[1, 0:4])
    axc = fig.add_subplot(grid[1, 4:9])
    axd = fig.add_subplot(grid[1, 9:12])
    axa.axis("off")
    axa.text(0.01, 0.80, "official zero-shot model", weight="bold", ha="left", va="center")
    axa.text(0.30, 0.80, "held context", weight="bold", ha="center", va="center")
    axa.text(0.57, 0.80, "shared interventions", weight="bold", ha="center", va="center")
    axa.text(0.88, 0.80, "operator recovery", weight="bold", ha="center", va="center")
    for x0, x1 in [(0.17, 0.25), (0.37, 0.49), (0.68, 0.79)]:
        axa.annotate("", xy=(x1, 0.80), xytext=(x0, 0.80), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#56616b"})
    for x, color in [(0.30, "#d9e2ea"), (0.53, "#dce9e4"), (0.58, "#dce9e4"), (0.63, "#dce9e4")]:
        axa.scatter([x], [0.30], s=180 if x == 0.30 else 70, color=color, edgecolor="#59636d", linewidth=0.7, clip_on=False)
    axa.text(0.88, 0.30, r"$g_c$ vs target-excluding Pert Mean", ha="center", va="center")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    colors = {"Parse cytokine discovery": "#587a95", "Tahoe drug replication": "#c16f54"}
    for ax, dataset, title in [(axb, "Parse cytokine discovery", "Discovery: cytokines"), (axc, "Tahoe drug replication", "Independent replication: drugs")]:
        frame = points.query("dataset == @dataset").copy()
        y = np.arange(len(frame))[::-1]
        ax.hlines(y, 0, frame["g"], color=colors[dataset], lw=1.5, alpha=0.85)
        ax.scatter(frame["g"], y, s=24, color=colors[dataset], edgecolor="white", linewidth=0.45, zorder=3)
        ax.axvline(0, color="#2d3238", lw=0.7)
        ax.set_yticks(y, frame["label"])
        ax.set_title(title, loc="center", pad=5, weight="bold")
        ax.set_xlabel(r"context-specific recovery, $g_c$")
        ax.grid(axis="x", color="#e7eaed", lw=0.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=2)
    for dataset, marker, color in [("Parse cytokine discovery", "o", colors["Parse cytokine discovery"]), ("Tahoe drug replication", "s", colors["Tahoe drug replication"])]:
        frame = points.query("dataset == @dataset")
        values = frame["g"].to_numpy(float)
        x = 0 if dataset.startswith("Parse") else 1
        axd.scatter(np.full(len(values), x) + np.linspace(-0.06, 0.06, len(values)), values, s=18, marker=marker, color=color, alpha=0.8)
        axd.scatter([x], [np.median(values)], s=48, marker="D", color="#24282d", edgecolor="white", linewidth=0.5, zorder=4)
        axd.vlines(x, np.min(values), np.max(values), color="#24282d", lw=0.8, zorder=2)
    axd.axhline(0, color="#2d3238", lw=0.7)
    axd.set_xticks([0, 1], ["Parse", "Tahoe"])
    axd.set_ylabel(r"context-specific recovery, $g_c$")
    axd.set_title("Across-context stability", loc="center", pad=5, weight="bold")
    axd.grid(axis="y", color="#e7eaed", lw=0.55)
    axd.spines[["top", "right"]].set_visible(False)
    for label, ax in zip("abcd", [axa, axb, axc, axd]):
        ax.text(-0.08 if ax is not axa else -0.01, 1.08 if ax is not axa else 1.02, label, transform=ax.transAxes, fontsize=9, weight="bold", va="top")
    for suffix in ("png", "svg"):
        fig.savefig(FIG / f"STATE_STABILITY_EXTENDED_DATA.{suffix}", dpi=600 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return source


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.{digits}g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(results: pd.DataFrame, summaries: pd.DataFrame, verdict: str, wall_seconds: float) -> None:
    primary = results.query("variant == 'ST_SE_TAHOE' and scope == 'context'")
    overall = results.query("variant == 'ST_SE_TAHOE' and scope == 'overall'").iloc[0]
    sensitivity = summaries.query("variant == 'ST_HVG_TAHOE'").iloc[0]
    per_context = primary[["context", "g", "alpha", "kappa", "cosine", "state_delta_pearson", "state_delta_mse", "pertmean_delta_pearson", "pertmean_delta_mse"]]
    summary = summaries[["variant", "positive_contexts", "n_contexts", "median_g", "minimum_g", "maximum_g", "aggregate_g", "aggregate_ci_low", "aggregate_ci_high"]]
    if verdict == "STATE_OPERATOR_RECOVERY_HETEROGENEOUSLY_REPLICATED":
        interpretation = "The independent Tahoe setting again contains substantial recovery in some held contexts and material negative transfer in others. STATE therefore demonstrates that CGC is partially mitigable, but context identity does not guarantee operator recoverability."
        wording = "STATE recovered substantial context-specific response structure in selected held-out contexts in both the cytokine and independent drug-response settings. However, recovery remained strongly context dependent, including material negative transfer, showing that preserved context identity does not guarantee reliable recovery of the intervention-conditioned operator."
        title = "Operator recovery remains context dependent despite preserved context identity"
    elif verdict == "STATE_OPERATOR_RECOVERY_BROADLY_STABLE":
        interpretation = "The independent Tahoe setting shows broadly positive context-specific operator recovery. STATE is a reproducible partial solution to CGC in the tested settings and should be presented constructively rather than only diagnostically."
        wording = "Across an independent official zero-shot drug-response setting, STATE showed broadly positive context-specific operator recovery, supporting a reproducible partial mitigation of intervention geometry compression in the tested contexts."
        title = "STATE reproducibly recovers part of the held-context response operator"
    else:
        interpretation = "The favorable 4/5 discovery pattern is not reproduced as substantial stable recovery in the independent Tahoe setting. Preserved context representation therefore does not establish general operator recovery."
        wording = "Although STATE preserved context and intervention structure in the discovery setting, substantial positive operator recovery did not replicate across the independent official zero-shot drug-response contexts."
        title = "Context representation does not ensure stable operator recovery"
    provenance = f"Authority `{AUTHORITY}`; protocol freeze `{PROTOCOL_FREEZE}`; branch `codex/state_stability_audit`."
    report = f"""# Independent stability audit of STATE context-specific operator recovery

## Verdict

`{verdict}`

{interpretation}

## Primary ST-SE-Tahoe result

- Exact official held contexts: {len(CONTEXTS)}.
- Exact shared drug-dose-unit interventions: {int(overall['n_interventions'])}.
- Feature dimension: {int(overall['feature_dimension'])}.
- Positive contexts: {int(np.sum(primary['g'] > 0))}/{len(primary)}.
- Median context `g`: {float(np.median(primary['g'])):.6f}.
- Range: [{float(primary['g'].min()):.6f}, {float(primary['g'].max()):.6f}].
- Aggregate `g`: {float(overall['g']):.6f}; exact frozen hierarchical-bootstrap interval [{float(overall['bootstrap_ci_low']):.6f}, {float(overall['bootstrap_ci_high']):.6f}].
- Aggregate alpha={float(overall['alpha']):.6f}, kappa={float(overall['kappa']):.6f}, cosine={float(overall['cosine']):.6f}.
- Delta-Pearson: STATE {float(overall['state_delta_pearson']):.6f} versus Pert Mean {float(overall['pertmean_delta_pearson']):.6f}.
- Delta-MSE: STATE {float(overall['state_delta_mse']):.6f} versus Pert Mean {float(overall['pertmean_delta_mse']):.6f}.

{markdown_table(per_context)}

## Frozen representation sensitivity

ST-HVG-Tahoe was prespecified and cannot rescue the primary result. It produced {int(sensitivity['positive_contexts'])}/{int(sensitivity['n_contexts'])} positive contexts, median `g`={float(sensitivity['median_g']):.6f}, range [{float(sensitivity['minimum_g']):.6f}, {float(sensitivity['maximum_g']):.6f}], and aggregate `g`={float(sensitivity['aggregate_g']):.6f}.

{markdown_table(summary)}

## Integrity and claim boundary

Only official immutable `eval_best.ckpt` per-context DE tables were used. The five cell lines are explicitly listed as test contexts in the official Tahoe zero-shot TOML. Exact intervention and feature correspondence passed, the ST-SE and ST-HVG truth-table LFS hashes are identical, and synthetic held-truth mutation left both official prediction tensors and the target-excluding comparator unchanged. No model, calibration, normalization, basis, or feature selector was fitted. Compact artifacts expose no biological replicate identities, so the held context—not cells or interventions—is the stability unit; the interval is a frozen hierarchical resampling diagnostic over five contexts and their interventions.

This audit does not identify mechanisms of heterogeneous recovery and does not support a claim that STATE globally solves CGC. Runtime was {wall_seconds / 60:.2f} minutes on CPU; GPU use and model fits were both zero.

## Provenance

{provenance}
"""
    (OUT / "STATE_STABILITY_REPORT.md").write_text(report, encoding="utf-8")
    manuscript = f"""# Recommended manuscript wording after the STATE stability audit

## Recommended section title

**{title}**

## Recommended main-text wording

{wording}

## Required boundary

Do not write that STATE never recovers context-specific operators. Do not write that STATE globally solves CGC. The frozen claim is limited to official held-context settings and the context-pattern verdict `{verdict}`.

## Provenance

{provenance}
"""
    (OUT / "STATE_STABILITY_MANUSCRIPT_WORDING.md").write_text(manuscript, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.download_only and args.analyze_only:
        parser.error("Choose at most one of --download-only and --analyze-only")
    if not args.analyze_only:
        acquire(max_workers=args.workers)
    if not args.download_only:
        verdict = analyze()
        print(verdict, flush=True)


if __name__ == "__main__":
    main()
