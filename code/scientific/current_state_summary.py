"""Current official-artifact summary; no State fitting or sampled context null.

Uses the byte-exact historical loader/metrics, with the already-adjudicated
finite five-context derangement calculation (44 unique nonidentity mappings).
It intentionally does not reconstruct bootstrap intervals or manuscript layout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT_AUTHORITY = "21b08f02d3371129bb5cbf49704c65a271240c7c"


def verified_metadata_path(manifest: dict, metadata_dir: Path, name: str) -> Path:
    key = "results/cgc_state_1/" + name
    expected = next(row["sha256"] for row in manifest["files"] if row["path"] == key)
    target = metadata_dir / name
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Frozen State metadata hash mismatch: {name}")
    return target


def verify_official_files(data_dir: Path, inventory_path: Path, split_rows: list[dict]) -> int:
    with inventory_path.open(newline="", encoding="utf-8-sig") as handle:
        inventory = {row["relative_path"].replace("\\", "/"): row["sha256"] for row in csv.DictReader(handle)}
    checked = set()
    for row in split_rows:
        directory = data_dir / row["variant"] / f"split_{row['split']}"
        for pattern in ("*_pred_de.csv", "*_real_de.csv", "*_results.csv", "*_agg_results.csv"):
            files = list(directory.glob(pattern))
            if not files:
                raise RuntimeError(f"Missing frozen official artifact: {directory}/{pattern}")
            for path in files:
                key = "data/state_official_predictions/" + path.relative_to(data_dir).as_posix()
                expected = inventory.get(key)
                if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Frozen official artifact hash mismatch: {key}")
                checked.add(key)
    return len(checked)


def exact_context_null(truth_gamma: np.ndarray, predicted_gamma: np.ndarray) -> dict:
    truth = np.asarray(truth_gamma, dtype=np.float64)
    prediction = np.asarray(predicted_gamma, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[0] != 5:
        raise ValueError("Frozen State correction requires exactly five aligned contexts")
    te, pe = float(np.square(truth).sum()), float(np.square(prediction).sum())
    if te <= 0:
        raise ValueError("Nonpositive context-operator truth energy")
    observed = 1.0 - float(np.square(truth - prediction).sum()) / te
    permutations = [p for p in itertools.permutations(range(5)) if all(i != p[i] for i in range(5))]
    cross = np.einsum("cpg,dpg->cd", truth, prediction, optimize=True)
    null = np.array([1 - (te + pe - 2 * cross[np.arange(5), p].sum()) / te for p in permutations])
    exceedances = int(np.sum(null >= observed))
    return {"observed_g": observed, "unique_derangements": len(permutations), "exceedances": exceedances, "conservative_context_null_p": (1 + exceedances) / (len(permutations) + 1), "formula": "(1 + exact exceedances) / (44 + 1)", "audit_authority": AUDIT_AUTHORITY}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-data-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=HERE / "bundles/state_official_summary/results/cgc_state_1")
    parser.add_argument("--out", type=Path, required=True, help="new JSON output; existing output is never overwritten")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new path")
    bundle = HERE / "bundles/state_official_summary"
    manifest = json.loads((bundle / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    from launch import verify_import_sources
    verify_import_sources(bundle, manifest)
    split_path = verified_metadata_path(manifest, args.metadata_dir, "zero_shot_split_manifest.csv")
    inventory_path = verified_metadata_path(manifest, args.metadata_dir, "artifact_inventory.csv")
    with split_path.open(newline="", encoding="utf-8-sig") as handle:
        split_records = list(csv.DictReader(handle))
    official_hash_count = verify_official_files(args.official_data_dir.resolve(), inventory_path, split_records)
    source = "scripts/cgc_state_1_analysis.py"
    source_row = next(row for row in manifest["files"] if row["path"] == source)
    if hashlib.sha256((bundle / source).read_bytes()).hexdigest() != source_row["sha256"]:
        raise RuntimeError("Frozen State loader source changed")
    sys.path.insert(0, str(bundle / "src"))
    spec = importlib.util.spec_from_file_location("_frozen_state_summary_loader", bundle / source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA, module.OUT = args.official_data_dir.resolve(), args.metadata_dir.resolve()
    split = module.pd.read_csv(module.OUT / "zero_shot_split_manifest.csv")
    for row in split.itertuples(index=False):
        for suffix, expected in (("pred", row.pred_sha256), ("real", row.real_sha256)):
            files = list((module.DATA / row.variant / f"split_{row.split}").glob(f"*_{suffix}_de.csv"))
            if len(files) != 1 or module.sha256(files[0]) != expected:
                raise RuntimeError(f"Frozen official artifact mismatch: {row.variant}/{row.split}/{suffix}")
    contexts, interventions, truth, baseline, predictions, official = module.load_mosaic()
    if (len(contexts), len(interventions), truth.shape[-1]) != (5, 90, 2000):
        raise RuntimeError("Official State frozen axes changed")
    pertmean = module.leave_one_context_mean(truth)
    gt = truth - pertmean
    rows = []
    for variant in module.VARIANTS:
        pred = predictions[variant]
        gp = pred - pertmean
        geometry, _ = module.geometry_metrics(gt, gp)
        row = {"variant": variant, **module.recovery_statistics(gt, gp), **geometry, **module.identity_metrics(gt, gp),
               "state_delta_pearson": module.correlation(truth, pred), "pertmean_delta_pearson": module.correlation(truth, pertmean),
               "state_delta_mse": float(np.square(truth - pred).mean()), "pertmean_delta_mse": float(np.square(truth - pertmean).mean()),
               "exact_context_null": exact_context_null(gt, gp)}
        rows.append(row)
    result = {"classification": "OFFICIAL_ARTIFACT_SUMMARY", "source_commit": manifest["commit"], "correction_authority": AUDIT_AUTHORITY,
              "contexts": contexts, "interventions": interventions, "metrics": rows, "official_table_reproduction": official.to_dict("records"),
              "scope": "Metrics and corrected exact context correspondence only; no model fitting, sampled nulls, CI recomputation, rendering or manuscript update.",
              "replicate_calibrated": False, "uncertainty": "Bootstrap intervals remain separate frozen artifacts; not regenerated by this summary."}
    result["official_artifact_hash_checks"] = official_hash_count
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
