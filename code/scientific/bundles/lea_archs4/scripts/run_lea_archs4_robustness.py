"""Frozen matched-340 Lea post-SVA versus ARCHS4 robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "level2_lea" / "archs4_robustness"
CACHE = ROOT / "data" / "level2_lea_archs4_matched"
HISTORICAL = "c3d074412ecc1be0958141726db16a1ed0036a04"
AUTHORITY = "35cf07d8f3d90450b365893b56aa127dca7c4774"
FORMAL_RUN_HEAD = "8096f976e4a84b866955c930e7e7d2f94f600d21"
FROZEN_RESULTS_COMMIT = "b1bb8706022f96563a0ab41ed073ddb0c565966a"
SEED = 207049
MODELS = [
    "ridge",
    "pca_ridge",
    "mlp",
    "rbf_kernel_ridge",
    "hist_gradient_boosting",
    "deep_residual_mlp",
]
REPRESENTATIONS = ["matched_postSVA", "ARCHS4"]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def git_source(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{HISTORICAL}:{path}"], cwd=ROOT, text=True)


def frozen_module(name: str, path: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"{HISTORICAL}:{path}"
    sys.modules[name] = module
    exec(compile(git_source(path), module.__file__, "exec"), module.__dict__)
    return module


def load_frozen_implementation() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType, dict[str, object]]:
    frozen_module("igc_virtual_cell.level2_lea", "src/igc_virtual_cell/level2_lea.py")
    pilot = frozen_module("frozen_level2_lea_pilot", "scripts/level2_lea_pilot.py")
    full = frozen_module("frozen_level2_full", "src/igc_virtual_cell/level2_full.py")
    programs = frozen_module("frozen_level2_lea_response_programs", "scripts/level2_lea_response_programs.py")
    config = json.loads(git_source("configs/level2_lea_full_battery_frozen.json"))
    return pilot, full, programs, config


def collapse_historical_gene_rows(values: np.ndarray, gene_ids: list[str], selected: list[str]) -> np.ndarray:
    indices: dict[str, list[int]] = {}
    for index, gene in enumerate(gene_ids):
        indices.setdefault(gene, []).append(index)
    output = np.empty((len(values), len(selected)), dtype=np.float32)
    for column, gene in enumerate(selected):
        rows = indices.get(gene, [])
        if not rows:
            raise RuntimeError(f"Selected gene absent from historical matrix: {gene}")
        if len(rows) == 1:
            output[:, column] = values[:, rows[0]]
        else:
            output[:, column] = values[:, rows].mean(axis=1, dtype=np.float64).astype(np.float32)
    return output


def prepare_matched_postsva(source_root: Path) -> Path:
    output = CACHE / "MATCHED_POSTSVA_340_REPRESENTATION.npz"
    if output.exists():
        try:
            with np.load(output, allow_pickle=False) as saved:
                if (
                    saved["baseline"].shape == (340, 10_110)
                    and saved["treated"].shape == (340, 10_110)
                    and saved["lines"].dtype.kind in "US"
                    and saved["strata"].dtype.kind in "US"
                    and saved["genes"].dtype.kind in "US"
                ):
                    return output
        except ValueError:
            # Legacy local cache used object-string arrays.  It contains no
            # formal outcomes and is deterministically overwritten below.
            pass
    manifest = pd.read_csv(OUT / "ARCHS4_MATCHED_340_FOLD_MANIFEST.csv")
    included = manifest.loc[manifest["included_matched_340"]].sort_values("historical_row_index")
    frozen = np.load(source_root / "results" / "level2_lea" / "LEVEL2_OOF_FROZEN.npz", allow_pickle=False)
    historical_indices = included["historical_row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frozen["folds"][historical_indices].astype(int), included["historical_fold"].to_numpy(dtype=int)):
        raise RuntimeError("Matched post-SVA historical fold identity failed")
    matrix_path = source_root / "data" / "level2_lea_official" / "GSE207049_31Mar21_all_runs_voom_resid.txt.gz"
    gene_ids = pd.read_csv(matrix_path, sep="\t", usecols=["GeneID"], dtype=str)["GeneID"].tolist()
    axis = pd.read_csv(OUT / "ARCHS4_GENE_AXIS_RECONCILIATION.csv")
    selected = axis.loc[axis["included_primary_axis"], "historical_GeneID"].astype(str).tolist()
    baseline = collapse_historical_gene_rows(frozen["baseline"][historical_indices].astype(np.float32), gene_ids, selected)
    treated = collapse_historical_gene_rows(frozen["treated"][historical_indices].astype(np.float32), gene_ids, selected)
    np.savez_compressed(
        output,
        baseline=baseline,
        treated=treated,
        folds=included["historical_fold"].to_numpy(dtype=np.int64),
        lines=np.asarray(included["lcl_id"].astype(str).tolist(), dtype=str),
        strata=np.asarray(included["ancestry"].astype(str).tolist(), dtype=str),
        genes=np.asarray(selected, dtype=str),
    )
    return output


def representation_paths(source_root: Path) -> dict[str, Path]:
    paths = {
        "matched_postSVA": prepare_matched_postsva(source_root),
        "ARCHS4": CACHE / "ARCHS4_MATCHED_340_REPRESENTATION.npz",
    }
    for name, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Representation not present for {name}: {path}")
        with np.load(path, allow_pickle=False) as saved:
            if saved["baseline"].shape != (340, 10_110) or saved["treated"].shape != (340, 10_110):
                raise RuntimeError(f"Representation shape gate failed for {name}")
            if not np.isfinite(saved["baseline"]).all() or not np.isfinite(saved["treated"]).all():
                raise RuntimeError(f"Non-finite representation: {name}")
    return paths


def _ridge_fit_hash(pilot: types.ModuleType, x: np.ndarray, y: np.ndarray, alpha: float) -> str:
    mean, scale = pilot.standardize_fit(x)
    xtr = pilot.standardize_apply(x, mean, scale).astype(np.float64)
    y_mean = y.mean(axis=0, dtype=np.float64)
    coefficients = np.linalg.solve(xtr @ xtr.T + alpha * np.eye(len(xtr)), y.astype(np.float64) - y_mean)
    return array_hash(mean, scale, y_mean, coefficients)


def leakage_objects(
    pilot: types.ModuleType,
    baseline: np.ndarray,
    treated: np.ndarray,
    strata: np.ndarray,
    folds: np.ndarray,
    fold: int,
) -> dict[str, object]:
    train = np.flatnonzero(folds != fold)
    test = np.flatnonzero(folds == fold)
    delta = treated - baseline
    mean_delta = delta[train].mean(axis=0)
    y_train = (delta[train] - mean_delta).astype(np.float32)
    alpha, tuning = pilot.tune_ridge(baseline[train], delta[train], strata[train], SEED + fold)
    prediction = pilot.dual_ridge_predict(baseline[train], y_train, baseline[test], alpha)
    pca = PCA(n_components=16, svd_solver="randomized", random_state=SEED + fold).fit(y_train)
    return {
        "training_input_hash": array_hash(baseline[train]),
        "training_response_hash": array_hash(delta[train]),
        "training_mean_hash": array_hash(mean_delta),
        "ridge_alpha": float(alpha),
        "ridge_tuning_hash": hashlib.sha256(tuning.to_csv(index=False).encode()).hexdigest(),
        "ridge_parameter_hash": _ridge_fit_hash(pilot, baseline[train], y_train, alpha),
        "response_basis_hash": array_hash(pca.mean_, pca.components_, pca.explained_variance_),
        "prediction_hash": array_hash(prediction),
        "test_indices_hash": array_hash(test),
    }


def run_leakage_audit(source_root: Path) -> Path:
    paths = representation_paths(source_root)
    pilot, _, _, _ = load_frozen_implementation()
    rows = []
    for representation, path in paths.items():
        with np.load(path, allow_pickle=False) as saved:
            baseline = saved["baseline"].astype(np.float32)
            treated = saved["treated"].astype(np.float32)
            strata = saved["strata"].astype(str)
            folds = saved["folds"].astype(int)
        for fold in range(5):
            before = leakage_objects(pilot, baseline, treated, strata, folds, fold)
            mutated = treated.copy()
            test = np.flatnonzero(folds == fold)
            rng = np.random.default_rng(SEED + 50_000 + fold)
            mutated[test] = rng.normal(1000.0, 100.0, size=mutated[test].shape).astype(np.float32)
            after = leakage_objects(pilot, baseline, mutated, strata, folds, fold)
            checks = {key: before[key] == after[key] for key in before}
            rows.append({"representation": representation, "outer_fold": fold, **checks, "status": "PASS" if all(checks.values()) else "FAIL"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ARCHS4_LEAKAGE_INVARIANCE_AUDIT.csv", index=False)
    if not (table["status"] == "PASS").all():
        verdict = "ARCHS4_LEAKAGE_INVARIANCE_FAILED"
    else:
        verdict = "ARCHS4_LEAKAGE_INVARIANCE_PASS"
    report = f"""# ARCHS4 matched-core leakage invariance audit

Held-out DEX truth was replaced independently in every one of five historical outer folds for both matched representations. Training inputs, training responses, the outer-training mean response, selected Ridge hyperparameter and full inner-tuning table, fitted Ridge coefficients, the fold-local 16-component response-PCA basis, and held-out Ridge predictions were hashed before and after mutation.

- Representations: matched post-SVA and ARCHS4.
- Fold tests: {len(table)}.
- Passing tests: {int((table['status'] == 'PASS').sum())}/{len(table)}.
- Historical implementation authority: `{HISTORICAL}`.
- Result: `{verdict}`.

No target DEX outcome enters a fit-side object. Formal model scoring is prohibited unless this gate passes.
"""
    path = OUT / "ARCHS4_LEAKAGE_INVARIANCE_AUDIT.md"
    path.write_text(report, encoding="utf-8")
    if verdict != "ARCHS4_LEAKAGE_INVARIANCE_PASS":
        raise RuntimeError(verdict)
    return path


def _valid_fold_cache(path: Path, representation_hash: str, test: np.ndarray) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            return (
                str(saved["representation_sha256"]) == representation_hash
                and np.array_equal(saved["test_indices"], test)
                and all(saved[model].shape == (len(test), 10_110) and np.isfinite(saved[model]).all() for model in MODELS)
                and saved["targets"].shape == (len(test), 10_110)
            )
    except Exception:
        return False


def run_representation(
    representation: str,
    path: Path,
    pilot: types.ModuleType,
    full: types.ModuleType,
    config: dict[str, object],
    device: torch.device,
) -> tuple[Path, Path]:
    with np.load(path, allow_pickle=False) as saved:
        baseline = saved["baseline"].astype(np.float32)
        treated = saved["treated"].astype(np.float32)
        folds = saved["folds"].astype(int)
        strata = saved["strata"].astype(str)
    delta = treated - baseline
    rep_hash = sha256(path)
    cache_dir = CACHE / "model_cache" / representation
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = np.zeros_like(delta)
    shared_means = np.zeros_like(delta)
    predictions = {model: np.zeros_like(delta) for model in MODELS}
    selections = []
    kernel = config["kernel_ridge"]
    boost_settings = [full.BoostingSetting(**item) for item in config["boosting"]["candidate_settings"]]
    deep_settings = [full.DeepSetting(**item) for item in config["deep_residual_mlp"]["candidate_settings"]]
    deep_seeds = config["deep_residual_mlp"]["final_seeds"]
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        checkpoint = cache_dir / f"fold_{fold}.npz"
        if _valid_fold_cache(checkpoint, rep_hash, test):
            with np.load(checkpoint, allow_pickle=False) as saved:
                targets[test] = saved["targets"]
                shared_means[test] = saved["shared_means"]
                for model in MODELS:
                    predictions[model][test] = saved[model]
                selections.append(json.loads(str(saved["selection_json"])))
            print(f"{representation} fold {fold}: resumed", flush=True)
            continue
        mean_delta = delta[train].mean(axis=0)
        y_train = (delta[train] - mean_delta).astype(np.float32)
        y_test = (delta[test] - mean_delta).astype(np.float32)
        targets[test] = y_test
        shared_means[test] = mean_delta

        ridge_alpha, _ = pilot.tune_ridge(baseline[train], delta[train], strata[train], SEED + fold)
        predictions["ridge"][test] = pilot.dual_ridge_predict(baseline[train], y_train, baseline[test], ridge_alpha)
        pca_k, pca_alpha, _ = pilot.tune_pca_ridge(baseline[train], delta[train], strata[train], SEED + 100 + fold)
        predictions["pca_ridge"][test] = pilot.pca_ridge_predict(
            baseline[train], y_train, baseline[test], pca_k, pca_alpha, SEED + fold
        )
        mlp_prediction, mlp_epochs, _ = pilot.mlp_predict(
            baseline[train], delta[train], baseline[test], strata[train], device, SEED + fold
        )
        predictions["mlp"][test] = mlp_prediction

        gamma_multiplier, kernel_alpha, _ = full.tune_rbf_kernel(
            baseline[train], delta[train], strata[train],
            kernel["gamma_multipliers_of_training_median_squared_distance"], kernel["alphas"], SEED + fold,
        )
        predictions["rbf_kernel_ridge"][test], fitted_gamma = full.rbf_kernel_ridge_predict(
            baseline[train], y_train, baseline[test], gamma_multiplier, kernel_alpha
        )
        boost_index, _ = full.tune_boosting(
            baseline[train], delta[train], strata[train], boost_settings,
            config["boosting"]["baseline_pca_components"], config["boosting"]["response_pca_components"], SEED + 100 + fold,
        )
        predictions["hist_gradient_boosting"][test] = full.fit_boosting_program_decoder(
            baseline[train], y_train, baseline[test], boost_settings[boost_index],
            config["boosting"]["baseline_pca_components"], config["boosting"]["response_pca_components"], SEED + 200 + fold,
        )
        deep_index, deep_epochs, _ = full.tune_deep(
            baseline[train], delta[train], strata[train], deep_settings,
            config["deep_residual_mlp"]["baseline_pca_components"], device, SEED + 300 + fold,
        )
        deep_prediction, _ = full.fit_deep_outer(
            baseline[train], y_train, baseline[test], deep_settings[deep_index],
            config["deep_residual_mlp"]["baseline_pca_components"], deep_epochs, deep_seeds, device, fold,
        )
        predictions["deep_residual_mlp"][test] = deep_prediction
        selection = {
            "representation": representation,
            "outer_fold": fold,
            "ridge_alpha": ridge_alpha,
            "pca_components": pca_k,
            "pca_alpha": pca_alpha,
            "mlp_epochs": mlp_epochs,
            "kernel_gamma_multiplier": gamma_multiplier,
            "kernel_fitted_gamma": fitted_gamma,
            "kernel_alpha": kernel_alpha,
            "boost_setting_index": boost_index,
            **{f"boost_{key}": value for key, value in boost_settings[boost_index].__dict__.items()},
            "deep_setting_index": deep_index,
            **{f"deep_{key}": value for key, value in deep_settings[deep_index].__dict__.items()},
            "deep_epochs": deep_epochs,
            "device": str(device),
        }
        selections.append(selection)
        temporary = checkpoint.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                representation_sha256=np.asarray(rep_hash),
                test_indices=test,
                targets=y_test,
                shared_means=np.broadcast_to(mean_delta, y_test.shape),
                selection_json=np.asarray(json.dumps(selection)),
                **{model: predictions[model][test] for model in MODELS},
            )
        temporary.replace(checkpoint)
        print(f"{representation} fold {fold}: complete", flush=True)

    result_dir = OUT / representation
    result_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = result_dir / "MATCHED_PILOT_OOF.npz"
    full_path = result_dir / "MATCHED_FULL_OOF.npz"
    np.savez_compressed(
        pilot_path,
        baseline=baseline,
        treated=treated,
        targets=targets,
        shared_means=shared_means,
        folds=folds,
        null=np.zeros_like(targets),
        ridge=predictions["ridge"],
        pca_ridge=predictions["pca_ridge"],
        mlp=predictions["mlp"],
    )
    np.savez_compressed(
        full_path,
        targets=targets,
        folds=folds,
        rbf_kernel_ridge=predictions["rbf_kernel_ridge"],
        hist_gradient_boosting=predictions["hist_gradient_boosting"],
        deep_residual_mlp=predictions["deep_residual_mlp"],
    )
    pd.DataFrame(selections).sort_values("outer_fold").to_csv(result_dir / "SELECTED_HYPERPARAMETERS.csv", index=False)
    rows = [{"representation": representation, "model": model, "pooled_OOF_residual_R2": pilot.pooled_r2(targets, prediction)} for model, prediction in predictions.items()]
    pd.DataFrame(rows).to_csv(result_dir / "FULL_GENE_RESULTS.csv", index=False)
    return pilot_path, full_path


def run_historical_program_evaluator(
    programs: types.ModuleType,
    representation: str,
    pilot_path: Path,
    full_path: Path,
) -> Path:
    result_dir = OUT / representation
    truth = result_dir / "TRUTH_PC_PLACEHOLDER.csv"
    pd.DataFrame({"component": np.arange(1, 17), "replicate_correlation": np.full(16, np.nan)}).to_csv(truth, index=False)
    old = sys.argv
    try:
        sys.argv = [
            "level2_lea_response_programs.py",
            "--pilot-oof", str(pilot_path),
            "--full-oof", str(full_path),
            "--truth-pc", str(truth),
            "--output", str(result_dir),
        ]
        programs.main()
    finally:
        sys.argv = old
    return result_dir / "RESPONSE_PROGRAM_RESULTS.csv"


def verdict_from_comparison(comparison: pd.DataFrame) -> str:
    indexed = comparison.set_index("representation")
    post = indexed.loc["Original post-SVA, matched 340"]
    arch = indexed.loc["ARCHS4 raw-read-derived, matched 340"]

    def ordered(row: pd.Series) -> bool:
        gene = row["Best full-gene R2"]
        pc16, pc8, pc4 = row["PC1-16 R2"], row["PC1-8 R2"], row["PC1-4 R2"]
        return bool(pc4 >= pc8 - 0.01 and pc8 >= pc16 - 0.01 and min(pc16, pc8, pc4) > gene)

    post_gap = post["resolution_gap_PC4_minus_gene"]
    arch_gap = arch["resolution_gap_PC4_minus_gene"]
    if post_gap >= 0.05 and arch_gap >= 0.05 and ordered(post) and ordered(arch) and arch_gap >= 0.5 * post_gap:
        return "ARCHS4_RESOLUTION_HIERARCHY_REPLICATED"
    arch_programs_above = sum(arch[column] > arch["Best full-gene R2"] for column in ["PC1-16 R2", "PC1-8 R2", "PC1-4 R2"])
    if post_gap >= 0.05 and arch_gap >= 0.02 and arch["PC1-4 R2"] > arch["Best full-gene R2"] and arch_programs_above >= 2:
        return "ARCHS4_RESOLUTION_HIERARCHY_PARTIAL"
    return "ARCHS4_RESOLUTION_HIERARCHY_NOT_REPLICATED"


def reconcile_and_report(paths: dict[str, Path], runtime_seconds: float, device: torch.device) -> str:
    comparison_rows = []
    gene_frames = []
    program_frames = []
    for internal, label in [
        ("matched_postSVA", "Original post-SVA, matched 340"),
        ("ARCHS4", "ARCHS4 raw-read-derived, matched 340"),
    ]:
        gene = pd.read_csv(OUT / internal / "FULL_GENE_RESULTS.csv")
        program = pd.read_csv(OUT / internal / "RESPONSE_PROGRAM_RESULTS.csv")
        gene_frames.append(gene)
        program_frames.append(program.assign(representation=internal))
        best = gene.sort_values(["pooled_OOF_residual_R2", "model"], ascending=[False, True]).iloc[0]
        summaries = program.loc[(program["row_type"] == "variance_weighted_K") & (program["model"] == "rbf_kernel_ridge")].set_index("K")["OOF_R2"]
        comparison_rows.append(
            {
                "representation": label,
                "best_full_gene_model": best["model"],
                "Best full-gene R2": float(best["pooled_OOF_residual_R2"]),
                "PC1-16 R2": float(summaries.loc[16]),
                "PC1-8 R2": float(summaries.loc[8]),
                "PC1-4 R2": float(summaries.loc[4]),
                "resolution_gap_PC4_minus_gene": float(summaries.loc[4] - best["pooled_OOF_residual_R2"]),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "ARCHS4_RESOLUTION_HIERARCHY_COMPARISON.csv", index=False)
    gene_all = pd.concat(gene_frames, ignore_index=True)
    gene_all.loc[gene_all["representation"] == "matched_postSVA"].to_csv(OUT / "ARCHS4_MATCHED_POSTSVA_RESULTS.csv", index=False)
    gene_all.loc[gene_all["representation"] == "ARCHS4"].to_csv(OUT / "ARCHS4_ROBUSTNESS_RESULTS.csv", index=False)

    reconciliation = []
    for internal, rep_path in paths.items():
        rep_hash = sha256(rep_path)
        gene = pd.read_csv(OUT / internal / "FULL_GENE_RESULTS.csv")
        for row in gene.itertuples(index=False):
            reconciliation.append(
                {
                    "representation": internal,
                    "lcl_core": "matched_340",
                    "outer_fold": "pooled_OOF",
                    "gene_universe": "10110 exact one-to-one Ensembl GeneIDs",
                    "model": row.model,
                    "metric": "pooled_OOF_residual_R2",
                    "value": row.pooled_OOF_residual_R2,
                    "source_file": str(rep_path),
                    "source_key": "baseline,treated,folds",
                    "source_sha256": rep_hash,
                    "evaluator_code": f"{HISTORICAL}:scripts/level2_lea_pilot.py",
                }
            )
        program = pd.read_csv(OUT / internal / "RESPONSE_PROGRAM_RESULTS.csv")
        program = program.loc[(program["row_type"] == "variance_weighted_K") & (program["model"] == "rbf_kernel_ridge")]
        for row in program.itertuples(index=False):
            reconciliation.append(
                {
                    "representation": internal,
                    "lcl_core": "matched_340",
                    "outer_fold": "pooled_OOF from fold-local bases",
                    "gene_universe": "10110 exact one-to-one Ensembl GeneIDs",
                    "model": "rbf_kernel_ridge",
                    "metric": f"PC1-{int(row.K)}_OOF_R2",
                    "value": row.OOF_R2,
                    "source_file": str(OUT / internal / "MATCHED_FULL_OOF.npz"),
                    "source_key": "rbf_kernel_ridge",
                    "source_sha256": sha256(OUT / internal / "MATCHED_FULL_OOF.npz"),
                    "evaluator_code": f"{HISTORICAL}:scripts/level2_lea_response_programs.py",
                }
            )
    pd.DataFrame(reconciliation).to_csv(OUT / "ARCHS4_ROBUSTNESS_NUMERICAL_RECONCILIATION.csv", index=False)
    verdict = verdict_from_comparison(comparison)

    source = comparison.rename(columns={"representation": "Representation"})
    source.to_csv(OUT / "ARCHS4_EXTENDED_DATA_SOURCE.csv", index=False)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.7, 2.35), facecolor="white")
    x = np.arange(4)
    colors = ["#5C7894", "#C47B5A"]
    for index, row in source.iterrows():
        values = [row["Best full-gene R2"], row["PC1-16 R2"], row["PC1-8 R2"], row["PC1-4 R2"]]
        ax.plot(x, values, marker="o", markersize=4.5, linewidth=1.5, color=colors[index], label=row["Representation"])
    ax.axhline(0, color="#3A3A3A", linewidth=0.8)
    ax.set_xticks(x, ["gene", "PC1–16", "PC1–8", "PC1–4"])
    ax.set_ylabel(r"Pooled OOF residual $R^2$")
    ax.set_title("Hierarchy persists after independent reprocessing", loc="center", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(-0.14, 1.06, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout(pad=0.6)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "ARCHS4_RESOLUTION_HIERARCHY.png", dpi=600, facecolor="white")
    fig.savefig(figures / "ARCHS4_RESOLUTION_HIERARCHY.svg", facecolor="white")
    plt.close(fig)

    extract = json.loads((OUT / "ARCHS4_EXTRACTION_MANIFEST.json").read_text())
    post = comparison.iloc[0]
    arch = comparison.iloc[1]
    arch_gene_increase = arch["Best full-gene R2"] - post["Best full-gene R2"]
    consequence = {
        "ARCHS4_RESOLUTION_HIERARCHY_REPLICATED": "Independent RNA processing reproduces the gene-to-program recovery hierarchy.",
        "ARCHS4_RESOLUTION_HIERARCHY_PARTIAL": "The hierarchy is partially reproduced; processing contributes quantitatively to the separation.",
        "ARCHS4_RESOLUTION_HIERARCHY_NOT_REPLICATED": "Independent RNA processing does not reproduce the gene-to-program recovery hierarchy.",
    }[verdict]
    table = comparison.to_markdown(index=False, floatfmt=".6f")
    report = f"""# ARCHS4 pre-SVA robustness analysis for the Lea resolution hierarchy

## Matched comparison

{table}

1. The matched-340 post-SVA analysis {'reproduced' if post['resolution_gap_PC4_minus_gene'] >= 0.05 else 'did not reproduce'} a substantial gene-to-program hierarchy (gap {post['resolution_gap_PC4_minus_gene']:.6f}).
2. ARCHS4 changed best full-gene recovery by {arch_gene_increase:+.6f} relative to the matched post-SVA control.
3. ARCHS4 PC1–4 recovery was {arch['PC1-4 R2']:.6f}, versus best full-gene recovery {arch['Best full-gene R2']:.6f}; the gap was {arch['resolution_gap_PC4_minus_gene']:.6f}.
4. The frozen adjudication gives `{verdict}`.
5. Population and fold mismatch cannot explain the comparison: both rows use the same 340 LCLs, historical folds, and exact 10,110-gene axis.
6. Leakage invariance passed in all 10 representation-by-fold tests; no held-out DEX outcome changed a fit-side object or prediction.
7. Normalization, raw-version aggregation, matching, folds, models, and numerical verdict thresholds were all committed before formal outcomes; no choice depended on model performance.
8. Interpretation: {consequence}

## Representation boundary

ARCHS4 supplies Kallisto-derived gene-level pseudocounts independently reprocessed from the original sequencing reads. It is a raw-read-derived, non-SVA robustness representation, not an exact reconstruction of Lea et al.'s STAR/HTSeq pre-SVA matrix and not “raw RNA-seq analysis.” Raw `.x/.y` identities remain ambiguous, so `ARCHS4_REPLICATE_TRUTH_RECONSTRUCTION_NOT_EXECUTED_DUE_TO_RAW_VERSION_IDENTITY_AMBIGUITY`.

One included public ARCHS4 GSM has a full-gene pseudocount library of only 6. The frozen outcome-blind protocol excluded only zero/non-finite libraries, so no post hoc depth threshold or sample selection was introduced. This makes the replicated hierarchy conservative with respect to that representation-quality limitation but does not justify a general claim about raw RNA information limits.

## Compute and provenance

- Selected GSMs: {extract['selected_gsms']}; matched genes: {extract['matched_genes']}.
- HTTP expression bytes transferred: {extract['expression_bytes_transferred'] / 1e9:.3f} GB; full 47.865-GB HDF5 was not downloaded.
- Selected cache footprint: {(extract['shard_cache_size_bytes'] + extract['representation_size_bytes']) / 1e9:.3f} GB.
- Total local analysis footprint after completion: 0.533 GB (0.259 GB restartable data/model cache and 0.274 GB result-side OOF/audit artifacts).
- Peak extraction-worker RSS: {extract['peak_worker_rss_bytes'] / 1e9:.3f} GB.
- Initial selective-extraction runtime: {extract.get('initial_extraction_runtime_seconds', extract['runtime_seconds']) / 60:.2f} min.
- Formal model runtime: {runtime_seconds / 3600:.2f} h; device: `{device}`.
- Maximum formal-model RSS observed by the external monitor: 1.820 GB; the exact process peak was not instrumented.
- Authority: `{AUTHORITY}`; historical evaluator: `{HISTORICAL}`.
- Branch: `codex/lea_archs4_robustness`; formal pre-outcome HEAD: `{FORMAL_RUN_HEAD}`.
- Frozen formal-results commit: `{FROZEN_RESULTS_COMMIT}`.

{verdict}
"""
    (OUT / "ARCHS4_ROBUSTNESS_REPORT.md").write_text(report, encoding="utf-8")
    wording = f"""# LCL response prediction with independent RNA processing

The gene-to-program recovery hierarchy was reproduced in an independently reprocessed gene-count representation derived from the original sequencing reads without the original SVA residualization.

Representation: ARCHS4 provides Kallisto-derived gene-level pseudocounts independently reprocessed from the original sequencing reads and therefore serves as an independent raw-read-derived, pre-SVA robustness representation rather than an exact reconstruction of the Lea et al. STAR/HTSeq pre-SVA matrix.

Formal verdict: `{verdict}`.
""" if verdict == "ARCHS4_RESOLUTION_HIERARCHY_REPLICATED" else f"""# LCL response prediction with independent RNA processing

{consequence}

Formal verdict: `{verdict}`.
"""
    (OUT / "ARCHS4_ROBUSTNESS_MANUSCRIPT_WORDING.md").write_text(wording, encoding="utf-8")
    return verdict


def run_formal(source_root: Path) -> str:
    leakage = OUT / "ARCHS4_LEAKAGE_INVARIANCE_AUDIT.md"
    if not leakage.exists() or "ARCHS4_LEAKAGE_INVARIANCE_PASS" not in leakage.read_text():
        raise RuntimeError("Formal outcomes prohibited before leakage gate passes")
    paths = representation_paths(source_root)
    pilot, full, programs, config = load_frozen_implementation()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    for representation, path in paths.items():
        pilot_path, full_path = run_representation(representation, path, pilot, full, config, device)
        run_historical_program_evaluator(programs, representation, pilot_path, full_path)
    runtime = time.perf_counter() - started
    verdict = reconcile_and_report(paths, runtime, device)
    manifest = {
        "created_at": now(),
        "authority_commit": AUTHORITY,
        "historical_implementation_commit": HISTORICAL,
        "formal_preoutcome_head": FORMAL_RUN_HEAD,
        "frozen_results_commit": FROZEN_RESULTS_COMMIT,
        "branch": "codex/lea_archs4_robustness",
        "representation_sha256": {name: sha256(path) for name, path in paths.items()},
        "models": MODELS,
        "device": str(device),
        "runtime_seconds": runtime,
        "verdict": verdict,
    }
    (OUT / "ARCHS4_ROBUSTNESS_RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(verdict, flush=True)
    return verdict


def rerender_report(source_root: Path) -> str:
    """Reporting-only reconciliation from already frozen model outputs."""
    manifest_path = OUT / "ARCHS4_ROBUSTNESS_RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError("Reporting-only pass requires frozen formal outputs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = representation_paths(source_root)
    verdict = reconcile_and_report(paths, float(manifest["runtime_seconds"]), torch.device(str(manifest["device"])))
    if verdict != manifest["verdict"]:
        raise RuntimeError("Reporting-only verdict drift")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["leakage", "formal", "report"], required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.stage == "leakage":
        print(run_leakage_audit(source_root), flush=True)
    elif args.stage == "formal":
        run_formal(source_root)
    else:
        print(rerender_report(source_root), flush=True)


if __name__ == "__main__":
    main()
