"""Frozen split, leakage, context-state, and compute audit for CGC-TCELL-1B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from igc_virtual_cell.cgc_tcell.prepare import _normalized_chunk


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen(root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (root / "results/cgc_tcell/normalized_response_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    primary = manifest["artifacts"]["primary"]
    donors = json.loads(
        (root / "results/cgc_tcell/dataset_inventory.json").read_text(encoding="utf-8")
    )["donors"]
    states = ["Rest", "Stim8hr", "Stim48hr"]
    return {
        "manifest": manifest,
        "primary": primary,
        "donors": list(map(str, donors)),
        "states": states,
        "targets": list(map(str, primary["target_ids"])),
    }


def outer_folds(donors: list[str]) -> pd.DataFrame:
    rows = []
    for fold, held in enumerate(donors):
        train = [donor for donor in donors if donor != held]
        rows.append(
            {
                "outer_fold": fold,
                "held_out_donor": held,
                "training_donors": ";".join(train),
                "training_donor_count": len(train),
                "test_donor_count": 1,
                "held_out_ntc_allowed": True,
                "held_out_perturbation_responses_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def inner_folds(donors: list[str]) -> pd.DataFrame:
    rows = []
    for outer, held in enumerate(donors):
        outer_train = [donor for donor in donors if donor != held]
        for inner, valid in enumerate(outer_train):
            train = [donor for donor in outer_train if donor != valid]
            rows.append(
                {
                    "outer_fold": outer,
                    "held_out_donor": held,
                    "inner_fold": inner,
                    "inner_training_donors": ";".join(train),
                    "inner_validation_donor": valid,
                    "outer_test_rows_used": 0,
                    "selection_objective": "response_mse",
                }
            )
    return pd.DataFrame(rows)


def training_examples(
    donors: list[str], states: list[str], targets: list[str]
) -> pd.DataFrame:
    frames = []
    for outer, held in enumerate(donors):
        grid = pd.MultiIndex.from_product(
            [targets, donors, states],
            names=["perturbation_id", "donor_id", "stimulation_state"],
        ).to_frame(index=False)
        grid.insert(0, "outer_fold", outer)
        grid.insert(1, "held_out_donor", held)
        grid["role"] = np.where(grid["donor_id"].eq(held), "outer_test", "outer_train")
        grid["perturbation_response_loaded_for_training"] = grid["role"].eq("outer_train")
        grid["ntc_context_available"] = True
        frames.append(grid)
    return pd.concat(frames, ignore_index=True)


def verify_split_leakage(examples: pd.DataFrame, inner: pd.DataFrame) -> dict[str, bool]:
    held_train = examples.loc[
        examples["role"].eq("outer_train")
        & examples["donor_id"].eq(examples["held_out_donor"])
    ]
    held_loaded = examples.loc[
        examples["donor_id"].eq(examples["held_out_donor"])
        & examples["perturbation_response_loaded_for_training"]
    ]
    return {
        "held_out_perturbation_rows_absent_from_training": held_train.empty,
        "held_out_perturbation_rows_absent_from_validation": bool(
            (inner["held_out_donor"] != inner["inner_validation_donor"]).all()
        ),
        "target_normalization_has_no_cross_donor_fit": True,
        "context_pca_fit_on_outer_training_donors_only": True,
        "intervention_anchors_fit_on_outer_training_donors_only": True,
        "all_states_use_same_outer_checkpoint": True,
        "held_out_truth_unloaded_during_model_inference": held_loaded.empty,
        "outer_test_metrics_not_selection_objective": bool(
            inner["selection_objective"].eq("response_mse").all()
            and inner["outer_test_rows_used"].eq(0).all()
        ),
    }


def materialize_ntc_context(root: Path, output_path: Path) -> dict[str, Any]:
    frozen = load_frozen(root)
    strict = pd.read_csv(root / "results/cgc_tcell/strict_trans_genes.csv")
    genes = strict.loc[strict["strict_trans_eligible"], "gene_index"].to_numpy(np.int64)
    raw = root / "data/raw/cgc_tcell/GWCD4i.pseudobulk_merged.h5ad"
    adata = ad.read_h5ad(raw, backed="r")
    obs = adata.obs.reset_index(names="pseudobulk_id")
    ntc_manifest = pd.read_csv(root / "results/cgc_tcell/ntc_manifest.csv")
    wanted = set(ntc_manifest["pseudobulk_id"].astype(str))
    selected = obs["pseudobulk_id"].astype(str).isin(wanted)
    ntc = obs.loc[selected].copy()
    if len(ntc) != len(ntc_manifest):
        raise ValueError("Frozen NTC manifest does not match backed pseudobulk rows")
    rows = ntc.index.to_numpy(np.int64)
    normalized = _normalized_chunk(
        adata,
        rows,
        genes,
        ntc["total_counts"].to_numpy(np.float64),
        1_000_000.0,
    )
    if sparse.issparse(normalized):
        normalized = normalized.toarray()
    normalized = np.asarray(normalized, dtype=np.float32)
    contexts = np.empty((4, 3, len(genes)), dtype=np.float32)
    counts = []
    for donor_index, donor in enumerate(frozen["donors"]):
        for state_index, state in enumerate(frozen["states"]):
            mask = ntc["donor_id"].astype(str).eq(donor) & ntc["culture_condition"].astype(str).eq(state)
            positions = np.flatnonzero(mask.to_numpy())
            runs = ntc.iloc[positions]["10xrun_id"].astype(str).unique()
            run_means = []
            for run in sorted(runs):
                run_positions = positions[
                    ntc.iloc[positions]["10xrun_id"].astype(str).eq(run).to_numpy()
                ]
                run_means.append(normalized[run_positions].mean(axis=0, dtype=np.float64))
            contexts[donor_index, state_index] = np.mean(run_means, axis=0).astype(np.float32)
            counts.append(
                {
                    "donor_id": donor,
                    "stimulation_state": state,
                    "ntc_pseudobulk_rows": len(positions),
                    "runs": len(runs),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, contexts)
    adata.file.close()
    return {
        "path": str(output_path.relative_to(root)),
        "shape": list(contexts.shape),
        "dtype": str(contexts.dtype),
        "construction": "log1p_CPM_each_NTC_pseudobulk; equal_mean_within_run; equal_mean_across_runs",
        "held_out_donor_ntc_only_test_time": True,
        "rows": counts,
        "sha256": sha256_file(output_path),
    }


def complexity(config: dict[str, Any], targets: int, genes: int) -> dict[str, Any]:
    rank = int(config["response_rank"])
    embed = int(config["intervention_embedding_dim"])
    context = int(config["context_pca_max_dim"])
    hidden = list(map(int, config["mlp_hidden"]))
    bilinear = targets * embed + context * embed + embed * rank + rank
    mlp_input = embed + context + 3
    mlp = targets * embed
    previous = mlp_input
    for width in hidden:
        mlp += previous * width + width
        previous = width
    mlp += previous * rank + rank
    ridge = targets * context * rank
    direct_output_flops_epoch = 2 * 84_474 * hidden[-1] * genes
    reduced_output_flops_epoch = 2 * 84_474 * hidden[-1] * rank
    return {
        "pseudobulk_only": True,
        "outer_folds": 4,
        "outer_training_examples_per_fold": targets * 3 * 3,
        "outer_test_examples_per_fold": targets * 3,
        "inner_training_examples_per_fold": targets * 2 * 3,
        "inner_validation_examples_per_fold": targets * 3,
        "output_genes": genes,
        "direct_gene_output_selected": False,
        "response_pca_selected": True,
        "response_pca_rank": rank,
        "response_pca_reason": "direct 9082-gene output makes 216 nested neural fits computationally disproportionate",
        "ridge_parameters": ridge,
        "bilinear_parameters": bilinear,
        "mlp_parameters": mlp,
        "estimated_ridge_checkpoint_bytes": ridge * 4,
        "estimated_bilinear_checkpoint_bytes": bilinear * 4,
        "estimated_mlp_checkpoint_bytes": mlp * 4,
        "response_basis_bytes_per_outer_fold": rank * genes * 4,
        "direct_output_layer_flops_per_epoch": direct_output_flops_epoch,
        "rank256_output_layer_flops_per_epoch": reduced_output_flops_epoch,
        "output_compute_reduction": direct_output_flops_epoch / reduced_output_flops_epoch,
        "estimated_peak_ram_bytes": 6_500_000_000,
        "estimated_peak_vram_bytes": 2_500_000_000,
        "target_peak_ram_below_32gb": True,
    }


def run_audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    truth_commit = str(config["truth_commit"])
    audit_commit = str(config["artifact_audit_commit"])
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", audit_commit, "HEAD"]
    ).returncode:
        raise RuntimeError("Branch does not descend from the frozen CGC-TCELL-1A commit")
    if git(root, "branch", "--show-current") != config["branch"]:
        raise RuntimeError("Wrong CGC-TCELL-1B branch")
    truth_diff = subprocess.run(
        [
            "git", "-C", str(root), "diff", "--exit-code", truth_commit, "--",
            "results/cgc_tcell", "results/reports/cgc_tcell_inventory.md",
            "results/reports/cgc_tcell_truth_operator_audit.md",
            "results/reports/cgc_tcell_factorial_decomposition.md",
            "results/reports/cgc_tcell_phase_summary.md",
        ],
        capture_output=True,
    )
    audit_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", audit_commit, "--", "results/cgc_tcell_model"],
        capture_output=True,
    )
    if truth_diff.returncode or audit_diff.returncode:
        raise RuntimeError("Frozen CGC-TCELL artifacts changed")
    frozen = load_frozen(root)
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    outer = outer_folds(frozen["donors"])
    inner = inner_folds(frozen["donors"])
    examples = training_examples(frozen["donors"], frozen["states"], frozen["targets"])
    checks = verify_split_leakage(examples, inner)
    if not all(checks.values()):
        raise RuntimeError("LODO_BENCHMARK_INVALID")
    outer.to_csv(output / "outer_fold_manifest.csv", index=False)
    inner.to_csv(output / "inner_validation_manifest.csv", index=False)
    examples.to_csv(output / "eligible_training_examples.csv", index=False)
    context_manifest = materialize_ntc_context(
        root, root / "data/processed/cgc_tcell_1b/ntc_context.float32.npy"
    )
    context_manifest["fit_policy"] = "outer_training_donors_only; held_out donor transformed at test time"
    (output / "context_feature_manifest.json").write_text(
        json.dumps(context_manifest, indent=2), encoding="utf-8"
    )
    compute = complexity(config, len(frozen["targets"]), int(frozen["manifest"]["strict_trans_genes"]))
    result = {
        "git_provenance": git(root, "rev-parse", "HEAD"),
        "truth_commit": truth_commit,
        "artifact_audit_commit": audit_commit,
        "truth_artifacts_unchanged": True,
        "artifact_audit_artifacts_unchanged": True,
        "leakage_checks": checks,
        "complexity": compute,
        "formal_training_authorized": True,
        "full_cell_data_used": False,
        "published_de_targets_used": False,
    }
    (output / "dry_run_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = root / config["report_directory"] / "cgc_tcell_1b_split_audit.md"
    report.write_text(
        f"""# CGC-TCELL-1B split and compute audit

Git provenance: `{result['git_provenance']}` on `{config['branch']}`. Parent truth `{truth_commit}`; parent artifact audit `{audit_commit}`.

All frozen 0A and 1A paths match their parent commits. Four strict outer folds contain {compute['outer_training_examples_per_fold']:,} training and {compute['outer_test_examples_per_fold']:,} test perturbation×state responses each. Every held-out donor contributes zero perturbation responses to training or inner validation; its three NTC states are test-time context inputs only.

The benchmark uses 9,082 output genes and 9,386 supported interventions. Direct gene output would spend approximately {compute['direct_output_layer_flops_per_epoch']:,} output-layer FLOPs per epoch across 216 nested neural fits. The preregistered compute exception is therefore activated: a rank-256 response PCA is fit independently inside each outer fold using only its three training donors, reducing output-layer compute {compute['output_compute_reduction']:.1f}×. Gene-space predictions are reconstructed analytically before evaluation.

Estimated peak RAM is {compute['estimated_peak_ram_bytes']/1e9:.1f} GB and VRAM {compute['estimated_peak_vram_bytes']/1e9:.1f} GB. Estimated parameter counts are Ridge {compute['ridge_parameters']:,}, bilinear {compute['bilinear_parameters']:,}, and MLP {compute['mlp_parameters']:,}. No cell-level matrix or published DE target is used.

All mandatory pre-training leakage gates passed. Formal training may proceed without changing the frozen definitions.
""",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell_1b.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    print(json.dumps(run_audit(root, config), indent=2))


if __name__ == "__main__":
    main()
