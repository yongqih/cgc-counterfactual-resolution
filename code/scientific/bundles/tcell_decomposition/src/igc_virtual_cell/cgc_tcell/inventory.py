"""Milestone-0 inventory for the CD4 T-cell pseudobulk release.

This module never reads expression values.  The H5AD is opened in backed mode
only to inventory observation/variable metadata and sparse storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from igc_virtual_cell.phase1.targeted_tensorized import _windows_memory_bytes


def _git(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_values(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.astype(str).value_counts(dropna=False).items()}


def _targeting_mask(obs: pd.DataFrame) -> pd.Series:
    return obs["guide_type"].astype(str).str.lower().eq("targeting")


def _ntc_mask(obs: pd.DataFrame) -> pd.Series:
    values = obs["guide_type"].astype(str).str.strip().str.lower()
    return (values.str.contains("non") & values.str.contains("target")) | values.isin(
        ["ntc", "control"]
    )


def run_inventory(root: Path, config_path: Path) -> dict[str, Any]:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = root / config["storage"]["raw_directory"]
    output = root / config["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    downloads = []
    for name, spec in config["downloads"].items():
        path = raw / name
        if not path.exists() or path.stat().st_size != int(spec["bytes"]):
            raise FileNotFoundError(f"Authorized download missing or wrong size: {path}")
        print(f"Hashing {name}", flush=True)
        downloads.append(
            {
                "file": name,
                "path": str(path.relative_to(root)),
                "url": spec["url"],
                "expected_bytes": int(spec["bytes"]),
                "actual_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    download_manifest = {
        "git_provenance": _git(root),
        "authorized_files_only": True,
        "cell_level_files_downloaded": False,
        "de_files_downloaded": False,
        "minimum_free_disk_gb": config["storage"]["minimum_free_disk_gb"],
        "free_disk_gb_after_download": round(shutil.disk_usage(root).free / 1e9, 3),
        "files": downloads,
    }
    (output / "download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2) + "\n", encoding="utf-8"
    )

    pseudobulk = raw / "GWCD4i.pseudobulk_merged.h5ad"
    adata = ad.read_h5ad(pseudobulk, backed="r")
    required_obs = {
        "10xrun_id",
        "donor_id",
        "culture_condition",
        "guide_id",
        "perturbed_gene_name",
        "perturbed_gene_id",
        "guide_type",
        "n_cells",
        "total_counts",
        "keep_min_cells",
        "keep_effective_guides",
        "keep_total_counts",
        "keep_for_DE",
        "keep_test_genes",
    }
    missing = required_obs - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Pseudobulk schema missing fields: {sorted(missing)}")
    obs = adata.obs.reset_index(names="pseudobulk_id").copy()
    var = adata.var.reset_index(names="variable_id").copy()
    states = list(map(str, config["frozen_states"]))
    donors = sorted(obs["donor_id"].astype(str).unique())
    observed_states = sorted(obs["culture_condition"].astype(str).unique())
    if len(donors) != 4 or set(observed_states) != set(states):
        raise ValueError(f"Unexpected donors/states: {donors}/{observed_states}")
    targeting, ntc = _targeting_mask(obs), _ntc_mask(obs)

    sample = pd.read_csv(raw / "sample_metadata.suppl_table.csv")
    lane_qc = pd.read_csv(raw / "QC_summaries_per_sample_lane.csv")
    library = pd.read_csv(raw / "sgrna_library_metadata.suppl_table.csv", low_memory=False)
    kd = pd.read_csv(raw / "guide_kd_efficiency.suppl_table.csv", low_memory=False)
    library_guide = "sgRNA" if "sgRNA" in library.columns else library.columns[0]
    kd_guide = "index" if "index" in kd.columns else kd.columns[0]
    library[library_guide] = library[library_guide].astype(str)
    kd[kd_guide] = kd[kd_guide].astype(str)

    grouping = ["perturbed_gene_id", "perturbed_gene_name", "donor_id", "culture_condition"]
    context = (
        obs.loc[targeting]
        .groupby(grouping, dropna=False, observed=True)
        .agg(
            pseudobulk_rows=("pseudobulk_id", "size"),
            guides=("guide_id", "nunique"),
            runs=("10xrun_id", "nunique"),
            n_cells_sum=("n_cells", "sum"),
            n_cells_min=("n_cells", "min"),
            n_cells_median=("n_cells", "median"),
            total_counts_sum=("total_counts", "sum"),
            keep_min_cells_fraction=("keep_min_cells", "mean"),
            keep_total_counts_fraction=("keep_total_counts", "mean"),
            keep_effective_guides_fraction=("keep_effective_guides", "mean"),
            keep_for_DE_fraction=("keep_for_DE", "mean"),
            keep_test_genes_fraction=("keep_test_genes", "mean"),
        )
        .reset_index()
    )
    context.to_csv(output / "context_coverage.csv", index=False)

    run_coverage = (
        obs.groupby(["donor_id", "culture_condition", "10xrun_id", "guide_type"], observed=True)
        .agg(
            pseudobulk_rows=("pseudobulk_id", "size"),
            guides=("guide_id", "nunique"),
            targets=("perturbed_gene_id", "nunique"),
            n_cells=("n_cells", "sum"),
            total_counts=("total_counts", "sum"),
        )
        .reset_index()
    )
    run_coverage.to_csv(output / "run_coverage.csv", index=False)

    guide = (
        obs.groupby(["guide_id", "guide_type", "perturbed_gene_id", "perturbed_gene_name"], dropna=False, observed=True)
        .agg(
            pseudobulk_rows=("pseudobulk_id", "size"),
            donors=("donor_id", "nunique"),
            states=("culture_condition", "nunique"),
            runs=("10xrun_id", "nunique"),
            donor_state_contexts=("pseudobulk_id", "size"),
            n_cells_min=("n_cells", "min"),
            n_cells_median=("n_cells", "median"),
            n_cells_max=("n_cells", "max"),
            total_counts_min=("total_counts", "min"),
            keep_min_cells_all=("keep_min_cells", "all"),
            keep_total_counts_all=("keep_total_counts", "all"),
            keep_effective_guides_all=("keep_effective_guides", "all"),
            keep_for_DE_all=("keep_for_DE", "all"),
            keep_test_genes_all=("keep_test_genes", "all"),
        )
        .reset_index()
    )
    guide = guide.merge(library, left_on="guide_id", right_on=library_guide, how="left", suffixes=("", "_library"))
    kd_summary = (
        kd.groupby(kd_guide, as_index=False)
        .agg(
            kd_conditions=("culture_condition", "nunique"),
            significant_knockdown_conditions=("signif_knockdown", "sum"),
            high_confidence_no_effect_any=("high_confidence_no_effect_guides", "any"),
            kd_t_statistic_median=("t_statistic", "median"),
            kd_adjusted_p_max=("adj_p_value", "max"),
        )
    )
    guide = guide.merge(kd_summary, left_on="guide_id", right_on=kd_guide, how="left")
    guide.to_csv(output / "guide_coverage.csv", index=False)

    qc_flags = [column for column in required_obs if column.startswith("keep_")]
    off_target_columns = [
        column
        for column in (
            "flag",
            "putative_bidirectional_promoter",
            "other_alignment_chromosome",
            "nearest_within2kb_nontarget_gene_id",
            "nearest_within2kb_nontarget_gene_name",
        )
        if column in library.columns
    ]
    target_context_counts = context.groupby(["perturbed_gene_id", "perturbed_gene_name"], dropna=False).size()
    guide_context_counts = (
        obs.loc[targeting]
        .drop_duplicates(["guide_id", "donor_id", "culture_condition"])
        .groupby(["perturbed_gene_id", "guide_id"], dropna=False)
        .size()
    )
    complete_guides = guide_context_counts.loc[guide_context_counts.eq(12)].reset_index()
    complete_guides_per_target = complete_guides.groupby("perturbed_gene_id")["guide_id"].nunique()
    guide_disjoint_pre_qc = int((complete_guides_per_target >= 2).sum())
    run_target_context = (
        obs.loc[targeting]
        .drop_duplicates(["perturbed_gene_id", "donor_id", "culture_condition", "10xrun_id"])
        .groupby(["perturbed_gene_id", "10xrun_id"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    run_disjoint_pre_qc = int((run_target_context.ge(12).sum(axis=1) >= 2).sum())

    inventory = {
        "git_provenance": _git(root),
        "outcome_signal_calculated": False,
        "h5ad_access_mode": "backed_read_only",
        "complete_matrix_toarray_called": False,
        "pseudobulk_rows": int(adata.n_obs),
        "measured_genes": int(adata.n_vars),
        "x_backed_type": type(adata.X).__name__,
        "obs_columns": list(map(str, adata.obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "donors": donors,
        "culture_conditions": observed_states,
        "runs": sorted(obs["10xrun_id"].astype(str).unique()),
        "guide_types": _json_values(obs["guide_type"]),
        "targeting_guides": int(obs.loc[targeting, "guide_id"].nunique()),
        "ntc_guides": int(obs.loc[ntc, "guide_id"].nunique()),
        "unique_perturbed_gene_ids": int(obs.loc[targeting, "perturbed_gene_id"].nunique()),
        "unique_perturbed_gene_names": int(obs.loc[targeting, "perturbed_gene_name"].nunique()),
        "targets_complete_all_12_pre_qc": int(target_context_counts.eq(12).sum()),
        "n_cells_summary": obs["n_cells"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict(),
        "total_counts_summary": obs["total_counts"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict(),
        "qc_flag_counts": {flag: _json_values(obs[flag]) for flag in sorted(qc_flags)},
        "library_flag_counts": _json_values(library["flag"]) if "flag" in library else {},
        "off_target_annotation_nonnull_counts": {
            column: int(library[column].notna().sum()) for column in off_target_columns
        },
        "guide_disjoint_complete_targets_pre_qc": guide_disjoint_pre_qc,
        "run_disjoint_complete_targets_pre_qc": run_disjoint_pre_qc,
        "sample_metadata_rows": len(sample),
        "lane_qc_rows": len(lane_qc),
        "guide_library_rows": len(library),
        "knockdown_rows": len(kd),
        "obs_memory_bytes": int(obs.memory_usage(deep=True).sum()),
        "peak_rss_bytes": int(_windows_memory_bytes()[1]),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, default=float) + "\n", encoding="utf-8"
    )
    report_path = root / config["outputs"]["inventory_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# CGC-TCELL-0A dataset inventory

- Git provenance: `{inventory['git_provenance']}`
- Pseudobulk rows: **{inventory['pseudobulk_rows']:,}**
- Measured genes: **{inventory['measured_genes']:,}**
- Donors: `{', '.join(donors)}`
- States: `{', '.join(observed_states)}`
- Runs: `{', '.join(inventory['runs'])}`
- Targeting guides: **{inventory['targeting_guides']:,}**
- NTC guides: **{inventory['ntc_guides']:,}**
- Unique curated target IDs: **{inventory['unique_perturbed_gene_ids']:,}**
- Complete 12-context targets before QC: **{inventory['targets_complete_all_12_pre_qc']:,}**
- Two complete guides before QC: **{guide_disjoint_pre_qc:,} targets**
- Two complete runs before QC: **{run_disjoint_pre_qc:,} targets**
- H5AD access: backed read-only; complete `.X` densification: **never**
- Peak inventory RSS: **{inventory['peak_rss_bytes'] / 1e9:.2f} GB**

No context-operator outcome was calculated during this inventory.
""",
        encoding="utf-8",
    )
    adata.file.close()
    print(json.dumps({key: inventory[key] for key in ("pseudobulk_rows", "measured_genes", "donors", "culture_conditions", "runs", "targeting_guides", "ntc_guides", "unique_perturbed_gene_ids", "guide_disjoint_complete_targets_pre_qc", "run_disjoint_complete_targets_pre_qc")}, indent=2))
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    run_inventory(root, config)


if __name__ == "__main__":
    main()
