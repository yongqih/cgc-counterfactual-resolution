"""Integrity gates for the frozen Tahoe-100M feasibility audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_OUTCOME_COLUMNS = {
    "gene_name",
    "basemean",
    "log2foldchange",
    "lfcse",
    "stat",
    "pvalue",
    "padj",
    "response",
    "delta",
}
ALLOWED_METADATA_FILES = {
    "sample_metadata.parquet",
    "cell_line_metadata.parquet",
    "drug_metadata.parquet",
    "gene_metadata.parquet",
}
FROZEN_GSE306429_FILES = (
    "results/cgc_support_0a/dataset_inventory.json",
    "results/cgc_support_0a/coverage_matrix.csv",
    "results/cgc_support_0a/leakage_audit.json",
    "results/cgc_support_0a/run_manifest.json",
    "results/cgc_support_0a/verdict.json",
    "results/reports/cgc_support_0a_phase_summary.md",
    "data/gse306429/filelist.txt",
)
FROZEN_TCELL_DIRS = (
    "results/cgc_tcell",
    "results/cgc_tcell_1b",
    "results/cgc_tcell_1c",
    "results/cgc_tcell_1d",
    "results/cgc_tcell_2a",
    "results/cgc_tcell_2b",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_changes(root: Path, pathspecs: tuple[str, ...]) -> list[str]:
    command = ["git", "status", "--short", "--", *pathspecs]
    result = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def run_integrity_audit(root: Path) -> dict[str, Any]:
    """Run all protocol gates and write the machine-readable audit."""

    root = root.resolve()
    output = root / "results" / "cgc_tahoe_0b"
    metadata = root / "data" / "tahoe100m_metadata"

    condition = pd.read_parquet(output / "plate6_14_condition_index.parquet")
    counts = pd.read_parquet(output / "plate6_14_cell_counts.parquet")
    core = pd.read_csv(output / "selected_replicate_core.csv")
    controls = pd.read_csv(output / "dmso_manifest.csv")
    sample_design = pd.read_csv(output / "sample_design.csv")
    cells = pd.read_csv(output / "cell_line_manifest.csv")
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    download_manifest = json.loads(
        (output / "download_manifest.json").read_text(encoding="utf-8")
    )
    verdict = json.loads((output / "verdict.json").read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    pair_plate_counts = condition.groupby(
        ["cell_line_id", "intervention_id"], observed=True
    )["plate"].nunique()
    separate_plate_columns = {
        "plate6_n_cells_trt",
        "plate14_n_cells_trt",
        "plate6_n_cells_ctrl",
        "plate14_n_cells_ctrl",
    }.issubset(core.columns)
    _record(
        checks,
        "plate6_and_plate14_not_pooled",
        set(condition["plate"].unique()) == {"plate6", "plate14"}
        and pair_plate_counts.max() == 2
        and separate_plate_columns,
        {
            "condition_plate_values": sorted(condition["plate"].unique().tolist()),
            "replicate_specific_core_columns_present": separate_plate_columns,
        },
    )

    noncontrol_design = sample_design.loc[~sample_design["is_control"]]
    reconstructed_ids = noncontrol_design.apply(
        lambda row: f"{row['parsed_drug']}__{float(row['concentration']):g}__{row['concentration_unit']}",
        axis=1,
    )
    _record(
        checks,
        "dose_is_part_of_intervention_identity",
        reconstructed_ids.eq(noncontrol_design["intervention_id"]).all()
        and noncontrol_design["concentration"].notna().all()
        and noncontrol_design["concentration_unit"].notna().all(),
        "identity reconstructed exactly as drug__concentration__unit",
    )

    control_markers = core["intervention_id"].astype(str).str.contains(
        "DMSO", case=False, regex=False
    ) | core["parsed_drug"].astype(str).str.contains("DMSO", case=False, regex=False)
    _record(
        checks,
        "dmso_excluded_from_interventions",
        not control_markers.any() and core["intervention_id"].nunique() == 93,
        {"selected_noncontrol_interventions": int(core["intervention_id"].nunique())},
    )

    coverage_flags = cells["selection_uses_expression_outcome"].eq(False).all() and core[
        "core_selected_by_coverage_only"
    ].eq(True).all()
    _record(
        checks,
        "core_selection_uses_coverage_only",
        coverage_flags,
        {"selection_inputs": run_manifest.get("selection_inputs", [])},
    )

    audited_tables = (condition, counts, core, cells)
    outcome_columns = sorted(
        {
            str(column)
            for table in audited_tables
            for column in table.columns
            if str(column).lower() in FORBIDDEN_OUTCOME_COLUMNS
        }
    )
    exclusions = {str(value).lower() for value in run_manifest["selection_excluded_inputs"]}
    _record(
        checks,
        "no_response_or_de_magnitude_in_selection",
        not outcome_columns
        and {"gene expression", "log2foldchange", "de magnitude"}.issubset(exclusions),
        {"forbidden_columns_found": outcome_columns},
    )

    _record(
        checks,
        "only_plate6_and_plate14_in_core_inputs",
        set(condition["plate"].unique()) == {"plate6", "plate14"}
        and set(controls["plate"].unique()) == {"plate6", "plate14"},
        sorted(condition["plate"].unique().tolist()),
    )

    core_complete = (
        len(core) == 50 * 93
        and core["cell_line_id"].nunique() == 50
        and core["intervention_id"].nunique() == 93
        and core["present_both_plates"].eq(True).all()
        and core[["plate6_n_cells_trt", "plate14_n_cells_trt"]].gt(0).all().all()
    )
    _record(
        checks,
        "every_core_pair_present_on_both_plates",
        core_complete,
        {
            "rows": int(len(core)),
            "contexts": int(core["cell_line_id"].nunique()),
            "interventions": int(core["intervention_id"].nunique()),
            "minimum_treatment_cells": int(
                core[["plate6_n_cells_trt", "plate14_n_cells_trt"]].min().min()
            ),
        },
    )

    selected_contexts = set(core["cell_line_id"])
    control_support = (
        core[["plate6_n_cells_ctrl", "plate14_n_cells_ctrl"]].gt(0).all().all()
        and set(controls["cell_line_id"]) == selected_contexts
        and controls.groupby(["cell_line_id", "plate"], observed=True)["n_control_cells"]
        .sum()
        .gt(0)
        .all()
    )
    _record(
        checks,
        "plate_matched_controls_for_every_context",
        control_support,
        {
            "control_manifest_rows": int(len(controls)),
            "minimum_individual_dmso_well_cells": int(controls["n_control_cells"].min()),
        },
    )

    local_metadata_files = {
        path.name for path in metadata.rglob("*") if path.is_file()
    }
    forbidden_expression_paths = [
        str(path.relative_to(root))
        for path in (root / "data").rglob("*")
        if path.is_file()
        and "tahoe" in str(path).lower()
        and any(token in str(path).lower() for token in ("expression_data", ".h5ad", ".zarr"))
    ]
    _record(
        checks,
        "full_expression_atlas_not_downloaded",
        not forbidden_expression_paths
        and run_manifest["full_expression_atlas_downloaded"] is False,
        forbidden_expression_paths,
    )

    forbidden_de_paths = [
        str(path.relative_to(root))
        for path in (root / "data").rglob("*")
        if path.is_file()
        and "tahoe" in str(path).lower()
        and "pseudobulk_differential_expression" in str(path).lower()
    ]
    _record(
        checks,
        "full_de_release_not_downloaded",
        not forbidden_de_paths
        and local_metadata_files == ALLOWED_METADATA_FILES
        and run_manifest["large_de_downloaded"] is False,
        {
            "local_metadata_files": sorted(local_metadata_files),
            "local_de_files": forbidden_de_paths,
        },
    )

    missing_gse = [path for path in FROZEN_GSE306429_FILES if not (root / path).exists()]
    frozen_verdict = json.loads(
        (root / "results/cgc_support_0a/verdict.json").read_text(encoding="utf-8")
    ).get("verdict")
    _record(
        checks,
        "gse306429_frozen_provenance_intact",
        not missing_gse and frozen_verdict == "SHARED_INTERVENTION_COVERAGE_INSUFFICIENT",
        {"missing": missing_gse, "frozen_verdict": frozen_verdict},
    )

    missing_tcell = [path for path in FROZEN_TCELL_DIRS if not (root / path).is_dir()]
    tcell_changes = _git_changes(root, FROZEN_TCELL_DIRS)
    _record(
        checks,
        "tcell_cgc_data_untouched",
        not missing_tcell and not tcell_changes,
        {"missing_directories": missing_tcell, "git_changes": tcell_changes},
    )

    manifest_hashes_ok = True
    hash_details: list[dict[str, Any]] = []
    for item in download_manifest["files"]:
        local_path = root / item["local_repo_path"]
        observed = _sha256(local_path)
        ok = observed == item["expected_sha256"] == item["observed_sha256"]
        manifest_hashes_ok &= ok
        hash_details.append({"path": item["local_repo_path"], "sha256_ok": ok})
    _record(checks, "small_metadata_hashes_match", manifest_hashes_ok, hash_details)

    sanity = run_manifest["sanity_check"]
    _record(
        checks,
        "publication_pair_sanity_check_matches",
        sanity["derived_plate6_pairs"] == 4800
        and sanity["derived_plate14_pairs"] == 4796
        and sanity["derived_common_pairs"] == 4796,
        sanity,
    )

    all_passed = all(check["passed"] for check in checks)
    result = {
        "phase": "CGC-SUPPORT-0B",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_passed else "FAIL",
        "invalid_verdict_if_failed": "CGC_TAHOE_0B_INVALID",
        "protocol_checks_passed": int(sum(check["passed"] for check in checks)),
        "protocol_checks_total": len(checks),
        "checks": checks,
        "frozen_feasibility_verdict": verdict["verdict"] if all_passed else None,
    }
    output_path = output / "integrity_audit.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    run_manifest["integrity_audit"] = {
        "status": result["status"],
        "checks_passed": result["protocol_checks_passed"],
        "checks_total": result["protocol_checks_total"],
    }
    run_manifest["status"] = "complete" if all_passed else "invalid"
    run_manifest["verdict"] = verdict["verdict"] if all_passed else "CGC_TAHOE_0B_INVALID"
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not all_passed:
        failed = [check["check"] for check in checks if not check["passed"]]
        raise RuntimeError(f"CGC_TAHOE_0B_INVALID: {failed}")
    return result
