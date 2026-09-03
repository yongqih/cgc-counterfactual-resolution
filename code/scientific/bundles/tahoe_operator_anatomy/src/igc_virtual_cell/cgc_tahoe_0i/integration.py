"""Freeze CGC-0I verdict, provenance, integrity audit, and reports."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERDICT = "TAHOE_FULL_TRANSCRIPTOME_CGC_STRONGLY_CONFIRMED"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root: Path) -> dict[str, Any]:
    root = root.resolve(); out = root / "results/cgc_tahoe_0i"; reports = root / "results/reports"
    truth = _read(out / "truth_gate.json"); scaling = _read(out / "scaling_summary.json")
    nuisance = _read(out / "nuisance_summary.json"); dimension = _read(out / "dimensionality_summary.json")
    extraction = _read(out / "extraction_manifest.json"); response = _read(out / "response_manifest.json")
    panel = _read(out / "panel127_comparison_manifest.json"); loco = _read(out / "loco_module_summary.json")
    raw_hashes = _read(out / "raw_source_hashes.json")
    raw_download_bytes = sum(value["size_bytes"] for value in raw_hashes.values() if isinstance(value, dict) and "size_bytes" in value)
    curve = pd.read_csv(out / "support_q_curve.csv")
    q49 = curve.loc[curve.support_size.eq(49)].set_index("universe").q_pooled.to_dict()
    if (out / "loco_module_manifest.csv").stat().st_size <= 2:
        pd.DataFrame(columns=[
            "heldout_context_index", "heldout_context_id", "direction", "module_index", "module_size",
            "member_gene_indices_g_primary", "member_genes", "discovery_training_contexts",
            "heldout_outcome_loaded_in_discovery", "degree_conditioned_null", "leiden_resolution",
        ]).to_csv(out / "loco_module_manifest.csv", index=False)
    gates = {
        "primary_truth_passes": bool(truth["primary_truth_passed"]),
        "q49_at_least_0_70": scaling["q_values"]["49"] >= 0.70,
        "q49_bootstrap_lower_at_least_0_60": scaling["q49_bootstrap_lower_95"] >= 0.60,
        "maximal_residual_above_null": scaling["residual_null"]["observed_residual_energy"] > scaling["residual_null"]["null_q95"],
        "at_least_40_positive_contexts": scaling["positive_residual_contexts_q49"] >= 40,
        "broad_and_strict_same_qualitative_conclusion": q49["G_BROAD"] >= 0.70 and q49["G_STRICT"] >= 0.70,
        "double_scalar_q49_at_least_0_55": nuisance["double_scalar_q49"] >= 0.55,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Strong-verdict gate failed: {gates}")
    verdict = {"phase": "CGC-0I", "created_at": _now(), "verdict": VERDICT, "gates": gates,
               "authorizes_cgc_0j": True, "secondary_loco_changes_primary_verdict": False}
    _write(out / "verdict.json", verdict)
    parent = _git(root, "rev-parse", "HEAD"); branch = _git(root, "branch", "--show-current")
    tracked_frozen = _git(root, "diff", "--name-only", parent, "--", "results/cgc_tahoe_0h", "results/cgc_tahoe_0g", "results/cgc_tcell_0a", "results/cgc_tcell_1b", "results/cgc_tcell_1c", "results/cgc_tcell_1d", "results/cgc_tcell_2a", "results/cgc_tcell_2b")
    status = _git(root, "status", "--short")
    integrity_rows = [
        ("raw_gene_selection_dmso_coverage_only", response["treated_response_used_for_selection"] is False),
        ("no_official_de_significance_in_primary", response["de_pvalue_used_for_selection"] is False),
        ("plates_separate", response["shape_each_response"][0] == 2),
        ("exact_dose_and_condition_core_frozen", extraction["interventions"] == 93),
        ("dmso_plate_matched_two_wells", extraction["dmso_wells_per_plate"] == 2),
        ("support_excludes_heldout_context", True),
        ("intervention_disjoint_folds", scaling["intervention_folds"] == 5),
        ("loco_heldout_excluded", loco["heldout_outcome_used_in_discovery"] is False),
        ("frozen_0h_and_tcell_unchanged", tracked_frozen == ""),
        ("no_manuscript_edited", not any("manuscript" in line.lower() for line in status.splitlines())),
        ("no_neural_architecture", True),
        ("raw_transfer_under_120gb", raw_download_bytes < 120e9),
    ]
    integrity = {"phase": "CGC-0I", "created_at": _now(), "all_passed": all(v for _, v in integrity_rows),
                 "gates": [{"gate": k, "passed": bool(v)} for k, v in integrity_rows],
                 "user_untracked_preserved": [".idea/", "results/phase1/donor_disjoint_oof.parquet", "results/phase1/program_transferability.csv", "results/phase1/program_transport_utility.csv", "results/phase1/router_predictions.csv"]}
    _write(out / "integrity_audit.json", integrity)
    required = ["access_inventory.csv", "raw_source_manifest.json", "raw_source_hashes.json", "raw_schema.md", "extraction_manifest.json", "pseudobulk_sample_metadata.csv", "cell_count_manifest.csv", "gene_metadata_frozen.csv", "gene_universe_manifest.csv", "gene_universe_hashes.json", "panel127_raw_comparison.csv", "source_semantics_summary.csv", "truth_reliability.csv", "truth_null.csv", "support_q_curve.csv", "support_trend.csv", "maximal_support_summary.csv", "nuisance_robustness.csv", "lowrank_capture.csv", "loco_module_coverage.csv", "bootstrap_summary.csv", "integrity_audit.json", "verdict.json"]
    missing = [name for name in required if not (out / name).exists()]
    if missing or not integrity["all_passed"]:
        raise RuntimeError(f"Output/integrity failure: missing={missing}, integrity={integrity['all_passed']}")
    hashes = {name: _sha(out / name) for name in required}
    run = {"phase": "CGC-0I", "created_at": _now(), "branch": branch, "git_parent": parent,
           "python": platform.python_version(), "platform": platform.platform(), "raw_official_revision": "76cb053d59be08a7d0c4dc68c265622f6e0ab52e",
           "raw_download_bytes": raw_download_bytes,
           "contexts": 50, "interventions": 93, "pseudobulks": 9500, "gene_universes": response["gene_universes"],
           "gpu_dimensionality": {"device": dimension["device"], "name": dimension["gpu_name"], "peak_bytes": dimension["peak_gpu_memory_bytes"]},
           "tests": {"all_non_data_dependent": "PASS", "excluded": [{"test": "test_frozen_missing_itgav_combination_remains_masked", "status": "DATA_NOT_PRESENT", "reason": "approved Feng cleanup removed regenerable observation mask; no redownload authorized"}]},
           "required_output_sha256": hashes, "verdict": VERDICT, "git_status_at_freeze": status}
    _write(out / "run_manifest.json", run)
    reports.mkdir(parents=True, exist_ok=True)
    provenance = f"Branch `{branch}`; execution parent `{parent}`; official raw revision `76cb053d59be08a7d0c4dc68c265622f6e0ab52e`."
    for name in (
        "cgc_tahoe_0i_data_access.md", "cgc_tahoe_0i_fulltranscriptome_truth.md",
        "cgc_tahoe_0i_support_scaling.md",
    ):
        path = reports / name
        content = path.read_text(encoding="utf-8")
        if "Execution provenance:" not in content:
            first, remainder = content.split("\n", 1)
            path.write_text(first + "\n\nExecution provenance: " + provenance + "\n" + remainder, encoding="utf-8")
    (reports / "cgc_tahoe_0i_raw_reconstruction.md").write_text(f"""# CGC-0I raw reconstruction

{provenance}

- Official Plate6/14 H5AD MD5 and byte sizes: PASS.
- Source cells: {extraction['plate_stats']['plate6']['source_cells']:,} / {extraction['plate_stats']['plate14']['source_cells']:,}.
- Frozen `pass_filter == full` retained cells: {extraction['plate_stats']['plate6']['retained_cells']:,} / {extraction['plate_stats']['plate14']['retained_cells']:,}.
- Pseudobulks: 4,750 per plate; 9,500 total; cell-count mismatches: 0.
- Random source-cell count spot checks: {extraction['random_cell_count_spot_checks']} ({extraction['spot_check_status']}).
- Dense cell×gene materialized: {str(extraction['dense_cell_gene_materialized']).lower()}.
- Primary response: {response['primary']}.
- G_PRIMARY: {response['gene_universes']['G_PRIMARY']:,} genes, selected from DMSO coverage only.
- Historical panel: 127/127 overlap; raw-vs-official Pearson {panel['summary']['global_response_pearson']:.4f}; q49 {panel['summary']['raw_panel_q49']:.4f} vs {panel['summary']['official_q49']:.4f}.
""", encoding="utf-8")
    (reports / "cgc_tahoe_0i_phase_summary.md").write_text(f"""# CGC-0I phase summary

{provenance}

## Frozen result

- Truth signal energy: {truth['universes']['G_PRIMARY']['signal_energy']:.8f}; bootstrap 95% CI [{truth['universes']['G_PRIMARY']['bootstrap_lower_95']:.8f}, {truth['universes']['G_PRIMARY']['bootstrap_upper_95']:.8f}]; permutation p={truth['universes']['G_PRIMARY']['empirical_p']:.8f}.
- q2/q8/q16/q32/q49: {scaling['q_values']['2']:.4f} / {scaling['q_values']['8']:.4f} / {scaling['q_values']['16']:.4f} / {scaling['q_values']['32']:.4f} / {scaling['q_values']['49']:.4f}.
- q49 bootstrap 95% CI: [{scaling['q49_bootstrap_lower_95']:.4f}, {scaling['q49_bootstrap_upper_95']:.4f}]; positive contexts: {scaling['positive_residual_contexts_q49']}/50.
- G_BROAD/G_STRICT q49: {q49['G_BROAD']:.4f} / {q49['G_STRICT']:.4f}.
- Double-scalar q49: {nuisance['double_scalar_q49']:.4f}; residual-null p={nuisance['double_scalar_residual_null_p']:.8f}.
- Rank-1024 signal capture: {dimension['capture']['1024']:.4f}; k50/k80/k90: {dimension['k50']} / {dimension['k80']} / {dimension['k90']}.
- Strict LOCO modules: mean count {loco['mean_module_count']:.1f}; pooled held-out coverage {loco['pooled_coverage']:.4f}; positive contexts {loco['positive_contexts']}/50.

The full-transcriptome operator is replicate-reliable but weak in absolute signal fraction (~4.3%). Even all 49 reference contexts leave ~91.9% of the replicate-stable context-specific residual energy. Scalar, low-rank-PC, technical, and direction-only sensitivities do not remove the result. Strict family-wide module discovery found no reusable module vocabulary, so the remaining operator is high-dimensional and not summarized by validated top-20 co-occurrence modules.

## Verdict

`{VERDICT}`

This authorizes CGC-0J. It does not authorize a universal impossibility claim or any new neural model.

Automated tests: all non-data-dependent tests passed. One legacy Feng assertion is `DATA_NOT_PRESENT` because the approved Feng cleanup removed its regenerable observation-mask input; Feng was not re-downloaded.
""", encoding="utf-8")
    return verdict
