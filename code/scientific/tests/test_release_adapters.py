"""Tiny input-free tests for bounded release reporting and execution guards."""
from __future__ import annotations

import importlib.util
import csv
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCI = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finite_context_correction_has_44_unique_derangements():
    module = load("current_state_summary")
    rng = np.random.default_rng(23)
    truth = rng.normal(size=(5, 3, 7))
    truth -= truth.mean(axis=0, keepdims=True)
    result = module.exact_context_null(truth, truth.copy())
    assert result["unique_derangements"] == 44
    assert result["exceedances"] == 0
    assert result["conservative_context_null_p"] == 1 / 45


def test_finite_context_correction_rejects_changed_context_panel():
    module = load("current_state_summary")
    with pytest.raises(ValueError, match="exactly five"):
        module.exact_context_null(np.ones((4, 2, 3)), np.ones((4, 2, 3)))


def test_resolution_selector_never_returns_invalid_random_rows():
    module = load("render_valid_resolution")
    rows = []
    for budget in ("m49_k92", "m40_k4"):
        for name in ("g_gene", "g_pathway"):
            rows.append({"family": "point_estimate", "contrast": f"{budget}_{name}", "estimate": ".1"})
        rows.append({"family": "PATHWAY_PRIMARY_2BUDGET_X_2CONTRAST", "contrast": f"{budget}_pathway_minus_gene", "estimate": ".2"})
        rows.append({"family": "point_estimate", "contrast": f"{budget}_g_matched_random_median", "estimate": "999"})
    result = module.select_clean_rows(rows)
    assert len(result) == 6
    assert not any("random" in row["contrast"] for row in result)


@pytest.mark.parametrize("bundle,old_entry", [
    ("state_official_summary", "scripts/cgc_state_1_analysis.py"),
    ("resolution_pathway_frozen", "scripts/cgc_resolution2_pathway.py"),
    ("resolution_pathway_frozen", "scripts/cgc_resolution2_inference.py"),
    ("resolution_pathway_frozen", "scripts/cgc_resolution2_report.py"),
])
def test_active_launcher_rejects_archival_all_stages(bundle, old_entry):
    result = subprocess.run([sys.executable, str(SCI / "launch.py"), bundle, "--entrypoint", old_entry, "--execute"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "archival-only" in result.stderr


def test_imported_source_tamper_is_rejected(tmp_path):
    module = load("launch")
    target = tmp_path / "src/example.py"
    target.parent.mkdir()
    target.write_bytes(b"x=1\n")
    manifest = {"files": [{"path": "src/example.py", "role": "scientific_source", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]}
    module.verify_import_sources(tmp_path, manifest)
    target.write_bytes(b"x=2\n")
    with pytest.raises(RuntimeError, match="source/config hash mismatch"):
        module.verify_import_sources(tmp_path, manifest)


def test_state_split_metadata_tamper_is_rejected(tmp_path):
    module = load("current_state_summary")
    file = tmp_path / "zero_shot_split_manifest.csv"
    file.write_bytes(b"frozen split\n")
    manifest = {"files": [{"path": "results/cgc_state_1/zero_shot_split_manifest.csv", "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}]}
    module.verified_metadata_path(manifest, tmp_path, file.name)
    file.write_bytes(b"substituted split\n")
    with pytest.raises(RuntimeError, match="metadata hash mismatch"):
        module.verified_metadata_path(manifest, tmp_path, file.name)


@pytest.mark.parametrize("tamper_suffix", ["_results.csv", "_agg_results.csv"])
def test_state_ancillary_official_metric_tamper_is_rejected(tmp_path, tamper_suffix):
    module = load("current_state_summary")
    directory = tmp_path / "ST_SE/split_0"
    directory.mkdir(parents=True)
    records = []
    for suffix in ("_pred_de.csv", "_real_de.csv", "_results.csv", "_agg_results.csv"):
        path = directory / ("artifact" + suffix)
        path.write_bytes(b"metric,value\nfrozen,1\n")
        records.append({"relative_path": "data/state_official_predictions/" + path.relative_to(tmp_path).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    splits = [{"variant": "ST_SE", "split": "0"}]
    assert module.verify_official_files(tmp_path, inventory, splits) == 4
    (directory / ("artifact" + tamper_suffix)).write_bytes(b"changed metric\n")
    with pytest.raises(RuntimeError, match="official artifact hash mismatch"):
        module.verify_official_files(tmp_path, inventory, splits)
