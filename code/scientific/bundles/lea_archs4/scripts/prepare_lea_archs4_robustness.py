from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.audit_lea_archs4_coverage import ARCHS4_URL, HTTPRangeFile, remote_head


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "level2_lea" / "archs4_robustness"
AUTHORITY = "35cf07d8f3d90450b365893b56aa127dca7c4774"
HISTORICAL = "c3d074412ecc1be0958141726db16a1ed0036a04"
PROTOCOL = "8fad55f7a39ff64383c8a0440bcb8c4e3892c7ee"
EXCLUDED = {"LineNA": "EXCLUDED_UNRESOLVED_GEO_IDENTITY", "Line223": "EXCLUDED_UNRESOLVED_GEO_IDENTITY"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_source(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{HISTORICAL}:{path}"], cwd=ROOT, text=True)


def historical_pairing_table(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    module = types.ModuleType("frozen_level2_lea")
    exec(compile(git_source("src/igc_virtual_cell/level2_lea.py"), f"{HISTORICAL}:level2_lea.py", "exec"), module.__dict__)
    return module.pairing_table(metadata)


def archs4_genes() -> tuple[list[str], list[str], dict[str, object]]:
    head = remote_head(ARCHS4_URL)
    range_file = HTTPRangeFile(ARCHS4_URL, int(head["content_length"]), block_size=256 * 1024, max_blocks=256)
    with h5py.File(range_file, "r", driver="fileobj") as handle:
        identifiers = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["meta/genes/ensembl_gene"]]
        symbols = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["meta/genes/symbol"]]
    return identifiers, symbols, {
        **head,
        "remote_url": ARCHS4_URL,
        "range_requests": range_file.request_count,
        "range_bytes_transferred": range_file.bytes_transferred,
    }


def write_fold_manifest(source_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata_path = source_root / "data" / "level2_lea_official" / "31Mar21_all_runs_voom_resid_metadata.txt"
    oof_path = source_root / "results" / "level2_lea" / "LEVEL2_OOF_FROZEN.npz"
    candidates_path = ROOT / "results" / "level2_lea" / "archs4_coverage_audit" / "ARCHS4_GSE207049_CANDIDATE_GSMS.csv"
    metadata = pd.read_csv(metadata_path, sep="\t")
    samples, pairing = historical_pairing_table(metadata)
    paired_lines = sorted(pairing.loc[pairing["paired"], "line"])
    frozen = np.load(oof_path, allow_pickle=False)
    folds = frozen["folds"].astype(int)
    if len(paired_lines) != 342 or len(folds) != 342:
        raise RuntimeError(f"Historical fold/core mismatch: lines={len(paired_lines)}, folds={len(folds)}")
    candidates = pd.read_csv(candidates_path, dtype=str)
    candidates = candidates.loc[candidates["archs4_v2_5_present"].str.lower() == "true"].copy()
    identity = pairing.set_index("line")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(paired_lines):
        line_candidates = candidates.loc[candidates["line"] == line]
        etoh = sorted(line_candidates.loc[line_candidates["treatment"] == "ETOH", "gsm"].unique())
        dex = sorted(line_candidates.loc[line_candidates["treatment"] == "DEX", "gsm"].unique())
        included = line not in EXCLUDED
        if included and (not etoh or not dex):
            raise RuntimeError(f"Resolved line lacks a treatment group: {line}")
        historical_etoh = sorted(samples.loc[(samples["line"] == line) & (samples["treatment"] == "ETOH"), "raw_file_name"].astype(str))
        historical_dex = sorted(samples.loc[(samples["line"] == line) & (samples["treatment"] == "DEX"), "raw_file_name"].astype(str))
        row = identity.loc[line]
        rows.append(
            {
                "lcl_id": line,
                "historical_row_index": index,
                "historical_fold": int(folds[index]),
                "ancestry": row["pop2"],
                "population": row["pop"],
                "individual_id": row["1000_genomes_id1"],
                "historical_ETOH_group": " | ".join(historical_etoh),
                "historical_DEX_group": " | ".join(historical_dex),
                "ARCHS4_ETOH_GSM_group": " | ".join(etoh),
                "ARCHS4_DEX_GSM_group": " | ".join(dex),
                "ARCHS4_ETOH_versions": len(etoh),
                "ARCHS4_DEX_versions": len(dex),
                "included_matched_340": included,
                "inclusion_exclusion_reason": "INCLUDED_EXACT_TITLE_IDENTITY" if included else EXCLUDED[line],
            }
        )
    frame = pd.DataFrame(rows)
    included = frame.loc[frame["included_matched_340"]]
    if len(included) != 340 or included["lcl_id"].isin(EXCLUDED).any():
        raise RuntimeError("Matched 340 fold manifest gate failed")
    if sorted(included["historical_fold"].unique()) != list(range(5)):
        raise RuntimeError("Historical fold identity incomplete")
    frame.to_csv(OUT / "ARCHS4_MATCHED_340_FOLD_MANIFEST.csv", index=False)
    return frame, {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "oof_path": str(oof_path),
        "oof_sha256": sha256(oof_path),
        "candidate_mapping_path": str(candidates_path.relative_to(ROOT)).replace("\\", "/"),
        "candidate_mapping_sha256": sha256(candidates_path),
    }


def write_gene_axis(source_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    matrix_path = source_root / "data" / "level2_lea_official" / "GSE207049_31Mar21_all_runs_voom_resid.txt.gz"
    historical = pd.read_csv(matrix_path, sep="\t", usecols=["GeneID"], dtype=str)["GeneID"]
    counts = historical.value_counts(sort=False)
    unique_order = list(dict.fromkeys(historical.tolist()))
    identifiers, symbols, archs4_meta = archs4_genes()
    arch_counts = pd.Series(identifiers).value_counts()
    arch_index = {identifier: index for index, identifier in enumerate(identifiers) if arch_counts[identifier] == 1}
    symbol_by_id = dict(zip(identifiers, symbols, strict=True))
    rows: list[dict[str, object]] = []
    for order, gene in enumerate(unique_order):
        arch_count = int(arch_counts.get(gene, 0))
        if arch_count == 1:
            status = "EXACT_ONE_TO_ONE"
            index = arch_index[gene]
            included = True
        elif arch_count == 0:
            status = "NOT_IN_ARCHS4"
            index = np.nan
            included = False
        else:
            status = "AMBIGUOUS_ARCHS4_DUPLICATE"
            index = np.nan
            included = False
        rows.append(
            {
                "historical_unique_order": order,
                "historical_GeneID": gene,
                "historical_row_count": int(counts[gene]),
                "historical_duplicate_collapse": "arithmetic_mean" if int(counts[gene]) > 1 else "not_required",
                "ARCHS4_ensembl_gene": gene if arch_count else "",
                "ARCHS4_symbol": symbol_by_id.get(gene, ""),
                "ARCHS4_row_index": index,
                "ARCHS4_identifier_count": arch_count,
                "mapping_status": status,
                "included_primary_axis": included,
                "exclusion_reason": "" if included else status,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ARCHS4_GENE_AXIS_RECONCILIATION.csv", index=False)
    matched = int(frame["included_primary_axis"].sum())
    coverage = matched / len(frame)
    if len(historical) != 10_157 or len(frame) != 10_120:
        raise RuntimeError("Historical gene-axis counts changed")
    if len(identifiers) != 67_186 or len(set(identifiers)) != 67_186:
        raise RuntimeError("ARCHS4 GeneID axis is not the expected unique 67,186 axis")
    if coverage < 0.95:
        raise RuntimeError(f"ARCHS4_GENE_AXIS_COVERAGE_REVIEW_REQUIRED: {matched}/{len(frame)}")
    return frame, {
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256(matrix_path),
        "historical_rows": len(historical),
        "historical_unique_geneids": len(frame),
        "archs4_gene_rows": len(identifiers),
        "archs4_unique_geneids": len(set(identifiers)),
        "matched_one_to_one": matched,
        "coverage_fraction": coverage,
        "excluded_missing": int((frame["mapping_status"] == "NOT_IN_ARCHS4").sum()),
        "excluded_ambiguous": int((frame["mapping_status"] == "AMBIGUOUS_ARCHS4_DUPLICATE").sum()),
        "archs4_remote_metadata": archs4_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    fold_manifest, input_meta = write_fold_manifest(source_root)
    gene_axis, gene_meta = write_gene_axis(source_root)
    gate = {
        "created_at": now(),
        "authority_commit": AUTHORITY,
        "historical_implementation_commit": HISTORICAL,
        "version_protocol_commit": PROTOCOL,
        "matched_core_lines": int(fold_manifest["included_matched_340"].sum()),
        "excluded_lines": sorted(fold_manifest.loc[~fold_manifest["included_matched_340"], "lcl_id"]),
        "fold_counts": fold_manifest.loc[fold_manifest["included_matched_340"], "historical_fold"].value_counts().sort_index().to_dict(),
        "gene_axis": gene_meta,
        "input_provenance": input_meta,
        "expression_outcomes_loaded": False,
        "model_outcomes_loaded": False,
        "gate": "ARCHS4_MATCHED_340_PREMODEL_GATE_PASS",
    }
    (OUT / "ARCHS4_PREMODEL_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"core": gate["matched_core_lines"], "fold_counts": gate["fold_counts"], **gene_meta}, indent=2, default=str))
    print(gate["gate"])


if __name__ == "__main__":
    main()
