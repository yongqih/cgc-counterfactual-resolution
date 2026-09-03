"""Freeze outcome-independent annotations for CGC-BIO-2 / CGC-MULTI-2.

This script is intentionally limited to provenance, axes, pharmacologic
annotations, target-family assignment, and predeclared feature eligibility.
It must run before any BIO-2 or MULTI-2 outcome calculation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell")
OUT = ROOT / "results" / "cgc_mechanism_followup"
HGNC = ROOT / "data" / "cgc_mechanism_followup" / "hgnc_complete_set_2026-07-07.tsv"

EXPECTED = {
    "bio1_gene_arrays.npz": "6b46053324175bfc34b2534bb59ae5519a1aa47b504ddeb2887956953f222228",
    "bio1_projected_programs.npz": "b481b44f29a22227527e9e9c05c48dbc203e0b33a48406e49b0bb7557ca60999",
    "drug_target_moa_manifest.csv": "33efd4cebbf7965bd20bc90a44b2b9d917b83740dad1991979d9e1dbc7d3ab59",
    "hgnc_complete_set_2026-07-07.tsv": "d63733c257ffeca6116f6fd401960da8ed113d213463bc0d1a6be26dc87e7f13",
}

# These official HGNC groups are structural, host-gene, or complex labels rather
# than molecular target families. The rule and exact patterns are frozen before
# outcomes and are applied uniformly to every target.
FAMILY_EXCLUDE_PATTERNS = (
    "domain containing",
    "complex subunits",
    "complex 1",
    "complex 2",
    "host genes",
    "CD molecules",
    "minor histocompatibility antigens",
    "protein phosphatase 1 regulatory subunits",
)
FAMILY_MIN_SIZE = 2
FAMILY_MAX_SIZE = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_set(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item.strip() for item in str(value).replace(",", ";").split(";") if item.strip()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "gene_residual": SOURCE_ROOT / "data/cgc_bio1_official/bio1_gene_arrays.npz",
        "program_residual": SOURCE_ROOT / "data/cgc_bio1_official/bio1_projected_programs.npz",
        "drug_annotation": ROOT / "results/cgc_tahoe_bio1/drug_target_moa_manifest.csv",
        "hgnc": HGNC,
    }
    observed_hashes = {key: sha256(path) for key, path in paths.items()}
    for key, path in paths.items():
        expected = EXPECTED[path.name]
        if observed_hashes[key] != expected:
            raise RuntimeError(f"Frozen hash mismatch for {key}: {observed_hashes[key]} != {expected}")

    hgnc = pd.read_csv(HGNC, sep="\t", low_memory=False)
    approved_pc = hgnc[(hgnc["status"] == "Approved") & (hgnc["locus_group"] == "protein-coding gene")]
    group_sizes: dict[str, int] = {}
    group_names: dict[str, str] = {}
    for row in approved_pc[["gene_group_id", "gene_group"]].dropna().itertuples(index=False):
        ids = str(row.gene_group_id).split("|")
        names = str(row.gene_group).split("|")
        for group_id, name in zip(ids, names, strict=True):
            group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
            group_names[group_id] = name

    symbol_groups: dict[str, list[str]] = {}
    symbol_group_names: dict[str, list[str]] = {}
    for row in hgnc[["symbol", "gene_group_id", "gene_group"]].dropna(subset=["symbol"]).itertuples(index=False):
        if pd.isna(row.gene_group_id) or pd.isna(row.gene_group):
            continue
        eligible: list[tuple[str, str]] = []
        for group_id, name in zip(str(row.gene_group_id).split("|"), str(row.gene_group).split("|"), strict=True):
            normalized = name.casefold()
            if not FAMILY_MIN_SIZE <= group_sizes.get(group_id, 0) <= FAMILY_MAX_SIZE:
                continue
            if any(pattern.casefold() in normalized for pattern in FAMILY_EXCLUDE_PATTERNS):
                continue
            eligible.append((group_id, name))
        symbol_groups[str(row.symbol)] = sorted({x[0] for x in eligible}, key=lambda x: (group_sizes[x], x))
        symbol_group_names[str(row.symbol)] = [group_names[x] for x in symbol_groups[str(row.symbol)]]

    drugs = pd.read_csv(paths["drug_annotation"])
    rows: list[dict[str, object]] = []
    all_targets: set[str] = set()
    mapped_targets: set[str] = set()
    targets_with_family: set[str] = set()
    hgnc_symbols = set(hgnc.symbol.astype(str))
    for row in drugs.itertuples(index=False):
        targets = split_set(row.target_list)
        all_targets.update(targets)
        family_ids = sorted({g for target in targets for g in symbol_groups.get(target, [])})
        family_names = [group_names[g] for g in family_ids]
        mapped_targets.update(target for target in targets if target in hgnc_symbols)
        targets_with_family.update(target for target in targets if symbol_groups.get(target))
        rows.append(
            {
                "intervention_axis": len(rows),
                "intervention_id": row.intervention_id,
                "drug_identity": row.drug,
                "concentration": row.concentration,
                "concentration_unit": row.concentration_unit,
                "moa_broad": getattr(row, "_6"),
                "moa_fine": getattr(row, "_7"),
                "exact_targets": ";".join(sorted(targets)),
                "n_exact_targets": len(targets),
                "hgnc_target_family_ids": ";".join(family_ids),
                "hgnc_target_family_names": ";".join(family_names),
                "n_hgnc_target_families": len(family_ids),
                "target_annotation_available": bool(targets),
                "target_family_annotation_available": bool(family_ids),
            }
        )
    frozen = pd.DataFrame(rows)
    frozen.to_csv(OUT / "BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv", index=False)

    created = datetime.now(timezone.utc).isoformat()
    provenance = {
        "phase": "CGC-BIO-2 annotation freeze",
        "created_at": created,
        "outcomes_inspected_before_freeze": False,
        "base_commit": "6d3b3203504eb7f18917c949cfd94536d6f5648d",
        "frozen_axes": {"plates": ["Plate6", "Plate14"], "contexts": 50, "interventions": 93, "genes": 25695},
        "sources": {
            "frozen_gene_residual": {"path": str(paths["gene_residual"]), "sha256": observed_hashes["gene_residual"]},
            "frozen_program_residual": {"path": str(paths["program_residual"]), "sha256": observed_hashes["program_residual"]},
            "bio1_drug_annotation": {"path": str(paths["drug_annotation"]), "sha256": observed_hashes["drug_annotation"]},
            "hgnc_gene_family": {
                "path": str(paths["hgnc"]),
                "url": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
                "declared_snapshot_date": "2026-07-07",
                "downloaded_at": created,
                "sha256": observed_hashes["hgnc"],
                "bytes": paths["hgnc"].stat().st_size,
                "fields": ["symbol", "status", "locus_group", "gene_group", "gene_group_id"],
            },
        },
        "target_family_rule": {
            "source": "HGNC gene_group_id",
            "approved_protein_coding_group_size": [FAMILY_MIN_SIZE, FAMILY_MAX_SIZE],
            "excluded_name_patterns": list(FAMILY_EXCLUDE_PATTERNS),
            "multi_target_aggregation": "union of eligible HGNC group IDs",
            "pair_match": "at least one shared eligible group ID; exact-target level is identical non-empty canonical target set",
        },
        "coverage": {
            "unique_target_strings": len(all_targets),
            "targets_mapped_to_hgnc": len(mapped_targets),
            "targets_with_eligible_family": len(targets_with_family),
            "interventions_with_exact_target": int(frozen.target_annotation_available.sum()),
            "interventions_with_target_family": int(frozen.target_family_annotation_available.sum()),
        },
    }
    write_json(OUT / "BIO2_ANNOTATION_PROVENANCE.json", provenance)

    multi_provenance = {
        "phase": "CGC-MULTI-2 annotation freeze",
        "created_at": created,
        "outcomes_inspected_before_freeze": False,
        "mechanism_mapping": "identical BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv target/family mapping plus frozen Reactome weights",
        "bio2_annotation_sha256": sha256(OUT / "BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv"),
        "reactome_weights": {
            "path": str(SOURCE_ROOT / "data/cgc_bio1_official/frozen_program_weights.parquet"),
            "sha256": "f5c4faa94fe03b2d088b72743da175380dfcf2edbfacd0077cc25eb19d2ca7e9",
        },
        "depmap_release": "DepMap Public 26Q1 / frozen MULTI-1 source manifest",
        "modality_source_manifest": "results/cgc_tahoe_multi1/modality_source_manifest.json",
        "functional_dependency_label": "FUNCTIONAL_PERTURBATIONAL_ORACLE",
    }
    write_json(OUT / "MULTI2_ANNOTATION_PROVENANCE.json", multi_provenance)

    print(f"Frozen {len(frozen)} intervention annotations in {OUT}")


if __name__ == "__main__":
    main()
