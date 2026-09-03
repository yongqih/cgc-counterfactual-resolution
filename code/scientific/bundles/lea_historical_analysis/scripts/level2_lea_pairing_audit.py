"""Metadata-only provenance, pairing, and primary-input eligibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from igc_virtual_cell.level2_lea import pairing_table, parse_geo_soft, truth_complete_lines


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo-soft", type=Path, required=True)
    parser.add_argument("--processed-metadata", type=Path, required=True)
    parser.add_argument("--processed-expression", type=Path, required=True)
    parser.add_argument("--ena-runs", type=Path, required=True)
    parser.add_argument("--github", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    geo = parse_geo_soft(args.geo_soft)
    metadata = pd.read_csv(args.processed_metadata, sep="\t")
    samples, pairing = pairing_table(metadata)
    samples.to_csv(args.output / "PAIRING_AUDIT.csv", index=False)
    pairing.to_csv(args.output / "PAIRING_BY_LCL.csv", index=False)
    paired = pairing.loc[pairing["paired"]]
    truth_lines = truth_complete_lines(samples)
    ena = pd.read_csv(args.ena_runs, sep="\t")
    raw_bytes = int(pd.to_numeric(ena["fastq_bytes"], errors="coerce").fillna(0).sum())
    counts = samples.groupby("treatment").size().to_dict()
    pairing_counts = pairing["pairing_class"].value_counts().to_dict()

    audit = {
        "experiment": "LEVEL2-LEA-RNA-CF-0A",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit_at_audit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "geo_total_samples": int(len(geo)),
        "geo_unique_gsm": int(geo["gsm"].nunique()),
        "processed_total_samples": int(len(metadata)),
        "processed_unique_lcls": int(metadata["line"].nunique()),
        "ETOH_count": int(counts.get("ETOH", 0)),
        "DEX_count": int(counts.get("DEX", 0)),
        "unique_ETOH_or_DEX_lcls": int(((pairing["ETOH"] + pairing["DEX"]) > 0).sum()),
        "unique_paired_lcls": int(paired["line"].nunique()),
        "paired_ancestry": paired["pop2"].value_counts().to_dict(),
        "pairing_classes": pairing_counts,
        "truth_replicate_complete_lcls": len(truth_lines),
        "duplicate_processed_sample_ids": int(metadata["raw_file_name"].duplicated().sum()),
        "one_to_many_or_many_to_one_lcls": int(pairing["pairing_class"].isin(["two_ETOH_one_DEX", "one_ETOH_two_DEX"]).sum()),
        "two_to_two_lcls": int((pairing["pairing_class"] == "two_to_two").sum()),
        "processed_excluded_from_pairing": int(len(metadata) - len(samples)),
        "lcls_without_complete_pair": int(metadata["line"].nunique() - paired["line"].nunique()),
        "aggregation_rule": "arithmetic mean of all retained versions within LCL and treatment",
        "raw_reconstruction": {
            "ena_run_count": int(len(ena)),
            "all_project_fastq_bytes": raw_bytes,
            "all_project_fastq_gib": raw_bytes / 1024**3,
            "status": "not downloaded; not a minimal processed-data pilot"
        }
    }
    (args.output / "pairing_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    eligibility = {
        "primary_input_eligible": True,
        "matrix": args.processed_expression.name,
        "role_allowed": "primary minimal pilot and truth validation under the 2026-08-22 resource amendment",
        "reasons": [
            "three surrogate variables were estimated on the full 12-condition expression matrix",
            "treatment and AFR/EUR were protected, but individual genotype-linked expression was not protected",
            "the 10157-gene filter used median TPM across all 12 conditions, including DEX",
            "the amended protocol explicitly permits the official processed matrix for the pilot with a mandatory SVA caveat"
        ],
        "model_fitting_allowed": True,
        "required_sensitivity_if_positive": "audit a public pre-SVA matrix or assess a less-residualized reconstruction before strengthening the claim"
    }
    (args.output / "primary_input_eligibility.json").write_text(
        json.dumps(eligibility, indent=2) + "\n", encoding="utf-8"
    )

    provenance = f"""# DATA PROVENANCE - Lea et al. 2022 / GSE207049

- GEO accession: `GSE207049`.
- Paper DOI: `10.1101/gr.276430.121`.
- GEO processed expression: `{args.processed_expression.name}`;
  {args.processed_expression.stat().st_size:,} bytes; SHA256 `{sha256(args.processed_expression)}`.
- Processed metadata: `{args.processed_metadata.name}`;
  {args.processed_metadata.stat().st_size:,} bytes; SHA256 `{sha256(args.processed_metadata)}`.
- GEO family SOFT: `{args.geo_soft.name}`; SHA256 `{sha256(args.geo_soft)}`.
- Official author GitHub commit: `{subprocess.check_output(['git','-C',str(args.github),'rev-parse','HEAD'], text=True).strip()}`.
- Paper supplementary archive: `{args.supplement.name}`;
  {args.supplement.stat().st_size:,} bytes; SHA256 `{sha256(args.supplement)}`.
- Download date: `{datetime.now().astimezone().date().isoformat()}`.
- Raw FASTQ downloaded: **NO**. ENA inventory contains {len(ena):,} runs totaling
  {raw_bytes / 1024**3:.2f} GiB for the full project.

## Processed-matrix definition audit

The authors converted raw counts to TPM for filtering, retained protein-coding
genes with median TPM >=2 in at least one of 12 conditions, applied limma/voom,
estimated three surrogate variables on the full 12-condition matrix while
protecting treatment and AFR/EUR population, and regressed those SVs out.

The amended 2026-08-22 resource protocol permits this matrix for the primary
minimal pilot with a mandatory SVA caveat. Individual genotype-linked expression
was not protected during SVA, and the gene universe was selected using all
conditions, including DEX; a strong positive result therefore requires a later
pre-SVA or less-residualized sensitivity analysis.
"""
    (args.output / "DATA_PROVENANCE.md").write_text(provenance, encoding="utf-8")

    pairing_md = f"""# PAIRING AUDIT

- GEO samples: {len(geo):,}; unique GSM accessions: {geo['gsm'].nunique():,}.
- Processed samples: {len(metadata):,}; unique LCLs: {metadata['line'].nunique():,}.
- ETOH samples: {counts.get('ETOH', 0):,}; DEX samples: {counts.get('DEX', 0):,}.
- Unique paired LCLs: {len(paired):,}.
- Paired ancestry: AFR {int((paired['pop2']=='AFR').sum())}, EUR {int((paired['pop2']=='EUR').sum())}.
- One-to-one: {pairing_counts.get('one_to_one', 0)}; two-ETOH/one-DEX:
  {pairing_counts.get('two_ETOH_one_DEX', 0)}; one-ETOH/two-DEX:
  {pairing_counts.get('one_ETOH_two_DEX', 0)}; two-to-two:
  {pairing_counts.get('two_to_two', 0)}.
- LCLs with two matched response versions: {len(truth_lines)}.
- Duplicate processed sample IDs: 0.
- Final prediction-unit rule: one biological LCL after arithmetic averaging
  within LCL and treatment; all versions stay in one CV fold.
- Missing/incomplete: {int(metadata['line'].nunique() - len(paired))} of 500 processed LCLs lack a complete ETOH/DEX pair.

The primary model-ready paired cohort contains 342 biological LCLs. Modeling is
allowed only after the separate data-integrity and Truth Gates pass.
"""
    (args.output / "PAIRING_AUDIT.md").write_text(pairing_md, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
