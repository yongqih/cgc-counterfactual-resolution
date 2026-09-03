"""Streamed integrity and linkage audit for the official GSE207049 matrix."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from igc_virtual_cell.level2_lea import pairing_table, parse_geo_soft


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def streamed_matrix_audit(path: Path) -> dict[str, object]:
    uncompressed_md5 = hashlib.md5()
    row_count = 0
    field_counts: set[int] = set()
    header: list[str] | None = None
    genes: list[str] = []
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            uncompressed_md5.update(raw)
            fields = raw.rstrip(b"\r\n").split(b"\t")
            row_count += 1
            field_counts.add(len(fields))
            if row_count == 1:
                header = [item.decode("utf-8") for item in fields]
            else:
                genes.append(fields[0].decode("utf-8"))
    if header is None:
        raise AssertionError("Expression matrix is empty")
    return {
        "uncompressed_md5": uncompressed_md5.hexdigest(),
        "matrix_total_rows_with_header": row_count,
        "matrix_gene_rows": row_count - 1,
        "matrix_total_columns": len(header),
        "matrix_sample_columns": len(header) - 1,
        "first_header": header[0],
        "sample_names": header[1:],
        "gene_ids": genes,
        "field_counts": sorted(field_counts),
    }


def matrix_title(sample_name: str) -> str:
    return sample_name.split(".R1.Aligned.counts", 1)[0]


GEO_TREATMENT_LABEL = {
    "DEX": "Dexamethasone",
    "ETOH": "Ethanol",
    "H20": "Water",
    "IFNG": "Interferon Gamma",
    "IGF": "Insulin-like Growth Factor 1",
    "GARD": "Gardiquimod",
    "BAFF": "B-cell activating factor",
    "FSL1": "FSL-1",
    "TUNIC": "Tunicamycin",
    "ACRYL": "ACRYL",
    "BPA": "BPA",
    "PFOA": "PFOA",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--filtered-info", type=Path, required=True)
    parser.add_argument("--gene-ids", type=Path, required=True)
    parser.add_argument("--geo-soft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    matrix = streamed_matrix_audit(args.matrix)
    metadata = pd.read_csv(args.metadata, sep="\t")
    filtered = pd.read_csv(args.filtered_info, sep="\t")
    gene_ids = pd.read_csv(args.gene_ids, sep="\t").iloc[:, 0].astype(str).tolist()
    geo = parse_geo_soft(args.geo_soft)
    samples, pairing = pairing_table(metadata)
    pairing.to_csv(args.output / "GSE207049_PAIRING_BY_LCL.csv", index=False)
    pairing.loc[~pairing["paired"]].to_csv(
        args.output / "GSE207049_EXCLUDED_LINES.csv", index=False
    )

    matrix_samples = matrix.pop("sample_names")
    matrix_genes = matrix.pop("gene_ids")
    ids = filtered["id"].astype(int).tolist()
    id_mapping_matches = [
        filtered.iloc[i]["name"] == matrix_samples[ids[i] - 1]
        for i in range(len(filtered))
    ]

    geo_by_title = {key: group for key, group in geo.groupby("title", sort=False)}
    rng = random.Random(207049)
    sampled_positions = sorted(rng.sample(range(len(matrix_samples)), 20))
    linkage_rows: list[dict[str, object]] = []
    for position in sampled_positions:
        name = matrix_samples[position]
        row = metadata.iloc[position]
        title = matrix_title(name)
        exact_title_matches = geo_by_title.get(title, pd.DataFrame())
        biological_matches = geo.loc[
            (geo["cell_line"] == row["1000_genomes_id1"])
            & (geo["treatment_label"] == GEO_TREATMENT_LABEL[row["treatment"]])
        ]
        geo_matches = exact_title_matches if len(exact_title_matches) else biological_matches
        linkage_method = "exact_GEO_title" if len(exact_title_matches) else "cell_line_and_treatment"
        linkage_rows.append({
            "matrix_sample_position_1based": position + 1,
            "matrix_name": name,
            "metadata_name": row["raw_file_name"],
            "line": row["line"],
            "treatment": row["treatment"],
            "geo_title": title,
            "geo_match_count": len(geo_matches),
            "geo_gsm": ";".join(geo_matches.get("gsm", pd.Series(dtype=str)).astype(str)),
            "linkage_method": linkage_method,
            "name_metadata_match": name == row["raw_file_name"],
            "line_title_match": title.startswith(f"{row['line']}_"),
            "pass": (
                name == row["raw_file_name"]
                and len(geo_matches) >= 1
                and (
                    title.startswith(f"{row['line']}_")
                    or linkage_method == "cell_line_and_treatment"
                )
            ),
        })
    linkage = pd.DataFrame(linkage_rows)
    linkage.to_csv(args.output / "GSE207049_LINKAGE_SPOTCHECK.csv", index=False)

    counts = samples.groupby("treatment").size().to_dict()
    paired = pairing.loc[pairing["paired"]].copy()
    pairing_classes = pairing["pairing_class"].value_counts().to_dict()
    matrix_order_matches_metadata = matrix_samples == metadata["raw_file_name"].tolist()
    matrix_order_matches_filtered = matrix_samples == filtered["name"].tolist()
    gene_order_matches = matrix_genes == gene_ids
    checks = {
        "gzip_stream_readable": True,
        "matrix_rectangular": matrix["field_counts"] == [matrix["matrix_total_columns"]],
        "matrix_first_header_is_GeneID": matrix["first_header"] == "GeneID",
        "matrix_sample_names_unique": len(set(matrix_samples)) == len(matrix_samples),
        "matrix_metadata_sample_count_match": len(matrix_samples) == len(metadata),
        "matrix_metadata_name_order_exact": matrix_order_matches_metadata,
        "matrix_filtered_info_name_order_exact": matrix_order_matches_filtered,
        "filtered_id_range_exact": ids == list(range(1, len(matrix_samples) + 1)),
        "filtered_id_to_sample_name_exact": all(id_mapping_matches),
        "matrix_gene_count_matches_companion": len(matrix_genes) == len(gene_ids),
        "matrix_gene_order_matches_companion": gene_order_matches,
        "uncompressed_md5_matches_Zenodo": matrix["uncompressed_md5"] == "fc2e3e6d082ab6431155830c9a7767a7",
        "twenty_sample_GEO_linkage_pass": bool(linkage["pass"].all()),
        "paired_LCL_count_plausible": 300 <= len(paired) <= 400,
    }
    verdict = "DATA_INTEGRITY_PASS" if all(checks.values()) else "DATA_INTEGRITY_FAIL"

    audit = {
        "experiment": "LEVEL2-LEA-RNA-CF-0A",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit_at_audit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": "GEO GSE207049; Zenodo DOI 10.5281/zenodo.6595427 checksum; AmandaJLea/LCLs_gene_exp metadata",
        "filename": args.matrix.name,
        "compressed_size_bytes": args.matrix.stat().st_size,
        "compressed_sha256": file_hash(args.matrix, "sha256"),
        **matrix,
        "metadata_rows": len(metadata),
        "metadata_columns": len(metadata.columns),
        "filtered_info_rows": len(filtered),
        "filtered_info_columns": len(filtered.columns),
        "gene_id_rows": len(gene_ids),
        "duplicate_gene_id_rows_beyond_first": len(gene_ids) - len(set(gene_ids)),
        "unique_lines_all_processed": int(metadata["line"].nunique()),
        "ETOH_samples": int(counts.get("ETOH", 0)),
        "DEX_samples": int(counts.get("DEX", 0)),
        "paired_ETOH_DEX_LCLs": int(len(paired)),
        "paired_ancestry": paired["pop2"].value_counts().to_dict(),
        "pairing_classes": pairing_classes,
        "duplicate_handling": "arithmetic mean of all retained samples within biological line and treatment; all versions remain in one outer fold",
        "checks": checks,
        "verdict": verdict,
    }
    (args.output / "GSE207049_DATA_INTEGRITY.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    failed = [name for name, passed in checks.items() if not passed]
    report = f"""# GSE207049 Data Integrity Report

## Verdict

`{verdict}`

## Provenance and file integrity

- Source: GEO `GSE207049`; Zenodo DOI `10.5281/zenodo.6595427`; official author repository `AmandaJLea/LCLs_gene_exp`.
- File: `{args.matrix.name}`.
- Downloaded compressed size: {args.matrix.stat().st_size:,} bytes ({args.matrix.stat().st_size / 1024**2:.1f} MiB).
- Compressed SHA-256: `{audit['compressed_sha256']}`.
- Streamed uncompressed MD5: `{matrix['uncompressed_md5']}`; exact Zenodo match: **{checks['uncompressed_md5_matches_Zenodo']}**.
- No FASTQ, BAM, complete SRA, genotype VCF/archive, or additional response-eQTL matrix was downloaded for this pilot.

## Dimensions and identity

- Expression: {matrix['matrix_gene_rows']:,} gene rows x {matrix['matrix_sample_columns']:,} sample columns; header field `GeneID` plus samples.
- Processed metadata: {len(metadata):,} rows x {len(metadata.columns)} columns.
- Filtered sample info: {len(filtered):,} records x {len(filtered.columns)} columns.
- Gene-ID companion: {len(gene_ids):,} rows; exact row-order match: **{gene_order_matches}**.
- Duplicate Ensembl rows beyond first occurrence: {audit['duplicate_gene_id_rows_beyond_first']:,}; original row structure is retained.
- Matrix sample names are unique and exactly equal to both metadata files in order: **{matrix_order_matches_metadata and matrix_order_matches_filtered}**.
- Author `id` range is 1...{max(ids):,}; it maps exactly to the one-based sample index after excluding the leading `GeneID` column. Name-level linkage is used in this project.
- Deterministic 20-sample matrix -> metadata -> GEO spot check: **{checks['twenty_sample_GEO_linkage_pass']}**; details in `GSE207049_LINKAGE_SPOTCHECK.csv`.

## Pairing audit

- All processed samples: {len(metadata):,}; unique LCL lines: {metadata['line'].nunique():,}.
- ETOH samples: {counts.get('ETOH', 0):,}; DEX samples: {counts.get('DEX', 0):,}.
- Exact paired ETOH/DEX biological LCLs: **{len(paired):,}**.
- Paired ancestry: AFR {int((paired['pop2'] == 'AFR').sum())}, EUR {int((paired['pop2'] == 'EUR').sum())}.
- One ETOH / one DEX: {pairing_classes.get('one_to_one', 0)}.
- Two ETOH / one DEX: {pairing_classes.get('two_ETOH_one_DEX', 0)}.
- One ETOH / two DEX: {pairing_classes.get('one_ETOH_two_DEX', 0)}.
- Two ETOH / two DEX: {pairing_classes.get('two_to_two', 0)}.
- ETOH-only: {pairing_classes.get('ETOH_only', 0)}; DEX-only: {pairing_classes.get('DEX_only', 0)}.
- Neither ETOH nor DEX: {pairing_classes.get('neither', 0)}.
- Duplicate rule is outcome-blind arithmetic averaging within `line x treatment`; all versions of a line remain in the same CV fold.

## SVA / residualization warning

`voom_resid` is **not unresidualized baseline RNA**. It is the authors' filtered,
limma/voom-normalized expression matrix after regression of three surrogate
variables estimated on the complete 12-environment dataset. The amended pilot
protocol explicitly permits it as the primary pilot representation, but every
result must carry this caveat. A positive Level-2 result requires a later
pre-SVA/less-residualized sensitivity analysis before strengthening the claim.

## Gate details

- Passed checks: {sum(checks.values())}/{len(checks)}.
- Failed checks: {', '.join(failed) if failed else 'none'}.

`{verdict}`
"""
    (args.output / "GSE207049_DATA_INTEGRITY_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
