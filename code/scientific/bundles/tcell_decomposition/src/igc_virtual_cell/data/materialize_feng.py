"""Materialize leakage-neutral Feng NTC states and validate processed LFC keys."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np
import pandas as pd

from igc_virtual_cell.data.ingest_feng import (
    COUNT_ARTICLE_ID,
    FILE_DECISIONS,
    LFC_ARTICLE_ID,
    SCREEN_FILES,
)


COUNT_FILES = {
    "GenomeWideScreen_FitnessGenes": "GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz",
    "GenomeWideScreen_NonFitnessGenes": "GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz",
    "TargetedScreen": "TargetedScreen_RNA-UMI-Counts.csv.gz",
}

LFC_FILES = {
    "genomewide_pooled": "GenomeWideScreen_LFC_byGene.tsv.gz",
    "targeted_pooled": "TargetedScreen_LFC_byGene.tsv.gz",
    "targeted_per_line": "TargetedScreen_LFC_byGene-perLine.tsv.gz",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selected_files(raw_dir: Path) -> pd.DataFrame:
    """Verify every selected/downloaded Figshare file against size and MD5."""
    records = []
    for article_id in (COUNT_ARTICLE_ID, LFC_ARTICLE_ID):
        with urlopen(f"https://api.figshare.com/v2/articles/{article_id}") as response:
            article = json.load(response)
        for file in article["files"]:
            name = str(file["name"])
            decision = FILE_DECISIONS.get(name, ("not_selected", ""))[0]
            if decision not in {"selected", "downloaded"}:
                continue
            path = raw_dir / name
            present = path.exists()
            local_md5 = _md5(path) if present and path.stat().st_size == int(file["size"]) else ""
            records.append(
                {
                    "name": name,
                    "size_bytes": int(file["size"]),
                    "local_present": present,
                    "size_matches": present and path.stat().st_size == int(file["size"]),
                    "expected_md5": file["computed_md5"],
                    "local_md5": local_md5,
                    "md5_matches": bool(local_md5 and local_md5 == file["computed_md5"]),
                }
            )
    result = pd.DataFrame(records)
    if result.empty or not result["md5_matches"].all():
        failed = result.loc[~result["md5_matches"], "name"].tolist() if not result.empty else []
        raise ValueError(f"Selected Feng files failed verification: {failed}")
    return result


def _normalized_cell_ids(metadata: pd.DataFrame, partition: str) -> pd.Series:
    ids = metadata["Cell_ID"].astype(str)
    if partition.startswith("GenomeWideScreen_"):
        return ids.str.removeprefix("MP-")

    def normalize_targeted(value: str) -> str:
        match = re.match(r"^PC-(P[0-9]+)-D[0-9]+_(I[0-9]+)_(.*)$", value)
        if not match:
            raise ValueError(f"Unexpected targeted Cell_ID format: {value}")
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"

    return ids.map(normalize_targeted)


def _lfc_gene_symbols(path: Path) -> list[str]:
    """Read the first complete target block, which defines the released gene universe."""
    sample = pd.read_csv(
        path,
        sep="\t",
        usecols=["Target", "Expressed_Gene_Symbol"],
        nrows=10_000,
    )
    first_target = sample["Target"].iloc[0]
    genes = sample.loc[sample["Target"].eq(first_target), "Expressed_Gene_Symbol"].astype(str)
    if genes.duplicated().any():
        raise ValueError(f"Duplicate expressed gene symbols in first target block of {path.name}")
    return genes.tolist()


def _line_map(project_root: Path, screen: str) -> dict[str, tuple[str, str]]:
    contexts = pd.read_csv(project_root / "results" / "tables" / "feng_2026_context_identifiers.csv")
    subset = contexts.loc[contexts["screen"].eq(screen)]
    return {
        str(row.metadata_cell_line): (str(row.context_id), str(row.hipsci_cell_line))
        for row in subset.itertuples(index=False)
    }


def _aggregate_count_file(
    *,
    count_path: Path,
    metadata_path: Path,
    partition: str,
    context_lookup: dict[str, tuple[str, str]],
    selected_genes: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = pd.read_csv(metadata_path, sep="\t")
    metadata["count_cell_id"] = _normalized_cell_ids(metadata, partition)
    if metadata["count_cell_id"].duplicated().any():
        raise ValueError(f"Normalized metadata Cell_ID is not unique for {partition}")
    metadata = metadata.set_index("count_cell_id")

    with gzip.open(count_path, "rb") as handle:
        header = handle.readline().rstrip(b"\r\n").decode("utf-8").split(",")[1:]
        if len(header) != len(set(header)):
            raise ValueError(f"Count matrix header has duplicate cells: {count_path.name}")
        aligned = metadata.reindex(header)
        matched = aligned["Cell_ID"].notna()
        guide = aligned["Guide_Call"].fillna("").astype(str)
        control = matched & (guide.eq("unassigned") | guide.str.startswith("NonTarget_"))
        context_ids = [value[0] for value in context_lookup.values()]
        context_index = {context_id: index for index, context_id in enumerate(context_ids)}
        short_to_code = {
            short: context_index[context_id]
            for short, (context_id, _) in context_lookup.items()
        }
        codes = aligned["Cell_Line"].map(short_to_code).fillna(-1).astype(int).to_numpy()
        selected_mask = control.to_numpy()
        selected_codes = codes[selected_mask]
        gene_set = set(selected_genes)
        aggregates: dict[str, np.ndarray] = {}
        gene_rows = 0
        for line in handle:
            gene_rows += 1
            comma = line.find(b",")
            if comma < 0:
                raise ValueError(f"Malformed count row {gene_rows} in {count_path.name}")
            label = line[:comma].decode("utf-8")
            parts = label.split(":")
            symbol = parts[1] if len(parts) > 1 else parts[0]
            if symbol not in gene_set:
                continue
            values = np.fromstring(line[comma + 1 :], dtype=np.int64, sep=",")
            if len(values) != len(header):
                raise ValueError(
                    f"Count row length mismatch in {count_path.name} for {label}: "
                    f"{len(values)} != {len(header)}"
                )
            sums = np.bincount(
                selected_codes,
                weights=values[selected_mask],
                minlength=len(context_ids),
            )
            aggregates[symbol] = aggregates.get(symbol, np.zeros(len(context_ids))) + sums

    missing_genes = sorted(gene_set - set(aggregates))
    matrix = pd.DataFrame(
        {
            gene: aggregates.get(gene, np.zeros(len(context_ids)))
            for gene in selected_genes
        },
        index=context_ids,
    )
    matrix.index.name = "context_id"
    qc = {
        "partition": partition,
        "count_file": count_path.name,
        "matrix_cell_count": len(header),
        "metadata_cell_count": len(metadata),
        "matched_cell_count": int(matched.sum()),
        "unmatched_matrix_cell_count": int((~matched).sum()),
        "control_cell_count": int(control.sum()),
        "gene_rows_in_count_matrix": gene_rows,
        "selected_gene_count": len(selected_genes),
        "missing_selected_gene_count": len(missing_genes),
        "missing_selected_genes": "|".join(missing_genes),
    }
    return matrix, qc


def materialize_ntc(raw_dir: Path, project_root: Path) -> pd.DataFrame:
    output_dir = project_root / "data" / "processed" / "feng_2026"
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = []
    for screen in ("genomewide", "targeted"):
        lfc_name = (
            LFC_FILES["genomewide_pooled"]
            if screen == "genomewide"
            else LFC_FILES["targeted_pooled"]
        )
        genes = _lfc_gene_symbols(raw_dir / lfc_name)
        context_lookup = _line_map(project_root, screen)
        raw_path = output_dir / f"feng_2026_{screen}_ntc_raw_counts.parquet"
        normalized_path = output_dir / f"feng_2026_{screen}_ntc_log1p_cpm.parquet"
        partitions = (
            ("GenomeWideScreen_FitnessGenes", "GenomeWideScreen_FitnessGenes_Cell-Metadata.tsv.gz"),
            ("GenomeWideScreen_NonFitnessGenes", "GenomeWideScreen_NonFitnessGenes_Cell-Metadata.tsv.gz"),
        ) if screen == "genomewide" else (
            ("TargetedScreen", "TargetedScreen_Cell-Metadata.tsv.gz"),
        )
        manifest_path = project_root / "data" / "manifests" / f"feng_2026_{screen}.csv"
        manifest = pd.read_csv(manifest_path, low_memory=False)
        control_mask = manifest["is_control"].astype(str).str.lower().eq("true")
        controls = manifest.loc[control_mask, ["context_id", "observation_id"]]
        observation_lookup = controls.set_index("context_id")["observation_id"]
        if raw_path.exists() and normalized_path.exists():
            qc_rows.append(
                {
                    "screen": screen,
                    "partition": "cached_complete_screen",
                    "count_file": "|".join(COUNT_FILES[partition] for partition, _ in partitions),
                    "selected_gene_count": len(genes),
                    "cache_reused": True,
                }
            )
        else:
            combined = pd.DataFrame(
                0.0, index=[value[0] for value in context_lookup.values()], columns=genes
            )
            for partition, metadata_name in partitions:
                partial, qc = _aggregate_count_file(
                    count_path=raw_dir / COUNT_FILES[partition],
                    metadata_path=raw_dir / metadata_name,
                    partition=partition,
                    context_lookup=context_lookup,
                    selected_genes=genes,
                )
                combined = combined.add(partial, fill_value=0)
                qc_rows.append({"screen": screen, **qc, "cache_reused": False})

            library_size = combined.sum(axis=1)
            if (library_size <= 0).any():
                raise ValueError(
                    f"Zero NTC library size in {screen}: "
                    f"{library_size[library_size <= 0].index.tolist()}"
                )
            log1p_cpm = np.log1p(combined.div(library_size, axis=0) * 1_000_000)
            combined.insert(0, "observation_id", combined.index.map(observation_lookup))
            log1p_cpm.insert(0, "observation_id", log1p_cpm.index.map(observation_lookup))
            combined.reset_index(drop=True).to_parquet(raw_path, index=False)
            log1p_cpm.reset_index(drop=True).to_parquet(normalized_path, index=False)

            for qc in qc_rows:
                if qc["screen"] == screen:
                    qc["ntc_library_size_min"] = int(library_size.min())
                    qc["ntc_library_size_max"] = int(library_size.max())

        manifest["expression_path"] = manifest["expression_path"].astype(object)
        manifest["data_format"] = manifest["data_format"].astype(object)
        manifest["expression_status"] = manifest["expression_status"].astype(object)
        manifest["normalization"] = manifest["normalization"].astype(object)
        manifest.loc[control_mask, "expression_path"] = (
            f"../processed/feng_2026/{normalized_path.name}"
        )
        manifest.loc[control_mask, "data_format"] = "parquet"
        manifest.loc[control_mask, "expression_status"] = "ntc_log1p_cpm_materialized"
        manifest.loc[control_mask, "normalization"] = "log1p_CPM_from_raw_UMI_counts"
        manifest.to_csv(manifest_path, index=False)
    return pd.DataFrame(qc_rows)


def validate_lfc_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    target_line_counts: Counter[tuple[str, str]] = Counter()
    for representation, name in LFC_FILES.items():
        path = raw_dir / name
        rows = 0
        targets: set[str] = set()
        genes: set[str] = set()
        lines: set[str] = set()
        usecols = ["Target", "Expressed_Gene_Symbol"]
        if representation == "targeted_per_line":
            usecols.append("Cell_Line")
        for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=1_000_000):
            rows += len(chunk)
            targets.update(chunk["Target"].dropna().astype(str).unique())
            genes.update(chunk["Expressed_Gene_Symbol"].dropna().astype(str).unique())
            if "Cell_Line" in chunk:
                lines.update(chunk["Cell_Line"].dropna().astype(str).unique())
                grouped = chunk.groupby(["Target", "Cell_Line"], sort=False).size()
                target_line_counts.update(
                    {(str(target), str(line)): int(count) for (target, line), count in grouped.items()}
                )
        summaries.append(
            {
                "representation": representation,
                "file": name,
                "row_count": rows,
                "target_count": len(targets),
                "expressed_gene_symbol_count": len(genes),
                "cell_line_count": len(lines),
            }
        )
    detail = pd.DataFrame(
        [
            {"perturbation_id": target, "metadata_cell_line": line, "lfc_gene_rows": count}
            for (target, line), count in sorted(target_line_counts.items())
        ]
    )
    return pd.DataFrame(summaries), detail


def run(raw_dir: Path, project_root: Path, *, skip_verification: bool = False) -> None:
    raw_dir = raw_dir.resolve()
    project_root = project_root.resolve()
    tables = project_root / "results" / "tables"
    manifests = project_root / "results" / "manifests"
    tables.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    verification_path = manifests / "feng_2026_download_verification.csv"
    if skip_verification:
        if not verification_path.exists():
            raise FileNotFoundError(
                "--skip-verification requires an existing successful verification report"
            )
        verification = pd.read_csv(verification_path)
        if not verification["md5_matches"].astype(bool).all():
            raise ValueError("Existing verification report contains failed files")
    else:
        verification = verify_selected_files(raw_dir)
        verification.to_csv(verification_path, index=False)
    ntc_qc = materialize_ntc(raw_dir, project_root)
    ntc_qc.to_csv(tables / "feng_2026_ntc_materialization_qc.csv", index=False)
    lfc_summary, lfc_detail = validate_lfc_tables(raw_dir)
    lfc_summary.to_csv(tables / "feng_2026_lfc_validation_summary.csv", index=False)
    lfc_detail.to_csv(tables / "feng_2026_targeted_lfc_target_line_counts.csv", index=False)
    print("Feng processed files verified; NTC states materialized; LFC keys audited.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/feng_2026"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--skip-verification", action="store_true")
    args = parser.parse_args()
    run(args.raw_dir, args.project_root, skip_verification=args.skip_verification)


if __name__ == "__main__":
    main()
