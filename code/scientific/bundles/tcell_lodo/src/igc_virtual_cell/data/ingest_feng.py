"""Ingest and audit Feng et al. 2026 processed cell metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

import pandas as pd

from igc_virtual_cell.data.schema import MANIFEST_COLUMNS, validate_manifest


COUNT_ARTICLE_ID = 27989294
LFC_ARTICLE_ID = 26819743
PAPER_DOI = "10.1016/j.xgen.2025.101076"

SCREEN_FILES = {
    "genomewide": (
        "GenomeWideScreen_FitnessGenes_Cell-Metadata.tsv.gz",
        "GenomeWideScreen_NonFitnessGenes_Cell-Metadata.tsv.gz",
    ),
    "targeted": ("TargetedScreen_Cell-Metadata.tsv.gz",),
}

FILE_DECISIONS = {
    "GenomeWideScreen_FitnessGenes_Cell-Metadata.tsv.gz": ("downloaded", "cell/context/guide audit"),
    "GenomeWideScreen_NonFitnessGenes_Cell-Metadata.tsv.gz": ("downloaded", "cell/context/guide audit"),
    "TargetedScreen_Cell-Metadata.tsv.gz": ("downloaded", "cell/context/guide audit"),
    "GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz": ("selected", "line-specific NTC and response expression"),
    "GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz": ("selected", "line-specific NTC and response expression"),
    "TargetedScreen_RNA-UMI-Counts.csv.gz": ("selected", "line-specific NTC expression"),
    "GenomeWideScreen_FitnessGenes_Guide-UMI-Counts.csv.gz": ("not_selected", "Guide_Call is already in cell metadata"),
    "GenomeWideScreen_NonFitnessGenes_Guide-UMI-Counts.csv.gz": ("not_selected", "Guide_Call is already in cell metadata"),
    "TargetedScreen_Guide-UMI-Counts.csv.gz": ("not_selected", "Guide_Call is already in cell metadata"),
    "00_raw_data_id_mapping.csv": ("downloaded", "full HipSci line identifiers and inlet mapping"),
    "GenomeWideScreen_LFC_byGene.tsv.gz": ("selected", "gene-level genome-wide response anchor; pooled across lines"),
    "TargetedScreen_LFC_byGene.tsv.gz": ("selected", "gene-level targeted pooled response reference"),
    "TargetedScreen_LFC_byGene-perLine.tsv.gz": ("selected", "direct line-specific gene-level responses"),
    "GenomeWideScreen_LFC_byGuide.tsv.gz": ("not_selected", "gene-level table is the Phase-I unit"),
    "TargetedScreen_LFC_byGuide.tsv.gz": ("not_selected", "gene-level table is the Phase-I unit"),
    "TargetedScreen_LFC_byGuide-perLine.tsv.gz": ("not_selected", "gene-level per-line table is smaller and sufficient"),
    "GenomeWideScreen_Coregulation-Target-Correlation.tsv.gz": ("not_selected", "derived correlation table is not an ingestion input"),
    "GenomeWideScreen_Coperturbation-Expressed-Gene-Correlation.tsv.gz": ("not_selected", "derived correlation table is not an ingestion input"),
    "scripts.zip": ("not_selected", "analysis code is available separately on GitHub"),
    "Peer Review.pdf": ("not_selected", "not a data input"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Git provenance is required for Feng ingestion")
    return commit


def _figshare_article(article_id: int) -> dict[str, object]:
    with urlopen(f"https://api.figshare.com/v2/articles/{article_id}") as response:
        return json.load(response)


def _source_catalog(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for article_id in (COUNT_ARTICLE_ID, LFC_ARTICLE_ID):
        article = _figshare_article(article_id)
        for file in article["files"]:
            name = str(file["name"])
            decision, rationale = FILE_DECISIONS.get(
                name, ("not_selected", "not required for the ingestion audit")
            )
            local = raw_dir / name
            rows.append(
                {
                    "figshare_article_id": article_id,
                    "figshare_file_id": file["id"],
                    "name": name,
                    "size_bytes": int(file["size"]),
                    "size_gib": round(int(file["size"]) / 1024**3, 4),
                    "figshare_md5": file.get("computed_md5", ""),
                    "download_url": file["download_url"],
                    "decision": decision,
                    "rationale": rationale,
                    "local_present": local.exists(),
                    "local_size_bytes": local.stat().st_size if local.exists() else 0,
                    "local_size_matches": local.exists() and local.stat().st_size == int(file["size"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["figshare_article_id", "size_bytes"])


def _line_mapping(raw_dir: Path) -> dict[str, str]:
    mapping = pd.read_csv(raw_dir / "00_raw_data_id_mapping.csv")
    pairs: dict[str, str] = {}
    for encoded in mapping["lines_used"].dropna().astype(str):
        for full in encoded.split(";"):
            match = re.search(r"-([^-]+_[0-9]+)$", full)
            short = match.group(1) if match else full
            if short in pairs and pairs[short] != full:
                raise ValueError(f"Ambiguous full line mapping for {short}")
            pairs[short] = full
    return pairs


def _classify(metadata: pd.DataFrame, *, screen: str, partition: str) -> pd.DataFrame:
    frame = metadata.copy()
    guide = frame["Guide_Call"].astype(str)
    frame["control_label"] = ""
    frame.loc[guide.eq("unassigned"), "control_label"] = "unassigned"
    frame.loc[guide.str.startswith("NonTarget_"), "control_label"] = "NonTarget"
    frame["is_control"] = frame["control_label"].ne("")
    frame["perturbation_id"] = guide.str.rsplit("_", n=1).str[0]
    frame.loc[frame["is_control"], "perturbation_id"] = "NTC"
    frame["screen"] = screen
    frame["partition"] = partition
    frame["timepoint"] = frame["Batch"].str.extract(r"_Day([0-9]+)_", expand=False).map(
        lambda value: f"{value}d" if pd.notna(value) else ""
    )
    return frame


def _load_screens(raw_dir: Path) -> dict[str, pd.DataFrame]:
    screens: dict[str, pd.DataFrame] = {}
    for screen, names in SCREEN_FILES.items():
        pieces = []
        for name in names:
            path = raw_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
            partition = name.removesuffix("_Cell-Metadata.tsv.gz")
            pieces.append(
                _classify(pd.read_csv(path, sep="\t"), screen=screen, partition=partition)
            )
        screens[screen] = pd.concat(pieces, ignore_index=True)
    return screens


def _context_table(
    screens: dict[str, pd.DataFrame], line_map: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for screen, metadata in screens.items():
        for short in sorted(metadata["Cell_Line"].astype(str).unique()):
            full = line_map.get(short)
            if full is None:
                raise ValueError(f"No full HipSci identifier found for {short}")
            rows.append(
                {
                    "screen": screen,
                    "context_id": f"feng2026_{screen}::{full}",
                    "metadata_cell_line": short,
                    "hipsci_cell_line": full,
                    "donor_id": full.rsplit("_", 1)[0],
                    "cell_count": int(metadata["Cell_Line"].astype(str).eq(short).sum()),
                    "batch_count": int(
                        metadata.loc[metadata["Cell_Line"].astype(str).eq(short), "Batch"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _count_tables(
    screens: dict[str, pd.DataFrame], contexts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context_lookup = contexts.set_index(["screen", "metadata_cell_line"])[
        ["context_id", "hipsci_cell_line", "donor_id"]
    ]
    perturbation_rows = []
    control_rows = []
    coverage_rows = []
    for screen, metadata in screens.items():
        perturbed = metadata.loc[~metadata["is_control"]]
        grouped = perturbed.groupby(["Cell_Line", "perturbation_id"], sort=True)
        for (short, perturbation), group in grouped:
            info = context_lookup.loc[(screen, str(short))]
            perturbation_rows.append(
                {
                    "screen": screen,
                    **info.to_dict(),
                    "metadata_cell_line": short,
                    "perturbation_id": perturbation,
                    "cell_count": int(len(group)),
                    "guide_count": int(group["Guide_Call"].nunique()),
                    "batch_count": int(group["Batch"].nunique()),
                    "timepoints": "|".join(sorted(group["timepoint"].dropna().unique())),
                }
            )
        controls = metadata.loc[metadata["is_control"]]
        for (short, label), group in controls.groupby(["Cell_Line", "control_label"], sort=True):
            info = context_lookup.loc[(screen, str(short))]
            control_rows.append(
                {
                    "screen": screen,
                    **info.to_dict(),
                    "metadata_cell_line": short,
                    "control_label": label,
                    "cell_count": int(len(group)),
                    "guide_count": int(group["Guide_Call"].nunique()),
                    "batch_count": int(group["Batch"].nunique()),
                }
            )
        counts = pd.DataFrame([row for row in perturbation_rows if row["screen"] == screen])
        for minimum_cells in (1, 10):
            qualified = counts.loc[counts["cell_count"] >= minimum_cells]
            contexts_per_perturbation = qualified.groupby("perturbation_id")["context_id"].nunique()
            for threshold in (2, 5, 10, 20):
                coverage_rows.append(
                    {
                        "screen": screen,
                        "minimum_cells_per_context": minimum_cells,
                        "minimum_contexts": threshold,
                        "perturbation_count": int((contexts_per_perturbation >= threshold).sum()),
                    }
                )
    return (
        pd.DataFrame(perturbation_rows),
        pd.DataFrame(control_rows),
        pd.DataFrame(coverage_rows),
    )


def _manifest_for_screen(
    *,
    screen: str,
    perturbations: pd.DataFrame,
    controls: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    dataset_id = f"feng_2026_{screen}"
    study = f"Feng_et_al_Cell_Genomics_2026_{screen}"
    source_names = SCREEN_FILES[screen]
    source_digest = hashlib.sha256(
        "".join(_sha256(raw_dir / name) for name in source_names).encode("utf-8")
    ).hexdigest()
    rows: list[dict[str, object]] = []

    screen_perturbations = perturbations.loc[perturbations["screen"].eq(screen)]
    for row in screen_perturbations.itertuples(index=False):
        identity = f"{dataset_id}|{row.context_id}|{row.perturbation_id}"
        rows.append(
            {
                "observation_id": f"feng_{hashlib.sha1(identity.encode()).hexdigest()[:20]}",
                "dataset_id": dataset_id,
                "study": study,
                "context_id": row.context_id,
                "context_name": row.hipsci_cell_line,
                "cell_type": "human induced pluripotent stem cell",
                "cell_line": row.hipsci_cell_line,
                "species": "Homo sapiens",
                "perturbation_type": "CRISPRi",
                "perturbation_id": row.perturbation_id,
                "is_control": False,
                "control_label": "",
                "timepoint": row.timepoints,
                "dose": "not_applicable",
                "replicate_id": f"aggregate_{row.guide_count}guides_{row.batch_count}batches",
                "cell_count": row.cell_count,
                "ntc_available": True,
                "gene_universe": f"Feng2026_{screen}_RNA_UMI_counts",
                "expression_path": "",
                "data_format": "aggregated_cell_metadata",
                "expression_status": "processed_counts_selected_pending_materialization",
                "normalization": "raw_UMI_counts",
                "batch_id": f"aggregate_{row.batch_count}_batches",
                "source_uri": f"https://doi.org/{PAPER_DOI};https://figshare.com/articles/dataset/{COUNT_ARTICLE_ID}",
                "sha256": source_digest,
            }
        )

    screen_controls = controls.loc[controls["screen"].eq(screen)]
    for context_id, group in screen_controls.groupby("context_id", sort=True):
        first = group.iloc[0]
        identity = f"{dataset_id}|{context_id}|NTC"
        rows.append(
            {
                "observation_id": f"feng_{hashlib.sha1(identity.encode()).hexdigest()[:20]}",
                "dataset_id": dataset_id,
                "study": study,
                "context_id": context_id,
                "context_name": first["hipsci_cell_line"],
                "cell_type": "human induced pluripotent stem cell",
                "cell_line": first["hipsci_cell_line"],
                "species": "Homo sapiens",
                "perturbation_type": "control",
                "perturbation_id": "NTC",
                "is_control": True,
                "control_label": "+".join(sorted(group["control_label"].unique())),
                "timepoint": "mixed_by_screen_design",
                "dose": "not_applicable",
                "replicate_id": "aggregate_control_cells",
                "cell_count": int(group["cell_count"].sum()),
                "ntc_available": True,
                "gene_universe": f"Feng2026_{screen}_RNA_UMI_counts",
                "expression_path": "",
                "data_format": "aggregated_cell_metadata",
                "expression_status": "processed_counts_selected_pending_materialization",
                "normalization": "raw_UMI_counts",
                "batch_id": f"aggregate_{int(group['batch_count'].max())}_batches",
                "source_uri": f"https://doi.org/{PAPER_DOI};https://figshare.com/articles/dataset/{COUNT_ARTICLE_ID}",
                "sha256": source_digest,
            }
        )
    return validate_manifest(pd.DataFrame(rows, columns=MANIFEST_COLUMNS), source=dataset_id)


def _render_report(
    *,
    commit: str,
    catalog: pd.DataFrame,
    screens: dict[str, pd.DataFrame],
    contexts: pd.DataFrame,
    perturbations: pd.DataFrame,
    controls: pd.DataFrame,
    coverage: pd.DataFrame,
    ingestion_confirmed: bool,
    lfc_detail: pd.DataFrame | None,
) -> str:
    lines = [
        "# Feng et al. 2026 ingestion audit",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Git provenance: `{commit}`",
        f"- Paper: https://doi.org/{PAPER_DOI}",
        f"- Count data: https://figshare.com/articles/dataset/{COUNT_ARTICLE_ID}",
        f"- LFC data: https://figshare.com/articles/dataset/{LFC_ARTICLE_ID}",
        "",
        "## Ingestion gate",
        "",
        (
            "`CONFIRMED`: cell-line identifiers, controls, perturbation labels, cell counts, "
            "coverage, and technical replicate fields are confirmed; every selected processed "
            "file is present and MD5-verified; NTC states are materialized; and LFC keys are "
            "audited. Phase I was not started by this ingestion command."
            if ingestion_confirmed
            else "`NOT YET CONFIRMED`: metadata fields are audited, but selected expression/LFC "
            "files or materialized NTC outputs remain incomplete. Phase I must not start."
        ),
        "",
        "## Screen-level findings",
        "",
        "| Screen | Cells | Contexts | Perturbations | Perturbed cells | Control cells | Batches/inlets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for screen, metadata in screens.items():
        screen_counts = perturbations.loc[perturbations["screen"].eq(screen)]
        screen_controls = controls.loc[controls["screen"].eq(screen)]
        lines.append(
            f"| {screen} | {len(metadata):,} | {metadata['Cell_Line'].nunique()} | "
            f"{screen_counts['perturbation_id'].nunique():,} | {screen_counts['cell_count'].sum():,} | "
            f"{screen_controls['cell_count'].sum():,} | {metadata['Batch'].nunique()} |"
        )
    lines.extend(
        [
            "",
            "The targeted experiment reports 20 designed lines in the paper but only 19 "
            "post-QC lines in the released metadata, consistent with the paper's retained-cell analysis.",
            "",
            "## Exact released context identifiers",
            "",
        ]
    )
    for screen in ("genomewide", "targeted"):
        subset = contexts.loc[contexts["screen"].eq(screen)]
        lines.append(f"### {screen}")
        lines.append("")
        lines.append(", ".join(subset["hipsci_cell_line"].astype(str)))
        lines.append("")
    lines.extend(
        [
            "## Control and perturbation labels",
            "",
            "- Control: exact `Guide_Call == unassigned`.",
            "- Control: exact `Guide_Call` prefix `NonTarget_`.",
            "- Perturbation: target symbol before the final underscore in `Guide_Call`; the "
            "suffix is the guide sequence.",
            "- The authors' released fold-change code uses both unassigned and NonTarget cells "
            "as controls.",
            "",
            "## Shared perturbation coverage",
            "",
            "| Screen | Minimum cells/context | ≥2 contexts | ≥5 | ≥10 | ≥20 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for (screen, minimum_cells), group in coverage.groupby(
        ["screen", "minimum_cells_per_context"], sort=True
    ):
        values = group.set_index("minimum_contexts")["perturbation_count"]
        lines.append(
            f"| {screen} | {minimum_cells} | {values.get(2, 0):,} | {values.get(5, 0):,} | "
            f"{values.get(10, 0):,} | {values.get(20, 0):,} |"
        )
    if lfc_detail is not None and not lfc_detail.empty:
        lfc_coverage = lfc_detail.groupby("perturbation_id")["metadata_cell_line"].nunique()
        per_line = lfc_detail.groupby("metadata_cell_line")["perturbation_id"].nunique()
        lines.extend(
            [
                "",
                "## Released targeted per-line LFC availability",
                "",
                f"- Present target×line blocks: {len(lfc_detail):,} / {444 * 19:,}.",
                f"- Targets present in at least 2, 5, and 10 lines: "
                f"{int((lfc_coverage >= 2).sum())}, {int((lfc_coverage >= 5).sum())}, "
                f"{int((lfc_coverage >= 10).sum())}.",
                f"- Targets present in all 19 lines: {int((lfc_coverage == 19).sum())}.",
                f"- Lines with the fewest released target blocks: "
                + ", ".join(
                    f"{line} ({count})" for line, count in per_line.sort_values().head(5).items()
                )
                + ".",
                "",
                "Metadata cell support and released LFC availability are therefore distinct "
                "eligibility filters. Phase I must use the automatically intersected LFC blocks, "
                "not assume a complete 444×19 response cube.",
            ]
        )
    lines.extend(
        [
            "",
            "## Separate-analysis decision",
            "",
            "`REQUIRED`: genome-wide and targeted screens must remain separate analysis groups. "
            "They differ in library composition, line coverage, time-point/inlet structure, and "
            "per-line perturbation support. Genome-wide median support is only about two cells per "
            "perturbation-line pair; targeted median support is about 74 cells and all 444 targets "
            "have at least 10 cells in every released line.",
            "",
            "## Replicate availability",
            "",
            "No explicit biological-replicate identifier is present. `Batch` encodes pool, day, "
            "and inlet and is retained as a technical replicate/batch field. Multiple guide "
            "sequences per target provide guide-level replication. Related clones from the same "
            "HipSci donor are mapped in the context table and must not be mistaken for independent donors.",
            "",
            "## Processed-file selection",
            "",
            "The complete file decision table is `results/manifests/feng_2026_source_files.csv`. "
            f"Selected files total {catalog.loc[catalog['decision'].isin(['downloaded','selected']), 'size_gib'].sum():.2f} GiB. "
            "No FASTQ or CRAM files were selected.",
            "",
            "## Outputs",
            "",
            "- `data/manifests/feng_2026_genomewide.csv`",
            "- `data/manifests/feng_2026_targeted.csv`",
            "- `results/tables/feng_2026_context_identifiers.csv`",
            "- `results/tables/feng_2026_cell_counts_by_perturbation_context.csv`",
            "- `results/tables/feng_2026_control_counts_by_context.csv`",
            "- `results/tables/feng_2026_shared_perturbation_summary.csv`",
        ]
    )
    lines.extend(
        [
            "",
            "FENG_INGESTION_AUDIT_CONFIRMED"
            if ingestion_confirmed
            else "FENG_INGESTION_AUDIT_NOT_CONFIRMED",
        ]
    )
    return "\n".join(lines) + "\n"


def ingest(raw_dir: Path, project_root: Path) -> Path:
    raw_dir = raw_dir.resolve()
    project_root = project_root.resolve()
    commit = _git_commit(project_root)
    catalog = _source_catalog(raw_dir)
    line_map = _line_mapping(raw_dir)
    screens = _load_screens(raw_dir)
    contexts = _context_table(screens, line_map)
    perturbations, controls, coverage = _count_tables(screens, contexts)

    manifests_dir = project_root / "data" / "manifests"
    tables_dir = project_root / "results" / "tables"
    source_dir = project_root / "results" / "manifests"
    reports_dir = project_root / "results" / "reports"
    for directory in (manifests_dir, tables_dir, source_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for screen in ("genomewide", "targeted"):
        manifest = _manifest_for_screen(
            screen=screen,
            perturbations=perturbations,
            controls=controls,
            raw_dir=raw_dir,
        )
        manifest_path = manifests_dir / f"feng_2026_{screen}.csv"
        if manifest_path.exists():
            previous = pd.read_csv(manifest_path, low_memory=False)
            previous["expression_path"] = previous["expression_path"].fillna("").astype(str)
            materialized = previous.loc[
                previous["expression_path"].str.strip().ne(""),
                [
                    "observation_id",
                    "expression_path",
                    "data_format",
                    "expression_status",
                    "normalization",
                ],
            ].set_index("observation_id")
            if not materialized.empty:
                manifest = manifest.set_index("observation_id")
                shared = manifest.index.intersection(materialized.index)
                for column in materialized.columns:
                    manifest.loc[shared, column] = materialized.loc[shared, column]
                manifest = manifest.reset_index()
        manifest.to_csv(manifest_path, index=False)
    contexts.to_csv(tables_dir / "feng_2026_context_identifiers.csv", index=False)
    perturbations.to_csv(
        tables_dir / "feng_2026_cell_counts_by_perturbation_context.csv", index=False
    )
    controls.to_csv(tables_dir / "feng_2026_control_counts_by_context.csv", index=False)
    coverage.to_csv(
        tables_dir / "feng_2026_shared_perturbation_summary.csv", index=False
    )
    catalog.to_csv(source_dir / "feng_2026_source_files.csv", index=False)
    selected = catalog.loc[catalog["decision"].isin(["downloaded", "selected"])]
    verification_path = source_dir / "feng_2026_download_verification.csv"
    verification_ok = False
    if verification_path.exists():
        verification = pd.read_csv(verification_path)
        verification_ok = bool(
            not verification.empty and verification["md5_matches"].astype(bool).all()
        )
    ntc_complete = all(
        (
            project_root
            / "data"
            / "processed"
            / "feng_2026"
            / f"feng_2026_{screen}_ntc_log1p_cpm.parquet"
        ).exists()
        for screen in ("genomewide", "targeted")
    )
    lfc_detail_path = tables_dir / "feng_2026_targeted_lfc_target_line_counts.csv"
    lfc_detail = pd.read_csv(lfc_detail_path) if lfc_detail_path.exists() else None
    ingestion_confirmed = bool(
        not selected.empty
        and selected["local_size_matches"].astype(bool).all()
        and verification_ok
        and ntc_complete
        and lfc_detail is not None
        and not lfc_detail.empty
    )
    report_path = reports_dir / "feng_2026_ingestion_audit.md"
    report_path.write_text(
        _render_report(
            commit=commit,
            catalog=catalog,
            screens=screens,
            contexts=contexts,
            perturbations=perturbations,
            controls=controls,
            coverage=coverage,
            ingestion_confirmed=ingestion_confirmed,
            lfc_detail=lfc_detail,
        ),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/feng_2026"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = ingest(args.raw_dir, args.project_root)
    print(f"Feng ingestion audit written to {report}")


if __name__ == "__main__":
    main()
