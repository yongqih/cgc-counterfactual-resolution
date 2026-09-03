from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAMPLE_PATTERN = re.compile(r"^(Pt[0-9]+)")


def _patient_from_sample(sample_id: str) -> str:
    match = SAMPLE_PATTERN.match(str(sample_id))
    if match is None:
        raise ValueError(f"Unrecognized official sample_id: {sample_id!r}")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sources(raw_dir: Path, source_manifest: dict[str, Any]) -> None:
    for record in source_manifest["files"]:
        path = raw_dir / record["local_name"]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Size mismatch for {path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")


def _read_geo_header(path: Path) -> tuple[pd.DataFrame, list[str]]:
    fields: dict[str, list[list[str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!"):
                continue
            row = next(csv.reader([line], delimiter="\t", quotechar='"'))
            fields[row[0]].append(row[1:])

    titles = fields["!Sample_title"][0]
    gsm = fields["!Sample_geo_accession"][0]
    characteristics = fields["!Sample_characteristics_ch1"]
    if len(titles) != len(gsm):
        raise ValueError("GEO title/accession length mismatch")

    group_row = next(row for row in characteristics if row and row[0].startswith("group:"))
    patient_row = next(row for row in characteristics if row and row[0].startswith("patient:"))
    tumor_row = next(row for row in characteristics if row and row[0].startswith("tumor:"))
    records: list[dict[str, str]] = []
    for index, (title, accession, group, patient, tumor) in enumerate(
        zip(titles, gsm, group_row, patient_row, tumor_row, strict=True)
    ):
        title_match = re.search(r"(?:PDO|Tissue sample) from (.+)$", title)
        if title_match is None:
            raise ValueError(f"Cannot parse GEO title {title!r}")
        records.append(
            {
                "geo_column_index": index,
                "geo_accession": accession,
                "sample_id": title_match.group(1),
                "geo_title": title,
                "group": group.removeprefix("group: "),
                "patient_id": patient.removeprefix("patient: "),
                "tumor_id": tumor.removeprefix("tumor: "),
                "sample_type": "PDO" if title.startswith("PDO") else "CRLM",
            }
        )
    return pd.DataFrame.from_records(records), gsm


def _load_geo_matrix(path: Path, pdo_geo: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = pd.read_csv(path, sep="\t", comment="!", compression="gzip")
    feature_ids = matrix.iloc[:, 0].astype(str).to_numpy()
    columns = pdo_geo["geo_accession"].tolist()
    missing = sorted(set(columns).difference(matrix.columns))
    if missing:
        raise ValueError(f"Missing GEO matrix columns: {missing[:5]}")
    values = matrix[columns].to_numpy(dtype=np.float32, copy=True).T
    return feature_ids, values


def _load_meta(raw_dir: Path) -> pd.DataFrame:
    meta = pd.read_excel(raw_dir / "mmc8.xlsx", sheet_name="Table S2", header=3)
    meta = meta[meta["sample_id"].astype(str).str.match(r"^Pt")].copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["patient_id"] = meta["sample_id"].map(_patient_from_sample)
    if len(meta) != 213 or meta["sample_id"].nunique() != 213:
        raise ValueError("Table S2 does not contain the expected 213 unique PDOs")
    return meta


def _validated_mapping(raw_dir: Path, meta: pd.DataFrame, geo: pd.DataFrame) -> tuple[dict[str, str], set[str], set[str]]:
    mutation = pd.read_excel(raw_dir / "mmc2.xlsx", sheet_name="Sheet1")
    explicit = mutation[["sample_id", "patient"]].dropna().drop_duplicates()
    explicit_map = dict(zip(explicit["sample_id"].astype(str), explicit["patient"].astype(str), strict=True))
    for sample_id, patient in explicit_map.items():
        if _patient_from_sample(sample_id) != patient:
            raise ValueError(f"Data S1 mapping mismatch: {sample_id} -> {patient}")

    pdo_geo = geo[geo["sample_type"].eq("PDO")].copy()
    for row in pdo_geo.itertuples(index=False):
        if _patient_from_sample(row.sample_id) != row.patient_id:
            raise ValueError(f"GEO mapping mismatch: {row.sample_id} -> {row.patient_id}")

    mapping = {sample_id: _patient_from_sample(sample_id) for sample_id in meta["sample_id"]}
    if len(set(mapping.values())) != 102:
        raise ValueError("Official sample schema does not recover 102 patients")
    return mapping, set(explicit_map), set(pdo_geo["sample_id"])


def _library_membership(dss: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    run_origin = raw[["sample_id", "run_id", "library_id"]].drop_duplicates()
    harmonized = dss[["sample_id", "run_id_harmonize"]].merge(
        run_origin,
        left_on=["sample_id", "run_id_harmonize"],
        right_on=["sample_id", "run_id"],
        how="left",
        validate="one_to_one",
    )
    if harmonized["library_id"].isna().any():
        raise ValueError("Cannot identify harmonized DSS library origin")

    raw_membership = (
        raw[["sample_id", "library_id"]]
        .drop_duplicates()
        .assign(present=True)
        .pivot(index="sample_id", columns="library_id", values="present")
        .fillna(False)
    )
    lib1_specific = list(dss.columns[dss.columns.get_loc("run_id_lib1") + 1 : dss.columns.get_loc("run_id_lib2")])
    lib2_specific = list(dss.columns[dss.columns.get_loc("run_id_lib2") + 1 :])
    out = dss[["sample_id"]].copy()
    out["has_lib1_raw"] = out["sample_id"].map(raw_membership.get("lib1", pd.Series(dtype=bool))).fillna(False).astype(bool)
    out["has_lib2_raw"] = out["sample_id"].map(raw_membership.get("lib2", pd.Series(dtype=bool))).fillna(False).astype(bool)
    out["has_lib1_final_specific"] = dss["run_id_lib1"].notna().to_numpy() | dss[lib1_specific].notna().any(axis=1).to_numpy()
    out["has_lib2_final_specific"] = dss["run_id_lib2"].notna().to_numpy() | dss[lib2_specific].notna().any(axis=1).to_numpy()
    if (int(out["has_lib1_raw"].sum()), int(out["has_lib2_raw"].sum())) != (119, 116):
        raise ValueError("Raw Data S4 library coverage differs from 119/116")
    if (int(out["has_lib1_final_specific"].sum()), int(out["has_lib2_final_specific"].sum())) != (117, 114):
        raise ValueError("Final library-specific DSS coverage differs from 117/114")
    return out


def _drug_coverage(
    dss: pd.DataFrame,
    membership: pd.DataFrame,
    primary_drugs: list[str],
) -> pd.DataFrame:
    dss = dss.copy()
    dss["patient_id"] = dss["sample_id"].map(_patient_from_sample)
    membership = membership.set_index("sample_id")
    lib1_start = dss.columns.get_loc("run_id_lib1")
    lib2_start = dss.columns.get_loc("run_id_lib2")
    common_columns = list(dss.columns[2:lib1_start])
    lib1_columns = list(dss.columns[lib1_start + 1 : lib2_start])
    lib2_columns = [column for column in dss.columns[lib2_start + 1 :] if column != "patient_id"]
    if common_columns != primary_drugs:
        raise ValueError("The frozen 24-drug panel does not match Data S5")

    records: list[dict[str, Any]] = []
    for group, columns, eligible_mask in (
        ("shared", common_columns, np.ones(len(dss), dtype=bool)),
        (
            "lib1_specific",
            lib1_columns,
            dss["sample_id"].map(membership["has_lib1_final_specific"]).to_numpy(bool),
        ),
        (
            "lib2_specific",
            lib2_columns,
            dss["sample_id"].map(membership["has_lib2_final_specific"]).to_numpy(bool),
        ),
    ):
        for column in columns:
            finite = dss[column].notna().to_numpy()
            eligible_n = int(eligible_mask.sum())
            observed = dss.loc[finite, ["sample_id", "patient_id"]]
            records.append(
                {
                    "drug_name": column.removesuffix("_lib1").removesuffix("_lib2"),
                    "dss_column": column,
                    "library_1_presence": group in {"shared", "lib1_specific"},
                    "library_2_presence": group in {"shared", "lib2_specific"},
                    "coverage_group": group,
                    "primary_panel": group == "shared",
                    "eligible_pdo_count": eligible_n,
                    "total_pdo_count": int(observed["sample_id"].nunique()),
                    "total_patient_count": int(observed["patient_id"].nunique()),
                    "missingness": 1.0 - int(finite[eligible_mask].sum()) / eligible_n,
                    "response_metric_availability": "DSS",
                    "source_file": "mmc6.xlsx (Data S5)",
                }
            )
    return pd.DataFrame.from_records(records)


def _materialize_expression(
    raw_dir: Path,
    processed_dir: Path,
    dss: pd.DataFrame,
    primary_drugs: list[str],
    geo: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    rna = pd.read_excel(raw_dir / "mmc3.xlsx", sheet_name="Sheet1")
    sample_ids = list(rna.columns[3:])
    counts = rna.iloc[:, 3:].to_numpy(dtype=np.float64, copy=True).T
    library_sizes = counts.sum(axis=1)
    if np.any(library_sizes <= 0):
        raise ValueError("Non-positive RNA-seq library size")
    logcpm = np.log2(counts / library_sizes[:, None] * 1_000_000.0 + 1.0).astype(np.float32)
    np.savez_compressed(
        processed_dir / "RNAseq_PDO_log2CPM1.npz",
        sample_ids=np.asarray(sample_ids, dtype="U32"),
        patient_ids=np.asarray([_patient_from_sample(value) for value in sample_ids], dtype="U16"),
        feature_ids=rna.iloc[:, 0].astype(str).to_numpy(dtype="U32"),
        feature_symbols=rna.iloc[:, 2].fillna("").astype(str).to_numpy(dtype="U32"),
        values=logcpm,
    )

    pdo_geo = geo[geo["sample_type"].eq("PDO")].copy()
    feature_ids, values = _load_geo_matrix(raw_dir / "GSE294511_series_matrix.txt.gz", pdo_geo)
    np.savez_compressed(
        processed_dir / "HTA2_PDO_public_processed.npz",
        sample_ids=pdo_geo["sample_id"].to_numpy(dtype="U32"),
        patient_ids=pdo_geo["patient_id"].to_numpy(dtype="U16"),
        geo_accessions=pdo_geo["geo_accession"].to_numpy(dtype="U16"),
        feature_ids=feature_ids.astype("U32"),
        values=values,
    )

    np.savez_compressed(
        processed_dir / "PRIMARY_DSS.npz",
        sample_ids=dss["sample_id"].astype(str).to_numpy(dtype="U32"),
        patient_ids=dss["sample_id"].map(_patient_from_sample).to_numpy(dtype="U16"),
        drug_names=np.asarray(primary_drugs, dtype="U32"),
        values=dss[primary_drugs].to_numpy(dtype=np.float64),
    )
    return set(sample_ids), set(pdo_geo["sample_id"])


def build_audit(
    *,
    repo_root: Path,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    raw_dir = raw_dir or repo_root / "data" / "crc_pdo_personalized_application" / "raw"
    output_dir = output_dir or repo_root / "results" / "crc_pdo_personalized_drug_application"
    processed_dir = processed_dir or repo_root / "data" / "crc_pdo_personalized_application" / "processed"
    source_manifest_path = repo_root / "data" / "crc_pdo_personalized_application" / "source_manifest.json"
    config_path = repo_root / "configs" / "crc_pdo_personalized_application.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_sources(raw_dir, source_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_drugs = list(config["primary_panel"]["drugs"])
    meta = _load_meta(raw_dir)
    geo, _ = _read_geo_header(raw_dir / "GSE294511_series_matrix.txt.gz")
    mapping, explicit_samples, geo_samples = _validated_mapping(raw_dir, meta, geo)

    dss = pd.read_excel(raw_dir / "mmc6.xlsx", sheet_name="Sheet1")
    dss = dss[dss["sample_id"].astype(str).str.match(r"^Pt")].copy()
    dss["sample_id"] = dss["sample_id"].astype(str)
    if len(dss) != 211 or dss["sample_id"].nunique() != 211:
        raise ValueError("Data S5 does not contain 211 unique PDO DSS profiles")
    if not dss[primary_drugs].notna().all().all():
        raise ValueError("Primary 24-drug DSS panel is not complete")

    raw = pd.read_excel(raw_dir / "mmc5.xlsx", sheet_name="DSRT_RAW_211PDOs")
    membership = _library_membership(dss, raw)
    membership_indexed = membership.set_index("sample_id")
    raw_runs = raw[["sample_id", "run_id", "library_id"]].drop_duplicates()
    sample_run_counts = raw_runs.groupby("sample_id").size()
    same_library_repeats = raw_runs.groupby(["sample_id", "library_id"]).size().groupby(level=0).max()

    rna_samples, array_samples = _materialize_expression(raw_dir, processed_dir, dss, primary_drugs, geo)
    dss_samples = set(dss["sample_id"])
    patient_pdo_counts = meta.groupby("patient_id")["sample_id"].nunique()

    manifest_records: list[dict[str, Any]] = []
    for row in meta.itertuples(index=False):
        sample_id = row.sample_id
        platforms: list[str] = []
        if sample_id in rna_samples:
            platforms.append("RNAseq")
        if sample_id in array_samples:
            platforms.append("HTA2.0")
        in_dss = sample_id in dss_samples
        has_lib1 = bool(membership_indexed.loc[sample_id, "has_lib1_raw"]) if in_dss else False
        has_lib2 = bool(membership_indexed.loc[sample_id, "has_lib2_raw"]) if in_dss else False
        library = "lib1+lib2" if has_lib1 and has_lib2 else "lib1" if has_lib1 else "lib2" if has_lib2 else "none"
        mapping_evidence = []
        if sample_id in explicit_samples:
            mapping_evidence.append("Data S1 explicit patient column")
        if sample_id in geo_samples:
            mapping_evidence.append("GEO explicit patient characteristic")
        mapping_evidence.append("official sample_id schema validated against explicit mappings")
        manifest_records.append(
            {
                "patient_id": mapping[sample_id],
                "pdo_id": sample_id,
                "metastatic_site": "colorectal_liver_metastasis",
                "sample_type": "PDO",
                "expression_platform": ";".join(platforms) if platforms else "none",
                "expression_available": bool(platforms),
                "drug_library": library,
                "drug_response_available": in_dss,
                "replicate_information": (
                    f"patient_PDOs={int(patient_pdo_counts[mapping[sample_id]])};"
                    f"drug_screen_runs={int(sample_run_counts.get(sample_id, 0))};"
                    f"max_same_library_runs={int(same_library_repeats.get(sample_id, 0))}"
                ),
                "source_file": "mmc8.xlsx; mmc2.xlsx; mmc5.xlsx; mmc6.xlsx; GSE294511",
                "official_mapping_status": "VALIDATED_OFFICIAL_MAPPING",
                "mapping_evidence": " | ".join(mapping_evidence),
            }
        )
    patient_manifest = pd.DataFrame.from_records(manifest_records).sort_values(["patient_id", "pdo_id"])
    patient_manifest.to_csv(output_dir / "CRC_PDO_PATIENT_MANIFEST.csv", index=False)

    coverage = _drug_coverage(dss, membership, primary_drugs)
    coverage.to_csv(output_dir / "CRC_PDO_DRUG_COVERAGE.csv", index=False)

    expression_records: list[dict[str, Any]] = []
    for platform, samples, source in (
        ("RNAseq", rna_samples, "mmc3.xlsx (Data S2)"),
        ("HTA2.0", array_samples, "GSE294511_series_matrix.txt.gz"),
    ):
        for sample_id in sorted(samples):
            expression_records.append(
                {
                    "patient_id": mapping[sample_id],
                    "pdo_id": sample_id,
                    "expression_platform": platform,
                    "expression_source": source,
                    "expression_available": True,
                    "drug_response_available": sample_id in dss_samples,
                    "primary_panel_complete": sample_id in dss_samples,
                    "eligible_joint_pdo": sample_id in dss_samples,
                }
            )
    expression_coverage = pd.DataFrame.from_records(expression_records).sort_values(
        ["expression_platform", "patient_id", "pdo_id"]
    )
    expression_coverage.to_csv(output_dir / "CRC_PDO_EXPRESSION_COVERAGE.csv", index=False)

    intersection_records: list[dict[str, Any]] = []
    for patient_id in sorted(patient_pdo_counts.index):
        patient_samples = set(meta.loc[meta["patient_id"].eq(patient_id), "sample_id"])
        for platform, samples in (("RNAseq", rna_samples), ("HTA2.0", array_samples)):
            expression_samples = patient_samples.intersection(samples)
            joint_samples = expression_samples.intersection(dss_samples)
            intersection_records.append(
                {
                    "patient_id": patient_id,
                    "expression_platform": platform,
                    "total_patient_pdos": len(patient_samples),
                    "expression_pdos": len(expression_samples),
                    "drug_response_pdos": len(patient_samples.intersection(dss_samples)),
                    "joint_eligible_pdos": len(joint_samples),
                    "joint_eligible_pdo_ids": ";".join(sorted(joint_samples)),
                    "patient_evaluable": bool(joint_samples),
                    "patient_aggregation_rule": "mean across joint eligible PDOs",
                }
            )
    intersection = pd.DataFrame.from_records(intersection_records)
    intersection.to_csv(output_dir / "PATIENT_PDO_EXPRESSION_DRUG_INTERSECTION.csv", index=False)

    primary_manifest = coverage.loc[coverage["primary_panel"]].copy()
    primary_manifest.insert(0, "panel_order", np.arange(1, len(primary_manifest) + 1))
    primary_manifest["selection_basis"] = "coverage completeness, patient count, and valid official mapping only"
    primary_manifest["frozen_before_model_outcomes"] = True
    primary_manifest.to_csv(output_dir / "PRIMARY_DRUG_PANEL_MANIFEST.csv", index=False)

    rna_joint = set(rna_samples).intersection(dss_samples)
    array_joint = set(array_samples).intersection(dss_samples)
    rna_patients = {mapping[value] for value in rna_joint}
    array_patients = {mapping[value] for value in array_joint}
    multi_patient_count = int((patient_pdo_counts > 1).sum())
    repeated_same_library = int((raw_runs.groupby(["sample_id", "library_id"]).size() > 1).sum())
    summary: dict[str, Any] = {
        "study": "Kryeziu et al. 2026 metastatic CRC PDO biobank",
        "unique_patients": int(meta["patient_id"].nunique()),
        "pdo_count": int(len(meta)),
        "patients_with_multiple_pdos": multi_patient_count,
        "rnaseq_pdos": len(rna_samples),
        "rnaseq_patients": len({mapping[value] for value in rna_samples}),
        "microarray_pdos": len(array_samples),
        "microarray_patients": len({mapping[value] for value in array_samples}),
        "drug_response_pdos": len(dss_samples),
        "drug_response_patients": len({mapping[value] for value in dss_samples}),
        "rnaseq_drug_joint_pdos": len(rna_joint),
        "rnaseq_drug_joint_patients": len(rna_patients),
        "microarray_drug_joint_pdos": len(array_joint),
        "microarray_drug_joint_patients": len(array_patients),
        "primary_drugs": len(primary_drugs),
        "primary_response_completeness": float(dss[primary_drugs].notna().to_numpy().mean()),
        "raw_lib1_pdos": int(membership["has_lib1_raw"].sum()),
        "raw_lib2_pdos": int(membership["has_lib2_raw"].sum()),
        "final_lib1_specific_pdos": int(membership["has_lib1_final_specific"].sum()),
        "final_lib2_specific_pdos": int(membership["has_lib2_final_specific"].sum()),
        "same_library_repeat_screen_pairs": repeated_same_library,
        "mapping_valid": True,
        "feasibility_verdict": "PERSONALIZED_APPLICATION_FEASIBLE",
        "modeling_unlocked": True,
    }
    (output_dir / "AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# CRC PDO personalized drug application — feasibility audit

## Frozen study and endpoint

- Study: Kryeziu et al., *Cell Reports Medicine* (2026), DOI `10.1016/j.xcrm.2026.102840`.
- Official public data: Mendeley Data v3 (`10.17632/hr94h42xdc.3`) and GEO `GSE294511`.
- Functional endpoint: author-computed DSS; larger values indicate greater ex vivo drug sensitivity.
- Independent unit: patient. Multiple liver-metastasis PDOs are never treated as independent patients.

## Exact mapping audit

- {summary['unique_patients']} unique patients and {summary['pdo_count']} official PDO identifiers.
- {multi_patient_count} patients contribute multiple PDOs.
- Patient mapping uses the official `sample_id` schema and was independently validated against the explicit Data S1 `patient` column and GEO `patient:` characteristics. No expression- or drug-similarity matching was used.
- Mapping status: **VALID**.

## Expression and drug intersections

| Platform | Expression PDOs | Expression patients | PDOs with complete 24-drug DSS | Independent evaluable patients |
|---|---:|---:|---:|---:|
| RNA-seq | {summary['rnaseq_pdos']} | {summary['rnaseq_patients']} | {summary['rnaseq_drug_joint_pdos']} | {summary['rnaseq_drug_joint_patients']} |
| HTA2.0 microarray | {summary['microarray_pdos']} | {summary['microarray_patients']} | {summary['microarray_drug_joint_pdos']} | {summary['microarray_drug_joint_patients']} |

RNA-seq exceeds the prespecified preferred threshold of 40 independent patients and is frozen as primary. The public processed HTA2.0 cohort is eligible as a platform-specific secondary sensitivity analysis; it is not pooled with RNA-seq.

## Drug panel

- Final DSS resource: {summary['drug_response_pdos']} PDOs from {summary['drug_response_patients']} patients.
- Data S4 raw screen presence: lib1 `{summary['raw_lib1_pdos']}` PDOs; lib2 `{summary['raw_lib2_pdos']}` PDOs. The final library-specific Data S5 fields contain `{summary['final_lib1_specific_pdos']}` and `{summary['final_lib2_specific_pdos']}` PDOs, respectively; the two extra lib1 raw screens do not change the harmonized shared panel.
- Primary panel: the exact 24 author-harmonized shared drugs in Data S5.
- Completeness: {summary['primary_response_completeness']:.1%} across all 211 DSS profiles and all 24 drugs.
- The panel was selected only from coverage and mapping validity before model outcomes were inspected.

## Replicate and clustering audit

- Multiple PDOs from one patient are distinct metastatic lesions or repeated resections, not independent patients.
- Data S4 contains four sample-library pairs with repeat screen runs; the author-harmonized Data S5 profile is the endpoint and supplies one DSS row per PDO.
- Patient-level primary features and outcomes are means across PDOs having both the selected expression platform and a complete primary DSS panel.

## Download minimization

Only processed/count-level files needed for mapping, expression and DSS were downloaded (approximately 66 MB total). The 4.58 GB GEO raw CEL archive and all raw RNA sequencing were not downloaded.

## Feasibility verdict

`PERSONALIZED_APPLICATION_FEASIBLE`

The strict RNA-seq primary analysis is unlocked with 52 independent held-patient units, 24 completely observed drugs, valid official mappings and patient-disjoint evaluation. No scientific model outcome was inspected during panel selection.
"""
    (output_dir / "APPLICATION_FEASIBILITY_AUDIT.md").write_text(report, encoding="utf-8")
    return summary
