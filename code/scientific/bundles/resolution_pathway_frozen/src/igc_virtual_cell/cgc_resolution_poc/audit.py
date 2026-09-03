from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_UTILITY_SHA256 = "bd3b4f5097b4f158f58da102849d52080bb9ae7b7a4c4325f37bf574fd1a7c08"
EXPECTED_ANNOTATION_SHA256 = "8e1f2619cde18d793c76674710aa9a2c9a58c0d394439fb288070d3a192d0bb2"
EXPECTED_REFERENCES = {(49, 92): 0.11068053088808294, (40, 4): 0.07718711664021238}
REFERENCE_ATOL = 1e-12
GENE_COUNT = 25_695


@dataclass(frozen=True)
class GeneReference:
    m: int
    k: int
    role: str
    expected_g: float
    reproduced_g: float
    absolute_difference: float
    vtruth_sum: float
    vafter_sum: float
    finite_entries: int
    passed: bool


PATHWAY_PATTERNS: dict[str, dict[str, Any]] = {
    "EGFR": {"targets": {"EGFR", "ERBB2", "ERBB3", "ERBB4"}, "prefixes": (), "moa": ("egfr", "erbb")},
    "MAPK": {
        "targets": {"KRAS", "NRAS", "HRAS", "BRAF", "ARAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3"},
        "prefixes": (),
        "moa": ("ras inhibitor", "raf inhibitor", "mek inhibitor", "mapk"),
    },
    "PI3K": {
        "targets": {"AKT1", "AKT2", "AKT3", "MTOR", "PTEN"},
        "prefixes": ("PIK3",),
        "moa": ("pi3k", "akt", "mtor"),
    },
    "JAK-STAT": {
        "targets": {"JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4", "STAT5", "STAT6"},
        "prefixes": (),
        "moa": ("jak/stat",),
    },
    "p53": {"targets": {"TP53", "MDM2", "MDM4"}, "prefixes": (), "moa": ("p53",)},
    "NFkB": {
        "targets": {"NFKB1", "NFKB2", "RELA", "RELB", "REL", "IKBKB", "CHUK"},
        "prefixes": (),
        "moa": ("nfkb", "nf-kb"),
    },
    "TGFb": {
        "targets": {"TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "TGFBR3", "SMAD2", "SMAD3", "SMAD4", "SMAD7"},
        "prefixes": (),
        "moa": ("tgfb", "tgf-beta"),
    },
    "WNT": {
        "targets": {"CTNNB1", "GSK3A", "GSK3B"},
        "prefixes": ("WNT", "FZD"),
        "moa": ("wnt",),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reproduce_gene_references(utility_path: Path) -> list[GeneReference]:
    utility_path = utility_path.resolve()
    if sha256(utility_path) != EXPECTED_UTILITY_SHA256:
        raise RuntimeError("FROZEN_UTILITY_HASH_MISMATCH")
    with np.load(utility_path, allow_pickle=False) as frozen:
        required = {"vtruth", "vafter", "models", "m_values", "k_values"}
        if not required.issubset(frozen.files):
            raise RuntimeError("FROZEN_UTILITY_SCHEMA_MISMATCH")
        models = list(map(str, frozen["models"]))
        m_values = frozen["m_values"].astype(int).tolist()
        k_values = frozen["k_values"].astype(int).tolist()
        model = models.index("M2_AFFINE_RIDGE")
        rows: list[GeneReference] = []
        for m, k in ((49, 92), (40, 4)):
            truth = np.asarray(frozen["vtruth"][m_values.index(m)], dtype=np.float64)
            after = np.asarray(
                frozen["vafter"][model, m_values.index(m), k_values.index(k)], dtype=np.float64
            )
            if truth.shape != (50, 93) or after.shape != truth.shape:
                raise RuntimeError("FROZEN_UTILITY_AXIS_MISMATCH")
            if not np.isfinite(truth).all() or not np.isfinite(after).all():
                raise RuntimeError("FROZEN_UTILITY_NONFINITE")
            truth_sum = float(truth.sum(dtype=np.float64))
            after_sum = float(after.sum(dtype=np.float64))
            observed = 1.0 - after_sum / truth_sum
            expected = EXPECTED_REFERENCES[(m, k)]
            difference = abs(observed - expected)
            rows.append(
                GeneReference(
                    m=m,
                    k=k,
                    role="primary" if (m, k) == (49, 92) else "secondary",
                    expected_g=expected,
                    reproduced_g=observed,
                    absolute_difference=difference,
                    vtruth_sum=truth_sum,
                    vafter_sum=after_sum,
                    finite_entries=int(np.isfinite(after).sum()),
                    passed=difference <= REFERENCE_ATOL,
                )
            )
    return rows


def _npz_schema(path: Path) -> dict[str, list[int]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: list(archive[name].shape) for name in archive.files}


def _has_gene_vector(schema: dict[str, list[int]]) -> bool:
    return any(GENE_COUNT in shape and len(shape) >= 2 for shape in schema.values())


def _pathway_coverage(annotation_path: Path) -> dict[str, int]:
    if sha256(annotation_path) != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("FROZEN_ANNOTATION_HASH_MISMATCH")
    annotations = pd.read_csv(annotation_path).fillna("")
    if len(annotations) != 93 or not annotations["intervention_axis"].tolist() == list(range(93)):
        raise RuntimeError("FROZEN_ANNOTATION_AXIS_MISMATCH")
    result: dict[str, int] = {}
    for pathway, rule in PATHWAY_PATTERNS.items():
        hits = 0
        for row in annotations.itertuples(index=False):
            targets = {value.strip().upper() for value in str(row.exact_targets).split(";") if value.strip()}
            exact = bool(targets & rule["targets"]) or any(
                target.startswith(prefix) for target in targets for prefix in rule["prefixes"]
            )
            moa = str(row.moa_fine).lower()
            if exact or any(token in moa for token in rule["moa"]):
                hits += 1
        result[pathway] = hits
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty-schema CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_resolution_inputs(
    output_root: Path,
    utility_path: Path,
    target_cache_root: Path,
    annotation_path: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    references = reproduce_gene_references(utility_path)
    reference_pass = all(row.passed for row in references)
    _write_csv(output_root / "RESOLUTION_GENE_REFERENCE.csv", [asdict(row) for row in references])

    audit_rows: list[dict[str, Any]] = []
    utility_schema = _npz_schema(utility_path)
    audit_rows.append(
        {
            "artifact": "frozen_utility_table",
            "path": str(utility_path.resolve()),
            "sha256": sha256(utility_path),
            "schema": json.dumps(utility_schema, separators=(",", ":")),
            "contains_gene_vector_truth": False,
            "contains_gene_vector_prediction": False,
            "eligible_for_projection": False,
            "reason": "ratio-of-sums sufficient statistics only",
        }
    )
    cache_paths = sorted(target_cache_root.glob("target_*.npz"))
    prediction_vectors = 0
    for path in cache_paths:
        schema = _npz_schema(path)
        has_vector = _has_gene_vector(schema)
        prediction_vectors += int(has_vector)
        audit_rows.append(
            {
                "artifact": path.stem,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "schema": json.dumps(schema, separators=(",", ":")),
                "contains_gene_vector_truth": False,
                "contains_gene_vector_prediction": has_vector,
                "eligible_for_projection": has_vector,
                "reason": "gene-vector artifact" if has_vector else "scalar utilities, metrics, and selected hyperparameters only",
            }
        )
    _write_csv(output_root / "RESOLUTION_INPUT_ARTIFACT_AUDIT.csv", audit_rows)

    coverage = _pathway_coverage(annotation_path)
    pathway_gate = sum(value >= 5 for value in coverage.values()) >= 4
    vector_gate = len(cache_paths) == 50 and prediction_vectors == 50
    if not reference_pass:
        gate_status = "RESOLUTION_AUDIT_REFERENCE_REPRODUCTION_FAIL"
        verdict = "COUNTERFACTUAL_RESOLUTION_AUDIT_INVALID"
    elif not vector_gate:
        gate_status = "FROZEN_M2_GENE_VECTOR_PREDICTIONS_NOT_PRESENT"
        verdict = "COUNTERFACTUAL_RESOLUTION_AUDIT_INVALID"
    elif not pathway_gate:
        gate_status = "PATHWAY_POC_COVERAGE_INSUFFICIENT"
        verdict = "COUNTERFACTUAL_RESOLUTION_AUDIT_INVALID"
    else:
        gate_status = "RESOLUTION_INPUT_GATES_PASS"
        verdict = "PENDING_FORMAL_RESOLUTION_ANALYSIS"
    return {
        "reference_pass": reference_pass,
        "vector_prediction_gate_pass": vector_gate,
        "target_cache_files": len(cache_paths),
        "target_caches_with_gene_vectors": prediction_vectors,
        "pathway_coverage": coverage,
        "eligible_pathway_count": sum(value >= 5 for value in coverage.values()),
        "pathway_gate_pass": pathway_gate,
        "gate_status": gate_status,
        "verdict": verdict,
    }

