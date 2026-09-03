"""Sample-level manifest schema and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


MANIFEST_COLUMNS = [
    "observation_id",
    "dataset_id",
    "study",
    "context_id",
    "context_name",
    "cell_type",
    "cell_line",
    "species",
    "perturbation_type",
    "perturbation_id",
    "is_control",
    "control_label",
    "timepoint",
    "dose",
    "replicate_id",
    "cell_count",
    "ntc_available",
    "gene_universe",
    "expression_path",
    "data_format",
    "expression_status",
    "normalization",
    "batch_id",
    "source_uri",
    "sha256",
]

REQUIRED_NONEMPTY = {
    "observation_id",
    "dataset_id",
    "study",
    "context_id",
    "species",
    "perturbation_type",
    "perturbation_id",
    "is_control",
    "replicate_id",
    "gene_universe",
    "expression_status",
}

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


class ManifestValidationError(ValueError):
    """Raised when a manifest violates the auditable data contract."""


@dataclass(frozen=True)
class ManifestLoadResult:
    frame: pd.DataFrame
    files: tuple[Path, ...]
    warnings: tuple[str, ...]


def parse_bool(value: object, *, field: str) -> bool:
    """Parse explicit booleans without Python's unsafe string truthiness."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ManifestValidationError(f"{field} contains a missing boolean value")
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ManifestValidationError(
        f"{field} contains {value!r}; expected one of {sorted(TRUE_VALUES | FALSE_VALUES)}"
    )


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def validate_manifest(frame: pd.DataFrame, *, source: str = "manifest") -> pd.DataFrame:
    """Validate and normalize a manifest, returning a defensive copy."""
    missing_columns = sorted(set(MANIFEST_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ManifestValidationError(
            f"{source} is missing required columns: {', '.join(missing_columns)}"
        )

    result = frame.loc[:, MANIFEST_COLUMNS].copy()
    if result.empty:
        return result

    problems: list[str] = []
    for column in sorted(REQUIRED_NONEMPTY):
        missing_rows = result.index[~_nonempty(result[column])].tolist()
        if missing_rows:
            problems.append(f"{column} empty at rows {missing_rows[:10]}")

    if result["observation_id"].duplicated().any():
        duplicate_ids = result.loc[
            result["observation_id"].duplicated(keep=False), "observation_id"
        ].astype(str).unique().tolist()
        problems.append(f"duplicate observation_id values: {duplicate_ids[:10]}")

    if problems:
        raise ManifestValidationError(f"{source}: " + "; ".join(problems))

    result["is_control"] = [
        parse_bool(value, field="is_control") for value in result["is_control"]
    ]
    ntc_mask = result["ntc_available"].notna() & result["ntc_available"].astype(str).str.strip().ne("")
    result.loc[ntc_mask, "ntc_available"] = [
        parse_bool(value, field="ntc_available")
        for value in result.loc[ntc_mask, "ntc_available"]
    ]
    result.loc[~ntc_mask, "ntc_available"] = False
    result["ntc_available"] = result["ntc_available"].astype(bool)

    result["cell_count"] = pd.to_numeric(result["cell_count"], errors="coerce")
    result["cell_count"] = result["cell_count"].fillna(0).astype(int)
    if (result["cell_count"] < 0).any():
        raise ManifestValidationError(f"{source}: cell_count cannot be negative")

    for context_id, group in result.groupby("context_id", sort=False):
        for stable_field in ("species", "context_name", "cell_type", "cell_line"):
            values = group[stable_field].dropna().astype(str).str.strip()
            values = values[values.ne("")].unique()
            if len(values) > 1:
                raise ManifestValidationError(
                    f"{source}: context {context_id!r} has conflicting {stable_field}: "
                    f"{values.tolist()}"
                )

    return result


def discover_manifest_files(project_root: Path, pattern: str) -> list[Path]:
    """Discover populated manifests, excluding the shipped empty template."""
    files = sorted(project_root.glob(pattern))
    return [path for path in files if path.name != "manifest_template.csv"]


def load_manifests(project_root: Path, pattern: str) -> ManifestLoadResult:
    """Load all discovered manifests and resolve expression paths."""
    files = discover_manifest_files(project_root, pattern)
    if not files:
        return ManifestLoadResult(
            frame=pd.DataFrame(columns=MANIFEST_COLUMNS), files=(), warnings=()
        )

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for path in files:
        raw = pd.read_csv(path, keep_default_na=True, low_memory=False)
        normalized = validate_manifest(raw, source=str(path))
        if normalized.empty:
            warnings.append(f"Ignored empty manifest: {path}")
            continue
        normalized["_manifest_path"] = str(path.resolve())

        def resolve_expression_path(value: object) -> str:
            if pd.isna(value) or not str(value).strip():
                return ""
            candidate = Path(str(value).strip())
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            return str(candidate.resolve())

        normalized["_expression_path_resolved"] = normalized["expression_path"].map(
            resolve_expression_path
        )
        frames.append(normalized)

    if not frames:
        return ManifestLoadResult(
            frame=pd.DataFrame(columns=MANIFEST_COLUMNS),
            files=tuple(files),
            warnings=tuple(warnings),
        )

    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined["observation_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = combined.loc[duplicate_mask, "observation_id"].unique().tolist()
        raise ManifestValidationError(
            "observation_id must be globally unique across manifests; duplicates: "
            + ", ".join(map(str, duplicate_ids[:10]))
        )
    return ManifestLoadResult(combined, tuple(files), tuple(warnings))


def hashable_manifest_view(frame: pd.DataFrame) -> Iterable[tuple[str, str]]:
    """Yield stable observation/dataset pairs for run provenance helpers."""
    for row in frame.sort_values("observation_id").itertuples(index=False):
        yield str(row.observation_id), str(row.dataset_id)
