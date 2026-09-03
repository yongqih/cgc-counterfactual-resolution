"""Auditable metadata and truth-gate helpers for the Lea Level-2 experiment."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_TREATMENTS = {"ETOH", "DEX"}


def parse_geo_soft(path: Path) -> pd.DataFrame:
    samples: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("^SAMPLE = "):
            if current:
                samples.append(current)
            current = {"gsm": raw.split(" = ", 1)[1]}
        elif current is not None and raw.startswith("!Sample_title = "):
            current["title"] = raw.split(" = ", 1)[1]
        elif current is not None and raw.startswith("!Sample_characteristics_ch1 = cell line: "):
            current["cell_line"] = raw.rsplit(": ", 1)[1]
        elif current is not None and raw.startswith("!Sample_characteristics_ch1 = treatment: "):
            current["treatment_label"] = raw.rsplit(": ", 1)[1]
        elif current is not None and raw.startswith("!Sample_relation = SRA: "):
            current["experiment_accession"] = raw.rsplit("=", 1)[1]
    if current:
        samples.append(current)
    result = pd.DataFrame(samples)
    if result["gsm"].duplicated().any():
        raise AssertionError("GEO sample accessions must be unique")
    return result


def replicate_label(raw_name: str) -> str:
    if raw_name.endswith(".x"):
        return "x"
    if raw_name.endswith(".y"):
        return "y"
    match = re.search(r"-v(\d+)-", raw_name, flags=re.IGNORECASE)
    if match:
        return f"v{match.group(1)}"
    return "single"


def validate_processed_metadata(metadata: pd.DataFrame) -> None:
    required = {
        "line", "1000_genomes_id1", "pop", "pop2", "raw_file_name", "treatment"
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Processed metadata fields missing: {sorted(missing)}")
    if metadata["raw_file_name"].duplicated().any():
        raise AssertionError("Processed expression sample names must be unique")
    if not set(metadata["pop2"].dropna()).issubset({"AFR", "EUR"}):
        raise AssertionError("Unexpected broad ancestry label")


def pairing_table(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_processed_metadata(metadata)
    samples = metadata.loc[metadata["treatment"].isin(ALLOWED_TREATMENTS)].copy()
    samples["replicate_label"] = samples["raw_file_name"].map(replicate_label)
    counts = samples.pivot_table(
        index="line", columns="treatment", values="raw_file_name", aggfunc="count", fill_value=0
    ).reindex(
        index=sorted(metadata["line"].unique()), columns=["ETOH", "DEX"], fill_value=0
    )
    identity = metadata.drop_duplicates("line").set_index("line")
    summary = counts.join(identity[["1000_genomes_id1", "pop", "pop2"]])
    summary["paired"] = (summary["ETOH"] > 0) & (summary["DEX"] > 0)
    summary["pairing_class"] = np.select(
        [
            (summary["ETOH"] == 0) & (summary["DEX"] == 0),
            (summary["ETOH"] == 1) & (summary["DEX"] == 1),
            (summary["ETOH"] == 2) & (summary["DEX"] == 1),
            (summary["ETOH"] == 1) & (summary["DEX"] == 2),
            (summary["ETOH"] == 2) & (summary["DEX"] == 2),
            (summary["ETOH"] == 0),
            (summary["DEX"] == 0),
        ],
        ["neither", "one_to_one", "two_ETOH_one_DEX", "one_ETOH_two_DEX", "two_to_two", "DEX_only", "ETOH_only"],
        default="other",
    )
    return samples, summary.reset_index()


def truth_complete_lines(samples: pd.DataFrame) -> dict[str, tuple[str, str, str, str]]:
    """Return matched ETOH/DEX columns for two independent versions per LCL."""
    output: dict[str, tuple[str, str, str, str]] = {}
    for line, group in samples.groupby("line"):
        by_treatment = {
            treatment: dict(zip(rows["replicate_label"], rows["raw_file_name"], strict=True))
            for treatment, rows in group.groupby("treatment")
        }
        if set(by_treatment) != ALLOWED_TREATMENTS:
            continue
        common = sorted(set(by_treatment["ETOH"]) & set(by_treatment["DEX"]))
        if len(common) < 2:
            continue
        first, second = common[:2]
        output[line] = (
            by_treatment["ETOH"][first], by_treatment["DEX"][first],
            by_treatment["ETOH"][second], by_treatment["DEX"][second],
        )
    return output


def outer_train_residuals(
    delta: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_train = delta[train_indices].mean(axis=0)
    return delta[train_indices] - mean_train, delta[test_indices] - mean_train, mean_train
