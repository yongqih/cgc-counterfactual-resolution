from __future__ import annotations

import numpy as np
import pandas as pd

from igc_virtual_cell.level2_lea import (
    outer_train_residuals,
    pairing_table,
    replicate_label,
    truth_complete_lines,
)


def sample_metadata() -> pd.DataFrame:
    rows = []
    for treatment in ["ETOH", "DEX"]:
        for suffix in ["x", "y"]:
            rows.append({
                "line": "Line1", "1000_genomes_id1": "HG1", "pop": "YRI", "pop2": "AFR",
                "raw_file_name": f"Line1_{treatment}.R1.Aligned.counts.{suffix}", "treatment": treatment,
            })
    return pd.DataFrame(rows)


def test_replicate_versions_share_biological_group() -> None:
    samples, summary = pairing_table(sample_metadata())
    assert summary.loc[0, "pairing_class"] == "two_to_two"
    assert samples["line"].nunique() == 1


def test_replicate_label_is_deterministic_and_outcome_blind() -> None:
    assert replicate_label("Line1_ETOH.R1.Aligned.counts.x") == "x"
    assert replicate_label("Line1_DEX.R1.Aligned.counts.x") == "x"
    assert replicate_label("Sample_Line1-v2-DEX.R1.Aligned.counts") == "v2"


def test_truth_pairs_require_two_versions_on_both_sides() -> None:
    samples, _ = pairing_table(sample_metadata())
    assert set(truth_complete_lines(samples)) == {"Line1"}
    assert truth_complete_lines(samples.iloc[:-1]) == {}


def test_target_center_uses_outer_training_only() -> None:
    delta = np.array([[0.0], [2.0], [100.0]])
    train, test, mean = outer_train_residuals(delta, np.array([0, 1]), np.array([2]))
    assert mean.item() == 1.0
    assert test.item() == 99.0
    assert np.allclose(train.ravel(), [-1.0, 1.0])
