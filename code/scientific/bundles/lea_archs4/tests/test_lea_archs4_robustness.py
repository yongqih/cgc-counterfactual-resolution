from __future__ import annotations

import numpy as np
import pytest

from scripts.extract_lea_archs4_matched import equal_version_mean, log1p_cpm
from scripts.run_lea_archs4_robustness import collapse_historical_gene_rows, verdict_from_comparison


def test_log1p_cpm_uses_full_gene_library() -> None:
    counts = np.asarray([10, 30, 60], dtype=np.uint32)
    observed, library = log1p_cpm(counts, np.asarray([0, 2]))
    expected = np.log1p(np.asarray([10, 60]) * 10_000.0).astype(np.float32)
    assert library == 100.0
    np.testing.assert_array_equal(observed, expected)


def test_log1p_cpm_fails_closed_for_zero_library() -> None:
    with pytest.raises(RuntimeError, match="Invalid full-gene library"):
        log1p_cpm(np.zeros(3, dtype=np.uint32), np.asarray([0, 2]))


def test_version_aggregation_is_equal_mean_not_sum() -> None:
    first = np.asarray([1.0, 3.0], dtype=np.float32)
    second = np.asarray([5.0, 7.0], dtype=np.float32)
    np.testing.assert_array_equal(equal_version_mean([first, second]), np.asarray([3.0, 5.0], dtype=np.float32))


def test_version_aggregation_rejects_empty_group() -> None:
    with pytest.raises(RuntimeError, match="empty GSM group"):
        equal_version_mean([])


def test_cache_labels_are_pickle_free_unicode() -> None:
    labels = np.asarray(["Line1", "Line2"], dtype=str)
    assert labels.dtype.kind == "U"


def test_historical_duplicate_rows_are_collapsed_by_equal_mean() -> None:
    values = np.asarray([[1, 3, 9], [2, 6, 8]], dtype=np.float32)
    observed = collapse_historical_gene_rows(values, ["g1", "g1", "g2"], ["g1", "g2"])
    np.testing.assert_array_equal(observed, np.asarray([[2, 9], [4, 8]], dtype=np.float32))


def _comparison(post: list[float], arch: list[float]) -> "pd.DataFrame":
    import pandas as pd

    columns = ["Best full-gene R2", "PC1-16 R2", "PC1-8 R2", "PC1-4 R2"]
    rows = []
    for label, values in [
        ("Original post-SVA, matched 340", post),
        ("ARCHS4 raw-read-derived, matched 340", arch),
    ]:
        row = {"representation": label, **dict(zip(columns, values, strict=True))}
        row["resolution_gap_PC4_minus_gene"] = row["PC1-4 R2"] - row["Best full-gene R2"]
        rows.append(row)
    return pd.DataFrame(rows)


def test_verdict_rules_were_operationalized_before_outcomes() -> None:
    assert verdict_from_comparison(_comparison([0.00, 0.08, 0.12, 0.18], [0.02, 0.08, 0.12, 0.15])) == "ARCHS4_RESOLUTION_HIERARCHY_REPLICATED"
    assert verdict_from_comparison(_comparison([0.00, 0.08, 0.12, 0.18], [0.08, 0.09, 0.10, 0.12])) == "ARCHS4_RESOLUTION_HIERARCHY_PARTIAL"
    assert verdict_from_comparison(_comparison([0.00, 0.08, 0.12, 0.18], [0.12, 0.11, 0.10, 0.09])) == "ARCHS4_RESOLUTION_HIERARCHY_NOT_REPLICATED"
