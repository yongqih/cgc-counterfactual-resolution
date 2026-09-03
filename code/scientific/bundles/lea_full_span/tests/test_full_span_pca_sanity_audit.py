from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_span_sanity_test_module", ROOT / "scripts/run_full_span_pca_sanity_audit.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_direct_svd_finds_centered_training_rank_and_reconstructs() -> None:
    rng = np.random.default_rng(21)
    matrix = rng.normal(size=(18, 30))
    svd = MODULE.direct_svd(matrix)
    assert svd.numerical_rank == len(matrix) - 1
    prediction = MODULE.affine_reconstruction(matrix, svd, svd.numerical_rank)
    _, _, score = MODULE.fidelity(matrix, prediction)
    assert score >= 1.0 - MODULE.RECONSTRUCTION_TOLERANCE
    assert svd.orthogonality_error <= MODULE.ORTHOGONALITY_TOLERANCE


def test_projection_energy_identity_and_bounds() -> None:
    rng = np.random.default_rng(22)
    train = rng.normal(size=(15, 24))
    held = rng.normal(size=(5, 24))
    svd = MODULE.direct_svd(train)
    energy, identity = MODULE.geometric_rows(
        held, svd, 0, np.arange(len(held)), np.asarray([f"l{i}" for i in range(len(held))])
    )
    assert all(0.0 <= row["q_raw"] <= 1.0 + MODULE.IDENTITY_TOLERANCE for row in energy)
    assert all(row["status"] == "PASS" for row in identity)


def test_affine_fidelity_is_exactly_recoverable_from_residual_energy() -> None:
    rng = np.random.default_rng(23)
    train = rng.normal(size=(17, 27))
    held = rng.normal(size=(6, 27))
    svd = MODULE.direct_svd(train)
    prediction = MODULE.affine_reconstruction(held, svd, svd.numerical_rank)
    denominator, sse, score = MODULE.fidelity(held, prediction)
    centered = held - svd.mean
    vectors = svd.retained
    residual = centered - (centered @ vectors.T) @ vectors
    assert np.isclose(sse, np.square(residual).sum(), rtol=0, atol=1e-12)
    assert np.isclose(score, 1.0 - np.square(residual).sum() / denominator, rtol=0, atol=1e-15)


def test_formal_script_trains_no_predictor_and_skips_nontrivial_loo() -> None:
    source = (ROOT / "scripts/run_full_span_pca_sanity_audit.py").read_text(encoding="utf-8")
    assert "RBFDesign(" not in source
    assert "fit_ladder_predictions(" not in source
    assert "SKIPPED_NOT_COMPUTATIONALLY_TRIVIAL" in source


def test_frozen_ranks_and_training_reconstruction_are_algebraically_complete() -> None:
    import pandas as pd

    output = ROOT / "results/cgc_full_span_pca_sanity"
    rank = pd.read_csv(output / "FULL_SPAN_PCA_RANK_AUDIT.csv")
    train = pd.read_csv(output / "FULL_SPAN_PCA_TRAIN_RECONSTRUCTION.csv")
    assert np.array_equal(rank["numerical_rank"], rank["theoretical_max_rank"])
    assert train["status"].eq("PASS").all()
    assert (train["relative_frobenius_error"] <= MODULE.RECONSTRUCTION_TOLERANCE).all()
    assert (train["train_fullrank_fidelity"] >= 1.0 - MODULE.RECONSTRUCTION_TOLERANCE).all()


def test_frozen_direct_svd_reproduces_D256_and_projection_identity() -> None:
    import pandas as pd

    output = ROOT / "results/cgc_full_span_pca_sanity"
    held = pd.read_csv(output / "FULL_SPAN_PCA_HELDOUT_FIDELITY.csv")
    discrepancy = held.loc[held["budget_label"].astype(str).eq("256"), "historical_discrepancy"].dropna()
    assert discrepancy.abs().max() <= MODULE.HISTORICAL_TOLERANCE
    identity = pd.read_csv(output / "FULL_SPAN_PCA_PROJECTION_IDENTITY_AUDIT.csv")
    assert identity["status"].eq("PASS").all()
    assert identity["relative_identity_error"].max() <= MODULE.IDENTITY_TOLERANCE


def test_frozen_energy_fractions_are_bounded_and_full_span_gain_is_small() -> None:
    import json
    import pandas as pd

    output = ROOT / "results/cgc_full_span_pca_sanity"
    energy = pd.read_csv(output / "FULL_SPAN_PCA_HELDOUT_ENERGY_FRACTION.csv")
    assert energy["q_raw"].between(0, 1).all()
    assert energy["q_affine"].between(0, 1).all()
    summary = json.loads((output / "FULL_SPAN_PCA_SANITY_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["train_reconstruction_pass"]
    assert summary["historical_D256_reproduction_pass"]
    assert summary["projection_identity_pass"]
    assert summary["increment_D256_to_fullspan"] < 0.01
    assert summary["predictive_model_trained"] is False
