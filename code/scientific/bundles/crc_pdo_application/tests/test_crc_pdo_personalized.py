from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.modeling import (
    PlatformData,
    _core_metrics,
    _kendall_tau_b_rows,
    _predict_from_kernel,
    _reversal_patient_metrics,
    _standardized_kernel,
    fit_oof_platform,
    primary_inference,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "crc_pdo_personalized_drug_application"


def test_frozen_primary_panel_is_exact_and_complete() -> None:
    config = json.loads((ROOT / "configs" / "crc_pdo_personalized_application.json").read_text())
    panel = pd.read_csv(RESULTS / "PRIMARY_DRUG_PANEL_MANIFEST.csv")
    assert panel["drug_name"].tolist() == config["primary_panel"]["drugs"]
    assert len(panel) == 24
    assert panel["frozen_before_model_outcomes"].all()
    assert (panel["missingness"] == 0).all()


def test_patient_mapping_and_intersection_are_patient_level() -> None:
    manifest = pd.read_csv(RESULTS / "CRC_PDO_PATIENT_MANIFEST.csv")
    intersection = pd.read_csv(RESULTS / "PATIENT_PDO_EXPRESSION_DRUG_INTERSECTION.csv")
    assert manifest["pdo_id"].nunique() == 213
    assert manifest["patient_id"].nunique() == 102
    assert (manifest["official_mapping_status"] == "VALIDATED_OFFICIAL_MAPPING").all()
    rna = intersection.query("expression_platform == 'RNAseq' and patient_evaluable")
    array = intersection.query("expression_platform == 'HTA2.0' and patient_evaluable")
    assert len(rna) == 52
    assert len(array) == 50
    assert rna["patient_id"].is_unique
    assert array["patient_id"].is_unique


def test_feasibility_gate_is_unlocked_without_outcomes() -> None:
    summary = json.loads((RESULTS / "AUDIT_SUMMARY.json").read_text())
    assert summary["mapping_valid"] is True
    assert summary["primary_drugs"] == 24
    assert summary["primary_response_completeness"] == 1.0
    assert summary["rnaseq_drug_joint_patients"] == 52
    assert summary["feasibility_verdict"] == "PERSONALIZED_APPLICATION_FEASIBLE"


def test_kendall_tau_b_and_reversal_metrics_have_expected_orientation() -> None:
    truth = np.tile(np.arange(24, dtype=float), (2, 1))
    same = truth.copy()
    reversed_order = truth[:, ::-1]
    assert np.allclose(_kendall_tau_b_rows(truth, same), 1.0)
    assert np.allclose(_kendall_tau_b_rows(truth, reversed_order), -1.0)

    population = truth.copy()
    patient_truth = truth.copy()
    patient_truth[:, [0, 1]] = patient_truth[:, [1, 0]]
    reversal = _reversal_patient_metrics(patient_truth, patient_truth, population, 1e-12)
    assert np.all(reversal["true_reversals"] >= 1)
    assert np.allclose(reversal["recall"], 1.0)
    assert np.allclose(reversal["precision"], 1.0)


def test_outer_predictions_are_patient_disjoint_and_population_excludes_held() -> None:
    rng = np.random.default_rng(17)
    n_patients = 10
    latent = rng.normal(size=(n_patients, 4))
    x = latent @ rng.normal(size=(4, 30)) + 0.01 * rng.normal(size=(n_patients, 30))
    y = latent @ rng.normal(size=(4, 24)) + np.arange(24, dtype=float)
    data = PlatformData(
        platform="synthetic",
        patient_ids=np.asarray([f"Pt{index + 1}" for index in range(n_patients)]),
        x=x,
        y=y,
        drug_names=np.asarray([f"drug_{index}" for index in range(24)]),
        eligible_pdo_ids=tuple((f"PDO_{index}",) for index in range(n_patients)),
    )
    config = {
        "model": {
            "alpha_grid": [0.01, 0.1, 1.0],
            "seed": 4,
            "pca_components": 3,
        }
    }
    predictions, folds, caches = fit_oof_platform(data, config)
    assert folds["patient_overlap"].eq(0).all()
    assert len(caches) == n_patients
    for held_index in range(n_patients):
        expected = np.mean(np.delete(y, held_index, axis=0), axis=0)
        assert np.allclose(predictions["Population"][held_index], expected)
        assert held_index not in caches[held_index].train_indices
    assert all(np.isfinite(values).all() for values in predictions.values())


def test_dual_ridge_matches_training_standardized_primal_solution() -> None:
    rng = np.random.default_rng(29)
    x_train = rng.normal(size=(8, 12))
    x_test = rng.normal(size=(3, 12))
    y = rng.normal(size=(8, 24))
    alpha = 0.37
    fit = _standardized_kernel(x_train, x_test)
    observed = _predict_from_kernel(fit, y, alpha)

    mean = x_train.mean(axis=0)
    variance = x_train.var(axis=0)
    keep = variance > 1e-12
    z_train = ((x_train[:, keep] - mean[keep]) / np.sqrt(variance[keep])) / np.sqrt(keep.sum())
    z_test = ((x_test[:, keep] - mean[keep]) / np.sqrt(variance[keep])) / np.sqrt(keep.sum())
    coefficient = np.linalg.solve(z_train.T @ z_train + alpha * np.eye(keep.sum()), z_train.T @ y)
    expected = z_test @ coefficient
    assert np.allclose(observed, expected, atol=1e-10)


def test_shared_signflip_is_restudentized_within_each_draw() -> None:
    rng = np.random.default_rng(101)
    population = np.tile(np.linspace(0, 6, 24), (60, 1)) + rng.normal(scale=0.15, size=(60, 24))
    truth = population + rng.normal(scale=2.0, size=(60, 24))
    prediction = population + 0.8 * (truth - population) + rng.normal(scale=0.25, size=(60, 24))
    config = {
        "evaluation": {"pair_tolerance": 1e-12},
        "inference": {"seed": 9, "bootstrap_draws": 1000, "signflip_draws": 2000},
    }
    table, _ = primary_inference(truth, prediction, population, config)
    assert table["positive_after_joint_correction"].all()
    assert (table["one_sided_maxT_adjusted_p"] <= 0.05).all()


def test_final_oof_tables_recompute_population_reference_and_primary_estimates() -> None:
    predictions = pd.read_csv(RESULTS / "HELD_PATIENT_FUNCTIONAL_PREDICTIONS.csv")
    primary = predictions.query("platform == 'RNAseq' and estimator == 'Ridge'")
    assert len(primary) == 52 * 24
    assert not primary.duplicated(["patient_id", "drug_name"]).any()
    patient_order = primary["patient_id"].drop_duplicates().tolist()
    drug_order = primary["drug_name"].drop_duplicates().tolist()

    def matrix(column: str) -> np.ndarray:
        return primary.pivot(index="patient_id", columns="drug_name", values=column).loc[
            patient_order, drug_order
        ].to_numpy(float)

    truth = matrix("true_DSS")
    personalized = matrix("predicted_DSS")
    population = matrix("population_DSS")
    for held_index in range(len(truth)):
        assert np.allclose(population[held_index], np.mean(np.delete(truth, held_index, axis=0), axis=0))

    metrics = _core_metrics(truth, personalized, population, 1e-12)
    inference = pd.read_csv(RESULTS / "PRIMARY_APPLICATION_INFERENCE.csv").set_index("metric")
    assert np.isclose(inference.loc["PG_macro", "estimate"], metrics["PG_macro"])
    assert np.isclose(inference.loc["g_func", "estimate"], metrics["g_func"])
    assert np.isclose(
        inference.loc["reversal_balanced_accuracy_minus_0.5", "estimate"],
        metrics["reversal_balanced_accuracy"] - 0.5,
    )
    assert inference["positive_after_joint_correction"].all()


def test_final_null_and_fold_manifests_are_complete() -> None:
    folds = pd.read_csv(RESULTS / "MODEL_FOLD_MANIFEST.csv")
    null_draws = pd.read_csv(RESULTS / "NULL_CONTROL_DRAWS.csv")
    qa = json.loads((RESULTS / "CRC_PDO_PERSONALIZED_APPLICATION_FINAL_QA.json").read_text())
    assert len(folds.query("platform == 'RNAseq'")) == 52
    assert len(folds.query("platform == 'HTA2.0'")) == 50
    assert folds["patient_overlap"].eq(0).all()
    assert len(null_draws.query("null_family == 'baseline_RNA_shuffle'")) == 1000
    assert len(null_draws.query("null_family == 'patient_response_profile_shuffle'")) == 1000
    assert qa["qa_pass"] is True
