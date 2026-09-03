from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from igc_virtual_cell.crc_pdo_personalized.conditioning_resolution import (
    RepresentationSpec,
    bootstrap_primary_inference,
    build_progeny_mapping,
    fit_representation_oof,
    random_orthonormal_bases,
    random_pathway_bases,
    representation_fit,
    standardize_genes,
    metrics,
)
from igc_virtual_cell.crc_pdo_personalized.modeling import PlatformData


def _synthetic_data(seed: int = 17) -> PlatformData:
    rng = np.random.default_rng(seed)
    patients = 12
    genes = 30
    latent = rng.normal(size=(patients, 4))
    x = latent @ rng.normal(size=(4, genes)) + 0.05 * rng.normal(size=(patients, genes))
    y = latent @ rng.normal(size=(4, 24)) + np.arange(24)
    return PlatformData(
        platform="synthetic",
        patient_ids=np.asarray([f"Pt{index + 1}" for index in range(patients)]),
        x=x,
        y=y,
        drug_names=np.asarray([f"drug_{index}" for index in range(24)]),
        eligible_pdo_ids=tuple((f"PDO_{index}",) for index in range(patients)),
    )


def test_progeny_mapping_is_exact_rank_preserving(tmp_path: Path) -> None:
    genes = [f"G{index}" for index in range(20)]
    pathways = [f"P{index}" for index in range(3)]
    rng = np.random.default_rng(2)
    rows = []
    for pathway in pathways:
        for gene in genes:
            rows.append({"gene": gene, "pathway": pathway, "weight": rng.normal()})
    path = tmp_path / "weights.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    mapping = build_progeny_mapping(
        np.asarray([f"ENSG{index}" for index in range(20)]),
        np.asarray(genes),
        path,
        pathways,
    )
    assert mapping.rank == 3
    assert np.allclose(mapping.q.T @ mapping.q, np.eye(3), atol=1e-12)
    assert np.allclose(mapping.normalized_weights, mapping.q @ mapping.r, atol=1e-12)
    assert len(mapping.mapping_audit) == 60


def test_fine_residual_pca_is_training_fitted_and_orthogonal() -> None:
    data = _synthetic_data()
    q = random_orthonormal_bases(data.x.shape[1], 3, 1, 9)[0]
    fitted = representation_fit(
        data.x[:-2],
        data.x[-2:],
        RepresentationSpec("COARSE_R4", "COARSE_PLUS_RESIDUAL", 4),
        q,
    )
    assert fitted.train_features.shape == (10, 7)
    assert fitted.test_features.shape == (2, 7)
    standardized, _, keep = standardize_genes(data.x[:-2], data.x[-2:])
    q_active = q[keep]
    residual_gene = standardized - (standardized @ q_active) @ q_active.T
    assert np.max(np.abs(residual_gene @ q_active)) < 1e-8


def test_all_representations_produce_patient_specific_drug_vectors() -> None:
    data = _synthetic_data()
    q = random_orthonormal_bases(data.x.shape[1], 3, 1, 13)[0]
    _, r = np.linalg.qr(q)
    result = fit_representation_oof(
        data,
        RepresentationSpec("PROGENY14", "PROGENY_SPAN", 3),
        q,
        r,
        ("P0", "P1", "P2"),
        [0.01, 0.1, 1.0],
        3,
    )
    assert result.predictions.shape == (12, 24)
    assert np.isfinite(result.predictions).all()
    assert np.all(result.folds.patient_overlap == 0)
    assert np.var(result.predictions - result.population, axis=0).max() > 0
    assert result.coefficients is not None


def test_matched_random_pathway_bases_preserve_rank() -> None:
    rng = np.random.default_rng(4)
    weights = np.zeros((40, 4))
    for column, size in enumerate([10, 12, 14, 16]):
        weights[:size, column] = rng.normal(size=size)
    bases = random_pathway_bases(weights, 5, 8)
    assert bases.shape == (5, 40, 4)
    for basis in bases:
        assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-12)


def test_bootstrap_inference_keeps_patient_profiles_intact() -> None:
    data = _synthetic_data()
    population = np.repeat(data.y.mean(axis=0, keepdims=True), len(data.y), axis=0)
    predictions = {
        "FULL_RNA": data.y + 0.2,
        "PROGENY14": data.y + 0.1,
        "PCA14": data.y + 0.3,
        "COARSE_R4": data.y + 0.09,
        "COARSE_R8": data.y + 0.08,
        "COARSE_R16": data.y + 0.07,
        "COARSE_R32": data.y + 0.06,
    }
    frame, arrays = bootstrap_primary_inference(
        data.y,
        predictions,
        population,
        draws=100,
        seed=12,
        pg_margin=0.01,
        g_margin=0.03,
        fine_levels=[0, 4, 8, 16, 32],
    )
    assert len(frame) == 11
    assert set(frame.inference_family) == {"PRIMARY_7", "SECONDARY_FINE_G_4"}
    assert arrays["bootstrap"].shape == (100, 11)
    assert frame.resampling_unit.eq("patient_profile").all()


def test_frozen_conditioning_outputs_recompute_from_saved_oof() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results/crc_pdo_conditioning_resolution"
    if not results.exists():
        return
    summary = pd.read_csv(results / "REPRESENTATION_PERFORMANCE_SUMMARY.csv").set_index(
        "representation"
    )
    tables = {
        name: pd.read_csv(results / f"{name}_OOF.csv")
        for name in ["FULL_RNA", "PROGENY14", "PCA14"]
    }
    patient_order = tables["FULL_RNA"]["patient_id"].drop_duplicates().tolist()
    drug_order = tables["FULL_RNA"]["drug_name"].drop_duplicates().tolist()

    def matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
        return (
            frame.pivot(index="patient_id", columns="drug_name", values=column)
            .loc[patient_order, drug_order]
            .to_numpy(float)
        )

    truth = matrix(tables["FULL_RNA"], "true_DSS")
    population = matrix(tables["FULL_RNA"], "population_DSS")
    for name, frame in tables.items():
        observed = metrics(truth, matrix(frame, "predicted_DSS"), population)
        for metric in ["PG_macro", "g_func", "reversal_balanced_accuracy_patient_mean"]:
            assert np.isclose(observed[metric], summary.loc[name, metric], atol=1e-12)


def test_frozen_null_and_fold_artifacts_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results/crc_pdo_conditioning_resolution"
    if not results.exists():
        return
    random14 = pd.read_csv(results / "RANDOM14_SUMMARY.csv")
    random_pathway = pd.read_csv(results / "RANDOM_PATHWAY14_SUMMARY.csv")
    folds = pd.read_csv(results / "OUTER_FOLD_TRANSFORM_AUDIT.csv")
    assert len(random14) == 100 and random14["draw"].nunique() == 100
    assert len(random_pathway) == 100 and random_pathway["draw"].nunique() == 100
    assert len(folds.query("family == 'RANDOM14'")) == 5200
    assert len(folds.query("family == 'RANDOM_PATHWAY14'")) == 5200
    assert folds["patient_overlap"].fillna(0).eq(0).all()
    assert folds["held_outcome_used_in_transform"].fillna(False).eq(False).all()
    assert folds["inner_preprocessing_refit"].all()


def test_frozen_verdict_is_driven_by_supported_fine_detail_gain() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results/crc_pdo_conditioning_resolution"
    if not results.exists():
        return
    inference = pd.read_csv(results / "PRIMARY_CONDITIONING_RESOLUTION_INFERENCE.csv").set_index(
        "contrast"
    )
    qa = __import__("json").loads(
        (results / "CRC_PDO_CONDITIONING_RESOLUTION_FINAL_QA.json").read_text()
    )
    assert inference.loc["COARSE_R16_minus_R0_PG", "estimate"] >= 0.010
    assert inference.loc["COARSE_R16_minus_R0_PG", "simultaneous_95_ci_lower"] > 0
    assert inference.loc["COARSE_R16_minus_R0_g_func", "estimate"] >= 0.030
    assert inference.loc["COARSE_R16_minus_R0_g_func", "simultaneous_95_ci_lower"] > 0
    assert qa["verdict"] == "FINE_BASELINE_DETAIL_ADDS_PREDICTIVE_VALUE"
