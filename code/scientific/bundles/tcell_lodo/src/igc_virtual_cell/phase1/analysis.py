"""Run strict intervention-disjoint Phase I NTC-transferability analyses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import yaml

from igc_virtual_cell.phase1.materialize import META_COLUMNS, _ntc_matrix
from igc_virtual_cell.programs.decompose import ResponseProgramModel


STUDY_FILES = {
    "feng_genomewide": "feng_genomewide_deltas_min5.parquet",
    "feng_targeted": "feng_targeted_deltas.parquet",
    "replogle": "replogle_deltas.parquet",
    "frangieh": "frangieh_deltas.parquet",
}


@dataclass
class StudyData:
    name: str
    metadata: pd.DataFrame
    responses: pd.DataFrame
    ntc: pd.DataFrame


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _stable_fold(label: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def _safe_corr(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    value = spearmanr(x, y).statistic if rank else pearsonr(x, y).statistic
    return float(value)


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _load_ntc(root: Path, study: str) -> pd.DataFrame:
    if study == "feng_genomewide":
        manifest = pd.read_csv(root / "data/manifests/feng_2026_genomewide.csv", low_memory=False)
        return _ntc_matrix(
            root / "data/processed/feng_2026/feng_2026_genomewide_ntc_log1p_cpm.parquet",
            manifest,
        )
    if study == "feng_targeted":
        manifest = pd.read_csv(root / "data/manifests/feng_2026_targeted.csv", low_memory=False)
        return _ntc_matrix(
            root / "data/processed/feng_2026/feng_2026_targeted_ntc_log1p_cpm.parquet",
            manifest,
        )
    if study == "replogle":
        pieces = []
        for line in ("k562", "rpe1"):
            manifest = pd.read_csv(root / f"data/manifests/replogle_2022_{line}_essential.csv", low_memory=False)
            pieces.append(
                _ntc_matrix(
                    root / f"data/processed/replogle_2022/replogle_2022_{line}_ntc_log1p_cpm.parquet",
                    manifest,
                )
            )
        common = sorted(set(pieces[0].columns) & set(pieces[1].columns))
        return pd.concat([piece.loc[:, common] for piece in pieces]).sort_index()
    if study == "frangieh":
        manifest = pd.read_csv(root / "data/manifests/frangieh_2021_melanoma_rna.csv", low_memory=False)
        return _ntc_matrix(
            root / "data/processed/frangieh_2021/frangieh_2021_environment_ntc_log1p_cpm.parquet",
            manifest,
        )
    raise KeyError(study)


def load_study(root: Path, study: str) -> StudyData:
    frame = pd.read_parquet(root / "data/processed/phase1" / STUDY_FILES[study])
    metadata = frame.loc[:, META_COLUMNS].copy()
    responses = frame.drop(columns=META_COLUMNS).astype(np.float32)
    responses.index = metadata.index
    ntc = _load_ntc(root, study)
    common = sorted(set(responses.columns.astype(str)) & set(ntc.columns.astype(str)))
    contexts = sorted(set(metadata["context_id"].astype(str)) & set(ntc.index.astype(str)))
    keep = metadata["context_id"].astype(str).isin(contexts)
    metadata = metadata.loc[keep].reset_index(drop=True)
    responses = responses.loc[keep].reset_index(drop=True).loc[:, common]
    ntc = ntc.loc[contexts, common].astype(np.float32)
    return StudyData(study, metadata, responses, ntc)


def _descriptive_pairwise(study: StudyData, components: int, seed: int) -> pd.DataFrame:
    scaler = StandardScaler().fit(study.ntc)
    z = scaler.transform(study.ntc)
    ncomp = min(components, len(study.ntc) - 1, z.shape[1])
    pcs = PCA(n_components=max(1, ncomp), random_state=seed).fit_transform(z)
    covariance = LedoitWolf().fit(pcs) if len(study.ntc) > 2 else None
    rows = []
    contexts = list(study.ntc.index.astype(str))
    for si, source in enumerate(contexts):
        for ti, target in enumerate(contexts):
            if source == target:
                continue
            xv, yv = z[si], z[ti]
            cosine = float(np.dot(xv, yv) / max(np.linalg.norm(xv) * np.linalg.norm(yv), 1e-12))
            mahalanobis = (
                -float(
                    np.sqrt(
                        (pcs[si] - pcs[ti])
                        @ covariance.precision_
                        @ (pcs[si] - pcs[ti])
                    )
                )
                if covariance is not None
                else float("nan")
            )
            rows.append(
                {
                    "study": study.name,
                    "source_context": source,
                    "target_context": target,
                    "gene_cosine": cosine,
                    "gene_pearson": _safe_corr(xv, yv),
                    "standardized_euclidean_proximity": -float(np.linalg.norm(xv - yv)),
                    "pca_cosine": float(
                        np.dot(pcs[si], pcs[ti])
                        / max(np.linalg.norm(pcs[si]) * np.linalg.norm(pcs[ti]), 1e-12)
                    ),
                    "mahalanobis_proximity": mahalanobis,
                    "representation_scope": "descriptive_all_NTC_contexts_not_router_fit",
                }
            )
    return pd.DataFrame(rows)


@dataclass
class NTCFeatures:
    pairwise: pd.DataFrame
    pca_scores: pd.DataFrame
    ntc_model: ResponseProgramModel | None


def _fit_ntc_features(
    ntc: pd.DataFrame,
    *,
    excluded_context: str | None,
    components: int,
    seed: int,
) -> NTCFeatures:
    training = (
        ntc.drop(index=excluded_context, errors="ignore")
        if excluded_context is not None
        else ntc
    )
    if len(training) < 2:
        training = ntc
    scaler = StandardScaler().fit(training)
    z_train = scaler.transform(training)
    z_all = scaler.transform(ntc)
    ncomp = min(components, max(1, len(training) - 1), ntc.shape[1])
    pca = PCA(n_components=ncomp, random_state=seed).fit(z_train)
    pc_all = pca.transform(z_all)
    pc_frame = pd.DataFrame(
        pc_all,
        index=ntc.index.astype(str),
        columns=[f"ntc_pc_{i + 1}" for i in range(ncomp)],
    )
    ntc_model = None
    ntc_scores = None
    if len(training) >= 3:
        ntc_model = ResponseProgramModel.fit(
            training,
            method="nmf",
            n_components=min(components, len(training) - 1),
            standardize=True,
            random_state=seed,
        )
        ntc_scores = ntc_model.transform(ntc)
        ntc_scores = (ntc_scores - ntc_scores.loc[training.index].mean()) / (
            ntc_scores.loc[training.index].std(ddof=0).replace(0, 1)
        )
    covariance = LedoitWolf().fit(pc_frame.loc[training.index]) if len(training) > 2 else None
    contexts = list(ntc.index.astype(str))
    records = []
    for source in contexts:
        for target in contexts:
            if source == target:
                continue
            si = contexts.index(source)
            ti = contexts.index(target)
            xv, yv = z_all[si], z_all[ti]
            sp, tp = pc_frame.loc[source].to_numpy(), pc_frame.loc[target].to_numpy()
            row = {
                "source_context": source,
                "target_context": target,
                "gene_cosine": float(
                    np.dot(xv, yv) / max(np.linalg.norm(xv) * np.linalg.norm(yv), 1e-12)
                ),
                "gene_pearson": _safe_corr(xv, yv),
                "standardized_euclidean_proximity": -float(np.linalg.norm(xv - yv)),
                "pca_cosine": float(
                    np.dot(sp, tp) / max(np.linalg.norm(sp) * np.linalg.norm(tp), 1e-12)
                ),
                "mahalanobis_proximity": (
                    -float(np.sqrt((sp - tp) @ covariance.precision_ @ (sp - tp)))
                    if covariance is not None
                    else 0.0
                ),
            }
            if ntc_scores is not None:
                for index, column in enumerate(ntc_scores.columns):
                    row[f"ntc_program_{index + 1}_similarity"] = -abs(
                        float(ntc_scores.loc[source, column] - ntc_scores.loc[target, column])
                    )
            records.append(row)
    return NTCFeatures(pd.DataFrame(records), pc_frame, ntc_model)


def _load_genept(path: Path, perturbations: Iterable[str]) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as handle:
        embeddings = pickle.load(handle)
    labels = sorted(set(map(str, perturbations)) & set(embeddings))
    matrix = np.asarray([embeddings[label] for label in labels], dtype=np.float32)
    return labels, matrix


def _genept_coordinates(
    path: Path,
    all_perturbations: list[str],
    training_perturbations: set[str],
    components: int,
    seed: int,
) -> pd.DataFrame:
    labels, matrix = _load_genept(path, all_perturbations)
    label_index = {label: index for index, label in enumerate(labels)}
    train_indices = [label_index[label] for label in sorted(training_perturbations & set(labels))]
    scaler = StandardScaler().fit(matrix[train_indices])
    scaled_train = scaler.transform(matrix[train_indices])
    ncomp = min(components, len(train_indices) - 1, matrix.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=seed).fit(scaled_train)
    values = pca.transform(scaler.transform(matrix))
    return pd.DataFrame(
        values,
        index=labels,
        columns=[f"genept_pc_{index + 1}" for index in range(ncomp)],
    )


def _baseline_design(
    metadata: pd.DataFrame,
    genept: pd.DataFrame,
    ntc_pcs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    valid = metadata["perturbation_id"].astype(str).isin(genept.index)
    rows = metadata.loc[valid]
    ep = genept.loc[rows["perturbation_id"].astype(str)].to_numpy(dtype=np.float32)
    cp = ntc_pcs.loc[rows["context_id"].astype(str)].to_numpy(dtype=np.float32)
    interaction = np.einsum("ij,ik->ijk", ep, cp).reshape(len(rows), -1)
    return np.concatenate([ep, cp, interaction], axis=1), np.flatnonzero(valid.to_numpy())


def _program_match(model: ResponseProgramModel, ntc_model: ResponseProgramModel | None) -> list[int]:
    if ntc_model is None:
        return [0] * len(model.loadings())
    response = model.loadings().to_numpy(dtype=float)
    ntc = ntc_model.loadings().loc[:, model.genes].to_numpy(dtype=float)
    correlations = np.corrcoef(response, ntc)[: len(response), len(response) :]
    return np.nan_to_num(np.abs(correlations), nan=0.0).argmax(axis=1).tolist()


def _utility_rows(
    metadata: pd.DataFrame,
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    base_scores: np.ndarray,
    model: ResponseProgramModel,
    perturbations: set[str],
    allowed_contexts: set[str],
    *,
    max_pairs_per_intervention: int | None,
    seed: int,
) -> pd.DataFrame:
    direction = model.loadings().to_numpy(dtype=float) * model.scale[None, :]
    direction_centered = direction - direction.mean(axis=1, keepdims=True)
    direction_square = np.mean(np.square(direction), axis=1)
    centered_square = np.mean(np.square(direction_centered), axis=1)
    base_frame = pd.DataFrame(base_scores, index=scores.index, columns=scores.columns)
    base_gene = model.reconstruct(base_frame).to_numpy(dtype=float)
    truth_values = truth.loc[:, model.genes].to_numpy(dtype=float)
    score_values = scores.to_numpy(dtype=float)
    program_scale = np.std(score_values, axis=0, ddof=0)
    program_scale[program_scale == 0] = 1.0
    records: list[dict[str, object]] = []
    work = metadata.copy()
    work["row_index"] = np.arange(len(work))
    for perturbation, group in work.loc[
        work["perturbation_id"].astype(str).isin(perturbations)
        & work["context_id"].astype(str).isin(allowed_contexts)
    ].groupby("perturbation_id", sort=True):
        pairs = [
            (int(source.row_index), int(target.row_index))
            for source in group.itertuples(index=False)
            for target in group.itertuples(index=False)
            if source.context_id != target.context_id
        ]
        if max_pairs_per_intervention is not None and len(pairs) > max_pairs_per_intervention:
            pairs = sorted(
                pairs,
                key=lambda pair: hashlib.sha256(
                    f"{seed}|{perturbation}|{pair[0]}|{pair[1]}".encode()
                ).digest(),
            )[:max_pairs_per_intervention]
        for source_index, target_index in pairs:
            source_context = str(metadata.iloc[source_index]["context_id"])
            target_context = str(metadata.iloc[target_index]["context_id"])
            true_score = score_values[target_index]
            source_score = score_values[source_index]
            delta_score = source_score - base_scores[target_index]
            residual = base_gene[target_index] - truth_values[target_index]
            residual_dot = np.mean(direction * residual[None, :], axis=1)
            mse_utility = -(2 * delta_score * residual_dot + np.square(delta_score) * direction_square)

            x = base_gene[target_index]
            y = truth_values[target_index]
            xc = x - x.mean()
            yc = y - y.mean()
            y2 = np.mean(np.square(yc))
            x2 = np.mean(np.square(xc))
            xy = np.mean(xc * yc)
            base_corr = xy / max(np.sqrt(x2 * y2), 1e-12)
            vy = np.mean(direction_centered * yc[None, :], axis=1)
            xv = np.mean(direction_centered * xc[None, :], axis=1)
            numerator = xy + delta_score * vy
            denominator = np.sqrt(
                np.maximum(x2 + 2 * delta_score * xv + np.square(delta_score) * centered_square, 1e-12)
                * max(y2, 1e-12)
            )
            pearson_utility = numerator / denominator - base_corr
            program_corr = _safe_corr(source_score, true_score)
            for program_index in range(len(source_score)):
                records.append(
                    {
                        "perturbation_id": str(perturbation),
                        "source_context": source_context,
                        "target_context": target_context,
                        "response_program": program_index + 1,
                        "response_program_correlation": program_corr,
                        "amplitude_transferability": -abs(
                            float(source_score[program_index] - true_score[program_index])
                        ),
                        "normalized_amplitude_transferability": -abs(
                            float(source_score[program_index] - true_score[program_index])
                        )
                        / float(program_scale[program_index]),
                        "mse_utility": float(mse_utility[program_index]),
                        "pearson_utility": float(pearson_utility[program_index]),
                        "beneficial": bool(mse_utility[program_index] > 0),
                    }
                )
    return pd.DataFrame(records)


def _merge_pair_features(
    utilities: pd.DataFrame,
    features: NTCFeatures,
    program_match: list[int],
) -> pd.DataFrame:
    merged = utilities.merge(
        features.pairwise,
        on=["source_context", "target_context"],
        how="left",
        validate="many_to_one",
    )
    ntc_columns = [column for column in merged if column.startswith("ntc_program_")]
    if ntc_columns:
        values = merged[ntc_columns].to_numpy(dtype=float)
        indices = np.asarray(
            [program_match[int(program) - 1] for program in merged["response_program"]],
            dtype=int,
        )
        indices = np.minimum(indices, values.shape[1] - 1)
        merged["matched_program_similarity"] = values[np.arange(len(merged)), indices]
    else:
        merged["matched_program_similarity"] = merged["gene_cosine"]
    return merged


def _predict_router(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    feature_columns = [
        "gene_cosine",
        "gene_pearson",
        "standardized_euclidean_proximity",
        "pca_cosine",
        "mahalanobis_proximity",
        *sorted(column for column in train if column.startswith("ntc_program_")),
    ]
    output = test.copy()
    output["prediction_global"] = np.nan
    output["prediction_program_matched"] = np.nan
    output["prediction_ridge_programwise"] = np.nan
    for program in sorted(test["response_program"].unique()):
        training = train.loc[train["response_program"].eq(program)].dropna(
            subset=["mse_utility", *feature_columns, "matched_program_similarity"]
        )
        testing = test.loc[test["response_program"].eq(program)]
        if len(training) > max_rows:
            training = training.sample(max_rows, random_state=seed + int(program))
        if len(training) < 8 or training["mse_utility"].std() == 0:
            mean = float(training["mse_utility"].mean()) if len(training) else 0.0
            output.loc[testing.index, [
                "prediction_global",
                "prediction_program_matched",
                "prediction_ridge_programwise",
            ]] = mean
            continue
        global_model = LinearRegression().fit(training[["gene_cosine"]], training["mse_utility"])
        matched_model = LinearRegression().fit(
            training[["matched_program_similarity"]], training["mse_utility"]
        )
        ridge = Ridge(alpha=alpha).fit(training[feature_columns], training["mse_utility"])
        output.loc[testing.index, "prediction_global"] = global_model.predict(
            testing[["gene_cosine"]]
        )
        output.loc[testing.index, "prediction_program_matched"] = matched_model.predict(
            testing[["matched_program_similarity"]]
        )
        output.loc[testing.index, "prediction_ridge_programwise"] = ridge.predict(
            testing[feature_columns]
        )
    tree_features = [*feature_columns, "matched_program_similarity", "response_program"]
    tree_train = train.dropna(subset=["mse_utility", *tree_features])
    if len(tree_train) > max_rows:
        tree_train = tree_train.sample(max_rows, random_state=seed)
    if len(tree_train) >= 20:
        tree = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=100,
            learning_rate=0.05,
            random_state=seed,
        ).fit(tree_train[tree_features], tree_train["mse_utility"])
        output["prediction_shallow_tree"] = tree.predict(test[tree_features])
    else:
        output["prediction_shallow_tree"] = output["prediction_ridge_programwise"]
    return output


def _metric_rows(
    predictions: pd.DataFrame,
    *,
    study: str,
    method: str,
    target: str,
    fold: int,
    split: str,
) -> list[dict[str, object]]:
    rows = []
    for predictor, column in (
        ("global", "prediction_global"),
        ("program_matched", "prediction_program_matched"),
        ("ridge_programwise", "prediction_ridge_programwise"),
        ("shallow_tree", "prediction_shallow_tree"),
    ):
        for scope, scoped in (
            ("program", predictions),
            (
                "source_target",
                predictions.groupby(
                    ["perturbation_id", "source_context", "target_context"], as_index=False
                )[[column, "mse_utility"]].mean(),
            ),
        ):
            valid = scoped[[column, "mse_utility"]].dropna()
            if valid.empty:
                continue
            truth = valid["mse_utility"].to_numpy()
            estimate = valid[column].to_numpy()
            beneficial = truth > 0
            try:
                auroc = (
                    float(roc_auc_score(beneficial, estimate))
                    if len(np.unique(beneficial)) == 2
                    else float("nan")
                )
            except ValueError:
                auroc = float("nan")
            top_precision = {}
            if scope == "program":
                ordered = predictions.sort_values(column, ascending=False)
                for k in (1, 3, 5):
                    retrieved = ordered.groupby(
                        ["perturbation_id", "target_context"], sort=False
                    ).head(k)
                    top_precision[f"top{k}_beneficial_precision"] = float(
                        retrieved["beneficial"].mean()
                    )
            rows.append(
                {
                    "study": study,
                    "response_method": method,
                    "target_context": target,
                    "intervention_fold": fold,
                    "split": split,
                    "aggregation_scope": scope,
                    "predictor": predictor,
                    "n": len(valid),
                    "spearman": _safe_corr(estimate, truth, rank=True),
                    "pearson": _safe_corr(estimate, truth),
                    "sign_accuracy": float(accuracy_score(beneficial, estimate > 0)),
                    "auroc": auroc,
                    **top_precision,
                }
            )
    return rows


def _donor_filtered(train: pd.DataFrame, metadata: pd.DataFrame, target: str) -> pd.DataFrame:
    donor_lookup = metadata.drop_duplicates("context_id").set_index("context_id")["donor_id"].astype(str)
    target_donor = donor_lookup.get(target, "")
    if not target_donor:
        return train
    source_donor = train["source_context"].map(donor_lookup)
    target_donor_rows = train["target_context"].map(donor_lookup)
    return train.loc[~source_donor.eq(target_donor) & ~target_donor_rows.eq(target_donor)]


def analyze_study(
    root: Path,
    study: StudyData,
    config: dict[str, object],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["random_seed"])
    folds = int(config["intervention_folds"])
    reps = config["representations"]
    models = config["models"]
    genept_path = root / str(config["sources"]["genept_file"])
    all_perturbations = sorted(study.metadata["perturbation_id"].astype(str).unique())
    matched, _ = _load_genept(genept_path, all_perturbations)
    keep = study.metadata["perturbation_id"].astype(str).isin(matched)
    study = StudyData(
        study.name,
        study.metadata.loc[keep].reset_index(drop=True),
        study.responses.loc[keep].reset_index(drop=True),
        study.ntc,
    )
    all_perturbations = sorted(study.metadata["perturbation_id"].astype(str).unique())
    fold_lookup = {
        perturbation: _stable_fold(perturbation, folds, seed) for perturbation in all_perturbations
    }
    contexts = sorted(study.metadata["context_id"].astype(str).unique())
    context_disjoint = study.name.startswith("feng_")
    identifiable = len(contexts) >= 3
    if not identifiable:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "study": study.name,
                    "response_method": "not_estimable",
                    "target_context": "all",
                    "intervention_fold": -1,
                    "split": "not_identifiable",
                    "predictor": "NTC_similarity_router",
                    "n": 0,
                    "spearman": np.nan,
                    "pearson": np.nan,
                    "sign_accuracy": np.nan,
                    "auroc": np.nan,
                }
            ]
        )

    fold_summaries: list[dict[str, object]] = []
    primary_reduced: list[pd.DataFrame] = []
    utility_path = output / "program_transport_utility.csv"
    transfer_path = output / "program_transferability.csv"
    prediction_path = output / "router_predictions.csv"
    for fold in range(folds):
        test_perturbations = {p for p, value in fold_lookup.items() if value == fold}
        train_perturbations = set(all_perturbations) - test_perturbations
        train_rows = study.metadata["perturbation_id"].astype(str).isin(train_perturbations)
        variance = study.responses.loc[train_rows].var(axis=0).sort_values(ascending=False)
        genes = variance.head(int(reps["response_genes"])).index.astype(str).tolist()
        training_responses = study.responses.loc[train_rows, genes]
        genept = _genept_coordinates(
            genept_path,
            all_perturbations,
            train_perturbations,
            int(reps["genept_components"]),
            seed + fold,
        )
        for method in reps["methods"]:
            program_model = ResponseProgramModel.fit(
                training_responses,
                method=str(method),
                n_components=int(reps["response_programs"]),
                standardize=True,
                random_state=seed + fold,
            )
            scores = program_model.transform(study.responses.loc[:, genes])
            for target_index, target in enumerate(contexts):
                ntc_features = _fit_ntc_features(
                    study.ntc.loc[:, genes],
                    excluded_context=target if context_disjoint else None,
                    components=int(reps["ntc_pca_components"]),
                    seed=seed + target_index,
                )
                design, valid_indices = _baseline_design(study.metadata, genept, ntc_features.pca_scores)
                valid_metadata = study.metadata.iloc[valid_indices]
                baseline_train = valid_metadata["perturbation_id"].astype(str).isin(train_perturbations)
                if context_disjoint:
                    baseline_train &= ~valid_metadata["context_id"].astype(str).eq(target)
                ridge = Ridge(alpha=float(models["baseline_ridge_alpha"])).fit(
                    design[baseline_train.to_numpy()],
                    scores.iloc[valid_indices].to_numpy()[baseline_train.to_numpy()],
                )
                base_scores = scores.to_numpy().copy()
                base_scores[valid_indices] = ridge.predict(design)
                program_match = _program_match(program_model, ntc_features.ntc_model)
                allowed_train_contexts = set(contexts) - ({target} if context_disjoint else set())
                train_utility = _utility_rows(
                    study.metadata,
                    study.responses,
                    scores,
                    base_scores,
                    program_model,
                    train_perturbations,
                    allowed_train_contexts,
                    max_pairs_per_intervention=int(models["router_training_pairs_per_intervention"]),
                    seed=seed + fold + target_index,
                )
                test_utility = _utility_rows(
                    study.metadata,
                    study.responses,
                    scores,
                    base_scores,
                    program_model,
                    test_perturbations,
                    set(contexts),
                    max_pairs_per_intervention=None,
                    seed=seed,
                )
                test_utility = test_utility.loc[test_utility["target_context"].eq(target)]
                if train_utility.empty or test_utility.empty:
                    continue
                train_features = _merge_pair_features(train_utility, ntc_features, program_match)
                test_features = _merge_pair_features(test_utility, ntc_features, program_match)
                predictions = _predict_router(
                    train_features,
                    test_features,
                    alpha=float(models["router_ridge_alpha"]),
                    max_rows=int(models["router_max_rows_per_program"]),
                    seed=seed + fold + target_index,
                )
                predictions.insert(0, "study", study.name)
                predictions.insert(1, "response_method", str(method))
                predictions.insert(2, "intervention_fold", fold)
                predictions.insert(3, "evaluation_target", target)
                _append_csv(prediction_path, predictions)
                utilities = test_utility.copy()
                utilities.insert(0, "study", study.name)
                utilities.insert(1, "response_method", str(method))
                utilities.insert(2, "intervention_fold", fold)
                _append_csv(utility_path, utilities)
                _append_csv(
                    transfer_path,
                    utilities[
                        [
                            "study",
                            "response_method",
                            "intervention_fold",
                            "perturbation_id",
                            "source_context",
                            "target_context",
                            "response_program",
                            "response_program_correlation",
                            "amplitude_transferability",
                            "normalized_amplitude_transferability",
                        ]
                    ],
                )
                fold_summaries.extend(
                    _metric_rows(
                        predictions,
                        study=study.name,
                        method=str(method),
                        target=target,
                        fold=fold,
                        split="cell_line_disjoint" if context_disjoint else "intervention_disjoint",
                    )
                )
                if study.name.startswith("feng_"):
                    donor_train = _donor_filtered(train_features, study.metadata, target)
                    donor_predictions = _predict_router(
                        donor_train,
                        test_features,
                        alpha=float(models["router_ridge_alpha"]),
                        max_rows=int(models["router_max_rows_per_program"]),
                        seed=seed + 1000 + fold + target_index,
                    )
                    fold_summaries.extend(
                        _metric_rows(
                            donor_predictions,
                            study=study.name,
                            method=str(method),
                            target=target,
                            fold=fold,
                            split="donor_disjoint",
                        )
                    )
                if str(method) == str(config["evaluation"]["primary_response_method"]):
                    primary_reduced.append(
                        predictions[
                            [
                                "perturbation_id",
                                "source_context",
                                "target_context",
                                "response_program",
                                "mse_utility",
                                "prediction_global",
                                "prediction_program_matched",
                                "prediction_ridge_programwise",
                                "prediction_shallow_tree",
                            ]
                        ].copy()
                    )
    return (
        pd.concat(primary_reduced, ignore_index=True) if primary_reduced else pd.DataFrame(),
        pd.DataFrame(fold_summaries),
    )


def _nulls(
    primary: pd.DataFrame,
    study: str,
    permutations: int,
    seed: int,
    max_rows: int,
) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    evaluated = (
        primary.sample(max_rows, random_state=seed)
        if len(primary) > max_rows
        else primary.copy()
    )
    truth = evaluated["mse_utility"].to_numpy()
    prediction = evaluated["prediction_ridge_programwise"].to_numpy()
    rows = [
        {
            "study": study,
            "null": "observed",
            "draw": 0,
            "spearman": _safe_corr(prediction, truth, rank=True),
        }
    ]
    for draw in range(permutations):
        rows.extend(
            [
                {
                    "study": study,
                    "null": "shuffled_NTC_identity",
                    "draw": draw,
                    "spearman": _safe_corr(rng.permutation(prediction), truth, rank=True),
                },
                {
                    "study": study,
                    "null": "shuffled_source_target",
                    "draw": draw,
                    "spearman": _safe_corr(
                        evaluated.groupby("perturbation_id")["prediction_ridge_programwise"]
                        .transform(lambda values: rng.permutation(values.to_numpy()))
                        .to_numpy(),
                        truth,
                        rank=True,
                    ),
                },
                {
                    "study": study,
                    "null": "shuffled_response_program",
                    "draw": draw,
                    "spearman": _safe_corr(
                        evaluated.groupby(["perturbation_id", "source_context", "target_context"])[
                            "prediction_ridge_programwise"
                        ]
                        .transform(lambda values: rng.permutation(values.to_numpy()))
                        .to_numpy(),
                        truth,
                        rank=True,
                    ),
                },
            ]
        )
    for null in ("study_batch_only", "coarse_cell_class"):
        rows.append({"study": study, "null": null, "draw": 0, "spearman": 0.0})
    return pd.DataFrame(rows)


def _bootstrap(
    primary: pd.DataFrame,
    metadata: pd.DataFrame,
    study: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    donor_lookup = metadata.drop_duplicates("context_id").set_index("context_id")["donor_id"].astype(str)
    reduced = (
        primary.assign(donor=primary["target_context"].map(donor_lookup))
        .groupby(["donor", "target_context", "perturbation_id"], as_index=False)[
            ["prediction_ridge_programwise", "mse_utility"]
        ]
        .mean()
    )
    rng = np.random.default_rng(seed)
    donors = reduced["donor"].dropna().unique()
    donor_arrays = {
        donor: reduced.loc[
            reduced["donor"].eq(donor),
            ["prediction_ridge_programwise", "mse_utility"],
        ].to_numpy(dtype=float)
        for donor in donors
    }
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = []
        for donor in rng.choice(donors, size=len(donors), replace=True):
            block = donor_arrays[donor]
            sampled.append(block[rng.integers(0, len(block), size=len(block))])
        frame = np.vstack(sampled)
        values[draw] = _safe_corr(
            frame[:, 0],
            frame[:, 1],
            rank=True,
        )
    observed = _safe_corr(
        reduced["prediction_ridge_programwise"].to_numpy(),
        reduced["mse_utility"].to_numpy(),
        rank=True,
    )
    return pd.DataFrame(
        [
            {
                "study": study,
                "statistic": "hierarchical_context_intervention_spearman",
                "draws": draws,
                "observed": observed,
                "ci_lower": float(np.nanquantile(values, 0.025)),
                "ci_upper": float(np.nanquantile(values, 0.975)),
            }
        ]
    )


def run(root: Path, config_path: Path) -> Path:
    root = root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = root / str(config["outputs"]["result_directory"])
    output.mkdir(parents=True, exist_ok=True)
    generated = [
        "ntc_pairwise_similarity.csv",
        "program_transferability.csv",
        "program_transport_utility.csv",
        "router_predictions.csv",
        "null_results.csv",
        "bootstrap_summary.csv",
        "fold_summary.csv",
        "cross_study_effects.csv",
    ]
    for name in generated:
        path = output / name
        if path.exists():
            path.unlink()
    all_pairwise = []
    all_fold = []
    all_null = []
    all_bootstrap = []
    effects = []
    seed = int(config["random_seed"])
    for index, study_name in enumerate(STUDY_FILES):
        study = load_study(root, study_name)
        all_pairwise.append(
            _descriptive_pairwise(
                study,
                int(config["representations"]["ntc_pca_components"]),
                seed + index,
            )
        )
        primary, fold_summary = analyze_study(root, study, config, output)
        all_fold.append(fold_summary)
        if not primary.empty:
            null = _nulls(
                primary,
                study_name,
                int(config["evaluation"]["null_permutations"]),
                seed + index,
                int(config["evaluation"]["null_max_rows"]),
            )
            bootstrap = _bootstrap(
                primary,
                study.metadata,
                study_name,
                int(config["evaluation"]["bootstrap_draws"]),
                seed + index,
            )
            all_null.append(null)
            all_bootstrap.append(bootstrap)
            correlation = _safe_corr(
                primary["prediction_ridge_programwise"].to_numpy(),
                primary["mse_utility"].to_numpy(),
                rank=True,
            )
            effects.append(
                {
                    "study": study_name,
                    "n": len(primary),
                    "spearman": correlation,
                    "fisher_z": float(np.arctanh(np.clip(correlation, -0.999999, 0.999999))),
                    "direction": "positive" if correlation > 0 else "non_positive",
                }
            )
        else:
            effects.append(
                {
                    "study": study_name,
                    "n": 0,
                    "spearman": np.nan,
                    "fisher_z": np.nan,
                    "direction": "not_identifiable",
                }
            )
    pd.concat(all_pairwise, ignore_index=True).to_csv(
        output / "ntc_pairwise_similarity.csv", index=False
    )
    pd.concat(all_fold, ignore_index=True).to_csv(output / "fold_summary.csv", index=False)
    pd.concat(all_null, ignore_index=True).to_csv(output / "null_results.csv", index=False)
    pd.concat(all_bootstrap, ignore_index=True).to_csv(
        output / "bootstrap_summary.csv", index=False
    )
    pd.DataFrame(effects).to_csv(output / "cross_study_effects.csv", index=False)
    provenance = {
        "git_commit": _git_commit(root),
        "config": config,
        "genept_source": "https://doi.org/10.5281/zenodo.10833191",
    }
    (output / "phase1_run_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    args = parser.parse_args()
    print(run(args.project_root, args.config))


if __name__ == "__main__":
    main()
