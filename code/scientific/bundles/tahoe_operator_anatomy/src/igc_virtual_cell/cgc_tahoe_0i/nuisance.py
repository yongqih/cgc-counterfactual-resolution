"""Frozen nuisance transformations for G_PRIMARY maximal-support robustness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from sklearn.utils.extmath import randomized_svd

from igc_virtual_cell.cgc_tahoe_0i.scaling import _batched_affine, _folds, _support_orders
from igc_virtual_cell.cgc_tahoe_0i.truth import interaction


TRANSFORMS = (
    "identity",
    "context_scalar",
    "intervention_potency",
    "double_scalar",
    "remove_pc1",
    "remove_pc3",
    "remove_pc5",
    "cell_count_library_residualization",
    "unit_norm_direction",
)
NULL_DRAWS = 5_000
NULL_SEED = 2_609
PC_SEED = 2_610


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rms(values: np.ndarray, axis: tuple[int, ...], keepdims: bool = False) -> np.ndarray:
    return np.sqrt(np.maximum(np.mean(values.astype(np.float64) ** 2, axis=axis, keepdims=keepdims), 1e-12))


def _global_transform(
    transform: str,
    train: np.ndarray,
    evaluation: np.ndarray,
    fold: int,
    cov_train: np.ndarray | None,
    cov_eval: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if transform == "identity" or transform in {"intervention_potency", "double_scalar"}:
        return train.copy(), evaluation.copy(), {"fit": "identity_or_target_specific"}
    if transform == "context_scalar":
        scale = _rms(train, axis=(2, 3), keepdims=True)
        return (train / scale).astype(np.float32), (evaluation / scale).astype(np.float32), {"fit": "training_interventions_context_rms"}
    if transform.startswith("remove_pc"):
        rank = int(transform.removeprefix("remove_pc"))
        matrix = train.reshape(-1, train.shape[-1]).astype(np.float64)
        center = matrix.mean(axis=0)
        _, _, basis = randomized_svd(
            matrix - center,
            n_components=rank,
            n_iter=3,
            random_state=PC_SEED + fold,
        )
        def remove(values: np.ndarray) -> np.ndarray:
            flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
            centered = flat - center
            return (centered - (centered @ basis.T) @ basis).reshape(values.shape).astype(np.float32)
        return remove(train), remove(evaluation), {"fit": "training_interventions_global_gene_PC", "rank": rank}
    if transform == "cell_count_library_residualization":
        if cov_train is None or cov_eval is None:
            raise RuntimeError("Technical covariates unavailable")
        transformed_train = np.empty_like(train)
        transformed_eval = np.empty_like(evaluation)
        for plate in range(2):
            x_train = cov_train[plate].reshape(-1, cov_train.shape[-1]).astype(np.float64)
            x_eval = cov_eval[plate].reshape(-1, cov_eval.shape[-1]).astype(np.float64)
            mean = x_train.mean(axis=0)
            std = np.maximum(x_train.std(axis=0), 1e-8)
            xs = (x_train - mean) / std
            xe = (x_eval - mean) / std
            design = np.column_stack([np.ones(len(xs)), xs])
            beta = np.linalg.pinv(design.T @ design, rcond=1e-10) @ design.T @ train[plate].reshape(len(xs), -1)
            # Preserve the fitted intercept; remove only technical variation.
            transformed_train[plate] = (train[plate].reshape(len(xs), -1) - xs @ beta[1:]).reshape(train[plate].shape)
            transformed_eval[plate] = (evaluation[plate].reshape(len(xe), -1) - xe @ beta[1:]).reshape(evaluation[plate].shape)
        return transformed_train, transformed_eval, {"fit": "training_interventions_multivariate_log_count_library_regression"}
    if transform == "unit_norm_direction":
        return (
            (train / _rms(train, axis=(3,), keepdims=True)).astype(np.float32),
            (evaluation / _rms(evaluation, axis=(3,), keepdims=True)).astype(np.float32),
            {"fit": "parameter_free_per_response_direction"},
        )
    raise ValueError(transform)


def _target_transform(
    transform: str,
    train: np.ndarray,
    evaluation: np.ndarray,
    target: int,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if transform == "intervention_potency":
        train_scale = _rms(train[:, support], axis=(1, 3))[:, None, :, None]
        eval_scale = _rms(evaluation[:, support], axis=(1, 3))[:, None, :, None]
        return (train / train_scale).astype(np.float32), (evaluation / eval_scale).astype(np.float32), {"fit": "support_context_only_intervention_RMS"}
    if transform == "double_scalar":
        train_norm = _rms(train, axis=(3,))
        eval_norm = _rms(evaluation, axis=(3,))
        output_train = np.empty_like(train)
        output_eval = np.empty_like(evaluation)
        for plate in range(2):
            log_train = np.log(train_norm[plate])
            grand = float(log_train[np.ix_(support, np.arange(log_train.shape[1]))].mean())
            context = log_train.mean(axis=1) - grand
            potency_train = log_train[support].mean(axis=0) - grand
            log_eval = np.log(eval_norm[plate])
            potency_eval = log_eval[support].mean(axis=0) - grand
            scale_train = np.exp(grand + context[:, None] + potency_train[None, :])[:, :, None]
            scale_eval = np.exp(grand + context[:, None] + potency_eval[None, :])[:, :, None]
            output_train[plate] = train[plate] / scale_train
            output_eval[plate] = evaluation[plate] / scale_eval
        return output_train, output_eval, {"fit": "support-only_context_plus_intervention_log_RMS"}
    return train, evaluation, {"fit": "global"}


def _technical_covariates(metadata: pd.DataFrame, contexts: list[str], interventions: list[str]) -> np.ndarray:
    treatment = metadata[~metadata["is_control"].astype(bool)].copy()
    control = metadata[metadata["is_control"].astype(bool)].groupby(["plate", "cell_line_id"], as_index=False).agg(
        control_cells=("n_cells", "sum"), control_library=("library_size", "sum")
    )
    treatment = treatment.merge(control, on=["plate", "cell_line_id"], validate="many_to_one")
    plate_map = {"plate6": 0, "plate14": 1}
    context_map = {value: index for index, value in enumerate(contexts)}
    intervention_map = {value: index for index, value in enumerate(interventions)}
    values = np.empty((2, 50, 93, 4), dtype=np.float64)
    for row in treatment.itertuples(index=False):
        values[plate_map[row.plate], context_map[row.cell_line_id], intervention_map[row.intervention_id]] = np.log1p(
            [row.n_cells, row.library_size, row.control_cells, row.control_library]
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Technical covariate tensor incomplete")
    return values


def _evaluate(
    transform: str,
    delta: np.ndarray,
    folds: np.ndarray,
    orders: np.ndarray,
    covariates: np.ndarray,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    base_total = residual_total = base_observed = residual_observed = 0.0
    oof = np.full_like(delta, np.nan, dtype=np.float32)
    null_matrices: list[np.ndarray] = []
    transform_states = []
    for fold in range(5):
        train_indices = np.flatnonzero(folds != fold)
        eval_indices = np.flatnonzero(folds == fold)
        raw_train, raw_eval = delta[:, :, train_indices], delta[:, :, eval_indices]
        global_train, global_eval, global_state = _global_transform(
            transform,
            raw_train,
            raw_eval,
            fold,
            covariates[:, :, train_indices],
            covariates[:, :, eval_indices],
        )
        fold_null = np.zeros((len(eval_indices), len(eval_indices)), dtype=np.float64)
        direction_residual6 = [np.empty((50, len(eval_indices), delta.shape[-1]), dtype=np.float32) for _ in range(2)]
        direction_residual14 = [np.empty_like(direction_residual6[0]) for _ in range(2)]
        for target in range(50):
            support = orders[target, 0, :49]
            train, evaluation, local_state = _target_transform(
                transform, global_train, global_eval, target, support
            )
            oof[:, target, eval_indices] = evaluation[:, target]
            qtrain = train - train.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
            qeval = evaluation - evaluation.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
            grams = np.einsum("rcpg,rdpg->rcd", qtrain, qtrain, dtype=np.float64, optimize=True)
            target6, target14 = qeval[0, target], qeval[1, target]
            mean6, mean14 = qeval[0, support].mean(axis=0), qeval[1, support].mean(axis=0)
            b6, b14 = target6 - mean6, target14 - mean14
            base_total += 2 * float(np.sum(b6.astype(np.float64) * b14.astype(np.float64)))
            base_observed += 2 * float(np.sum((b6.astype(np.float64) ** 2 + b14.astype(np.float64) ** 2) / 2))
            for direction in range(2):
                gram = grams[direction]
                weight = _batched_affine(
                    gram[np.ix_(support, support)][None], gram[support, target][None]
                )[0]
                prediction6 = np.einsum("c,cpg->pg", weight, qeval[0, support], optimize=True)
                prediction14 = np.einsum("c,cpg->pg", weight, qeval[1, support], optimize=True)
                r6, r14 = target6 - prediction6, target14 - prediction14
                direction_residual6[direction][target] = r6
                direction_residual14[direction][target] = r14
                residual_total += float(np.sum(r6.astype(np.float64) * r14.astype(np.float64)))
                residual_observed += float(np.sum((r6.astype(np.float64) ** 2 + r14.astype(np.float64) ** 2) / 2))
            if target == 0:
                transform_states.append({"fold": fold, **global_state, **local_state})
        for direction in range(2):
            fold_null += np.einsum(
                "cpg,cqg->pq",
                direction_residual6[direction],
                direction_residual14[direction],
                dtype=np.float64,
                optimize=True,
            )
        null_matrices.append(fold_null)
        print(f"nuisance {transform}: fold {fold + 1}/5", flush=True)
    gamma = interaction(oof.astype(np.float64))
    truth_signal = float(np.mean(gamma[0] * gamma[1]))
    truth_observed = float(np.mean((gamma[0] ** 2 + gamma[1] ** 2) / 2))
    rng = np.random.default_rng(NULL_SEED)
    null = np.empty(NULL_DRAWS, dtype=np.float64)
    denominator = 50 * 2 * 93 * delta.shape[-1]
    for draw in range(NULL_DRAWS):
        total = 0.0
        for matrix in null_matrices:
            total += matrix[np.arange(len(matrix)), rng.permutation(len(matrix))].sum()
        null[draw] = total / denominator
    residual_energy = residual_total / denominator
    result = {
        "transformation": transform,
        "truth_signal": truth_signal,
        "truth_observed": truth_observed,
        "truth_signal_fraction": truth_signal / truth_observed,
        "q49": residual_total / base_total,
        "raw_r2": 1 - residual_observed / base_observed,
        "residual_energy": residual_energy,
        "residual_null_q95": float(np.quantile(null, 0.95)),
        "residual_null_p": float((1 + np.count_nonzero(null >= residual_energy)) / (1 + NULL_DRAWS)),
        "parameters_fit_on_training_interventions_only": transform not in {"intervention_potency", "double_scalar", "unit_norm_direction"},
        "heldout_context_excluded_from_eval_intervention_scalars": transform in {"intervention_potency", "double_scalar"},
        "parameter_free_per_response": transform == "unit_norm_direction",
        "states": json.dumps(transform_states),
    }
    return result, null_matrices


def run_nuisance(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    group = zarr.open_group(out / "response_tensors.zarr", mode="r")
    indices = np.load(out / "gene_indices_g_primary.npy")
    delta = np.asarray(
        group["delta_primary"].get_orthogonal_selection((slice(None), slice(None), slice(None), indices)),
        dtype=np.float32,
    )
    contexts, interventions = list(group.attrs["contexts"]), list(group.attrs["interventions"])
    folds = _folds(root, interventions)
    orders = _support_orders(root, contexts)
    metadata = pd.read_csv(out / "pseudobulk_sample_metadata.csv")
    covariates = _technical_covariates(metadata, contexts, interventions)
    rows = []
    for transform in TRANSFORMS:
        result, _ = _evaluate(transform, delta, folds, orders, covariates)
        rows.append(result)
    table = pd.DataFrame(rows)
    table.to_csv(out / "nuisance_robustness.csv", index=False)
    double = table.set_index("transformation").loc["double_scalar"]
    result = {
        "transformations": len(table),
        "double_scalar_q49": float(double["q49"]),
        "double_scalar_truth_signal": float(double["truth_signal"]),
        "double_scalar_residual_null_p": float(double["residual_null_p"]),
        "all_required_transformations_complete": set(table["transformation"]) == set(TRANSFORMS),
        "pc_parameters_training_only": True,
        "technical_residualization_training_only": True,
        "unit_norm_is_parameter_free_descriptive_sensitivity": True,
    }
    _write_json(out / "nuisance_summary.json", result)
    return result

