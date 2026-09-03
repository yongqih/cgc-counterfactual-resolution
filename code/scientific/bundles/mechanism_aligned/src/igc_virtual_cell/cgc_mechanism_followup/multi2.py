"""Frozen CGC-MULTI-2 mechanism-aligned susceptibility analysis.

This module is deliberately data-source agnostic.  It accepts arrays whose axes
have already been checked against the frozen protocol and never discovers a
feature, context, intervention, rank, or annotation from a response outcome.
The real-data loader/reporting entry point is kept separate so the numerical
core can be tested with sealed synthetic outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEED = 8201
EPS = 1e-12
RESPONSE_RANKS = (8, 16, 32, 64)
GENERIC_RANKS = (8, 16)
RNA_RANK = 16
INTERVENTION_EMBED_RANK = 8
RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
INNER_FOLDS = 5
NULL_DRAWS = 10_000
POWER_REPLICATES = 200
POWER_LEVELS = (0.02, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class CompleteCaseMatrix:
    """A raw modality restricted to outcome-independent, fully observed columns."""

    name: str
    context_axes: np.ndarray
    feature_indices: np.ndarray
    feature_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.shape != (len(self.context_axes), len(self.feature_names)):
            raise ValueError("complete-case values must be context x selected-feature")
        if len(self.feature_indices) != values.shape[1]:
            raise ValueError("feature index count does not match complete-case values")
        if not np.isfinite(values).all():
            raise ValueError("complete-case matrix contains a missing or non-finite value")
        if len(np.unique(self.context_axes)) != len(self.context_axes):
            raise ValueError("context axes must be unique")


@dataclass(frozen=True)
class AlignedBlock:
    """Low-dimensional Z[c,p] with no missing value encoded as biological zero."""

    name: str
    context_axes: np.ndarray
    intervention_axes: np.ndarray
    feature_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.context_axes), len(self.intervention_axes), len(self.feature_names))
        if np.asarray(self.values).shape != expected:
            raise ValueError(f"aligned block must have shape {expected}")
        if not np.isfinite(self.values).all():
            raise ValueError("eligible aligned block must be finite; filter ineligible interventions first")
        if len(np.unique(self.context_axes)) != len(self.context_axes):
            raise ValueError("context axes must be unique")
        if len(np.unique(self.intervention_axes)) != len(self.intervention_axes):
            raise ValueError("intervention axes must be unique")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_rna: bool
    use_generic: bool
    use_aligned: bool


DEFAULT_MODEL_SPECS = (
    ModelSpec("M0_RNA", True, False, False),
    ModelSpec("M1_GENERIC", False, True, False),
    ModelSpec("M2_ALIGNED", False, False, True),
    ModelSpec("M3_RNA_GENERIC", True, True, False),
    ModelSpec("M4_RNA_ALIGNED", True, False, True),
)


@dataclass
class FoldPredictions:
    predictions: dict[tuple[str, str, int], np.ndarray]
    fits: pd.DataFrame
    prepared: dict[int, dict[str, object]] | None = None


@dataclass
class OOFResult:
    blocks: pd.DataFrame
    fits: pd.DataFrame


@dataclass(frozen=True)
class PowerFold:
    held_context: int
    train_contexts: np.ndarray
    source_plate: int
    x_base_train: np.ndarray
    x_base_test: np.ndarray
    x_aug_train: np.ndarray
    x_aug_test: np.ndarray
    intervention_axes: np.ndarray


@dataclass
class PowerResult:
    curves: pd.DataFrame
    replicates: pd.DataFrame
    detection_limit: str | float


@dataclass(frozen=True)
class FoldLocalPowerFold:
    """Direction-local sufficient statistics for exact coefficient-space power.

    Every response-dependent field is fitted without the held context.  The
    affine response coordinates and the omitted orthogonal gene-space energy
    together reconstruct the same cross-product numerator and denominator that
    the real gene-space scorer uses.
    """

    held_context: int
    train_contexts: np.ndarray
    source_plate: int
    intervention_axes: np.ndarray
    basis: np.ndarray
    x_base_train: np.ndarray
    x_base_test: np.ndarray
    x_aug_train: np.ndarray
    x_aug_test: np.ndarray
    memory_coeff: np.ndarray
    affine_coeff: np.ndarray
    signal_coeff: np.ndarray
    residual_cross: np.ndarray
    residual_norm_source: np.ndarray
    memory_orthogonal_norm: np.ndarray
    memory_orthogonal_dot_affine_sum: np.ndarray


def select_complete_features(
    raw_values: np.ndarray,
    candidate_context_axes: Sequence[int],
    feature_names: Sequence[str] | None = None,
    *,
    name: str = "modality",
) -> CompleteCaseMatrix:
    """Select columns observed in every predeclared context without outcomes.

    ``candidate_context_axes`` address rows in ``raw_values``.  Rows are never
    admitted because another feature happens to be present, which was the
    historical MULTI-1 semantic error.
    """

    raw = np.asarray(raw_values, dtype=np.float64)
    contexts = np.asarray(candidate_context_axes, dtype=np.int64)
    if raw.ndim != 2:
        raise ValueError("raw modality must be a two-dimensional context x feature matrix")
    if not len(contexts) or contexts.min() < 0 or contexts.max() >= raw.shape[0]:
        raise ValueError("candidate context axes are empty or out of range")
    if len(np.unique(contexts)) != len(contexts):
        raise ValueError("candidate context axes must be unique")
    names = tuple(str(x) for x in (feature_names or [f"feature_{i}" for i in range(raw.shape[1])]))
    if len(names) != raw.shape[1]:
        raise ValueError("feature name count does not match raw matrix")
    selected = np.flatnonzero(np.isfinite(raw[contexts]).all(axis=0))
    if not len(selected):
        raise ValueError(f"{name} has no fully observed feature on the frozen context panel")
    return CompleteCaseMatrix(
        name=name,
        context_axes=contexts.copy(),
        feature_indices=selected,
        feature_names=tuple(names[i] for i in selected),
        values=raw[np.ix_(contexts, selected)].copy(),
    )


def _feature_matches(label: str, member: str, match: str) -> bool:
    left, right = str(label).upper(), str(member).upper()
    if match == "exact":
        return left == right
    if match == "protein_label":
        return left == right or left.endswith(f" ({right})")
    if match == "rppa_target":
        return left == right or left.startswith(right + "P")
    raise ValueError(f"unsupported feature matching rule: {match}")


def aggregate_mapped_features(
    complete: CompleteCaseMatrix,
    intervention_members: Sequence[Iterable[str]],
    intervention_axes: Sequence[int] | None = None,
    *,
    aggregation: str = "mean",
    match: str = "exact",
    name: str = "aligned",
) -> tuple[AlignedBlock, np.ndarray]:
    """Aggregate a frozen target/pathway mapping and return its eligibility mask.

    Interventions without a mapped, fully observed feature are excluded from
    the returned block.  The full-length Boolean mask makes that exclusion
    auditable; no unavailable mapping is represented by a numeric zero.
    """

    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be the frozen mean or max sensitivity")
    members = [tuple(str(x) for x in values if str(x).strip()) for values in intervention_members]
    axes = np.arange(len(members), dtype=np.int64) if intervention_axes is None else np.asarray(intervention_axes, dtype=np.int64)
    if len(axes) != len(members):
        raise ValueError("intervention axis count does not match mapping count")
    eligible = np.zeros(len(members), dtype=bool)
    columns: list[np.ndarray] = []
    kept_axes: list[int] = []
    for local, mapped in enumerate(members):
        hit = sorted({i for i, label in enumerate(complete.feature_names) for member in mapped if _feature_matches(label, member, match)})
        if not hit:
            continue
        values = complete.values[:, hit]
        aggregate = values.mean(axis=1) if aggregation == "mean" else values.max(axis=1)
        if not np.isfinite(aggregate).all():
            raise AssertionError("complete-feature aggregation unexpectedly produced missing values")
        eligible[local] = True
        columns.append(aggregate)
        kept_axes.append(int(axes[local]))
    if not columns:
        raise ValueError(f"{name} has no intervention with a measured frozen mapping")
    array = np.stack(columns, axis=1)[:, :, None]
    return (
        AlignedBlock(
            name=name,
            context_axes=complete.context_axes.copy(),
            intervention_axes=np.asarray(kept_axes, dtype=np.int64),
            feature_names=(name,),
            values=array,
        ),
        eligible,
    )


def intersect_aligned_blocks(blocks: Sequence[AlignedBlock], *, name: str) -> AlignedBlock:
    """Stack frozen aligned axes on the exact common context/intervention panel."""

    if not blocks:
        raise ValueError("at least one aligned block is required")
    contexts = blocks[0].context_axes
    for block in blocks[1:]:
        if not np.array_equal(block.context_axes, contexts):
            raise ValueError("aligned blocks use different context panels")
    interventions = blocks[0].intervention_axes.copy()
    for block in blocks[1:]:
        interventions = np.intersect1d(interventions, block.intervention_axes, assume_unique=True)
    if not len(interventions):
        raise ValueError("aligned blocks have no common eligible intervention")
    arrays, names = [], []
    for block in blocks:
        loc = {int(axis): i for i, axis in enumerate(block.intervention_axes)}
        arrays.append(block.values[:, [loc[int(axis)] for axis in interventions], :])
        names.extend(block.feature_names)
    return AlignedBlock(name, contexts.copy(), interventions, tuple(names), np.concatenate(arrays, axis=2))


def _dual_context_pca(
    train: np.ndarray,
    test: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-local standardization and deterministic context-dual PCA."""

    xtr = np.asarray(train, dtype=np.float64)
    xte = np.asarray(test, dtype=np.float64)
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
        raise ValueError("MULTI-2 does not impute a missing generic feature")
    mean = xtr.mean(axis=0)
    scale = xtr.std(axis=0)
    keep = scale > 1e-10
    if not keep.any():
        return np.zeros((len(xtr), rank)), np.zeros((len(xte), rank))
    xtr = (xtr[:, keep] - mean[keep]) / scale[keep]
    xte = (xte[:, keep] - mean[keep]) / scale[keep]
    gram = xtr @ xtr.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    order = order[values[order] > 1e-10][: min(rank, xtr.shape[1], max(len(xtr) - 1, 0))]
    result_tr = np.zeros((len(xtr), rank), dtype=np.float64)
    result_te = np.zeros((len(xte), rank), dtype=np.float64)
    if len(order):
        components = xtr.T @ (vectors[:, order] / np.sqrt(values[order]))
        result_tr[:, : len(order)] = xtr @ components
        result_te[:, : len(order)] = xte @ components
    return result_tr, result_te


def _standardize_aligned(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = np.asarray(train, dtype=np.float64)
    te = np.asarray(test, dtype=np.float64)
    if tr.ndim != 3 or te.ndim != 3 or tr.shape[1:] != te.shape[1:]:
        raise ValueError("aligned feature arrays must be context x intervention x feature")
    if not np.isfinite(tr).all() or not np.isfinite(te).all():
        raise ValueError("ineligible aligned entries must be filtered, not imputed")
    flat = tr.reshape(-1, tr.shape[-1])
    mean, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-10] = 1.0
    return (tr - mean) / scale, (te - mean) / scale


def _randomized_response_basis(matrix: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Return a training-only orthonormal gene basis, nested up to ``rank``."""

    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("response adaptation matrix must be finite and two-dimensional")
    k = min(rank, min(x.shape))
    if k < 1:
        raise ValueError("response adaptation has no estimable component")
    # Exact SVD keeps small synthetic/invariance tests maximally transparent.
    if x.size <= 2_000_000 or k >= min(x.shape) // 2:
        return np.linalg.svd(x, full_matrices=False)[2][:k]
    rng = np.random.default_rng(seed)
    width = min(min(x.shape), k + 12)
    omega = rng.normal(size=(x.shape[1], width))
    q, _ = np.linalg.qr(x @ omega, mode="reduced")
    for _ in range(2):
        q, _ = np.linalg.qr(x @ (x.T @ q), mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ x, full_matrices=False)
    return vt[:k]


def _intervention_embedding(memory: np.ndarray, rank: int = INTERVENTION_EMBED_RANK) -> np.ndarray:
    x = np.asarray(memory, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("intervention memory must be a finite intervention x output matrix")
    centered = x - x.mean(axis=0, keepdims=True)
    # Only U*S is needed.  The intervention-dual Gram avoids materializing a
    # 93 x 25,695 right-singular matrix in every outer fold.
    values, u = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(values)[::-1]
    order = order[values[order] > 1e-12][:rank]
    result = np.zeros((len(x), rank), dtype=np.float64)
    if len(order):
        result[:, : len(order)] = u[:, order] * np.sqrt(values[order])
    return result


def build_design(
    context_blocks: Sequence[np.ndarray],
    aligned_blocks: Sequence[np.ndarray],
    intervention_embedding: np.ndarray,
    *,
    bilinear: bool,
) -> np.ndarray:
    """Build additive or compact bilinear context×intervention design.

    Every supplied context or aligned feature participates in the interaction
    block.  This explicitly prevents the historical first-eight-RNA-PC
    truncation from recurring.
    """

    embed = np.asarray(intervention_embedding, dtype=np.float64)
    if embed.ndim != 2:
        raise ValueError("intervention embedding must be intervention x component")
    if context_blocks:
        nctx = len(context_blocks[0])
    elif aligned_blocks:
        nctx = len(aligned_blocks[0])
    else:
        raise ValueError("model design has no context or aligned feature")
    npert = len(embed)
    pieces = [np.ones((nctx, npert, 1), dtype=np.float64)]
    interaction_inputs: list[np.ndarray] = []
    for block in context_blocks:
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != nctx:
            raise ValueError("context block must be context x feature")
        expanded = np.repeat(values[:, None, :], npert, axis=1)
        pieces.append(expanded)
        interaction_inputs.append(expanded)
    for block in aligned_blocks:
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 3 or values.shape[:2] != (nctx, npert):
            raise ValueError("aligned block must be context x intervention x feature")
        pieces.append(values)
        interaction_inputs.append(values)
    if bilinear:
        for values in interaction_inputs:
            pieces.append((values[:, :, :, None] * embed[None, :, None, :]).reshape(nctx, npert, -1))
    design = np.concatenate(pieces, axis=2).reshape(nctx * npert, -1)
    if not np.isfinite(design).all():
        raise ValueError("model design contains a non-finite value")
    return design


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = x.T @ x
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + penalty) @ rhs


def _select_response_rank_ridge(
    design: np.ndarray,
    adaptation: np.ndarray,
    basis: np.ndarray,
    response_ranks: Sequence[int],
    ridge_grid: Sequence[float],
    intervention_axes: np.ndarray,
) -> tuple[int, float, float]:
    """Fixed intervention-fold CV with full-gene-equivalent squared loss."""

    nctx, npert, ngene = adaptation.shape
    if design.shape[0] != nctx * npert or basis.shape[1] != ngene:
        raise ValueError("design, adaptation, and response basis axes disagree")
    flat = adaptation.reshape(-1, ngene)
    max_scores = flat @ basis.T
    truth_norm = np.einsum("ng,ng->n", flat, flat)
    return _select_response_rank_ridge_scores(
        design, max_scores, truth_norm, nctx, npert, response_ranks,
        ridge_grid, intervention_axes,
    )


def _select_response_rank_ridge_scores(
    design: np.ndarray,
    max_scores: np.ndarray,
    truth_norm: np.ndarray,
    nctx: int,
    npert: int,
    response_ranks: Sequence[int],
    ridge_grid: Sequence[float],
    intervention_axes: np.ndarray,
) -> tuple[int, float, float]:
    """Rank/ridge CV from one fold/plate projection reused by all models."""

    if design.shape[0] != nctx * npert or max_scores.shape[0] != nctx * npert:
        raise ValueError("design and cached response-score axes disagree")
    if truth_norm.shape != (nctx * npert,):
        raise ValueError("cached response norm axis disagrees")
    frozen_axes = np.asarray(intervention_axes, dtype=np.int64)
    if frozen_axes.shape != (npert,) or len(np.unique(frozen_axes)) != npert:
        raise ValueError("frozen intervention axes must uniquely label the fitted panel")
    paxis = np.tile(np.arange(npert), nctx)
    folds = frozen_axes % INNER_FOLDS
    candidates = [int(rank) for rank in response_ranks if rank <= max_scores.shape[1]]
    if not candidates:
        candidates = [max_scores.shape[1]]
    # Ridge is output-separable: one max-rank solve supplies every nested rank.
    # This prevents ranks {8,16,32,64} from multiplying the number of fits.
    losses = np.zeros((len(candidates), len(ridge_grid)), dtype=np.float64)
    for fold in range(INNER_FOLDS):
        train = folds[paxis] != fold
        valid = ~train
        if not train.any() or not valid.any():
            continue
        valid_scores = max_scores[valid]
        cumulative_truth = np.cumsum(np.square(valid_scores), axis=1)
        for ridge_axis, ridge in enumerate(ridge_grid):
            coefficient = _ridge_coefficients(design[train], max_scores[train], float(ridge))
            predicted = design[valid] @ coefficient
            cumulative_error = np.cumsum(np.square(valid_scores - predicted), axis=1)
            for rank_axis, rank in enumerate(candidates):
                projected_error = cumulative_error[:, rank - 1].sum()
                orthogonal = (truth_norm[valid] - cumulative_truth[:, rank - 1]).sum()
                losses[rank_axis, ridge_axis] += float(projected_error + max(orthogonal, 0.0))
    best = (np.inf, candidates[0], float(ridge_grid[0]))
    for rank_axis, rank in enumerate(candidates):
        for ridge_axis, ridge in enumerate(ridge_grid):
            candidate = (float(losses[rank_axis, ridge_axis]), rank, float(ridge))
            if candidate < best:
                best = candidate
    return best[1], best[2], best[0]


def _validate_panel_inputs(
    response: np.ndarray,
    rna_raw: np.ndarray,
    generic_raw: np.ndarray | None,
    aligned: np.ndarray | None,
) -> tuple[int, int, int]:
    y = np.asarray(response)
    if y.ndim != 4 or y.shape[0] != 2 or not np.isfinite(y).all():
        raise ValueError("response must be finite Plate6/Plate14 x context x intervention x gene")
    _, nctx, npert, ngene = y.shape
    if np.asarray(rna_raw).ndim != 2 or len(rna_raw) != nctx or not np.isfinite(rna_raw).all():
        raise ValueError("RNA matrix must be finite and share the response context axis")
    if generic_raw is not None and (np.asarray(generic_raw).ndim != 2 or len(generic_raw) != nctx or not np.isfinite(generic_raw).all()):
        raise ValueError("generic modality must be a finite complete-case context matrix")
    if aligned is not None and (np.asarray(aligned).ndim != 3 or np.asarray(aligned).shape[:2] != (nctx, npert) or not np.isfinite(aligned).all()):
        raise ValueError("aligned modality must be finite on the exact eligible context/intervention panel")
    return nctx, npert, ngene


def fit_predict_outer_fold(
    response: np.ndarray,
    held_context: int,
    rna_raw: np.ndarray,
    generic_raw: np.ndarray | None,
    aligned: np.ndarray | None,
    *,
    specs: Sequence[ModelSpec] = DEFAULT_MODEL_SPECS,
    response_ranks: Sequence[int] = RESPONSE_RANKS,
    generic_ranks: Sequence[int] = GENERIC_RANKS,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    intervention_axes: Sequence[int] | None = None,
    seed: int = SEED,
) -> FoldPredictions:
    """Fit one sealed LOCO fold; held response is accessed only by the caller.

    The returned predictions can therefore be subjected to adversarial held-
    outcome replacement tests without conflating fitting with evaluation.
    """

    y = np.asarray(response, dtype=np.float64)
    nctx, npert, _ = _validate_panel_inputs(y, rna_raw, generic_raw, aligned)
    paxes = np.arange(npert, dtype=np.int64) if intervention_axes is None else np.asarray(intervention_axes, dtype=np.int64)
    if paxes.shape != (npert,) or len(np.unique(paxes)) != npert:
        raise ValueError("intervention axes must uniquely cover the supplied panel")
    if not 0 <= held_context < nctx:
        raise ValueError("held context is out of range")
    train = np.delete(np.arange(nctx), held_context)
    rna_train, rna_test = _dual_context_pca(np.asarray(rna_raw)[train], np.asarray(rna_raw)[[held_context]], RNA_RANK)
    if generic_raw is not None:
        generic_max = max(int(x) for x in generic_ranks)
        generic_train, generic_test = _dual_context_pca(np.asarray(generic_raw)[train], np.asarray(generic_raw)[[held_context]], generic_max)
    else:
        generic_train = generic_test = None
    if aligned is not None:
        aligned_train, aligned_test = _standardize_aligned(np.asarray(aligned)[train], np.asarray(aligned)[[held_context]])
    else:
        aligned_train = aligned_test = None

    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    fit_rows: list[dict[str, object]] = []
    prepared: dict[int, dict[str, object]] = {}
    for source_plate in (0, 1):
        training_response = y[source_plate, train]
        memory = training_response.mean(axis=0)
        adaptation = training_response - memory[None]
        max_rank = min(max(int(x) for x in response_ranks), min(adaptation.reshape(-1, adaptation.shape[-1]).shape))
        basis = _randomized_response_basis(
            adaptation.reshape(-1, adaptation.shape[-1]), max_rank, seed + 1009 * held_context + source_plate
        )
        flat_adaptation = adaptation.reshape(-1, adaptation.shape[-1])
        max_scores = flat_adaptation @ basis.T
        truth_norm = np.einsum("ng,ng->n", flat_adaptation, flat_adaptation)
        embedding = _intervention_embedding(memory)
        selected_designs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for spec in specs:
            if spec.use_generic and generic_train is None:
                continue
            if spec.use_aligned and aligned_train is None:
                continue
            rank_options: Sequence[int | None] = generic_ranks if spec.use_generic else (None,)
            for estimator, bilinear in (("ridge", False), ("bilinear_ridge", True)):
                best: tuple[float, int, float, int | None, np.ndarray, np.ndarray] | None = None
                for generic_rank in rank_options:
                    context_train: list[np.ndarray] = []
                    context_test: list[np.ndarray] = []
                    aligned_train_blocks: list[np.ndarray] = []
                    aligned_test_blocks: list[np.ndarray] = []
                    if spec.use_rna:
                        context_train.append(rna_train)
                        context_test.append(rna_test)
                    if spec.use_generic:
                        assert generic_train is not None and generic_test is not None and generic_rank is not None
                        context_train.append(generic_train[:, : int(generic_rank)])
                        context_test.append(generic_test[:, : int(generic_rank)])
                    if spec.use_aligned:
                        assert aligned_train is not None and aligned_test is not None
                        aligned_train_blocks.append(aligned_train)
                        aligned_test_blocks.append(aligned_test)
                    xtrain = build_design(context_train, aligned_train_blocks, embedding, bilinear=bilinear)
                    xtest = build_design(context_test, aligned_test_blocks, embedding, bilinear=bilinear)
                    rank, ridge, loss = _select_response_rank_ridge_scores(
                        xtrain, max_scores, truth_norm, len(train), npert,
                        response_ranks, ridge_grid, paxes,
                    )
                    candidate = (loss, rank, ridge, generic_rank, xtrain, xtest)
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
                assert best is not None
                loss, rank, ridge, generic_rank, xtrain, xtest = best
                selected_designs[f"{spec.name}::{estimator}"] = (xtrain, xtest)
                scores = max_scores[:, :rank]
                coefficient = _ridge_coefficients(xtrain, scores, ridge)
                predicted_scores = xtest @ coefficient
                prediction = memory + predicted_scores @ basis[:rank]
                predictions[(spec.name, estimator, source_plate)] = prediction
                fit_rows.append(
                    {
                        "held_context": held_context,
                        "source_plate": source_plate,
                        "model": spec.name,
                        "estimator": estimator,
                        "response_rank": rank,
                        "generic_rank": generic_rank,
                        "ridge": ridge,
                        "inner_loss": loss,
                        "design_columns": xtrain.shape[1],
                        "training_contexts": len(train),
                    }
                )
        prepared[source_plate] = {
            "basis": basis,
            "memory": memory,
            "max_scores": max_scores,
            "truth_norm": truth_norm,
            "embedding": embedding,
            "selected_designs": selected_designs,
        }
    return FoldPredictions(predictions, pd.DataFrame(fit_rows), prepared)


def run_loco_models(
    response: np.ndarray,
    rna_raw: np.ndarray,
    generic_raw: np.ndarray | None,
    aligned: np.ndarray | None,
    *,
    context_axes: Sequence[int] | None = None,
    intervention_axes: Sequence[int] | None = None,
    specs: Sequence[ModelSpec] = DEFAULT_MODEL_SPECS,
    response_ranks: Sequence[int] = RESPONSE_RANKS,
    generic_ranks: Sequence[int] = GENERIC_RANKS,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    seed: int = SEED,
) -> OOFResult:
    """Run the sealed LOCO battery and return the frozen utility table."""

    y = np.asarray(response, dtype=np.float64)
    nctx, npert, _ = _validate_panel_inputs(y, rna_raw, generic_raw, aligned)
    caxes = np.arange(nctx) if context_axes is None else np.asarray(context_axes, dtype=np.int64)
    paxes = np.arange(npert) if intervention_axes is None else np.asarray(intervention_axes, dtype=np.int64)
    if len(caxes) != nctx or len(paxes) != npert:
        raise ValueError("reported axes must exactly cover the supplied response panel")
    denominator = np.einsum("cpg,cpg->cp", y[0], y[1], dtype=np.float64)
    block_rows: list[dict[str, object]] = []
    fits: list[pd.DataFrame] = []
    for held in range(nctx):
        fold = fit_predict_outer_fold(
            y,
            held,
            rna_raw,
            generic_raw,
            aligned,
            specs=specs,
            response_ranks=response_ranks,
            generic_ranks=generic_ranks,
            ridge_grid=ridge_grid,
            intervention_axes=paxes,
            seed=seed,
        )
        fits.append(fold.fits.assign(context_axis=int(caxes[held])))
        grouped: dict[tuple[str, str], list[np.ndarray]] = {}
        for (model, estimator, source), prediction in fold.predictions.items():
            error6 = y[0, held] - prediction
            error14 = y[1, held] - prediction
            numerator = np.einsum("pg,pg->p", error6, error14, dtype=np.float64)
            grouped.setdefault((model, estimator), [None, None])[source] = numerator
        for (model, estimator), directional in grouped.items():
            if directional[0] is None or directional[1] is None:
                raise AssertionError("both plate-direction fits are required")
            averaged = 0.5 * (directional[0] + directional[1])
            for p in range(npert):
                block_rows.append(
                    {
                        "model": model,
                        "estimator": estimator,
                        "context_axis": int(caxes[held]),
                        "intervention_axis": int(paxes[p]),
                        "numerator_after": float(averaged[p]),
                        "numerator_plate6_fit": float(directional[0][p]),
                        "numerator_plate14_fit": float(directional[1][p]),
                        "denominator_u": float(denominator[held, p]),
                    }
                )
    return OOFResult(pd.DataFrame(block_rows), pd.concat(fits, ignore_index=True))


def _paired_arrays(
    blocks: pd.DataFrame,
    model: str,
    comparator: str,
    estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"model", "estimator", "context_axis", "intervention_axis", "numerator_after", "denominator_u"}
    if not required.issubset(blocks.columns):
        raise ValueError(f"utility table lacks columns {sorted(required - set(blocks.columns))}")
    local = blocks[blocks.estimator.eq(estimator)]
    a = local[local.model.eq(model)].pivot(index="context_axis", columns="intervention_axis", values="numerator_after")
    b = local[local.model.eq(comparator)].pivot(index="context_axis", columns="intervention_axis", values="numerator_after")
    d = local[local.model.eq(model)].pivot(index="context_axis", columns="intervention_axis", values="denominator_u")
    if not a.index.equals(b.index) or not a.columns.equals(b.columns) or not a.index.equals(d.index) or not a.columns.equals(d.columns):
        raise ValueError("paired models do not use the exact same context/intervention panel")
    return a.to_numpy(float), b.to_numpy(float), d.to_numpy(float), a.index.to_numpy(int)


def paired_incremental_summary(
    blocks: pd.DataFrame,
    model: str,
    comparator: str,
    estimator: str,
    *,
    draws: int = NULL_DRAWS,
    seed: int = SEED,
) -> dict[str, float | int | str]:
    model_num, comparator_num, denominator, _ = _paired_arrays(blocks, model, comparator, estimator)
    delta = comparator_num - model_num
    estimate = float(delta.sum() / denominator.sum())
    bootstrap = hierarchical_bootstrap(delta, denominator, draws=draws, seed=seed)
    return {
        "model": model,
        "comparator": comparator,
        "estimator": estimator,
        "delta_g": estimate,
        "ci_low": float(np.nanquantile(bootstrap, 0.025)),
        "ci_high": float(np.nanquantile(bootstrap, 0.975)),
        "bootstrap_draws": int(draws),
    }


def hierarchical_bootstrap(delta: np.ndarray, denominator: np.ndarray, *, draws: int, seed: int) -> np.ndarray:
    """Frozen-table paired bootstrap over contexts and interventions."""

    x = np.asarray(delta, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    if x.ndim != 2 or x.shape != d.shape:
        raise ValueError("paired delta and denominator must be equal context x intervention tables")
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        ci = rng.integers(0, x.shape[0], x.shape[0])
        pi = rng.integers(0, x.shape[1], x.shape[1])
        local_d = d[np.ix_(ci, pi)].sum()
        values[draw] = x[np.ix_(ci, pi)].sum() / local_d if abs(local_d) > EPS else np.nan
    return values


def paired_signflip_null(
    delta: np.ndarray,
    denominator: np.ndarray,
    *,
    draws: int = NULL_DRAWS,
    seed: int = SEED,
) -> np.ndarray:
    """Paired frozen-table null with context and intervention block signs."""

    x, d = np.asarray(delta, float), np.asarray(denominator, float)
    if x.ndim != 2 or x.shape != d.shape:
        raise ValueError("delta and denominator must be equal 2D frozen tables")
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    total_den = d.sum()
    for draw in range(draws):
        context_sign = rng.choice((-1.0, 1.0), size=x.shape[0])
        intervention_sign = rng.choice((-1.0, 1.0), size=x.shape[1])
        result[draw] = np.sum(x * context_sign[:, None] * intervention_sign[None, :]) / total_den
    return result


def blocked_permutation_orders(
    annotations: pd.DataFrame,
    block_columns: Sequence[str],
    *,
    draws: int = NULL_DRAWS,
    seed: int = SEED,
) -> np.ndarray:
    """Generate outcome-independent intervention mapping permutations."""

    if not set(block_columns).issubset(annotations.columns):
        raise ValueError("a frozen mapping-permutation block column is absent")
    n = len(annotations)
    groups = annotations.groupby(list(block_columns), dropna=False, sort=False).indices
    rng = np.random.default_rng(seed)
    orders = np.tile(np.arange(n, dtype=np.int32), (draws, 1))
    for draw in range(draws):
        for members in groups.values():
            members = np.asarray(members, dtype=np.int64)
            orders[draw, members] = rng.permutation(members)
    return orders


def context_permutation_orders(
    lineage: Sequence[str],
    *,
    draws: int = NULL_DRAWS,
    seed: int = SEED,
) -> np.ndarray:
    frame = pd.DataFrame({"lineage": [str(x) for x in lineage]})
    return blocked_permutation_orders(frame, ["lineage"], draws=draws, seed=seed)


def apply_intervention_order(aligned: np.ndarray, order: Sequence[int]) -> np.ndarray:
    values = np.asarray(aligned)
    permutation = np.asarray(order, dtype=np.int64)
    if values.ndim != 3 or len(permutation) != values.shape[1] or not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
        raise ValueError("order must be a permutation of the aligned intervention axis")
    return values[:, permutation, :]


def per_intervention_gains(
    blocks: pd.DataFrame,
    *,
    model: str,
    comparator: str,
    estimator: str,
    modality: str,
    intervention_ids: Mapping[int, str] | Sequence[str],
    annotations: pd.DataFrame | None = None,
    response_target: str = "PRIMARY_RESIDUAL",
) -> pd.DataFrame:
    """Return bridge-ready paired gains with context-cluster ratio uncertainty."""

    model_num, comparator_num, denominator, contexts = _paired_arrays(blocks, model, comparator, estimator)
    local = blocks[(blocks.model.eq(model)) & blocks.estimator.eq(estimator)]
    interventions = np.sort(local.intervention_axis.unique()).astype(int)
    rows: list[dict[str, object]] = []
    nctx = len(contexts)
    for local_p, axis in enumerate(interventions):
        contribution = comparator_num[:, local_p] - model_num[:, local_p]
        den = denominator[:, local_p]
        total_den = den.sum()
        estimate = float(contribution.sum() / total_den)
        influence = contribution - estimate * den
        se = float(np.sqrt(nctx / max(nctx - 1, 1) * np.square(influence).sum()) / abs(total_den))
        if isinstance(intervention_ids, Mapping):
            intervention_id = intervention_ids[int(axis)]
        else:
            intervention_id = intervention_ids[int(axis)]
        rows.append(
            {
                "response_target": response_target,
                "modality": modality,
                "model": model,
                "comparator": comparator,
                "estimator": estimator,
                "intervention_axis": int(axis),
                "intervention_id": str(intervention_id),
                "delta_g_aligned_minus_rna": estimate,
                "paired_se": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "contexts": nctx,
                "positive_context_fraction": float(np.mean(contribution / np.where(np.abs(den) > EPS, den, np.nan) > 0)),
            }
        )
    result = pd.DataFrame(rows)
    if annotations is not None:
        annotation_columns = [
            column
            for column in (
                "intervention_axis",
                "moa_broad",
                "moa_fine",
                "hgnc_target_family_ids",
                "hgnc_target_family_names",
                "exact_targets",
            )
            if column in annotations.columns
        ]
        result = result.merge(annotations[annotation_columns], on="intervention_axis", how="left", validate="one_to_one")
    return result


def build_power_folds(
    coefficient_noise: np.ndarray,
    rna_raw: np.ndarray,
    aligned: np.ndarray,
    *,
    intervention_axes: Sequence[int] | None = None,
    bilinear: bool = True,
) -> list[PowerFold]:
    """Freeze fold-local M0/M4 designs for vectorized CPU power calibration."""

    noise = np.asarray(coefficient_noise, dtype=np.float64)
    if noise.ndim != 4 or noise.shape[0] != 2:
        raise ValueError("coefficient noise must be plate x context x intervention x coefficient")
    nctx, npert = noise.shape[1:3]
    paxes = np.arange(npert, dtype=np.int64) if intervention_axes is None else np.asarray(intervention_axes, dtype=np.int64)
    if paxes.shape != (npert,) or len(np.unique(paxes)) != npert:
        raise ValueError("power intervention axes must uniquely cover the supplied panel")
    if np.asarray(aligned).shape[:2] != (nctx, npert):
        raise ValueError("aligned feature panel does not match coefficient noise")
    folds: list[PowerFold] = []
    for held in range(nctx):
        train = np.delete(np.arange(nctx), held)
        rna_train, rna_test = _dual_context_pca(np.asarray(rna_raw)[train], np.asarray(rna_raw)[[held]], RNA_RANK)
        aligned_train, aligned_test = _standardize_aligned(np.asarray(aligned)[train], np.asarray(aligned)[[held]])
        for source in (0, 1):
            memory = noise[source, train].mean(axis=0)
            embedding = _intervention_embedding(memory)
            folds.append(
                PowerFold(
                    held,
                    train,
                    source,
                    build_design([rna_train], [], embedding, bilinear=bilinear),
                    build_design([rna_test], [], embedding, bilinear=bilinear),
                    build_design([rna_train], [aligned_train], embedding, bilinear=bilinear),
                    build_design([rna_test], [aligned_test], embedding, bilinear=bilinear),
                    paxes.copy(),
                )
            )
    return folds


def make_aligned_synthetic_signal(aligned: np.ndarray, output_rank: int, *, seed: int = SEED) -> np.ndarray:
    """Outcome-independent signal confined to the frozen aligned feature map."""

    z = np.asarray(aligned, dtype=np.float64)
    if z.ndim != 3 or not np.isfinite(z).all():
        raise ValueError("synthetic signal requires a finite eligible aligned panel")
    centered = z - z.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    rotation = rng.normal(size=(z.shape[-1], output_rank)) / np.sqrt(max(z.shape[-1], 1))
    signal = np.einsum("cpf,fk->cpk", centered, rotation)
    norm = np.sqrt(np.mean(np.square(signal)))
    if norm < EPS:
        raise ValueError("aligned feature map has no context-varying synthetic direction")
    return signal / norm


def _batched_ridge_predict_cpu(
    xtrain: np.ndarray,
    xtest: np.ndarray,
    ytrain: np.ndarray,
    ridge_grid: Sequence[float],
    intervention_axes: np.ndarray,
) -> np.ndarray:
    """Select ridge and predict a replicate×level batch using sufficient stats."""

    y = np.asarray(ytrain, dtype=np.float64)
    if y.ndim != 5:
        raise ValueError("batched targets must be replicate x level x context x intervention x output")
    nrep, nlevel, nctx, npert, nout = y.shape
    memory = y.mean(axis=2)
    centered = (y - memory[:, :, None]).reshape(nrep, nlevel, nctx * npert, nout)
    frozen_axes = np.asarray(intervention_axes, dtype=np.int64)
    if frozen_axes.shape != (npert,):
        raise ValueError("power intervention axes do not match the target panel")
    paxis = np.tile(np.arange(npert), nctx)
    folds = frozen_axes % INNER_FOLDS
    losses = np.zeros((nrep, nlevel, len(ridge_grid)), dtype=np.float64)
    for fold in range(INNER_FOLDS):
        train_mask = folds[paxis] != fold
        valid_mask = ~train_mask
        xt, xv = xtrain[train_mask], xtrain[valid_mask]
        rhs = np.einsum("nd,blnk->bldk", xt, centered[:, :, train_mask], optimize=True)
        truth = centered[:, :, valid_mask]
        for ridge_axis, ridge in enumerate(ridge_grid):
            coefficient = np.einsum(
                "de,blek->bldk",
                np.linalg.pinv(xt.T @ xt + _ridge_penalty(xt.shape[1], float(ridge))),
                rhs,
                optimize=True,
            )
            prediction = np.einsum("nd,bldk->blnk", xv, coefficient, optimize=True)
            losses[:, :, ridge_axis] += np.square(truth - prediction).sum(axis=(2, 3))
    selected = losses.argmin(axis=2)
    rhs = np.einsum("nd,blnk->bldk", xtrain, centered, optimize=True)
    prediction = np.empty((nrep, nlevel, npert, nout), dtype=np.float64)
    for ridge_axis, ridge in enumerate(ridge_grid):
        coefficient = np.einsum(
            "de,blek->bldk",
            np.linalg.pinv(xtrain.T @ xtrain + _ridge_penalty(xtrain.shape[1], float(ridge))),
            rhs,
            optimize=True,
        )
        all_prediction = np.einsum("pd,bldk->blpk", xtest, coefficient, optimize=True) + memory
        mask = selected == ridge_axis
        prediction[mask] = all_prediction[mask]
    return prediction


def _ridge_penalty(size: int, ridge: float) -> np.ndarray:
    penalty = np.eye(size, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return penalty


def _scale_for_target(
    noise: np.ndarray,
    signal: np.ndarray,
    orthogonal_denominator: np.ndarray,
    target: float,
) -> float:
    base = float(np.einsum("cpk,cpk->", noise[0], noise[1]) + orthogonal_denominator.sum())

    def oracle(scale: float) -> float:
        y0, y1 = noise[0] + scale * signal, noise[1] + scale * signal
        denominator = float(np.einsum("cpk,cpk->", y0, y1) + orthogonal_denominator.sum())
        return 1.0 - base / denominator

    low, high = 0.0, 1.0
    while oracle(high) < target and high < 1e6:
        high *= 2.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        if oracle(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def vectorized_power_calibration(
    coefficient_noise: np.ndarray,
    signal: np.ndarray,
    folds: Sequence[PowerFold],
    *,
    orthogonal_denominator: np.ndarray | None = None,
    levels: Sequence[float] = POWER_LEVELS,
    replicates: int = POWER_REPLICATES,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    bootstrap_draws: int = 1_000,
    batch_size: int = 20,
    seed: int = SEED,
) -> PowerResult:
    """CPU-only 200×4 matched injection audit with no full-gene refits.

    Replicates and effect levels are vectorized together; only outer folds and
    bounded replicate batches are iterated.  ``coefficient_noise`` is a frozen
    training-only response-coordinate table, while omitted gene-space energy is
    retained exactly through ``orthogonal_denominator``.
    """

    noise = np.asarray(coefficient_noise, dtype=np.float64)
    injected = np.asarray(signal, dtype=np.float64)
    if noise.ndim != 4 or noise.shape[0] != 2 or injected.shape != noise.shape[1:]:
        raise ValueError("power noise/signal axes must be 2 x C x P x K and C x P x K")
    nctx, npert = noise.shape[1:3]
    orthogonal = np.zeros((nctx, npert), dtype=np.float64) if orthogonal_denominator is None else np.asarray(orthogonal_denominator, dtype=np.float64)
    if orthogonal.shape != (nctx, npert):
        raise ValueError("orthogonal denominator must be context x intervention")
    fold_map = {(fold.held_context, fold.source_plate): fold for fold in folds}
    if set(fold_map) != {(c, p) for c in range(nctx) for p in (0, 1)}:
        raise ValueError("power folds must cover every held context and source plate exactly once")
    level_values = np.asarray([0.0, *levels], dtype=np.float64)
    scales = np.asarray([0.0, *[_scale_for_target(noise, injected, orthogonal, float(x)) for x in levels]])
    rng = np.random.default_rng(seed)
    context_orders = np.stack([rng.permutation(nctx) for _ in range(replicates)])
    intervention_orders = np.stack([rng.permutation(npert) for _ in range(replicates)])
    estimates = np.empty((replicates, len(level_values)), dtype=np.float64)
    lower = np.empty_like(estimates)
    upper = np.empty_like(estimates)
    for lo in range(0, replicates, batch_size):
        hi = min(lo + batch_size, replicates)
        batch = hi - lo
        permuted = np.empty((batch, 2, nctx, npert, noise.shape[-1]), dtype=np.float64)
        permuted_orthogonal = np.empty((batch, nctx, npert), dtype=np.float64)
        for local, rep in enumerate(range(lo, hi)):
            ci, pi = context_orders[rep], intervention_orders[rep]
            permuted[local] = noise[:, ci][:, :, pi]
            permuted_orthogonal[local] = orthogonal[np.ix_(ci, pi)]
        truth = permuted[:, :, None] + scales[None, None, :, None, None, None] * injected[None, None, None]
        numerator_base = np.zeros((batch, len(level_values), nctx, npert), dtype=np.float64)
        numerator_aug = np.zeros_like(numerator_base)
        for held in range(nctx):
            direction_base, direction_aug = [], []
            for source in (0, 1):
                fold = fold_map[(held, source)]
                # ``np.take`` preserves B,L,C,P,K order; mixed advanced indexing
                # would move the selected context axis to the front.
                ytrain = np.take(truth[:, source], fold.train_contexts, axis=2)
                pred_base = _batched_ridge_predict_cpu(
                    fold.x_base_train, fold.x_base_test, ytrain, ridge_grid, fold.intervention_axes
                )
                pred_aug = _batched_ridge_predict_cpu(
                    fold.x_aug_train, fold.x_aug_test, ytrain, ridge_grid, fold.intervention_axes
                )
                truth0, truth1 = truth[:, 0, :, held], truth[:, 1, :, held]
                direction_base.append(np.einsum("blpk,blpk->blp", truth0 - pred_base, truth1 - pred_base))
                direction_aug.append(np.einsum("blpk,blpk->blp", truth0 - pred_aug, truth1 - pred_aug))
            numerator_base[:, :, held] = 0.5 * (direction_base[0] + direction_base[1])
            numerator_aug[:, :, held] = 0.5 * (direction_aug[0] + direction_aug[1])
        denominator = np.einsum("blcpk,blcpk->blcp", truth[:, 0], truth[:, 1]) + permuted_orthogonal[:, None]
        delta = numerator_base - numerator_aug
        estimates[lo:hi] = delta.sum(axis=(2, 3)) / denominator.sum(axis=(2, 3))
        for local in range(batch):
            for level_axis in range(len(level_values)):
                boot = hierarchical_bootstrap(
                    delta[local, level_axis], denominator[local, level_axis], draws=bootstrap_draws,
                    seed=seed + 1_000_003 * (lo + local) + level_axis,
                )
                lower[lo + local, level_axis], upper[lo + local, level_axis] = np.nanquantile(boot, [0.025, 0.975])
    replicate_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for level_axis, target in enumerate(level_values):
        detected = lower[:, level_axis] > 0
        for replicate in range(replicates):
            replicate_rows.append(
                {
                    "target_delta_g": float(target),
                    "replicate": replicate,
                    "estimate": float(estimates[replicate, level_axis]),
                    "ci_low": float(lower[replicate, level_axis]),
                    "ci_high": float(upper[replicate, level_axis]),
                    "detected": bool(detected[replicate]),
                    "covers_target": bool(lower[replicate, level_axis] <= target <= upper[replicate, level_axis]),
                }
            )
        curve_rows.append(
            {
                "target_delta_g": float(target),
                "estimate": float(estimates[:, level_axis].mean()),
                "bias": float(estimates[:, level_axis].mean() - target),
                "power": float(detected.mean()),
                "false_positive_rate": float(detected.mean()) if target == 0 else np.nan,
                "ci_coverage": float(np.mean((lower[:, level_axis] <= target) & (target <= upper[:, level_axis]))),
                "replicates": replicates,
                "synthetic_scale": float(scales[level_axis]),
                "device": "cpu",
            }
        )
    curves = pd.DataFrame(curve_rows)
    positive = curves[curves.target_delta_g > 0].sort_values("target_delta_g")
    reached = positive[positive.power >= 0.80]
    detection: str | float = float(reached.iloc[0].target_delta_g) if len(reached) else ">0.20"
    return PowerResult(curves, pd.DataFrame(replicate_rows), detection)


def _nonidentity_intervention_order(size: int, seed: int) -> np.ndarray:
    """Return the single predeclared outcome-independent power-null mapping."""

    if size < 2:
        raise ValueError("power calibration requires at least two interventions")
    order = np.random.default_rng(seed).permutation(size)
    if np.array_equal(order, np.arange(size)):
        order = np.roll(order, 1)
    return order.astype(np.int64)


def _training_only_signal_coefficients(
    aligned: np.ndarray,
    train: np.ndarray,
    output_rank: int,
    *,
    seed: int,
) -> np.ndarray:
    """Construct S(Z,p) using only outer-training centering and scaling."""

    z = np.asarray(aligned, dtype=np.float64)
    ztrain, ztest = _standardize_aligned(z[train], z[np.setdiff1d(np.arange(len(z)), train)])
    standardized = np.empty_like(z, dtype=np.float64)
    standardized[train] = ztrain
    held = np.setdiff1d(np.arange(len(z)), train)
    standardized[held] = ztest
    rng = np.random.default_rng(seed)
    rotation = rng.normal(size=(z.shape[-1], output_rank)) / np.sqrt(max(z.shape[-1], 1))
    signal = np.einsum("cpf,fk->cpk", standardized, rotation, optimize=True)
    # The real fold memory is the source-plate mean over training contexts.
    # Centering S within that identical set keeps memory and its intervention
    # embedding fixed for every injection level.
    signal -= signal[train].mean(axis=0, keepdims=True)
    norm = float(np.sqrt(np.mean(np.square(signal[train]))))
    if norm < EPS:
        raise ValueError("aligned panel has no training-fold-varying synthetic direction")
    return signal / norm


def prepare_fold_local_power_folds(
    response: np.ndarray,
    rna_raw: np.ndarray,
    aligned: np.ndarray,
    *,
    intervention_axes: Sequence[int] | None = None,
    response_ranks: Sequence[int] = RESPONSE_RANKS,
    bilinear: bool = True,
    seed: int = SEED,
    intervention_order: Sequence[int] | None = None,
    held_contexts: Sequence[int] | None = None,
) -> list[FoldLocalPowerFold]:
    """Prepare exact fold-local coefficient and orthogonal-energy statistics.

    The fixed intervention order destroys real feature-response correspondence
    before power injection.  It does not alter the row-space covariance used to
    fit a response basis.  No held response contributes to a basis, memory,
    RNA transform, aligned transform, intervention embedding, or signal map.
    Held responses are projected only for the final synthetic score, exactly as
    held outcomes are used only for evaluation in the real analysis.
    """

    y = np.asarray(response)
    nctx, npert, ngene = _validate_panel_inputs(y, rna_raw, None, aligned)
    paxes = np.arange(npert, dtype=np.int64) if intervention_axes is None else np.asarray(intervention_axes, dtype=np.int64)
    if paxes.shape != (npert,) or len(np.unique(paxes)) != npert:
        raise ValueError("power intervention axes must uniquely cover the supplied panel")
    order = (_nonidentity_intervention_order(npert, seed + 313)
             if intervention_order is None else np.asarray(intervention_order, dtype=np.int64))
    if order.shape != (npert,) or not np.array_equal(np.sort(order), np.arange(npert)):
        raise ValueError("power intervention order must be a complete permutation")
    max_requested = max(int(rank) for rank in response_ranks)
    folds: list[FoldLocalPowerFold] = []
    all_contexts = np.arange(nctx, dtype=np.int64)
    selected_held = (np.arange(nctx, dtype=np.int64) if held_contexts is None
                     else np.asarray(held_contexts, dtype=np.int64))
    if selected_held.ndim != 1 or not len(selected_held) or selected_held.min() < 0 or selected_held.max() >= nctx:
        raise ValueError("power held-context selection is empty or out of range")
    if len(np.unique(selected_held)) != len(selected_held):
        raise ValueError("power held-context selection contains duplicates")
    for held in selected_held:
        held = int(held)
        train = np.delete(all_contexts, held)
        rna_train, rna_test = _dual_context_pca(
            np.asarray(rna_raw)[train], np.asarray(rna_raw)[[held]], RNA_RANK,
        )
        aligned_train, aligned_test = _standardize_aligned(
            np.asarray(aligned)[train], np.asarray(aligned)[[held]],
        )
        for source in (0, 1):
            training_response = np.asarray(y[source, train][:, order], dtype=np.float64)
            memory = training_response.mean(axis=0)
            adaptation = training_response - memory[None]
            flat = adaptation.reshape(-1, ngene)
            max_rank = min(max_requested, min(flat.shape))
            basis = _randomized_response_basis(
                flat, max_rank, seed + 1009 * held + source,
            )
            memory_coeff = memory @ basis.T
            embedding = _intervention_embedding(memory)
            x_base_train = build_design([rna_train], [], embedding, bilinear=bilinear)
            x_base_test = build_design([rna_test], [], embedding, bilinear=bilinear)
            x_aug_train = build_design([rna_train], [aligned_train], embedding, bilinear=bilinear)
            x_aug_test = build_design([rna_test], [aligned_test], embedding, bilinear=bilinear)
            signal = _training_only_signal_coefficients(
                np.asarray(aligned), train, len(basis), seed=seed + 1709 * held + source,
            )

            affine_coeff = np.empty((2, nctx, npert, len(basis)), dtype=np.float64)
            residual_cross = np.empty((nctx, npert), dtype=np.float64)
            residual_norm_source = np.empty((nctx, npert), dtype=np.float64)
            memory_orthogonal_norm = (
                np.einsum("pg,pg->p", memory, memory, dtype=np.float64)
                - np.einsum("pk,pk->p", memory_coeff, memory_coeff, dtype=np.float64)
            )
            memory_dot_sum = np.empty((nctx, npert), dtype=np.float64)
            for context in range(nctx):
                values = np.asarray(y[:, context][:, order], dtype=np.float64)
                affine = values - memory[None]
                coeff = np.einsum("tpg,kg->tpk", affine, basis, optimize=True)
                affine_coeff[:, context] = coeff
                full_cross = np.einsum("pg,pg->p", affine[0], affine[1], dtype=np.float64)
                coeff_cross = np.einsum("pk,pk->p", coeff[0], coeff[1], dtype=np.float64)
                residual_cross[context] = full_cross - coeff_cross
                full_norm = np.einsum("pg,pg->p", affine[source], affine[source], dtype=np.float64)
                coeff_norm = np.einsum("pk,pk->p", coeff[source], coeff[source], dtype=np.float64)
                residual_norm_source[context] = np.maximum(full_norm - coeff_norm, 0.0)
                full_memory_dot = np.einsum(
                    "pg,tpg->tp", memory, affine, dtype=np.float64,
                ).sum(axis=0)
                coeff_memory_dot = np.einsum(
                    "pk,tpk->tp", memory_coeff, coeff, dtype=np.float64,
                ).sum(axis=0)
                memory_dot_sum[context] = full_memory_dot - coeff_memory_dot
            folds.append(
                FoldLocalPowerFold(
                    held_context=held,
                    train_contexts=train.copy(),
                    source_plate=source,
                    intervention_axes=paxes.copy(),
                    basis=basis,
                    x_base_train=x_base_train,
                    x_base_test=x_base_test,
                    x_aug_train=x_aug_train,
                    x_aug_test=x_aug_test,
                    memory_coeff=memory_coeff,
                    affine_coeff=affine_coeff,
                    signal_coeff=signal,
                    residual_cross=residual_cross,
                    residual_norm_source=residual_norm_source,
                    memory_orthogonal_norm=np.maximum(memory_orthogonal_norm, 0.0),
                    memory_orthogonal_dot_affine_sum=memory_dot_sum,
                )
            )
            del training_response, adaptation, flat, memory
    return folds


def _batched_rank_ridge_predict_cpu(
    xtrain: np.ndarray,
    xtest: np.ndarray,
    ytrain: np.ndarray,
    orthogonal_norm: np.ndarray,
    response_ranks: Sequence[int],
    ridge_grid: Sequence[float],
    intervention_axes: np.ndarray,
    operators: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Exact batched analogue of real inner rank/ridge selection."""

    y = np.asarray(ytrain, dtype=np.float64)
    if y.ndim != 5:
        raise ValueError("batched power targets must be B x L x C x P x K")
    nrep, nlevel, nctx, npert, nout = y.shape
    orth = np.asarray(orthogonal_norm, dtype=np.float64)
    if orth.shape != (nctx, npert):
        raise ValueError("orthogonal training energy must be C x P")
    memory = y.mean(axis=2)
    centered = (y - memory[:, :, None]).reshape(nrep, nlevel, nctx * npert, nout)
    truth_norm = np.square(centered).sum(axis=-1) + orth.reshape(1, 1, -1)
    axes = np.asarray(intervention_axes, dtype=np.int64)
    if axes.shape != (npert,):
        raise ValueError("power intervention axes do not match the target panel")
    ranks = [int(rank) for rank in response_ranks if int(rank) <= nout] or [nout]
    paxis = np.tile(np.arange(npert), nctx)
    folds = axes % INNER_FOLDS
    if operators is None:
        operators = _power_ridge_operators(
            xtrain, xtest, ridge_grid, axes, nctx=nctx, npert=npert,
        )
    losses = np.zeros((nrep, nlevel, len(ranks), len(ridge_grid)), dtype=np.float64)
    for inner, local in enumerate(operators["cv"]):
        keep = np.asarray(local["train_mask"], dtype=bool)
        valid = np.asarray(local["valid_mask"], dtype=bool)
        if not keep.any() or not valid.any():
            continue
        xt, xv = xtrain[keep], xtrain[valid]
        valid_truth = centered[:, :, valid]
        cumulative_truth = np.cumsum(np.square(valid_truth), axis=-1)
        for ridge_axis, ridge in enumerate(ridge_grid):
            operator = np.asarray(local["coefficient_operator"][ridge_axis])
            coefficient = np.einsum(
                "dn,blnk->bldk", operator, centered[:, :, keep], optimize=True,
            )
            predicted = np.einsum("nd,bldk->blnk", xv, coefficient, optimize=True)
            cumulative_error = np.cumsum(np.square(valid_truth - predicted), axis=-1)
            for rank_axis, rank in enumerate(ranks):
                projected = cumulative_error[..., rank - 1].sum(axis=-1)
                omitted = (truth_norm[:, :, valid] - cumulative_truth[..., rank - 1]).sum(axis=-1)
                losses[:, :, rank_axis, ridge_axis] += projected + np.maximum(omitted, 0.0)
    selected = losses.reshape(nrep, nlevel, -1).argmin(axis=-1)
    selected_rank_axis = selected // len(ridge_grid)
    selected_ridge_axis = selected % len(ridge_grid)
    output = np.zeros((nrep, nlevel, npert, nout), dtype=np.float64)
    for ridge_axis, ridge in enumerate(ridge_grid):
        operator = np.asarray(operators["full_coefficient_operator"][ridge_axis])
        coefficient = np.einsum("dn,blnk->bldk", operator, centered, optimize=True)
        predicted = np.einsum("pd,bldk->blpk", xtest, coefficient, optimize=True)
        for rank_axis, rank in enumerate(ranks):
            mask = (selected_ridge_axis == ridge_axis) & (selected_rank_axis == rank_axis)
            if mask.any():
                output[mask, :, :rank] = predicted[mask, :, :rank]
    return output + memory


def _power_ridge_operators(
    xtrain: np.ndarray,
    xtest: np.ndarray,
    ridge_grid: Sequence[float],
    intervention_axes: np.ndarray,
    *,
    nctx: int,
    npert: int,
) -> dict[str, object]:
    """Cache design-only inverses once per outer fold, not once per batch."""

    axes = np.asarray(intervention_axes, dtype=np.int64)
    paxis = np.tile(np.arange(npert), nctx)
    inner_assignment = axes % INNER_FOLDS
    cv: list[dict[str, object]] = []
    for inner in range(INNER_FOLDS):
        train_mask = inner_assignment[paxis] != inner
        valid_mask = ~train_mask
        operators = _unpenalized_intercept_operator_family(xtrain[train_mask], ridge_grid)
        cv.append({
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "coefficient_operator": operators,
        })
    full = _unpenalized_intercept_operator_family(xtrain, ridge_grid)
    return {"cv": cv, "full_coefficient_operator": full, "test_rows": len(xtest)}


def _unpenalized_intercept_operator_family(
    design: np.ndarray,
    ridge_grid: Sequence[float],
) -> tuple[np.ndarray, ...]:
    """Exact ridge maps with one eigendecomposition for the full lambda grid.

    The first design column is the unpenalized intercept used by the real
    solver.  Centering all other columns is algebraically identical to setting
    penalty[0,0]=0, but avoids repeating a pseudoinverse for every ridge value.
    """

    x = np.asarray(design, dtype=np.float64)
    if x.ndim != 2 or not np.allclose(x[:, 0], 1.0, rtol=0, atol=1e-12):
        raise ValueError("power ridge design must begin with an intercept column")
    count = len(x)
    features = x[:, 1:]
    if not features.shape[1]:
        return tuple(np.full((1, count), 1.0 / count) for _ in ridge_grid)
    mean = features.mean(axis=0)
    centered = features - mean
    values, vectors = np.linalg.eigh(centered.T @ centered)
    values = np.maximum(values, 0.0)
    operators: list[np.ndarray] = []
    for ridge in ridge_grid:
        ridge = float(ridge)
        if ridge <= 0:
            raise ValueError("power ridge values must be positive")
        inverse = (vectors * (1.0 / (values + ridge))) @ vectors.T
        slope = inverse @ centered.T
        intercept = np.full((1, count), 1.0 / count) - mean[None] @ slope
        operators.append(np.vstack([intercept, slope]))
    return tuple(operators)


def fold_local_metric_components(
    fold: FoldLocalPowerFold,
    signs: np.ndarray,
    scales: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct held gene-space numerator/denominator from sufficient stats."""

    s = np.asarray(signs, dtype=np.float64)
    lam = np.asarray(scales, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != len(fold.intervention_axes):
        raise ValueError("power signs must be replicate x intervention")
    held = fold.held_context
    signal = fold.signal_coeff[held]
    coeff = (
        fold.memory_coeff[None, None, None]
        + s[:, None, None, :, None] * fold.affine_coeff[:, held][None, None]
        + lam[None, :, None, None, None] * signal[None, None, None]
    )
    # coeff is B x L x plate x intervention x coefficient.
    error = coeff - prediction[:, :, None]
    numerator = np.einsum("blpk,blpk->blp", error[:, :, 0], error[:, :, 1], optimize=True)
    numerator += fold.residual_cross[held][None, None]
    denominator = np.einsum("blpk,blpk->blp", coeff[:, :, 0], coeff[:, :, 1], optimize=True)
    denominator += fold.memory_orthogonal_norm[None, None]
    denominator += s[:, None] * fold.memory_orthogonal_dot_affine_sum[held][None, None]
    denominator += fold.residual_cross[held][None, None]
    return numerator, denominator


def _fold_local_scale(
    folds: Sequence[FoldLocalPowerFold],
    target: float,
) -> float:
    if target <= 0:
        return 0.0
    fold_map = {(f.held_context, f.source_plate): f for f in folds}
    nctx = max(f.held_context for f in folds) + 1

    def oracle(scale: float) -> float:
        contrast = 0.0
        denominator = 0.0
        for held in range(nctx):
            for source in (0, 1):
                fold = fold_map[(held, source)]
                signal = fold.signal_coeff[held]
                memory = fold.memory_coeff
                a0, a1 = fold.affine_coeff[:, held]
                contrast += 0.5 * scale * scale * float(np.square(signal).sum())
                denominator += 0.5 * float(
                    np.einsum("pk,pk->", memory + scale * signal, memory + scale * signal)
                    + np.einsum("pk,pk->", a0, a1)
                    + fold.memory_orthogonal_norm.sum()
                    + fold.residual_cross[held].sum()
                )
        return contrast / denominator if abs(denominator) > EPS else np.nan

    low, high = 0.0, 1.0
    while oracle(high) < target and high < 1e6:
        high *= 2.0
    if not np.isfinite(oracle(high)) or oracle(high) < target:
        raise RuntimeError(f"power target delta_g={target} is not calibratable")
    for _ in range(64):
        middle = 0.5 * (low + high)
        if oracle(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _bootstrap_ratio_batch(
    delta: np.ndarray,
    denominator: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized frozen-table hierarchical intervals for a B x L batch."""

    x, d = np.asarray(delta, float), np.asarray(denominator, float)
    if x.ndim != 4 or x.shape != d.shape:
        raise ValueError("power utility batches must be B x L x context x intervention")
    rng = np.random.default_rng(seed)
    context_counts = np.stack([
        np.bincount(rng.integers(0, x.shape[2], x.shape[2]), minlength=x.shape[2])
        for _ in range(draws)
    ]).astype(np.float64)
    intervention_counts = np.stack([
        np.bincount(rng.integers(0, x.shape[3], x.shape[3]), minlength=x.shape[3])
        for _ in range(draws)
    ]).astype(np.float64)
    num = np.einsum("qc,blcp,qp->blq", context_counts, x, intervention_counts, optimize=True)
    den = np.einsum("qc,blcp,qp->blq", context_counts, d, intervention_counts, optimize=True)
    ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=np.abs(den) > EPS)
    return np.nanquantile(ratio, 0.025, axis=-1), np.nanquantile(ratio, 0.975, axis=-1)


def fold_specific_power_calibration(
    folds: Sequence[FoldLocalPowerFold],
    *,
    levels: Sequence[float] = POWER_LEVELS,
    replicates: int = POWER_REPLICATES,
    response_ranks: Sequence[int] = RESPONSE_RANKS,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    bootstrap_draws: int = 500,
    batch_size: int = 10,
    seed: int = SEED,
) -> PowerResult:
    """Run the formal 200x4 fold-local coefficient-space power calibration."""

    if not folds:
        raise ValueError("fold-local power calibration received no folds")
    fold_map = {(fold.held_context, fold.source_plate): fold for fold in folds}
    nctx = max(fold.held_context for fold in folds) + 1
    npert = len(folds[0].intervention_axes)
    if set(fold_map) != {(c, source) for c in range(nctx) for source in (0, 1)}:
        raise ValueError("fold-local power data must cover every context and source direction")
    level_values = np.asarray([0.0, *levels], dtype=np.float64)
    scales = np.asarray([_fold_local_scale(folds, float(level)) for level in level_values])
    rng = np.random.default_rng(seed + 991)
    signs = rng.choice((-1.0, 1.0), size=(replicates, npert))
    delta = np.empty((replicates, len(level_values), nctx, npert), dtype=np.float64)
    denominator = np.empty_like(delta)
    # Outer-fold-first ordering lets every expensive design inverse be reused
    # across all replicate batches, while peak target memory stays bounded.
    for held in range(nctx):
        numerator_base, numerator_aug, directional_den = [], [], []
        for source in (0, 1):
            fold = fold_map[(held, source)]
            train = fold.train_contexts
            orth = fold.residual_norm_source[train]
            base_operators = _power_ridge_operators(
                fold.x_base_train, fold.x_base_test, ridge_grid,
                fold.intervention_axes, nctx=len(train), npert=npert,
            )
            aug_operators = _power_ridge_operators(
                fold.x_aug_train, fold.x_aug_test, ridge_grid,
                fold.intervention_axes, nctx=len(train), npert=npert,
            )
            local_base = np.empty((replicates, len(level_values), npert), dtype=np.float64)
            local_aug = np.empty_like(local_base)
            local_den = np.empty_like(local_base)
            for lo in range(0, replicates, batch_size):
                hi = min(lo + batch_size, replicates)
                local_signs = signs[lo:hi]
                training = (
                    fold.memory_coeff[None, None, None]
                    + local_signs[:, None, None, :, None]
                    * fold.affine_coeff[source, train][None, None]
                    + scales[None, :, None, None, None] * fold.signal_coeff[train][None, None]
                )
                pred_base = _batched_rank_ridge_predict_cpu(
                    fold.x_base_train, fold.x_base_test, training, orth,
                    response_ranks, ridge_grid, fold.intervention_axes,
                    operators=base_operators,
                )
                pred_aug = _batched_rank_ridge_predict_cpu(
                    fold.x_aug_train, fold.x_aug_test, training, orth,
                    response_ranks, ridge_grid, fold.intervention_axes,
                    operators=aug_operators,
                )
                base, den = fold_local_metric_components(fold, local_signs, scales, pred_base)
                aug, den_aug = fold_local_metric_components(fold, local_signs, scales, pred_aug)
                if not np.allclose(den, den_aug, rtol=0, atol=1e-9):
                    raise AssertionError("power comparator changed the scoring denominator")
                local_base[lo:hi], local_aug[lo:hi], local_den[lo:hi] = base, aug, den
            numerator_base.append(local_base)
            numerator_aug.append(local_aug)
            directional_den.append(local_den)
        delta[:, :, held] = 0.5 * (
            numerator_base[0] + numerator_base[1]
            - numerator_aug[0] - numerator_aug[1]
        )
        denominator[:, :, held] = 0.5 * (directional_den[0] + directional_den[1])
    estimates = delta.sum(axis=(2, 3)) / denominator.sum(axis=(2, 3))
    lower = np.empty_like(estimates)
    upper = np.empty_like(estimates)
    for lo in range(0, replicates, batch_size):
        hi = min(lo + batch_size, replicates)
        low, high = _bootstrap_ratio_batch(
            delta[lo:hi], denominator[lo:hi], draws=bootstrap_draws,
            seed=seed + 100_003 * lo,
        )
        lower[lo:hi], upper[lo:hi] = low, high

    replicate_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for level_axis, target in enumerate(level_values):
        detected = lower[:, level_axis] > 0.0
        for replicate in range(replicates):
            replicate_rows.append({
                "target_delta_g": float(target), "replicate": replicate,
                "estimate": float(estimates[replicate, level_axis]),
                "ci_low": float(lower[replicate, level_axis]),
                "ci_high": float(upper[replicate, level_axis]),
                "detected": bool(detected[replicate]),
                "covers_target": bool(lower[replicate, level_axis] <= target <= upper[replicate, level_axis]),
            })
        curve_rows.append({
            "target_delta_g": float(target),
            "estimate": float(estimates[:, level_axis].mean()),
            "bias": float(estimates[:, level_axis].mean() - target),
            "power": float(detected.mean()),
            "false_positive_rate": float(detected.mean()) if target == 0 else np.nan,
            "ci_coverage": float(np.mean((lower[:, level_axis] <= target) & (target <= upper[:, level_axis]))),
            "replicates": replicates,
            "synthetic_scale": float(scales[level_axis]),
            "device": "cpu",
        })
    curves = pd.DataFrame(curve_rows)
    reached = curves[(curves.target_delta_g > 0) & (curves.power >= 0.80)].sort_values("target_delta_g")
    detection: str | float = float(reached.iloc[0].target_delta_g) if len(reached) else ">0.20"
    return PowerResult(curves, pd.DataFrame(replicate_rows), detection)


__all__ = [
    "AlignedBlock",
    "CompleteCaseMatrix",
    "DEFAULT_MODEL_SPECS",
    "FoldPredictions",
    "FoldLocalPowerFold",
    "GENERIC_RANKS",
    "ModelSpec",
    "NULL_DRAWS",
    "OOFResult",
    "POWER_LEVELS",
    "POWER_REPLICATES",
    "PowerFold",
    "PowerResult",
    "RESPONSE_RANKS",
    "RIDGE_GRID",
    "aggregate_mapped_features",
    "apply_intervention_order",
    "blocked_permutation_orders",
    "build_design",
    "build_power_folds",
    "context_permutation_orders",
    "fit_predict_outer_fold",
    "fold_local_metric_components",
    "fold_specific_power_calibration",
    "hierarchical_bootstrap",
    "intersect_aligned_blocks",
    "make_aligned_synthetic_signal",
    "paired_incremental_summary",
    "paired_signflip_null",
    "prepare_fold_local_power_folds",
    "per_intervention_gains",
    "run_loco_models",
    "select_complete_features",
    "vectorized_power_calibration",
]
