from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .core import (
    FactorFit,
    ProfiledWorkspace,
    build_profiled_workspace,
    fit_profiled_cp,
    prediction_weights,
    weighted_squared_error,
)
from .design import RANK_GRID, RIDGE_GRID, STARTS


@dataclass(frozen=True)
class SelectedFit:
    fit: FactorFit
    rows: list[dict[str, object]]
    selection_seconds: float
    final_seconds: float


def fit_seed(plate: int, fold: int, rank: int, ridge_index: int, start: int, stage: str) -> int:
    stage_offset = 0 if stage == "inner" else 500_000_000
    return int(202_608_251 + stage_offset + plate * 10_000_000 + fold * 100_000 + rank * 100 + ridge_index * 10 + start)


def _fit_two_starts(
    gram: np.ndarray,
    train: np.ndarray,
    rank: int,
    ridge: float,
    plate: int,
    fold: int,
    ridge_index: int,
    stage: str,
    device: str,
    workspace: ProfiledWorkspace,
) -> tuple[FactorFit, list[FactorFit], float]:
    started = perf_counter()
    fits = [
        fit_profiled_cp(
            gram,
            train,
            rank,
            ridge,
            fit_seed(plate, fold, rank, ridge_index, start, stage),
            device=device,
            max_steps=400,
            min_steps=80,
            convergence_window=30,
            tolerance=1e-3,
            gradient_tolerance=1e-4,
            learning_rate=0.03,
            workspace=workspace,
        )
        for start in range(STARTS)
    ]
    eligible = [fit for fit in fits if fit.converged]
    pool = eligible if eligible else fits
    best = min(pool, key=lambda fit: (fit.objective_best, fit.seed))
    return best, fits, perf_counter() - started


def select_and_refit(
    full_gram: np.ndarray,
    outer_train: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    *,
    plate: int,
    fold: int,
    device: str = "cuda",
) -> SelectedFit:
    """Select rank/L2 on inner targets, then refit on outer observations."""

    inner_gram = np.asarray(full_gram[np.ix_(inner_train, inner_train)], dtype=np.float32)
    inner_workspace = build_profiled_workspace(inner_gram, inner_train, device=device)
    rows: list[dict[str, object]] = []
    selection_started = perf_counter()
    candidates: list[tuple[float, int, float, int, FactorFit]] = []
    for rank in RANK_GRID:
        for ridge_index, ridge in enumerate(RIDGE_GRID):
            best, starts, seconds = _fit_two_starts(
                inner_gram,
                inner_train,
                rank,
                ridge,
                plate,
                fold,
                ridge_index,
                "inner",
                device,
                inner_workspace,
            )
            weights = prediction_weights(inner_train, inner_validation, best).full
            errors = weighted_squared_error(full_gram, inner_train, inner_validation, weights)
            validation_sse = float(np.sum(errors, dtype=np.float64))
            validation_mean = float(np.mean(errors, dtype=np.float64))
            for start_index, fitted in enumerate(starts):
                row = asdict(fitted)
                row.pop("u")
                row.pop("v")
                row.update(
                    {
                        "outer_fold": fold,
                        "plate": plate,
                        "stage": "inner_selection",
                        "start_index": start_index,
                        "start_selected": fitted.seed == best.seed,
                        "inner_train_entries": len(inner_train),
                        "inner_validation_entries": len(inner_validation),
                        "validation_sse_selected_start": validation_sse,
                        "validation_mean_sse_selected_start": validation_mean,
                        "candidate_wall_seconds": seconds,
                        "target_outcome_used": False,
                    }
                )
                rows.append(row)
            candidates.append((validation_sse, rank, -ridge, ridge_index, best))
    selection_seconds = perf_counter() - selection_started
    # Primary key is validation SSE. The fixed secondary ordering implements
    # smaller-rank then stronger-ridge preference for numerical ties.
    minimum = min(value[0] for value in candidates)
    tolerance = max(abs(minimum), 1.0) * 1e-9
    tied = [value for value in candidates if value[0] <= minimum + tolerance]
    _, selected_rank, negative_ridge, selected_ridge_index, selected_inner = min(
        tied, key=lambda value: (value[1], value[2], value[4].seed)
    )
    selected_ridge = -negative_ridge
    outer_gram = np.asarray(full_gram[np.ix_(outer_train, outer_train)], dtype=np.float32)
    outer_workspace = build_profiled_workspace(outer_gram, outer_train, device=device)
    final, final_starts, final_seconds = _fit_two_starts(
        outer_gram,
        outer_train,
        selected_rank,
        selected_ridge,
        plate,
        fold,
        selected_ridge_index,
        "final",
        device,
        outer_workspace,
    )
    for start_index, fitted in enumerate(final_starts):
        row = asdict(fitted)
        row.pop("u")
        row.pop("v")
        row.update(
            {
                "outer_fold": fold,
                "plate": plate,
                "stage": "outer_refit",
                "start_index": start_index,
                "start_selected": fitted.seed == final.seed,
                "inner_train_entries": len(inner_train),
                "inner_validation_entries": len(inner_validation),
                "validation_sse_selected_start": minimum,
                "validation_mean_sse_selected_start": minimum / len(inner_validation),
                "candidate_wall_seconds": final_seconds,
                "target_outcome_used": False,
                "selected_rank": selected_rank,
                "selected_ridge": selected_ridge,
                "selected_inner_seed": selected_inner.seed,
            }
        )
        rows.append(row)
    return SelectedFit(final, rows, selection_seconds, final_seconds)
