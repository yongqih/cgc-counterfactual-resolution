"""Strict target-hidden low-rank interaction completion for Tahoe."""

from .core import (
    FactorFit,
    ProfiledWorkspace,
    PredictionWeights,
    additive_design,
    build_profiled_workspace,
    fit_profiled_cp,
    prediction_weights,
)

__all__ = [
    "FactorFit",
    "ProfiledWorkspace",
    "PredictionWeights",
    "additive_design",
    "build_profiled_workspace",
    "fit_profiled_cp",
    "prediction_weights",
]
