"""Strict prospective entrywise experimental-compression analysis."""

from igc_virtual_cell.cgc_entrywise.core import (
    LinearPrediction,
    SealedGramView,
    affine_ridge_weights,
    lowrank_context_weights,
)

__all__ = [
    "LinearPrediction",
    "SealedGramView",
    "affine_ridge_weights",
    "lowrank_context_weights",
]
