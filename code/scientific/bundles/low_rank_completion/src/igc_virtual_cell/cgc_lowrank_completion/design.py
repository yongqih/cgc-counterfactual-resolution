from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .core import CONTEXTS, ENTRIES, INTERVENTIONS


OUTER_SEED = 202_608_251
INNER_SEED = 202_608_252
SYNTHETIC_SEED = 202_608_253
OUTER_FOLDS = 100
RANK_GRID = (2, 4, 8, 16, 32)
RIDGE_GRID = (0.001, 0.01, 0.1)
STARTS = 2


@dataclass(frozen=True)
class FrozenMasks:
    outer_fold: np.ndarray
    inner_color: np.ndarray

    def outer_targets(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.outer_fold == int(fold)).astype(np.int64)

    def inner_validation(self, fold: int) -> np.ndarray:
        color = (37 * int(fold) + 11) % OUTER_FOLDS
        return np.flatnonzero((self.inner_color == color) & (self.outer_fold != int(fold))).astype(np.int64)

    def inner_training(self, fold: int) -> np.ndarray:
        hidden = np.zeros(ENTRIES, dtype=bool)
        hidden[self.outer_targets(fold)] = True
        hidden[self.inner_validation(fold)] = True
        return np.flatnonzero(~hidden).astype(np.int64)

    def outer_training(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.outer_fold != int(fold)).astype(np.int64)


def frozen_masks() -> FrozenMasks:
    context_rng = np.random.default_rng(OUTER_SEED)
    intervention_rng = np.random.default_rng(OUTER_SEED + 17)
    context_order = context_rng.permutation(CONTEXTS)
    intervention_order = intervention_rng.permutation(INTERVENTIONS)
    context_rank = np.empty(CONTEXTS, dtype=np.int64)
    intervention_rank = np.empty(INTERVENTIONS, dtype=np.int64)
    context_rank[context_order] = np.arange(CONTEXTS)
    intervention_rank[intervention_order] = np.arange(INTERVENTIONS)

    inner_context_rng = np.random.default_rng(INNER_SEED)
    inner_intervention_rng = np.random.default_rng(INNER_SEED + 17)
    inner_context_order = inner_context_rng.permutation(CONTEXTS)
    inner_intervention_order = inner_intervention_rng.permutation(INTERVENTIONS)
    inner_context_rank = np.empty(CONTEXTS, dtype=np.int64)
    inner_intervention_rank = np.empty(INTERVENTIONS, dtype=np.int64)
    inner_context_rank[inner_context_order] = np.arange(CONTEXTS)
    inner_intervention_rank[inner_intervention_order] = np.arange(INTERVENTIONS)

    indices = np.arange(ENTRIES, dtype=np.int64)
    context = indices // INTERVENTIONS
    intervention = indices % INTERVENTIONS
    outer = (intervention_rank[intervention] + 2 * context_rank[context]) % OUTER_FOLDS
    inner = (
        inner_intervention_rank[intervention] + 2 * inner_context_rank[context]
    ) % OUTER_FOLDS
    result = FrozenMasks(outer.astype(np.int16), inner.astype(np.int16))
    validate_masks(result)
    return result


def validate_masks(masks: FrozenMasks) -> None:
    if masks.outer_fold.shape != (ENTRIES,) or masks.inner_color.shape != (ENTRIES,):
        raise RuntimeError("LOW_RANK_MASK_AXIS_FAIL")
    seen = np.zeros(ENTRIES, dtype=np.int8)
    for fold in range(OUTER_FOLDS):
        targets = masks.outer_targets(fold)
        seen[targets] += 1
        contexts = targets // INTERVENTIONS
        interventions = targets % INTERVENTIONS
        if len(targets) not in {46, 47}:
            raise RuntimeError("LOW_RANK_OUTER_FOLD_SIZE_FAIL")
        if len(np.unique(contexts)) != len(targets):
            raise RuntimeError("LOW_RANK_OUTER_CONTEXT_BALANCE_FAIL")
        if len(np.unique(interventions)) != len(targets):
            raise RuntimeError("LOW_RANK_OUTER_INTERVENTION_BALANCE_FAIL")
        inner = masks.inner_validation(fold)
        if np.intersect1d(targets, inner).size:
            raise RuntimeError("LOW_RANK_INNER_OUTER_OVERLAP")
        if len(inner) < 43 or len(inner) > 47:
            raise RuntimeError("LOW_RANK_INNER_FOLD_SIZE_FAIL")
    if not np.all(seen == 1):
        raise RuntimeError("LOW_RANK_OOF_COVERAGE_FAIL")


def benchmark_targets(contexts: list[str], interventions: list[str]) -> np.ndarray:
    records: list[tuple[str, int]] = []
    for context, context_id in enumerate(contexts):
        for intervention, intervention_id in enumerate(interventions):
            payload = f"{context_id}|{intervention_id}".encode("utf-8")
            records.append((hashlib.sha256(payload).hexdigest(), context * INTERVENTIONS + intervention))
    ordered = np.asarray([index for _, index in sorted(records)], dtype=np.int64)
    positions = np.asarray([0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered) - 1], dtype=np.int64)
    return ordered[positions]


def write_mask_manifest(root: Path, source_root: Path) -> Path:
    root = root.resolve()
    source_root = source_root.resolve()
    split = json.loads(
        (root / "results/cgc_entrywise_compression/ENTRYWISE_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    masks = frozen_masks()
    benchmark = set(map(int, benchmark_targets(split["contexts"], split["interventions"])))
    rows: list[dict[str, object]] = []
    for index in range(ENTRIES):
        context = index // INTERVENTIONS
        intervention = index % INTERVENTIONS
        outer = int(masks.outer_fold[index])
        inner_color = int(masks.inner_color[index])
        rows.append(
            {
                "entry_index": index,
                "context_index": context,
                "context_id": split["contexts"][context],
                "intervention_index": intervention,
                "intervention_id": split["interventions"][intervention],
                "outer_fold": outer,
                "outer_fold_target_count": len(masks.outer_targets(outer)),
                "outer_observed_count": ENTRIES - len(masks.outer_targets(outer)),
                "outer_observed_fraction": (ENTRIES - len(masks.outer_targets(outer))) / ENTRIES,
                "inner_color": inner_color,
                "inner_validation_color_for_own_outer_fold": (37 * outer + 11) % OUTER_FOLDS,
                "benchmark_episode": index in benchmark,
                "mask_depends_on_outcome": False,
            }
        )
    output = root / "results/cgc_lowrank_interaction_completion/LOW_RANK_MASK_MANIFEST.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
