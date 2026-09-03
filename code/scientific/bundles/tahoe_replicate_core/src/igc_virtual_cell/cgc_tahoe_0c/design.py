"""Pre-register deterministic support sequences and intervention folds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_tahoe_0c.extraction import write_json


SUPPORT_SIZES = (2, 4, 8, 16, 24, 32, 40, 49)
SUPPORT_SEQUENCES = 128
SUPPORT_BASE_SEED = 202608203
INTERVENTION_FOLDS = 5


def _seed(context: str, sequence: int) -> int:
    value = f"{SUPPORT_BASE_SEED}|{context}|{sequence}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def intervention_fold(intervention: str) -> int:
    digest = hashlib.sha256(intervention.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % INTERVENTION_FOLDS


def freeze_design(root: Path) -> dict[str, object]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    axes = json.loads(
        (root / "data/tahoe100m_plate6_14_core/axes.json").read_text(
            encoding="utf-8"
        )
    )
    contexts = axes["contexts"]
    interventions = axes["interventions"]
    rows = []
    for heldout in contexts:
        available = np.array([value for value in contexts if value != heldout], dtype=object)
        for sequence in range(SUPPORT_SEQUENCES):
            seed = _seed(heldout, sequence)
            permutation = np.random.default_rng(seed).permutation(available)
            for rank, support_context in enumerate(permutation, start=1):
                rows.append(
                    {
                        "heldout_context": heldout,
                        "sequence": sequence,
                        "sequence_seed": seed,
                        "rank": rank,
                        "support_context": support_context,
                    }
                )
    support = pd.DataFrame(rows)
    support.to_csv(result_dir / "support_sequences.csv", index=False)

    folds = pd.DataFrame(
        {
            "intervention_id": interventions,
            "fold": [intervention_fold(value) for value in interventions],
            "assignment": "sha256_first8_little_endian_mod_5",
            "outcome_balancing": False,
        }
    )
    if folds["fold"].nunique() != INTERVENTION_FOLDS:
        raise RuntimeError("Deterministic intervention folds are incomplete")
    folds.to_csv(result_dir / "intervention_folds.csv", index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_coordinate_fitting": True,
        "support_sizes": list(SUPPORT_SIZES),
        "support_sequences_per_heldout": SUPPORT_SEQUENCES,
        "support_base_seed": SUPPORT_BASE_SEED,
        "nested_prefix_design": True,
        "n49_computationally_deduplicated": True,
        "heldout_contexts": len(contexts),
        "available_support_contexts_per_heldout": len(contexts) - 1,
        "intervention_folds": INTERVENTION_FOLDS,
        "intervention_fold_assignment": "sha256 identity hash; no outcome balancing",
        "interventions": len(interventions),
        "fold_counts": {
            str(key): int(value) for key, value in folds["fold"].value_counts().sort_index().items()
        },
        "response_values_read_during_design": False,
    }
    write_json(result_dir / "design_manifest.json", manifest)
    return manifest
