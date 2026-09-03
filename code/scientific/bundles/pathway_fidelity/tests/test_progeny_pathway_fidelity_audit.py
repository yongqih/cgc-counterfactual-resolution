from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "progeny_pathway_fidelity_audit", ROOT / "scripts/progeny_pathway_fidelity_audit.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_but_one_centering_identity() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(2, 50, 3, 4))
    observed = MODULE.all_but_one_deltas(values)
    expected = np.empty_like(observed)
    for plate in range(2):
        for target in range(50):
            sources = [index for index in range(50) if index != target]
            expected[plate, target] = values[plate, target] - values[plate, sources].mean(axis=0)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)


def test_orthogonal_projection_uses_span_not_raw_overlapping_scores(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "GENE_COUNT", 4)
    monkeypatch.setattr(MODULE, "PATHWAYS", ("a", "b"))
    w = np.asarray([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    q, audit = MODULE.orthogonal_basis(w)
    assert audit["numerical_rank"] == 2
    projector = q @ q.T
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-14, rtol=0)
    np.testing.assert_allclose(projector @ w, w, atol=1e-14, rtol=0)
