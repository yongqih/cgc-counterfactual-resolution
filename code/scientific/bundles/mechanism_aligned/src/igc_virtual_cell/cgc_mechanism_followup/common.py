"""Shared, outcome-agnostic helpers for the frozen mechanism follow-up."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell")
OUT = ROOT / "results" / "cgc_mechanism_followup"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment with NaNs preserved."""

    values = np.asarray(pvalues, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_hierarchical_bootstrap(
    numerator_delta: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Resample frozen context and intervention axes without model refits."""

    delta = np.asarray(numerator_delta, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if delta.shape != den.shape or delta.ndim != 2:
        raise ValueError("numerator_delta and denominator must be equal 2D context x intervention arrays")
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    nc, np_ = delta.shape
    for draw in range(draws):
        ci = rng.integers(0, nc, nc)
        pi = rng.integers(0, np_, np_)
        local_den = den[np.ix_(ci, pi)].sum()
        values[draw] = delta[np.ix_(ci, pi)].sum() / local_den if abs(local_den) > 1e-12 else np.nan
    return values

