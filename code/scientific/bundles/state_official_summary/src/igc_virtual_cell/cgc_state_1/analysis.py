from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def leave_one_context_mean(delta: np.ndarray) -> np.ndarray:
    """Return the target-excluding context mean for a C x P x G tensor."""
    if delta.ndim != 3 or delta.shape[0] < 2:
        raise ValueError("delta must be a C x P x G tensor with at least two contexts")
    return (delta.sum(axis=0, keepdims=True) - delta) / (delta.shape[0] - 1)


def recovery_statistics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Deterministic recovery metrics relative to the zero-interaction baseline."""
    t = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    energy = float(t @ t)
    pred_energy = float(p @ p)
    dot = float(t @ p)
    error = float(np.square(t - p).sum())
    return {
        "truth_energy": energy,
        "prediction_energy": pred_energy,
        "error_energy": error,
        "g": 1.0 - error / energy if energy else np.nan,
        "alpha": dot / energy if energy else np.nan,
        "kappa": pred_energy / energy if energy else np.nan,
        "cosine": dot / np.sqrt(energy * pred_energy) if energy and pred_energy else np.nan,
        "pearson": float(stats.pearsonr(t, p).statistic) if np.std(t) and np.std(p) else np.nan,
    }


def bootstrap_recovery(
    truth: np.ndarray,
    prediction: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Hierarchically resample frozen context x intervention energy contributions."""
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("truth and prediction must be equal C x P x G tensors")
    numerator = np.square(truth - prediction).sum(axis=2)
    denominator = np.square(truth).sum(axis=2)
    contexts, interventions = numerator.shape
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        c = rng.integers(0, contexts, size=contexts)
        p = rng.integers(0, interventions, size=interventions)
        den = denominator[np.ix_(c, p)].sum()
        values[draw] = 1.0 - numerator[np.ix_(c, p)].sum() / den if den else np.nan
    return values


def correspondence_null(
    truth: np.ndarray,
    prediction: np.ndarray,
    axis: str,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Permute context or intervention correspondence using precomputed dot products."""
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("truth and prediction must be equal C x P x G tensors")
    truth_energy = float(np.square(truth).sum())
    pred_energy = float(np.square(prediction).sum())
    if axis == "context":
        cross = np.einsum("cpg,dpg->cd", truth, prediction, optimize=True)
        n = truth.shape[0]
        derangements = np.array([p for p in itertools.permutations(range(n)) if all(i != p[i] for i in range(n))], dtype=int)
        rng = np.random.default_rng(seed)
        chosen = derangements[rng.integers(0, len(derangements), size=draws)]
        dots = np.array([cross[np.arange(n), row].sum() for row in chosen])
    elif axis == "intervention":
        cross = np.einsum("cpg,cqg->pq", truth, prediction, optimize=True)
        n = truth.shape[1]
        rng = np.random.default_rng(seed)
        dots = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            permutation = rng.permutation(n)
            while np.any(permutation == np.arange(n)):
                permutation = rng.permutation(n)
            dots[draw] = cross[np.arange(n), permutation].sum()
    else:
        raise ValueError("axis must be context or intervention")
    errors = truth_energy + pred_energy - 2.0 * dots
    return 1.0 - errors / truth_energy


def pairwise_context_geometry(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pair labels and distances for every intervention and context pair."""
    contexts, interventions, _ = tensor.shape
    pairs = np.array(list(itertools.combinations(range(contexts), 2)), dtype=int)
    distances = np.empty((interventions, len(pairs)), dtype=np.float64)
    for index, (left, right) in enumerate(pairs):
        distances[:, index] = np.linalg.norm(tensor[left] - tensor[right], axis=1)
    return pairs, distances


def spectral_summary(tensor: np.ndarray) -> dict[str, float]:
    matrix = tensor.reshape(-1, tensor.shape[-1]).astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    gram = matrix @ matrix.T
    values = np.linalg.eigvalsh(gram)
    values = np.maximum(values, 0)[::-1]
    total = values.sum()
    fractions = values / total if total else values
    cumulative = np.cumsum(fractions)
    positive = fractions[fractions > 0]
    entropy_rank = float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else np.nan
    return {
        "pc1_fraction": float(fractions[0]) if len(fractions) else np.nan,
        "pc80": int(np.searchsorted(cumulative, 0.8) + 1) if total else 0,
        "entropy_effective_rank": entropy_rank,
    }
