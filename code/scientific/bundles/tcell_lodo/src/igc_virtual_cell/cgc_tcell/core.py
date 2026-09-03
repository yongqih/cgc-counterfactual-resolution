"""Model-free numerical primitives for the CD4 T-cell truth-operator audit."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc.core import two_way_interaction


def hash_block(label: str, blocks: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:8], "little") % blocks


def donor_partitions(donors: list[str]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(donors) != 4 or len(set(donors)) != 4:
        raise ValueError("Exactly four distinct donors are required")
    return [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def donor_interactions(delta: np.ndarray) -> np.ndarray:
    """Return P x D x S x G donor-local two-way interactions."""
    values = np.asarray(delta, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:3] != (4, 3) or not np.isfinite(values).all():
        raise ValueError("Expected finite P x 4 donors x 3 states x G tensor")
    result = np.empty_like(values)
    for donor in range(4):
        result[:, donor] = two_way_interaction(values[:, donor])
    return result


def donor_disjoint_distances(
    delta: np.ndarray,
    *,
    donors: list[str],
    states: list[str],
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Compute all 3 donor partitions x 3 stimulation contrasts.

    The returned per-target arrays have shape P x 3 contrasts for each donor
    partition; their target means exactly reconstruct the reported distances.
    """
    if len(states) != 3:
        raise ValueError("Exactly three frozen stimulation states are required")
    values = np.asarray(delta)
    if values.ndim != 4 or values.shape[1:3] != (4, 3):
        raise ValueError("Expected P x 4 donors x 3 states x G tensor")
    rows, contributions = [], []
    state_pairs = list(combinations(range(3), 2))
    for partition_index, (left_donors, right_donors) in enumerate(donor_partitions(donors)):
        partition_contributions = []
        for contrast_index, (state_a, state_b) in enumerate(state_pairs):
            # In a state contrast, the intervention mean and donor grand mean
            # cancel from Gamma. Centering the raw state difference over targets
            # is therefore algebraically identical and avoids materializing Gamma.
            difference_left = np.zeros(
                (values.shape[0], values.shape[-1]), dtype=np.float32
            )
            difference_right = np.zeros_like(difference_left)
            for donor in left_donors:
                block = np.asarray(values[:, donor, state_a]) - np.asarray(
                    values[:, donor, state_b]
                )
                block -= block.mean(axis=0, keepdims=True, dtype=np.float64).astype(
                    np.float32
                )
                difference_left += block * 0.5
            for donor in right_donors:
                block = np.asarray(values[:, donor, state_a]) - np.asarray(
                    values[:, donor, state_b]
                )
                block -= block.mean(axis=0, keepdims=True, dtype=np.float64).astype(
                    np.float32
                )
                difference_right += block * 0.5
            per_target = np.einsum(
                "pg,pg->p", difference_left, difference_right, optimize=True
            ) / values.shape[-1]
            partition_contributions.append(per_target)
            rows.append(
                {
                    "partition_index": partition_index,
                    "donor_half_a": "|".join(donors[index] for index in left_donors),
                    "donor_half_b": "|".join(donors[index] for index in right_donors),
                    "contrast_index": contrast_index,
                    "state_a": states[state_a],
                    "state_b": states[state_b],
                    "crossvalidated_distance": float(
                        per_target.astype(np.float64).mean()
                    ),
                    "positive": bool(per_target.astype(np.float64).mean() > 0),
                }
            )
        contributions.append(np.asarray(partition_contributions, dtype=np.float64).T)
    return pd.DataFrame(rows), contributions


def hierarchical_target_bootstrap(
    contributions: list[np.ndarray],
    *,
    states: list[str],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if len(contributions) != 3 or any(values.shape[1] != 3 for values in contributions):
        raise ValueError("Expected three partition arrays with three contrasts")
    rng = np.random.default_rng(seed)
    overall = np.empty(draws, dtype=np.float64)
    contrasts = np.empty((draws, 3), dtype=np.float64)
    perturbations = contributions[0].shape[0]
    for draw in range(draws):
        selected = rng.integers(perturbations, size=perturbations)
        partition_means = np.asarray(
            [values[selected].mean(axis=0) for values in contributions], dtype=np.float64
        )
        contrasts[draw] = partition_means.mean(axis=0)
        overall[draw] = contrasts[draw].mean()
    rows = []
    for contrast_index, (left, right) in enumerate(combinations(range(3), 2)):
        values = contrasts[:, contrast_index]
        rows.append(
            {
                "statistic": "contrast_mean",
                "state_a": states[left],
                "state_b": states[right],
                "draws": draws,
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_above_zero": float(np.mean(values > 0)),
            }
        )
    rows.append(
        {
            "statistic": "overall_mean",
            "state_a": "",
            "state_b": "",
            "draws": draws,
            "bootstrap_mean": float(overall.mean()),
            "ci_low": float(np.quantile(overall, 0.025)),
            "ci_high": float(np.quantile(overall, 0.975)),
            "probability_above_zero": float(np.mean(overall > 0)),
        }
    )
    return pd.DataFrame(rows)


def stratified_permutation_indices(
    nuisance: np.ndarray,
    *,
    draws: int,
    bins: int,
    seed: int,
    output_path: Path | None = None,
) -> np.ndarray:
    """Freeze independent target-label permutations for every donor/state unit.

    ``nuisance`` is U x P (normally mean pseudobulk cell count). Labels are
    permuted only within deterministic quantile bins. The returned map has
    shape U x draws x P and maps each output target to an input target.
    """
    values = np.asarray(nuisance, dtype=np.float64)
    if values.ndim != 2 or draws < 1 or bins < 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite U x P nuisance values and positive draws/bins")
    shape = (values.shape[0], draws, values.shape[1])
    if output_path is None:
        result = np.empty(shape, dtype=np.int32)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = np.memmap(output_path, mode="w+", dtype=np.int32, shape=shape)
    rng = np.random.default_rng(seed)
    for unit in range(values.shape[0]):
        order = np.argsort(values[unit], kind="stable")
        labels = np.empty(values.shape[1], dtype=np.int16)
        labels[order] = np.minimum(
            np.arange(values.shape[1], dtype=np.int64) * bins // values.shape[1],
            bins - 1,
        )
        members = [np.flatnonzero(labels == block) for block in range(bins)]
        for draw in range(draws):
            result[unit, draw] = np.arange(values.shape[1], dtype=np.int32)
            for indices in members:
                result[unit, draw, indices] = rng.permutation(indices)
    if isinstance(result, np.memmap):
        result.flush()
    return result


def null_term_plan() -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Return unique cross-Gram terms and coefficients for 3 x 3 statistics."""
    term_coefficients: dict[tuple[int, int, int, int], np.ndarray] = {}
    state_pairs = list(combinations(range(3), 2))
    for partition_index, (left, right) in enumerate(
        donor_partitions(["0", "1", "2", "3"])
    ):
        for contrast_index, (state_a, state_b) in enumerate(state_pairs):
            statistic = partition_index * 3 + contrast_index
            for donor_left in left:
                for donor_right in right:
                    for left_state, right_state, sign in (
                        (state_a, state_a, 1.0),
                        (state_a, state_b, -1.0),
                        (state_b, state_a, -1.0),
                        (state_b, state_b, 1.0),
                    ):
                        if donor_left < donor_right:
                            key = (donor_left, left_state, donor_right, right_state)
                        else:
                            key = (donor_right, right_state, donor_left, left_state)
                        if key not in term_coefficients:
                            term_coefficients[key] = np.zeros(9, dtype=np.float64)
                        term_coefficients[key][statistic] += sign / 4.0
    terms = sorted(term_coefficients)
    return terms, np.stack([term_coefficients[key] for key in terms])


def permutation_null_from_crossgrams(
    centered: np.ndarray,
    permutations: np.ndarray,
    *,
    batch_draws: int = 25,
    device: str = "cpu",
) -> np.ndarray:
    """Calculate the exact nuisance null from target cross-Gram matrices.

    ``centered`` is P x 4 x 3 x G and ``permutations`` is 12 x R x P in
    donor-major/state-minor order. A memory map is accepted for either input.
    """
    values = np.asarray(centered)
    maps = np.asarray(permutations)
    if values.ndim != 4 or values.shape[1:3] != (4, 3):
        raise ValueError("centered must have shape P x 4 x 3 x G")
    if maps.ndim != 3 or maps.shape[0] != 12 or maps.shape[2] != values.shape[0]:
        raise ValueError("permutations must have shape 12 x draws x P")
    if maps.min() < 0 or maps.max() >= values.shape[0]:
        raise ValueError("permutation index out of bounds")
    terms, coefficients = null_term_plan()
    result = np.zeros((maps.shape[1], 9), dtype=np.float64)
    use_torch = device != "cpu"
    if use_torch:
        import torch

        torch_device = torch.device(device)
    for term_index, (donor_a, state_a, donor_b, state_b) in enumerate(terms):
        left = np.ascontiguousarray(values[:, donor_a, state_a], dtype=np.float32)
        right = np.ascontiguousarray(values[:, donor_b, state_b], dtype=np.float32)
        unit_a, unit_b = donor_a * 3 + state_a, donor_b * 3 + state_b
        if use_torch:
            left_t = torch.from_numpy(left).to(torch_device)
            right_t = torch.from_numpy(right).to(torch_device)
            gram_t = left_t @ right_t.T
            del left_t, right_t
            term_sums = np.empty(maps.shape[1], dtype=np.float64)
            for start in range(0, maps.shape[1], batch_draws):
                stop = min(start + batch_draws, maps.shape[1])
                index_a = torch.from_numpy(
                    np.ascontiguousarray(maps[unit_a, start:stop])
                ).to(torch_device)
                index_b = torch.from_numpy(
                    np.ascontiguousarray(maps[unit_b, start:stop])
                ).to(torch_device)
                gathered = gram_t[index_a, index_b].sum(dim=1, dtype=torch.float64)
                term_sums[start:stop] = gathered.cpu().numpy()
            del gram_t
        else:
            gram = left @ right.T
            term_sums = np.empty(maps.shape[1], dtype=np.float64)
            for start in range(0, maps.shape[1], batch_draws):
                stop = min(start + batch_draws, maps.shape[1])
                term_sums[start:stop] = gram[
                    maps[unit_a, start:stop], maps[unit_b, start:stop]
                ].sum(axis=1, dtype=np.float64)
            del gram
        result += term_sums[:, None] * coefficients[term_index][None, :]
    result /= float(values.shape[0] * values.shape[-1])
    return result


def crossmeasurement_context_distances(
    left: np.ndarray, right: np.ndarray
) -> pd.DataFrame:
    """Full context geometry from independent P x C x G measurements."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.ndim != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected two finite, shape-matched P x C x G tensors")
    rows = []
    for context_a, context_b in combinations(range(a.shape[1]), 2):
        difference_a = np.asarray(a[:, context_a]) - np.asarray(a[:, context_b])
        difference_b = np.asarray(b[:, context_a]) - np.asarray(b[:, context_b])
        difference_a -= difference_a.mean(axis=0, keepdims=True, dtype=np.float64).astype(
            np.float32
        )
        difference_b -= difference_b.mean(axis=0, keepdims=True, dtype=np.float64).astype(
            np.float32
        )
        per_target = np.einsum("pg,pg->p", difference_a, difference_b, optimize=True) / a.shape[-1]
        rows.append(
            {
                "context_a": context_a,
                "context_b": context_b,
                "crossvalidated_distance": float(per_target.astype(np.float64).mean()),
                "positive": bool(per_target.astype(np.float64).mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def crossmeasurement_permutation_null(
    left: np.ndarray,
    right: np.ndarray,
    permutations: np.ndarray,
    *,
    batch_draws: int = 25,
    device: str = "cpu",
) -> np.ndarray:
    """Exact full-context label null for independent response measurements."""
    a, b = np.asarray(left), np.asarray(right)
    maps = np.asarray(permutations)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Expected shape-matched P x C x G tensors")
    if maps.shape[:1] != (a.shape[1],) or maps.shape[2] != a.shape[0]:
        raise ValueError("Permutation shape must be C x draws x P")
    pairs = list(combinations(range(a.shape[1]), 2))
    result = np.zeros((maps.shape[1], len(pairs)), dtype=np.float64)
    # Each term contributes to one or more (c,d) distances.
    plan: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for pair_index, (context_a, context_b) in enumerate(pairs):
        for term, coefficient in (
            ((context_a, context_a), 1.0),
            ((context_a, context_b), -1.0),
            ((context_b, context_a), -1.0),
            ((context_b, context_b), 1.0),
        ):
            plan.setdefault(term, []).append((pair_index, coefficient))
    use_torch = device != "cpu"
    if use_torch:
        import torch

        torch_device = torch.device(device)
    for (context_a, context_b), destinations in sorted(plan.items()):
        x = np.ascontiguousarray(a[:, context_a], dtype=np.float32)
        y = np.ascontiguousarray(b[:, context_b], dtype=np.float32)
        x -= x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        y -= y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        sums = np.empty(maps.shape[1], dtype=np.float64)
        if use_torch:
            x_t, y_t = torch.from_numpy(x).to(torch_device), torch.from_numpy(y).to(torch_device)
            gram_t = x_t @ y_t.T
            del x_t, y_t
            for start in range(0, maps.shape[1], batch_draws):
                stop = min(start + batch_draws, maps.shape[1])
                index_a = torch.from_numpy(np.ascontiguousarray(maps[context_a, start:stop])).to(
                    torch_device
                )
                index_b = torch.from_numpy(np.ascontiguousarray(maps[context_b, start:stop])).to(
                    torch_device
                )
                sums[start:stop] = gram_t[index_a, index_b].sum(
                    dim=1, dtype=torch.float64
                ).cpu().numpy()
            del gram_t
        else:
            gram = x @ y.T
            for start in range(0, maps.shape[1], batch_draws):
                stop = min(start + batch_draws, maps.shape[1])
                sums[start:stop] = gram[
                    maps[context_a, start:stop], maps[context_b, start:stop]
                ].sum(axis=1, dtype=np.float64)
        for pair_index, coefficient in destinations:
            result[:, pair_index] += coefficient * sums
    result /= float(a.shape[0] * a.shape[-1])
    return result
