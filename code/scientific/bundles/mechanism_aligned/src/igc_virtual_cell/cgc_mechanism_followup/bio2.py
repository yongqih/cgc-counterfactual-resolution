"""Frozen CGC-BIO-2 hierarchical attribution analysis.

The module implements the protocol frozen at ``de548ac``.  Importing it is
side-effect free: real BIO-2 outcomes are read only by :func:`run_bio2`.
Pairwise inference is intervention-clustered throughout; no naive pairwise
standard errors are computed.
"""

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .common import OUT, ROOT, SOURCE_ROOT, bh_adjust, git, sha256, write_json


SEED = 7_201
N_DRAW = 10_000
EPS = 1e-12
LEVELS = ("BROAD_ACTION", "MOA", "TARGET_FAMILY", "EXACT_TARGET")
DIRECTIONS = ("plate6_to_plate14", "plate14_to_plate6", "pooled")
REPRESENTATIONS = ("Gene", "PROGENy", "CollecTRI")
EXPECTED_GENE_HASH = "6b46053324175bfc34b2534bb59ae5519a1aa47b504ddeb2887956953f222228"
EXPECTED_PROGRAM_HASH = "b481b44f29a22227527e9e9c05c48dbc203e0b33a48406e49b0bb7557ca60999"
EXPECTED_ANNOTATION_HASH = "33efd4cebbf7965bd20bc90a44b2b9d917b83740dad1991979d9e1dbc7d3ab59"


@dataclass(frozen=True)
class PairAxes:
    """Canonical unordered intervention-pair axes."""

    left: np.ndarray
    right: np.ndarray

    @classmethod
    def build(cls, interventions: int) -> "PairAxes":
        left, right = np.triu_indices(interventions, 1)
        return cls(left.astype(np.int32), right.astype(np.int32))

    def subset(self, keep_interventions: np.ndarray) -> np.ndarray:
        keep = np.asarray(keep_interventions, dtype=bool)
        return keep[self.left] & keep[self.right]

    def lookup(self, interventions: int) -> np.ndarray:
        table = np.full((interventions, interventions), -1, dtype=np.int32)
        table[self.left, self.right] = np.arange(len(self.left), dtype=np.int32)
        table[self.right, self.left] = np.arange(len(self.left), dtype=np.int32)
        return table


@dataclass
class CrossPlateStatistics:
    """Cross-plate Gram matrices and plate-specific squared norms."""

    cross: np.ndarray
    residual_norm6_sq: np.ndarray
    residual_norm14_sq: np.ndarray
    gamma_norm6_sq: np.ndarray
    gamma_norm14_sq: np.ndarray

    def similarities(self, pairs: PairAxes) -> np.ndarray:
        i, j = pairs.left, pairs.right
        d6 = self.cross[i, j] / np.sqrt(
            np.maximum(self.residual_norm6_sq[i] * self.residual_norm14_sq[j], EPS)
        )
        d14 = self.cross[j, i] / np.sqrt(
            np.maximum(self.residual_norm14_sq[i] * self.residual_norm6_sq[j], EPS)
        )
        return np.column_stack([d6, d14, 0.5 * (d6 + d14)])

    def without(self, contribution: "CrossPlateStatistics") -> "CrossPlateStatistics":
        return CrossPlateStatistics(
            cross=self.cross - contribution.cross,
            residual_norm6_sq=np.maximum(self.residual_norm6_sq - contribution.residual_norm6_sq, 0),
            residual_norm14_sq=np.maximum(self.residual_norm14_sq - contribution.residual_norm14_sq, 0),
            gamma_norm6_sq=np.maximum(self.gamma_norm6_sq - contribution.gamma_norm6_sq, 0),
            gamma_norm14_sq=np.maximum(self.gamma_norm14_sq - contribution.gamma_norm14_sq, 0),
        )


@dataclass
class GeneStatistics:
    total: CrossPlateStatistics
    lineage: dict[str, CrossPlateStatistics]
    relation_kernels: dict[str, np.ndarray]


@dataclass(frozen=True)
class FrozenInputs:
    annotations: pd.DataFrame
    relation_manifest: pd.DataFrame
    context_mapping: pd.DataFrame
    contexts: list[str]
    interventions: list[str]
    lineages: np.ndarray
    program_manifest: pd.DataFrame
    gene_gamma: np.ndarray
    gene_residual: np.ndarray
    program_gamma: np.ndarray
    program_residual: np.ndarray


def split_set(value: object) -> frozenset[str]:
    if pd.isna(value) or not str(value).strip():
        return frozenset()
    return frozenset(item.strip() for item in str(value).replace(",", ";").split(";") if item.strip())


def _valid_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "unclear", "nan", "none"} else text


def open_stored_npz_member(path: Path, member: str) -> np.memmap:
    """Memory-map one uncompressed NPY member inside an NPZ archive.

    BIO-1's 1.9 GB gene cache is ZIP_STORED.  ``numpy.load(...,
    mmap_mode='r')`` does not mmap NPZ members, so this parser locates the NPY
    payload without allocating the full array.  It fails closed for compressed
    or encrypted members.
    """

    name = member if member.endswith(".npy") else f"{member}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            raise ValueError(f"{name} is not an unencrypted ZIP_STORED member")
        header_offset = info.header_offset
    with path.open("rb") as handle:
        handle.seek(header_offset)
        local = handle.read(30)
        if len(local) != 30:
            raise ValueError("Truncated ZIP local header")
        signature, *_, name_length, extra_length = struct.unpack("<IHHHHHIIIHH", local)
        if signature != 0x04034B50:
            raise ValueError("Invalid ZIP local header signature")
        handle.seek(name_length + extra_length, 1)
        major, minor = np.lib.format.read_magic(handle)
        if (major, minor) == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif (major, minor) in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version {(major, minor)}")
        offset = handle.tell()
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape, order="F" if fortran else "C")


def _assert_sha(path: Path, expected: str, verify: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if verify:
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen hash mismatch for {path}: {observed} != {expected}")


def load_frozen_inputs(verify_hashes: bool = True) -> FrozenInputs:
    """Load immutable BIO-2 inputs without calculating any outcome statistic."""

    provenance = json.loads((OUT / "BIO2_ANNOTATION_PROVENANCE.json").read_text(encoding="utf-8"))
    gene_path = Path(provenance["sources"]["frozen_gene_residual"]["path"])
    program_path = Path(provenance["sources"]["frozen_program_residual"]["path"])
    drug_path = ROOT / "results/cgc_tahoe_bio1/drug_target_moa_manifest.csv"
    _assert_sha(gene_path, EXPECTED_GENE_HASH, verify_hashes)
    _assert_sha(program_path, EXPECTED_PROGRAM_HASH, verify_hashes)
    _assert_sha(drug_path, EXPECTED_ANNOTATION_HASH, verify_hashes)

    annotations = pd.read_csv(OUT / "BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv").sort_values("intervention_axis")
    if annotations.intervention_axis.tolist() != list(range(93)):
        raise RuntimeError("Frozen intervention axes are not contiguous")
    meta = pd.read_csv(ROOT / "results/cgc_tahoe_0i/pseudobulk_sample_metadata.csv")
    contexts = (
        meta[["context_axis", "cell_line_id"]].drop_duplicates().sort_values("context_axis").cell_line_id.tolist()
    )
    interventions = (
        meta.loc[~meta.is_control.astype(bool), ["sample_axis", "intervention_id"]]
        .drop_duplicates()
        .sort_values("sample_axis")
        .intervention_id.tolist()
    )
    if interventions != annotations.intervention_id.tolist() or len(contexts) != 50:
        raise RuntimeError("BIO-2 annotation and frozen response axes disagree")
    mapping = pd.read_csv(ROOT / "results/cgc_tahoe_bio1/cellline_depmap_mapping.csv")
    mapping = mapping.set_index("cell_line_id").loc[contexts].reset_index()
    lineages = mapping.OncotreeLineage.fillna(mapping.Organ).fillna("UNKNOWN").astype(str).to_numpy()

    gene_gamma = open_stored_npz_member(gene_path, "gamma")
    gene_residual = open_stored_npz_member(gene_path, "residual")
    with np.load(program_path, allow_pickle=False) as archive:
        program_gamma = np.asarray(archive["gamma"], dtype=np.float32)
        program_residual = np.asarray(archive["residual"], dtype=np.float32)
    expected_gene = (2, 50, 93, 25_695)
    expected_program = (2, 50, 93, 2_822)
    if gene_gamma.shape != expected_gene or gene_residual.shape != expected_gene:
        raise RuntimeError("Frozen BIO-1 gene cache shape mismatch")
    if program_gamma.shape != expected_program or program_residual.shape != expected_program:
        raise RuntimeError("Frozen BIO-1 program cache shape mismatch")
    relations = pd.read_csv(ROOT / "results/cgc_tahoe_bio1/genotype_drug_match_manifest.csv")
    relations = relations[relations.formal_eligible.astype(bool)].reset_index(drop=True)
    if len(relations) != 424 or relations.relation_id.duplicated().any():
        raise RuntimeError("Frozen 424-relation cohort changed")
    program_manifest = pd.read_csv(ROOT / "results/cgc_tahoe_bio1/program_manifest.csv")
    return FrozenInputs(
        annotations=annotations.reset_index(drop=True),
        relation_manifest=relations,
        context_mapping=mapping,
        contexts=contexts,
        interventions=interventions,
        lineages=lineages,
        program_manifest=program_manifest,
        gene_gamma=gene_gamma,
        gene_residual=gene_residual,
        program_gamma=program_gamma,
        program_residual=program_residual,
    )


def annotation_relation_matrices(annotations: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compile the four immutable partial-order indicators."""

    n = len(annotations)
    broad = np.asarray([_valid_label(x) for x in annotations.moa_broad], dtype=object)
    fine = np.asarray([_valid_label(x) for x in annotations.moa_fine], dtype=object)
    targets = [split_set(x) for x in annotations.exact_targets]
    families = [split_set(x) for x in annotations.hgnc_target_family_ids]
    same_broad = (broad[:, None] == broad[None, :]) & (broad[:, None] != "")
    same_fine = (fine[:, None] == fine[None, :]) & (fine[:, None] != "")
    exact = np.zeros((n, n), dtype=bool)
    family = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            is_exact = bool(targets[i]) and targets[i] == targets[j]
            exact[i, j] = exact[j, i] = is_exact
            is_family = bool(families[i].intersection(families[j])) and not is_exact
            family[i, j] = family[j, i] = is_family
    np.fill_diagonal(same_broad, False)
    np.fill_diagonal(same_fine, False)
    return {
        "BROAD_ACTION": same_broad,
        "MOA": same_fine,
        "TARGET_FAMILY": family,
        "EXACT_TARGET": exact,
    }


def pair_indicators(annotations: pd.DataFrame, pairs: PairAxes) -> np.ndarray:
    matrices = annotation_relation_matrices(annotations)
    return np.column_stack([matrices[level][pairs.left, pairs.right] for level in LEVELS]).astype(np.float64)


def _empty_statistics(interventions: int) -> CrossPlateStatistics:
    return CrossPlateStatistics(
        cross=np.zeros((interventions, interventions), dtype=np.float64),
        residual_norm6_sq=np.zeros(interventions, dtype=np.float64),
        residual_norm14_sq=np.zeros(interventions, dtype=np.float64),
        gamma_norm6_sq=np.zeros(interventions, dtype=np.float64),
        gamma_norm14_sq=np.zeros(interventions, dtype=np.float64),
    )


def _accumulate_statistics(
    target: CrossPlateStatistics,
    gamma6: np.ndarray,
    gamma14: np.ndarray,
    residual6: np.ndarray,
    residual14: np.ndarray,
) -> None:
    target.cross += np.einsum("cpg,cqg->pq", residual6, residual14, dtype=np.float64, optimize=True)
    target.residual_norm6_sq += np.einsum("cpg,cpg->p", residual6, residual6, dtype=np.float64, optimize=True)
    target.residual_norm14_sq += np.einsum("cpg,cpg->p", residual14, residual14, dtype=np.float64, optimize=True)
    target.gamma_norm6_sq += np.einsum("cpg,cpg->p", gamma6, gamma6, dtype=np.float64, optimize=True)
    target.gamma_norm14_sq += np.einsum("cpg,cpg->p", gamma14, gamma14, dtype=np.float64, optimize=True)


def compute_cross_plate_statistics(
    gamma: np.ndarray,
    residual: np.ndarray,
    lineages: Sequence[str] | None = None,
    feature_block: int = 512,
) -> tuple[CrossPlateStatistics, dict[str, CrossPlateStatistics]]:
    """Accumulate fingerprint sufficient statistics without concatenation."""

    if gamma.shape != residual.shape or gamma.ndim != 4 or gamma.shape[0] != 2:
        raise ValueError("gamma and residual must have shape 2 x context x intervention x feature")
    contexts, interventions, features = gamma.shape[1:]
    if lineages is not None and len(lineages) != contexts:
        raise ValueError("lineage axis length mismatch")
    total = _empty_statistics(interventions)
    unique = sorted(set(map(str, lineages))) if lineages is not None else []
    group_stats = {name: _empty_statistics(interventions) for name in unique}
    masks = {name: np.flatnonzero(np.asarray(lineages, dtype=object) == name) for name in unique}
    for lo in range(0, features, feature_block):
        hi = min(features, lo + feature_block)
        g6 = np.asarray(gamma[0, :, :, lo:hi], dtype=np.float32)
        g14 = np.asarray(gamma[1, :, :, lo:hi], dtype=np.float32)
        u6 = np.asarray(residual[0, :, :, lo:hi], dtype=np.float32)
        u14 = np.asarray(residual[1, :, :, lo:hi], dtype=np.float32)
        _accumulate_statistics(total, g6, g14, u6, u14)
        for name, axes in masks.items():
            _accumulate_statistics(group_stats[name], g6[axes], g14[axes], u6[axes], u14[axes])
    return total, group_stats


def _robust_scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, median)
    mad = np.nanmedian(np.abs(filled - median), axis=0)
    constant = ~np.isfinite(mad) | (mad <= EPS)
    scale = np.where(constant, 1.0, mad)
    scaled = (filled - median) / scale
    scaled[:, constant] = 0.0
    return scaled, median, mad, constant


def pair_nuisance_features(
    annotations: pd.DataFrame,
    statistics: CrossPlateStatistics,
    pairs: PairAxes,
    available_context_count: int = 50,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Construct the exact ten frozen matching nuisance features."""

    i, j = pairs.left, pairs.right
    log_gamma = 0.25 * (
        np.log(np.maximum(statistics.gamma_norm6_sq, EPS))
        + np.log(np.maximum(statistics.gamma_norm14_sq, EPS))
    )
    log_residual = 0.25 * (
        np.log(np.maximum(statistics.residual_norm6_sq, EPS))
        + np.log(np.maximum(statistics.residual_norm14_sq, EPS))
    )
    dose = pd.to_numeric(annotations.concentration, errors="coerce").to_numpy(float)
    unit = annotations.concentration_unit.fillna("").astype(str).to_numpy()
    log_dose = np.log10(np.maximum(dose, EPS))
    dose_difference = np.where(unit[i] == unit[j], np.abs(log_dose[i] - log_dose[j]), np.nan)
    target_count = pd.to_numeric(annotations.n_exact_targets, errors="coerce").fillna(0).to_numpy(float)
    raw = np.column_stack(
        [
            np.full(len(i), available_context_count, dtype=float),
            np.full(len(i), available_context_count, dtype=float),
            0.5 * (log_gamma[i] + log_gamma[j]),
            np.abs(log_gamma[i] - log_gamma[j]),
            0.5 * (log_residual[i] + log_residual[j]),
            np.abs(log_residual[i] - log_residual[j]),
            dose_difference,
            np.zeros(len(i), dtype=float),
            0.5 * (target_count[i] + target_count[j]),
            np.abs(target_count[i] - target_count[j]),
        ]
    )
    names = [
        "context_overlap",
        "available_context_count",
        "mean_log_global_response_magnitude",
        "absolute_log_global_response_magnitude_difference",
        "mean_log_residual_energy",
        "absolute_log_residual_energy_difference",
        "absolute_log10_dose_difference_within_unit",
        "lineage_composition_distance",
        "mean_target_annotation_count",
        "absolute_target_annotation_count_difference",
    ]
    scaled, median, mad, constant = _robust_scale(raw)
    audit = {
        "names": names,
        "median": median.tolist(),
        "mad": mad.tolist(),
        "constant": constant.tolist(),
        "scaling": "median/raw-MAD; zero-MAD columns retained as zeros",
    }
    return scaled, names, audit


def _solve(gram: np.ndarray, cross: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(gram, rcond=1e-10) @ cross


def fit_conditional_ladder(
    nuisance: np.ndarray,
    indicators: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit all four nested coefficients; return level x outcome."""

    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    if nuisance.shape[0] != n or indicators.shape != (n, 4):
        raise ValueError("Pair design axes disagree")
    base = np.column_stack([np.ones(n, dtype=float), nuisance])
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    result = np.full((4, y.shape[1]), np.nan, dtype=np.float64)
    for level in range(4):
        design = np.column_stack([base, indicators[:, : level + 1]])
        gram = design.T @ (w[:, None] * design)
        cross = design.T @ (w[:, None] * y)
        result[level] = _solve(gram, cross)[-1]
    return result


def qap_strata(annotations: pd.DataFrame) -> np.ndarray:
    """Target-count x dose-unit/tertile strata from the frozen rules."""

    count = pd.to_numeric(annotations.n_exact_targets, errors="coerce").fillna(0).to_numpy(int)
    count_bin = np.digitize(count, [1, 2, 4], right=False)
    unit = annotations.concentration_unit.fillna("MISSING").astype(str).to_numpy()
    dose = pd.to_numeric(annotations.concentration, errors="coerce").to_numpy(float)
    tertile = np.full(len(annotations), -1, dtype=int)
    for label in sorted(set(unit)):
        axes = np.flatnonzero(unit == label)
        valid = axes[np.isfinite(dose[axes]) & (dose[axes] > 0)]
        if len(valid):
            values = np.log10(dose[valid])
            cuts = np.unique(np.quantile(values, [1 / 3, 2 / 3]))
            tertile[valid] = np.searchsorted(cuts, values, side="right")
    return np.asarray([f"T{a}|{u}|D{d}" for a, u, d in zip(count_bin, unit, tertile, strict=True)], dtype=object)


def qap_permutations(annotations: pd.DataFrame, draws: int, seed: int = SEED) -> np.ndarray:
    strata = qap_strata(annotations)
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
    rng = np.random.default_rng(seed)
    permutations = np.broadcast_to(np.arange(len(annotations), dtype=np.int32), (draws, len(annotations))).copy()
    for draw in range(draws):
        for axes in groups:
            if len(axes) > 1:
                permutations[draw, axes] = axes[rng.permutation(len(axes))]
    return permutations


def qap_conditional_null(
    annotations: pd.DataFrame,
    pairs: PairAxes,
    nuisance: np.ndarray,
    outcomes: np.ndarray,
    draws: int = N_DRAW,
    seed: int = SEED,
) -> np.ndarray:
    """Whole-annotation-row stratified QAP null (draw x level x outcome)."""

    matrices = annotation_relation_matrices(annotations)
    permutations = qap_permutations(annotations, draws, seed)
    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    null = np.empty((draws, 4, y.shape[1]), dtype=np.float64)
    base = np.column_stack([np.ones(len(y), dtype=float), nuisance])
    base_pinv = np.linalg.pinv(base, rcond=1e-10)
    y_residual = y - base @ (base_pinv @ y)
    for draw, perm in enumerate(permutations):
        z = np.column_stack(
            [matrices[level][perm[pairs.left], perm[pairs.right]] for level in LEVELS]
        ).astype(float)
        z_residual = z - base @ (base_pinv @ z)
        for level in range(4):
            local = z_residual[:, : level + 1]
            null[draw, level] = _solve(local.T @ local, local.T @ y_residual)[-1]
    return null


def cluster_bootstrap_ladder(
    nuisance: np.ndarray,
    indicators: np.ndarray,
    outcomes: np.ndarray,
    pairs: PairAxes,
    interventions: int,
    draws: int = N_DRAW,
    seed: int = SEED,
    batch_size: int = 128,
) -> np.ndarray:
    """Intervention-cluster bootstrap using induced-pair multiplicities."""

    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    result = np.empty((draws, 4, y.shape[1]), dtype=np.float64)
    base = np.column_stack([np.ones(len(y), dtype=float), nuisance])
    design = np.column_stack([base, indicators])
    columns = design.shape[1]
    # Every weighted fit is recovered from one batch matrix multiplication.
    xx_terms = (design[:, :, None] * design[:, None, :]).reshape(len(design), columns * columns)
    xy_terms = (design[:, :, None] * y[:, None, :]).reshape(len(design), columns * y.shape[1])
    moments = np.column_stack([xx_terms, xy_terms])
    rng = np.random.default_rng(seed)
    for lo in range(0, draws, batch_size):
        hi = min(draws, lo + batch_size)
        counts = rng.multinomial(interventions, np.full(interventions, 1 / interventions), size=hi - lo)
        weights = counts[:, pairs.left] * counts[:, pairs.right]
        sufficient = weights @ moments
        for local, row in enumerate(sufficient):
            gram = row[: columns * columns].reshape(columns, columns)
            cross = row[columns * columns :].reshape(columns, y.shape[1])
            for level in range(4):
                selected = np.r_[np.arange(base.shape[1]), base.shape[1] + np.arange(level + 1)]
                local_gram = gram[np.ix_(selected, selected)]
                local_cross = cross[selected]
                result[lo + local, level] = _solve(local_gram, local_cross)[-1]
    return result


def _endpoint_annotation_availability(annotations: pd.DataFrame, level: str) -> np.ndarray:
    if level == "BROAD_ACTION":
        return np.asarray([bool(_valid_label(x)) for x in annotations.moa_broad])
    if level == "MOA":
        return np.asarray([bool(_valid_label(x)) for x in annotations.moa_fine])
    if level == "TARGET_FAMILY":
        return annotations.target_family_annotation_available.astype(bool).to_numpy()
    return annotations.target_annotation_available.astype(bool).to_numpy()


def match_lower_resolution_controls(
    annotations: pd.DataFrame,
    pairs: PairAxes,
    indicators: np.ndarray,
    nuisance: np.ndarray,
    level: str,
    controls_per_pair: int = 5,
    maximum_distance: float = 2.5,
) -> pd.DataFrame:
    """Deterministic nearest lower-resolution controls for one ladder level."""

    k = LEVELS.index(level)
    positive = np.flatnonzero(indicators[:, k] > 0)
    broad, fine, family, exact = (indicators[:, axis] > 0 for axis in range(4))
    available = _endpoint_annotation_availability(annotations, level)
    endpoint_available = available[pairs.left] & available[pairs.right]
    different = ~np.asarray(indicators[:, k], dtype=bool) & endpoint_available
    if level == "BROAD_ACTION":
        tiers = [("different broad action", different)]
    elif level == "MOA":
        tiers = [
            ("same broad action and different fine MOA", different & broad),
            ("different fine MOA", different),
        ]
    elif level == "TARGET_FAMILY":
        different = different & ~exact
        tiers = [
            ("same fine MOA and different family", different & fine),
            ("same broad action and different family", different & broad),
            ("different family", different),
        ]
    else:
        tiers = [
            ("same target family and different canonical target set", different & family),
            ("same fine MOA and different canonical target set", different & fine),
            ("same broad action and different canonical target set", different & broad),
        ]
    rows: list[dict[str, object]] = []
    for pos in positive:
        selected: np.ndarray | None = None
        selected_distance: np.ndarray | None = None
        selected_tier = ""
        for tier_name, tier_mask in tiers:
            candidates = np.flatnonzero(tier_mask)
            if not len(candidates):
                continue
            distance = np.linalg.norm(nuisance[candidates] - nuisance[pos], axis=1)
            valid = distance <= maximum_distance
            candidates, distance = candidates[valid], distance[valid]
            if not len(candidates):
                continue
            order = np.lexsort((pairs.right[candidates], pairs.left[candidates], distance))
            selected = candidates[order[:controls_per_pair]]
            selected_distance = distance[order[:controls_per_pair]]
            selected_tier = tier_name
            break
        if selected is None:
            rows.append(
                {
                    "level": level,
                    "positive_pair_axis": pos,
                    "positive_i": int(pairs.left[pos]),
                    "positive_j": int(pairs.right[pos]),
                    "control_pair_axis": -1,
                    "control_rank": -1,
                    "distance": np.nan,
                    "priority_tier": "NOT_ESTIMABLE_AT_REQUESTED_CONDITIONAL_RESOLUTION",
                }
            )
            continue
        for rank, (control, distance) in enumerate(zip(selected, selected_distance, strict=True), 1):
            rows.append(
                {
                    "level": level,
                    "positive_pair_axis": pos,
                    "positive_i": int(pairs.left[pos]),
                    "positive_j": int(pairs.right[pos]),
                    "control_pair_axis": int(control),
                    "control_rank": rank,
                    "distance": float(distance),
                    "priority_tier": selected_tier,
                }
            )
    return pd.DataFrame(rows)


def matched_pair_deltas(matches: pd.DataFrame, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-pair axes, endpoints and positive-minus-control outcomes."""

    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    valid = matches[matches.control_pair_axis >= 0]
    rows: list[np.ndarray] = []
    axes: list[int] = []
    endpoints: list[tuple[int, int]] = []
    for pos, local in valid.groupby("positive_pair_axis", sort=True):
        controls = local.control_pair_axis.to_numpy(int)
        rows.append(y[int(pos)] - y[controls].mean(axis=0))
        axes.append(int(pos))
        endpoints.append((int(local.positive_i.iloc[0]), int(local.positive_j.iloc[0])))
    if not rows:
        return np.empty(0, int), np.empty((0, 2), int), np.empty((0, y.shape[1]))
    return np.asarray(axes), np.asarray(endpoints), np.vstack(rows)


def cluster_bootstrap_pair_mean(
    endpoints: np.ndarray,
    values: np.ndarray,
    interventions: int,
    draws: int = N_DRAW,
    seed: int = SEED,
) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    rng = np.random.default_rng(seed)
    result = np.full((draws, y.shape[1]), np.nan, dtype=float)
    for draw in range(draws):
        count = rng.multinomial(interventions, np.full(interventions, 1 / interventions))
        weight = count[endpoints[:, 0]] * count[endpoints[:, 1]]
        denominator = weight.sum()
        if denominator > 0:
            result[draw] = (weight[:, None] * y).sum(axis=0) / denominator
    return result


def annotation_cluster_count(annotations: pd.DataFrame, level: str) -> int:
    if level == "BROAD_ACTION":
        labels = pd.Series([_valid_label(x) for x in annotations.moa_broad])
        return int((labels[labels.ne("")].value_counts() >= 2).sum())
    if level == "MOA":
        labels = pd.Series([_valid_label(x) for x in annotations.moa_fine])
        return int((labels[labels.ne("")].value_counts() >= 2).sum())
    if level == "EXACT_TARGET":
        labels = annotations.exact_targets.fillna("").astype(str)
        return int((labels[labels.ne("")].value_counts() >= 2).sum())
    families: dict[str, set[int]] = {}
    for axis, value in enumerate(annotations.hgnc_target_family_ids):
        for family in split_set(value):
            families.setdefault(family, set()).add(axis)
    targets = [split_set(value) for value in annotations.exact_targets]
    contributes = 0
    for axes in families.values():
        ordered = sorted(axes)
        if any(targets[i] != targets[j] for offset, i in enumerate(ordered) for j in ordered[offset + 1 :]):
            contributes += 1
    return contributes


def compute_gene_statistics(
    gamma: np.ndarray,
    residual: np.ndarray,
    lineages: Sequence[str],
    relation_interventions: Sequence[int],
    feature_block: int = 512,
) -> GeneStatistics:
    """One-pass gene-space fingerprint and genotype-kernel accumulation."""

    if gamma.shape != residual.shape or gamma.ndim != 4 or gamma.shape[0] != 2:
        raise ValueError("gamma and residual must have shape 2 x context x intervention x gene")
    contexts, interventions, features = gamma.shape[1:]
    relation_axes = np.asarray(sorted(set(map(int, relation_interventions))), dtype=int)
    total = _empty_statistics(interventions)
    names = sorted(set(map(str, lineages)))
    lineage_stats = {name: _empty_statistics(interventions) for name in names}
    lineage_masks = {name: np.flatnonzero(np.asarray(lineages, dtype=object) == name) for name in names}
    kernels = {
        "residual_cross_plate": np.zeros((interventions, contexts, contexts), dtype=np.float64),
        "plate6_gamma_to_plate14_residual": np.zeros((interventions, contexts, contexts), dtype=np.float64),
        "plate14_gamma_to_plate6_residual": np.zeros((interventions, contexts, contexts), dtype=np.float64),
    }
    for lo in range(0, features, feature_block):
        hi = min(features, lo + feature_block)
        g6 = np.asarray(gamma[0, :, :, lo:hi], dtype=np.float32)
        g14 = np.asarray(gamma[1, :, :, lo:hi], dtype=np.float32)
        u6 = np.asarray(residual[0, :, :, lo:hi], dtype=np.float32)
        u14 = np.asarray(residual[1, :, :, lo:hi], dtype=np.float32)
        _accumulate_statistics(total, g6, g14, u6, u14)
        for name, axes in lineage_masks.items():
            _accumulate_statistics(lineage_stats[name], g6[axes], g14[axes], u6[axes], u14[axes])
        if len(relation_axes):
            kernels["residual_cross_plate"][relation_axes] += np.einsum(
                "crg,drg->rcd", u6[:, relation_axes], u14[:, relation_axes], dtype=np.float64, optimize=True
            )
            kernels["plate6_gamma_to_plate14_residual"][relation_axes] += np.einsum(
                "crg,drg->rcd", g6[:, relation_axes], u14[:, relation_axes], dtype=np.float64, optimize=True
            )
            kernels["plate14_gamma_to_plate6_residual"][relation_axes] += np.einsum(
                "crg,drg->rcd", g14[:, relation_axes], u6[:, relation_axes], dtype=np.float64, optimize=True
            )
    kernels["directional_mean"] = 0.5 * (
        kernels["plate6_gamma_to_plate14_residual"]
        + np.transpose(kernels["plate14_gamma_to_plate6_residual"], (0, 2, 1))
    )
    return GeneStatistics(total=total, lineage=lineage_stats, relation_kernels=kernels)


def relation_weights(
    relations: pd.DataFrame,
    context_mapping: pd.DataFrame,
    contexts: Sequence[str],
) -> np.ndarray:
    """Compile the immutable altered-minus-unaltered context contrasts."""

    aligned = context_mapping.set_index("cell_line_id").loc[list(contexts)]
    depmap_to_axis = {
        str(value): axis for axis, value in enumerate(aligned.depmap_id) if pd.notna(value) and str(value).strip()
    }
    weights = np.zeros((len(relations), len(contexts)), dtype=np.float64)
    for row_axis, row in enumerate(relations.itertuples(index=False)):
        altered = sorted(depmap_to_axis[x] for x in split_set(row.altered_depmap_ids) if x in depmap_to_axis)
        unaltered = sorted(depmap_to_axis[x] for x in split_set(row.unaltered_depmap_ids) if x in depmap_to_axis)
        if not altered or not unaltered:
            raise RuntimeError(f"Relation {row.relation_id} lacks a two-class contrast")
        weights[row_axis, altered] = 1 / len(altered)
        weights[row_axis, unaltered] = -1 / len(unaltered)
    return weights


def _relation_features(
    relations: pd.DataFrame,
    annotations: pd.DataFrame,
    weights: np.ndarray,
    lineages: np.ndarray,
    statistics: CrossPlateStatistics,
) -> np.ndarray:
    pmap = annotations.set_index("intervention_id").intervention_axis.astype(int).to_dict()
    pidx = relations.intervention_id.map(pmap).to_numpy(int)
    counts = relations.groupby("intervention_id").size()
    log_gamma = 0.25 * (
        np.log(np.maximum(statistics.gamma_norm6_sq, EPS))
        + np.log(np.maximum(statistics.gamma_norm14_sq, EPS))
    )
    log_residual = 0.25 * (
        np.log(np.maximum(statistics.residual_norm6_sq, EPS))
        + np.log(np.maximum(statistics.residual_norm14_sq, EPS))
    )
    lineage_levels = sorted(set(map(str, lineages)))
    lineage_profile = np.zeros((len(relations), len(lineage_levels)), dtype=float)
    for row in range(len(relations)):
        positive = weights[row] > 0
        for axis, name in enumerate(lineage_levels):
            lineage_profile[row, axis] = np.count_nonzero(positive & (lineages == name))
        if lineage_profile[row].sum():
            lineage_profile[row] /= lineage_profile[row].sum()
    altered = pd.to_numeric(relations.n_altered, errors="coerce").to_numpy(float)
    unaltered = pd.to_numeric(relations.n_unaltered, errors="coerce").to_numpy(float)
    continuous = np.column_stack(
        [
            altered / np.maximum(altered + unaltered, 1),
            np.abs(altered - unaltered) / np.maximum(altered + unaltered, 1),
            altered + unaltered,
            relations.intervention_id.map(counts).to_numpy(float),
            log_gamma[pidx],
            log_residual[pidx],
            lineage_profile,
        ]
    )
    scaled, *_ = _robust_scale(continuous)
    return scaled


def _fallback_relation_mask(
    row: pd.Series,
    relations: pd.DataFrame,
    annotations_by_id: pd.DataFrame,
) -> tuple[str, np.ndarray]:
    source = annotations_by_id.loc[row.intervention_id]
    candidates = relations.intervention_id.ne(row.intervention_id).to_numpy()
    source_target = "" if pd.isna(source.exact_targets) else str(source.exact_targets).strip()
    target = relations.intervention_id.map(annotations_by_id.exact_targets).fillna("").astype(str)
    if source_target and np.count_nonzero(candidates & target.eq(source_target).to_numpy()):
        return "exact_target_set", candidates & target.eq(source_target).to_numpy()
    source_fine = _valid_label(source.moa_fine)
    fine = relations.intervention_id.map(annotations_by_id.moa_fine).map(_valid_label)
    if source_fine and np.count_nonzero(candidates & fine.eq(source_fine).to_numpy()):
        return "moa_fine", candidates & fine.eq(source_fine).to_numpy()
    source_broad = _valid_label(source.moa_broad)
    broad = relations.intervention_id.map(annotations_by_id.moa_broad).map(_valid_label)
    if source_broad and np.count_nonzero(candidates & broad.eq(source_broad).to_numpy()):
        return "moa_broad", candidates & broad.eq(source_broad).to_numpy()
    return "NOT_ESTIMABLE", np.zeros(len(relations), dtype=bool)


def genotype_control_map(
    relations: pd.DataFrame,
    annotations: pd.DataFrame,
    relation_features: np.ndarray,
    controls_per_relation: int = 5,
    caliper: float = 2.5,
) -> pd.DataFrame:
    """Freeze G2 controls before evaluating any relation score."""

    by_id = annotations.set_index("intervention_id")
    counts = relations.groupby("intervention_id").size()
    rows: list[dict[str, object]] = []
    for axis, relation in relations.iterrows():
        same_drug = relations.intervention_id.eq(relation.intervention_id).to_numpy(copy=True)
        same_drug[axis] = False
        if counts[relation.intervention_id] > 1:
            resolution, candidates = "exact_drug", same_drug
        else:
            resolution, candidates = _fallback_relation_mask(relation, relations, by_id)
        candidates &= relations.alteration_gene.ne(relation.alteration_gene).to_numpy()
        candidate_axes = np.flatnonzero(candidates)
        if len(candidate_axes):
            distance = np.linalg.norm(relation_features[candidate_axes] - relation_features[axis], axis=1)
            keep = distance <= caliper
            candidate_axes, distance = candidate_axes[keep], distance[keep]
        else:
            distance = np.empty(0)
        if not len(candidate_axes):
            rows.append(
                {
                    "relation_axis": axis,
                    "relation_id": relation.relation_id,
                    "candidate_relation_axis": -1,
                    "candidate_relation_id": "",
                    "g2_resolution": "NOT_ESTIMABLE_AT_REQUESTED_CONDITIONAL_RESOLUTION",
                    "distance": np.nan,
                    "candidate_rank": -1,
                }
            )
            continue
        order = np.lexsort((relations.iloc[candidate_axes].relation_id.to_numpy(), distance))
        for rank, local in enumerate(order[:controls_per_relation], 1):
            candidate = int(candidate_axes[local])
            rows.append(
                {
                    "relation_axis": axis,
                    "relation_id": relation.relation_id,
                    "candidate_relation_axis": candidate,
                    "candidate_relation_id": relations.at[candidate, "relation_id"],
                    "g2_resolution": resolution,
                    "distance": float(distance[local]),
                    "candidate_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def _quadratic_scores(kernels: Mapping[str, np.ndarray], pidx: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name, kernel in kernels.items():
        scores[name] = np.asarray(
            [weights[row] @ kernel[pidx[row]] @ weights[row] for row in range(len(weights))], dtype=float
        )
    return scores


def genotype_relation_deltas(
    kernels: Mapping[str, np.ndarray],
    relations: pd.DataFrame,
    annotations: pd.DataFrame,
    weights: np.ndarray,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate G3 minus deterministic G2 for each frozen relation."""

    pmap = annotations.set_index("intervention_id").intervention_axis.astype(int).to_dict()
    pidx = relations.intervention_id.map(pmap).to_numpy(int)
    observed = _quadratic_scores(kernels, pidx, weights)
    rows: list[dict[str, object]] = []
    for relation_axis, local in controls.groupby("relation_axis", sort=True):
        candidates = local.loc[local.candidate_relation_axis >= 0, "candidate_relation_axis"].to_numpy(int)
        relation = relations.iloc[int(relation_axis)]
        for direction, kernel in kernels.items():
            correct = observed[direction][int(relation_axis)]
            if len(candidates):
                wrong = np.asarray(
                    [weights[candidate] @ kernel[pidx[int(relation_axis)]] @ weights[candidate] for candidate in candidates]
                )
                g2 = float(wrong.mean())
                delta = float(correct - g2)
            else:
                g2 = delta = np.nan
            rows.append(
                {
                    "relation_axis": int(relation_axis),
                    "relation_id": relation.relation_id,
                    "intervention_id": relation.intervention_id,
                    "alteration_gene": relation.alteration_gene,
                    "match_level": relation.match_level,
                    "direction": direction,
                    "g3_correct": float(correct),
                    "g2_wrong": g2,
                    "g3_minus_g2": delta,
                    "g2_resolution": local.g2_resolution.iloc[0],
                    "n_g2_controls": len(candidates),
                }
            )
    return pd.DataFrame(rows)


def _equal_drug_mean(frame: pd.DataFrame, value: str = "g3_minus_g2") -> float:
    local = frame[np.isfinite(pd.to_numeric(frame[value], errors="coerce"))]
    if not len(local):
        return np.nan
    return float(local.groupby("intervention_id")[value].mean().mean())


def genotype_cluster_bootstrap(
    relation_deltas: pd.DataFrame,
    direction: str,
    draws: int = N_DRAW,
    seed: int = SEED,
) -> np.ndarray:
    local = relation_deltas[
        relation_deltas.direction.eq(direction)
        & relation_deltas.match_level.eq("LEVEL2_PATHWAY")
        & np.isfinite(relation_deltas.g3_minus_g2)
    ]
    grouped = local.groupby("intervention_id").g3_minus_g2.mean()
    values = grouped.to_numpy(float)
    if not len(values):
        return np.full(draws, np.nan)
    rng = np.random.default_rng(seed)
    axes = rng.integers(0, len(values), size=(draws, len(values)))
    return values[axes].mean(axis=1)


def genotype_block_permutation_null(
    kernels: Mapping[str, np.ndarray],
    relations: pd.DataFrame,
    annotations: pd.DataFrame,
    weights: np.ndarray,
    controls: pd.DataFrame,
    direction: str = "residual_cross_plate",
    draws: int = N_DRAW,
    seed: int = SEED,
) -> np.ndarray:
    """Move whole alteration contrasts within frozen drug/fallback blocks."""

    pmap = annotations.set_index("intervention_id").intervention_axis.astype(int).to_dict()
    pidx = relations.intervention_id.map(pmap).to_numpy(int)
    valid = controls[controls.candidate_relation_axis >= 0].groupby("relation_axis").size().index.to_numpy(int)
    pool = valid
    primary = pool[relations.iloc[pool].match_level.to_numpy() == "LEVEL2_PATHWAY"]
    resolution = controls.groupby("relation_axis").g2_resolution.first()
    pool_block_labels = []
    annotation_by_id = annotations.set_index("intervention_id")
    for axis in pool:
        row = relations.iloc[axis]
        res = str(resolution.get(axis, "NOT_ESTIMABLE"))
        if res == "exact_drug":
            pool_block_labels.append(f"DRUG:{row.intervention_id}")
        elif res in {"exact_target_set", "moa_fine", "moa_broad"}:
            annotation = annotation_by_id.loc[row.intervention_id]
            value = {
                "exact_target_set": annotation.exact_targets,
                "moa_fine": annotation.moa_fine,
                "moa_broad": annotation.moa_broad,
            }[res]
            pool_block_labels.append(f"{res}:{value}")
        else:
            pool_block_labels.append(f"UNMOVED:{axis}")
    pool_block_labels = np.asarray(pool_block_labels, dtype=object)
    pool_position = {int(axis): position for position, axis in enumerate(pool)}
    primary_block_labels = np.asarray([pool_block_labels[pool_position[int(axis)]] for axis in primary], dtype=object)
    block_groups = [
        (
            np.flatnonzero(primary_block_labels == label),
            np.flatnonzero(pool_block_labels == label),
        )
        for label in sorted(set(primary_block_labels))
    ]
    kernel = kernels[direction]
    # Score every primary response row under every exchangeable alteration contrast.
    assignment = np.empty((len(primary), len(pool)), dtype=np.float64)
    for row, relation_axis in enumerate(primary):
        local_kernel = kernel[pidx[relation_axis]]
        assignment[row] = np.einsum(
            "ic,cd,id->i", weights[pool], local_kernel, weights[pool], optimize=True
        )
    g2 = (
        relation_deltas_from_controls(controls, relations, primary, pidx, weights, kernel)
        .set_index("relation_axis")
        .reindex(primary)
        .g2_wrong.to_numpy(float)
    )
    drugs = relations.iloc[primary].intervention_id.to_numpy()
    unique_drugs = sorted(set(drugs))
    rng = np.random.default_rng(seed)
    null = np.full(draws, np.nan, dtype=float)
    for draw in range(draws):
        assigned = np.asarray([pool_position[int(axis)] for axis in primary], dtype=int)
        for response_group, candidate_group in block_groups:
            if len(candidate_group) > 1:
                order = candidate_group[rng.permutation(len(candidate_group))]
                assigned[response_group] = np.resize(order, len(response_group))
        delta = assignment[np.arange(len(primary)), assigned] - g2
        drug_values = [np.nanmean(delta[drugs == drug]) for drug in unique_drugs]
        null[draw] = float(np.nanmean(drug_values))
    return null


def relation_deltas_from_controls(
    controls: pd.DataFrame,
    relations: pd.DataFrame,
    relation_axes: np.ndarray,
    pidx: np.ndarray,
    weights: np.ndarray,
    kernel: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for axis in relation_axes:
        local = controls[(controls.relation_axis == axis) & (controls.candidate_relation_axis >= 0)]
        candidates = local.candidate_relation_axis.to_numpy(int)
        correct = float(weights[axis] @ kernel[pidx[axis]] @ weights[axis])
        wrong = (
            float(np.mean([weights[j] @ kernel[pidx[axis]] @ weights[j] for j in candidates]))
            if len(candidates)
            else np.nan
        )
        rows.append({"relation_axis": axis, "g3_correct": correct, "g2_wrong": wrong, "g3_minus_g2": correct - wrong})
    return pd.DataFrame(rows)


def hierarchy_leave_group_out(
    annotations: pd.DataFrame,
    pairs: PairAxes,
    nuisance: np.ndarray,
    indicators: np.ndarray,
    outcomes: np.ndarray,
    outcome_labels: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Leave-one drug/MOA/target-family estimates on frozen full-context fingerprints."""

    groups: list[tuple[str, str, np.ndarray]] = []
    for axis, name in zip(annotations.intervention_axis, annotations.intervention_id, strict=True):
        keep = np.ones(len(annotations), dtype=bool)
        keep[int(axis)] = False
        groups.append(("drug", str(name), keep))
    fine = np.asarray([_valid_label(x) for x in annotations.moa_fine], dtype=object)
    for label in sorted(set(fine) - {""}):
        groups.append(("MOA", label, fine != label))
    family_by_axis = [split_set(value) for value in annotations.hgnc_target_family_ids]
    for family in sorted(set().union(*family_by_axis)):
        groups.append(("target_family", family, np.asarray([family not in values for values in family_by_axis])))
    rows: list[dict[str, object]] = []
    for group_type, omitted, keep in groups:
        pair_keep = pairs.subset(keep)
        if pair_keep.sum() < 10:
            continue
        estimate = fit_conditional_ladder(
            nuisance[pair_keep], indicators[pair_keep], outcomes[pair_keep]
        )
        for level_axis, level in enumerate(LEVELS):
            status = "ESTIMABLE" if indicators[pair_keep, level_axis].sum() > 0 else "NOT_ESTIMABLE"
            for outcome_axis, (representation, direction) in enumerate(outcome_labels):
                rows.append(
                    {
                        "group_type": group_type,
                        "omitted_group": omitted,
                        "level": level,
                        "representation": representation,
                        "direction": direction,
                        "effect": estimate[level_axis, outcome_axis] if status == "ESTIMABLE" else np.nan,
                        "n_interventions": int(keep.sum()),
                        "n_pairs": int(pair_keep.sum()),
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def lineage_leave_out(
    annotations: pd.DataFrame,
    pairs: PairAxes,
    indicators: np.ndarray,
    representation_stats: Mapping[str, CrossPlateStatistics],
    lineage_stats: Mapping[str, Mapping[str, CrossPlateStatistics]],
    gene_stats: CrossPlateStatistics,
    gene_lineage_stats: Mapping[str, CrossPlateStatistics],
    lineage_context_counts: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lineage in sorted(gene_lineage_stats):
        reduced_gene = gene_stats.without(gene_lineage_stats[lineage])
        available_contexts = 50 - int(lineage_context_counts[lineage])
        nuisance, _, _ = pair_nuisance_features(annotations, reduced_gene, pairs, available_contexts)
        outcomes, labels = [], []
        for representation in REPRESENTATIONS:
            reduced = representation_stats[representation].without(lineage_stats[representation][lineage])
            similarity = reduced.similarities(pairs)
            for direction_axis, direction in enumerate(DIRECTIONS):
                outcomes.append(similarity[:, direction_axis])
                labels.append((representation, direction))
        y = np.column_stack(outcomes)
        estimate = fit_conditional_ladder(nuisance, indicators, y)
        for level_axis, level in enumerate(LEVELS):
            status = "ESTIMABLE" if indicators[:, level_axis].sum() > 0 else "NOT_ESTIMABLE"
            for outcome_axis, (representation, direction) in enumerate(labels):
                rows.append(
                    {
                        "group_type": "lineage",
                        "omitted_group": lineage,
                        "level": level,
                        "representation": representation,
                        "direction": direction,
                        "effect": estimate[level_axis, outcome_axis] if status == "ESTIMABLE" else np.nan,
                        "n_interventions": len(annotations),
                        "n_pairs": len(pairs.left),
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def per_intervention_organization(
    annotations: pd.DataFrame,
    pairs: PairAxes,
    indicators: np.ndarray,
    outcomes: np.ndarray,
    outcome_labels: Sequence[tuple[str, str]],
    match_maps: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Frozen bridge score: finest annotated positive pair minus its controls."""

    labels = list(outcome_labels)
    primary_columns = {
        direction: labels.index(("Gene", direction)) for direction in DIRECTIONS
    }
    delta_by_pair: dict[int, np.ndarray] = {}
    selected_level: dict[int, str] = {}
    for level_axis in range(3, -1, -1):
        level = LEVELS[level_axis]
        _, _, deltas = matched_pair_deltas(match_maps[level], outcomes)
        valid = match_maps[level][match_maps[level].control_pair_axis >= 0]
        grouped = list(valid.groupby("positive_pair_axis", sort=True))
        for (pair_axis, _), delta in zip(grouped, deltas, strict=True):
            pair_axis = int(pair_axis)
            if indicators[pair_axis, level_axis] and pair_axis not in delta_by_pair:
                delta_by_pair[pair_axis] = delta
                selected_level[pair_axis] = level
    rows = []
    for intervention_axis, annotation in annotations.iterrows():
        incident = [
            pair_axis
            for pair_axis in delta_by_pair
            if pairs.left[pair_axis] == intervention_axis or pairs.right[pair_axis] == intervention_axis
        ]
        values = np.vstack([delta_by_pair[axis] for axis in incident]) if incident else np.empty((0, outcomes.shape[1]))
        row = {
            "intervention_axis": int(annotation.intervention_axis),
            "intervention_id": annotation.intervention_id,
            "organization_score": float(values[:, primary_columns["pooled"]].mean()) if len(values) else np.nan,
            "organization_score_plate6_to_plate14": float(values[:, primary_columns["plate6_to_plate14"]].mean()) if len(values) else np.nan,
            "organization_score_plate14_to_plate6": float(values[:, primary_columns["plate14_to_plate6"]].mean()) if len(values) else np.nan,
            "n_matched_pair_contrasts": len(values),
            "n_eligible_neighbors": len(incident),
            "organization_eligible": bool(len(values)),
            "organization_levels": ";".join(sorted({selected_level[axis] for axis in incident}, key=LEVELS.index)),
            "organization_moa_broad": annotation.moa_broad,
            "organization_moa_fine": annotation.moa_fine,
            "organization_exact_targets": annotation.exact_targets,
            "organization_target_family_ids": annotation.hgnc_target_family_ids,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _renormalize_without_contexts(weights: np.ndarray, omitted: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=float).copy()
    result[:, omitted] = 0
    positive = result > 0
    negative = result < 0
    pden = positive.sum(axis=1)
    nden = negative.sum(axis=1)
    positive_value = np.divide(
        np.ones_like(pden, dtype=float), pden, out=np.zeros_like(pden, dtype=float), where=pden > 0
    )
    negative_value = np.divide(
        -np.ones_like(nden, dtype=float), nden, out=np.zeros_like(nden, dtype=float), where=nden > 0
    )
    result = np.where(positive, positive_value[:, None], result)
    result = np.where(negative, negative_value[:, None], result)
    invalid = (pden == 0) | (nden == 0)
    result[invalid] = np.nan
    return result


def genotype_leave_group_out(
    relation_deltas: pd.DataFrame,
    relations: pd.DataFrame,
    annotations: pd.DataFrame,
    kernels: Mapping[str, np.ndarray],
    weights: np.ndarray,
    controls: pd.DataFrame,
    lineages: np.ndarray,
) -> pd.DataFrame:
    """Leave drug/MOA/family/lineage from the pathway-linked G3-G2 estimand."""

    annotation_by_id = annotations.set_index("intervention_id")
    relation_annotation = annotation_by_id.loc[relations.intervention_id].reset_index()
    rows: list[dict[str, object]] = []

    def append_subset(group_type: str, omitted: str, keep_relations: np.ndarray, source: pd.DataFrame) -> None:
        for direction in sorted(source.direction.unique()):
            local = source[
                source.direction.eq(direction)
                & source.relation_axis.isin(np.flatnonzero(keep_relations))
                & source.match_level.eq("LEVEL2_PATHWAY")
            ]
            rows.append(
                {
                    "group_type": group_type,
                    "omitted_group": omitted,
                    "level": "GENOTYPE_DRUG",
                    "representation": "Gene",
                    "direction": direction,
                    "effect": _equal_drug_mean(local),
                    "n_interventions": local.intervention_id.nunique(),
                    "n_pairs": len(local),
                    "status": "ESTIMABLE" if local.intervention_id.nunique() >= 3 else "NOT_ESTIMABLE",
                }
            )

    for drug in sorted(relations.intervention_id.unique()):
        append_subset("drug", drug, relations.intervention_id.ne(drug).to_numpy(), relation_deltas)
    fine = relation_annotation.moa_fine.map(_valid_label).to_numpy()
    for label in sorted(set(fine) - {""}):
        append_subset("MOA", label, fine != label, relation_deltas)
    family_values = [split_set(value) for value in relation_annotation.hgnc_target_family_ids]
    for family in sorted(set().union(*family_values)):
        append_subset(
            "target_family",
            family,
            np.asarray([family not in values for values in family_values]),
            relation_deltas,
        )
    for lineage in sorted(set(map(str, lineages))):
        omitted = np.flatnonzero(lineages == lineage)
        local_weights = _renormalize_without_contexts(weights, omitted)
        valid_weights = np.isfinite(local_weights).all(axis=1)
        if not valid_weights.all():
            # Candidate map remains frozen; affected rows become non-estimable.
            local_weights[~valid_weights] = 0
        local_delta = genotype_relation_deltas(
            kernels, relations, annotations, local_weights, controls
        )
        local_delta.loc[~local_delta.relation_axis.map(dict(enumerate(valid_weights))).astype(bool), "g3_minus_g2"] = np.nan
        append_subset("lineage", lineage, valid_weights, local_delta)
    return pd.DataFrame(rows)


def _outcome_matrix(
    representation_stats: Mapping[str, CrossPlateStatistics],
    pairs: PairAxes,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    values, labels = [], []
    for representation in REPRESENTATIONS:
        similarities = representation_stats[representation].similarities(pairs)
        for direction_axis, direction in enumerate(DIRECTIONS):
            values.append(similarities[:, direction_axis])
            labels.append((representation, direction))
    return np.column_stack(values), labels


def _null_pvalues(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    two = np.full(observed.shape, np.nan, dtype=float)
    positive = np.full(observed.shape, np.nan, dtype=float)
    for level in range(observed.shape[0]):
        for outcome in range(observed.shape[1]):
            values = null[:, level, outcome]
            finite = np.isfinite(values)
            if not finite.any() or not np.isfinite(observed[level, outcome]):
                continue
            two[level, outcome] = (1 + np.count_nonzero(np.abs(values[finite]) >= abs(observed[level, outcome]))) / (
                1 + finite.sum()
            )
            positive[level, outcome] = (1 + np.count_nonzero(values[finite] >= observed[level, outcome])) / (
                1 + finite.sum()
            )
    return two, positive


def _loo_sign_summary(leave_out: pd.DataFrame, level: str) -> dict[str, float]:
    primary = leave_out[
        leave_out.level.eq(level)
        & leave_out.representation.eq("Gene")
        & leave_out.direction.eq("pooled" if level != "GENOTYPE_DRUG" else "residual_cross_plate")
        & leave_out.status.eq("ESTIMABLE")
    ]
    return {
        group: float(np.mean(local.effect.to_numpy(float) > 0))
        for group, local in primary.groupby("group_type", sort=True)
        if len(local)
    }


def _decision_status(
    cluster_count: int,
    effect: float,
    ci_low: float,
    qap_p_positive: float,
    matched_effect: float,
    directional_effects: Sequence[float],
    loo_sign: Mapping[str, float],
) -> str:
    if cluster_count < 3:
        return "NOT_ESTIMABLE"
    if cluster_count < 5:
        return "LIMITED_POWER"
    robust = (
        effect > 0
        and ci_low > 0
        and qap_p_positive < 0.05
        and matched_effect > 0
        and all(value > 0 for value in directional_effects)
        and bool(loo_sign)
        and all(value >= 0.80 for value in loo_sign.values())
    )
    return "ROBUST" if robust else "NOT_ROBUST"


def _write_final_report(hierarchy: pd.DataFrame, out: Path) -> None:
    primary = hierarchy[
        hierarchy.representation.eq("Gene")
        & hierarchy.direction.isin(["pooled", "residual_cross_plate"])
    ].copy()
    robust = primary[primary.decision_status.eq("ROBUST")]
    order = {level: axis for axis, level in enumerate((*LEVELS, "GENOTYPE_DRUG"))}
    highest = max(robust.level, key=lambda value: order.get(value, -1)) if len(robust) else "NONE"
    lines = [
        "# CGC-BIO-2 — Hierarchical biological attribution",
        "",
        "This report applies the frozen partial-order hierarchy and never treats pairwise rows as independent.",
        "",
        f"`HIGHEST_ROBUST_ATTRIBUTION_LEVEL={highest}`",
        "",
        "| Level | Effect | 95% cluster CI | conditional QAP p+ | matched effect | clusters | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in primary.sort_values("level", key=lambda s: s.map(order)).itertuples(index=False):
        lines.append(
            f"| {row.level} | {row.conditional_effect:.6g} | [{row.ci_low:.6g}, {row.ci_high:.6g}] "
            f"| {row.qap_p_positive:.6g} | {row.matched_effect:.6g} | {row.independent_clusters} | {row.decision_status} |"
        )
    lines += [
        "",
        "Exact-target inference is capped at `LIMITED_POWER` by its frozen three-cluster/five-pair design. "
        "The four direct genotype-target relations are descriptive only; the 420 pathway-linked relations define the primary genotype contrast.",
        "",
    ]
    (out / "BIO2_HIERARCHICAL_ATTRIBUTION_FINAL.md").write_text("\n".join(lines), encoding="utf-8")


def run_bio2(
    out: Path = OUT,
    draws: int = N_DRAW,
    seed: int = SEED,
    verify_hashes: bool = True,
    feature_block: int = 512,
) -> dict[str, object]:
    """Run the frozen BIO-2 analysis and write all preregistered outputs."""

    if out.resolve() == OUT.resolve() and (draws != N_DRAW or seed != SEED or not verify_hashes):
        raise ValueError("Formal BIO-2 output requires seed 7201, 10,000 draws, and hash verification")
    inputs = load_frozen_inputs(verify_hashes=verify_hashes)
    out.mkdir(parents=True, exist_ok=True)
    annotations = inputs.annotations
    pairs = PairAxes.build(len(annotations))
    pmap = {name: axis for axis, name in enumerate(inputs.interventions)}
    relation_pidx = inputs.relation_manifest.intervention_id.map(pmap).to_numpy(int)
    gene = compute_gene_statistics(
        inputs.gene_gamma,
        inputs.gene_residual,
        inputs.lineages,
        relation_pidx,
        feature_block,
    )
    representation_stats: dict[str, CrossPlateStatistics] = {"Gene": gene.total}
    representation_lineage: dict[str, dict[str, CrossPlateStatistics]] = {"Gene": gene.lineage}
    for representation in ("PROGENy", "CollecTRI"):
        axes = np.flatnonzero(inputs.program_manifest.space.to_numpy() == representation)
        total, lineage = compute_cross_plate_statistics(
            inputs.program_gamma[..., axes],
            inputs.program_residual[..., axes],
            inputs.lineages,
            feature_block=max(len(axes), 1),
        )
        representation_stats[representation] = total
        representation_lineage[representation] = lineage

    indicators = pair_indicators(annotations, pairs)
    nuisance, nuisance_names, nuisance_audit = pair_nuisance_features(annotations, gene.total, pairs)
    outcomes, outcome_labels = _outcome_matrix(representation_stats, pairs)
    observed = fit_conditional_ladder(nuisance, indicators, outcomes)
    bootstrap = cluster_bootstrap_ladder(
        nuisance, indicators, outcomes, pairs, len(annotations), draws, seed + 1
    )
    qap = qap_conditional_null(annotations, pairs, nuisance, outcomes, draws, seed)
    p_two, p_positive = _null_pvalues(observed, qap)

    match_maps: dict[str, pd.DataFrame] = {}
    matched_effect: dict[str, np.ndarray] = {}
    matched_bootstrap: dict[str, np.ndarray] = {}
    match_parts = []
    for level_axis, level in enumerate(LEVELS):
        matches = match_lower_resolution_controls(annotations, pairs, indicators, nuisance, level)
        match_maps[level] = matches
        match_parts.append(matches)
        _, endpoints, deltas = matched_pair_deltas(matches, outcomes)
        matched_effect[level] = np.nanmean(deltas, axis=0) if len(deltas) else np.full(outcomes.shape[1], np.nan)
        matched_bootstrap[level] = (
            cluster_bootstrap_pair_mean(endpoints, deltas, len(annotations), draws, seed + 10 + level_axis)
            if len(deltas)
            else np.full((draws, outcomes.shape[1]), np.nan)
        )
    pd.concat(match_parts, ignore_index=True).to_csv(out / "BIO2_MATCHED_PAIR_MAP.csv", index=False)

    loo = hierarchy_leave_group_out(annotations, pairs, nuisance, indicators, outcomes, outcome_labels)
    lineage_counts = pd.Series(inputs.lineages).value_counts().to_dict()
    loo_lineage = lineage_leave_out(
        annotations,
        pairs,
        indicators,
        representation_stats,
        representation_lineage,
        gene.total,
        gene.lineage,
        lineage_counts,
    )
    loo = pd.concat([loo, loo_lineage], ignore_index=True)

    hierarchy_rows: list[dict[str, object]] = []
    for level_axis, level in enumerate(LEVELS):
        cluster_count = annotation_cluster_count(annotations, level)
        loo_sign = _loo_sign_summary(loo, level)
        primary_axis = outcome_labels.index(("Gene", "pooled"))
        direction_axes = [
            outcome_labels.index(("Gene", "plate6_to_plate14")),
            outcome_labels.index(("Gene", "plate14_to_plate6")),
        ]
        primary_status = _decision_status(
            cluster_count,
            observed[level_axis, primary_axis],
            np.nanquantile(bootstrap[:, level_axis, primary_axis], 0.025),
            p_positive[level_axis, primary_axis],
            matched_effect[level][primary_axis],
            observed[level_axis, direction_axes],
            loo_sign,
        )
        for outcome_axis, (representation, direction) in enumerate(outcome_labels):
            hierarchy_rows.append(
                {
                    "level": level,
                    "representation": representation,
                    "direction": direction,
                    "conditional_effect": observed[level_axis, outcome_axis],
                    "ci_low": np.nanquantile(bootstrap[:, level_axis, outcome_axis], 0.025),
                    "ci_high": np.nanquantile(bootstrap[:, level_axis, outcome_axis], 0.975),
                    "qap_p_two_sided": p_two[level_axis, outcome_axis],
                    "qap_p_positive": p_positive[level_axis, outcome_axis],
                    "matched_effect": matched_effect[level][outcome_axis],
                    "matched_ci_low": np.nanquantile(matched_bootstrap[level][:, outcome_axis], 0.025),
                    "matched_ci_high": np.nanquantile(matched_bootstrap[level][:, outcome_axis], 0.975),
                    "positive_pairs": int(indicators[:, level_axis].sum()),
                    "matched_positive_pairs": int(
                        match_maps[level].loc[match_maps[level].control_pair_axis >= 0, "positive_pair_axis"].nunique()
                    ),
                    "independent_clusters": cluster_count,
                    "decision_status": primary_status if (representation, direction) == ("Gene", "pooled") else "CONFIRMATION_AXIS",
                    "loo_sign_consistency_json": json.dumps(loo_sign, sort_keys=True),
                }
            )

    # G3-G2 genotype hierarchy, kept distinct from pharmacologic pair QAP.
    weights = relation_weights(inputs.relation_manifest, inputs.context_mapping, inputs.contexts)
    relation_features = _relation_features(
        inputs.relation_manifest, annotations, weights, inputs.lineages, gene.total
    )
    genotype_controls = genotype_control_map(inputs.relation_manifest, annotations, relation_features)
    genotype_controls.to_csv(out / "BIO2_GENOTYPE_G2_MAP.csv", index=False)
    relation_delta = genotype_relation_deltas(
        gene.relation_kernels, inputs.relation_manifest, annotations, weights, genotype_controls
    )
    relation_delta.to_csv(out / "BIO2_GENOTYPE_RELATION_RESULTS.csv", index=False)
    genotype_loo = genotype_leave_group_out(
        relation_delta,
        inputs.relation_manifest,
        annotations,
        gene.relation_kernels,
        weights,
        genotype_controls,
        inputs.lineages,
    )
    loo = pd.concat([loo, genotype_loo], ignore_index=True)
    genotype_null = genotype_block_permutation_null(
        gene.relation_kernels,
        inputs.relation_manifest,
        annotations,
        weights,
        genotype_controls,
        "residual_cross_plate",
        draws,
        seed + 100,
    )
    genotype_primary = relation_delta[
        relation_delta.direction.eq("residual_cross_plate")
        & relation_delta.match_level.eq("LEVEL2_PATHWAY")
    ]
    genotype_effect = _equal_drug_mean(genotype_primary)
    genotype_boot = genotype_cluster_bootstrap(relation_delta, "residual_cross_plate", draws, seed + 101)
    finite_null = genotype_null[np.isfinite(genotype_null)]
    genotype_p_two = (
        (1 + np.count_nonzero(np.abs(finite_null) >= abs(genotype_effect))) / (1 + len(finite_null))
        if len(finite_null)
        else np.nan
    )
    genotype_p_positive = (
        (1 + np.count_nonzero(finite_null >= genotype_effect)) / (1 + len(finite_null))
        if len(finite_null)
        else np.nan
    )
    genotype_direction = {
        name: _equal_drug_mean(
            relation_delta[
                relation_delta.direction.eq(name) & relation_delta.match_level.eq("LEVEL2_PATHWAY")
            ]
        )
        for name in ("plate6_gamma_to_plate14_residual", "plate14_gamma_to_plate6_residual")
    }
    genotype_loo_sign = _loo_sign_summary(loo, "GENOTYPE_DRUG")
    genotype_status = _decision_status(
        genotype_primary.intervention_id.nunique(),
        genotype_effect,
        np.nanquantile(genotype_boot, 0.025),
        genotype_p_positive,
        genotype_effect,
        list(genotype_direction.values()),
        genotype_loo_sign,
    )
    hierarchy_rows.append(
        {
            "level": "GENOTYPE_DRUG",
            "representation": "Gene",
            "direction": "residual_cross_plate",
            "conditional_effect": genotype_effect,
            "ci_low": np.nanquantile(genotype_boot, 0.025),
            "ci_high": np.nanquantile(genotype_boot, 0.975),
            "qap_p_two_sided": genotype_p_two,
            "qap_p_positive": genotype_p_positive,
            "matched_effect": genotype_effect,
            "matched_ci_low": np.nanquantile(genotype_boot, 0.025),
            "matched_ci_high": np.nanquantile(genotype_boot, 0.975),
            "positive_pairs": len(genotype_primary),
            "matched_positive_pairs": int(np.isfinite(genotype_primary.g3_minus_g2).sum()),
            "independent_clusters": genotype_primary.intervention_id.nunique(),
            "decision_status": genotype_status,
            "loo_sign_consistency_json": json.dumps(genotype_loo_sign, sort_keys=True),
        }
    )
    direct = relation_delta[
        relation_delta.direction.eq("residual_cross_plate")
        & relation_delta.match_level.eq("LEVEL1_DIRECT")
    ]
    hierarchy_rows.append(
        {
            "level": "GENOTYPE_DRUG_DIRECT_SENSITIVITY",
            "representation": "Gene",
            "direction": "residual_cross_plate",
            "conditional_effect": _equal_drug_mean(direct),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "qap_p_two_sided": np.nan,
            "qap_p_positive": np.nan,
            "matched_effect": _equal_drug_mean(direct),
            "matched_ci_low": np.nan,
            "matched_ci_high": np.nan,
            "positive_pairs": len(direct),
            "matched_positive_pairs": int(np.isfinite(direct.g3_minus_g2).sum()),
            "independent_clusters": direct.intervention_id.nunique(),
            "decision_status": "DESCRIPTIVE_NOT_ESTIMABLE",
            "loo_sign_consistency_json": "{}",
        }
    )

    hierarchy = pd.DataFrame(hierarchy_rows)
    hierarchy["qap_q_two_sided"] = np.nan
    for _, axes in hierarchy.groupby(["representation", "direction"]).groups.items():
        hierarchy.loc[axes, "qap_q_two_sided"] = bh_adjust(
            hierarchy.loc[axes, "qap_p_two_sided"].to_numpy(float)
        )
    hierarchy.to_csv(out / "BIO2_HIERARCHY_RESULTS.csv", index=False)
    loo.to_csv(out / "BIO2_LEAVE_GROUP_OUT.csv", index=False)
    hierarchy[
        [
            "level",
            "representation",
            "direction",
            "conditional_effect",
            "ci_low",
            "ci_high",
            "matched_effect",
            "decision_status",
        ]
    ].to_csv(out / "BIO2_PLATE_REPLICATION.csv", index=False)

    null_rows = []
    for draw in range(draws):
        for level_axis, level in enumerate(LEVELS):
            for outcome_axis, (representation, direction) in enumerate(outcome_labels):
                null_rows.append(
                    {
                        "null": "QAP_ANNOTATION",
                        "draw": draw,
                        "level": level,
                        "representation": representation,
                        "direction": direction,
                        "value": qap[draw, level_axis, outcome_axis],
                    }
                )
        null_rows.append(
            {
                "null": "GENOTYPE_DRUG_BLOCK",
                "draw": draw,
                "level": "GENOTYPE_DRUG",
                "representation": "Gene",
                "direction": "residual_cross_plate",
                "value": genotype_null[draw],
            }
        )
    pd.DataFrame(null_rows).to_csv(out / "BIO2_CONDITIONAL_NULLS.csv", index=False)
    organization = per_intervention_organization(
        annotations, pairs, indicators, outcomes, outcome_labels, match_maps
    )
    organization.to_csv(out / "BIO2_PER_INTERVENTION_ORGANIZATION.csv", index=False)
    _write_final_report(hierarchy, out)
    result = {
        "phase": "CGC-BIO-2",
        "git_commit": git("rev-parse", "HEAD"),
        "seed": seed,
        "draws": draws,
        "representations": list(REPRESENTATIONS),
        "nuisance_audit": nuisance_audit,
        "nuisance_names": nuisance_names,
        "highest_robust_attribution_level": (
            hierarchy.loc[
                hierarchy.decision_status.eq("ROBUST")
                & hierarchy.representation.eq("Gene")
                & hierarchy.direction.isin(["pooled", "residual_cross_plate"]),
                "level",
            ].iloc[-1]
            if np.any(
                hierarchy.decision_status.eq("ROBUST")
                & hierarchy.representation.eq("Gene")
                & hierarchy.direction.isin(["pooled", "residual_cross_plate"])
            )
            else "NONE"
        ),
        "frozen_protocol_commit": "de548acea86b4da93b736feb0fcab9ebd830cf48",
    }
    write_json(out / "BIO2_RUN_MANIFEST.json", result)
    return result


if __name__ == "__main__":  # pragma: no cover - use the repository script
    print(json.dumps(run_bio2(), indent=2))
