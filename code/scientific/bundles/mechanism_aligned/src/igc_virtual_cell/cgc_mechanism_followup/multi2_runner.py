"""Real-data orchestration for the frozen CGC-MULTI-2 experiment.

Importing this module never opens a response outcome.  The outcome-bearing
``bio1_gene_arrays.npz`` is opened only by :func:`run_real_analysis` or
:func:`run_power_analysis`, both of which require an explicit caller action.
Feature audits, eligibility tables, hashes, and placeholder reports are safe to
run before unsealing outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import OUT, ROOT, SOURCE_ROOT, sha256, write_json
from .multi2 import (
    DEFAULT_MODEL_SPECS,
    GENERIC_RANKS,
    NULL_DRAWS,
    POWER_LEVELS,
    POWER_REPLICATES,
    RESPONSE_RANKS,
    RIDGE_GRID,
    SEED,
    AlignedBlock,
    CompleteCaseMatrix,
    FoldLocalPowerFold,
    ModelSpec,
    aggregate_mapped_features,
    apply_intervention_order,
    context_permutation_orders,
    fold_specific_power_calibration,
    fit_predict_outer_fold,
    hierarchical_bootstrap,
    intersect_aligned_blocks,
    paired_incremental_summary,
    per_intervention_gains,
    prepare_fold_local_power_folds,
    select_complete_features,
)


FROZEN_COMMIT = "de548acea86b4da93b736feb0fcab9ebd830cf48"
FROZEN_HASHES = {
    "results/cgc_mechanism_followup/MULTI2_MECHANISM_ALIGNED_PROTOCOL.md":
        "422ae0426dedd807ca26c37ea3478d696dcbf27980b78f85ab93cdd1187ec166",
    "results/cgc_mechanism_followup/MULTI2_FEATURE_MAP.json":
        "fcf8947e0d436d045a64f4ad877649840fbbf8e1660c1334600006fd4921c435",
    "results/cgc_mechanism_followup/MULTI2_ANNOTATION_PROVENANCE.json":
        "9af2af9c197c4c2dd70287ccc4ee8a02ee0465d553a41c7f66126ac83f1ff538",
    "results/cgc_mechanism_followup/BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv":
        "8e1f2619cde18d793c76674710aa9a2c9a58c0d394439fb288070d3a192d0bb2",
}
SOURCE_HASHES = {
    "data/cgc_bio1_official/bio1_gene_arrays.npz":
        "6b46053324175bfc34b2534bb59ae5519a1aa47b504ddeb2887956953f222228",
    "data/cgc_bio1_official/frozen_program_weights.parquet":
        "f5c4faa94fe03b2d088b72743da175380dfcf2edbfacd0077cc25eb19d2ca7e9",
    "data/cgc_multi1_official/rppa500-mclp_tahoe49.parquet":
        "2f60e2bc45b1ece9eb12bf43e06d7dfa393489bf85ade1c3051925407a928f9c",
    "data/cgc_multi1_official/harmonized_Sanger_MS_2022_tahoe49.parquet":
        "7e76ab775a5d606f40458fbbbf90dfa186b9619f9fc6bcd9962079ef3e9428af",
    "data/cgc_multi1_official/Chronos_Combined_tahoe49.parquet":
        "2dc521c366207a279dc83a1d48bf7d506b70096424b4364bf87d237fcf397ece",
}

REQUIRED_OUTPUTS: dict[str, tuple[str, ...]] = {
    "MULTI2_COMPLETE_CASE_MATRIX.csv": (
        "panel", "modality", "context_axis", "context_id", "intervention_axis",
        "intervention_id", "context_profile_available", "context_in_panel",
        "mapping_available", "pair_eligible", "status", "reason",
    ),
    "MULTI2_INCREMENTAL_RECOVERY.csv": (
        "response_target", "panel", "modality", "model", "comparator", "estimator",
        "delta_g", "ci_low", "ci_high", "bootstrap_draws", "status",
    ),
    "MULTI2_ALIGNMENT_NULLS.csv": (
        "response_target", "panel", "modality", "null_type", "estimator",
        "observed_delta_g", "null_delta_g", "contrast_delta_g", "contrast_ci_low",
        "contrast_ci_high", "null_p", "table_draws", "oof_refits", "status",
    ),
    "MULTI2_POWER_CURVES.csv": (
        "response_target", "panel", "modality", "target_delta_g", "estimate", "bias",
        "power", "false_positive_rate", "ci_coverage", "replicates", "device", "status",
    ),
    "MULTI2_DETECTION_LIMITS.csv": (
        "response_target", "panel", "modality", "estimator", "mde80", "replicates",
        "levels", "device", "status", "reason",
    ),
    "MULTI2_PER_INTERVENTION_GAINS.csv": (
        "response_target", "panel", "modality", "model", "comparator", "estimator",
        "intervention_axis", "intervention_id", "delta_g_aligned_minus_rna",
        "paired_se", "ci_low", "ci_high", "moa_broad", "moa_fine",
        "hgnc_target_family_ids", "hgnc_target_family_names", "bridge_primary", "status",
    ),
}


@dataclass(frozen=True)
class RunnerPaths:
    root: Path = ROOT
    source_root: Path = SOURCE_ROOT
    out: Path = OUT

    @property
    def cache(self) -> Path:
        return self.out / "multi2_fold_cache"


@dataclass
class Foundation:
    context_table: pd.DataFrame
    intervention_table: pd.DataFrame
    annotations: pd.DataFrame
    rna: np.ndarray
    lineages: np.ndarray
    depmap_ids: tuple[str | None, ...]
    reactome_members: list[tuple[str, ...]]
    exact_targets: list[tuple[str, ...]]


@dataclass
class PanelBundle:
    name: str
    modality: str
    context_axes: np.ndarray
    intervention_axes: np.ndarray
    generic_raw: np.ndarray
    aligned: np.ndarray
    aligned_feature_names: tuple[str, ...]
    context_ids: tuple[str, ...]
    intervention_ids: tuple[str, ...]
    random_aligned: np.ndarray | None = None
    status: str = "ELIGIBLE"
    notes: list[str] = field(default_factory=list)
    raw_complete_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nc, np_ = len(self.context_axes), len(self.intervention_axes)
        if np.asarray(self.generic_raw).ndim != 2 or len(self.generic_raw) != nc:
            raise ValueError(f"{self.name}: generic matrix does not cover the panel contexts")
        if np.asarray(self.aligned).shape[:2] != (nc, np_):
            raise ValueError(f"{self.name}: aligned matrix does not cover the exact panel")
        if not np.isfinite(self.generic_raw).all() or not np.isfinite(self.aligned).all():
            raise ValueError(f"{self.name}: an eligible value is missing; missingness may not be zero-filled")
        if self.random_aligned is not None:
            if np.asarray(self.random_aligned).shape != np.asarray(self.aligned).shape:
                raise ValueError(f"{self.name}: matched random control does not cover the exact aligned panel")
            if not np.isfinite(self.random_aligned).all():
                raise ValueError(f"{self.name}: matched random control contains a missing value")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status_frame(columns: Sequence[str], status: str, reason: str) -> pd.DataFrame:
    row = {column: np.nan for column in columns}
    if "status" in row:
        row["status"] = status
    if "reason" in row:
        row["reason"] = reason
    return pd.DataFrame([row], columns=columns)


def initialize_outputs(paths: RunnerPaths, *, status: str = "NOT_RUN") -> None:
    """Create schema-stable report files without opening outcomes."""

    paths.out.mkdir(parents=True, exist_ok=True)
    for name, columns in REQUIRED_OUTPUTS.items():
        destination = paths.out / name
        if not destination.exists():
            _status_frame(columns, status, "stage has not been explicitly executed").to_csv(destination, index=False)


def verify_frozen_hashes(
    paths: RunnerPaths,
    *,
    include_large_response: bool = False,
    tracked_expected: Mapping[str, str] = FROZEN_HASHES,
    source_expected: Mapping[str, str] = SOURCE_HASHES,
) -> pd.DataFrame:
    """Verify the preregistration and official inputs without reading outcomes.

    By default the 1.9 GB response is existence/size checked but not rehashed.
    ``include_large_response`` performs the preregistered full digest before a
    real run.  Hashing bytes does not inspect scientific outcomes.
    """

    rows: list[dict[str, object]] = []
    for relative, expected in tracked_expected.items():
        path = paths.root / relative
        observed = sha256(path) if path.exists() else None
        rows.append({"scope": "frozen_tracked", "path": relative, "exists": path.exists(),
                     "bytes": path.stat().st_size if path.exists() else 0,
                     "expected_sha256": expected, "observed_sha256": observed,
                     "status": "PASS" if observed == expected else "FAIL"})
    for relative, expected in source_expected.items():
        path = paths.source_root / relative
        is_large_response = relative.endswith("bio1_gene_arrays.npz")
        observed = sha256(path) if path.exists() and (include_large_response or not is_large_response) else None
        if not path.exists():
            status = "FAIL"
        elif is_large_response and not include_large_response:
            status = "EXISTS_HASH_DEFERRED"
        else:
            status = "PASS" if observed == expected else "FAIL"
        rows.append({"scope": "official_source", "path": relative, "exists": path.exists(),
                     "bytes": path.stat().st_size if path.exists() else 0,
                     "expected_sha256": expected, "observed_sha256": observed, "status": status})
    result = pd.DataFrame(rows)
    hard_fail = result.status.eq("FAIL")
    if hard_fail.any():
        failures = result.loc[hard_fail, "path"].tolist()
        raise RuntimeError(f"MULTI2_FROZEN_HASH_MISMATCH: {failures}")
    return result


def _axis_tables(paths: RunnerPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(paths.root / "results/cgc_tahoe_0i/pseudobulk_sample_metadata.csv")
    contexts = (samples[["context_axis", "cell_line_id"]].drop_duplicates()
                .sort_values("context_axis").reset_index(drop=True))
    interventions = (samples.loc[~samples["is_control"].astype(bool), ["intervention_id"]]
                     .drop_duplicates().reset_index(drop=True))
    annotations = pd.read_csv(paths.root / "results/cgc_mechanism_followup/BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv")
    interventions = annotations[["intervention_axis", "intervention_id"]].sort_values("intervention_axis").reset_index(drop=True)
    if contexts.context_axis.tolist() != list(range(50)):
        raise RuntimeError("MULTI2_CONTEXT_AXIS_NOT_0_TO_49")
    if interventions.intervention_axis.tolist() != list(range(93)):
        raise RuntimeError("MULTI2_INTERVENTION_AXIS_NOT_0_TO_92")
    return contexts, interventions, annotations


def _split_targets(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(sorted({part.strip().upper() for part in str(value).split(";") if part.strip()}))


def _reactome_sets(paths: RunnerPaths, exact_targets: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    weights = pd.read_parquet(paths.source_root / "data/cgc_bio1_official/frozen_program_weights.parquet")
    reactome = weights.loc[weights["space"].eq("Reactome"), ["program", "gene_symbol"]].copy()
    reactome["gene_symbol"] = reactome.gene_symbol.astype(str).str.upper()
    target_to_programs = reactome.groupby("gene_symbol").program.agg(lambda x: frozenset(x)).to_dict()
    program_to_genes = reactome.groupby("program").gene_symbol.agg(lambda x: frozenset(x)).to_dict()
    result: list[tuple[str, ...]] = []
    for targets in exact_targets:
        programs: set[str] = set()
        for target in targets:
            programs.update(target_to_programs.get(str(target).upper(), ()))
        genes: set[str] = set()
        for program in programs:
            genes.update(program_to_genes[program])
        result.append(tuple(sorted(genes)))
    return result


def _select_primary_baseline(controls: np.ndarray, primary_indices: np.ndarray) -> np.ndarray:
    """Map the frozen all-mapped NTC tensor onto the frozen G_PRIMARY order."""

    values = np.asarray(controls, dtype=np.float64)
    indices = np.asarray(primary_indices, dtype=np.int64)
    if values.ndim != 3 or values.shape[:2] != (2, 50):
        raise RuntimeError(f"MULTI2_BASELINE_RNA_AXIS_MISMATCH: {values.shape}")
    if indices.shape != (25_695,) or len(np.unique(indices)) != len(indices):
        raise RuntimeError(f"MULTI2_G_PRIMARY_INDEX_MISMATCH: {indices.shape}")
    if indices.min() < 0 or indices.max() >= values.shape[2]:
        raise RuntimeError("MULTI2_G_PRIMARY_INDEX_OUT_OF_RANGE")
    selected = values[:, :, indices]
    if selected.shape != (2, 50, 25_695) or not np.isfinite(selected).all():
        raise RuntimeError("MULTI2_PRIMARY_BASELINE_INVALID")
    return np.log1p(selected).mean(axis=0)


def _load_rna(paths: RunnerPaths) -> np.ndarray:
    """Load the frozen NTC baseline only; this is not a perturbation outcome."""

    import zarr

    store = zarr.open_group(paths.source_root / "results/cgc_tahoe_0i/response_tensors.zarr", mode="r")
    controls = np.asarray(store["primary_dmso_cpm"], dtype=np.float64)
    primary_indices = np.load(
        paths.source_root / "results/cgc_tahoe_0i/gene_indices_g_primary.npy",
        allow_pickle=False,
    )
    return _select_primary_baseline(controls, primary_indices)


def load_foundation(paths: RunnerPaths, *, rna_override: np.ndarray | None = None) -> Foundation:
    """Load outcome-independent axes, baseline RNA, and frozen mappings."""

    contexts, interventions, annotations = _axis_tables(paths)
    mapping = pd.read_csv(paths.root / "results/cgc_tahoe_bio1/cellline_depmap_mapping.csv")
    mapping = contexts.merge(mapping, on="cell_line_id", how="left", validate="one_to_one")
    depmap = tuple(None if pd.isna(value) else str(value) for value in mapping.depmap_id)
    lineage = mapping.OncotreeLineage.fillna(mapping.Organ).fillna("UNKNOWN").astype(str).to_numpy()
    exact = [_split_targets(value) for value in annotations.sort_values("intervention_axis").exact_targets]
    reactome = _reactome_sets(paths, exact)
    rna = np.asarray(rna_override, dtype=np.float64) if rna_override is not None else _load_rna(paths)
    if rna.shape[0] != 50 or not np.isfinite(rna).all():
        raise RuntimeError("MULTI2_BASELINE_RNA_INVALID")
    return Foundation(contexts, interventions, annotations, rna, lineage, depmap, reactome, exact)


def _matrix_by_context(
    matrix_path: Path,
    feature_path: Path | None,
    foundation: Foundation,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read an official feature×DepMap parquet into the frozen 50-context axis."""

    matrix = pd.read_parquet(matrix_path)
    if feature_path is not None:
        labels = pd.read_parquet(feature_path)
        if not {"id", "label"}.issubset(labels.columns):
            raise RuntimeError(f"feature-map schema mismatch for {matrix_path.name}")
        by_id = labels.assign(id=labels["id"].astype(str)).drop_duplicates("id").set_index("id")["label"]
        mapped = pd.Index(matrix.index.astype(str)).map(by_id)
        # Official parquets retain the feature ID as their index.  Row-order
        # fallback is accepted only when the complete official table is an
        # exact-length companion and identifier matching is wholly absent.
        if pd.notna(mapped).mean() >= 0.80:
            feature_names = tuple(pd.Series(mapped).fillna(pd.Series(matrix.index.astype(str))).astype(str))
        elif len(labels) == len(matrix) and pd.isna(mapped).all():
            feature_names = tuple(labels["label"].fillna(labels["id"]).astype(str))
        else:
            raise RuntimeError(
                f"feature-map identifier coverage mismatch for {matrix_path.name}: "
                f"{pd.notna(mapped).mean():.3f}"
            )
    else:
        feature_names = tuple(matrix.index.astype(str).str.replace(r" \(\d+\)$", "", regex=True).str.upper())
    values = np.full((50, len(matrix)), np.nan, dtype=np.float64)
    for context_axis, depmap_id in enumerate(foundation.depmap_ids):
        if depmap_id is not None and depmap_id in matrix.columns:
            values[context_axis] = pd.to_numeric(matrix[depmap_id], errors="coerce").to_numpy(float)
    return values, feature_names


def _candidate_contexts(raw: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isfinite(raw).any(axis=1)).astype(np.int64)


def _make_complete(raw: np.ndarray, names: Sequence[str], contexts: np.ndarray, name: str) -> CompleteCaseMatrix:
    return select_complete_features(raw, contexts, names, name=name)


def _mapped_feature_hits(
    complete: CompleteCaseMatrix,
    members: Sequence[Iterable[str]],
    *,
    match: str,
) -> list[np.ndarray]:
    normalized = [str(label).upper() for label in complete.feature_names]
    exact_index: dict[str, list[int]] = {}
    for index, label in enumerate(normalized):
        exact_index.setdefault(label, []).append(index)
        if match == "protein_label":
            parsed = re.search(r"\(([^()]+)\)\s*$", label)
            if parsed:
                exact_index.setdefault(parsed.group(1), []).append(index)
    result: list[np.ndarray] = []
    for raw_members in members:
        member_set = {str(member).upper() for member in raw_members if str(member).strip()}
        hits: set[int] = set()
        for member in member_set:
            hits.update(exact_index.get(member, ()))
        if match == "rppa_target" and member_set:
            # Exact frozen rule: label == target or label starts target + P.
            for feature_axis, label in enumerate(normalized):
                if any(label.startswith(member + "P") for member in member_set):
                    hits.add(feature_axis)
        result.append(np.asarray(sorted(hits), dtype=np.int64))
    return result


def _aggregate(
    complete: CompleteCaseMatrix,
    members: Sequence[Iterable[str]],
    feature_name: str,
    *,
    match: str = "exact",
) -> tuple[AlignedBlock | None, np.ndarray]:
    hits_by_axis = _mapped_feature_hits(complete, members, match=match)
    kept_axes, values = [], []
    eligible = np.zeros(93, dtype=bool)
    for axis, hits in enumerate(hits_by_axis):
        if not len(hits):
            continue
        eligible[axis] = True
        kept_axes.append(axis)
        values.append(complete.values[:, hits].mean(axis=1))
    if not values:
        return None, eligible
    array = np.stack(values, axis=1)[:, :, None]
    return AlignedBlock(
        name=feature_name, context_axes=complete.context_axes.copy(),
        intervention_axes=np.asarray(kept_axes, dtype=np.int64),
        feature_names=(feature_name,), values=array,
    ), eligible


def _matched_random_block(
    complete: CompleteCaseMatrix,
    members: Sequence[Iterable[str]],
    observed: AlignedBlock | None,
    feature_name: str,
    *,
    match: str = "exact",
) -> AlignedBlock | None:
    """Deterministic measured-size/coverage control excluding true members."""

    if observed is None:
        return None
    hits_by_axis = _mapped_feature_hits(complete, members, match=match)
    pool = np.arange(complete.values.shape[1], dtype=np.int64)
    values: list[np.ndarray] = []
    for axis in observed.intervention_axes:
        true_hits = hits_by_axis[int(axis)]
        candidates = np.setdiff1d(pool, true_hits, assume_unique=True)
        if len(candidates) < len(true_hits):
            return None
        token = int.from_bytes(
            hashlib.sha256(f"{feature_name}|{int(axis)}".encode("utf-8")).digest()[:8], "little",
        )
        rng = np.random.default_rng(np.random.SeedSequence([SEED, token & 0xFFFFFFFF, token >> 32]))
        selected = rng.choice(candidates, size=len(true_hits), replace=False)
        values.append(complete.values[:, selected].mean(axis=1))
    return AlignedBlock(
        name=f"random_{feature_name}", context_axes=complete.context_axes.copy(),
        intervention_axes=observed.intervention_axes.copy(), feature_names=(feature_name,),
        values=np.stack(values, axis=1)[:, :, None],
    )


def _assemble_panel(
    *,
    name: str,
    modality: str,
    complete: CompleteCaseMatrix,
    blocks: Sequence[AlignedBlock | None],
    random_blocks: Sequence[AlignedBlock | None],
    foundation: Foundation,
    notes: Sequence[str] = (),
) -> PanelBundle | None:
    available = [block for block in blocks if block is not None]
    if not available:
        return None
    aligned = intersect_aligned_blocks(available, name=name)
    random_available = [block for block in random_blocks if block is not None]
    random_aligned: np.ndarray | None = None
    if len(random_available) == len(available):
        random = intersect_aligned_blocks(random_available, name=f"{name}_RANDOM")
        if np.array_equal(random.intervention_axes, aligned.intervention_axes):
            random_aligned = random.values
    context_axes = complete.context_axes
    if not np.array_equal(aligned.context_axes, context_axes):
        raise RuntimeError(f"{name}: complete/aligned context axes differ")
    return PanelBundle(
        name=name,
        modality=modality,
        context_axes=context_axes,
        intervention_axes=aligned.intervention_axes,
        generic_raw=complete.values,
        aligned=aligned.values,
        aligned_feature_names=aligned.feature_names,
        context_ids=tuple(foundation.context_table.set_index("context_axis").loc[context_axes, "cell_line_id"].astype(str)),
        intervention_ids=tuple(foundation.intervention_table.set_index("intervention_axis").loc[aligned.intervention_axes, "intervention_id"].astype(str)),
        random_aligned=random_aligned,
        notes=list(notes),
        raw_complete_feature_names=complete.feature_names,
    )


def _combine_mutation_sources(
    hotspot: tuple[np.ndarray, tuple[str, ...]],
    damaging: tuple[np.ndarray, tuple[str, ...]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Union two measured binary alteration tables without treating NaN as zero."""

    frames = []
    for values, names in (hotspot, damaging):
        frame = pd.DataFrame(values, columns=[str(x).upper() for x in names]).T
        # Duplicate gene records within a source are a measured maximum, not an imputation.
        frame = frame.groupby(level=0, sort=True).max()
        frames.append(frame)
    joined = pd.concat(frames, keys=("hotspot", "damaging"), names=("source", "gene"))
    genes = sorted(set(frames[0].index).union(frames[1].index))
    result = np.full((hotspot[0].shape[0], len(genes)), np.nan, dtype=np.float64)
    for column, gene in enumerate(genes):
        source_values = []
        for frame in frames:
            if gene in frame.index:
                source_values.append(frame.loc[gene].to_numpy(float))
        stack = np.stack(source_values)
        measured = np.isfinite(stack)
        value = np.full(stack.shape[1], np.nan)
        any_measured = measured.any(axis=0)
        value[any_measured] = np.nanmax(stack[:, any_measured], axis=0)
        result[:, column] = value
    return result, tuple(genes)


def build_panels(paths: RunnerPaths, foundation: Foundation) -> tuple[dict[str, PanelBundle], pd.DataFrame]:
    """Build every frozen panel using only baseline modalities and annotations."""

    bio = paths.source_root / "data/cgc_bio1_official"
    multi = paths.source_root / "data/cgc_multi1_official"
    depmap_gene_map = multi / "Chronos_Combined_features.parquet"
    hotspot = _matrix_by_context(bio / "mutations_hotspot_tahoe49.parquet", depmap_gene_map, foundation)
    damaging = _matrix_by_context(bio / "mutations_damaging_tahoe49.parquet", depmap_gene_map, foundation)
    mutation_raw, mutation_names = _combine_mutation_sources(hotspot, damaging)
    cnv_raw, cnv_names = _matrix_by_context(bio / "OmicsCNGeneWGS_tahoe49.parquet", depmap_gene_map, foundation)
    rppa_raw, rppa_names = _matrix_by_context(
        multi / "rppa500-mclp_tahoe49.parquet", multi / "rppa500-mclp_features.parquet", foundation,
    )
    total_raw, total_names = _matrix_by_context(
        multi / "harmonized_Sanger_MS_2022_tahoe49.parquet",
        multi / "harmonized_Sanger_MS_2022_features.parquet", foundation,
    )
    dependency_raw, dependency_names = _matrix_by_context(
        multi / "Chronos_Combined_tahoe49.parquet", multi / "Chronos_Combined_features.parquet", foundation,
    )

    mutation_contexts = _candidate_contexts(mutation_raw)
    cnv_contexts = _candidate_contexts(cnv_raw)
    rppa_contexts = _candidate_contexts(rppa_raw)
    total_contexts = _candidate_contexts(total_raw)
    dependency_contexts = _candidate_contexts(dependency_raw)
    expected_counts = {
        "WIRING": (len(mutation_contexts), 49),
        "ACTIVITY": (len(rppa_contexts), 44),
        "TOTAL_PROTEIN": (len(total_contexts), 45),
        "DEPENDENCY_ORACLE": (len(dependency_contexts), 39),
    }
    bad = {name: observed for name, (observed, expected) in expected_counts.items() if observed != expected}
    if bad:
        raise RuntimeError(f"MULTI2_PREDECLARED_CONTEXT_COUNT_MISMATCH: {bad}")

    panels: dict[str, PanelBundle] = {}
    audit_rows: list[dict[str, object]] = []

    mutation_complete = _make_complete(mutation_raw, mutation_names, mutation_contexts, "WIRING_MUTATION")
    target_mut, target_mut_mask = _aggregate(mutation_complete, foundation.exact_targets, "target_mut")
    pathway_mut, pathway_mut_mask = _aggregate(mutation_complete, foundation.reactome_members, "pathway_mut")
    random_target_mut = _matched_random_block(
        mutation_complete, foundation.exact_targets, target_mut, "target_mut",
    )
    random_pathway_mut = _matched_random_block(
        mutation_complete, foundation.reactome_members, pathway_mut, "pathway_mut",
    )
    wiring = _assemble_panel(
        name="WIRING", modality="WIRING", complete=mutation_complete,
        blocks=[target_mut, pathway_mut], random_blocks=[random_target_mut, random_pathway_mut],
        foundation=foundation,
        notes=("CNV is not encoded as zero on the 49-context mutation panel; it is audited separately.",),
    )
    if wiring is not None:
        panels[wiring.name] = wiring
    # CNV cannot honestly be appended to a 49-context matched comparison when only 39 profiles exist.
    audit_rows.append({"panel": "WIRING", "feature_axis": "target_cnv/pathway_cnv",
                       "status": "NOT_EXECUTABLE_CONTEXT_PANEL_MISMATCH", "available_contexts": len(cnv_contexts),
                       "selected_complete_features": 0, "eligible_interventions": 0,
                       "reason": "official CNV profiles cover 39, not the frozen 49-context WIRING panel"})

    rppa_complete = _make_complete(rppa_raw, rppa_names, rppa_contexts, "ACTIVITY_RPPA")
    pathway_rppa, pathway_rppa_mask = _aggregate(
        rppa_complete, foundation.reactome_members, "pathway_rppa", match="rppa_target",
    )
    random_pathway_rppa = _matched_random_block(
        rppa_complete, foundation.reactome_members, pathway_rppa, "pathway_rppa", match="rppa_target",
    )
    activity = _assemble_panel(name="ACTIVITY", modality="ACTIVITY_PATHWAY", complete=rppa_complete,
                               blocks=[pathway_rppa], random_blocks=[random_pathway_rppa], foundation=foundation)
    if activity is not None:
        panels[activity.name] = activity

    phospho_idx = np.asarray(
        [i for i, name in enumerate(rppa_names) if re.search(r"P[STY]\d", str(name).upper())], dtype=int,
    )
    phospho_raw = rppa_raw[:, phospho_idx]
    phospho_names = tuple(rppa_names[i] for i in phospho_idx)
    phospho_complete = _make_complete(phospho_raw, phospho_names, rppa_contexts, "PHOSPHO")
    phospho_block, phospho_mask = _aggregate(
        phospho_complete, foundation.exact_targets, "target_phospho", match="rppa_target",
    )
    random_phospho = _matched_random_block(
        phospho_complete, foundation.exact_targets, phospho_block, "target_phospho", match="rppa_target",
    )
    phospho = _assemble_panel(name="PHOSPHO", modality="PHOSPHO_EXACT_TARGET", complete=phospho_complete,
                              blocks=[phospho_block], random_blocks=[random_phospho], foundation=foundation,
                              notes=("Sparse exact-target secondary axis.",))
    if phospho is not None:
        panels[phospho.name] = phospho

    total_complete = _make_complete(total_raw, total_names, total_contexts, "TOTAL_PROTEIN")
    total_block, total_mask = _aggregate(
        total_complete, foundation.exact_targets, "target_total_protein", match="protein_label",
    )
    random_total = _matched_random_block(
        total_complete, foundation.exact_targets, total_block, "target_total_protein", match="protein_label",
    )
    total = _assemble_panel(name="TOTAL_PROTEIN", modality="TOTAL_PROTEIN_EXACT_TARGET",
                            complete=total_complete, blocks=[total_block], random_blocks=[random_total],
                            foundation=foundation,
                            notes=("Sparse exact-target secondary axis.",))
    if total is not None:
        panels[total.name] = total

    dependency_complete = _make_complete(
        dependency_raw, dependency_names, dependency_contexts, "DEPENDENCY_ORACLE",
    )
    dep_target, dep_target_mask = _aggregate(dependency_complete, foundation.exact_targets, "target_dependency")
    dep_pathway, dep_pathway_mask = _aggregate(
        dependency_complete, foundation.reactome_members, "pathway_dependency",
    )
    random_dep_target = _matched_random_block(
        dependency_complete, foundation.exact_targets, dep_target, "target_dependency",
    )
    random_dep_pathway = _matched_random_block(
        dependency_complete, foundation.reactome_members, dep_pathway, "pathway_dependency",
    )
    oracle = _assemble_panel(
        name="DEPENDENCY_ORACLE", modality="FUNCTIONAL_PERTURBATIONAL_ORACLE",
        complete=dependency_complete, blocks=[dep_target, dep_pathway],
        random_blocks=[random_dep_target, random_dep_pathway], foundation=foundation,
        notes=("Perturbational functional oracle; never pooled with ordinary modalities.",),
    )
    if oracle is not None:
        panels[oracle.name] = oracle

    # M5 uses the predeclared 40-context wiring/RPPA/Sanger intersection, even
    # though the fitted aligned blocks are wiring + activity only.
    joint_contexts = np.intersect1d(np.intersect1d(mutation_contexts, rppa_contexts), total_contexts)
    if len(joint_contexts) != 40:
        raise RuntimeError(f"MULTI2_JOINT_PANEL_EXPECTED_40_OBSERVED_{len(joint_contexts)}")
    joint_mut = _make_complete(mutation_raw, mutation_names, joint_contexts, "M5_WIRING")
    joint_rppa = _make_complete(rppa_raw, rppa_names, joint_contexts, "M5_ACTIVITY")
    jm_target, _ = _aggregate(joint_mut, foundation.exact_targets, "target_mut")
    jm_path, _ = _aggregate(joint_mut, foundation.reactome_members, "pathway_mut")
    jr_path, _ = _aggregate(joint_rppa, foundation.reactome_members, "pathway_rppa", match="rppa_target")
    rjm_target = _matched_random_block(joint_mut, foundation.exact_targets, jm_target, "target_mut")
    rjm_path = _matched_random_block(joint_mut, foundation.reactome_members, jm_path, "pathway_mut")
    rjr_path = _matched_random_block(
        joint_rppa, foundation.reactome_members, jr_path, "pathway_rppa", match="rppa_target",
    )
    aligned_joint = intersect_aligned_blocks(
        [block for block in (jm_target, jm_path, jr_path) if block is not None], name="M5",
    )
    joint_generic = np.concatenate([joint_mut.values, joint_rppa.values], axis=1)
    random_joint_blocks = [block for block in (rjm_target, rjm_path, rjr_path) if block is not None]
    random_joint = None
    if len(random_joint_blocks) == 3:
        candidate = intersect_aligned_blocks(random_joint_blocks, name="M5_RANDOM")
        if np.array_equal(candidate.intervention_axes, aligned_joint.intervention_axes):
            random_joint = candidate.values
    panels["M5"] = PanelBundle(
        name="M5", modality="JOINT_WIRING_ACTIVITY", context_axes=joint_contexts,
        intervention_axes=aligned_joint.intervention_axes, generic_raw=joint_generic,
        aligned=aligned_joint.values, aligned_feature_names=aligned_joint.feature_names,
        context_ids=tuple(foundation.context_table.set_index("context_axis").loc[joint_contexts, "cell_line_id"].astype(str)),
        intervention_ids=tuple(foundation.intervention_table.set_index("intervention_axis").loc[aligned_joint.intervention_axes, "intervention_id"].astype(str)),
        random_aligned=random_joint,
        notes=["Frozen 40-context wiring/RPPA/Sanger intersection; fitted M5 is RNA + aligned wiring + aligned activity."],
        raw_complete_feature_names=tuple([f"WIRING::{x}" for x in joint_mut.feature_names]
                                         + [f"ACTIVITY::{x}" for x in joint_rppa.feature_names]),
    )

    masks = {
        "WIRING": target_mut_mask & pathway_mut_mask,
        "ACTIVITY": pathway_rppa_mask,
        "PHOSPHO": phospho_mask,
        "TOTAL_PROTEIN": total_mask,
        "DEPENDENCY_ORACLE": dep_target_mask & dep_pathway_mask,
    }
    for name, panel in panels.items():
        audit_rows.append({"panel": name, "feature_axis": ";".join(panel.aligned_feature_names),
                           "status": panel.status, "available_contexts": len(panel.context_axes),
                           "selected_complete_features": panel.generic_raw.shape[1],
                           "eligible_interventions": len(panel.intervention_axes),
                           "reason": "; ".join(panel.notes)})
    return panels, pd.DataFrame(audit_rows)


def complete_case_table(foundation: Foundation, panels: Mapping[str, PanelBundle]) -> pd.DataFrame:
    """Materialize the honest context×intervention eligibility audit."""

    rows: list[dict[str, object]] = []
    context_lookup = foundation.context_table.set_index("context_axis").cell_line_id.astype(str).to_dict()
    intervention_lookup = foundation.intervention_table.set_index("intervention_axis").intervention_id.astype(str).to_dict()
    for panel in panels.values():
        contexts = set(map(int, panel.context_axes))
        interventions = set(map(int, panel.intervention_axes))
        for c in range(50):
            for p in range(93):
                context_ok = c in contexts
                mapping_ok = p in interventions
                eligible = context_ok and mapping_ok
                if not context_ok:
                    reason = "NO_OFFICIAL_PROFILE_ON_PREDECLARED_PANEL"
                elif not mapping_ok:
                    reason = "NO_FROZEN_MAPPING_TO_FULLY_OBSERVED_FEATURE"
                else:
                    reason = "ELIGIBLE"
                rows.append({
                    "panel": panel.name, "modality": panel.modality,
                    "context_axis": c, "context_id": context_lookup[c],
                    "intervention_axis": p, "intervention_id": intervention_lookup[p],
                    "context_profile_available": context_ok, "context_in_panel": context_ok,
                    "mapping_available": mapping_ok, "pair_eligible": eligible,
                    "status": "ELIGIBLE" if eligible else "INELIGIBLE", "reason": reason,
                    "selected_complete_features": panel.generic_raw.shape[1],
                    "aligned_features": ";".join(panel.aligned_feature_names),
                })
    return pd.DataFrame(rows)


def _panel_specs(panel: PanelBundle) -> tuple[ModelSpec, ...]:
    if panel.name == "M5":
        return (ModelSpec("M0_RNA", True, False, False),
                ModelSpec("M5_RNA_WIRING_ACTIVITY", True, False, True))
    if panel.name == "DEPENDENCY_ORACLE":
        return (ModelSpec("M0_RNA", True, False, False),
                ModelSpec("M_ORACLE_RNA_DEPENDENCY", True, False, True))
    return DEFAULT_MODEL_SPECS


def _panel_comparison(panel: PanelBundle) -> tuple[str, str]:
    if panel.name == "M5":
        return "M5_RNA_WIRING_ACTIVITY", "M0_RNA"
    if panel.name == "DEPENDENCY_ORACLE":
        return "M_ORACLE_RNA_DEPENDENCY", "M0_RNA"
    return "M4_RNA_ALIGNED", "M0_RNA"


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _cache_key(panel: PanelBundle, response_target: str, null_label: str,
               response: np.ndarray, rna: np.ndarray, response_cache_id: str | None) -> str:
    payload = {
        "frozen_commit": FROZEN_COMMIT,
        "response_target": response_target,
        "null_label": null_label,
        "panel": panel.name,
        "context_axes": panel.context_axes.tolist(),
        "intervention_axes": panel.intervention_axes.tolist(),
        "response_sha256_or_frozen_id": response_cache_id or _array_digest(response),
        "rna_sha256": _array_digest(rna),
        "generic_sha256": _array_digest(panel.generic_raw),
        "aligned_sha256": _array_digest(panel.aligned),
        "random_aligned_sha256": None if panel.random_aligned is None else _array_digest(panel.random_aligned),
        "specs": [spec.__dict__ for spec in _panel_specs(panel)],
        "response_ranks": RESPONSE_RANKS,
        "generic_ranks": GENERIC_RANKS,
        "ridge_grid": RIDGE_GRID,
        "seed": SEED,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _fold_blocks(
    response: np.ndarray,
    held: int,
    fold_predictions: object,
    panel: PanelBundle,
) -> pd.DataFrame:
    denominator = np.einsum(
        "pg,pg->p", response[0, held], response[1, held], dtype=np.float64,
    )
    grouped: dict[tuple[str, str], list[np.ndarray | None]] = {}
    predictions = getattr(fold_predictions, "predictions")
    for (model, estimator, source), prediction in predictions.items():
        error6 = response[0, held] - prediction
        error14 = response[1, held] - prediction
        numerator = np.einsum("pg,pg->p", error6, error14, dtype=np.float64)
        grouped.setdefault((model, estimator), [None, None])[source] = numerator
    rows: list[dict[str, object]] = []
    for (model, estimator), directional in grouped.items():
        if directional[0] is None or directional[1] is None:
            raise AssertionError("both plate-direction fits are required")
        averaged = 0.5 * (directional[0] + directional[1])
        for local_p, intervention_axis in enumerate(panel.intervention_axes):
            rows.append({
                "model": model, "estimator": estimator,
                "context_axis": int(panel.context_axes[held]),
                "intervention_axis": int(intervention_axis),
                "numerator_after": float(averaged[local_p]),
                "numerator_plate6_fit": float(directional[0][local_p]),
                "numerator_plate14_fit": float(directional[1][local_p]),
                "denominator_u": float(denominator[local_p]),
            })
    return pd.DataFrame(rows)


def run_panel_cached(
    response: np.ndarray,
    rna: np.ndarray,
    panel: PanelBundle,
    paths: RunnerPaths,
    *,
    response_target: str,
    null_label: str = "OBSERVED",
    response_cache_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run/resume exact LOCO fits, checkpointing each frozen sufficient table.

    Fold preprocessing, response bases, designs, sufficient statistics, and
    model selection are training-only inside :func:`fit_predict_outer_fold`.
    The compact cache retains their final OOF sufficient table rather than
    multi-gigabyte gene-space predictions; its key hashes every frozen input.
    """

    key = _cache_key(panel, response_target, null_label, response, rna, response_cache_id)
    folder = paths.cache / response_target / panel.name / null_label
    folder.mkdir(parents=True, exist_ok=True)
    block_parts, fit_parts = [], []
    for held in range(len(panel.context_axes)):
        block_path = folder / f"fold_{held:02d}_blocks.parquet"
        fit_path = folder / f"fold_{held:02d}_fits.parquet"
        manifest_path = folder / f"fold_{held:02d}_manifest.json"
        prepared_path = folder / f"fold_{held:02d}_prepared.npz"
        valid = False
        if block_path.exists() and fit_path.exists() and prepared_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = manifest.get("cache_key") == key and manifest.get("status") == "COMPLETE"
        if valid:
            blocks = pd.read_parquet(block_path)
            fits = pd.read_parquet(fit_path)
        else:
            fold = fit_predict_outer_fold(
                response, held, rna, panel.generic_raw, panel.aligned,
                specs=_panel_specs(panel), response_ranks=RESPONSE_RANKS,
                generic_ranks=GENERIC_RANKS, ridge_grid=RIDGE_GRID,
                intervention_axes=panel.intervention_axes, seed=SEED,
            )
            blocks = _fold_blocks(response, held, fold, panel)
            fits = fold.fits.assign(
                context_axis=int(panel.context_axes[held]), panel=panel.name,
                modality=panel.modality, response_target=response_target, null_label=null_label,
            )
            _atomic_parquet(blocks, block_path)
            _atomic_parquet(fits, fit_path)
            prepared_arrays: dict[str, np.ndarray] = {}
            for source, prepared in (fold.prepared or {}).items():
                for field_name in ("basis", "memory", "max_scores", "truth_norm", "embedding"):
                    prepared_arrays[f"source{source}_{field_name}"] = np.asarray(prepared[field_name], dtype=np.float32)
                for design_name, pair in prepared["selected_designs"].items():
                    safe = re.sub(r"[^A-Za-z0-9]+", "_", design_name).strip("_")
                    prepared_arrays[f"source{source}_{safe}_train"] = np.asarray(pair[0], dtype=np.float32)
                    prepared_arrays[f"source{source}_{safe}_test"] = np.asarray(pair[1], dtype=np.float32)
            _atomic_npz(prepared_path, prepared_arrays)
            write_json(manifest_path, {
                "status": "COMPLETE", "created_at": _utc(), "cache_key": key,
                "held_local_axis": held, "held_context_axis": int(panel.context_axes[held]),
                "training_only_preprocessing": True, "fold_local_response_basis": True,
                "fold_local_design": True, "compact_sufficient_table_cache": True,
                "full_predictions_cached": False,
                "prepared_cache": str(prepared_path.name),
            })
        block_parts.append(blocks.assign(panel=panel.name, modality=panel.modality,
                                         response_target=response_target, null_label=null_label))
        fit_parts.append(fits)
    return pd.concat(block_parts, ignore_index=True), pd.concat(fit_parts, ignore_index=True)


def _load_response(paths: RunnerPaths, member: str) -> np.ndarray:
    if member not in {"residual", "gamma"}:
        raise ValueError("response member must be residual or gamma")
    path = paths.source_root / "data/cgc_bio1_official/bio1_gene_arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        if member not in archive.files:
            raise RuntimeError(f"MULTI2_RESPONSE_MEMBER_ABSENT: {member}")
        response = np.asarray(archive[member], dtype=np.float32)
    if response.shape != (2, 50, 93, 25_695) or not np.isfinite(response).all():
        raise RuntimeError(f"MULTI2_RESPONSE_AXIS_MISMATCH: {member} {response.shape}")
    return response


def _summaries_for_panel(
    blocks: pd.DataFrame,
    panel: PanelBundle,
    response_target: str,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[tuple[str, str]]
    if panel.name in {"M5", "DEPENDENCY_ORACLE"}:
        comparisons = [_panel_comparison(panel)]
    else:
        comparisons = [
            ("M4_RNA_ALIGNED", "M0_RNA"),
            ("M4_RNA_ALIGNED", "M3_RNA_GENERIC"),
            ("M2_ALIGNED", "M1_GENERIC"),
            ("M3_RNA_GENERIC", "M0_RNA"),
        ]
    summary_rows, gain_parts = [], []
    model, comparator = _panel_comparison(panel)
    id_map = annotations.set_index("intervention_axis").intervention_id.astype(str).to_dict()
    for estimator in ("ridge", "bilinear_ridge"):
        for model_name, comparator_name in comparisons:
            row = paired_incremental_summary(
                blocks, model_name, comparator_name, estimator, draws=NULL_DRAWS,
                seed=SEED + (0 if estimator == "ridge" else 1),
            )
            summary_rows.append({"response_target": response_target, "panel": panel.name,
                                 "modality": panel.modality, **row, "status": "COMPLETE"})
        gain = per_intervention_gains(
            blocks, model=model, comparator=comparator, estimator=estimator,
            modality=panel.modality, intervention_ids=id_map, annotations=annotations,
            response_target=response_target,
        )
        gain.insert(1, "panel", panel.name)
        gain["bridge_primary"] = bool(
            response_target == "PRIMARY_RESIDUAL"
            and panel.name == "M5"
            and estimator == "bilinear_ridge"
        )
        gain["status"] = "COMPLETE"
        gain_parts.append(gain)
    return pd.DataFrame(summary_rows), pd.concat(gain_parts, ignore_index=True)


def run_real_analysis(
    paths: RunnerPaths,
    foundation: Foundation,
    panels: Mapping[str, PanelBundle],
    *,
    targets: Sequence[str] = ("residual", "gamma"),
) -> dict[str, pd.DataFrame]:
    """Explicitly unseal and execute the frozen real-data primary/secondary run."""

    verify_frozen_hashes(paths, include_large_response=True)
    summary_parts, gain_parts, fit_parts, block_parts = [], [], [], []
    for member in targets:
        response_all = _load_response(paths, member)
        response_target = "PRIMARY_RESIDUAL" if member == "residual" else "SECONDARY_GAMMA"
        for panel in panels.values():
            response = response_all[:, panel.context_axes][:, :, panel.intervention_axes]
            blocks, fits = run_panel_cached(
                response, foundation.rna[panel.context_axes], panel, paths,
                response_target=response_target,
                response_cache_id=f"{SOURCE_HASHES['data/cgc_bio1_official/bio1_gene_arrays.npz']}::{member}",
            )
            summaries, gains = _summaries_for_panel(blocks, panel, response_target, foundation.annotations)
            summary_parts.append(summaries); gain_parts.append(gains)
            fit_parts.append(fits); block_parts.append(blocks)
        del response_all
    outputs = {
        "incremental": pd.concat(summary_parts, ignore_index=True),
        "gains": pd.concat(gain_parts, ignore_index=True),
        "fits": pd.concat(fit_parts, ignore_index=True),
        "blocks": pd.concat(block_parts, ignore_index=True),
    }
    outputs["incremental"].to_csv(paths.out / "MULTI2_INCREMENTAL_RECOVERY.csv", index=False)
    outputs["gains"].to_csv(paths.out / "MULTI2_PER_INTERVENTION_GAINS.csv", index=False)
    _atomic_parquet(outputs["fits"], paths.out / "MULTI2_FITS.parquet")
    _atomic_parquet(outputs["blocks"], paths.out / "MULTI2_OOF_UTILITY_BLOCKS.parquet")
    return outputs


def _copy_panel(panel: PanelBundle, *, name: str | None = None,
                generic: np.ndarray | None = None, aligned: np.ndarray | None = None,
                random_aligned: np.ndarray | None = None,
                note: str = "") -> PanelBundle:
    return PanelBundle(
        name=panel.name if name is None else name, modality=panel.modality,
        context_axes=panel.context_axes.copy(), intervention_axes=panel.intervention_axes.copy(),
        generic_raw=panel.generic_raw.copy() if generic is None else np.asarray(generic, float),
        aligned=panel.aligned.copy() if aligned is None else np.asarray(aligned, float),
        aligned_feature_names=panel.aligned_feature_names, context_ids=panel.context_ids,
        intervention_ids=panel.intervention_ids,
        random_aligned=(panel.random_aligned.copy() if random_aligned is None and panel.random_aligned is not None
                        else random_aligned),
        status=panel.status,
        notes=[*panel.notes, note] if note else list(panel.notes),
        raw_complete_feature_names=panel.raw_complete_feature_names,
    )


def _first_nonidentity_order(annotations: pd.DataFrame, columns: Sequence[str], *, seed: int) -> np.ndarray | None:
    from .multi2 import blocked_permutation_orders

    for offset in range(100):
        order = blocked_permutation_orders(annotations, columns, draws=1, seed=seed + offset)[0]
        if not np.array_equal(order, np.arange(len(order))):
            return order.astype(int)
    return None


def _delta_table(blocks: pd.DataFrame, model: str, comparator: str, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    local = blocks.loc[blocks.estimator.eq(estimator)]
    a = local.loc[local.model.eq(model)].pivot(index="context_axis", columns="intervention_axis", values="numerator_after")
    b = local.loc[local.model.eq(comparator)].pivot(index="context_axis", columns="intervention_axis", values="numerator_after")
    d = local.loc[local.model.eq(model)].pivot(index="context_axis", columns="intervention_axis", values="denominator_u")
    if not a.index.equals(b.index) or not a.columns.equals(b.columns) or not a.index.equals(d.index) or not a.columns.equals(d.columns):
        raise RuntimeError("MULTI2_NULL_UNPAIRED_OOF_TABLE")
    return b.to_numpy(float) - a.to_numpy(float), d.to_numpy(float)


def _mapping_null_panel(
    panel: PanelBundle,
    foundation: Foundation,
    response_magnitude: np.ndarray,
    *,
    pathway: bool,
) -> tuple[PanelBundle | None, str]:
    axes = panel.intervention_axes
    ann = foundation.annotations.set_index("intervention_axis").loc[axes].reset_index()
    ann["target_set_size"] = [len(foundation.exact_targets[int(axis)]) for axis in axes]
    ann["pathway_set_size"] = [len(foundation.reactome_members[int(axis)]) for axis in axes]
    ann["annotation_coverage"] = ann.exact_targets.fillna("").astype(str).ne("").astype(int)
    ann["moa_frequency"] = ann.groupby("moa_fine", dropna=False).moa_fine.transform("size")
    ann["response_magnitude_quintile"] = pd.qcut(
        pd.Series(response_magnitude, index=ann.index).rank(method="first"), 5, labels=False,
    ).astype(int)
    columns = ["pathway_set_size", "annotation_coverage"] if pathway else [
        "target_set_size", "moa_frequency", "annotation_coverage", "response_magnitude_quintile",
    ]
    order = _first_nonidentity_order(ann, columns, seed=SEED + (101 if pathway else 0))
    if order is None:
        return None, "NOT_EXECUTABLE_NO_EXCHANGEABLE_FROZEN_MAPPING_BLOCK"
    selected = [i for i, name in enumerate(panel.aligned_feature_names)
                if ("pathway" in name.lower()) == pathway]
    if not selected:
        return None, "NOT_EXECUTABLE_NO_MATCHING_ALIGNED_FEATURE_AXIS"
    values = panel.aligned.copy()
    values[:, :, selected] = panel.aligned[:, order][:, :, selected]
    label = "PATHWAY_MAPPING_PERMUTATION" if pathway else "DRUG_TARGET_MAPPING_PERMUTATION"
    return _copy_panel(panel, aligned=values, note=f"actual OOF rerun with {label}"), label


def _context_null_panel(panel: PanelBundle, foundation: Foundation) -> tuple[PanelBundle | None, str]:
    lineage = foundation.lineages[panel.context_axes]
    order = context_permutation_orders(lineage, draws=1, seed=SEED + 211)[0].astype(int)
    if np.array_equal(order, np.arange(len(order))):
        return None, "NOT_EXECUTABLE_NO_WITHIN_LINEAGE_EXCHANGE"
    return _copy_panel(
        panel, generic=panel.generic_raw[order], aligned=panel.aligned[order],
        note="actual OOF rerun with context modality profiles permuted within frozen lineage",
    ), "CONTEXT_MODALITY_WITHIN_LINEAGE"


def _random_gene_null_panel(panel: PanelBundle) -> tuple[PanelBundle | None, str]:
    if panel.random_aligned is None:
        return None, "NOT_EXECUTABLE_INSUFFICIENT_EXCLUDED_FULLY_OBSERVED_FEATURE_POOL"
    return _copy_panel(
        panel, aligned=panel.random_aligned,
        note="actual OOF rerun with deterministic measured-size/coverage matched random features",
    ), "RANDOM_GENE_SET_MATCHED"


def run_null_analysis(
    paths: RunnerPaths,
    foundation: Foundation,
    panels: Mapping[str, PanelBundle],
    *,
    response_member: str = "residual",
) -> pd.DataFrame:
    """Run feature-level nulls as actual OOF refits, never output-label shuffles."""

    verify_frozen_hashes(paths, include_large_response=True)
    response_all = _load_response(paths, response_member)
    response_target = "PRIMARY_RESIDUAL" if response_member == "residual" else "SECONDARY_GAMMA"
    rows: list[dict[str, object]] = []
    for panel in panels.values():
        response = response_all[:, panel.context_axes][:, :, panel.intervention_axes]
        observed_blocks, _ = run_panel_cached(
            response, foundation.rna[panel.context_axes], panel, paths, response_target=response_target,
            response_cache_id=f"{SOURCE_HASHES['data/cgc_bio1_official/bio1_gene_arrays.npz']}::{response_member}",
        )
        response_magnitude = np.mean(np.abs(response), axis=(0, 1, 3))
        null_builders = [
            ("DRUG_TARGET_MAPPING_PERMUTATION", lambda: _mapping_null_panel(
                panel, foundation, response_magnitude, pathway=False)),
            ("PATHWAY_MAPPING_PERMUTATION", lambda: _mapping_null_panel(
                panel, foundation, response_magnitude, pathway=True)),
            ("CONTEXT_MODALITY_WITHIN_LINEAGE", lambda: _context_null_panel(panel, foundation)),
            ("RANDOM_GENE_SET_MATCHED", lambda: _random_gene_null_panel(panel)),
        ]
        for requested_label, builder in null_builders:
            null_panel, label_or_status = builder()
            if null_panel is None:
                rows.append({"response_target": response_target, "panel": panel.name,
                             "modality": panel.modality, "null_type": requested_label,
                             "estimator": np.nan, "observed_delta_g": np.nan, "null_delta_g": np.nan,
                             "contrast_delta_g": np.nan, "contrast_ci_low": np.nan,
                             "contrast_ci_high": np.nan,
                             "null_p": np.nan, "table_draws": 0, "oof_refits": 0,
                             "status": label_or_status})
                continue
            label = label_or_status
            null_blocks, _ = run_panel_cached(
                response, foundation.rna[panel.context_axes], null_panel, paths,
                response_target=response_target, null_label=label,
                response_cache_id=f"{SOURCE_HASHES['data/cgc_bio1_official/bio1_gene_arrays.npz']}::{response_member}",
            )
            model, comparator = _panel_comparison(panel)
            for estimator in ("ridge", "bilinear_ridge"):
                observed_delta, observed_den = _delta_table(observed_blocks, model, comparator, estimator)
                null_delta, null_den = _delta_table(null_blocks, model, comparator, estimator)
                if not np.allclose(observed_den, null_den, rtol=0, atol=1e-10):
                    raise RuntimeError("MULTI2_NULL_DENOMINATOR_CHANGED_ACROSS_PAIRED_OOF_REFIT")
                observed = float(observed_delta.sum() / observed_den.sum())
                null_estimate = float(null_delta.sum() / null_den.sum())
                contrast_delta = observed_delta - null_delta
                contrast = float(contrast_delta.sum() / observed_den.sum())
                draws = hierarchical_bootstrap(
                    contrast_delta, observed_den, draws=NULL_DRAWS, seed=SEED + len(rows),
                )
                ci_low, ci_high = np.nanquantile(draws, [0.025, 0.975])
                pvalue = float((1 + np.sum(draws <= 0.0)) / (NULL_DRAWS + 1))
                rows.append({"response_target": response_target, "panel": panel.name,
                             "modality": panel.modality, "null_type": label, "estimator": estimator,
                             "observed_delta_g": observed, "null_delta_g": null_estimate,
                             "contrast_delta_g": contrast, "contrast_ci_low": float(ci_low),
                             "contrast_ci_high": float(ci_high),
                             "null_p": pvalue, "table_draws": NULL_DRAWS, "oof_refits": 1,
                             "status": "COMPLETE_ACTUAL_FEATURE_NULL_OOF_REFIT"})
    del response_all
    result = pd.DataFrame(rows)
    result.to_csv(paths.out / "MULTI2_ALIGNMENT_NULLS.csv", index=False)
    return result


POWER_NUMERIC_IMPLEMENTATION_COMMIT = "3ef635e6e619ef76b1274aac77bca1be533875ab"
POWER_CACHE_COMPATIBLE_COMMITS = (
    POWER_NUMERIC_IMPLEMENTATION_COMMIT,
    # Scheduling-only --panel support; no numerical power code changed.
    "c27ae6948d6ca466893e245e89b8d8fd91a9dfe1",
)


def _power_cache_key(
    panel: PanelBundle,
    response_member: str,
    response: np.ndarray,
    rna: np.ndarray,
    *,
    implementation: str = POWER_NUMERIC_IMPLEMENTATION_COMMIT,
) -> str:
    payload = {
        "implementation": implementation,
        "protocol_commit": FROZEN_COMMIT,
        "response_member": response_member,
        "response_frozen_id": SOURCE_HASHES["data/cgc_bio1_official/bio1_gene_arrays.npz"],
        "panel": panel.name,
        "contexts": panel.context_axes.tolist(),
        "interventions": panel.intervention_axes.tolist(),
        "rna_sha256": _array_digest(rna),
        "aligned_sha256": _array_digest(panel.aligned),
        "levels": POWER_LEVELS,
        "replicates": POWER_REPLICATES,
        "response_ranks": RESPONSE_RANKS,
        "ridge_grid": RIDGE_GRID,
        "seed": SEED,
        "adapter": "FOLD_LOCAL_COEFFICIENT_PLUS_ORTHOGONAL_GENE_METRIC_V1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _power_cache_keys(
    panel: PanelBundle, response_member: str, response: np.ndarray, rna: np.ndarray,
) -> set[str]:
    """Accept caches from the frozen numeric kernel and scheduling-only wrapper."""

    return {
        _power_cache_key(
            panel, response_member, response, rna, implementation=implementation,
        )
        for implementation in POWER_CACHE_COMPATIBLE_COMMITS
    }


_POWER_FOLD_ARRAYS = (
    "basis", "x_base_train", "x_base_test", "x_aug_train", "x_aug_test",
    "memory_coeff", "affine_coeff", "signal_coeff", "residual_cross",
    "residual_norm_source", "memory_orthogonal_norm",
    "memory_orthogonal_dot_affine_sum",
)


def _load_or_prepare_power_folds(
    response: np.ndarray,
    rna: np.ndarray,
    panel: PanelBundle,
    paths: RunnerPaths,
    *,
    response_member: str,
    cache_key: str,
) -> list[FoldLocalPowerFold]:
    """Checkpoint every expensive outer-fold basis before calibration."""

    folder = paths.cache / "power" / f"{response_member}_{panel.name}_prepared"
    folder.mkdir(parents=True, exist_ok=True)
    folds: list[FoldLocalPowerFold] = []
    for held in range(len(panel.context_axes)):
        array_path = folder / f"fold_{held:02d}.npz"
        manifest_path = folder / f"fold_{held:02d}.json"
        valid = False
        if array_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (
                manifest.get("status") == "COMPLETE"
                and manifest.get("cache_key") == cache_key
                and manifest.get("held_context") == held
            )
        if valid:
            with np.load(array_path, allow_pickle=False) as archive:
                for source in (0, 1):
                    prefix = f"source{source}_"
                    values = {name: np.asarray(archive[prefix + name]) for name in _POWER_FOLD_ARRAYS}
                    folds.append(FoldLocalPowerFold(
                        held_context=held,
                        train_contexts=np.asarray(archive[prefix + "train_contexts"], dtype=np.int64),
                        source_plate=source,
                        intervention_axes=panel.intervention_axes.copy(),
                        **values,
                    ))
            continue
        prepared = prepare_fold_local_power_folds(
            response, rna, panel.aligned,
            intervention_axes=panel.intervention_axes,
            response_ranks=RESPONSE_RANKS,
            bilinear=True,
            seed=SEED,
            held_contexts=(held,),
        )
        if len(prepared) != 2 or {fold.source_plate for fold in prepared} != {0, 1}:
            raise RuntimeError("MULTI2_POWER_FOLD_PREPARATION_INCOMPLETE")
        arrays: dict[str, np.ndarray] = {}
        for fold in prepared:
            prefix = f"source{fold.source_plate}_"
            arrays[prefix + "train_contexts"] = np.asarray(fold.train_contexts, dtype=np.int16)
            for name in _POWER_FOLD_ARRAYS:
                arrays[prefix + name] = np.asarray(getattr(fold, name), dtype=np.float32)
        _atomic_npz(array_path, arrays)
        write_json(manifest_path, {
            "status": "COMPLETE", "created_at": _utc(), "cache_key": cache_key,
            "held_context": held, "sources": [0, 1],
            "training_only_response_basis": True,
            "held_outcome_used_for_fit_or_injection": False,
        })
        folds.extend(prepared)
    return folds


def run_power_analysis(
    paths: RunnerPaths,
    foundation: Foundation,
    panels: Mapping[str, PanelBundle],
    *,
    response_member: str = "residual",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run/resume the reviewed fold-local 200x4 coefficient-space audit."""

    verify_frozen_hashes(paths, include_large_response=True)
    response_all = _load_response(paths, response_member)
    response_target = "PRIMARY_RESIDUAL" if response_member == "residual" else "SECONDARY_GAMMA"
    curve_parts, limit_rows = [], []
    power_dir = paths.cache / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    for panel in panels.values():
        response = response_all[:, panel.context_axes][:, :, panel.intervention_axes]
        rna = foundation.rna[panel.context_axes]
        key = _power_cache_key(panel, response_member, response, rna)
        compatible_keys = _power_cache_keys(panel, response_member, response, rna)
        curve_path = power_dir / f"{response_target}_{panel.name}_curves.csv"
        replicate_path = power_dir / f"{response_target}_{panel.name}_replicates.parquet"
        manifest_path = power_dir / f"{response_target}_{panel.name}_manifest.json"
        valid = False
        if curve_path.exists() and replicate_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (
                manifest.get("status") == "COMPLETE"
                and manifest.get("cache_key") in compatible_keys
            )
        if valid:
            curves = pd.read_csv(curve_path)
            detection = manifest["mde80"]
        else:
            folds = _load_or_prepare_power_folds(
                response, rna, panel, paths,
                response_member=response_member,
                cache_key=key,
            )
            result = fold_specific_power_calibration(
                folds,
                levels=POWER_LEVELS,
                replicates=POWER_REPLICATES,
                response_ranks=RESPONSE_RANKS,
                ridge_grid=RIDGE_GRID,
                bootstrap_draws=500,
                batch_size=10,
                seed=SEED,
            )
            curves, detection = result.curves, result.detection_limit
            curves.to_csv(curve_path, index=False)
            _atomic_parquet(result.replicates, replicate_path)
            write_json(manifest_path, {
                "status": "COMPLETE", "created_at": _utc(), "cache_key": key,
                "response_target": response_target, "panel": panel.name,
                "contexts": len(panel.context_axes), "interventions": len(panel.intervention_axes),
                "folds": len(folds), "replicates": POWER_REPLICATES,
                "levels": list(POWER_LEVELS), "bootstrap_draws_per_replicate": 500,
                "mde80": detection, "device": "CPU_ONLY",
                "training_only_response_basis": True,
                "held_outcome_used_for_fit_or_injection": False,
                "gene_metric_reconstruction": "coefficient cross-product plus exact omitted orthogonal energy",
            })
            del folds
        curves = curves.assign(
            response_target=response_target,
            panel=panel.name,
            modality=panel.modality,
            status="COMPLETE_FOLD_LOCAL_GENE_METRIC_EQUIVALENT",
        )
        curve_parts.append(curves)
        limit_rows.append({
            "response_target": response_target, "panel": panel.name,
            "modality": panel.modality, "estimator": "bilinear_ridge",
            "mde80": detection, "replicates": POWER_REPLICATES,
            "levels": ";".join(map(str, POWER_LEVELS)), "device": "cpu",
            "status": "COMPLETE_FOLD_LOCAL_GENE_METRIC_EQUIVALENT",
            "reason": "training-only response basis and exact coefficient+orthogonal cross-product reconstruction",
        })
    del response_all
    curves = pd.concat(curve_parts, ignore_index=True)
    limits = pd.DataFrame(limit_rows)
    curves.to_csv(paths.out / "MULTI2_POWER_CURVES.csv", index=False)
    limits.to_csv(paths.out / "MULTI2_DETECTION_LIMITS.csv", index=False)
    return curves, limits


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def run_audit(paths: RunnerPaths, *, rna_override: np.ndarray | None = None) -> tuple[Foundation, dict[str, PanelBundle]]:
    """Outcome-sealed feature audit and eligibility report."""

    initialize_outputs(paths)
    hashes = verify_frozen_hashes(paths, include_large_response=False)
    foundation = load_foundation(paths, rna_override=rna_override)
    panels, panel_audit = build_panels(paths, foundation)
    complete = complete_case_table(foundation, panels)
    complete.to_csv(paths.out / "MULTI2_COMPLETE_CASE_MATRIX.csv", index=False)
    hashes.to_csv(paths.out / "MULTI2_INPUT_HASH_AUDIT.csv", index=False)
    panel_audit.to_csv(paths.out / "MULTI2_PANEL_AUDIT.csv", index=False)
    manifest = {
        "phase": "CGC-MULTI-2 outcome-sealed input audit",
        "created_at": _utc(), "frozen_commit": FROZEN_COMMIT,
        "current_branch": _git_value(paths.root, "branch", "--show-current"),
        "current_commit": _git_value(paths.root, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value(paths.root, "status", "--porcelain")),
        "source_root": str(paths.source_root), "outcomes_opened": False,
        "device": "CPU_ONLY", "seed": SEED,
        "panel_audit_sha256": _hash_frame(panel_audit),
        "complete_case_sha256": _hash_frame(complete),
        "panels": {
            name: {"contexts": len(panel.context_axes), "interventions": len(panel.intervention_axes),
                   "complete_features": panel.generic_raw.shape[1],
                   "aligned_features": list(panel.aligned_feature_names)}
            for name, panel in panels.items()
        },
    }
    write_json(paths.out / "MULTI2_RUN_MANIFEST.json", manifest)
    return foundation, panels


def _report(paths: RunnerPaths) -> None:
    panel = pd.read_csv(paths.out / "MULTI2_PANEL_AUDIT.csv")
    manifest = json.loads((paths.out / "MULTI2_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    columns = panel.columns.tolist()
    markdown = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in panel.itertuples(index=False, name=None):
        markdown.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    lines = [
        "# CGC-MULTI-2 frozen runner status", "",
        f"- Frozen commit: `{manifest['frozen_commit']}`",
        f"- Current provenance: `{manifest['current_branch']}` / `{manifest['current_commit']}`",
        f"- Outcomes opened during audit: `{manifest['outcomes_opened']}`",
        f"- Device: `{manifest['device']}`", "", "## Outcome-independent panels", "",
        *markdown, "",
        "The real outcome run, feature-null refits, and power stage require separate explicit commands.",
        "A stage that cannot preserve the frozen feature correspondence is written as `NOT_EXECUTABLE`; it is never replaced by an output-label shuffle.",
    ]
    (paths.out / "MULTI2_RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen CGC-MULTI-2 real-data runner (CPU only)")
    parser.add_argument("command", choices=("audit", "run", "nulls", "power", "all"))
    parser.add_argument("--formal", action="store_true",
                        help="required acknowledgement before any audit or outcome-bearing stage")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--panel", action="append", choices=(
            "WIRING", "ACTIVITY", "PHOSPHO", "TOTAL_PROTEIN",
            "DEPENDENCY_ORACLE", "M5",
        ),
        help="run only the named frozen panel during the power stage; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.formal:
        parser.error("--formal is required; no data were opened")
    paths = RunnerPaths(args.root.resolve(), args.source_root.resolve(), args.out.resolve())
    foundation, panels = run_audit(paths)
    if args.panel and args.command not in {"power"}:
        parser.error("--panel is a scheduling-only option for the power command")
    if args.panel:
        selected = set(args.panel)
        panels = {name: panel for name, panel in panels.items() if name in selected}
    if args.command in {"run", "all"}:
        run_real_analysis(paths, foundation, panels)
    if args.command in {"nulls", "all"}:
        run_null_analysis(paths, foundation, panels)
    if args.command in {"power", "all"}:
        run_power_analysis(paths, foundation, panels)
    _report(paths)
    return 0


__all__ = [
    "FROZEN_COMMIT", "FROZEN_HASHES", "Foundation", "PanelBundle", "REQUIRED_OUTPUTS",
    "RunnerPaths", "SOURCE_HASHES", "build_panels", "build_parser", "complete_case_table",
    "initialize_outputs", "load_foundation", "main", "run_audit", "run_null_analysis",
    "run_panel_cached", "run_power_analysis", "run_real_analysis", "verify_frozen_hashes",
]


if __name__ == "__main__":
    raise SystemExit(main())
