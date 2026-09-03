"""Secondary BIO-2 × MULTI-2 bridge adjudication on frozen primary outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .common import OUT, git, write_json


SEED = 9201
N_DRAW = 10_000


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    rx = rankdata(x[valid])
    ry = rankdata(y[valid])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _exchange_blocks(frame: pd.DataFrame) -> np.ndarray:
    """Freeze repeated MOA/family groups before observing bridge correlation."""

    fine = frame["moa_fine"].fillna("").astype(str)
    family = frame["hgnc_target_family_ids"].fillna("").astype(str)
    broad = frame["moa_broad"].fillna("UNCLEAR").astype(str)
    fine_count = fine.value_counts()
    family_count = family.value_counts()
    labels: list[str] = []
    for f, fam, b in zip(fine, family, broad, strict=True):
        if f and f.casefold() != "unclear" and fine_count.get(f, 0) >= 2:
            labels.append(f"MOA:{f}")
        elif fam and family_count.get(fam, 0) >= 2:
            labels.append(f"FAMILY:{fam}")
        else:
            labels.append(f"BROAD:{b}")
    return np.asarray(labels, dtype=object)


def run_bridge(out: Path = OUT) -> dict[str, object]:
    bio = pd.read_csv(out / "BIO2_PER_INTERVENTION_ORGANIZATION.csv")
    multi = pd.read_csv(out / "MULTI2_PER_INTERVENTION_GAINS.csv")
    annotations = pd.read_csv(out / "BIO2_FROZEN_INTERVENTION_ANNOTATIONS.csv")

    # Freeze exchangeability from the outcome-independent annotation table
    # before merging.  The MULTI-2 gain table intentionally carries copies of
    # these labels for reporting; computing after the merge would suffix the
    # duplicate column names and make the preregistered bridge non-executable.
    annotations = annotations.copy()
    annotations["bridge_exchange_block"] = _exchange_blocks(annotations)

    # The primary MULTI-2 row is predeclared by the implementation; bridge code
    # never chooses the best modality from outcomes.
    if "bridge_primary" in multi:
        multi = multi[multi.bridge_primary.astype(bool)]
    if multi.intervention_id.duplicated().any():
        raise RuntimeError("MULTI2 bridge-primary table must contain one row per intervention")
    joined = annotations.merge(bio, on=["intervention_axis", "intervention_id"], how="left")
    joined = joined.merge(multi, on=["intervention_axis", "intervention_id"], how="left", suffixes=("_bio", "_multi"))
    bcol = "organization_score"
    mcol = "delta_g_aligned_minus_rna"
    observed = _spearman(joined[bcol].to_numpy(float), joined[mcol].to_numpy(float))

    blocks = joined["bridge_exchange_block"].to_numpy(dtype=object)
    valid = np.isfinite(joined[bcol]) & np.isfinite(joined[mcol])
    work = joined.loc[valid].reset_index(drop=True)
    work_blocks = blocks[valid]
    groups = [np.flatnonzero(work_blocks == name) for name in sorted(set(work_blocks))]
    rng = np.random.default_rng(SEED)
    null = np.empty(N_DRAW, dtype=np.float64)
    original_m = work[mcol].to_numpy(float)
    for draw in range(N_DRAW):
        permuted = original_m.copy()
        for axes in groups:
            if len(axes) > 1:
                permuted[axes] = original_m[rng.permutation(axes)]
        null[draw] = _spearman(work[bcol].to_numpy(float), permuted)
    finite = np.isfinite(null)
    p_two = float((1 + np.sum(np.abs(null[finite]) >= abs(observed))) / (1 + finite.sum())) if np.isfinite(observed) else np.nan
    p_pos = float((1 + np.sum(null[finite] >= observed)) / (1 + finite.sum())) if np.isfinite(observed) else np.nan

    joined.to_csv(out / "BIO2_MULTI2_BRIDGE.csv", index=False)
    pd.DataFrame({"draw": np.arange(N_DRAW), "rho": null}).to_parquet(out / "bio2_multi2_bridge_null.parquet", index=False)
    result = {
        "phase": "BIO2_MULTI2_BRIDGE",
        "secondary_analysis": True,
        "n_complete_interventions": int(valid.sum()),
        "spearman_rho": observed,
        "blocked_permutation_p_two_sided": p_two,
        "blocked_permutation_p_positive": p_pos,
        "draws": N_DRAW,
        "seed": SEED,
        "permutation": "within frozen repeated MOA, then target-family, then broad-action exchange blocks",
        "git_commit": git("rev-parse", "HEAD"),
    }
    write_json(out / "BIO2_MULTI2_BRIDGE_RESULT.json", result)
    report = f"""# BIO-2 × MULTI-2 Bridge Report

This was run only after both primary analyses were frozen. It is secondary and
was not used to choose a model, modality, hierarchy level, or subgroup.

- Complete intervention pairs: **{int(valid.sum())}**
- Spearman rho(B_p, M_p): **{observed:.6f}**
- Structure-preserving permutation p (two-sided): **{p_two:.6g}**
- Positive-direction p: **{p_pos:.6g}**

The null permutes MULTI-2 gains only within the predeclared repeated MOA,
target-family, or broad-action exchange blocks. A null result means that the
groups with stronger residual organization were not preferentially rescued by
the available aligned modalities; it does not erase either primary analysis.
"""
    (out / "BIO2_MULTI2_BRIDGE_REPORT.md").write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_bridge(), indent=2))
