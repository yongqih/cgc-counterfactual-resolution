#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frozen_wgcna_data_authority", ROOT / "scripts/run_wgcna_resolution_frontier.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen Lea authority")
FRONTIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FRONTIER
SPEC.loader.exec_module(FRONTIER)

OUT = ROOT / "results/cgc_full_span_pca_sanity"
HISTORICAL_CURVE = ROOT / "results/cgc_dimension_matched_compression/GLOBAL_PCA_ORACLE_CURVE.csv"
HISTORICAL_SUMMARY = ROOT / "results/cgc_dimension_matched_compression/DIMENSION_MATCHED_COMPRESSION_SUMMARY.json"
EXPECTED_HASHES = {
    FRONTIER.REPRESENTATION: "7f44947ec59a43eb954ec2032e4b9c3733306542bcd117e03f584d72b3ecf59b",
    HISTORICAL_CURVE: "31c7e72d76becb9fd78f2f52ad85fd8dff5f91195981f70649bfd78bac4d21e6",
    HISTORICAL_SUMMARY: "220b98089ff92008b86b94920127ec322a547ae27995fe414e1920960c565516",
}
DIMENSIONS = (4, 8, 16, 32, 64, 128, 256)
N_GENES = 10_110
ORTHOGONALITY_TOLERANCE = 1e-10
RECONSTRUCTION_TOLERANCE = 1e-12
HISTORICAL_TOLERANCE = 1e-10
IDENTITY_TOLERANCE = 1e-10
SEED = 207049


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_authorities() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"FULL_SPAN_AUTHORITY_HASH_MISMATCH:{path.name}")
    summary = json.loads(HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("verdict") != "BLOCK_STRUCTURED_RESPONSE_BASIS_NOT_SUPPORTED":
        raise RuntimeError("FULL_SPAN_DIMENSION_AUDIT_AUTHORITY_INVALID")


@dataclass(frozen=True)
class DirectSVD:
    mean: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    numerical_rank: int
    tolerance: float

    @property
    def retained(self) -> np.ndarray:
        return self.right_vectors[: self.numerical_rank]

    @property
    def orthogonality_error(self) -> float:
        gram = self.retained @ self.retained.T
        return float(np.max(np.abs(gram - np.eye(self.numerical_rank))))


def direct_svd(matrix: np.ndarray) -> DirectSVD:
    values = np.asarray(matrix, dtype=np.float64)
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
    numerical_rank = int(np.sum(singular_values > tolerance))
    if numerical_rank < 1:
        raise RuntimeError("GLOBAL_PCA_METRIC_OR_IMPLEMENTATION_ISSUE:ZERO_RANK")
    oriented = right.copy()
    for row in oriented[:numerical_rank]:
        largest = int(np.argmax(np.abs(row)))
        if row[largest] < 0:
            row *= -1
    return DirectSVD(mean, singular_values, oriented, numerical_rank, float(tolerance))


def affine_reconstruction(truth: np.ndarray, svd: DirectSVD, dimension: int) -> np.ndarray:
    vectors = svd.right_vectors[:dimension]
    centered = np.asarray(truth, dtype=np.float64) - svd.mean
    return svd.mean + (centered @ vectors.T) @ vectors


def fidelity(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    denominator = float(np.square(truth, dtype=np.float64).sum())
    sse = float(np.square(np.asarray(truth, dtype=np.float64) - prediction).sum())
    return denominator, sse, 1.0 - sse / denominator


def geometric_rows(
    truth: np.ndarray,
    svd: DirectSVD,
    fold: int,
    sample_indices: np.ndarray,
    lines: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vectors = svd.retained
    raw = np.asarray(truth, dtype=np.float64)
    raw_scores = raw @ vectors.T
    raw_projection = raw_scores @ vectors
    raw_residual = raw - raw_projection
    centered = raw - svd.mean
    affine_scores = centered @ vectors.T
    affine_projection = affine_scores @ vectors
    affine_residual = centered - affine_projection
    energy_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for row, sample_index in enumerate(sample_indices):
        raw_energy = float(np.square(raw[row]).sum())
        projected_energy = float(np.square(raw_projection[row]).sum())
        residual_energy = float(np.square(raw_residual[row]).sum())
        identity_error = abs(raw_energy - projected_energy - residual_energy)
        relative_error = identity_error / max(raw_energy, np.finfo(np.float64).tiny)
        affine_energy = float(np.square(centered[row]).sum())
        affine_projected = float(np.square(affine_projection[row]).sum())
        affine_residual_energy = float(np.square(affine_residual[row]).sum())
        energy_rows.append({
            "outer_fold": fold,
            "sample_index": int(sample_index),
            "lcl_id": str(lines[sample_index]),
            "raw_response_energy": raw_energy,
            "raw_projected_energy": projected_energy,
            "raw_residual_energy": residual_energy,
            "q_raw": projected_energy / raw_energy,
            "affine_centered_energy": affine_energy,
            "affine_projected_energy": affine_projected,
            "affine_residual_energy": affine_residual_energy,
            "q_affine": affine_projected / affine_energy,
        })
        identity_rows.append({
            "outer_fold": fold,
            "sample_index": int(sample_index),
            "lcl_id": str(lines[sample_index]),
            "raw_response_energy": raw_energy,
            "raw_projected_plus_residual_energy": projected_energy + residual_energy,
            "absolute_identity_error": identity_error,
            "relative_identity_error": relative_error,
            "status": "PASS" if relative_error <= IDENTITY_TOLERANCE else "FAIL",
        })
    return energy_rows, identity_rows


def pooled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("budget_label", sort=False):
        denominator = float(group["denominator"].sum())
        sse = float(group["oracle_sse"].sum())
        row = {
            "outer_fold": "pooled",
            "budget_label": label,
            "dimension": float(group["dimension"].mean()),
            "dimension_min": int(group["dimension"].min()),
            "dimension_max": int(group["dimension"].max()),
            "denominator": denominator,
            "oracle_sse": sse,
            "oracle_fidelity": 1.0 - sse / denominator,
            "historical_frozen_fidelity": np.nan,
            "historical_discrepancy": np.nan,
        }
        if label == "256":
            historical = group["historical_frozen_fidelity"].to_numpy(dtype=np.float64)
            historical_denominator = group["denominator"].to_numpy(dtype=np.float64)
            historical_pooled = float(np.sum(historical * historical_denominator) / denominator)
            row["historical_frozen_fidelity"] = historical_pooled
            row["historical_discrepancy"] = row["oracle_fidelity"] - historical_pooled
        rows.append(row)
    output = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    d256_rows = output.loc[output["budget_label"].eq("256")]
    d256 = dict(zip(
        d256_rows["outer_fold"].astype(str), d256_rows["oracle_fidelity"], strict=True
    ))
    output["increment_vs_D256"] = [
        float(row.oracle_fidelity - d256[str(row.outer_fold)])
        if row.budget_label == "full_span" else np.nan
        for row in output.itertuples(index=False)
    ]
    return output


def make_metric_reconciliation(
    held: pd.DataFrame,
    energy: pd.DataFrame,
    identity: pd.DataFrame,
) -> dict[str, float]:
    full = held.loc[
        held["outer_fold"].astype(str).eq("pooled") & held["budget_label"].eq("full_span")
    ].iloc[0]
    raw_energy = float(energy["raw_response_energy"].sum())
    raw_projected = float(energy["raw_projected_energy"].sum())
    affine_energy = float(energy["affine_centered_energy"].sum())
    affine_projected = float(energy["affine_projected_energy"].sum())
    affine_residual = float(energy["affine_residual_energy"].sum())
    pooled_q_raw = raw_projected / raw_energy
    pooled_q_affine = affine_projected / affine_energy
    reconstructed_f = 1.0 - affine_residual / raw_energy
    discrepancy = float(full["oracle_fidelity"] - pooled_q_raw)
    text = f"""# Full-span PCA metric reconciliation

The frozen oracle fidelity is:

`F_affine = 1 - sum_i ||(I-P)(r_i-mu_train)||^2 / sum_i ||r_i||^2`.

The requested pure zero-origin geometric fraction is:

`q_raw,pooled = sum_i ||P r_i||^2 / sum_i ||r_i||^2`.

They are not algebraically identical because PCA uses the training-column mean `mu_train` as a fixed affine origin, whereas `q_raw` projects the already train-shared-centered response from zero. The corresponding centered geometric fraction is:

`q_affine,pooled = sum_i ||P(r_i-mu_train)||^2 / sum_i ||r_i-mu_train||^2`.

Exact results:

- frozen-metric full-span fidelity: `{float(full['oracle_fidelity']):.15f}`;
- pooled raw projection-energy fraction: `{pooled_q_raw:.15f}`;
- pooled affine-centered projection-energy fraction: `{pooled_q_affine:.15f}`;
- `F_affine - q_raw`: `{discrepancy:.3e}`;
- fidelity reconstructed independently from affine residual energy: `{reconstructed_f:.15f}`;
- maximum per-LCL raw projection-identity relative error: `{identity['relative_identity_error'].max():.3e}`.

Thus any tiny fidelity-versus-`q_raw` difference is fully explained by the fixed training-mean origin, not by a metric or projector inconsistency.
"""
    (OUT / "FULL_SPAN_PCA_METRIC_RECONCILIATION.md").write_text(text, encoding="utf-8")
    return {
        "pooled_q_raw": pooled_q_raw,
        "pooled_q_affine": pooled_q_affine,
        "metric_minus_q_raw": discrepancy,
        "reconstructed_fidelity": reconstructed_f,
    }


def render_figure(
    rank: pd.DataFrame,
    held: pd.DataFrame,
    energy: pd.DataFrame,
    train_reconstruction: pd.DataFrame,
) -> None:
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.65, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.2, 4.25), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], hspace=0.62, wspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    navy, orange, light = "#315a7d", "#c97850", "#b9c8d2"

    dimensions = list(DIMENSIONS)
    for row in rank.itertuples(index=False):
        values = [getattr(row, f"train_variance_D{dimension}") for dimension in dimensions]
        ax_a.plot(dimensions, values, color=light, lw=0.7, alpha=0.65)
        ax_a.scatter([row.numerical_rank], [1.0], color=light, s=9, alpha=0.65)
    means = [rank[f"train_variance_D{dimension}"].mean() for dimension in dimensions]
    ax_a.plot(dimensions, means, color=navy, lw=1.5, marker="o", ms=3.2)
    mean_full_rank = float(rank["numerical_rank"].mean())
    ax_a.scatter([mean_full_rank], [1.0], color=navy, s=20, marker="D", zorder=3)
    ax_a.annotate("full span", (mean_full_rank, 1.0), xytext=(-4, -10), textcoords="offset points", ha="right", fontsize=6.5, color=navy)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(dimensions, [str(value) for value in dimensions])
    ax_a.set_ylim(0, 1.035)
    ax_a.set_xlabel("training PCs")
    ax_a.set_ylabel("cumulative training variance")
    ax_a.set_title("Training-span saturation", loc="center", fontweight="semibold")

    fold_held = held.loc[~held["outer_fold"].astype(str).eq("pooled")]
    for _, group in fold_held.groupby("outer_fold"):
        ordered = group.copy()
        ax_b.plot(ordered["dimension"], ordered["oracle_fidelity"], color="#d6b5a4", lw=0.65, alpha=0.65)
    pooled = held.loc[held["outer_fold"].astype(str).eq("pooled")].copy()
    ax_b.plot(pooled["dimension"], pooled["oracle_fidelity"], color=orange, lw=1.5, marker="o", ms=3.2)
    pooled_full = pooled.loc[pooled["budget_label"].eq("full_span")].iloc[0]
    ax_b.annotate(
        "full span", (pooled_full["dimension"], pooled_full["oracle_fidelity"]),
        xytext=(-5, 5), textcoords="offset points", ha="right", fontsize=6.5, color=orange,
    )
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(dimensions, [str(value) for value in dimensions])
    ax_b.set_xlabel("training-span dimensions")
    ax_b.set_ylabel("held oracle fidelity, $F$")
    ax_b.set_title("Held-span transfer", loc="center", fontweight="semibold")

    rng = np.random.default_rng(SEED)
    q = energy["q_raw"].to_numpy(dtype=float)
    jitter = rng.uniform(-0.14, 0.14, len(q))
    ax_c.scatter(jitter, q, color="#7999b0", alpha=0.28, s=9, linewidth=0)
    ax_c.boxplot(
        [q], positions=[0], widths=0.35, showfliers=False, patch_artist=True,
        medianprops={"color": "#263f55", "linewidth": 1.1},
        boxprops={"facecolor": "white", "edgecolor": "#516b7f", "linewidth": 0.8},
        whiskerprops={"color": "#516b7f", "linewidth": 0.8},
        capprops={"color": "#516b7f", "linewidth": 0.8},
    )
    ax_c.set_xticks([0], ["held LCLs"])
    ax_c.set_ylabel("full-span projection fraction, $q_i$")
    ax_c.set_title("Held response energy in training span", loc="center", fontweight="semibold")

    held_full = held.loc[held["budget_label"].eq("full_span") & ~held["outer_fold"].astype(str).eq("pooled")].sort_values("outer_fold")
    train_values = train_reconstruction.sort_values("outer_fold")["train_fullrank_fidelity"].to_numpy()
    held_values = held_full["oracle_fidelity"].to_numpy()
    offsets = np.linspace(-0.045, 0.045, 5)
    for offset, train_value, held_value in zip(offsets, train_values, held_values, strict=True):
        ax_d.plot([offset, 1 + offset], [train_value, held_value], color="#c9d0d5", lw=0.75, zorder=1)
    ax_d.scatter(offsets, train_values, color=navy, s=20, zorder=2)
    ax_d.scatter(1 + offsets, held_values, color=orange, s=20, zorder=2)
    ax_d.set_xticks([0, 1], ["training", "held-out"])
    ax_d.set_ylabel("full-span oracle fidelity, $F$")
    ax_d.set_ylim(0, 1.04)
    ax_d.set_title("Complete in-sample, weak transfer", loc="center", fontweight="semibold")

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        axis.text(-0.12, 1.10, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(width=0.65, length=3)
    fig.savefig(OUT / "FULL_SPAN_PCA_SANITY_FIGURE.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "FULL_SPAN_PCA_SANITY_FIGURE.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def formal() -> dict[str, Any]:
    validate_authorities()
    OUT.mkdir(parents=True, exist_ok=True)
    values = FRONTIER.load_inputs()
    historical = pd.read_csv(HISTORICAL_CURVE)
    rank_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    held_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    for fold in range(5):
        arrays = FRONTIER.fold_arrays(values, fold)
        svd = direct_svd(arrays["y_train"])
        theoretical_rank = len(arrays["train"]) - 1
        retained_singular = svd.singular_values[: svd.numerical_rank]
        singular_energy = np.square(retained_singular, dtype=np.float64)
        total_singular_energy = float(singular_energy.sum())
        rank_row: dict[str, Any] = {
            "outer_fold": fold,
            "n_train": len(arrays["train"]),
            "n_test": len(arrays["test"]),
            "n_genes": N_GENES,
            "theoretical_max_rank": theoretical_rank,
            "numerical_rank": svd.numerical_rank,
            "rank_tolerance": svd.tolerance,
            "largest_singular_value": float(retained_singular[0]),
            "smallest_retained_singular_value": float(retained_singular[-1]),
            "retained_condition_ratio": float(retained_singular[0] / retained_singular[-1]),
            "max_abs_training_column_mean": float(np.max(np.abs(svd.mean))),
            "max_abs_orthogonality_error": svd.orthogonality_error,
        }
        for dimension in DIMENSIONS:
            rank_row[f"train_variance_D{dimension}"] = float(singular_energy[:dimension].sum() / total_singular_energy)
        rank_row["train_variance_full_rank"] = 1.0
        rank_rows.append(rank_row)

        train_prediction = affine_reconstruction(arrays["y_train"], svd, svd.numerical_rank)
        train_denominator, train_sse, train_fidelity = fidelity(arrays["y_train"], train_prediction)
        max_error = float(np.max(np.abs(arrays["y_train"].astype(np.float64) - train_prediction)))
        relative_error = train_sse / train_denominator
        train_raw = np.asarray(arrays["y_train"], dtype=np.float64)
        train_raw_projection = (train_raw @ svd.retained.T) @ svd.retained
        train_q = np.square(train_raw_projection, dtype=np.float64).sum(axis=1) / np.square(
            train_raw, dtype=np.float64
        ).sum(axis=1)
        train_rows.append({
            "outer_fold": fold,
            "n_train": len(arrays["train"]),
            "numerical_rank": svd.numerical_rank,
            "training_denominator": train_denominator,
            "training_reconstruction_sse": train_sse,
            "relative_frobenius_error": relative_error,
            "maximum_absolute_reconstruction_error": max_error,
            "train_fullrank_fidelity": train_fidelity,
            "train_raw_projection_fraction_mean": float(np.mean(train_q)),
            "train_raw_projection_fraction_median": float(np.median(train_q)),
            "status": "PASS" if (
                relative_error <= RECONSTRUCTION_TOLERANCE
                and train_fidelity >= 1.0 - RECONSTRUCTION_TOLERANCE
                and svd.orthogonality_error <= ORTHOGONALITY_TOLERANCE
            ) else "FAIL",
        })
        if train_rows[-1]["status"] != "PASS":
            raise RuntimeError("GLOBAL_PCA_ORACLE_IMPLEMENTATION_OR_METRIC_INCONSISTENCY:TRAIN")

        for dimension in (*DIMENSIONS, svd.numerical_rank):
            label = str(dimension) if dimension != svd.numerical_rank else "full_span"
            prediction = affine_reconstruction(arrays["y_test"], svd, dimension)
            denominator, sse, oracle_fidelity = fidelity(arrays["y_test"], prediction)
            historical_fidelity = np.nan
            discrepancy = np.nan
            if dimension == 256:
                selected = historical.loc[
                    historical["outer_fold"].astype(str).eq(str(fold))
                    & historical["dimension"].eq(256)
                ]
                if len(selected) != 1:
                    raise RuntimeError("GLOBAL_PCA_HISTORICAL_ROW_MISSING")
                historical_fidelity = float(selected["oracle_fidelity"].iloc[0])
                discrepancy = oracle_fidelity - historical_fidelity
            held_rows.append({
                "outer_fold": fold,
                "budget_label": label,
                "dimension": dimension,
                "dimension_min": dimension,
                "dimension_max": dimension,
                "denominator": denominator,
                "oracle_sse": sse,
                "oracle_fidelity": oracle_fidelity,
                "historical_frozen_fidelity": historical_fidelity,
                "historical_discrepancy": discrepancy,
            })
        fold_energy, fold_identity = geometric_rows(
            arrays["y_test"], svd, fold, arrays["test"], values["lines"]
        )
        energy_rows.extend(fold_energy)
        identity_rows.extend(fold_identity)
        print(f"full-span PCA fold {fold}: COMPLETE", flush=True)

    rank = pd.DataFrame(rank_rows)
    train = pd.DataFrame(train_rows)
    held = pooled_rows(pd.DataFrame(held_rows))
    energy = pd.DataFrame(energy_rows)
    identity = pd.DataFrame(identity_rows)
    if not identity["status"].eq("PASS").all():
        raise RuntimeError("GLOBAL_PCA_ORACLE_IMPLEMENTATION_OR_METRIC_INCONSISTENCY:IDENTITY")
    direct_256 = held.loc[held["budget_label"].eq("256"), "historical_discrepancy"].dropna().abs()
    if direct_256.max() > HISTORICAL_TOLERANCE:
        raise RuntimeError("GLOBAL_PCA_ORACLE_IMPLEMENTATION_OR_METRIC_INCONSISTENCY:HISTORICAL")

    rank.to_csv(OUT / "FULL_SPAN_PCA_RANK_AUDIT.csv", index=False)
    train.to_csv(OUT / "FULL_SPAN_PCA_TRAIN_RECONSTRUCTION.csv", index=False)
    held.to_csv(OUT / "FULL_SPAN_PCA_HELDOUT_FIDELITY.csv", index=False)
    energy.to_csv(OUT / "FULL_SPAN_PCA_HELDOUT_ENERGY_FRACTION.csv", index=False)
    identity.to_csv(OUT / "FULL_SPAN_PCA_PROJECTION_IDENTITY_AUDIT.csv", index=False)
    reconciliation = make_metric_reconciliation(held, energy, identity)
    render_figure(rank, held, energy, train)

    pooled_full = held.loc[
        held["outer_fold"].astype(str).eq("pooled") & held["budget_label"].eq("full_span")
    ].iloc[0]
    pooled_256 = held.loc[
        held["outer_fold"].astype(str).eq("pooled") & held["budget_label"].eq("256")
    ].iloc[0]
    q = energy["q_raw"]
    summary = {
        "status": "COMPLETE",
        "numerical_ranks": rank["numerical_rank"].astype(int).tolist(),
        "train_reconstruction_pass": bool(train["status"].eq("PASS").all()),
        "historical_D256_reproduction_pass": bool(direct_256.max() <= HISTORICAL_TOLERANCE),
        "max_abs_D256_fidelity_discrepancy": float(direct_256.max()),
        "projection_identity_pass": bool(identity["status"].eq("PASS").all()),
        "max_projection_identity_relative_error": float(identity["relative_identity_error"].max()),
        "pooled_D256_fidelity": float(pooled_256["oracle_fidelity"]),
        "pooled_fullspan_fidelity": float(pooled_full["oracle_fidelity"]),
        "increment_D256_to_fullspan": float(pooled_full["oracle_fidelity"] - pooled_256["oracle_fidelity"]),
        "held_q_raw_mean": float(q.mean()),
        "held_q_raw_median": float(q.median()),
        "held_q_raw_q25": float(q.quantile(0.25)),
        "held_q_raw_q75": float(q.quantile(0.75)),
        "held_q_raw_q05": float(q.quantile(0.05)),
        "held_q_raw_q95": float(q.quantile(0.95)),
        **reconciliation,
        "optional_train_LOO": "SKIPPED_NOT_COMPUTATIONALLY_TRIVIAL",
        "predictive_model_trained": False,
        "manuscript_modified": False,
    }
    (OUT / "FULL_SPAN_PCA_SANITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen full-training-span PCA sanity audit")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if not args.formal:
        parser.error("--formal is required")
    print(json.dumps(formal(), indent=2))


if __name__ == "__main__":
    main()
