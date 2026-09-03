"""Diagnostic mapping of the frozen official-DE 127-gene panel to raw counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy.stats import rankdata, spearmanr

from igc_virtual_cell.cgc_tahoe_0i.scaling import _batched_affine, _folds, _support_orders
from igc_virtual_cell.cgc_tahoe_0i.truth import interaction


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _q49(delta: np.ndarray, folds: np.ndarray, orders: np.ndarray) -> float:
    total_base = 0.0
    total_residual = 0.0
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        evaluation = np.flatnonzero(folds == fold)
        q_train = delta[:, :, train] - delta[:, :, train].mean(axis=2, keepdims=True)
        q_eval = delta[:, :, evaluation] - delta[:, :, evaluation].mean(axis=2, keepdims=True)
        grams = np.einsum("rcpg,rdpg->rcd", q_train, q_train, optimize=True)
        for target in range(50):
            support = orders[target, 0, :49]
            base6 = q_eval[0, support].mean(axis=0)
            base14 = q_eval[1, support].mean(axis=0)
            target6, target14 = q_eval[0, target], q_eval[1, target]
            total_base += 2 * float(np.sum((target6 - base6) * (target14 - base14)))
            for direction in range(2):
                gram = grams[direction]
                weight = _batched_affine(
                    gram[np.ix_(support, support)][None], gram[support, target][None]
                )[0]
                prediction6 = np.einsum("c,cpg->pg", weight, q_eval[0, support], optimize=True)
                prediction14 = np.einsum("c,cpg->pg", weight, q_eval[1, support], optimize=True)
                total_residual += float(np.sum((target6 - prediction6) * (target14 - prediction14)))
    return total_residual / total_base


def compare_panel(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "results/cgc_tahoe_0i"
    group = zarr.open_group(out / "response_tensors.zarr", mode="r")
    contexts = list(group.attrs["contexts"])
    interventions = list(group.attrs["interventions"])
    genes = pd.read_csv(out / "gene_metadata_frozen.csv")["gene_symbol"].astype(str).tolist()
    gene_map = {value: index for index, value in enumerate(genes)}
    old_axes = json.loads((root / "data/tahoe100m_plate6_14_core/axes.json").read_text(encoding="utf-8"))
    if old_axes["contexts"] != contexts or old_axes["interventions"] != interventions:
        raise RuntimeError("Frozen 127 panel axes differ from raw reconstruction axes")
    overlap = [gene for gene in old_axes["genes"] if gene in gene_map]
    raw_indices = np.asarray([gene_map[gene] for gene in overlap], dtype=np.int64)
    old_indices = np.asarray([old_axes["genes"].index(gene) for gene in overlap], dtype=np.int64)
    official_all = np.load(root / "data/tahoe100m_plate6_14_core/delta_float32.npy", mmap_mode="r")
    official = np.asarray(official_all[..., old_indices], dtype=np.float64)
    selection = (slice(None), slice(None), slice(None), raw_indices)
    raw_primary = np.asarray(group["delta_primary"].get_orthogonal_selection(selection), dtype=np.float64)
    raw_logfc = np.asarray(group["delta_log2fc"].get_orthogonal_selection(selection), dtype=np.float64)
    rows = []
    for plate in range(2):
        for context in range(50):
            for intervention in range(93):
                x = official[plate, context, intervention]
                y = raw_primary[plate, context, intervention]
                z = raw_logfc[plate, context, intervention]
                rows.append(
                    {
                        "plate": ("plate6", "plate14")[plate],
                        "context_id": contexts[context],
                        "intervention_id": interventions[intervention],
                        "shared_genes": len(overlap),
                        "official_vs_raw_primary_pearson": float(np.corrcoef(x, y)[0, 1]),
                        "official_vs_raw_primary_spearman": float(spearmanr(x, y).statistic),
                        "official_vs_raw_log2fc_pearson": float(np.corrcoef(x, z)[0, 1]),
                        "official_vs_raw_log2fc_spearman": float(spearmanr(x, z).statistic),
                    }
                )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "panel127_raw_comparison.csv", index=False)

    controls = group["primary_dmso_cpm"]
    mean_expression = np.zeros(len(genes), dtype=np.float64)
    effect_energy = np.zeros(len(genes), dtype=np.float64)
    for start in range(0, len(genes), 512):
        stop = min(len(genes), start + 512)
        mean_expression[start:stop] = np.asarray(controls[:, :, start:stop], dtype=np.float64).mean(axis=(0, 1))
        values = np.asarray(group["delta_primary"][:, :, :, start:stop], dtype=np.float64)
        effect_energy[start:stop] = np.sqrt(np.mean(values * values, axis=(0, 1, 2)))
    expression_percentile = rankdata(mean_expression, method="average") / len(genes)
    effect_percentile = rankdata(effect_energy, method="average") / len(genes)
    panel_mask = np.zeros(len(genes), dtype=bool)
    panel_mask[raw_indices] = True
    enrichment = pd.DataFrame(
        {
            "gene_symbol": genes,
            "in_frozen_127_panel": panel_mask,
            "mean_primary_dmso_cpm": mean_expression,
            "dmso_expression_percentile": expression_percentile,
            "raw_primary_response_rms": effect_energy,
            "response_effect_percentile": effect_percentile,
        }
    )
    enrichment.to_csv(out / "panel127_gene_enrichment.csv", index=False)

    folds = _folds(root, interventions)
    orders = _support_orders(root, contexts)
    raw_q49 = _q49(raw_primary, folds, orders)
    official_q49 = float(
        pd.read_csv(root / "results/cgc_tahoe_0c/support_q_curve.csv")
        .set_index("support_size")
        .loc[49, "q_pooled"]
    )
    gamma_official = interaction(official)
    gamma_raw = interaction(raw_primary)
    official_truth = float(np.mean(gamma_official[0] * gamma_official[1]))
    raw_truth = float(np.mean(gamma_raw[0] * gamma_raw[1]))
    global_pearson = float(np.corrcoef(official.ravel(), raw_primary.ravel())[0, 1])
    global_spearman = float(spearmanr(official.ravel(), raw_primary.ravel()).statistic)
    panel_expression = float(np.median(expression_percentile[panel_mask]))
    panel_effect = float(np.median(effect_percentile[panel_mask]))

    # Frozen diagnostic adjudication rule.  It is deliberately conservative:
    # incomplete mapping is inconclusive; a large q49 shift or weak response
    # correspondence indicates non-representativeness; otherwise strong effect
    # enrichment is reported before the generic semantics-resolved label.
    if len(overlap) < 120:
        conclusion = "TAHOE_RAW_COUNT_RECONSTRUCTION_INCONCLUSIVE"
    elif abs(raw_q49 - official_q49) > 0.15 or global_pearson < 0.30:
        conclusion = "TAHOE_127_PANEL_NOT_REPRESENTATIVE"
    elif panel_effect >= 0.75:
        conclusion = "TAHOE_OFFICIAL_DE_PANEL_EFFECT_ENRICHED"
    else:
        conclusion = "TAHOE_RAW_COUNTS_RESOLVE_READOUT_SEMANTICS"
    summary = pd.DataFrame(
        [
            {
                "panel_genes": len(old_axes["genes"]),
                "raw_overlap_genes": len(overlap),
                "global_response_pearson": global_pearson,
                "global_response_spearman": global_spearman,
                "median_condition_pearson": float(comparison["official_vs_raw_primary_pearson"].median()),
                "official_truth_signal": official_truth,
                "raw_panel_truth_signal": raw_truth,
                "official_q49": official_q49,
                "raw_panel_q49": raw_q49,
                "panel_median_dmso_expression_percentile": panel_expression,
                "panel_median_raw_effect_percentile": panel_effect,
                "source_semantics_conclusion": conclusion,
            }
        ]
    )
    summary.to_csv(out / "source_semantics_summary.csv", index=False)
    _write_json(
        out / "panel127_comparison_manifest.json",
        {
            "diagnostic_only": True,
            "gene_selection_changed": False,
            "outcome_based_panel_not_used_for_primary_analysis": True,
            "adjudication_rule": [
                "overlap <120 -> inconclusive",
                "else abs(raw_q49-official_q49)>0.15 or global Pearson<0.30 -> not representative",
                "else median panel raw-effect percentile>=0.75 -> effect enriched",
                "else raw counts resolve semantics",
            ],
            "summary": summary.iloc[0].to_dict(),
        },
    )
    return summary.iloc[0].to_dict()
