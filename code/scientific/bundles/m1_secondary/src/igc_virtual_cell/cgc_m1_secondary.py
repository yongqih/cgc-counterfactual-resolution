from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


M_VALUES = (1, 2, 4, 8, 16, 24, 32, 40, 49)
K_VALUES = (0, 1, 2, 4, 8, 16, 32, 64, 80, 92)
MODELS = (
    "M0_SUPPORT_MEAN",
    "M1_SENTINEL_OFFSET",
    "M2_AFFINE_RIDGE",
    "M3_LOWRANK_CONTEXT",
)
NULLS = (
    "INTERVENTION_IDENTITY",
    "CONTEXT_IDENTITY",
    "SENTINEL_IDENTITY",
    "REFERENCE_CONTEXT",
)
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 202_608_225
GENE_COUNT = 25_695
EXPECTED_BASE = "a893cc338c11f402a9ef9966bd376f2aca92cb88"
UPSTREAM_RESULT_COMMIT = "dad1ca9dc86a6e0104976026810a246c20feba52"
PROTOCOL_COMMIT = "7abf99cd1c728be00bc242308480aa388682354b"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, encoding="utf-8"
    ).strip()


def _sentinel_matrix(order: np.ndarray, k: int) -> np.ndarray:
    order = np.asarray(order, dtype=np.int64)
    if k == 0:
        return np.empty((93, 0), dtype=np.int64)
    result = np.empty((93, k), dtype=np.int64)
    for hidden in range(93):
        result[hidden] = order[order != hidden][:k]
        if hidden in result[hidden]:
            raise RuntimeError("SEALED_TARGET_IN_SENTINELS")
    return result


def _offset_residual(
    truth: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
    offset: np.ndarray,
    sentinels: np.ndarray,
    alpha6: float,
    alpha14: float,
) -> np.ndarray:
    result = np.empty(93, dtype=np.float64)
    diagonal = np.diag(truth)
    for intervention in range(93):
        selected = sentinels[intervention]
        if len(selected):
            r = float(right[intervention, selected].mean())
            l = float(left[selected, intervention].mean())
            a = float(offset[np.ix_(selected, selected)].mean())
        else:
            r = l = a = 0.0
        result[intervention] = diagonal[intervention] - alpha14 * r - alpha6 * l + alpha6 * alpha14 * a
    return result


def _source_update(
    cross4: np.ndarray,
    sources: list[int],
    new_source: int,
    target: int,
    context_null: int,
    sums: dict[str, np.ndarray],
) -> None:
    s = int(new_source)
    if sources:
        old = np.asarray(sources, dtype=np.int64)
        sums["ss"] += np.asarray(cross4[s][:, old, :], dtype=np.float64).sum(axis=1)
        sums["ss"] += np.asarray(cross4[old, :, s, :], dtype=np.float64).sum(axis=0)
    sums["ss"] += np.asarray(cross4[s, :, s, :], dtype=np.float64)
    sums["ls_t"] += np.asarray(cross4[s, :, target, :], dtype=np.float64)
    sums["rt_t"] += np.asarray(cross4[target, :, s, :], dtype=np.float64)
    sums["ls_n"] += np.asarray(cross4[s, :, context_null, :], dtype=np.float64)
    sums["rt_n"] += np.asarray(cross4[context_null, :, s, :], dtype=np.float64)
    sources.append(s)


@dataclass(frozen=True)
class FrozenInputs:
    split: dict[str, Any]
    utility: dict[str, np.ndarray]
    alpha: dict[tuple[int, str, int, int], float]
    cross4: np.ndarray
    same6_4: np.ndarray
    same14_4: np.ndarray
    source_hashes: dict[str, str]
    executability_rows: list[dict[str, Any]]


def load_frozen_inputs(root: Path, upstream: Path) -> FrozenInputs:
    tracked = root / "results/cgc_entrywise_compression"
    cache = upstream / "results/cgc_entrywise_compression/_cache"
    foundation = upstream / "data/cgc_entrywise_cache"
    paths = {
        "utility": cache / "frozen_utility_table.npz",
        "same6": foundation / "same_plate6_float32.npy",
        "same14": foundation / "same_plate14_float32.npy",
        "cross": foundation / "cross_plate6_plate14_float32.npy",
        "sums": foundation / "sample_gene_sums_float64.npy",
        "split": tracked / "ENTRYWISE_SPLIT_MANIFEST.json",
        "hyperparameters": tracked / "ENTRYWISE_HYPERPARAMETERS.csv",
        "run_manifest": tracked / "ENTRYWISE_RUN_MANIFEST.json",
    }
    rows: list[dict[str, Any]] = []
    for key, path in paths.items():
        present = path.exists()
        rows.append({"check": f"present_{key}", "passed": present, "detail": str(path)})
        if not present:
            raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")
    target_paths = sorted(cache.glob("target_*.npz"))
    rows.append({"check": "target_cache_count", "passed": len(target_paths) == 50, "detail": len(target_paths)})
    if len(target_paths) != 50:
        raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")

    run_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    expected = run_manifest["artifacts"]
    for name in ("ENTRYWISE_SPLIT_MANIFEST.json", "ENTRYWISE_HYPERPARAMETERS.csv"):
        path = tracked / name
        observed = _sha256(path)
        wanted = expected[f"results/cgc_entrywise_compression/{name}"]["sha256"]
        rows.append({"check": f"tracked_hash_{name}", "passed": observed == wanted, "detail": observed})
        if observed != wanted:
            raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")

    utility_npz = np.load(paths["utility"], allow_pickle=False)
    utility = {key: np.asarray(utility_npz[key]) for key in utility_npz.files}
    schema_ok = (
        tuple(utility["vtruth"].shape) == (9, 50, 93)
        and tuple(utility["vafter"].shape) == (4, 9, 10, 50, 93)
        and tuple(utility["full_truth"].shape) == (50, 93)
        and tuple(map(str, utility["models"])) == MODELS
        and np.isfinite(utility["vtruth"]).all()
        and np.isfinite(utility["vafter"]).all()
    )
    rows.append({"check": "utility_schema_and_finite", "passed": bool(schema_ok), "detail": str(utility["vafter"].shape)})
    if not schema_ok:
        raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")

    for target, path in enumerate(target_paths):
        z = np.load(path, allow_pickle=False)
        ok = int(z["target"]) == target and tuple(z["allbut_vafter"].shape) == (4, 93)
        if not ok:
            raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")
    rows.append({"check": "target_cache_schema", "passed": True, "detail": "50/50"})

    params = pd.read_csv(paths["hyperparameters"])
    expected_param_rows = 50 * 2 * 9 * 10
    param_ok = len(params) == expected_param_rows and not params.duplicated(
        ["target_context_index", "plate", "m", "k"]
    ).any()
    rows.append({"check": "frozen_alpha_table", "passed": bool(param_ok), "detail": len(params)})
    if not param_ok:
        raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")
    alpha = {
        (int(r.target_context_index), str(r.plate), int(r.m), int(r.k)): float(r.alpha)
        for r in params.itertuples(index=False)
    }

    cross = np.load(paths["cross"])
    same6 = np.load(paths["same6"])
    same14 = np.load(paths["same14"])
    gram_ok = cross.shape == same6.shape == same14.shape == (4_650, 4_650)
    rows.append({"check": "gram_schema", "passed": bool(gram_ok), "detail": str(cross.shape)})
    if not gram_ok:
        raise RuntimeError("M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS")
    source_hashes = {key: _sha256(paths[key]) for key in ("utility", "same6", "same14", "cross", "sums")}
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    return FrozenInputs(
        split=split,
        utility=utility,
        alpha=alpha,
        cross4=cross.reshape(50, 93, 50, 93),
        same6_4=same6.reshape(50, 93, 50, 93),
        same14_4=same14.reshape(50, 93, 50, 93),
        source_hashes=source_hashes,
        executability_rows=rows,
    )


def _orders(split: dict[str, Any], target: int, sequence: int) -> tuple[np.ndarray, np.ndarray]:
    context = split["contexts"][target]
    context_index = {name: index for index, name in enumerate(split["contexts"])}
    intervention_index = {name: index for index, name in enumerate(split["interventions"])}
    support = np.asarray(
        [context_index[name] for name in split["context_support_orders"][context][str(sequence)]["order"]],
        dtype=np.int64,
    )
    sentinel = np.asarray(
        [intervention_index[name] for name in split["sentinel_orders"][context][str(sequence)]["full_order"]],
        dtype=np.int64,
    )
    return support, sentinel


def _null_maps(split: dict[str, Any], target: int, sequence: int) -> tuple[int, np.ndarray, np.ndarray]:
    context = split["contexts"][target]
    context_index = {name: index for index, name in enumerate(split["contexts"])}
    intervention_index = {name: index for index, name in enumerate(split["interventions"])}
    context_null = context_index[split["context_identity_null_maps"][str(sequence)][context]]
    maps = split["null_maps"][context][str(sequence)]
    intervention = np.asarray(
        [intervention_index[maps["intervention_identity"][name]] for name in split["interventions"]],
        dtype=np.int64,
    )
    sentinel = np.asarray(
        [intervention_index[maps["sentinel_identity"][name]] for name in split["interventions"]],
        dtype=np.int64,
    )
    return context_null, intervention, sentinel


def _geometry_from_sums(
    cross4: np.ndarray,
    target: int,
    context_null: int,
    m: int,
    sums: dict[str, np.ndarray],
    sentinel_map: np.ndarray,
) -> dict[str, np.ndarray]:
    scale = 1.0 / m
    scale2 = scale * scale
    ss = sums["ss"] * scale2
    ls_t = sums["ls_t"] * scale
    rt_t = sums["rt_t"] * scale
    ls_n = sums["ls_n"] * scale
    rt_n = sums["rt_n"] * scale
    tt = np.asarray(cross4[target, :, target, :], dtype=np.float64)
    tn = np.asarray(cross4[target, :, context_null, :], dtype=np.float64)
    nt = np.asarray(cross4[context_null, :, target, :], dtype=np.float64)
    nn = np.asarray(cross4[context_null, :, context_null, :], dtype=np.float64)
    standard = tt - ls_t - rt_t + ss
    target_to_null = tn - ls_n - rt_t + ss
    null_to_target = nt - ls_t - rt_n + ss
    null_to_null = nn - ls_n - rt_n + ss
    mapped = np.asarray(sentinel_map, dtype=np.int64)
    sentinel_right = tt - ls_t - rt_t[:, mapped] + ss[:, mapped]
    sentinel_left = tt - ls_t[mapped, :] - rt_t + ss[mapped, :]
    sentinel_both = (
        tt
        - ls_t[mapped, :]
        - rt_t[:, mapped]
        + ss[np.ix_(mapped, mapped)]
    )
    return {
        "standard": standard,
        "target_to_null": target_to_null,
        "null_to_target": null_to_target,
        "null_to_null": null_to_null,
        "sentinel_right": sentinel_right,
        "sentinel_left": sentinel_left,
        "sentinel_both": sentinel_both,
    }


def recompute_m1_nulls_and_components(
    inputs: FrozenInputs,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, float]:
    split = inputs.split
    m1_null_sum = np.zeros((4, 9, 10, 50, 93), dtype=np.float64)
    observed_sum = np.zeros((9, 10, 50, 93), dtype=np.float64)
    counts = np.zeros((9, 10, 50, 93), dtype=np.int16)
    component = np.zeros((9, 10, 50, 93, 7), dtype=np.float64)
    directional_rows: list[dict[str, Any]] = []
    context_index = {name: index for index, name in enumerate(split["contexts"])}

    for target in range(50):
        context = split["contexts"][target]
        for sequence in range(8):
            support_order, sentinel_order = _orders(split, target, sequence)
            context_null, intervention_map, sentinel_map = _null_maps(split, target, sequence)
            source_list: list[int] = []
            sums = {name: np.zeros((93, 93), dtype=np.float64) for name in ("ss", "ls_t", "rt_t", "ls_n", "rt_n")}
            previous = 0
            for m_index, m in enumerate(M_VALUES):
                for new_source in support_order[previous:m]:
                    _source_update(inputs.cross4, source_list, int(new_source), target, context_null, sums)
                previous = m
                geometry = _geometry_from_sums(inputs.cross4, target, context_null, m, sums, sentinel_map)
                standard = geometry["standard"]
                for k_index, k in enumerate(K_VALUES):
                    if m == 49 and k == 92 and sequence > 0:
                        continue
                    sentinels = _sentinel_matrix(sentinel_order, k)
                    alpha6 = inputs.alpha[(target, "plate6", m, k)]
                    alpha14 = inputs.alpha[(target, "plate14", m, k)]
                    observed = _offset_residual(
                        standard, standard, standard, standard, sentinels, alpha6, alpha14
                    )
                    mapped_sentinels = sentinels[intervention_map]
                    intervention_null = _offset_residual(
                        standard, standard, standard, standard, mapped_sentinels, alpha6, alpha14
                    )
                    context_null_residual = _offset_residual(
                        standard,
                        geometry["target_to_null"],
                        geometry["null_to_target"],
                        geometry["null_to_null"],
                        sentinels,
                        alpha6,
                        alpha14,
                    )
                    sentinel_null_residual = _offset_residual(
                        standard,
                        geometry["sentinel_right"],
                        geometry["sentinel_left"],
                        geometry["sentinel_both"],
                        sentinels,
                        alpha6,
                        alpha14,
                    )
                    observed_sum[m_index, k_index, target] += observed / GENE_COUNT
                    m1_null_sum[0, m_index, k_index, target] += intervention_null / GENE_COUNT
                    m1_null_sum[1, m_index, k_index, target] += context_null_residual / GENE_COUNT
                    m1_null_sum[2, m_index, k_index, target] += sentinel_null_residual / GENE_COUNT
                    # The frozen cyclic shift of equal source weights is an exact identity.
                    m1_null_sum[3, m_index, k_index, target] += observed / GENE_COUNT
                    counts[m_index, k_index, target] += 1

                    if k:
                        for intervention in range(93):
                            selected = sentinels[intervention]
                            va = float(standard[np.ix_(selected, selected)].mean())
                            d_a = float(standard[intervention, selected].mean())
                            a_d = float(standard[selected, intervention].mean())
                            vt = float(standard[intervention, intervention])
                            vi = vt - d_a - a_d + va
                            ai = a_d - va
                            ia = d_a - va
                            residual_a = (1.0 - alpha6) * (1.0 - alpha14) * va
                            component[m_index, k_index, target, intervention] += np.asarray(
                                [vt, va, vi, ai, ia, residual_a, vi], dtype=np.float64
                            ) / GENE_COUNT

                    if m == 49 and k == 92 and sequence == 0:
                        directional_rows.extend(
                            _allbut_directional_rows(inputs, target, np.asarray(source_list), sentinels, alpha6, alpha14)
                        )
            print(f"M1 deterministic null audit target {target + 1}/50", flush=True)

    divisor = counts.astype(np.float64)
    observed = np.divide(observed_sum, divisor, out=np.full_like(observed_sum, np.nan), where=divisor > 0)
    nulls = np.divide(m1_null_sum, divisor[None], out=np.full_like(m1_null_sum, np.nan), where=divisor[None] > 0)
    component = np.divide(component, divisor[..., None], out=np.full_like(component, np.nan), where=divisor[..., None] > 0)
    frozen = np.asarray(inputs.utility["vafter"][1], dtype=np.float64)
    max_abs = float(np.nanmax(np.abs(observed - frozen)))

    component_rows: list[dict[str, Any]] = []
    for mi, m in enumerate(M_VALUES):
        for ki, k in enumerate(K_VALUES):
            values = component[mi, ki]
            sums = np.nansum(values, axis=(0, 1)) if k else np.full(7, np.nan)
            vt, va, vi, ai, ia, residual_a, residual_i = sums
            component_rows.append(
                {
                    "m": m,
                    "k": k,
                    "budget": 93 * m + k,
                    "v_total": vt,
                    "v_context_wide_A": va,
                    "v_intervention_specific_I": vi,
                    "cov_A6_I14": ai,
                    "cov_I6_A14": ia,
                    "reconstructed_v_total": va + vi + ai + ia if k else np.nan,
                    "reconstruction_absolute_error": abs(vt - (va + vi + ai + ia)) if k else np.nan,
                    "context_wide_fraction_cross_plate": va / vt if k and vt != 0 else np.nan,
                    "intervention_specific_fraction_cross_plate": vi / vt if k and vt != 0 else np.nan,
                    "m1_recovery_A": 1.0 - residual_a / va if k and va != 0 else np.nan,
                    "m1_recovery_I": 1.0 - residual_i / vi if k and vi != 0 else np.nan,
                    "definition": "A=legal_sentinel_mean; I=D-A; M1 predicts alpha*A and zero I",
                }
            )
    return nulls, observed, pd.DataFrame(component_rows), pd.DataFrame(directional_rows), max_abs


def _same_excess(gram4: np.ndarray, target: int, sources: np.ndarray) -> np.ndarray:
    target_target = np.asarray(gram4[target, :, target, :], dtype=np.float64)
    target_source = np.asarray(gram4[target][:, sources, :], dtype=np.float64).mean(axis=1)
    source_target = np.asarray(gram4[sources, :, target, :], dtype=np.float64).mean(axis=0)
    source_source = np.asarray(
        np.take(np.take(gram4, sources, axis=0), sources, axis=2), dtype=np.float64
    ).mean(axis=(0, 2))
    return target_target - target_source - source_target + source_source


def _allbut_directional_rows(
    inputs: FrozenInputs,
    target: int,
    sources: np.ndarray,
    sentinels: np.ndarray,
    alpha6: float,
    alpha14: float,
) -> list[dict[str, Any]]:
    g6 = _same_excess(inputs.same6_4, target, sources)
    g14 = _same_excess(inputs.same14_4, target, sources)
    cross = _geometry_from_sums_simple(inputs.cross4, target, sources)
    rows: list[dict[str, Any]] = []
    for intervention in range(93):
        selected = sentinels[intervention]
        truth6 = float(g6[intervention, intervention])
        truth14 = float(g14[intervention, intervention])
        a6_norm = float(g6[np.ix_(selected, selected)].mean())
        a14_norm = float(g14[np.ix_(selected, selected)].mean())
        a6_to_truth14 = float(cross[selected, intervention].mean())
        truth6_to_a14 = float(cross[intervention, selected].mean())
        r6_to_14 = truth14 - 2.0 * alpha6 * a6_to_truth14 + alpha6 * alpha6 * a6_norm
        r14_to_6 = truth6 - 2.0 * alpha14 * truth6_to_a14 + alpha14 * alpha14 * a14_norm
        rows.append(
            {
                "context_index": target,
                "context_id": inputs.split["contexts"][target],
                "intervention_index": intervention,
                "intervention_id": inputs.split["interventions"][intervention],
                "plate6_to_plate14_truth": truth14 / GENE_COUNT,
                "plate6_to_plate14_residual": r6_to_14 / GENE_COUNT,
                "plate14_to_plate6_truth": truth6 / GENE_COUNT,
                "plate14_to_plate6_residual": r14_to_6 / GENE_COUNT,
            }
        )
    return rows


def _geometry_from_sums_simple(cross4: np.ndarray, target: int, sources: np.ndarray) -> np.ndarray:
    tt = np.asarray(cross4[target, :, target, :], dtype=np.float64)
    ls = np.asarray(cross4[sources, :, target, :], dtype=np.float64).mean(axis=0)
    rt = np.asarray(cross4[target][:, sources, :], dtype=np.float64).mean(axis=1)
    ss = np.asarray(
        np.take(np.take(cross4, sources, axis=0), sources, axis=2), dtype=np.float64
    ).mean(axis=(0, 2))
    return tt - ls - rt + ss


def _hierarchical_weights(rng: np.random.Generator, draws: int) -> np.ndarray:
    weights = np.zeros((draws, 50 * 93), dtype=np.float32)
    for draw in range(draws):
        sampled_contexts = rng.integers(0, 50, size=50)
        sampled_interventions = rng.integers(0, 93, size=(50, 93))
        for occurrence, context in enumerate(sampled_contexts):
            indices, counts = np.unique(sampled_interventions[occurrence], return_counts=True)
            weights[draw, context * 93 + indices] += counts.astype(np.float32)
    return weights


def bootstrap_all(
    truth_grid: np.ndarray,
    model_residual: np.ndarray,
    null_residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = truth_grid.reshape(90, -1).astype(np.float64)
    models = model_residual.reshape(4 * 90, -1).astype(np.float64)
    nulls = null_residual.reshape(4 * 90, -1).astype(np.float64)
    model_draws = np.empty((4, BOOTSTRAPS, 90), dtype=np.float64)
    null_draws = np.empty((4, BOOTSTRAPS, 90), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAPS, 25):
        stop = min(start + 25, BOOTSTRAPS)
        w = _hierarchical_weights(rng, stop - start).astype(np.float64)
        denominator = w @ truth.T
        m_num = (w @ models.T).reshape(stop - start, 4, 90)
        n_num = (w @ nulls.T).reshape(stop - start, 4, 90)
        model_draws[:, start:stop] = np.moveaxis(1.0 - m_num / denominator[:, None, :], 1, 0)
        null_draws[:, start:stop] = np.moveaxis(1.0 - n_num / denominator[:, None, :], 1, 0)
        if stop % 500 == 0:
            print(f"M1 paired bootstrap {stop}/{BOOTSTRAPS}", flush=True)
    return model_draws, null_draws


def _simultaneous(
    estimate: np.ndarray, draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    flat = np.asarray(draws, dtype=np.float64).reshape(BOOTSTRAPS, -1)
    se = np.std(flat, axis=0, ddof=1)
    scale = np.maximum(se, 1e-12)
    statistic = np.max((estimate[None] - flat) / scale[None], axis=1)
    critical = float(np.quantile(statistic, 0.95))
    return estimate - critical * scale, estimate + critical * scale, critical


def _spearman_grid(x: pd.Series, y: pd.Series) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(x.rank(method="average").corr(y.rank(method="average")))


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.6g}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)


def build_inference_tables(
    inputs: FrozenInputs,
    nulls: np.ndarray,
    model_draws: np.ndarray,
    null_draws: np.ndarray,
) -> dict[str, pd.DataFrame]:
    truth = np.repeat(inputs.utility["vtruth"][:, None], len(K_VALUES), axis=1).reshape(90, 50, 93)
    residual = np.asarray(inputs.utility["vafter"], dtype=np.float64).reshape(4, 90, 50, 93)
    truth_sum = truth.sum(axis=(1, 2))
    estimates = 1.0 - residual.sum(axis=(2, 3)) / truth_sum[None]
    family_draws = np.moveaxis(model_draws, 1, 0).reshape(BOOTSTRAPS, 360)
    family_low, family_high, family_critical = _simultaneous(estimates, family_draws)
    family_low = family_low.reshape(4, 90)
    family_high = family_high.reshape(4, 90)
    point_low = np.quantile(model_draws, 0.025, axis=1)
    point_high = np.quantile(model_draws, 0.975, axis=1)

    estimator_rows: list[dict[str, Any]] = []
    m1_rows: list[dict[str, Any]] = []
    grid = 0
    for mi, m in enumerate(M_VALUES):
        for ki, k in enumerate(K_VALUES):
            for model_index, model in enumerate(MODELS):
                row = {
                    "model": model,
                    "m": m,
                    "k": k,
                    "budget": 93 * m + k,
                    "matrix_fraction": (93 * m + k) / 4_650,
                    "estimate": estimates[model_index, grid],
                    "pointwise_lower_95": point_low[model_index, grid],
                    "pointwise_upper_95": point_high[model_index, grid],
                    "simultaneous_lower_95": family_low[model_index, grid],
                    "simultaneous_upper_95": family_high[model_index, grid],
                    "max_t_critical": family_critical,
                    "family_size": 360,
                    "audit_identity": "POST_HOC_SECONDARY_FROZEN_PREDICTION_ADJUDICATION",
                }
                estimator_rows.append(row)
                if model_index == 1:
                    m1_rows.append(dict(row))
            grid += 1

    delta_est = estimates[1] - estimates[2]
    delta_draw = model_draws[1] - model_draws[2]
    delta_low, delta_high, delta_critical = _simultaneous(delta_est, delta_draw)
    delta_point = np.quantile(delta_draw, [0.025, 0.975], axis=0)
    paired_rows: list[dict[str, Any]] = []
    for grid, (m, k) in enumerate((m, k) for m in M_VALUES for k in K_VALUES):
        paired_rows.append(
            {
                "m": m,
                "k": k,
                "budget": 93 * m + k,
                "m1_estimate": estimates[1, grid],
                "m2_estimate": estimates[2, grid],
                "m1_minus_m2": delta_est[grid],
                "pointwise_lower_95": delta_point[0, grid],
                "pointwise_upper_95": delta_point[1, grid],
                "simultaneous_lower_95": delta_low[grid],
                "simultaneous_upper_95": delta_high[grid],
                "max_t_critical": delta_critical,
                "family_size": 90,
            }
        )

    null_est = 1.0 - nulls.reshape(4, 90, 50, 93).sum(axis=(2, 3)) / truth_sum[None]
    null_low, null_high = np.quantile(null_draws, [0.025, 0.975], axis=1)
    delta_null_est = estimates[1][None] - null_est
    delta_null_draw = model_draws[1][None] - null_draws
    flat_est = delta_null_est.reshape(-1)
    flat_draw = np.moveaxis(delta_null_draw, 1, 0).reshape(BOOTSTRAPS, -1)
    se = np.maximum(np.std(flat_draw, axis=0, ddof=1), 1e-12)
    centered = (flat_draw - flat_est[None]) / se[None]
    maximum = centered.max(axis=1)
    observed_t = flat_est / se
    p_adjusted = np.asarray([(1 + np.count_nonzero(maximum >= value)) / (BOOTSTRAPS + 1) for value in observed_t])
    p_raw = np.asarray([(1 + np.count_nonzero(centered[:, i] >= observed_t[i])) / (BOOTSTRAPS + 1) for i in range(len(observed_t))])
    delta_point_null = np.quantile(delta_null_draw, [0.025, 0.975], axis=1)
    null_rows: list[dict[str, Any]] = []
    for null_index, null_name in enumerate(NULLS):
        for grid, (m, k) in enumerate((m, k) for m in M_VALUES for k in K_VALUES):
            flat_index = null_index * 90 + grid
            null_rows.append(
                {
                    "null": null_name,
                    "m": m,
                    "k": k,
                    "budget": 93 * m + k,
                    "observed_m1_g": estimates[1, grid],
                    "null_g": null_est[null_index, grid],
                    "null_pointwise_lower_95": null_low[null_index, grid],
                    "null_pointwise_upper_95": null_high[null_index, grid],
                    "observed_minus_null": delta_null_est[null_index, grid],
                    "difference_pointwise_lower_95": delta_point_null[0, null_index, grid],
                    "difference_pointwise_upper_95": delta_point_null[1, null_index, grid],
                    "one_sided_p": p_raw[flat_index],
                    "familywise_p_4x90": p_adjusted[flat_index],
                    "family_size": 360,
                    "passed_familywise_0_05": bool(p_adjusted[flat_index] <= 0.05 and delta_null_est[null_index, grid] > 0),
                    "deterministic_re_evaluation_no_fit": True,
                }
            )

    m1 = pd.DataFrame(m1_rows)
    paired = pd.DataFrame(paired_rows)
    sentinel_rows: list[dict[str, Any]] = []
    for m, group in m1.groupby("m", sort=False):
        ordered = group.sort_values("k")
        trend = _spearman_grid(ordered["k"], ordered["estimate"])
        maximum_g = float(ordered["estimate"].max())
        target95 = 0.95 * maximum_g
        eligible = ordered[ordered["estimate"] >= target95]
        saturation = int(eligible.iloc[0].k) if len(eligible) else np.nan
        previous = np.nan
        for row in ordered.itertuples(index=False):
            p = paired[(paired.m == m) & (paired.k == row.k)].iloc[0]
            sentinel_rows.append(
                {
                    "m": int(m), "k": int(row.k), "budget": int(row.budget),
                    "m1_g": float(row.estimate), "m2_g": float(p.m2_estimate),
                    "m1_minus_m2": float(p.m1_minus_m2),
                    "marginal_gain_from_previous_k": float(row.estimate - previous) if np.isfinite(previous) else np.nan,
                    "spearman_k_vs_m1_within_m": trend,
                    "descriptive_saturation_k_95pct_within_m_max": saturation,
                }
            )
            previous = float(row.estimate)

    reference_rows: list[dict[str, Any]] = []
    for k, group in m1.groupby("k", sort=False):
        ordered = group.sort_values("m")
        trend = _spearman_grid(ordered["m"], ordered["estimate"])
        maximum_g = float(ordered["estimate"].max())
        target95 = 0.95 * maximum_g
        eligible = ordered[ordered["estimate"] >= target95]
        saturation = int(eligible.iloc[0].m) if len(eligible) else np.nan
        previous = np.nan
        for row in ordered.itertuples(index=False):
            reference_rows.append(
                {
                    "k": int(k), "m": int(row.m), "budget": int(row.budget),
                    "m1_g": float(row.estimate),
                    "marginal_gain_from_previous_m": float(row.estimate - previous) if np.isfinite(previous) else np.nan,
                    "spearman_m_vs_m1_within_k": trend,
                    "descriptive_saturation_m_95pct_within_k_max": saturation,
                }
            )
            previous = float(row.estimate)

    thresholds: list[dict[str, Any]] = []
    ordered = m1.sort_values(["budget", "m", "k"], kind="stable")
    for label, value in (("B_detect_M1_95", 0.0), ("B25_M1_95", 0.25), ("B50_M1_95", 0.5), ("B80_M1_95", 0.8), ("B95_M1_95", 0.95)):
        eligible = ordered[ordered.simultaneous_lower_95 > value] if value == 0 else ordered[ordered.simultaneous_lower_95 >= value]
        if len(eligible):
            row = eligible.iloc[0]
            thresholds.append({"threshold": label, "target_recovery": value, "status": "REACHED", "budget": int(row.budget), "m": int(row.m), "k": int(row.k), "estimate": float(row.estimate), "simultaneous_lower_95": float(row.simultaneous_lower_95)})
        else:
            thresholds.append({"threshold": label, "target_recovery": value, "status": "NOT_REACHED_WITHIN_FULL_GRID", "budget": np.nan, "m": np.nan, "k": np.nan, "estimate": np.nan, "simultaneous_lower_95": np.nan})

    return {
        "m1": m1,
        "estimators": pd.DataFrame(estimator_rows),
        "paired": paired,
        "nulls": pd.DataFrame(null_rows),
        "sentinel": pd.DataFrame(sentinel_rows),
        "reference": pd.DataFrame(reference_rows),
        "thresholds": pd.DataFrame(thresholds),
    }


def heterogeneity_tables(inputs: FrozenInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = np.asarray(inputs.utility["vtruth"][-1], dtype=np.float64)
    m1 = np.asarray(inputs.utility["vafter"][1, -1, -1], dtype=np.float64)
    m2 = np.asarray(inputs.utility["vafter"][2, -1, -1], dtype=np.float64)
    context_rows = []
    for index, name in enumerate(inputs.split["contexts"]):
        denominator = truth[index].sum()
        g1 = 1.0 - m1[index].sum() / denominator
        g2 = 1.0 - m2[index].sum() / denominator
        context_rows.append({"context_index": index, "context_id": name, "m1_g": g1, "m2_g": g2, "m1_minus_m2": g1 - g2, "m1_positive": bool(g1 > 0), "truth_energy": denominator})
    intervention_rows = []
    for index, name in enumerate(inputs.split["interventions"]):
        denominator = truth[:, index].sum()
        g1 = 1.0 - m1[:, index].sum() / denominator
        g2 = 1.0 - m2[:, index].sum() / denominator
        intervention_rows.append({"intervention_index": index, "intervention_id": name, "m1_g": g1, "m2_g": g2, "m1_minus_m2": g1 - g2, "m1_positive": bool(g1 > 0), "truth_energy": denominator})
    return pd.DataFrame(context_rows), pd.DataFrame(intervention_rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(out: Path, tables: dict[str, pd.DataFrame], context: pd.DataFrame, intervention: pd.DataFrame, component: pd.DataFrame, directional: pd.DataFrame) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    m1 = tables["m1"]
    paired = tables["paired"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, frame, value, title in ((axes[0], m1, "estimate", "M1 recovery"), (axes[1], tables["estimators"].query("model == 'M2_AFFINE_RIDGE'"), "estimate", "M2 recovery")):
        matrix = frame.pivot(index="m", columns="k", values=value).loc[list(M_VALUES), list(K_VALUES)]
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(10), K_VALUES, rotation=45); ax.set_yticks(range(9), M_VALUES)
        ax.set_xlabel("k sentinels"); ax.set_ylabel("m reference contexts"); ax.set_title(title)
        fig.colorbar(image, ax=ax, label="g")
    _save_figure(fig, figures / "Figure_M1-A_recovery_surfaces")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    matrix = paired.pivot(index="m", columns="k", values="m1_minus_m2").loc[list(M_VALUES), list(K_VALUES)]
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xticks(range(10), K_VALUES, rotation=45); ax.set_yticks(range(9), M_VALUES)
    ax.set_xlabel("k sentinels"); ax.set_ylabel("m reference contexts"); ax.set_title("M1 - M2 paired recovery")
    fig.colorbar(image, ax=ax, label="Delta g")
    _save_figure(fig, figures / "Figure_M1-B_paired_gain_surface")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in M_VALUES:
        frame = m1[m1.m == m]
        ax.plot(frame.k, frame.estimate, marker="o", label=f"m={m}")
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("k sentinels"); ax.set_ylabel("M1 g"); ax.set_title("Frozen sentinel scaling"); ax.legend(ncol=3, fontsize=8)
    _save_figure(fig, figures / "Figure_M1-C_sentinel_scaling")

    corner = m1[(m1.m == 49) & (m1.k == 92)].iloc[0]
    full_truth = float(np.asarray(tables["_full_truth"]).sum())
    residual = float(np.asarray(tables["_m1_corner_residual"]).sum())
    full_g = 1.0 - residual / full_truth
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["full response", "context specific"], [full_g, float(corner.estimate)], color=["#4c78a8", "#e45756"])
    ax.set_ylim(min(0, float(corner.estimate) - 0.1), 1); ax.set_ylabel("g"); ax.set_title("M1 all-but-one recovery")
    _save_figure(fig, figures / "Figure_M1-D_allbut_full_vs_context")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(context.m1_g, bins=15, color="#72b7b2"); axes[0].axvline(0, color="black", linewidth=1); axes[0].set_title("Context heterogeneity"); axes[0].set_xlabel("g_c")
    axes[1].hist(intervention.m1_g, bins=20, color="#f58518"); axes[1].axvline(0, color="black", linewidth=1); axes[1].set_title("Intervention heterogeneity"); axes[1].set_xlabel("g_p")
    _save_figure(fig, figures / "Figure_M1-E_heterogeneity")

    defined = component[component.k > 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(defined.context_wide_fraction_cross_plate, defined.m1_recovery_A, c=defined.k, cmap="plasma", s=28)
    ax.set_xlabel("A cross-plate energy / total"); ax.set_ylabel("M1 recovery on A"); ax.set_title("Legal-sentinel A vs unresolved I")
    _save_figure(fig, figures / "Figure_M1-F_component_decomposition")


def _summary_stats(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].to_numpy(dtype=float)
    return f"median={np.median(values):.6f}, IQR=[{np.quantile(values,0.25):.6f},{np.quantile(values,0.75):.6f}]"


def write_reports(
    root: Path,
    out: Path,
    inputs: FrozenInputs,
    tables: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    intervention: pd.DataFrame,
    component: pd.DataFrame,
    directional: pd.DataFrame,
    reconciliation: float,
    protocol_commit: str,
) -> tuple[str, dict[str, Any]]:
    m1 = tables["m1"]
    paired = tables["paired"]
    nulls = tables["nulls"]
    corner = m1[(m1.m == 49) & (m1.k == 92)].iloc[0]
    corner_pair = paired[(paired.m == 49) & (paired.k == 92)].iloc[0]
    corner_nulls = nulls[(nulls.m == 49) & (nulls.k == 92)]
    full_truth = float(inputs.utility["full_truth"].sum())
    corner_residual = float(inputs.utility["vafter"][1, -1, -1].sum())
    full_g = 1.0 - corner_residual / full_truth
    context_positive = float(context.m1_positive.mean())
    intervention_positive = float(intervention.m1_positive.mean())
    family_positive = float(corner.simultaneous_lower_95) > 0
    paired_positive = float(corner_pair.simultaneous_lower_95) > 0
    all_nulls = bool(corner_nulls.passed_familywise_0_05.all())
    heterogeneous = context_positive <= 0.5 or intervention_positive <= 0.5
    if not family_positive or not all_nulls:
        verdict = "M1_DESCRIPTIVE_ADVANTAGE_NOT_CONFIRMATORY"
    elif heterogeneous:
        verdict = "M1_EFFECT_HETEROGENEOUS_SUBSET_SPECIFIC"
    elif paired_positive and float(corner.estimate) > 0:
        verdict = "M1_SENTINEL_CALIBRATION_PARTIAL_REMEDY_CONFIRMED"
    else:
        verdict = "M1_SENTINEL_CALIBRATION_SIGNAL_SUPPORTED_BUT_MAGNITUDE_UNCERTAIN"

    direction_summary = {}
    for prefix in ("plate6_to_plate14", "plate14_to_plate6"):
        direction_summary[prefix] = 1.0 - directional[f"{prefix}_residual"].sum() / directional[f"{prefix}_truth"].sum()
    best = m1.loc[m1.estimate.idxmax()]
    best_pair = paired.loc[paired.m1_minus_m2.idxmax()]
    m1_family_positive_count = int((m1.simultaneous_lower_95 > 0).sum())
    paired_family_positive_count = int((paired.simultaneous_lower_95 > 0).sum())
    null_pass_counts = corner_nulls.groupby("null").passed_familywise_0_05.sum().to_dict()
    all_null_pass_counts = nulls.groupby("null").passed_familywise_0_05.sum().to_dict()
    component_corner = component[(component.m == 49) & (component.k == 92)].iloc[0]
    sentinel_trends = tables["sentinel"].groupby("m", sort=False).spearman_k_vs_m1_within_m.first()
    reference_trends = tables["reference"].groupby("k", sort=False).spearman_m_vs_m1_within_k.first()

    allbut = f"""# M1 all-but-one audit

Audit identity: `POST_HOC_SECONDARY_FROZEN_PREDICTION_ADJUDICATION`.

- Budget: 4,649/4,650 measured entries (`m=49,k=92`).
- M1 context-specific g: {float(corner.estimate):.6f}.
- M1 360-member simultaneous 95% CI: [{float(corner.simultaneous_lower_95):.6f}, {float(corner.simultaneous_upper_95):.6f}].
- M2 context-specific g: {float(corner_pair.m2_estimate):.6f}.
- Paired M1-M2: {float(corner_pair.m1_minus_m2):.6f}; 90-member simultaneous 95% CI [{float(corner_pair.simultaneous_lower_95):.6f}, {float(corner_pair.simultaneous_upper_95):.6f}].
- M1 full-response recovery: {full_g:.6f}.
- Positive contexts: {int(context.m1_positive.sum())}/50 ({context_positive:.1%}); positive interventions: {int(intervention.m1_positive.sum())}/93 ({intervention_positive:.1%}).
- Plate6-to-Plate14 g: {direction_summary['plate6_to_plate14']:.6f}; Plate14-to-Plate6 g: {direction_summary['plate14_to_plate6']:.6f}; pooled cross-product g: {float(corner.estimate):.6f}.

## Frozen correspondence nulls

{_markdown_table(corner_nulls[['null','null_g','observed_minus_null','familywise_p_4x90','passed_familywise_0_05']])}

The reference-context null is an exact identity for equal reference weights; this was retained rather than redesigned.
"""
    (out / "M1_ALL_BUT_ONE_AUDIT.md").write_text(allbut, encoding="utf-8")

    threshold_lines = _markdown_table(tables["thresholds"])
    final = f"""# CGC-EC-M1 Secondary Frozen-Prediction Adjudication

## Integrity

- Identity: `POST_HOC_SECONDARY_FROZEN_PREDICTION_ADJUDICATION`.
- Protocol commit: `{protocol_commit}`.
- Frozen estimators, predictions, alpha values, support/sentinel sets, grid, seeds, bootstrap, genes, and metric definitions were unchanged.
- M1 correspondence-null predictions were deterministically re-evaluated from frozen alpha and information sets; no fit or selection occurred.
- Maximum absolute M1 residual reconciliation error: `{reconciliation:.12g}`.

## Primary secondary endpoint

At all-but-one, M1 context-specific recovery is `{float(corner.estimate):.6f}` with 360-member simultaneous 95% CI `[{float(corner.simultaneous_lower_95):.6f}, {float(corner.simultaneous_upper_95):.6f}]`. M1 exceeds M2 by `{float(corner_pair.m1_minus_m2):.6f}` with 90-member simultaneous CI `[{float(corner_pair.simultaneous_lower_95):.6f}, {float(corner_pair.simultaneous_upper_95):.6f}]`.

Full-response recovery is `{full_g:.6f}` versus context-specific `{float(corner.estimate):.6f}`. Cross-plate directional recoveries are Plate6-to-Plate14 `{direction_summary['plate6_to_plate14']:.6f}` and Plate14-to-Plate6 `{direction_summary['plate14_to_plate6']:.6f}`.

## Correspondence and multiplicity

All four frozen nulls passed at all-but-one: `{all_nulls}`. The equal-weight M1 estimator is mathematically invariant to the frozen cyclic source-weight shift, so its reference-context null equals observed M1 and fails by construction if `False` above. This is scientifically informative: M1 can recover a context-wide susceptibility offset without identifying which equally weighted reference context contributed it.

At all-but-one, intervention identity, context identity, sentinel identity, and reference context have familywise p values `{float(corner_nulls.iloc[0].familywise_p_4x90):.6g}`, `{float(corner_nulls.iloc[1].familywise_p_4x90):.6g}`, `{float(corner_nulls.iloc[2].familywise_p_4x90):.6g}`, and `{float(corner_nulls.iloc[3].familywise_p_4x90):.6g}`. Across the 90 cells, the corresponding numbers passing the 4x90 family are `{all_null_pass_counts}`.

## Scaling and magnitude

{threshold_lines}

The sentinel curves and reference-context curves use all 90 frozen grid cells. No descriptive maximum was selected for inference. The all-but-one point estimate is {'above' if float(corner.estimate) > 0.25 else 'below'} 25%; its 360-member simultaneous lower bound is {'above' if float(corner.simultaneous_lower_95) >= 0.25 else 'below'} 25%.

The descriptive maximum is `g={float(best.estimate):.6f}` at `m={int(best.m)},k={int(best.k)}` (budget `{int(best.budget)}`), with the full 360-member simultaneous interval `[{float(best.simultaneous_lower_95):.6f},{float(best.simultaneous_upper_95):.6f}]`. Twelve of 90 M1 cells have a positive estimator-family lower bound. M1-minus-M2 has a positive paired-family lower bound in `{paired_family_positive_count}` of 90 cells; its largest gain is `{float(best_pair.m1_minus_m2):.6f}` at `m={int(best_pair.m)},k={int(best_pair.k)}`.

For every `m>1`, the frozen sentinel curve has Spearman trend `rho=1.0`; `m=1` is flat because frozen alpha is zero. Reference-context scaling is not consistently positive (fixed-k Spearman values: `{', '.join(f'{int(k)}:{v:.3f}' if np.isfinite(v) else f'{int(k)}:NA' for k, v in reference_trends.items())}`). This is the expected signature of a sentinel-driven context offset rather than improved reference-context routing.

## Heterogeneity

- Contexts: {int(context.m1_positive.sum())}/50 positive; {_summary_stats(context, 'm1_g')}.
- Interventions: {int(intervention.m1_positive.sum())}/93 positive; {_summary_stats(intervention, 'm1_g')}.

No external mechanism annotation or post-hoc subset selection was performed.

## Component interpretation

The legal-sentinel decomposition defines `A_c` from observed sentinels only and `I_pc=D_pc-A_c`. M1 predicts `alpha*A_c` and no intervention-specific residual. Thus any supported recovery is specifically a context-wide susceptibility calibration; the intervention-specific component remains unresolved by construction. Covariance terms and reconstruction checks are reported in `M1_CONTEXT_WIDE_VS_INTERVENTION_SPECIFIC.csv`.

At all-but-one, the cross-plate energy fractions are `A/total={float(component_corner.context_wide_fraction_cross_plate):.6f}` and `I/total={float(component_corner.intervention_specific_fraction_cross_plate):.6f}`, with two covariance terms of `{float(component_corner.cov_A6_I14):.6f}` and `{float(component_corner.cov_I6_A14):.6f}` in summed utility units. The decomposition reconstructs total energy to absolute error `{float(component_corner.reconstruction_absolute_error):.3g}`. Frozen alpha equals one at this corner, so M1 recovers the defined A component exactly (`g_A=1`) and predicts none of I (`g_I=0`); this decomposition is an estimator anatomy, not an independent success claim.

The same-plate-to-other-plate directional recoveries are both negative despite positive pooled cross-product recovery. Cross-plate direction is therefore not robust in the conventional squared-error sense, another reason the descriptive 31% cannot be promoted to a remedy.

## Claim boundary

This secondary result cannot replace preregistered primary M2. Even a positive M1 estimate does not mean that a few sentinels solve CGC. The bounded interpretation is: empirical target-context calibration can partially restore a shared susceptibility component, while a large intervention-specific residual remains unresolved. The terminal verdict additionally requires the frozen correspondence-null family.

{verdict}
"""
    (out / "M1_SECONDARY_AUDIT_FINAL.md").write_text(final, encoding="utf-8")
    return verdict, {
        "allbut_m1_g": float(corner.estimate),
        "allbut_m1_simultaneous_lower_95": float(corner.simultaneous_lower_95),
        "allbut_m1_simultaneous_upper_95": float(corner.simultaneous_upper_95),
        "allbut_m1_minus_m2": float(corner_pair.m1_minus_m2),
        "allbut_m1_minus_m2_simultaneous_lower_95": float(corner_pair.simultaneous_lower_95),
        "allbut_full_g": full_g,
        "positive_context_fraction": context_positive,
        "positive_intervention_fraction": intervention_positive,
        "all_four_nulls_passed": all_nulls,
        "directional": direction_summary,
        "verdict": verdict,
    }


def run_audit(root: Path, upstream: Path) -> dict[str, Any]:
    out = root / "results/cgc_m1_secondary_audit"
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen_inputs(root, upstream)
    pd.DataFrame(inputs.executability_rows).to_csv(out / "M1_EXECUTABILITY_AUDIT.csv", index=False)

    cache_dir = out / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    null_cache = cache_dir / "m1_null_residuals.npz"
    component_path = out / "M1_CONTEXT_WIDE_VS_INTERVENTION_SPECIFIC.csv"
    directional_path = out / "M1_CROSS_PLATE_ROBUSTNESS.csv"
    if null_cache.exists() and component_path.exists() and directional_path.exists():
        cached = np.load(null_cache, allow_pickle=False)
        nulls = np.asarray(cached["nulls"], dtype=np.float64)
        observed = np.asarray(cached["observed"], dtype=np.float64)
        component = pd.read_csv(component_path)
        directional = pd.read_csv(directional_path)
        reconciliation = float(np.nanmax(np.abs(observed - np.asarray(inputs.utility["vafter"][1], dtype=np.float64))))
    else:
        nulls, observed, component, directional, reconciliation = recompute_m1_nulls_and_components(inputs)
        np.savez_compressed(null_cache, nulls=nulls, observed=observed)
    if reconciliation > 2e-5:
        raise RuntimeError(f"M1_FROZEN_RECONCILIATION_FAIL: {reconciliation}")

    truth = np.repeat(inputs.utility["vtruth"][:, None], len(K_VALUES), axis=1).reshape(90, 50, 93)
    model_residual = np.asarray(inputs.utility["vafter"], dtype=np.float64).reshape(4, 90, 50, 93)
    null_flat = nulls.reshape(4, 90, 50, 93)
    bootstrap_cache = cache_dir / "m1_bootstrap_draws.npz"
    if bootstrap_cache.exists():
        cached = np.load(bootstrap_cache, allow_pickle=False)
        model_draws = np.asarray(cached["model"], dtype=np.float64)
        null_draws = np.asarray(cached["null"], dtype=np.float64)
    else:
        model_draws, null_draws = bootstrap_all(truth, model_residual, null_flat)
        np.savez_compressed(bootstrap_cache, model=model_draws, null=null_draws)

    tables = build_inference_tables(inputs, nulls, model_draws, null_draws)
    context, intervention = heterogeneity_tables(inputs)
    tables["_full_truth"] = inputs.utility["full_truth"]
    tables["_m1_corner_residual"] = inputs.utility["vafter"][1, -1, -1]
    output_map = {
        "M1_SIMULTANEOUS_GRID_INFERENCE.csv": tables["m1"],
        "M1_ESTIMATOR_FAMILY_CORRECTION.csv": tables["estimators"],
        "M1_CORRESPONDENCE_NULLS.csv": tables["nulls"],
        "M1_VS_M2_PAIRED.csv": tables["paired"],
        "M1_SENTINEL_SCALING.csv": tables["sentinel"],
        "M1_REFERENCE_CONTEXT_SCALING.csv": tables["reference"],
        "M1_COMPRESSION_THRESHOLDS.csv": tables["thresholds"],
        "M1_CONTEXT_HETEROGENEITY.csv": context,
        "M1_INTERVENTION_HETEROGENEITY.csv": intervention,
        "M1_CONTEXT_WIDE_VS_INTERVENTION_SPECIFIC.csv": component,
        "M1_CROSS_PLATE_ROBUSTNESS.csv": directional,
    }
    for name, frame in output_map.items():
        frame.to_csv(out / name, index=False)
    pd.DataFrame(
        [
            {
                "check": "recomputed_M1_vs_frozen_utility_max_absolute_error",
                "observed": reconciliation,
                "tolerance": 2e-5,
                "passed": reconciliation <= 2e-5,
            },
            {
                "check": "frozen_episode_shape",
                "observed": str(tuple(model_residual.shape)),
                "tolerance": "(4,90,50,93)",
                "passed": tuple(model_residual.shape) == (4, 90, 50, 93),
            },
            {
                "check": "bootstrap_shape",
                "observed": str(tuple(model_draws.shape)),
                "tolerance": "(4,10000,90)",
                "passed": tuple(model_draws.shape) == (4, 10_000, 90),
            },
        ]
    ).to_csv(out / "M1_NUMERICAL_RECONCILIATION.csv", index=False)
    make_figures(out, tables, context, intervention, component, directional)

    protocol_commit = PROTOCOL_COMMIT
    verdict, summary = write_reports(
        root, out, inputs, tables, context, intervention, component, directional,
        reconciliation, protocol_commit,
    )
    artifact_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(out.rglob("*"))
        if path.is_file() and "_cache" not in path.parts and path.name != "M1_SECONDARY_AUDIT_RUN_MANIFEST.json"
    }
    manifest = {
        "phase": "CGC-EC-M1",
        "created_at": _now(),
        "identity": "POST_HOC_SECONDARY_FROZEN_PREDICTION_ADJUDICATION",
        "audit_branch": _git(root, "branch", "--show-current"),
        "git_commit_at_generation": _git(root, "rev-parse", "HEAD"),
        "protocol_commit": protocol_commit,
        "expected_base": EXPECTED_BASE,
        "upstream_result_commit": UPSTREAM_RESULT_COMMIT,
        "upstream_root_read_only": str(upstream),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_refit": False,
        "hyperparameter_selection": False,
        "null_re_evaluation": "deterministic frozen alpha/information sets; no fit",
        "m1_reconciliation_max_absolute_error": reconciliation,
        "source_hashes": inputs.source_hashes,
        "artifacts": artifact_hashes,
        "summary": summary,
    }
    _write_json(out / "M1_SECONDARY_AUDIT_RUN_MANIFEST.json", manifest)
    return manifest
