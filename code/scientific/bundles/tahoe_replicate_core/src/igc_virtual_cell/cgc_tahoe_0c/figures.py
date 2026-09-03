"""Dependency-free SVG figures for the frozen Tahoe 0C analysis."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from igc_virtual_cell.cgc_tahoe_0c.analysis import interaction_tensor, load_core


WIDTH = 1000
HEIGHT = 650


def _svg(title: str, body: str, width: int = WIDTH, height: int = HEIGHT) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="50" y="45" font-family="Arial" font-size="24" font-weight="bold">{html.escape(title)}</text>'
        f'{body}</svg>'
    )


def _text(x: float, y: float, value: object, size: int = 13, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" fill="#222">{html.escape(str(value))}</text>'


def _line_chart(
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    color: str = "#1769aa",
) -> str:
    left, top, right, bottom = 100, 85, 940, 555
    ymin = float(np.nanmin(lower if lower is not None else y))
    ymax = float(np.nanmax(upper if upper is not None else y))
    padding = max((ymax - ymin) * 0.12, 0.02)
    ymin -= padding
    ymax += padding
    sx = lambda value: left + (float(value) - float(x.min())) / (float(x.max()) - float(x.min())) * (right - left)
    sy = lambda value: bottom - (float(value) - ymin) / (ymax - ymin) * (bottom - top)
    parts = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for tick in np.linspace(ymin, ymax, 6):
        yy = sy(tick)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" stroke="#e8e8e8"/>')
        parts.append(_text(left - 10, yy + 4, f"{tick:.2f}", 12, "end"))
    if lower is not None and upper is not None:
        polygon = [(sx(a), sy(b)) for a, b in zip(x, lower, strict=True)] + [
            (sx(a), sy(b)) for a, b in zip(x[::-1], upper[::-1], strict=True)
        ]
        points = " ".join(f"{a:.2f},{b:.2f}" for a, b in polygon)
        parts.append(f'<polygon points="{points}" fill="{color}" opacity="0.15"/>')
    points = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a, b in zip(x, y, strict=True))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
    for a, b in zip(x, y, strict=True):
        parts.append(f'<circle cx="{sx(a):.2f}" cy="{sy(b):.2f}" r="6" fill="{color}"/>')
        parts.append(_text(sx(a), bottom + 24, int(a), 12, "middle"))
    parts.append(_text((left + right) / 2, 615, "Observed context support N", 15, "middle"))
    parts.append(
        f'<text x="25" y="{(top + bottom) / 2}" transform="rotate(-90 25 {(top + bottom) / 2})" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(ylabel)}</text>'
    )
    return _svg(title, "".join(parts))


def generate_figures(root: Path) -> list[str]:
    root = root.resolve()
    result_dir = root / "results/cgc_tahoe_0c"
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    # Figure 1: complete replicated core.
    parts = []
    for panel, (label, x0) in enumerate((("Plate 6", 80), ("Plate 14", 530))):
        parts.append(_text(x0 + 180, 90, label, 17, "middle"))
        cell_w, cell_h = 3.8, 8.5
        for context in range(50):
            for intervention in range(93):
                parts.append(
                    f'<rect x="{x0 + intervention * cell_w:.2f}" y="{110 + context * cell_h:.2f}" '
                    f'width="{cell_w:.2f}" height="{cell_h:.2f}" fill="#248f5a"/>'
                )
        parts.append(_text(x0 + 180, 560, "93 exact drug×dose interventions", 13, "middle"))
    parts.append(_text(25, 320, "50 contexts", 14, "middle"))
    path = figure_dir / "figure1_replicated_core.svg"
    path.write_text(_svg("Figure 1 — Tahoe replicated core", "".join(parts)), encoding="utf-8")
    outputs.append(str(path.relative_to(root)))

    # Figure 2: truth interaction scatter.
    delta, _ = load_core(root)
    gamma = interaction_tensor(delta)
    first, second = gamma[0].ravel(), gamma[1].ravel()
    rng = np.random.default_rng(202608206)
    sample = rng.choice(len(first), size=min(8000, len(first)), replace=False)
    first, second = first[sample], second[sample]
    limit = float(np.quantile(np.abs(np.concatenate([first, second])), 0.995))
    left, top, right, bottom = 110, 85, 900, 570
    sx = lambda value: left + (value + limit) / (2 * limit) * (right - left)
    sy = lambda value: bottom - (value + limit) / (2 * limit) * (bottom - top)
    parts = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{sx(-limit)}" y1="{sy(-limit)}" x2="{sx(limit)}" y2="{sy(limit)}" stroke="#999" stroke-dasharray="5 5"/>',
    ]
    parts.extend(
        f'<circle cx="{sx(a):.2f}" cy="{sy(b):.2f}" r="1.5" fill="#1769aa" opacity="0.25"/>'
        for a, b in zip(first, second, strict=True)
        if abs(a) <= limit and abs(b) <= limit
    )
    truth = json.loads((result_dir / "truth_gate.json").read_text(encoding="utf-8"))
    parts.append(_text(650, 110, f"cosine = {truth['operator_cosine']:.3f}", 15))
    parts.append(_text(650, 135, f"signal fraction = {truth['signal_fraction']:.3f}", 15))
    parts.append(_text((left + right) / 2, 620, "Plate 6 interaction", 15, "middle"))
    parts.append(_text(30, 330, "Plate 14 interaction", 15, "middle"))
    path = figure_dir / "figure2_truth_replicate_reliability.svg"
    path.write_text(_svg("Figure 2 — Truth replicate reliability", "".join(parts)), encoding="utf-8")
    outputs.append(str(path.relative_to(root)))

    curve = pd.read_csv(result_dir / "support_q_curve.csv")
    bootstrap = pd.read_csv(result_dir / "bootstrap_summary.csv")
    q_boot = bootstrap[bootstrap["metric"].str.fullmatch(r"q\d+")].copy()
    x = curve["support_size"].to_numpy(dtype=float)
    q = curve["q_pooled"].to_numpy(dtype=float)
    lower = q_boot["lower_95"].to_numpy(dtype=float)
    upper = q_boot["upper_95"].to_numpy(dtype=float)
    path = figure_dir / "figure3_support_q_curve.svg"
    path.write_text(
        _line_chart("Figure 3 — Replicate-calibrated residual", x, q, "q_N", lower, upper),
        encoding="utf-8",
    )
    outputs.append(str(path.relative_to(root)))
    path = figure_dir / "figure4_recoverable_coverage.svg"
    path.write_text(
        _line_chart(
            "Figure 4 — Recoverable context coverage",
            x,
            1 - q,
            "g_N = 1 - q_N",
            1 - upper,
            1 - lower,
            "#248f5a",
        ),
        encoding="utf-8",
    )
    outputs.append(str(path.relative_to(root)))

    maximal = pd.read_csv(result_dir / "maximal_support_summary.csv").sort_values("q49")
    left, top, right, bottom = 100, 85, 950, 565
    ymin, ymax = float(maximal["q49"].min()), float(maximal["q49"].max())
    sx = lambda index: left + index / 49 * (right - left)
    sy = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    parts = [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>']
    for index, row in enumerate(maximal.itertuples(index=False)):
        parts.append(f'<circle cx="{sx(index):.2f}" cy="{sy(row.q49):.2f}" r="5" fill="#7b2cbf"/>')
    parts.append(_text(520, 615, "50 held-out contexts (sorted)", 15, "middle"))
    parts.append(_text(30, 330, "q49", 15, "middle"))
    path = figure_dir / "figure5_max_support_contexts.svg"
    path.write_text(_svg("Figure 5 — Max-support context distribution", "".join(parts)), encoding="utf-8")
    outputs.append(str(path.relative_to(root)))

    null = pd.read_csv(result_dir / "residual_shuffle_null.csv")
    null49 = null.loc[null["support_size"] == 49, "residual_energy"].to_numpy()
    verdict = json.loads((result_dir / "verdict.json").read_text(encoding="utf-8"))
    counts, edges = np.histogram(null49, bins=40)
    left, top, right, bottom = 100, 90, 940, 565
    sx = lambda value: left + (value - edges.min()) / (max(verdict["max_support_residual_energy"], edges.max()) - edges.min()) * (right - left)
    sy = lambda value: bottom - value / counts.max() * (bottom - top)
    parts = []
    for count, a, b in zip(counts, edges[:-1], edges[1:], strict=True):
        parts.append(
            f'<rect x="{sx(a):.2f}" y="{sy(count):.2f}" width="{max(sx(b)-sx(a)-1,1):.2f}" height="{bottom-sy(count):.2f}" fill="#9ecae1"/>'
        )
    observed_x = sx(verdict["max_support_residual_energy"])
    parts.append(f'<line x1="{observed_x}" y1="{top}" x2="{observed_x}" y2="{bottom}" stroke="#c62828" stroke-width="4"/>')
    parts.append(_text(observed_x - 5, top + 20, "observed", 14, "end"))
    path = figure_dir / "figure6_max_support_null.svg"
    path.write_text(_svg("Figure 6 — Residual-vs-shuffle null at N=49", "".join(parts)), encoding="utf-8")
    outputs.append(str(path.relative_to(root)))

    budget = pd.read_csv(result_dir / "measurement_budget.csv")
    budget = budget[budget["support_size"].isin([2, 8, 16, 32, 49])]
    parts = []
    colors = ["#2ca25f", "#756bb1", "#bdbdbd"]
    labels = ["captured", "reproducible unexplained", "measurement / unresolved"]
    for index, row in enumerate(budget.itertuples(index=False)):
        x0 = 150 + index * 155
        cumulative = 0.0
        values = [
            row.captured_fraction_of_observed,
            row.residual_fraction_of_observed,
            row.measurement_fraction_of_observed,
        ]
        for value, color in zip(values, colors, strict=True):
            y1 = 540 - cumulative * 430
            y2 = 540 - (cumulative + value) * 430
            parts.append(
                f'<rect x="{x0}" y="{min(y1,y2):.2f}" width="85" height="{abs(y2-y1):.2f}" fill="{color}"/>'
            )
            cumulative += value
        parts.append(_text(x0 + 42, 575, f"N={row.support_size}", 13, "middle"))
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        parts.append(f'<rect x="{650}" y="{75+index*25}" width="16" height="16" fill="{color}"/>')
        parts.append(_text(675, 88 + index * 25, label, 13))
    path = figure_dir / "figure7_measurement_budget.svg"
    path.write_text(_svg("Figure 7 — Measurement-calibrated budget", "".join(parts)), encoding="utf-8")
    outputs.append(str(path.relative_to(root)))

    heterogeneity = pd.read_csv(result_dir / "context_heterogeneity.csv")
    matrix = heterogeneity.pivot(index="context_id", columns="support_size", values="q_context")
    matrix = matrix[list(map(int, x))]
    vmin, vmax = float(np.nanquantile(matrix, 0.02)), float(np.nanquantile(matrix, 0.98))
    parts = []
    for row_index, (_, row) in enumerate(matrix.iterrows()):
        for column_index, value in enumerate(row):
            scaled = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
            red = int(40 + 200 * scaled)
            blue = int(220 - 170 * scaled)
            color = f"rgb({red},90,{blue})"
            parts.append(
                f'<rect x="{180+column_index*85}" y="{85+row_index*9.5:.2f}" width="85" height="9.5" fill="{color}"/>'
            )
    for column_index, value in enumerate(matrix.columns):
        parts.append(_text(222 + column_index * 85, 585, value, 12, "middle"))
    parts.append(_text(520, 620, "Support size N", 14, "middle"))
    parts.append(_text(50, 330, "50 contexts", 14, "middle"))
    path = figure_dir / "figure8_context_heterogeneity.svg"
    path.write_text(_svg("Figure 8 — Context heterogeneity in q_N", "".join(parts)), encoding="utf-8")
    outputs.append(str(path.relative_to(root)))
    return outputs
