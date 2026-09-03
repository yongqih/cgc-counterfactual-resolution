"""Dependency-free SVG figures for the frozen CGC-TCELL-1C outputs."""

from __future__ import annotations

from pathlib import Path
import xml.sax.saxutils as sax

import numpy as np
import pandas as pd


COLORS = {"Ridge": "#2563eb", "Bilinear": "#d97706", "MLP": "#7c3aed"}


def _start(title: str, width=1100, height=680):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{sax.escape(title)}</text>',
    ]


def _finish(parts, path):
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def energy(path: Path, frame: pd.DataFrame):
    means = frame.groupby("model")[["e_parallel", "e_perpendicular"]].mean()
    parts = _start("Figure 1 — Predicted operator energy decomposition")
    labels = ["Truth", "Ridge", "Bilinear", "MLP"]
    x0, y0, h, w = 130, 560, 460, 130
    maximum = 1.0
    for i, label in enumerate(labels):
        x = x0 + i * 235
        if label == "Truth":
            parallel, perpendicular = 1.0, 0.0
        else:
            parallel, perpendicular = means.loc[label]
        hp, ho = h * parallel / maximum, h * perpendicular / maximum
        parts.append(f'<rect x="{x}" y="{y0-ho}" width="{w}" height="{ho}" fill="#d97706"/>')
        parts.append(f'<rect x="{x}" y="{y0-ho-hp}" width="{w}" height="{hp}" fill="#2563eb"/>')
        parts.append(f'<text x="{x+w/2}" y="{y0+28}" text-anchor="middle" font-family="Arial" font-size="15">{label}</text>')
        parts.append(f'<text x="{x+w/2}" y="{y0-ho-8}" text-anchor="middle" font-family="Arial" font-size="13">total {parallel+perpendicular:.4g}</text>')
    parts += [
        '<rect x="790" y="70" width="18" height="18" fill="#2563eb"/><text x="815" y="84" font-family="Arial" font-size="14">truth-aligned</text>',
        '<rect x="790" y="98" width="18" height="18" fill="#d97706"/><text x="815" y="112" font-family="Arial" font-size="14">truth-orthogonal</text>',
        f'<line x1="80" y1="{y0}" x2="1030" y2="{y0}" stroke="#111"/>',
    ]
    _finish(parts, path)


def shuffle_histograms(path: Path, null: pd.DataFrame, summary: pd.DataFrame):
    parts = _start("Figure 2 — Intervention-identity shuffle null")
    pooled = null.loc[null.scope.eq("pooled")]
    observed = summary.loc[summary.scope.eq("pooled")].set_index("model")
    for panel, model in enumerate(("Ridge", "Bilinear", "MLP")):
        values = pooled.loc[pooled.model.eq(model), "cosine"].to_numpy()
        lo, hi = float(np.quantile(values, .001)), float(observed.loc[model, "observed_cosine"] * 1.08)
        counts, edges = np.histogram(values, bins=100, range=(lo, hi), density=True)
        x0, y0, pw, ph = 100, 190 + panel * 165, 890, 105
        maxc = max(counts.max(), 1e-12)
        points = []
        for i, count in enumerate(counts):
            x = x0 + pw * ((edges[i] + edges[i + 1]) / 2 - lo) / (hi - lo)
            y = y0 - ph * count / maxc
            points.append(f"{x:.1f},{y:.1f}")
        obs = float(observed.loc[model, "observed_cosine"])
        xo = x0 + pw * (obs - lo) / (hi - lo)
        parts += [
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[model]}" stroke-width="2"/>',
            f'<line x1="{xo:.1f}" y1="{y0-ph}" x2="{xo:.1f}" y2="{y0}" stroke="#dc2626" stroke-width="3"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0+pw}" y2="{y0}" stroke="#222"/>',
            f'<text x="20" y="{y0-45}" font-family="Arial" font-size="15">{model}</text>',
            f'<text x="{xo-6:.1f}" y="{y0-ph-8}" text-anchor="end" font-family="Arial" font-size="12">observed {obs:.5f}</text>',
            f'<text x="{x0}" y="{y0+20}" font-family="Arial" font-size="11">{lo:.4g}</text>',
            f'<text x="{x0+pw}" y="{y0+20}" text-anchor="end" font-family="Arial" font-size="11">{hi:.4g}</text>',
        ]
    _finish(parts, path)


def matched_mismatched(path: Path, root: Path, retrieval: pd.DataFrame):
    parts = _start("Figure 3 — Matched vs mismatched intervention fingerprints")
    rng = np.random.default_rng(20260820)
    donors = sorted(retrieval.held_out_donor.unique())
    for panel, model in enumerate(("Ridge", "Bilinear", "MLP")):
        matched = retrieval.loc[retrieval.model.eq(model), "matched_cosine"].to_numpy()
        mismatch = []
        for donor in donors:
            matrix = np.load(root / f"results/cgc_tcell_1c/fingerprint_similarity_matrix/{model}_{donor}.float32.npy", mmap_mode="r")
            rows = rng.integers(0, matrix.shape[0], 250_000)
            cols = rng.integers(0, matrix.shape[1], 250_000)
            keep = rows != cols
            mismatch.append(np.asarray(matrix[rows[keep], cols[keep]], np.float32))
        mismatch = np.concatenate(mismatch)
        lo = float(min(np.quantile(matched, .005), np.quantile(mismatch, .005)))
        hi = float(max(np.quantile(matched, .995), np.quantile(mismatch, .995)))
        x0, y0, pw, ph = 100, 190 + panel * 165, 890, 105
        for values, color, label in ((mismatch, "#9ca3af", "mismatched"), (matched, COLORS[model], "matched")):
            counts, edges = np.histogram(values, bins=120, range=(lo, hi), density=True)
            maxc = max(counts.max(), 1e-12)
            points = []
            for i, count in enumerate(counts):
                x = x0 + pw * ((edges[i] + edges[i+1])/2-lo)/(hi-lo)
                y = y0 - ph * count/maxc
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        parts += [f'<line x1="{x0}" y1="{y0}" x2="{x0+pw}" y2="{y0}" stroke="#222"/>',f'<text x="20" y="{y0-45}" font-family="Arial" font-size="15">{model}</text>']
    parts += ['<line x1="780" y1="72" x2="820" y2="72" stroke="#2563eb" stroke-width="3"/><text x="830" y="77" font-family="Arial" font-size="14">matched</text>','<line x1="780" y1="98" x2="820" y2="98" stroke="#9ca3af" stroke-width="3"/><text x="830" y="103" font-family="Arial" font-size="14">mismatched</text>']
    _finish(parts, path)


def retrieval_cdf(path: Path, retrieval: pd.DataFrame, p: int):
    parts = _start("Figure 5 — True-intervention retrieval rank CDF")
    x0, y0, pw, ph = 100, 570, 900, 470
    grid = np.linspace(0, 1, 501)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+pw}" y2="{y0-ph}" stroke="#9ca3af" stroke-dasharray="7 7" stroke-width="2"/>')
    for model in ("Ridge", "Bilinear", "MLP"):
        ranks = retrieval.loc[retrieval.model.eq(model), "true_match_rank"].to_numpy()/p
        ranks.sort()
        y = np.searchsorted(ranks, grid, side="right")/len(ranks)
        points = [f"{x0+pw*x:.1f},{y0-ph*yy:.1f}" for x,yy in zip(grid,y)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[model]}" stroke-width="3"/>')
    parts += [f'<line x1="{x0}" y1="{y0}" x2="{x0+pw}" y2="{y0}" stroke="#111"/>',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0-ph}" stroke="#111"/>','<text x="550" y="640" text-anchor="middle" font-family="Arial" font-size="16">normalized true-match rank (lower is better)</text>','<text x="25" y="330" transform="rotate(-90 25 330)" text-anchor="middle" font-family="Arial" font-size="16">fraction retrieved by rank</text>']
    for i,model in enumerate(("Ridge","Bilinear","MLP")): parts.append(f'<line x1="790" y1="{70+i*25}" x2="825" y2="{70+i*25}" stroke="{COLORS[model]}" stroke-width="3"/><text x="835" y="{75+i*25}" font-family="Arial" font-size="14">{model}</text>')
    _finish(parts, path)


def generate(root: Path):
    output = root / "results/cgc_tcell_1c"
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    energy(figures / "figure_1_energy_decomposition.svg", pd.read_csv(output / "parallel_orthogonal_energy.csv"))
    shuffle_histograms(figures / "figure_2_intervention_shuffle.svg", pd.read_csv(output / "intervention_shuffle_null.csv"), pd.read_csv(output / "intervention_shuffle_summary.csv"))
    retrieval = pd.read_csv(output / "intervention_retrieval.csv")
    matched_mismatched(figures / "figure_3_matched_mismatched.svg", root, retrieval)
    retrieval_cdf(figures / "figure_5_intervention_retrieval.svg", retrieval, 9386)


if __name__ == "__main__":
    generate(Path.cwd())
