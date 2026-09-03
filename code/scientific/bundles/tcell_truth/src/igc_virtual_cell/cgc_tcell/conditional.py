"""Conditional post-confirmation analyses for CGC-TCELL-0A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml


def _git(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _anova_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return balanced P×D, P×S and P×D×S ANOVA components."""
    z = np.asarray(values, dtype=np.float32)
    grand = z.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    p = z.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32) - grand
    d = z.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32) - grand
    s = z.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32) - grand
    pd_component = (
        z.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
        - grand
        - p
        - d
    )
    ps_component = (
        z.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        - grand
        - p
        - s
    )
    ds = (
        z.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        - grand
        - d
        - s
    )
    pds_component = z - grand - p - d - s - pd_component - ps_component - ds
    return pd_component, ps_component, pds_component


def factorial_signal_decomposition(
    guide_a: np.ndarray,
    guide_b: np.ndarray,
    *,
    chunk_genes: int,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Noise-corrected reliable energy from independent guide halves."""
    a, b = np.asarray(guide_a), np.asarray(guide_b)
    if a.shape != b.shape or a.ndim != 4 or a.shape[1:3] != (4, 3):
        raise ValueError("Expected matched P x 4 x 3 x G guide tensors")
    contributions = np.zeros((a.shape[0], 3), dtype=np.float64)
    total_genes = a.shape[-1]
    for start in range(0, total_genes, chunk_genes):
        stop = min(start + chunk_genes, total_genes)
        components_a = _anova_components(a[:, :, :, start:stop])
        components_b = _anova_components(b[:, :, :, start:stop])
        contributions[:, 0] += np.einsum(
            "pdkg,pdkg->p", components_a[0], components_b[0], optimize=True
        ) / (4 * total_genes)
        contributions[:, 1] += np.einsum(
            "pdsg,pdsg->p", components_a[1], components_b[1], optimize=True
        ) / (3 * total_genes)
        contributions[:, 2] += np.einsum(
            "pdsg,pdsg->p", components_a[2], components_b[2], optimize=True
        ) / (4 * 3 * total_genes)
    energy = contributions.mean(axis=0)
    positive = np.maximum(energy, 0.0)
    shares = positive / positive.sum() if positive.sum() else np.full(3, np.nan)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((bootstrap_draws, 3), dtype=np.float64)
    for draw in range(bootstrap_draws):
        indices = rng.integers(a.shape[0], size=a.shape[0])
        bootstrap[draw] = contributions[indices].mean(axis=0)
    labels = [
        "intervention_x_donor",
        "intervention_x_stimulation",
        "intervention_x_donor_x_stimulation",
    ]
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "component": label,
                "reliable_crossguide_energy": float(energy[index]),
                "positive_truncated_relative_share": float(shares[index]),
                "bootstrap_ci_low": float(np.quantile(bootstrap[:, index], 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap[:, index], 0.975)),
                "bootstrap_probability_positive": float(np.mean(bootstrap[:, index] > 0)),
                "targets": a.shape[0],
                "genes": a.shape[-1],
                "guide_measurements": "A_x_B",
            }
        )
    return pd.DataFrame(rows)


def run_factorial(root: Path, config_path: Path) -> pd.DataFrame:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["outputs"]["directory"]
    verdict = json.loads((output / "verdict.json").read_text(encoding="utf-8"))
    if verdict["verdict"] != "TCELL_STIMULATION_OPERATOR_SIGNAL_CONFIRMED":
        raise RuntimeError("Factorial decomposition is conditional on confirmed primary signal")
    manifest = json.loads(
        (output / "guide_disjoint_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["status"] != "GUIDE_DISJOINT_FULL_CONTEXT_MATERIALIZED":
        raise RuntimeError("Noise-corrected factorial decomposition requires guide halves")
    shape = tuple(manifest["response_shape"])
    guide_a = np.memmap(
        root / manifest["response_a_path"], mode="r", dtype=np.float32, shape=shape
    )
    guide_b = np.memmap(
        root / manifest["response_b_path"], mode="r", dtype=np.float32, shape=shape
    )
    result = factorial_signal_decomposition(
        guide_a,
        guide_b,
        chunk_genes=int(config["normalization"]["chunk_genes"]),
        bootstrap_draws=int(config["bootstrap"]["draws"]),
        seed=int(config["random_seed"]) + 4,
    )
    result.insert(0, "git_provenance", _git(root))
    result.to_csv(output / "factorial_signal_decomposition.csv", index=False)
    report = root / config["outputs"]["factorial_report"]
    lookup = result.set_index("component")
    report.write_text(
        f"""# CGC-TCELL-0A factorial signal decomposition

Git provenance: `{_git(root)}`.

Noise correction uses independent guide A×B cross-energy over **{shape[0]:,}** targets and **{shape[-1]:,}** strict-trans genes.

| Component | Reliable energy | Positive-share | 95% target bootstrap CI |
|---|---:|---:|---:|
| intervention × donor | {lookup.loc['intervention_x_donor', 'reliable_crossguide_energy']:.8g} | {lookup.loc['intervention_x_donor', 'positive_truncated_relative_share']:.3f} | [{lookup.loc['intervention_x_donor', 'bootstrap_ci_low']:.8g}, {lookup.loc['intervention_x_donor', 'bootstrap_ci_high']:.8g}] |
| intervention × stimulation | {lookup.loc['intervention_x_stimulation', 'reliable_crossguide_energy']:.8g} | {lookup.loc['intervention_x_stimulation', 'positive_truncated_relative_share']:.3f} | [{lookup.loc['intervention_x_stimulation', 'bootstrap_ci_low']:.8g}, {lookup.loc['intervention_x_stimulation', 'bootstrap_ci_high']:.8g}] |
| intervention × donor × stimulation | {lookup.loc['intervention_x_donor_x_stimulation', 'reliable_crossguide_energy']:.8g} | {lookup.loc['intervention_x_donor_x_stimulation', 'positive_truncated_relative_share']:.3f} | [{lookup.loc['intervention_x_donor_x_stimulation', 'bootstrap_ci_low']:.8g}, {lookup.loc['intervention_x_donor_x_stimulation', 'bootstrap_ci_high']:.8g}] |

These are balanced-factorial, noise-corrected interaction energies; they are not model predictions and are not a CGC claim.
""",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/cgc_tcell.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    print(run_factorial(root, config).to_string(index=False))


if __name__ == "__main__":
    main()
