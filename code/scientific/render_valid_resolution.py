"""Render only the valid frozen pathway-minus-gene result, never the invalid null.

This reconstructs an analytical figure from prediction-derived frozen contrasts,
not the manual R2 manuscript SVG layout. No prediction, bootstrap or null is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_COMMIT = "46b25eec4a97e3001cc43262980af8f69c533478"
AUDIT_AUTHORITY = "21b08f02d3371129bb5cbf49704c65a271240c7c"
TABLE = "results/cgc_resolution_poc_v2/RESOLUTION2_PAIRED_COMPARISONS.csv"


def select_clean_rows(rows: list[dict[str, str]]) -> list[dict]:
    selected = []
    corrected = any(row["family"] == "VALID_PATHWAY_MINUS_GENE_2_BUDGETS" for row in rows)
    comparison_family = "VALID_PATHWAY_MINUS_GENE_2_BUDGETS" if corrected else "PATHWAY_PRIMARY_2BUDGET_X_2CONTRAST"
    for budget in ("m49_k92", "m40_k4"):
        keys = [("point_estimate", f"{budget}_g_gene"), ("point_estimate", f"{budget}_g_pathway"),
                (comparison_family, f"{budget}_pathway_minus_gene")]
        for family, contrast in keys:
            matching = [row for row in rows if row["family"] == family and row["contrast"] == contrast]
            if len(matching) != 1:
                raise ValueError(f"Expected one frozen valid contrast: {family}/{contrast}")
            row = matching[0]
            selected.append({**row, "source_file": TABLE, "source_commit": SOURCE_COMMIT,
                             "audit_authority": AUDIT_AUTHORITY,
                             "post_freeze_correction": "EPISODE_REFERENCE_SUPPORT_ONLY_V2" if corrected else ""})
    if any("random" in row["contrast"].lower() for row in selected):
        raise AssertionError("Invalid random-projection arm reached current rendering")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    corrected_table = TABLE.replace("RESOLUTION2_PAIRED_COMPARISONS.csv", "RESOLUTION2_VALID_COMPARISONS_CORRECTED.csv")
    default_table = corrected_table if (HERE / "bundles/resolution_pathway_frozen" / corrected_table).exists() else TABLE
    parser.add_argument("--source-table", type=Path, default=HERE / "bundles/resolution_pathway_frozen" / default_table)
    parser.add_argument("--out-dir", type=Path, required=True, help="new output directory, never an existing manuscript figure directory")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    bundle = HERE / "bundles/resolution_pathway_frozen"
    manifest = json.loads((bundle / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    table_relative = args.source_table.resolve().relative_to(bundle.resolve()).as_posix()
    expected = next(row["sha256"] for row in manifest["files"] if row["path"] == table_relative)
    digest = hashlib.sha256(args.source_table.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError("Frozen resolution contrast-table hash mismatch")
    with args.source_table.open(newline="", encoding="utf-8-sig") as handle:
        rows = select_clean_rows(list(csv.DictReader(handle)))
    for row in rows:
        row["source_file"] = table_relative
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.25))
    for ax, budget, letter in zip(axes, ("m49_k92", "m40_k4"), ("a", "b")):
        local = {row["contrast"]: row for row in rows if row["contrast"].startswith(budget)}
        values = [float(local[f"{budget}_g_gene"]["estimate"]), float(local[f"{budget}_g_pathway"]["estimate"])]
        ax.plot([0, 1], values, color="#8a8f98", lw=1)
        ax.scatter([0, 1], values, c=["#708bac", "#549f97"], s=42, zorder=3)
        ax.set_xticks([0, 1], ["Gene", "Predefined pathways"])
        ax.set_ylabel(r"Recovery $g$")
        ax.set_title(budget.replace("_", ", ").replace("m", "m=").replace("k", "k="), loc="center")
        contrast = local[f"{budget}_pathway_minus_gene"]
        ax.text(.5, .03, f"Δg={float(contrast['estimate']):.6f}\nCorrected interval [{float(contrast['simultaneous_lower_95']):.6f}, {float(contrast['simultaneous_upper_95']):.6f}]", ha="center", transform=ax.transAxes, fontsize=7)
        ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontweight="bold")
        ax.margins(x=.35, y=.35)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True)
    fig.savefig(args.out_dir / "VALID_PATHWAY_MINUS_GENE.png", dpi=300)
    fig.savefig(args.out_dir / "VALID_PATHWAY_MINUS_GENE.svg")
    plt.close(fig)
    for row in rows:
        row["source_sha256"] = digest
    with (args.out_dir / "VALID_PATHWAY_MINUS_GENE_SOURCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
