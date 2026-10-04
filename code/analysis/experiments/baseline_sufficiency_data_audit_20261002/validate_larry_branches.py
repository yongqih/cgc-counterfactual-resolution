"""Audit real endpoint ambiguity for a shared, experimentally recorded baseline.

This tests realized clonal branch outcomes, not a noiseless daughter-cell state
or an impossibility of predicting conditional means/distributions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "larry"
MARKERS = {
    "Monocyte": ["Ms4a6d", "Fabp5", "Ctss", "Ms4a6c", "Tgfbi", "Olfm1", "Csf1r", "Ccr2", "Klf4", "F13a1"],
    "Neutrophil": ["S100a9", "Itgb2l", "Elane", "Fcnb", "Mpo", "Prtn3", "S100a6", "S100a8", "Lcn2", "Lrg1"],
}


def exclude_ambiguous_late_records(meta, counts):
    """Discard potential repeated cells across libraries, retaining neither copy.

    A droplet barcode is only unique within a library. A matching droplet
    barcode AND clonal tag in the same culture/time is suspicious across
    libraries. This conservative screen does not claim every match is a
    duplicate. It avoids relying on the ambiguous records for replication.
    """
    keys = ['culture_family', 'Time point', 'Well', 'clone_index', 'Cell barcode']
    ambiguous = (meta['Time point'] == 6) & meta.duplicated(keys, keep=False)
    audit = {'n_late_records_excluded_as_potential_repeated_cells': int(ambiguous.sum()),
             'n_affected_clones': int(meta.loc[ambiguous, 'clone_index'].nunique()),
             'clone1978_records_excluded': int((ambiguous & (meta.clone_index == 1978)).sum()),
             'clone1261_records_excluded': int((ambiguous & (meta.clone_index == 1261)).sum()),
             'screen': 'Same family, day, well, clonal tag and droplet barcode; exclude every ambiguous copy.',
             'remaining_interpretation': 'Disjoint measured-cell records within the observed culture branches; not independent culture experiments.'}
    keep = ~ambiguous.to_numpy()
    return meta.loc[keep].reset_index(drop=True), counts[keep], audit


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def permutation_mean_p(a, b, rng, repetitions=10000):
    y = np.r_[a, b]
    contrast = abs(a.mean() - b.mean())
    n = len(a)
    count = 0
    for _ in range(repetitions):
        indices = rng.permutation(len(y))
        d = abs(y[indices[:n]].mean() - y[indices[n:]].mean())
        count += d >= contrast - 1e-12
    return (count + 1) / (repetitions + 1)


def cross_library_signal(x, meta, rng, repetitions=3000):
    """Independent-library cross product removes zero-mean sampling variance.

    Libraries partitioned by their published IDs, without using RNA values.
    Bootstrap is conditional on the observed cultures; it does not supply
    independent biological culture replicates or generalize across donors.
    """
    groups = {}
    for well in (1, 2):
        libraries = sorted(meta.loc[meta.Well == well, "Library"].unique())
        if len(libraries) < 2:
            return None
        groups[well, "A"] = np.flatnonzero((meta.Well == well) & (meta.Library == libraries[0]))
        groups[well, "B"] = np.flatnonzero((meta.Well == well) & (meta.Library != libraries[0]))
    n = len(meta)
    a = np.zeros(n)
    b = np.zeros(n)
    boot_a = np.zeros((repetitions, n))
    boot_b = np.zeros((repetitions, n))
    for (well, group), indices in groups.items():
        sign = 1 if well == 1 else -1
        target, boots = (a, boot_a) if group == "A" else (b, boot_b)
        target[indices] = sign / len(indices)
        boots[:, indices] = sign * rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repetitions) / len(indices)
    gram = (x @ x.T).toarray().astype(np.float64)
    estimate = float(a @ gram @ b) / 4
    draws = np.einsum("bi,bi->b", boot_a @ gram, boot_b, optimize=True) / 4
    return {"estimate": estimate, "conditional_bootstrap_CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "group_sample_sizes": {f"well{w}_{g}": len(v) for (w, g), v in groups.items()},
            "n_bootstrap": repetitions}


def main():
    rng = np.random.default_rng(20261002)
    meta = pd.read_csv(FOLDER / "branch_validation_metadata.csv")
    counts = load_npz(FOLDER / "branch_validation_full_gene.npz").astype(np.float64)
    genes = json.loads((FOLDER / "branch_validation_gene_names.json").read_text(encoding="utf-8"))
    assert counts.shape == (len(meta), len(genes))
    meta, counts, duplicate_audit = exclude_ambiguous_late_records(meta, counts)
    (FOLDER / 'branch_duplicate_record_screen.json').write_text(json.dumps(duplicate_audit, indent=2), encoding='utf-8')
    expression = counts.copy()
    expression.data = np.log1p(expression.data)
    late = meta[meta["Time point"] == 6]
    sizes = late.groupby(["clone_index", "Well"]).size().unstack(fill_value=0)
    eligible = sizes.index[(sizes[1] >= 10) & (sizes[2] >= 10)].tolist()
    fate_names = sorted(pd.read_csv(FOLDER / "stateFate_inVitro_metadata.txt.gz", sep="\t")["Cell type annotation"].unique())
    fate_tests, marker_tests, signals, baseline_rows = [], [], [], []
    program = {}
    for name, panel in MARKERS.items():
        indices = [genes.index(g) for g in panel]
        program[name] = np.asarray(expression[:, indices].mean(axis=1)).ravel()
    for clone in eligible:
        baseline = meta[(meta.clone_index == clone) & (meta["Time point"] == 2)]
        baseline_rows.append({"clone_index": clone, "n_baseline_cells": len(baseline),
                              "baseline_cell_indices": baseline.original_cell_index.tolist(),
                              "n_input_gene_features": len(genes),
                              "culture_family": baseline.culture_family.iloc[0]})
        selected = (meta.clone_index == clone) & (meta["Time point"] == 6)
        local = meta.loc[selected].reset_index(drop=True)
        for fate in fate_names:
            totals = [int((local.Well == w).sum()) for w in (1, 2)]
            hits = [int(((local.Well == w) & (local["Cell type annotation"] == fate)).sum()) for w in (1, 2)]
            _, p = fisher_exact([[hits[0], totals[0] - hits[0]], [hits[1], totals[1] - hits[1]]])
            fate_tests.append({"clone_index": clone, "fate": fate, "n_well1": totals[0], "n_well2": totals[1],
                               "count_well1": hits[0], "count_well2": hits[1], "p": float(p)})
        for name, values in program.items():
            a = values[selected & (meta.Well == 1)]
            b = values[selected & (meta.Well == 2)]
            marker_tests.append({"clone_index": clone, "program": name,
                                 "mean_well1": float(a.mean()), "mean_well2": float(b.mean()),
                                 "p": permutation_mean_p(a, b, rng)})
        signal = cross_library_signal(expression[selected.to_numpy()], local, rng)
        if signal is not None:
            signals.append({"clone_index": clone, "culture_family": local.culture_family.iloc[0], **signal})
    fate_tests = pd.DataFrame(fate_tests)
    fate_tests["p_holm"] = holm(fate_tests.p)
    marker_tests = pd.DataFrame(marker_tests)
    marker_tests["p_holm"] = holm(marker_tests.p)
    signal_frame = pd.DataFrame(signals)
    fate_tests.to_csv(FOLDER / "branch_fate_tests.csv", index=False)
    marker_tests.to_csv(FOLDER / "branch_marker_program_tests.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(FOLDER / "branch_baseline_records.csv", index=False)
    (FOLDER / "branch_full_gene_signal.json").write_text(json.dumps(signals, indent=2), encoding="utf-8")
    library_rows = []
    for (clone, library), rows in late[late.clone_index.isin(eligible)].groupby(["clone_index", "Library"]):
        record = {"clone_index": int(clone), "Library": library, "Well": int(rows.Well.iloc[0]), "n_cells": len(rows)}
        for fate in fate_names:
            record[f"n_{fate}"] = int((rows["Cell type annotation"] == fate).sum())
        for name, values in program.items():
            record[f"mean_{name}_program"] = float(values[rows.index].mean())
        library_rows.append(record)
    pd.DataFrame(library_rows).to_csv(FOLDER / "branch_library_replication.csv", index=False)
    other_controls = []
    for library, rows in late[late.clone_index != 1978].groupby("Library"):
        mono = rows[rows["Cell type annotation"] == "Monocyte"]
        other_controls.append({"Library": library, "n_other_clone_cells": len(rows), "n_other_clone_monocytes": len(mono),
                               "mean_monocyte_program_in_annotated_monocytes": None if len(mono) == 0 else float(program["Monocyte"][mono.index].mean())})
    pd.DataFrame(other_controls).to_csv(FOLDER / "branch_marker_detection_controls.csv", index=False)
    result = {
        "n_common_clones_after_culture_family_screen": int(meta.clone_index.nunique()),
        "potential_repeated_cell_screen": duplicate_audit,
        "n_eligible_clones": len(eligible), "minimum_day6_cells_per_well": 10,
        "n_fate_tests": len(fate_tests), "n_fate_tests_Holm_below_0_05": int((fate_tests.p_holm < .05).sum()),
        "n_clones_with_fate_test_Holm_below_0_05": int(fate_tests.loc[fate_tests.p_holm < .05, "clone_index"].nunique()),
        "n_marker_program_tests": len(marker_tests), "n_marker_program_tests_Holm_below_0_05": int((marker_tests.p_holm < .05).sum()),
        "marker_source": "Published Table 2, https://pmc.ncbi.nlm.nih.gov/articles/PMC7608074/",
        "full_gene_output": "Mean per-cell log1p of published normalized counts across all 25289 genes",
        "full_gene_signal_statistic": "dot(mean RNA well1 library A - mean RNA well2 library A, mean RNA well1 library B - mean RNA well2 library B) / 4",
        "statistic_interpretation": "Under unbiased independent library sampling, estimates one quarter of the squared difference between the two realized branch population profiles. Any common deterministic prediction has at least this paired mean squared error against the two true profiles.",
        "strongest_fate_results": fate_tests.sort_values("p").head(12).to_dict("records"),
        "strongest_marker_results": marker_tests.sort_values("p").head(12).to_dict("records"),
        "clone1978_signal": next(s for s in signals if s["clone_index"] == 1978),
        "proof_scope": "A shared experimentally recorded early clonal RNA input cannot specify a unique realized culture branch outcome at day 6 when branch population profiles genuinely differ. This is an operational limit of the available antecedent measurement.",
        "not_established": ["Insufficiency of ideal noiseless RNA from the individual daughter cell", "Population Bayes-risk ceiling for all contexts", "Inability to predict full-gene conditional means/distributions", "Whether biological branching, incompletely sampled sister-cell states, or unmeasured culture conditions cause the ambiguity"],
        "assumptions_and_limits": ["Early record sampled before partition, as stated in original experiment design", "Nominal treatment/time matched within culture family", "Barcode links correspond to real common source; known cross-family collisions excluded", "Sequenced cells represent realized culture populations", "Library contrasts are not entirely due to well-associated technical artifacts", "Conditional bootstrap quantifies sampled-cell uncertainty, not independent biological replication across new experiments", "Cell-type labels and marker scores derive from the same RNA measurements and are not independent assay confirmations"]
    }
    (FOLDER / "branch_validation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    chosen = late[late.clone_index == 1978]
    labels = sorted(chosen.Library.unique())
    for j, fate in enumerate(["Neutrophil", "Monocyte", "Undifferentiated"]):
        fraction = [float((chosen.loc[chosen.Library == lib, "Cell type annotation"] == fate).mean()) for lib in labels]
        bottom = np.zeros(4) if j == 0 else bottom + previous
        axes[0].bar(np.arange(4), fraction, bottom=bottom, color=["#16827c", "#d8753e", "#c7cbd2"][j], label=fate)
        previous = np.array(fraction)
    axes[0].set_xticks(np.arange(4), ["W1 / L1", "W1 / L2", "W2 / L1", "W2 / L2"], rotation=25)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Observed cell fraction")
    axes[0].set_title("Same early clonal record, different endpoints")
    axes[0].legend(fontsize=8, loc="lower left")
    values = program["Monocyte"]
    for k, lib in enumerate(labels):
        rows = chosen[chosen.Library == lib].index
        jitter = np.random.default_rng(k).uniform(-.12, .12, len(rows))
        axes[1].scatter(k + jitter, values[rows], color="#d8753e" if k < 2 else "#16827c", alpha=.7, s=20)
        axes[1].plot([k - .18, k + .18], [values[rows].mean()] * 2, color="black", lw=2)
    axes[1].set_xticks(np.arange(4), ["W1 / L1", "W1 / L2", "W2 / L1", "W2 / L2"], rotation=25)
    axes[1].set_ylabel("Mean log1p RNA, 10 published markers")
    axes[1].set_title("Monocyte RNA program / clone 1978")
    top = sorted(signals, key=lambda r: r["estimate"], reverse=True)
    for j, row in enumerate(top):
        low, high = row["conditional_bootstrap_CI95"]
        axes[2].plot([j, j], [low, high], color="#b1b8c0", lw=1)
        axes[2].scatter(j, row["estimate"], color="#d8753e" if row["clone_index"] == 1978 else "#354b66", s=24)
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set_xlabel(f"{len(top)} clones, ordered by observed estimate")
    axes[2].set_ylabel("Paired full-gene error bound estimate")
    axes[2].set_title("Cross-library full-gene contrast")
    fig.text(.01, .01, "Day-2 common-source RNA → day-6 split cultures. Intervals quantify sampled-cell uncertainty within these cultures.", fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, 1])
    fig.savefig(FOLDER / "branch_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
