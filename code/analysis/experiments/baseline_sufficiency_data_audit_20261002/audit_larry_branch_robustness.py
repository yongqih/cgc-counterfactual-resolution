"""Check endpoint contrasts without an outcome-selected test family.

All tests are conditional on observed cultures and sampled cells. Library
controls are sensitivity analyses, not extra biological culture replicates.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import fisher_exact
from validate_larry_branches import holm, cross_library_signal, exclude_ambiguous_late_records

ROOT = Path(__file__).resolve().parent / 'larry'


def main():
    meta = pd.read_csv(ROOT / 'branch_validation_metadata.csv')
    x = load_npz(ROOT / 'branch_validation_full_gene.npz').astype(np.float64)
    meta, x, duplicate_audit = exclude_ambiguous_late_records(meta, x)
    x.data = np.log1p(x.data)
    late = meta[meta['Time point'] == 6]
    fates = sorted(pd.read_csv(ROOT / 'stateFate_inVitro_metadata.txt.gz', sep='\t')['Cell type annotation'].unique())
    tests = []
    for clone, cells in late.groupby('clone_index'):
        totals = [int((cells.Well == w).sum()) for w in (1, 2)]
        for fate in fates:
            hits = [int(((cells.Well == w) & (cells['Cell type annotation'] == fate)).sum()) for w in (1, 2)]
            _, p = fisher_exact([[hits[0], totals[0]-hits[0]], [hits[1], totals[1]-hits[1]]])
            tests.append({'clone_index': int(clone), 'fate': fate, 'p': float(p),
                          'n_well1': totals[0], 'n_well2': totals[1],
                          'hits_well1': hits[0], 'hits_well2': hits[1]})
    frame = pd.DataFrame(tests)
    frame['p_holm_all_207_clones'] = holm(frame.p)
    frame.to_csv(ROOT / 'branch_fate_tests_all_207_clones.csv', index=False)

    sensitivities = []
    rng = np.random.default_rng(20261003)
    for clone in (1978, 1261):
        selected = (meta.clone_index == clone) & (meta['Time point'] == 6)
        local = meta.loc[selected].reset_index(drop=True)
        raw = x[selected.to_numpy()].toarray()
        corrected = raw.copy()
        references = []
        for library in sorted(local.Library.unique()):
            target = (local.Library == library).to_numpy()
            control = ((meta.Library == library) & (meta.clone_index != clone)
                       & (meta['Cell type annotation'] == 'Neutrophil')).to_numpy()
            assert control.sum() >= 20
            corrected[target] -= np.asarray(x[control].mean(axis=0))
            references.append({'library': library, 'n_other_clone_neutrophils': int(control.sum())})
        from scipy.sparse import csr_matrix
        signal = cross_library_signal(csr_matrix(corrected), local, rng)
        sensitivities.append({'clone_index': clone, 'reference_cell_type': 'Neutrophil',
                              'correction': 'Subtract the same-library other-clone mean neutrophil profile',
                              'interpretation': 'Sensitivity to library-associated shifts common to sampled neutrophils; this reference need not be biologically invariant.',
                              'controls': references, **signal})

    # Stronger cell-identity check: exclude any droplet barcode reused in the
    # same family/day/well in the FULL metadata, even if its clone differs.
    original = pd.read_csv(ROOT / 'stateFate_inVitro_metadata.txt.gz', sep='\t')
    from prepare_larry_branch_validation import family
    original['culture_family'] = original.Library.map(family)
    original['original_cell_index'] = np.arange(len(original))
    suspect = original.duplicated(['culture_family', 'Time point', 'Well', 'Cell barcode'], keep=False)
    suspect_ids = set(original.loc[suspect, 'original_cell_index'])
    selected = ((meta.clone_index == 1978) & (meta['Time point'] == 6)
                & ~meta.original_cell_index.isin(suspect_ids))
    stricter = meta.loc[selected].reset_index(drop=True)
    totals = [int((stricter.Well == w).sum()) for w in (1, 2)]
    hits = [int(((stricter.Well == w) & (stricter['Cell type annotation'] == 'Neutrophil')).sum()) for w in (1, 2)]
    _, p = fisher_exact([[hits[0], totals[0]-hits[0]], [hits[1], totals[1]-hits[1]]])
    stronger_identity = {'clone_index': 1978, 'screen': 'Exclude any same-family/day/well droplet barcode reuse across the full metadata, including different clones and unlabelled cells.',
                         'n_well1': totals[0], 'n_well2': totals[1],
                         'Neutrophil_well1': hits[0], 'Neutrophil_well2': hits[1],
                         'Neutrophil_p': float(p), 'Neutrophil_p_Bonferroni_2277_tests': min(1, float(p)*len(frame)),
                         'full_gene_signal': cross_library_signal(x[selected.to_numpy()], stricter, rng)}

    result = {'n_tested_clones': int(late.clone_index.nunique()), 'n_tests': len(frame),
              'potential_repeated_cell_screen': duplicate_audit,
              'all_207_clone_family_significant_results': frame[frame.p_holm_all_207_clones < .05].sort_values('p').to_dict('records'),
              'reference_corrected_full_gene_sensitivity': sensitivities,
              'stronger_droplet_barcode_screen': stronger_identity,
              'clone1978_early_records': meta[(meta.clone_index == 1978) & (meta['Time point'] == 2)].to_dict('records'),
              'limits': ['Correction is a post-hoc artifact check, not a deployment input.',
                         'Reference profiles can themselves differ biologically.',
                         'Confidence intervals condition on observed sampled cultures.',
                         'Bootstrap treats reference means as fixed; it omits reference-estimation uncertainty.',
                         'A well-specific biological effect is information absent from the recorded pre-split input.',
                         'A well-specific measurement artifact would not establish a true biological endpoint contrast.']}
    (ROOT / 'branch_robustness_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
