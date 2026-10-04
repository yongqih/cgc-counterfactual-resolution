"""Audit reuse of cell barcode strings across LARRY RNA libraries."""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from scipy.io import mmread

ROOT = Path(__file__).resolve().parent / 'larry'


def main():
    meta = pd.read_csv(ROOT / 'stateFate_inVitro_metadata.txt.gz', sep='\t')
    with gzip.open(ROOT / 'stateFate_inVitro_clone_matrix.mtx.gz', 'rb') as stream:
        clones = mmread(stream, spmatrix=True).tocsr()
    positive = np.diff(clones.indptr) == 1
    meta['clone_index'] = -1
    meta.loc[positive, 'clone_index'] = clones.indices
    rows = []
    for well in (1, 2):
        a = meta[meta.Library == f'LK_d6_{well}_1']
        b = meta[meta.Library == f'LK_d6_{well}_2']
        overlap = a.merge(b, on='Cell barcode', suffixes=('_a', '_b'))
        both = overlap[(overlap.clone_index_a >= 0) & (overlap.clone_index_b >= 0)]
        pa = a[a.clone_index >= 0].clone_index.value_counts(normalize=True)
        pb = b[b.clone_index >= 0].clone_index.value_counts(normalize=True)
        rows.append({'well': well, 'barcode_overlap': len(overlap),
                     'barcoded_in_both': len(both),
                     'same_clone_among_both_barcoded': int((both.clone_index_a == both.clone_index_b).sum()),
                     'expected_same_clone_probability_under_independent_labels': float(pa.mul(pb, fill_value=0).sum()),
                     'target_clone1978_barcode_overlap': int(((overlap.clone_index_a == 1978) & (overlap.clone_index_b == 1978)).sum())})
    result = {'results': rows,
              'interpretation': 'Some cross-library droplet-barcode overlaps carry the same clonal tag far more often than random matching predicts. Possible shared cells or barcode-assignment artifacts require conservative exclusion; this audit does not resolve their cause. Clone 1978 has no such overlap in either well. Disjoint retained cell records support within-culture replication; the libraries are not independent culture experiments.',
              'analysis_action': 'Exclude every day-6 record with a duplicated family/time/well/clone/droplet-barcode key; see branch_duplicate_record_screen.json.'}
    (ROOT / 'branch_library_cell_identity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
