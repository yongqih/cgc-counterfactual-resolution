"""Early RNA counts only, with no integration against later cell states."""
from pathlib import Path
import json
import hashlib
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import argparse

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--six',action='store_true')
args=ap.parse_args()
suffix='_six' if args.six else ''
base=np.load(ROOT/f'dataset_tf_object{suffix}.npz')
ids=base['clone_ids']
meta=pd.read_csv(ROOT/'data/clone_table.csv').set_index('cell.bc')
# Keep exactly the original analysis' cells and clones.
rna_export=ROOT/'data/authors/85d6689d256ec2b18f87.csv'
rids=pd.read_csv(rna_export,usecols=[0]).iloc[:,0].unique()
rids=[s for s in rids if meta.loc[s,'clone.id'] in ids]
path=ROOT/'data/day2_rna_counts.h5'
with h5py.File(path,'r') as f:
    a=f['matrix']
    counts=csc_matrix((a['data'][:],a['indices'][:],a['indptr'][:]),shape=tuple(a['shape'][:])).T.tocsr()
    barcodes=['d2_5-rna-'+s.decode() for s in a['barcodes'][:]]
    features=np.array([s.decode() for s in a['features/name'][:]])
    lookup={s:i for i,s in enumerate(barcodes)}
    assert all(s in lookup for s in rids)
    x=counts[[lookup[s] for s in rids]].astype(float)
    totals=np.asarray(x.sum(axis=1)).ravel()
    assert (totals>0).all()
    x=x.multiply((1e4/totals)[:,None]).tocsr()
    x.data=np.log1p(x.data)
    rdf=pd.DataFrame(x.toarray(),index=rids)
    rdf['clone.id']=meta.loc[rids,'clone.id'].to_numpy()
    grouped=rdf.groupby('clone.id').mean().loc[ids].to_numpy(np.float32)
np.savez_compressed(ROOT/f'dataset_early_counts{suffix}.npz',clone_ids=ids,y=base['y'],rna=grouped,atac=base['atac'],
                    rna_features=features,atac_features=base['atac_features'])
if not args.six:
    cohort=pd.read_csv(ROOT/'cohort.csv').set_index('clone_id').loc[ids]
    one_rna=rdf.loc[cohort.rna_example_cell].drop(columns='clone.id').to_numpy(np.float32)
    from scipy.sparse import csr_matrix
    with h5py.File(ROOT/'data/authors/bcdc06c9c1b2e6ce42d3.h5ad','r') as f:
        a=f['X'];matrix=csr_matrix((a['data'][:],a['indices'][:],a['indptr'][:]),shape=tuple(a.attrs['shape']))
        obs=[v.decode().replace('-ATAC-','-atac-') for v in f['obs/_index'][:]]
        var=[v.decode().replace('_','-') for v in f['var/_index'][:]]
        rows={s:i for i,s in enumerate(obs)};cols={s:i for i,s in enumerate(var)}
        one_atac=matrix[[rows[s] for s in cohort.atac_example_cell]][:,[cols[s.replace('_','-')] for s in base['atac_features']]].toarray()
    np.savez_compressed(ROOT/'dataset_early_onecell.npz',clone_ids=ids,y=base['y'],rna=one_rna,atac=one_atac,
                        rna_features=features,atac_features=base['atac_features'])
else:
    # Select cell identities with a seed specific to each clone and modality;
    # this selection uses no fate labels and is invariant to cohort order.
    atac_rids=pd.read_csv(ROOT/'data/authors/912a3570ed91642988ac.csv',usecols=[0]).iloc[:,0].unique()
    choice_r=[];choice_a=[]
    for clone in ids:
        rpool=sorted([s for s in rids if meta.loc[s,'clone.id']==clone])
        apool=sorted([s for s in atac_rids if meta.loc[s,'clone.id']==clone])
        choice_r.append(np.random.default_rng(20261001+int(clone)*2).choice(rpool))
        choice_a.append(np.random.default_rng(20261002+int(clone)*2).choice(apool))
    one_rna=rdf.loc[choice_r].drop(columns='clone.id').to_numpy(np.float32)
    from scipy.sparse import csr_matrix
    with h5py.File(ROOT/'data/authors/bcdc06c9c1b2e6ce42d3.h5ad','r') as f:
        a=f['X'];matrix=csr_matrix((a['data'][:],a['indices'][:],a['indptr'][:]),shape=tuple(a.attrs['shape']))
        rows={v.decode().replace('-ATAC-','-atac-'):i for i,v in enumerate(f['obs/_index'][:])}
        cols={v.decode().replace('_','-'):i for i,v in enumerate(f['var/_index'][:])}
        one_atac=matrix[[rows[s] for s in choice_a]][:,[cols[s.replace('_','-')] for s in base['atac_features']]].toarray()
    np.savez_compressed(ROOT/'dataset_early_onecell_six.npz',clone_ids=ids,y=base['y'],rna=one_rna,atac=one_atac,
                        rna_features=features,atac_features=base['atac_features'])
    pd.DataFrame({'clone_id':ids,'rna_cell':choice_r,'atac_cell':choice_a,'y':base['y']}).to_csv(ROOT/'six_onecell_cohort.csv',index=False)
audit={'RNA':'GSM6681127 early day-2.5 raw counts; per-cell library normalization 1e4 and log1p; within-clone mean; 3000 variable genes selected only inside training folds',
       'ATAC':'Early rows of deposited lsk_tf_mtx.h5ad X/raw.X; no authors kNN-imputed ML CSV. Motif/peak reference remains the authors preprocessing.',
       'clones':len(ids),'rna_cells':len(rids),'rna_genes':len(features),'rna_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'reason':'Predeclared feature provenance check: remove dependence on integrated RNA and smoothed ATAC representations. Same clones and validation partitions as the processed-feature analysis.'}
(ROOT/f'early_counts_audit{suffix}.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
