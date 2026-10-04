"""Recover observed early-cell features; audit repeated CV exports and pairing."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parent
inventory=json.loads((ROOT/'data/archive_inventory.json').read_text())
def locate(needle):
    matches=[x for x in inventory if needle in x['name'] and 'local_path' in x]
    assert len(matches)==1,(needle,[x['name'] for x in matches])
    return ROOT/matches[0]['local_path']

def read_export(needle):
    path=locate(needle)
    full=pd.read_csv(path,index_col=0)
    unique=full[~full.index.duplicated()].copy()
    assert all(str(i).startswith('d2_5-') for i in unique.index)
    reconstructed=unique.loc[full.index].to_numpy()
    assert np.allclose(full.to_numpy(),reconstructed,rtol=0,atol=1e-12,equal_nan=True)
    print(path.name,'export rows',len(full),'unique early cells/pairs',len(unique),'features',full.shape[1],flush=True)
    return full,unique

def main():
    meta=pd.read_csv(ROOT/'data/clone_table.csv').set_index('cell.bc',drop=False)
    rfull,rna=read_export('/test_data_mtx/X_test_hsc.rna.r1&2_integrated_data_')
    afull,atac=read_export('/rna_atac/test_data_mtx/X_test_hsc.atac.r1&2_tfact_knnimputed20_all_')
    pfull,pairs=read_export('/test_data_mtx/X_test_hsc.rna&atac.r1&2_')
    audit={'exports':{k:{'rows':len(full),'unique_rows':len(unique),'features':full.shape[1]}
           for k,full,unique in [('RNA',rfull,rna),('ATAC',afull,atac),('paired',pfull,pairs)]}}
    pair_rna=[s.split('_d2_5-atac-')[0] for s in pairs.index]
    pair_atac=['d2_5-atac-'+s.split('_d2_5-atac-')[1] for s in pairs.index]
    cr=meta.loc[pair_rna,'clone.id'].to_numpy()
    ca=meta.loc[pair_atac,'clone.id'].to_numpy()
    assert np.array_equal(cr,ca)
    audit['published_pair_table']={'pairs':len(pairs),'clones':len(set(cr)),
        'rna_cells':len(set(pair_rna)),'atac_cells':len(set(pair_atac)),
        'fate_counts_pairs':meta.loc[pair_rna,'fate'].value_counts().to_dict(),
        'fate_counts_clones':meta.groupby('clone.id').fate.first().loc[np.unique(cr)].value_counts().to_dict()}
    # An audit of potential cell/clone leakage if the published repeated
    # StratifiedKFold were applied to these rows. This is not reconstructed
    # evidence of the exact original folds, whose indices were not deposited.
    from sklearn.model_selection import StratifiedKFold
    pair_y=meta.loc[pair_rna,'fate'].replace({'MPP':'Prog','MPP/GMP':'Prog','MEP':'Prog','Lym':'Lym/DC','pDC':'Lym/DC','Ccr7_DC':'Lym/DC'}).to_numpy()
    fractions=[]
    for seed in range(5):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(pairs)),pair_y):
            fractions.append(float(np.isin(cr[te],cr[tr]).mean()))
    audit['ordinary_row_split_clone_overlap_fraction']={'mean':float(np.mean(fractions)),'min':min(fractions),'max':max(fractions),
        'interpretation':'Design-risk calculation on published paired rows, not verified original fold assignments.'}
    rna.index.name=atac.index.name='cell.bc'
    xr=rna.join(meta[['clone.id','day']],how='inner')
    xa=atac.join(meta[['clone.id','day']],how='inner')
    assert xr.day.eq('d2').all() and xa.day.eq('d2').all()
    gr=xr.drop(columns='day').groupby('clone.id').mean()
    ga=xa.drop(columns='day').groupby('clone.id').mean()
    truth=meta.groupby('clone.id').first()
    common=gr.index.intersection(ga.index)
    ids=np.array(sorted(set(common)&set(truth.index[truth.fate.isin(['Mono','Neutro'])])),dtype=int)
    outcomes=truth.loc[ids]
    for name,df in [('RNA',rna),('ATAC',atac)]:assert np.isfinite(df.to_numpy()).all()
    def save(name,rmat,amat,ids=ids):
        np.savez_compressed(ROOT/name,clone_ids=ids,y=(truth.loc[ids].fate=='Neutro').to_numpy(int),
            rna=rmat.astype(np.float32),atac=amat.astype(np.float32),
            rna_features=rna.columns.to_numpy(str),atac_features=atac.columns.to_numpy(str))
    save('dataset.npz',gr.loc[ids].to_numpy(),ga.loc[ids].to_numpy())
    # Fixed, fate-blind one-cell-per-modality selection.
    rng=np.random.default_rng(20261001)
    rid=[rng.choice(xr.index[xr['clone.id']==i]) for i in ids]
    aid=[rng.choice(xa.index[xa['clone.id']==i]) for i in ids]
    save('dataset_onecell.npz',rna.loc[rid].to_numpy(),atac.loc[aid].to_numpy())
    # Read the separately deposited chromVAR object. Whether these are
    # unimputed scores is checked by comparison with the ML CSV and metadata.
    with h5py.File(locate('/lsk_tf_mtx.h5ad'),'r') as f:
        a=f['X'];mat=csr_matrix((a['data'][:],a['indices'][:],a['indptr'][:]),shape=tuple(a.attrs['shape']))
        obs=np.array([v.decode().replace('-ATAC-','-atac-') for v in f['obs/_index'][:]])
        var=np.array([v.decode() for v in f['var/_index'][:]])
        colmap={s:j for j,s in enumerate(var)}
        audit['tf_object']={'shape':list(mat.shape),'features_sample':var[:5].tolist(),'csv_feature_sample':atac.columns[:5].tolist()}
        # Harmonize only dash/underscore separators, preserving motif IDs.
        canonical=lambda s:s.replace('_','-')
        cm={canonical(s):j for j,s in enumerate(var)}
        j=[cm[canonical(s)] for s in atac.columns]
        idx={s:i for i,s in enumerate(obs)}
        ar=mat[[idx[s] for s in atac.index]][:,j].toarray()
        audit['tf_object']['difference_from_imputed_csv_max']=float(np.max(np.abs(ar-atac.to_numpy())))
        rawdf=pd.DataFrame(ar,index=atac.index,columns=atac.columns)
        rawgroup=rawdf.join(meta[['clone.id']]).groupby('clone.id').mean()
        save('dataset_tf_object.npz',gr.loc[ids].to_numpy(),rawgroup.loc[ids].to_numpy())
        fate_map={'Mono':0,'Neutro':1,'MPP':2,'MPP/GMP':2,'MEP':2,'Ery/Meg':3,'Baso/Eos/Mast':4,'Lym':5,'pDC':5,'Ccr7_DC':5}
        all_ids=np.array(sorted(common),dtype=int)
        all_y=truth.loc[all_ids].fate.map(fate_map)
        assert all_y.notna().all()
        np.savez_compressed(ROOT/'dataset_six.npz',clone_ids=all_ids,y=all_y.to_numpy(int),
            rna=gr.loc[all_ids].to_numpy(np.float32),atac=ga.loc[all_ids].to_numpy(np.float32),
            rna_features=rna.columns.to_numpy(str),atac_features=atac.columns.to_numpy(str))
        np.savez_compressed(ROOT/'dataset_tf_object_six.npz',clone_ids=all_ids,y=all_y.to_numpy(int),
            rna=gr.loc[all_ids].to_numpy(np.float32),atac=rawgroup.loc[all_ids].to_numpy(np.float32),
            rna_features=rna.columns.to_numpy(str),atac_features=atac.columns.to_numpy(str))
    cohort=outcomes[['fate','fate_pct','# of D5 cells (RNA & ATAC)']].copy()
    cohort['n_early_rna']=xr.groupby('clone.id').size().loc[ids]
    cohort['n_early_atac']=xa.groupby('clone.id').size().loc[ids]
    cohort['rna_example_cell']=rid;cohort['atac_example_cell']=aid
    cohort.to_csv(ROOT/'cohort.csv',index_label='clone_id')
    audit['analysis_cohort']={'n':len(ids),'fates':outcomes.fate.value_counts().to_dict(),
       'common_all_fate_clones':len(common),'common_all_fates':truth.loc[common].fate.value_counts().to_dict(),
       'n_rna_genes':rna.shape[1],'n_atac_motifs':atac.shape[1]}
    (ROOT/'feature_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
