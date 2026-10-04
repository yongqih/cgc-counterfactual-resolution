"""Audit frozen CellTag predictions and tabulate specificity at two criteria."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CT=HERE.parent/'celltag_information_resolution_20261001'
OUT=CT/'outputs_six_resolution'
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    inputs=[OUT/'predictions.csv',OUT/'fit_diagnostics.json',OUT/'summary.csv',OUT/'contrasts.csv',
            CT/'dataset_early_onecell_six.npz']
    hashes={str(p.relative_to(CT)):SHA(p) for p in inputs}
    d=pd.read_csv(inputs[0]); fits=json.loads(inputs[1].read_text())
    z=np.load(inputs[4]); labels=dict(zip(z['clone_ids'],z['y']))
    assert len(d)==4950 and len(fits)==150
    assert d.groupby(['model','condition','clone_id']).size().eq(3).all()
    assert (d.y==d.clone_id.map(labels)).all()
    checks=0; splitmaps={}
    for f in fits:
        tr,ca,te=[set(f[k]) for k in ['train_ids','cal_ids','test_ids']]
        assert not(tr&ca or tr&te or ca&te)
        assert len(tr)==99 and len(ca)==33 and len(te)==33 and tr|ca|te==set(labels)
        key=f['rep'],f['fold']
        split=(tr,ca,te)
        if key in splitmaps:assert split==splitmaps[key]
        splitmaps[key]=split
        q=d[(d.rep==f['rep'])&(d.fold==f['fold'])&(d.model==f['model'])&(d.condition==f['condition'])]
        assert set(q.clone_id)==te
        p=q[[f'p{i}' for i in range(6)]].to_numpy()
        assert np.allclose(p.sum(axis=1),1,atol=1e-12)
        truth=q.y.to_numpy()
        assert np.allclose(q.brier,((p-np.eye(6)[truth])**2).sum(axis=1)/2)
        assert np.allclose(q.logloss,-np.log(np.clip(p[np.arange(len(q)),truth],1e-15,1)))
        if f['condition']=='PRIOR':
            prior=np.bincount([labels[i] for i in tr],minlength=6)/99
            assert np.allclose(p,prior)
        for tag,cut in f['quantiles'].items():
            pred=(1-p)<=np.asarray(cut)+1e-12
            empty=~pred.any(axis=1);pred[empty]=True
            assert np.array_equal(pred.sum(axis=1),q[f'size_{tag}'])
            assert np.array_equal(pred[np.arange(len(q)),truth],q[f'covered_{tag}'])
            assert np.array_equal(empty,q[f'empty_replaced_{tag}'])
            checks+=len(q)
    old=pd.read_csv(OUT/'summary.csv').set_index(['model','condition'])
    new=d.groupby(['model','condition'])[old.columns.tolist()].mean()
    assert np.allclose(old.sort_index(),new.sort_index(),atol=1e-12)
    # Existing paired contrasts and their intervals are preserved verbatim.
    contrast=pd.read_csv(OUT/'contrasts.csv')
    for _,r in contrast.iterrows():
        a=d[(d.model==r.model)&(d.condition==r.a)].groupby('clone_id')[r.metric].mean()
        b=d[(d.model==r.model)&(d.condition==r.b)].groupby('clone_id')[r.metric].mean()
        assert np.isclose((a-b).mean(),r.delta,atol=1e-12)
    rows=[]; prior=[]
    rng=np.random.default_rng(20261003)
    for model in ['LR','RF']:
        boot=rng.integers(165,size=(10000,165))
        for condition in ['PRIOR','RNA','ATAC','RNA_ATAC','RNA_unrelated_ATAC']:
            g=d[(d.model==model)&(d.condition==condition)].groupby('clone_id').mean(numeric_only=True).sort_index()
            for level in [80,90]:
                rec={'model':model,'condition':condition,'nominal_coverage':level/100,'n_clones':165,'n_repeats':3}
                for name,col in [('set_size',f'size_{level}'),('coverage',f'covered_{level}')]:
                    a=g[col].to_numpy();lo,hi=np.quantile(a[boot].mean(axis=1),[.025,.975])
                    rec.update({name:float(a.mean()),name+'_ci95_low':lo,name+'_ci95_high':hi})
                rows.append(rec)
            if condition!='PRIOR':
                base=d[(d.model==model)&(d.condition=='PRIOR')].groupby('clone_id').mean(numeric_only=True).sort_index()
                for level in [80,90]:
                    for metric in [f'size_{level}',f'covered_{level}']:
                        a=(g[metric]-base[metric]).to_numpy();lo,hi=np.quantile(a[boot].mean(axis=1),[.025,.975])
                        prior.append({'model':model,'a':condition,'b':'PRIOR','metric':metric,'delta':a.mean(),'ci95_low':lo,'ci95_high':hi})
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'celltag_specificity.csv',index=False)
    pd.concat([contrast,pd.DataFrame(prior)],ignore_index=True).to_csv(HERE/'celltag_paired_contrasts.csv',index=False)
    report={'input_sha256':hashes,'prediction_rows':len(d),'fits_audited':len(fits),'set_reconstructions':checks,
            'all_split_and_probability_checks_pass':True,'frozen_summary_reproduced':True,
            'existing_contrasts_reproduced_and_CIs_preserved':True,'models_retrained':0,
            'uncertainty':'Paired clone bootstrap after averaging three repeated held-out predictions; conditional on fitted models and splits.',
            'limits':['Nominal marginal calibration, with empirical coverage reported; not per-clone or per-class coverage.',
                      'Same nominal criterion does not mean exactly equal realized coverage.',
                      'No formal equivalence or modality synergy follows from similar ATAC and joint results.',
                      'Set size is task-specific remaining ambiguity; lower is more specific.']}
    assert hashes=={str(p.relative_to(CT)):SHA(p) for p in inputs}
    (HERE/'celltag_reuse_audit.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False))
    print(contrast[contrast.metric.isin(['size_80','size_90'])].to_string(index=False))

if __name__=='__main__':main()
