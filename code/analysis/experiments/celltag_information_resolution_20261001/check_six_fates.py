"""Task-scope sensitivity: all six published clonal fate groups.

Uses 80% of clones for training per outer split, with nested model selection.
Tiny rare-fate samples preclude a reliable class-conditional resolution claim.
"""
from analyze import ROOT, SEED, specification
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

K=6
def proba(model,x):
    p=np.zeros((len(x),K))
    p[:,model.classes_.astype(int)]=model.predict_proba(x)
    return p
def scorer(model,x,y):return -log_loss(y,proba(model,x),labels=np.arange(K))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='dataset_six.npz')
    ap.add_argument('--out',default='outputs_six')
    ap.add_argument('--select-rna',action='store_true')
    args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=True)
    z=np.load(ROOT/args.dataset)
    xr,xa,y,ids=z['rna'],z['atac'],z['y'],z['clone_ids']
    print('Six-fate cohort',len(ids),'counts',np.bincount(y,minlength=K),flush=True)
    warnings.filterwarnings('ignore',message='The least populated class')
    rows=[]
    splits=[]
    for rep in range(3):
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED+rep).split(xr,y)):
            assert not set(ids[tr])&set(ids[te])
            rng=np.random.default_rng(SEED+100*rep+fold)
            shtr,shte=rng.permutation(tr),rng.permutation(te)
            for family in ['LR','RF']:
                for cond in ['RNA','ATAC','RNA_ATAC','RNA_unrelated_ATAC','PRIOR']:
                    if cond=='RNA':a,b=xr[tr],xr[te]
                    elif cond=='ATAC':a,b=xa[tr],xa[te]
                    elif cond=='RNA_ATAC':a,b=np.c_[xr[tr],xa[tr]],np.c_[xr[te],xa[te]]
                    elif cond=='RNA_unrelated_ATAC':a,b=np.c_[xr[tr],xa[shtr]],np.c_[xr[te],xa[shte]]
                    if cond=='PRIOR':p=np.tile(np.bincount(y[tr],minlength=K)/len(tr),(len(te),1));params={}
                    else:
                        nr=xr.shape[1] if args.select_rna and cond!='ATAC' else 0
                        model,grid=specification(family,SEED+100*rep+fold,nr)
                        search=GridSearchCV(model,grid,cv=StratifiedKFold(3,shuffle=True,random_state=SEED+fold),
                                            scoring=scorer,n_jobs=4,error_score='raise')
                        with threadpool_limits(limits=1):search.fit(a,y[tr]);p=proba(search,b)
                        params=search.best_params_
                    for j,i in enumerate(te):
                        rows.append({'clone_id':int(ids[i]),'y':int(y[i]),'rep':rep,'fold':fold,'model':family,'condition':cond,
                            'logloss':float(-np.log(np.clip(p[j,y[i]],1e-15,1))),
                            'brier':float(np.sum((p[j]-np.eye(K)[y[i]])**2)/2),
                            'correct':int(np.argmax(p[j])==y[i]),**{f'p{k}':float(p[j,k]) for k in range(K)}})
                    splits.append({'rep':rep,'fold':fold,'model':family,'condition':cond,'params':params,
                                   'train_ids':ids[tr].tolist(),'test_ids':ids[te].tolist()})
                    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
                print(rep,fold,family,'done',flush=True)
    d=pd.DataFrame(rows)
    summary=d.groupby(['model','condition'])[['brier','logloss','correct']].mean()
    summary.to_csv(out/'summary.csv')
    (out/'fit_diagnostics.json').write_text(json.dumps(splits,indent=2))
    contrasts=[]
    rng=np.random.default_rng(SEED)
    for model,g in d.groupby('model'):
        for a,b in [('RNA_ATAC','RNA'),('RNA_ATAC','ATAC'),('RNA_ATAC','RNA_unrelated_ATAC')]:
            aa=g[g.condition==a].groupby('clone_id').mean(numeric_only=True)
            bb=g[g.condition==b].groupby('clone_id').mean(numeric_only=True)
            boot=rng.integers(len(aa),size=(10000,len(aa)))
            for metric in ['brier','logloss','correct']:
                delta=(aa[metric]-bb[metric]).to_numpy()
                lo,hi=np.quantile(delta[boot].mean(axis=1),[.025,.975])
                contrasts.append({'model':model,'a':a,'b':b,'metric':metric,'delta':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    print(summary.round(4).to_string(),flush=True)

if __name__=='__main__':main()
