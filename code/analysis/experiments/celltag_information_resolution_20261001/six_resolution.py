"""All-six-fate information/resolution test with one early cell per assay.

The same real observations are used throughout; random pairing is solely a
negative control. No eventual-fate filter restricts this task to two classes.
"""
from analyze import ROOT,SEED,specification,prediction_sets
from check_six_fates import proba,scorer,K
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from threadpoolctl import threadpool_limits

def main():
    out=ROOT/'outputs_six_resolution';out.mkdir(exist_ok=True)
    z=np.load(ROOT/'dataset_early_onecell_six.npz')
    xr,xa,y,ids=z['rna'],z['atac'],z['y'],z['clone_ids']
    assert len(ids)==len(np.unique(ids))==165
    warnings.filterwarnings('ignore',message='The least populated class')
    warnings.filterwarnings('ignore',message='The y_prob values do not sum to one')
    rows=[];diagnostics=[]
    for rep in range(3):
        splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED+rep).split(xr,y))
        for fold in range(5):
            te=splits[fold][1];ca=splits[(fold+1)%5][1]
            tr=np.setdiff1d(np.arange(len(ids)),np.r_[te,ca])
            rng=np.random.default_rng(SEED+rep*100+fold)
            blocks=[tr,ca,te];shuffled=[rng.permutation(b) for b in blocks]
            for family in ['LR','RF']:
                for cond in ['RNA','ATAC','RNA_ATAC','RNA_unrelated_ATAC','PRIOR']:
                    if cond=='RNA':inputs=[xr[b] for b in blocks]
                    elif cond=='ATAC':inputs=[xa[b] for b in blocks]
                    elif cond=='RNA_ATAC':inputs=[np.c_[xr[b],xa[b]] for b in blocks]
                    elif cond=='RNA_unrelated_ATAC':inputs=[np.c_[xr[b],xa[s]] for b,s in zip(blocks,shuffled)]
                    if cond=='PRIOR':
                        prior=np.bincount(y[tr],minlength=K)/len(tr)
                        pc=np.tile(prior,(len(ca),1));pt=np.tile(prior,(len(te),1));params={}
                    else:
                        nr=0 if cond=='ATAC' else xr.shape[1]
                        model,grid=specification(family,SEED+100*rep+fold,nr)
                        search=GridSearchCV(model,grid,cv=StratifiedKFold(3,shuffle=True,random_state=SEED+fold),scoring=scorer,n_jobs=4,error_score='raise')
                        with threadpool_limits(limits=1):
                            search.fit(inputs[0],y[tr]);pc=proba(search,inputs[1]);pt=proba(search,inputs[2])
                        pc/=pc.sum(axis=1,keepdims=True);pt/=pt.sum(axis=1,keepdims=True)
                        params=search.best_params_
                    sets={}
                    for a in [.1,.2]:
                        for cc in [False,True]:
                            tag=('cc' if cc else '')+str(int((1-a)*100))
                            sets[tag]=prediction_sets(pc,y[ca],pt,a,cc)
                    for j,i in enumerate(te):
                        row={'clone_id':int(ids[i]),'y':int(y[i]),'rep':rep,'fold':fold,'model':family,'condition':cond,
                             'brier':float(np.sum((pt[j]-np.eye(K)[y[i]])**2)/2),
                             'logloss':float(-np.log(np.clip(pt[j,y[i]],1e-15,1))),
                             'correct':int(np.argmax(pt[j])==y[i]),**{f'p{k}':float(pt[j,k]) for k in range(K)}}
                        for tag,(s,q,empty) in sets.items():
                            row.update({f'size_{tag}':int(s[j].sum()),f'covered_{tag}':int(s[j,y[i]]),
                                        f'singleton_{tag}':int(s[j].sum()==1),f'empty_replaced_{tag}':int(empty[j])})
                        rows.append(row)
                    diagnostics.append({'rep':rep,'fold':fold,'model':family,'condition':cond,'params':params,
                        'train_ids':ids[tr].tolist(),'cal_ids':ids[ca].tolist(),'test_ids':ids[te].tolist(),
                        'quantiles':{tag:v[1] for tag,v in sets.items()},'cal_label_counts':np.bincount(y[ca],minlength=K).tolist()})
                    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
                    (out/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
                print(rep,fold,family,'complete',flush=True)
    d=pd.DataFrame(rows)
    metrics=['brier','logloss','correct']+[f'{m}_{s}' for m in ['size','covered','singleton'] for s in ['90','80','cc90','cc80']]
    summary=d.groupby(['model','condition'])[metrics].mean();summary.to_csv(out/'summary.csv')
    d.groupby(['model','condition','y'])[metrics].mean().to_csv(out/'class_summary.csv')
    rng=np.random.default_rng(SEED);contrasts=[]
    for model,g in d.groupby('model'):
        for a,b in [('RNA_ATAC','RNA'),('RNA_ATAC','ATAC'),('RNA_ATAC','RNA_unrelated_ATAC')]:
            aa=g[g.condition==a].groupby('clone_id').mean(numeric_only=True)
            bb=g[g.condition==b].groupby('clone_id').mean(numeric_only=True)
            boot=rng.integers(len(aa),size=(10000,len(aa)))
            for metric in metrics:
                delta=(aa[metric]-bb[metric]).to_numpy()
                lo,hi=np.quantile(delta[boot].mean(axis=1),[.025,.975])
                contrasts.append({'model':model,'a':a,'b':b,'metric':metric,'delta':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    print(summary.round(4).to_string(),flush=True)

if __name__=='__main__':main()
