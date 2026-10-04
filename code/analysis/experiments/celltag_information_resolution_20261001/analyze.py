"""Matched-cohort, clone-disjoint state-to-fate prediction audit.

Inputs are prepared separately and must contain only day-2.5 molecular data.
All learned transformations, model selection and set calibration are nested.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 20261001

class RNASelection(TransformerMixin, BaseEstimator):
    def __init__(self, n_rna=0, k=3000):
        self.n_rna=n_rna
        self.k=k
    def fit(self,X,y=None):
        if self.n_rna>self.k:
            variance=np.var(X[:,:self.n_rna],axis=0)
            chosen=np.argsort(variance,kind='stable')[-self.k:]
            self.columns_=np.r_[chosen,np.arange(self.n_rna,X.shape[1])]
        else:self.columns_=np.arange(X.shape[1])
        return self
    def transform(self,X):return X[:,self.columns_]

def specification(family, seed, n_rna=0):
    selection=[('select',RNASelection(n_rna=n_rna))] if n_rna else []
    if family == 'LR':
        return Pipeline(selection+[('scale', StandardScaler()), ('model', LogisticRegression(
            solver='lbfgs', max_iter=1500, random_state=seed))]), {'model__C': [.001, .01, .1, 1, 10]}
    return Pipeline(selection+[('model',RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1))]), {
        'model__min_samples_leaf': [1, 4], 'model__max_features': ['sqrt', .25]}

def fit_predict(family, train, cal, test, y, seed, n_rna=0):
    model, grid = specification(family, seed, n_rna)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(model, grid, scoring='neg_log_loss', cv=inner,
                          n_jobs=4, error_score='raise', refit=True)
    with threadpool_limits(limits=1):
        search.fit(train, y)
        pc = search.predict_proba(cal)
        pt = search.predict_proba(test)
    return pc, pt, search.best_params_, float(search.best_score_)

def prediction_sets(cal_prob, cal_y, test_prob, alpha, class_conditional=False):
    # LAC split conformal; replacement of empty sets with the full set can only
    # increase coverage. Calibration outcomes are never used to fit the model.
    if class_conditional:
        qs=[]
        for label in range(cal_prob.shape[1]):
            scores=1-cal_prob[cal_y==label,label]
            k=int(np.ceil((len(scores)+1)*(1-alpha)))
            qs.append(np.sort(scores)[k-1] if k<=len(scores) else np.inf)
        q=np.asarray(qs)
        sets=(1-test_prob)<=q[None,:]+1e-12
        empty=~sets.any(axis=1)
        sets[empty,:]=True
        return sets,q.tolist(),empty
    scores = 1 - cal_prob[np.arange(len(cal_y)), cal_y]
    k = int(np.ceil((len(scores)+1)*(1-alpha)))
    q = np.sort(scores)[k-1] if k <= len(scores) else np.inf
    sets = (1-test_prob) <= q + 1e-12
    empty = ~sets.any(axis=1)
    sets[empty, :] = True
    return sets, float(q), empty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset.npz')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--families', nargs='+', default=['LR', 'RF'])
    ap.add_argument('--select-rna',action='store_true')
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    data = np.load(ROOT / args.dataset, allow_pickle=False)
    ids, y = data['clone_ids'], data['y']
    xr, xa = data['rna'].astype(np.float32), data['atac'].astype(np.float32)
    assert len(ids) == len(set(ids)) == len(y) == len(xr) == len(xa)
    assert set(np.unique(y)) == {0,1}
    assert np.isfinite(xr).all() and np.isfinite(xa).all()
    config = {'dataset':args.dataset, 'sha256': hashlib.sha256((ROOT/args.dataset).read_bytes()).hexdigest(),
              'n':len(y), 'rna_features':xr.shape[1], 'atac_features':xa.shape[1],
              'labels':{int(k):int(v) for k,v in zip(*np.unique(y, return_counts=True))},
              'repeats':args.repeats, 'seed':SEED, 'fit_fraction':.6, 'cal_fraction':.2, 'test_fraction':.2,
              'alpha':[.1,.2], 'families':args.families,'train_only_rna_variance_selection':args.select_rna}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    all_rows, diagnostics = [], []
    for rep in range(args.repeats):
        folds = list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+rep).split(xr,y))
        for fold in range(5):
            test_idx = folds[fold][1]
            cal_idx = folds[(fold+1)%5][1]
            train_idx = np.setdiff1d(np.arange(len(y)), np.r_[test_idx,cal_idx])
            assert not set(ids[train_idx]) & set(ids[test_idx])
            assert not set(ids[train_idx]) & set(ids[cal_idx])
            assert not set(ids[test_idx]) & set(ids[cal_idx])
            blocks = [train_idx,cal_idx,test_idx]
            rng = np.random.default_rng(SEED + rep*100 + fold)
            shuffle_idx = [rng.permutation(z) for z in blocks]
            for family in args.families:
                for condition in ['RNA','ATAC','RNA_ATAC','RNA_unrelated_ATAC','PRIOR']:
                    seed = SEED + rep*100 + fold
                    if condition == 'RNA': inputs = [xr[z] for z in blocks]
                    elif condition == 'ATAC': inputs = [xa[z] for z in blocks]
                    elif condition == 'RNA_ATAC': inputs = [np.c_[xr[z],xa[z]] for z in blocks]
                    elif condition == 'RNA_unrelated_ATAC':
                        inputs = [np.c_[xr[z],xa[w]] for z,w in zip(blocks,shuffle_idx)]
                    start = time.monotonic()
                    if condition == 'PRIOR':
                        prior = np.bincount(y[train_idx],minlength=2)/len(train_idx)
                        pc,pt = np.tile(prior,(len(cal_idx),1)),np.tile(prior,(len(test_idx),1))
                        params,cv = {},None
                    else:
                        nr=xr.shape[1] if args.select_rna and condition!='ATAC' else 0
                        pc,pt,params,cv=fit_predict(family,*inputs,y[train_idx],seed,nr)
                    assert np.allclose(pt.sum(axis=1),1)
                    sets = {a:prediction_sets(pc,y[cal_idx],pt,a) for a in [.1,.2]}
                    conditional_sets={a:prediction_sets(pc,y[cal_idx],pt,a,True) for a in [.1,.2]}
                    for k,j in enumerate(test_idx):
                        row={'clone_id':int(ids[j]),'y':int(y[j]),'rep':rep,'fold':fold,'model':family,
                             'condition':condition,'p_neutro':float(pt[k,1]),
                             'brier':float((pt[k,1]-y[j])**2),
                             'logloss':float(-np.log(np.clip(pt[k,y[j]],1e-15,1))),
                             'correct':int(np.argmax(pt[k])==y[j])}
                        for a,(s,q,empty) in sets.items():
                            tag=str(int((1-a)*100))
                            row.update({f'size_{tag}':int(s[k].sum()),f'covered_{tag}':int(s[k,y[j]]),
                                        f'singleton_{tag}':int(s[k].sum()==1),f'empty_replaced_{tag}':int(empty[k])})
                        for a,(s,q,empty) in conditional_sets.items():
                            tag='cc'+str(int((1-a)*100))
                            row.update({f'size_{tag}':int(s[k].sum()),f'covered_{tag}':int(s[k,y[j]]),
                                        f'singleton_{tag}':int(s[k].sum()==1),f'empty_replaced_{tag}':int(empty[k])})
                        all_rows.append(row)
                    diagnostics.append({'rep':rep,'fold':fold,'model':family,'condition':condition,
                        'train_ids':ids[train_idx].tolist(),'cal_ids':ids[cal_idx].tolist(),
                        'test_ids':ids[test_idx].tolist(),'params':params,'inner_score':cv,
                        'quantiles':{str(a):x[1] for a,x in sets.items()},
                        'class_conditional_quantiles':{str(a):x[1] for a,x in conditional_sets.items()},'seconds':time.monotonic()-start})
                    pd.DataFrame(all_rows).to_csv(out/'predictions.csv',index=False)
                    (out/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
                    print(f'rep={rep} fold={fold} {family} {condition} loss={log_loss(y[test_idx],pt,labels=[0,1]):.3f} {time.monotonic()-start:.1f}s',flush=True)
    summarize(out)

def summarize(out):
    d=pd.read_csv(out/'predictions.csv')
    summaries=[]
    for (model,condition),g in d.groupby(['model','condition']):
        row={'model':model,'condition':condition,'unique_clones':g.clone_id.nunique()}
        metrics=['brier','logloss','correct']+[f'{m}_{s}' for m in ['size','covered','singleton'] for s in ['90','80','cc90','cc80']]
        for metric in metrics:
            row[metric]=float(g[metric].mean())
        row['auroc']=float(np.mean([roc_auc_score(x.y,x.p_neutro) for _,x in g.groupby('rep')]))
        row['balanced_accuracy']=float(np.mean([balanced_accuracy_score(x.y,x.p_neutro>=.5) for _,x in g.groupby('rep')]))
        for lab in [0,1]:
            row[f'recall_{lab}']=float(g[g.y==lab].correct.mean())
            row[f'coverage90_{lab}']=float(g[g.y==lab].covered_90.mean())
            row[f'coverage_cc90_{lab}']=float(g[g.y==lab].covered_cc90.mean())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    contrasts=[]
    rng=np.random.default_rng(SEED)
    for model,gm in d.groupby('model'):
        for a,b in [('RNA_ATAC','RNA'),('RNA_ATAC','ATAC'),('RNA_ATAC','RNA_unrelated_ATAC'),('RNA','PRIOR')]:
            ga=gm[gm.condition==a].groupby('clone_id').mean(numeric_only=True)
            gb=gm[gm.condition==b].groupby('clone_id').mean(numeric_only=True)
            assert ga.index.equals(gb.index)
            samples=rng.integers(len(ga),size=(10000,len(ga)))
            for metric in ['brier','logloss','correct']+[f'{m}_{s}' for m in ['size','covered','singleton'] for s in ['90','80','cc90','cc80']]:
                delta=(ga[metric]-gb[metric]).to_numpy()
                ci=np.quantile(delta[samples].mean(axis=1),[.025,.975])
                contrasts.append({'model':model,'a':a,'b':b,'metric':metric,'a_minus_b':float(delta.mean()),
                                  'ci95_low':float(ci[0]),'ci95_high':float(ci[1])})
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':main()
