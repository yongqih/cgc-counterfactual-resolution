from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
import torch
from core import *

def main():
    torch.set_num_threads(4)
    started=time.perf_counter();out=HERE/'outputs';cache=HERE/'cache'
    write_json(out/'analysis_freeze.json',{'protocol_sha256':digest(HERE/'protocol.json'),'code':{p.name:digest(p) for p in HERE.glob('*.py')},'outcome_scoring_started':False})
    x=np.load(cache/'baseline.npy');names=json.loads((cache/'axes.json').read_text())['contexts']
    base=[x[p]@x[p].T for p in range(2)]
    cpu=[np.load(cache/f'{name}.npy',mmap_mode='r') for name in ('same6','same14','cross')]
    cross=np.asarray(cpu[2]).reshape(50,93,50,93)
    diag=np.stack([cross[:,p,:,p] for p in range(93)])
    grams=[torch.as_tensor(np.array(a),device='cuda',dtype=torch.float64) for a in cpu]
    outer=folds(np.arange(50),names)
    shape=(len(MODELS),len(DIMS),50,93,len(RANKS))
    S=np.zeros((50,93,len(RANKS)));E=np.zeros(shape)
    selected_S=np.zeros((len(MODELS),50,93));selected_E=np.zeros_like(selected_S)
    selected_full_S=np.zeros_like(selected_S);selected_full_E=np.zeros_like(selected_S)
    input_S=np.zeros(50);input_captured=np.zeros((50,len(DIMS)))
    all_weights=np.zeros((len(MODELS),len(DIMS),2,50,50))
    params_rows=[];policy_rows=[];basis_rows=[];direction_rows=[]
    for fi,query in enumerate(outer):
        tr=np.setdiff1d(np.arange(50),query);t=time.perf_counter()
        params,inner,rows=tune(x,diag,tr,names)
        params_rows.extend([dict(fold=fi,**row) for row in rows])
        wmap={}
        for mi,family in enumerate(MODELS):
            for di,d in enumerate(DIMS):
                w=fitted_weights(x,tr,query,family,d,params[(family,str(d))],base);wmap[(mi,di)]=w
                all_weights[mi,di,:,query[:,None],tr[None,:]]=w.transpose(1,2,0)
                fs,fe=energies(diag,tr,query,w)
                S[query,:,-1]=fs;E[mi,di,query,:,-1]=fe
        input_S[query],input_captured[query]=input_fidelity(x,tr,query)
        print(f'Fold {fi+1}: nested input tuning complete in {time.perf_counter()-t:.1f}s',flush=True)
        # Select output resolution using inner-context-disjoint projections.
        chosen_d={mi:selected_input(params,family) for mi,family in enumerate(MODELS)}
        inner_S=np.zeros((len(MODELS),len(RANKS)));inner_E=np.zeros_like(inner_S)
        for ji,val in enumerate(inner):
            it=np.setdiff1d(tr,val);iw={}
            for mi,family in enumerate(MODELS):
                d=chosen_d[mi]
                iw[mi]=fitted_weights(x,it,val,family,d,params[(family,str(d))],base)
                fs,fe=energies(diag,it,val,iw[mi])
                inner_S[mi,-1]+=fs.sum();inner_E[mi,-1]+=fe.sum()
            for direction in range(2):
                coord,info=response_coordinates(grams,it,93,direction)
                basis_rows.append(dict(outer_fold=fi,inner_fold=ji,direction=direction,**info))
                for mi in range(len(MODELS)):
                    sp,ep=coordinate_energies(coord,it,val,iw[mi])
                    inner_S[mi,:-1]+=sp.sum(axis=(0,1))/2;inner_E[mi,:-1]+=ep.sum(axis=(0,1))/2
                del coord
            print(f'Fold {fi+1}: inner output basis {ji+1}/5 complete',flush=True)
        chosen_r={}
        for mi,family in enumerate(MODELS):
            u=(inner_S[mi]-inner_E[mi])/inner_S[mi,-1]
            best=float(np.max(u));eligible=np.flatnonzero(u>=0.95*best) if best>0 else np.array([],dtype=int)
            ri=int(eligible[0]) if len(eligible) else None;chosen_r[mi]=ri
            policy_rows.append({'fold':fi,'model':family,'input':chosen_d[mi],'output':None if ri is None else RANKS[ri],'max_inner_U':best,'selected_inner_U':None if ri is None else float(u[ri]),'inner_U_by_rank':json.dumps(dict(zip(map(str,RANKS),map(float,u))))})
        for direction in range(2):
            coord,info=response_coordinates(grams,tr,93,direction)
            basis_rows.append(dict(outer_fold=fi,inner_fold=-1,direction=direction,**info))
            for mi,family in enumerate(MODELS):
                for di,d in enumerate(DIMS):
                    sp,ep=coordinate_energies(coord,tr,query,wmap[(mi,di)])
                    E[mi,di,query,:,:-1]+=ep/2
                    if mi==0 and di==0:S[query,:,:-1]+=sp/2
                    for ri,r in enumerate(RANKS[:-1]):
                        direction_rows.append({'fold':fi,'basis_plate':direction,'model':family,'input':d,'output':r,**ratio(sp[:,:,ri].sum(),ep[:,:,ri].sum(),S[query,:,-1].sum())})
            del coord
        for mi in range(len(MODELS)):
            di=DIMS.index(chosen_d[mi]);ri=chosen_r[mi]
            selected_full_S[mi,query]=S[query,:,-1]
            selected_full_E[mi,query]=E[mi,di,query,:,-1]
            if ri is None:
                selected_S[mi,query]=0;selected_E[mi,query]=0
            else:
                selected_S[mi,query]=S[query,:,ri];selected_E[mi,query]=E[mi,di,query,:,ri]
        np.savez_compressed(out/'contributions_partial.npz',S=S,E=E,input_S=input_S,input_captured=input_captured,completed_fold=fi)
        print(f'Fold {fi+1}/5 finished; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    np.savez_compressed(out/'contributions.npz',S=S,E=E,input_S=input_S,input_captured=input_captured,selected_S=selected_S,selected_E=selected_E,selected_full_S=selected_full_S,selected_full_E=selected_full_E,weights=all_weights)
    pd.DataFrame(params_rows).to_csv(out/'training_selection.csv',index=False)
    pd.DataFrame(policy_rows).to_csv(out/'selected_policy.csv',index=False)
    pd.DataFrame(basis_rows).to_csv(out/'basis_diagnostics.csv',index=False)
    pd.DataFrame(direction_rows).to_csv(out/'fold_plate_direction_results.csv',index=False)
    write_json(out/'runtime.json',{'seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'status':'BASELINE_JOINT_COMPLETE','outer_folds':[v.tolist() for v in outer]})
    print('BASELINE_JOINT_COMPLETE',flush=True)
if __name__=='__main__':main()
