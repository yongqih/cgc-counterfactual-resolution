"""Train-only output-detail selection, scored on the unchanged full response."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch,zarr

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'tahoe_joint_resolution_20261001'
sys.path.insert(0,str(PREV))
from core import ROOT,folds,fitted_weights,digest,write_json,MODELS,DIMS
from summarize import interval,simultaneous


def contributions(y,train,query,w):
    weights=torch.as_tensor(w,device=y.device,dtype=y.dtype)
    truth=y[:,query]-y[:,train].mean(dim=1,keepdim=True)
    error=y[:,query]-torch.einsum('bct,btpg->bcpg',weights,y[:,train])
    s=(truth[0]*truth[1]).sum(dim=1)
    e=(error[0]*error[1]).sum(dim=1)
    return s.cpu().numpy(),e.cpu().numpy()


def support_mask(s,e):
    return (s-e).sum(axis=0)>0


def main():
    torch.set_num_threads(4)
    out=HERE/'outputs';out.mkdir(exist_ok=True)
    config=json.loads((HERE/'protocol.json').read_text())
    write_json(out/'design_freeze.json',{'protocol_sha256':digest(HERE/'protocol.json'),'code_sha256':digest(Path(__file__)),
        'previous_prediction_sha256':digest(PREV/'outputs/contributions.npz'),'status':'FROZEN_BEFORE_NEW_POLICY_SCORING'})
    start=time.perf_counter()
    group=zarr.open_group(ROOT/'results/cgc_tahoe_0i/response_tensors.zarr',mode='r')
    genes=np.load(ROOT/'results/cgc_tahoe_0i/gene_indices_g_primary.npy')
    names=json.loads((PREV/'cache/axes.json').read_text())['contexts']
    data=np.asarray(group['delta_primary'].get_orthogonal_selection((slice(None),slice(None),slice(None),genes)),dtype=np.float32)
    y=torch.as_tensor(data,device='cuda',dtype=torch.float64)/np.sqrt(len(genes));del data
    x=np.load(PREV/'cache/baseline.npy');base=[x[b]@x[b].T for b in range(2)]
    parameters=pd.read_csv(PREV/'outputs/training_selection.csv')
    previous=np.load(PREV/'outputs/contributions.npz')
    policies=['full','support','measurement','random']
    S=np.zeros((50,len(genes)));R=np.zeros((2,50,len(genes)))
    policy_R=np.zeros((2,len(policies),50));omitted_S=np.zeros((2,50))
    masks=np.zeros((2,5,2,len(genes)),dtype=bool)
    rows=[];checks=[]
    for fi,query in enumerate(folds(np.arange(50),names)):
        tr=np.setdiff1d(np.arange(50),query)
        for mi,m in enumerate(MODELS):
            row=parameters[(parameters.fold==fi)&(parameters.model==m)&(parameters.input=='full')&parameters.selected].iloc[0]
            pars={'alpha':float(row.alpha),'factor':float(row.factor)}
            inner_s=[];inner_e=[]
            for val in folds(tr,names,'inner'):
                it=np.setdiff1d(tr,val)
                w=fitted_weights(x,it,val,m,'full',pars,base)
                s,e=contributions(y,it,val,w)
                inner_s.append(s);inner_e.append(e)
            inner_s=np.concatenate(inner_s);inner_e=np.concatenate(inner_e)
            mask=support_mask(inner_s,inner_e);count=int(mask.sum())
            measured=np.zeros(len(genes),dtype=bool)
            measured[np.argsort(-inner_s.sum(axis=0),kind='stable')[:count]]=True
            masks[mi,fi]=np.stack([mask,measured])
            w=previous['weights'][mi,DIMS.index('full')][:,query][:,:,tr]
            s,e=contributions(y,tr,query,w);r=s-e
            S[query]=s;R[mi,query]=r
            np.testing.assert_allclose(s.sum(1),previous['S'][query,:,-1].sum(1),atol=1e-10)
            np.testing.assert_allclose(e.sum(1),previous['E'][mi,-1,query,:,-1].sum(1),atol=1e-10)
            policy_R[mi,0,query]=r.sum(1)
            policy_R[mi,1,query]=r[:,mask].sum(1)
            policy_R[mi,2,query]=r[:,measured].sum(1)
            rng=np.random.default_rng(config['seed']+fi*10+mi)
            random_values=[]
            for rep in range(20):
                take=rng.permutation(len(genes))[:count]
                random_values.append(r[:,take].sum(1))
            policy_R[mi,3,query]=np.mean(random_values,axis=0)
            omitted_S[mi,query]=s[:,~mask].sum(1)
            rows.append({'fold':fi,'model':m,'retained_genes':count,'withheld_genes':len(genes)-count,
                'inner_S':float(inner_s.sum()),'inner_R_full':float((inner_s-inner_e).sum()),
                'inner_R_selected':float((inner_s-inner_e)[:,mask].sum()),
                'outer_contexts':json.dumps(query.tolist())})
            print(f'Fold {fi+1}/5 {m}: retained {count}/{len(genes)}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    np.savez_compressed(out/'contributions.npz',S=S,R=R,policy_R=policy_R,omitted_S=omitted_S,masks=masks)
    pd.DataFrame(rows).to_csv(out/'training_selection.csv',index=False)
    s=S.sum(1);full=s.sum()
    rng=np.random.default_rng(config['seed']+1);bw=rng.multinomial(50,np.full(50,1/50),size=10000)
    denom=bw@s
    assert np.all(denom>0)
    point=policy_R.sum(2)/full
    samples=(bw@policy_R.transpose(2,0,1).reshape(50,-1)).reshape(10000,2,4)/denom[:,None,None]
    omitted=omitted_S.sum(1)/full;omitted_b=(bw@omitted_S.T)/denom[:,None]
    c=np.stack([omitted,point[:,1]-point[:,0],point[:,1]-point[:,2],point[:,1]-point[:,3]],axis=-1)
    cb=np.stack([omitted_b,samples[:,:,1]-samples[:,:,0],samples[:,:,1]-samples[:,:,2],samples[:,:,1]-samples[:,:,3]],axis=-1)
    lo,hi,q,valid=simultaneous(c,cb)
    results=[];contrasts=[]
    labels=['withheld_reproducible_signal','support_minus_full','support_minus_measurement','support_minus_random']
    for mi,m in enumerate(MODELS):
        for pi,p in enumerate(policies):
            results.append({'model':m,'policy':p,'U':point[mi,pi],'interval95':interval(samples[:,mi,pi])})
        for ci,label in enumerate(labels):
            contrasts.append({'model':m,'contrast':label,'estimate':c[mi,ci],
                'pointwise95':interval(cb[:,mi,ci]),'simultaneous95':[lo[mi,ci],hi[mi,ci]]})
    write_json(out/'results.json',{'policies':results,'contrasts':contrasts,'simultaneous_q':q,'valid_draws':valid,
        'seconds':time.perf_counter()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'full_response_match':'PASS'})
    print(json.dumps({'policies':results,'contrasts':contrasts},indent=2,default=lambda v:v.item()))


if __name__=='__main__':main()
