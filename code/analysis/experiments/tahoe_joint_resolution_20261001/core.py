"""Joint resolution analysis. SPDX-License-Identifier: MIT."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CGC_INPUT_ROOT', HERE.parents[1]))
CONFIG = json.loads((HERE / 'protocol.json').read_text())
DIMS = CONFIG['input_dimensions']
RANKS = CONFIG['output_ranks']
MODELS = CONFIG['models']

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n', encoding='utf-8')

def folds(indices, names, salt='outer'):
    ordered = sorted(map(int, indices), key=lambda i: hashlib.sha256(f"{CONFIG['seed']}|{salt}|{names[i]}".encode()).hexdigest())
    return [np.array(ordered[f::5], dtype=int) for f in range(5)]

def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    x = x-x.mean(axis=-1, keepdims=True)
    return x/np.linalg.norm(x, axis=-1, keepdims=True)

def representation(x, train, dim):
    """Only training rows define centering and PCA. Return all projected rows."""
    centered = x-x[train].mean(axis=0)
    if dim == 'full': return centered
    a = centered[train]
    val, vec = np.linalg.eigh(a@a.T)
    take = np.flatnonzero(val > max(val[-1]*1e-10, 1e-14))[::-1]
    if dim != 'span': take=take[:int(dim)]
    return (centered@a.T@vec[:, take])/np.sqrt(val[take])

def kernel(x, train, dim, family, factor, base=None):
    base=x@x.T if base is None else base
    k=base-base[:,train].mean(axis=1,keepdims=True)-base[train,:].mean(axis=0,keepdims=True)+base[np.ix_(train,train)].mean()
    if dim!='full':
        val,vec=np.linalg.eigh(k[np.ix_(train,train)])
        take=np.flatnonzero(val>max(val[-1]*1e-10,1e-14))[::-1]
        if dim!='span':take=take[:int(dim)]
        z=k[:,train]@vec[:,take]/np.sqrt(val[take])
        k=z@z.T
    if family == 'linear':
        return k/max(float(np.trace(k[np.ix_(train, train)])/len(train)),1e-14)
    distance = np.maximum(np.diag(k)[:,None]+np.diag(k)[None,:]-2*k, 0)
    dt = distance[np.ix_(train,train)]
    positive = dt[np.triu_indices(len(train),1)]
    positive = positive[positive>1e-15]
    median = float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-float(factor)*distance/median)

def weights(k, train, query, alpha):
    raw=np.linalg.solve(k[np.ix_(train,train)]+alpha*np.eye(len(train)), k[np.ix_(query,train)].T).T
    return raw+(1-raw.sum(axis=1,keepdims=True))/len(train)

def coefficients(train, query, w, n):
    v=np.zeros((len(query),n),dtype=np.float64)
    v[np.arange(len(query)),query]=1
    v[:,train]-=w
    return v

def energies(diag_cross, train, query, w):
    n=diag_cross.shape[1]
    a=coefficients(train,query,w[0],n);b=coefficients(train,query,w[1],n)
    t=coefficients(train,query,np.full((len(query),len(train)),1/len(train)),n)
    return np.einsum('vi,pij,vj->vp',t,diag_cross,t,optimize=True),np.einsum('vi,pij,vj->vp',a,diag_cross,b,optimize=True)

def candidates(family):
    if family=='linear': return [(a,1.0) for a in CONFIG['linear_alphas']]
    return [(a,f) for f in CONFIG['rbf_bandwidth_factors'] for a in CONFIG['rbf_alphas']]

def tune(x, diag_cross, outer_train, names):
    """All candidate selection sees outer-training responses only."""
    inner=folds(outer_train,names,'inner')
    output={}; all_rows=[]
    # Cache fold-local PCA representations and kernels across penalties.
    cache={};base=[x[p]@x[p].T for p in range(2)]
    for dim in DIMS:
        for family in MODELS:
            cs=candidates(family); loss=np.zeros(len(cs)); truth=0.0
            for j,val in enumerate(inner):
                tr=np.setdiff1d(outer_train,val)
                for ci,(alpha,factor) in enumerate(cs):
                    key=(j,str(dim),family,factor)
                    if key not in cache:
                        cache[key]=[kernel(x[p],tr,dim,family,factor,base[p]) for p in range(2)]
                    w=np.stack([weights(cache[key][p],tr,val,alpha) for p in range(2)])
                    s,e=energies(diag_cross,tr,val,w)
                    loss[ci]+=e.sum()
                    if ci==0: truth+=s.sum()
            best=int(np.argmin(loss))
            output[(family,str(dim))]={'alpha':cs[best][0],'factor':cs[best][1], 'inner_S':truth,'inner_E':float(loss[best]),'inner_g':float(1-loss[best]/truth)}
            for ci,(alpha,factor) in enumerate(cs):
                all_rows.append({'model':family,'input':str(dim),'alpha':alpha,'factor':factor,'inner_S':truth,'inner_E':float(loss[ci]),'selected':ci==best})
    return output,inner,all_rows

def fitted_weights(x, train, query, family, dim, parameters, base=None):
    return np.stack([weights(kernel(x[p],train,dim,family,parameters['factor'],None if base is None else base[p]),train,query,parameters['alpha']) for p in range(2)])

def selected_input(parameters,family):
    best=max(parameters[(family,str(d))]['inner_g'] for d in DIMS)
    for d in DIMS:
        if parameters[(family,str(d))]['inner_g']>=best-0.02: return d
    raise RuntimeError('input selection failed')

def response_coordinates(grams,train,pcount,direction,device='cuda'):
    """Exact one-plate training PCA using float64 sample Grams. No held rows in eigensystem."""
    index=np.array([c*pcount+p for c in train for p in range(pcount)],dtype=int)
    ii=torch.as_tensor(index,device=device)
    same=grams[direction]
    k=same[ii][:,ii].reshape(len(train),pcount,len(train),pcount)
    kc=k-k.mean(dim=0,keepdim=True)-k.mean(dim=2,keepdim=True)+k.mean(dim=(0,2),keepdim=True)
    kc=kc.reshape(len(index),len(index));kc=(kc+kc.T)*0.5
    val,vec=torch.linalg.eigh(kc)
    take=torch.where(val>val[-1]*1e-8)[0].flip(0)
    spectrum=val[take]
    v=vec[:,take]/torch.sqrt(spectrum)[None,:]
    coords=[]
    for plate in range(2):
        if plate==direction: cross=grams[plate][:,ii]
        elif direction==0: cross=grams[2][ii,:].T
        else: cross=grams[2][:,ii]
        cross=cross.reshape(-1,len(train),pcount)
        cross=cross-cross.mean(dim=1,keepdim=True)
        coords.append(cross.reshape(-1,len(index))@v)
    rank=len(take)
    result=torch.stack(coords).reshape(2,-1,pcount,rank)
    info={'rank':rank,'lambda_max':float(spectrum[0]),'lambda_min':float(spectrum[-1]),'train_reconstruction_fraction':float(spectrum.sum()/torch.trace(kc))}
    return result,info

def coordinate_energies(coords,train,query,w):
    wt=torch.as_tensor(w,device=coords.device,dtype=coords.dtype)
    truth=coords[:,query]-coords[:,train].mean(dim=1,keepdim=True)
    pred=torch.einsum('bvt,btpr->bvpr',wt,coords[:,train])
    error=coords[:,query]-pred
    s=torch.cumsum(truth[0]*truth[1],dim=-1)
    e=torch.cumsum(error[0]*error[1],dim=-1)
    indices=[min(int(r),coords.shape[-1])-1 if isinstance(r,int) else coords.shape[-1]-1 for r in RANKS[:-1]]
    return s[...,indices].cpu().numpy(),e[...,indices].cpu().numpy()

def input_fidelity(x,train,query):
    """One input-plate basis, same projector on both plates, role averaged."""
    centered=x-x[:,train].mean(axis=1,keepdims=True)
    full=np.sum(centered[0,query]*centered[1,query],axis=1)
    result=np.zeros((len(query),len(DIMS)))
    for direction in range(2):
        a=centered[direction,train]
        val,vec=np.linalg.eigh(a@a.T)
        take=np.flatnonzero(val>max(val[-1]*1e-10,1e-14))[::-1]
        basis=a.T@vec[:,take]/np.sqrt(val[take])
        cumulative=np.cumsum((centered[0,query]@basis)*(centered[1,query]@basis),axis=1)
        for j,d in enumerate(DIMS):
            if d=='full':result[:,j]+=full/2
            else:result[:,j]+=cumulative[:,-1 if d=='span' else min(int(d),len(take))-1]/2
    return full,result

def ratio(s,e,full):
    return {'F':float(s/full) if full>0 else None,'g':float(1-e/s) if s>0 else None,'U':float((s-e)/full) if full>0 else None}
