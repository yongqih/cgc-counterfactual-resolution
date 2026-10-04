from __future__ import annotations
import numpy as np
import pandas as pd
import zarr
from core import *

def main():
    out=HERE/'outputs';a=np.load(out/'contributions.npz');checks={}
    names=json.loads((HERE/'cache/axes.json').read_text())['contexts']
    outer=folds(np.arange(50),names);weights=a['weights']
    checks['weight_sum_max_error']=float(np.max(np.abs(weights.sum(axis=-1)-1)))
    for fi,query in enumerate(outer):
        block=weights[:,:,:,query][:,:,:,:,query]
        checks[f'fold_{fi}_forbidden_weights_max']=float(np.max(np.abs(block)))
    group=zarr.open_group(ROOT/'results/cgc_tahoe_0i/response_tensors.zarr',mode='r')
    genes=np.load(ROOT/'results/cgc_tahoe_0i/gene_indices_g_primary.npy')
    selected_p=np.array([0,17,92])
    y=np.asarray(group['delta_primary'].get_orthogonal_selection((slice(None),slice(None),selected_p,genes)),dtype=np.float64)
    max_s=max_e=0.0
    for mi in range(len(MODELS)):
        for di in (0,3,len(DIMS)-1):
            for query in outer:
                tr=np.setdiff1d(np.arange(50),query)
                pred=np.einsum('bvt,btpg->bvpg',weights[mi,di][:,query][:,:,tr],y[:,tr])
                error=y[:,query]-pred
                truth=y[:,query]-y[:,tr].mean(axis=1,keepdims=True)
                ss=np.sum(truth[0]*truth[1],axis=-1)/len(genes)
                ee=np.sum(error[0]*error[1],axis=-1)/len(genes)
                max_s=max(max_s,float(np.max(np.abs(ss-a['S'][query][:,selected_p,-1]))))
                max_e=max(max_e,float(np.max(np.abs(ee-a['E'][mi,di,query][:,selected_p,-1]))))
    checks['real_direct_truth_max_error']=max_s;checks['real_direct_error_max_error']=max_e
    assert max_s<1e-9 and max_e<1e-9
    assert checks['weight_sum_max_error']<1e-10
    assert all(v==0 for k,v in checks.items() if 'forbidden' in k)
    diagnostics=pd.read_csv(out/'basis_diagnostics.csv')
    checks['number_train_only_bases']=len(diagnostics)
    checks['min_training_reconstruction']=float(diagnostics.train_reconstruction_fraction.min())
    checks['minimum_basis_rank']=int(diagnostics['rank'].min())
    checks['maximum_basis_rank']=int(diagnostics['rank'].max())
    checks['status']='PASS'
    write_json(out/'verification.json',checks)
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
