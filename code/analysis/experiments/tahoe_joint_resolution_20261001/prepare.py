from __future__ import annotations
import json,time,hashlib
import numpy as np
import torch,zarr
from core import HERE,ROOT,CONFIG,digest,write_json,normalize,folds,response_coordinates

def main():
    torch.set_num_threads(4)
    out=HERE/'outputs';out.mkdir(exist_ok=True)
    cache=HERE/'cache';cache.mkdir(exist_ok=True)
    freeze={'protocol_sha256':digest(HERE/'protocol.json'),'created_before_scoring':True,'code':{p.name:digest(p) for p in HERE.glob('*.py')}}
    write_json(out/'design_freeze.json',freeze)
    t=time.perf_counter()
    control=ROOT/'data/tahoe100m_control_state'
    axes=json.loads((control/'axes.json').read_text())
    wells=np.load(control/'control_wells_log1p_cpm_float32.npy')
    x=normalize(np.stack([wells[:2].mean(axis=0,dtype=np.float64),wells[2:].mean(axis=0,dtype=np.float64)]))
    group=zarr.open_group(ROOT/'results/cgc_tahoe_0i/response_tensors.zarr',mode='r')
    genes=np.load(ROOT/'results/cgc_tahoe_0i/gene_indices_g_primary.npy')
    names=list(map(str,group.attrs['contexts'])); interventions=list(map(str,group.attrs['interventions']))
    assert names==list(map(str,axes['contexts']))
    np.save(cache/'baseline.npy',x)
    write_json(cache/'axes.json',{'contexts':names,'interventions':interventions})
    print('Reading frozen response gene subset',flush=True)
    delta=np.asarray(group['delta_primary'].get_orthogonal_selection((slice(None),slice(None),slice(None),genes)),dtype=np.float32)
    assert delta.shape==(2,50,93,25695)
    manifests={'baseline_source_sha256':digest(control/'control_wells_log1p_cpm_float32.npy'),'gene_index_sha256':digest(ROOT/'results/cgc_tahoe_0i/gene_indices_g_primary.npy'),'delta_shape':delta.shape,'delta_float32_sha256':hashlib.sha256(delta.tobytes()).hexdigest(),'context_folds':[v.tolist() for v in folds(np.arange(50),names)]}
    a=torch.as_tensor(delta.reshape(2,4650,-1),device='cuda',dtype=torch.float64)/np.sqrt(25695)
    grams=[]
    for name,left,right in [('same6',0,0),('same14',1,1),('cross',0,1)]:
        k=a[left]@a[right].T
        np.save(cache/f'{name}.npy',k.cpu().numpy());grams.append(k)
        print(name,'Gram saved',flush=True)
    # Random output controls defined independently of any biological outcome.
    rng=np.random.default_rng(CONFIG['seed']+9)
    random=[]
    for j in range(20):
        q,_=torch.linalg.qr(torch.as_tensor(rng.normal(size=(25695,64)),device='cuda',dtype=torch.float64),mode='reduced')
        random.append((a@q).cpu().numpy().reshape(2,50,93,64))
    np.save(cache/'random_output64.npy',np.stack(random))
    del a,delta
    train=np.setdiff1d(np.arange(50),folds(np.arange(50),names)[0])
    tt=time.perf_counter()
    coord,info=response_coordinates(grams,train,93,0)
    torch.cuda.synchronize()
    manifests['feasibility']={'preparation_seconds':time.perf_counter()-t,'one_basis_seconds':time.perf_counter()-tt,'basis':info,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),'outcome_scores_inspected':False}
    manifests['cache_sha256']={p.name:digest(p) for p in cache.glob('*.npy')}
    write_json(out/'inputs_and_feasibility.json',manifests)
    print(json.dumps(manifests['feasibility']),flush=True)
if __name__=='__main__':main()
