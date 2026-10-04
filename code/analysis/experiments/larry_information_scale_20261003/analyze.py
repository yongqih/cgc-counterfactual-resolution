"""Conditional, fixed-cohort cross-library endpoint decomposition."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'experiments/baseline_sufficiency_data_audit_20261002'
sys.path.insert(0, str(OLD))
from validate_larry_branches import exclude_ambiguous_late_records
from prepare_larry_branch_validation import family

B = 3000
SEED = 20261003
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def interval(v):
    return {'estimate': float(v[0]), 'conditional_CI95': np.quantile(v[1:], [.025,.975]).tolist()}


def run_variant(meta, x, ids, variant):
    target = (meta['Time point']==6) & meta.clone_index.isin(ids)
    ref = ((meta['Time point']==6) & ~meta.clone_index.isin(ids)
           & (meta['Cell type annotation']=='Neutrophil')) if variant=='reference' else np.zeros(len(meta),bool)
    keep = target | ref
    m = meta.loc[keep].reset_index(drop=True)
    x = x[keep.to_numpy() if hasattr(keep,'to_numpy') else keep]
    gram = (x@x.T).toarray()
    n = len(m)
    # Each H block is an empirical mean's resampling weights. Shared reference
    # blocks are drawn once and reused, retaining their induced covariance.
    h = np.zeros((B+1,n))
    rng = np.random.default_rng(SEED)
    block_sizes = np.zeros(n)
    target_local = m.clone_index.isin(ids)
    groups = list(m[target_local].groupby(['clone_index','Well','Library']).groups.values())
    groups += list(m[~target_local].groupby('Library').groups.values())
    for ii in groups:
        ii = np.asarray(ii); ng = len(ii)
        h[0,ii] = 1/ng
        h[1:,ii] = rng.multinomial(ng,np.full(ng,1/ng),size=B)/ng
        block_sizes[ii] = ng
    assert np.all(block_sizes>0)

    coeff = np.zeros((len(ids),2,2,n)) # clone, well, replication half, cell
    sizes = []
    families = []
    for ci,clone in enumerate(ids):
        cells = m[m.clone_index==clone]
        families.append(cells.culture_family.iloc[0])
        record = {'clone_index':int(clone),'culture_family':families[-1]}
        for wi,well in enumerate([1,2]):
            libs = sorted(cells.loc[cells.Well==well,'Library'].unique())
            assert len(libs)>=2, (variant,clone,well,libs)
            for ri,chosen in enumerate([libs[:1],libs[1:]]):
                total = int(((cells.Well==well)&cells.Library.isin(chosen)).sum())
                record[f'well{well}_{"AB"[ri]}'] = total
                for lib in chosen:
                    ii = np.flatnonzero((m.clone_index==clone)&(m.Library==lib))
                    coeff[ci,wi,ri,ii] = len(ii)/total
                    if variant=='reference':
                        jj = np.flatnonzero((~target_local)&(m.Library==lib))
                        assert len(jj)>=10, (variant,lib,len(jj))
                        coeff[ci,wi,ri,jj] -= len(ii)/total
        sizes.append(record)

    def cross(a,b):
        ia=np.flatnonzero(a); ib=np.flatnonzero(b)
        aa=h[:,ia]*a[ia]; bb=h[:,ib]*b[ib]
        out=np.empty(B+1)
        gg=gram[np.ix_(ia,ib)]
        for start in range(0,B+1,500):
            end=min(start+500,B+1)
            out[start:end]=np.sum((aa[start:end]@gg)*bb[start:end],axis=1)
        return out

    floor=[]; end=[]
    for ci in range(len(ids)):
        q=coeff[ci]
        floor.append(cross(q[0,0]-q[1,0],q[0,1]-q[1,1])/4)
        end.append((cross(q[0,0],q[0,1])+cross(q[1,0],q[1,1]))/2)
    floor=np.array(floor); end=np.array(end); families=np.array(families)
    mid=coeff.mean(axis=1)
    summaries=[]; draws={}
    def summarize(mask,label,balanced=False):
        indices=np.flatnonzero(mask)
        fams=np.unique(families[indices]); weights=np.zeros(len(ids))
        if balanced:
            for fam in fams:
                ii=indices[families[indices]==fam]; weights[ii]=1/len(fams)/len(ii)
        else: weights[indices]=1/len(indices)
        ll=weights@floor
        vv=weights@end
        common=np.zeros(B+1)
        for fam in fams:
            ii=indices[families[indices]==fam]; wf=weights[ii].sum()
            av=np.einsum('c,crn->rn',weights[ii]/wf,mid[ii])
            vv-=wf*cross(av[0],av[1])
            dd=np.einsum('c,crn->rn',weights[ii]/wf,coeff[ii,0]-coeff[ii,1])
            common+=wf*cross(dd[0],dd[1])/4
        globalav=np.einsum('c,crn->rn',weights,mid)
        globalv=weights@end-cross(globalav[0],globalav[1])
        assert np.all(np.isfinite(vv))
        rec={'variant':variant,'aggregation':label,'n_clones':len(indices),
             'n_families':len(fams),'n_positive_points':int((floor[indices,0]>0).sum()),
             'floor':interval(ll),'endpoint_variation':interval(vv),
             'midpoint_variation':interval(vv-ll),'normalized_floor':interval(ll/vv),
             'global_centered_variation':interval(globalv),
             'global_centered_normalized_floor':interval(ll/globalv),
             'shared_well_component':interval(common),
             'well_centered_floor':interval(ll-common),
             'well_centered_variation':interval(vv-common),
             'well_centered_normalized_floor':interval((ll-common)/(vv-common)),
             'n_nonpositive_denominator_draws':int((vv[1:]<=0).sum())}
        summaries.append(rec)
        for key,val in [('floor',ll),('variation',vv),('ratio',ll/vv),('well_centered_floor',ll-common)]:
            draws[f'{label}_{key}']=val
        print(variant,label,json.dumps({k:rec[k] for k in ['floor','endpoint_variation','normalized_floor','well_centered_floor']}),flush=True)

    summarize(np.ones(len(ids),bool),'clone_equal')
    summarize(np.ones(len(ids),bool),'family_equal',True)
    summarize(np.array(ids)!=1978,'exclude1978')
    for fam in np.unique(families): summarize(families==fam,'family_'+fam)

    # Independent direct-vector identity check and all leave-one-out point values.
    means=np.einsum('cwrn,n->cwrn',coeff,h[0]).reshape(-1,n)@x
    means=np.asarray(means).reshape(len(ids),2,2,-1)
    direct=np.sum((means[:,0,0]-means[:,1,0])*(means[:,0,1]-means[:,1,1]),axis=1)/4
    assert np.allclose(direct,floor[:,0],rtol=1e-9,atol=1e-10)
    loo=[]
    for omit in ids:
        mask=np.array(ids)!=omit; nn=int(mask.sum())
        endpoints=means[mask]; ff=families[mask]
        v=0
        for fam in np.unique(ff):
            z=endpoints[ff==fam]; center=z.mean(axis=(0,1)); centered=z-center
            v+=np.sum(centered[:,:,0]*centered[:,:,1])/(2*nn)
        l=float(direct[mask].mean())
        loo.append({'omitted_clone':int(omit),'floor':l,'variation':float(v),'ratio':l/v})
    records=[]
    for i,record in enumerate(sizes):
        records.append({**record,**interval(floor[i])})
    reference_counts=m[~target_local].groupby('Library').size().to_dict()
    np.savez_compressed(HERE/f'{variant}_conditional_draws.npz',**draws,clone_floor=floor,clone_ids=ids)
    return {'variant':variant,'n_target_cells':int(target_local.sum()),'reference_cells_by_library':reference_counts,
            'clones':records,'summary':summaries,'leave_one_out':loo,
            'direct_vector_max_absolute_discrepancy':float(np.max(abs(direct-floor[:,0])))}


def main():
    p=OLD/'larry'
    names=['branch_validation_metadata.csv','branch_validation_full_gene.npz','branch_validation_gene_names.json',
           'branch_full_gene_signal.json','stateFate_inVitro_metadata.txt.gz']
    manifest={name:SHA(p/name) for name in names}
    meta=pd.read_csv(p/names[0]);x=load_npz(p/names[1]).astype(np.float64)
    meta,x,screen=exclude_ambiguous_late_records(meta,x);x.data=np.log1p(x.data)
    late=meta[meta['Time point']==6]
    sizes=late.groupby(['clone_index','Well']).size().unstack(fill_value=0)
    ids=sizes.index[(sizes[1]>=10)&(sizes[2]>=10)].tolist()
    previous=json.loads((p/'branch_full_gene_signal.json').read_text())
    assert ids==[r['clone_index'] for r in previous]
    results=[]
    with threadpool_limits(limits=6):
        primary=run_variant(meta,x,ids,'raw');results.append(primary)
        assert np.allclose([r['estimate'] for r in primary['clones']],[r['estimate'] for r in previous],rtol=1e-9)
        original=pd.read_csv(p/'stateFate_inVitro_metadata.txt.gz',sep='\t')
        original['culture_family']=original.Library.map(family)
        suspect=original.duplicated(['culture_family','Time point','Well','Cell barcode'],keep=False)
        suspect_ids=set(np.flatnonzero(suspect))
        keep=~meta.original_cell_index.isin(suspect_ids)
        strict=run_variant(meta[keep].reset_index(drop=True),x[keep.to_numpy()],ids,'barcode');results.append(strict)
        reference=run_variant(meta,x,ids,'reference');results.append(reference)
    report={'plan_sha256':SHA(HERE/'ANALYSIS_PLAN.md'),'input_sha256':manifest,
            'seed':SEED,'bootstrap_repetitions':B,'n_genes':x.shape[1],
            'bootstrap_unit':'Cells within clone x well x published library; references resampled jointly per library; cultures fixed.',
            'primary_duplicate_screen':screen,'results':results,
            'interpretation':'Estimated paired common-point error for observed branch population profiles. Cross-library sampling-unbiased interpretation requires independent zero-mean errors; systematic well-linked artifacts are not excluded by bootstrap.',
            'normalization':'Within-family reproducible endpoint variation, equal clone and well weight; not a population Bayes-risk fraction.'}
    (HERE/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([{**r,'variant':z['variant'],'ci95_low':r['conditional_CI95'][0],'ci95_high':r['conditional_CI95'][1]}
                  for z in results for r in z['clones']]).drop(columns='conditional_CI95').to_csv(HERE/'clone_estimates.csv',index=False)
    flat=[]
    for z in results:
        for s in z['summary']:
            for metric,v in s.items():
                if isinstance(v,dict) and 'estimate' in v:
                    flat.append({'variant':z['variant'],'aggregation':s['aggregation'],'metric':metric,
                                 'estimate':v['estimate'],'ci95_low':v['conditional_CI95'][0],'ci95_high':v['conditional_CI95'][1]})
    pd.DataFrame(flat).to_csv(HERE/'aggregate_estimates.csv',index=False)
    assert manifest=={name:SHA(p/name) for name in names}
    print('COMPLETE: frozen inputs unchanged.',flush=True)


if __name__=='__main__':main()
