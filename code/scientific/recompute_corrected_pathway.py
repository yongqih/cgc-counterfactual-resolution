"""Recompute only the two retained pathway-minus-gene contrasts from V2 replay inputs.

No model fitting, null selection or new random arms. The historical four-term
max-T calibration is retained; its random-panel terms are calibration-only.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bundles/resolution_pathway_frozen/src'))
from igc_virtual_cell.cgc_resolution_poc import replay as r, pathway as p, inference as i

def recompute(source_root,truth_root,replay_root,matching_table):
    context=r.ReplayContext(source_root,source_root)
    if context.correction is None:raise RuntimeError('Corrected V2 support manifest required')
    with np.load(replay_root/'m40_k4_frozen_weights.npz') as z:
        if str(z['tuning_protocol'])!='EPISODE_REFERENCE_SUPPORT_ONLY_V2':
            raise RuntimeError('Corrected sequence-aware weights required')
    checks,_=context.metric_replay()
    if not all(row['passed'] for row in checks):raise RuntimeError('Gene replay validation failed')
    w,_=p.load_pathway_weights(truth_root)
    raw=p.project_raw(np.load(replay_root/'truth_g_primary_float32.npy',mmap_mode='r'),w)
    boot=np.load(replay_root/'hierarchical_bootstrap_weights_int16.npy',mmap_mode='r')
    matching=pd.read_csv(matching_table)
    rows=[];names=[];observed=[];draws=[]
    for m,k,_ in r.BUDGETS:
        truth,pred=p.episode_scores(raw,replay_root,m,k)
        vt,va=p.utility(truth,pred);gt,ga=i._gene_utility(source_root,m,k)
        gp,gg=p.recovery(vt,va),p.recovery(gt,ga)
        for suffix,value in [('g_gene',gg),('g_pathway',gp)]:
            rows.append({'family':'point_estimate','contrast':f'm{m}_k{k}_{suffix}','estimate':value})
        names.append(f'm{m}_k{k}_pathway_minus_gene');observed.append(gp-gg)
        draws.append(i.bootstrap_g(boot,vt,va).ravel()-i.bootstrap_g(boot,gt,ga).ravel())
        if m==49:
            with np.load(replay_root/'pathway_null/pathway_frozen_utilities.npz') as frozen:
                nvt=frozen[f'null_vtruth_m{m}_k{k}'];nva=frozen[f'null_vafter_m{m}_k{k}']
        else:
            nvt=np.zeros((500,50,93));nva=np.zeros_like(nvt)
            for pathway in p.PATHWAYS:
                selected=matching[(matching.m==m)&(matching.k==k)&(matching.pathway==pathway)].candidate.to_numpy(int)
                if len(selected)!=500:raise RuntimeError('Frozen calibration membership changed')
                scores=np.load(replay_root/f'pathway_null/{pathway}_candidate_raw_scores_float32.npy',mmap_mode='r')
                nt,npred=p.episode_scores(np.asarray(scores[...,selected],dtype=np.float64),replay_root,m,k)
                tv,av=p.utility(nt,npred);nvt+=np.moveaxis(tv,-1,0);nva+=np.moveaxis(av,-1,0)
        null_point=float(np.median(1-nva.reshape(500,-1).sum(axis=1)/nvt.reshape(500,-1).sum(axis=1)))
        null_draw=np.median(i.bootstrap_g(boot,nvt,nva),axis=1)
        names.append(f'm{m}_k{k}_pathway_minus_matched_random');observed.append(gp-null_point)
        draws.append(i.bootstrap_g(boot,vt,va).ravel()-null_draw)
    calibrated=i.simultaneous_rows(names,np.asarray(observed),np.column_stack(draws),'VALID_PATHWAY_MINUS_GENE_2_BUDGETS')
    rows.extend(row for row in calibrated if row['contrast'].endswith('pathway_minus_gene'))
    for row in rows:
        row.update(tuning_protocol='EPISODE_REFERENCE_SUPPORT_ONLY_V2',calibration_family_size=4,
                   comparison_family_note='Original four-term max-T calibration retained; only two pathway-minus-gene comparisons are claim-eligible. Random terms are calibration-only, not biological control evidence.')
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','truth-root','replay-root','matching-table','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output CSV; frozen results are never overwritten by this command')
    result=recompute(a.source_root,a.truth_root,a.replay_root,a.matching_table)
    result.to_csv(a.output,index=False);print(result.to_string(index=False))

if __name__=='__main__':main()
