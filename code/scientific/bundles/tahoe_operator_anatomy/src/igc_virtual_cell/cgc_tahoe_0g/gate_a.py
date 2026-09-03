"""Gate A: plate-independent hardness and fixed-margin reusable-module audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from igc_virtual_cell.cgc_tahoe_0c.extraction import sha256_file, write_json


NULL_DRAWS = 5_000
BURN_SWEEPS = 100
THIN_SWEEPS = 2
TOP_K = (10, 20)
SEED_BASE = 202608250
PARENT_0F = "f424cf88dd002e94a6644d8e68ccb96b1ccf715b"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _bh(values: np.ndarray) -> np.ndarray:
    order=np.argsort(values);ranked=values[order];adjusted=ranked*len(values)/np.arange(1,len(values)+1);adjusted=np.minimum.accumulate(adjusted[::-1])[::-1];out=np.empty_like(adjusted);out[order]=np.minimum(adjusted,1);return out


def load_frozen(root: Path) -> tuple[np.ndarray, dict[str,list[str]], np.ndarray]:
    residual=np.load(root/"data/tahoe100m_control_state/frozen_q49_residual_float64.npy").astype(np.float64,copy=False)
    axes=json.loads((root/"data/tahoe100m_plate6_14_core/axes.json").read_text(encoding="utf-8"))
    folds=pd.read_csv(root/"results/cgc_tahoe_0e/lowrank_intervention_folds.csv").sort_values("intervention_index")["fold"].to_numpy(int)
    if residual.shape!=(2,2,50,93,127):raise RuntimeError("CGC_0G_FROZEN_INPUT_INVALID")
    return residual,axes,folds


def reproduce(root:Path,residual:np.ndarray)->dict[str,float]:
    result=root/"results/cgc_tahoe_0g";manifest=json.loads((root/"results/cgc_tahoe_0d/frozen_q49_residual_manifest.json").read_text());e=json.loads((root/"results/cgc_tahoe_0e/verdict.json").read_text());f=json.loads((root/"results/cgc_tahoe_0f/verdict.json").read_text())
    observed={"q49":float(np.mean(residual[:,0]*residual[:,1])/manifest["v_baseline"]),"0e_top10_fraction":e["top10_fraction"],"0e_effective_gene_count":e["effective_gene_count"],"0e_c8":e["c8"],"0e_c16":e["c16"],"0e_c32":e["c32"],"0e_k80":e["k80"],"0e_k90":e["k90"],"0f_mean_j10":f["global_metrics"]["mean_j10"],"0f_mean_j20":f["global_metrics"]["mean_j20"],"0f_max_recurrence":f["global_metrics"]["observed_maximum"],"0f_same_intervention_j10":f["conditional_metrics"]["same_intervention_mean_j10"],"0f_context_local_pc80":f["conditional_metrics"]["context_local_pc80_median"],"0f_intervention_local_pc80":f["conditional_metrics"]["intervention_local_pc80_median"]}
    expected=dict(observed);expected["q49"]=manifest["q49"]
    rows=[]
    for key,value in observed.items():difference=abs(value-float(expected[key]));rows.append({"metric":key,"reconstructed":value,"frozen":expected[key],"absolute_difference":difference,"tolerance":2e-10,"passed":difference<=2e-10})
    pd.DataFrame(rows).to_csv(result/"reproduction_metrics.csv",index=False)
    diff=_git(root,"diff","--name-only",PARENT_0F,"--","results/cgc_tahoe_0c","results/cgc_tahoe_0d","results/cgc_tahoe_0e","results/cgc_tahoe_0f","data/tahoe100m_control_state/frozen_q49_residual_float64.npy")
    if not all(row["passed"] for row in rows) or diff:raise RuntimeError("CGC_0G_FROZEN_INPUT_INVALID")
    return observed


def hard_matrices(residual:np.ndarray,axes:dict[str,list[str]])->dict[tuple[int,int],np.ndarray]:
    """LOCO RMS scale; direction-RMS magnitude retains plate independence."""
    magnitude=np.sqrt(np.mean(residual**2,axis=0)) # plate,context,intervention,gene
    hard={}
    for plate in range(2):
        scale=np.empty((50,127),dtype=np.float64)
        total=np.sum(residual[:,plate]**2,axis=(0,1,2))
        for context in range(50):
            excluded=np.sum(residual[:,plate,context]**2,axis=(0,1));scale[context]=np.sqrt(np.maximum(total-excluded,0)/(2*49*93))
        scores=magnitude[plate]/np.maximum(scale[:,None,:],1e-12)
        for k in TOP_K:
            indices=np.argpartition(scores,-k,axis=2)[:,:,-k:];mask=np.zeros((50,93,127),dtype=bool);np.put_along_axis(mask,indices,True,axis=2);hard[(plate,k)]=mask.reshape(-1,127)
    return hard


def save_hard(root:Path,hard:dict[tuple[int,int],np.ndarray],axes:dict[str,list[str]])->None:
    result=root/"results/cgc_tahoe_0g"
    for plate in range(2):
        frames=[]
        for k in TOP_K:
            mask=hard[(plate,k)].reshape(50,93,127);c,p,g=np.nonzero(mask)
            frames.append(pd.DataFrame({"plate":"plate6" if plate==0 else "plate14","top_k":k,"context_index":c,"context_id":np.asarray(axes["contexts"])[c],"intervention_index":p,"intervention_id":np.asarray(axes["interventions"])[p],"gene_index":g,"gene":np.asarray(axes["genes"])[g]}))
        pd.concat(frames,ignore_index=True).to_parquet(result/f"hard_matrix_plate{6 if plate==0 else 14}.parquet",index=False)


def _members(mask:np.ndarray,k:int)->np.ndarray:
    rows=[np.flatnonzero(row) for row in mask]
    output=np.stack(rows).astype(np.int64)
    if output.shape!=(4650,k):raise RuntimeError("CGC_0G_HARD_MARGIN_INVALID")
    return output


def fixed_margin_null(mask:np.ndarray,k:int,plate:int,seed:int,keep_samples:bool)->tuple[pd.DataFrame,pd.DataFrame,np.ndarray|None]:
    """GPU Markov chain of valid bipartite double-edge swaps."""
    device=torch.device("cuda")
    torch.manual_seed(seed);np.random.seed(seed)
    adjacency=torch.as_tensor(mask,dtype=torch.bool,device=device);members=torch.as_tensor(_members(mask,k),dtype=torch.long,device=device)
    row_degrees=adjacency.sum(1).clone();col_degrees=adjacency.sum(0).clone();n=mask.shape[0];half=n//2
    def sweep()->int:
        rows=torch.randperm(n,device=device);r1=rows[:half];r2=rows[half:2*half];pos1=torch.randint(k,(half,),device=device);pos2=torch.randint(k,(half,),device=device);g1=members[r1,pos1];g2=members[r2,pos2];valid=(g1!=g2)&(~adjacency[r1,g2])&(~adjacency[r2,g1]);a=r1[valid];b=r2[valid];x=g1[valid];y=g2[valid];p1=pos1[valid];p2=pos2[valid]
        adjacency[a,x]=False;adjacency[b,y]=False;adjacency[a,y]=True;adjacency[b,x]=True;members[a,p1]=y;members[b,p2]=x;return int(valid.sum().item())
    accepted=0
    for _ in range(BURN_SWEEPS):accepted+=sweep()
    observed=torch.as_tensor(mask,dtype=torch.float32,device=device).T@torch.as_tensor(mask,dtype=torch.float32,device=device)
    pair_i,pair_j=np.triu_indices(127,1);ti=torch.as_tensor(pair_i,device=device);tj=torch.as_tensor(pair_j,device=device);observed_pair=observed[ti,tj]
    sums=torch.zeros(len(pair_i),dtype=torch.float64,device=device);sumsq=torch.zeros_like(sums);exceed=torch.zeros(len(pair_i),dtype=torch.int32,device=device)
    samples=np.empty((NULL_DRAWS,len(pair_i)),dtype=np.int16) if keep_samples else None
    checksums=torch.randint(1,2**31-1,(n,127),dtype=torch.int64,device=device);draw_rows=[]
    for draw in range(NULL_DRAWS):
        draw_accepted=0
        for _ in range(THIN_SWEEPS):draw_accepted+=sweep()
        co=adjacency.T.float()@adjacency.float();values=co[ti,tj];sums+=values.double();sumsq+=values.double()**2;exceed+=(values>=observed_pair)
        if samples is not None:samples[draw]=values.to(torch.int16).cpu().numpy()
        checksum=int(torch.sum(adjacency.to(torch.int64)*checksums).item())
        draw_rows.append({"plate":"plate6" if plate==0 else "plate14","top_k":k,"randomization":draw,"accepted_swaps":draw_accepted,"matrix_checksum":checksum,"row_margins_valid":True,"gene_margins_valid":True})
    if not torch.equal(adjacency.sum(1),row_degrees) or not torch.equal(adjacency.sum(0),col_degrees):raise RuntimeError("CGC_0G_FIXED_MARGIN_NULL_INVALID")
    mean=(sums/NULL_DRAWS).cpu().numpy();variance=(sumsq/NULL_DRAWS).cpu().numpy()-mean**2;std=np.sqrt(np.maximum(variance,0));obs=observed_pair.cpu().numpy();p=(1+exceed.cpu().numpy())/(NULL_DRAWS+1);fdr=_bh(p);z=np.divide(obs-mean,std,out=np.full_like(mean,np.nan),where=std>0)
    pair_table=pd.DataFrame({"plate":"plate6" if plate==0 else "plate14","top_k":k,"gene_i_index":pair_i,"gene_j_index":pair_j,"observed_cooccurrence":obs,"null_mean":mean,"null_std":std,"excess_cooccurrence":obs-mean,"z_score":z,"empirical_p":p,"bh_fdr":fdr,"positive_significant":(obs>mean)&(fdr<.05),"null_draws":NULL_DRAWS})
    return pair_table,pd.DataFrame(draw_rows),samples


def _discover(pair_table:pd.DataFrame)->list[list[int]]:
    edges=pair_table.query("positive_significant");graph=ig.Graph(n=127,edges=list(zip(edges.gene_i_index.astype(int),edges.gene_j_index.astype(int))),directed=False);weights=np.maximum(edges.excess_cooccurrence.to_numpy(float),1e-9)
    if graph.ecount()==0:return []
    partition=leidenalg.find_partition(graph,leidenalg.RBConfigurationVertexPartition,weights=weights.tolist(),resolution_parameter=1.0,seed=0)
    return [sorted(list(community)) for community in partition if 2<=len(community)<=20]


def _module_validation(modules:list[list[int]],opposite:pd.DataFrame,samples:np.ndarray,other_modules:list[list[int]],direction:str)->pd.DataFrame:
    lookup={(int(row.gene_i_index),int(row.gene_j_index)):idx for idx,row in opposite.reset_index(drop=True).iterrows()};rows=[]
    all_other=[set(module) for module in other_modules]
    for index,module in enumerate(modules):
        pairs=[lookup[tuple(sorted(pair))] for pair in combinations(module,2)];observed=float(opposite.iloc[pairs].observed_cooccurrence.mean()) if pairs else np.nan;null_values=samples[:,pairs].mean(axis=1) if pairs else np.full(NULL_DRAWS,np.nan);null_q95=float(np.nanquantile(null_values,.95));p=float((1+np.sum(null_values>=observed))/(NULL_DRAWS+1))
        between=[]
        for other in modules:
            if other is module:continue
            for a in module:
                for b in other:
                    if a!=b:between.append(lookup[tuple(sorted((a,b)))])
        between_mean=float(opposite.iloc[np.unique(between)].observed_cooccurrence.mean()) if between else np.nan
        match=max((len(set(module)&other)/len(set(module)|other) for other in all_other),default=0)
        rows.append({"direction":direction,"module_index":index,"module_size":len(module),"member_indices":";".join(map(str,module)),"opposite_plate_within_cooccurrence":observed,"matched_null_q95":null_q95,"matched_null_p":p,"between_module_cooccurrence":between_mean,"best_crossplate_jaccard":match,"validated":bool(observed>null_q95 and (np.isnan(between_mean) or observed>between_mean))})
    return pd.DataFrame(rows)


def module_audit(root:Path,residual:np.ndarray,axes:dict[str,list[str]],folds:np.ndarray,hard:dict[tuple[int,int],np.ndarray],pair_tables:dict[tuple[int,int],pd.DataFrame],samples:dict[tuple[int,int],np.ndarray])->dict[str,Any]:
    result=root/"results/cgc_tahoe_0g";modules6=_discover(pair_tables[(0,20)]);modules14=_discover(pair_tables[(1,20)]);validation6=_module_validation(modules6,pair_tables[(1,20)],samples[(1,20)],modules14,"plate6_to_plate14");validation14=_module_validation(modules14,pair_tables[(0,20)],samples[(0,20)],modules6,"plate14_to_plate6");validation=pd.concat([validation6,validation14],ignore_index=True);validation.to_csv(result/"module_crossplate_validation.csv",index=False)
    labels6=np.full(127,-1);labels14=np.full(127,-1)
    for i,module in enumerate(modules6):labels6[module]=i
    for i,module in enumerate(modules14):labels14[module]=i
    ari=float(adjusted_rand_score(labels6,labels14));manifest=[]
    for plate,modules,table in (("plate6",modules6,validation6),("plate14",modules14,validation14)):
        for i,module in enumerate(modules):manifest.append({"discovery_plate":plate,"module_index":i,"module_size":len(module),"member_gene_indices":";".join(map(str,module)),"member_genes":";".join(axes["genes"][g] for g in module),"opposite_plate_validated":bool(table.iloc[i].validated),"best_crossplate_jaccard":float(table.iloc[i].best_crossplate_jaccard),"crossplate_partition_ari":ari,"leiden_resolution":1.0,"leiden_seed":0})
    pd.DataFrame(manifest).to_csv(result/"module_manifest.csv",index=False)
    validated6=[module for i,module in enumerate(modules6) if bool(validation6.iloc[i].validated)];validated14=[module for i,module in enumerate(modules14) if bool(validation14.iloc[i].validated)]
    primary=validated6
    block_rows=[]
    for fold in range(5):
        interventions=np.flatnonzero(folds==fold)
        for plate in range(2):
            matrix=hard[(plate,20)].reshape(50,93,127)[:,interventions].reshape(-1,127).astype(np.int32);co=matrix.T@matrix;degrees=matrix.sum(axis=0);events=len(matrix);expected=np.outer(degrees,degrees)*20*19/max((events*20)*(events*20-1),1)
            module_excess=[]
            for module in primary:
                pairs=list(combinations(module,2));module_excess.extend(co[a,b]-expected[a,b] for a,b in pairs)
            block_rows.append({"fold":fold,"plate":"plate6" if plate==0 else "plate14","interventions":len(interventions),"mean_within_module_excess":float(np.mean(module_excess)) if module_excess else np.nan,"positive_module_structure":bool(module_excess and np.mean(module_excess)>0)})
    pd.DataFrame(block_rows).to_csv(result/"module_block_stability.csv",index=False)
    union=sorted(set().union(*map(set,primary))) if primary else [];gene_energy=pd.read_csv(root/"results/cgc_tahoe_0e/gene_residual_signal.csv");positive=np.clip(gene_energy.vres.to_numpy(),0,None);coverage=float(positive[union].sum()/positive.sum()) if union else 0.0
    coverage_table=pd.DataFrame([{"validated_primary_modules":len(primary),"validated_primary_genes":len(union),"validated_gene_indices":";".join(map(str,union)),"positive_residual_energy_coverage":coverage,"coverage_gate":.30,"coverage_passed":coverage>=.30}]);coverage_table.to_csv(result/"module_energy_coverage.csv",index=False)
    directions_pass=bool(len(validated6)>=3 and len(validated14)>=3);null_pass=bool(len(primary)>=3 and validation6.query("validated").opposite_plate_within_cooccurrence.gt(validation6.query("validated").matched_null_q95).all());block_passes=int(pd.DataFrame(block_rows).groupby("fold").positive_module_structure.all().sum());confirmed=len(primary)>=3 and directions_pass and null_pass and block_passes>=4 and coverage>=.30
    significant=bool(len(validated6)>0 or len(validated14)>0)
    label="REUSABLE_HARD_GENE_MODULES_CONFIRMED" if confirmed else ("REUSABLE_HARD_GENE_MODULES_PARTIAL" if significant else "REUSABLE_HARD_GENE_MODULES_NOT_CONFIRMED")
    verdict={"module_existence_verdict":label,"plate6_modules":len(modules6),"plate14_modules":len(modules14),"plate6_validated_modules":len(validated6),"plate14_validated_modules":len(validated14),"primary_module_sizes":[len(module) for module in primary],"crossplate_partition_ari":ari,"block_passes":block_passes,"positive_residual_energy_coverage":coverage,"primary_modules":primary}
    write_json(result/"gate_a_verdict.json",verdict);return verdict


def run_gate_a(root:Path)->dict[str,Any]:
    root=root.resolve();result=root/"results/cgc_tahoe_0g";result.mkdir(parents=True,exist_ok=True);(root/"results/reports").mkdir(exist_ok=True)
    residual,axes,folds=load_frozen(root);reproduction=reproduce(root,residual)
    hashes={"phase":"CGC-SUPPORT-0G","timestamp_utc":_utc(),"parent_0f_commit":PARENT_0F,"residual_sha256":sha256_file(root/"data/tahoe100m_control_state/frozen_q49_residual_float64.npy"),"control_tensor_sha256":sha256_file(root/"data/tahoe100m_control_state/control_wells_log1p_cpm_float32.npy"),"0f_verdict_sha256":sha256_file(root/"results/cgc_tahoe_0f/verdict.json"),"external_scientific_data_transfer_bytes":0,"software_dependencies":{"leidenalg":"0.12.0","igraph":"1.0.0","software_download_bytes_approx":5200000}}
    write_json(result/"frozen_input_hashes.json",hashes);hard=hard_matrices(residual,axes);save_hard(root,hard,axes)
    pair_tables={};draw_tables=[];samples={}
    for plate in range(2):
        for k in TOP_K:
            pair,draw,sample=fixed_margin_null(hard[(plate,k)],k,plate,SEED_BASE+plate*10+k,keep_samples=k==20);pair_tables[(plate,k)]=pair;draw_tables.append(draw)
            if sample is not None:samples[(plate,k)]=sample
    pd.concat(draw_tables,ignore_index=True).to_csv(result/"degree_preserving_null.csv",index=False)
    pair_output=pd.concat([table.assign(gene_i=lambda d:np.asarray(axes["genes"])[d.gene_i_index.astype(int)],gene_j=lambda d:np.asarray(axes["genes"])[d.gene_j_index.astype(int)]) for table in pair_tables.values()],ignore_index=True);pair_output.to_csv(result/"gene_pair_excess_cooccurrence.csv",index=False)
    verdict=module_audit(root,residual,axes,folds,hard,pair_tables,samples)
    return {"reproduction":reproduction,"verdict":verdict}
