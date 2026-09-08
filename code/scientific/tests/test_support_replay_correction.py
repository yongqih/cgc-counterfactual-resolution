"""The replay must use the same support-sequence parameter as the fit."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

SCI=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCI/'bundles/resolution_pathway_frozen/src'))
from igc_virtual_cell.cgc_resolution_poc.replay import ReplayContext


def make_cache(tmp_path,corrected=True,pooled_partial=False):
    cache=tmp_path/'results/cgc_entrywise_compression/_cache';cache.mkdir(parents=True)
    for target in range(50):
        rows=[]
        for plate in ['plate6','plate14']:
            for m,k in [(49,92),(40,4)]:
                seqs=range(8) if corrected and m==40 and not pooled_partial else [-1]
                for seq in seqs:
                    rows.append({'m':m,'k':k,'target_context_index':target,'target_outcome_used':False,
                                 'plate':plate,'support_sequence':seq,'ridge_lambda':float(seq+2),'rank':0})
        fields={'parameter_rows_json':np.asarray(json.dumps(rows))}
        if corrected: fields['tuning_protocol']=np.asarray('EPISODE_REFERENCE_SUPPORT_ONLY_V2')
        np.savez(cache/f'target_{target:02d}.npz',**fields)
    obj=object.__new__(ReplayContext);obj.source_root=tmp_path
    obj.correction={'protocol':'EPISODE_REFERENCE_SUPPORT_ONLY_V2'} if corrected else None
    return obj


def test_sequence_specific_parameters_survive_replay(tmp_path):
    loaded=make_cache(tmp_path)._load_parameters()
    assert len(loaded)==1600
    assert [loaded[(0,0,40,4,s)].ridge for s in range(8)]==list(range(2,10))
    assert len({loaded[(0,0,49,92,s)].ridge for s in range(8)})==1


def test_legacy_frozen_cache_still_replays_explicitly(tmp_path):
    assert len(make_cache(tmp_path,corrected=False)._load_parameters())==1600


def test_partial_support_cannot_reuse_pooled_tuning(tmp_path):
    with pytest.raises(RuntimeError,match='POOLED_PARAMETER_OUTSIDE_SUPPORT'):
        make_cache(tmp_path,pooled_partial=True)._load_parameters()


def test_corrected_and_legacy_protocols_cannot_mix(tmp_path):
    obj=make_cache(tmp_path);obj.correction=None
    with pytest.raises(RuntimeError,match='MIXED_SUPPORT_PROTOCOL'):obj._load_parameters()


def test_base_export_cannot_discard_registered_correction(tmp_path):
    spec=importlib.util.spec_from_file_location('export_for_correction_test',SCI/'export_frozen_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(RuntimeError,match='discard'):
        module.Exporter(tmp_path,tmp_path/'out').export({'id':'corrected','post_freeze_correction':{'date':'2026-09-08'}})


def test_renderer_accepts_only_corrected_valid_contrast_family():
    spec=importlib.util.spec_from_file_location('render_for_correction_test',SCI/'render_valid_resolution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for budget in ['m49_k92','m40_k4']:
        for suffix in ['g_gene','g_pathway']:
            rows.append({'family':'point_estimate','contrast':f'{budget}_{suffix}','estimate':'.1'})
        rows.append({'family':'VALID_PATHWAY_MINUS_GENE_2_BUDGETS','contrast':f'{budget}_pathway_minus_gene','estimate':'.2'})
    selected=module.select_clean_rows(rows)
    assert len(selected)==6
    assert all(row['post_freeze_correction']=='EPISODE_REFERENCE_SUPPORT_ONLY_V2' for row in selected)
