import numpy as np
import torch
from core import energies,response_coordinates,coordinate_energies,RANKS,tune,representation,input_fidelity,kernel

def test_projection_and_gram_identity():
    rng=np.random.default_rng(10);n,p,g=10,4,20
    y=rng.normal(size=(2,n,p,g));tr=np.arange(8);query=np.array([8,9])
    w=rng.normal(size=(2,2,8));w+=(1-w.sum(axis=2,keepdims=True))/8
    flat=y.reshape(2,n*p,g)
    grams=[torch.tensor(flat[0]@flat[0].T),torch.tensor(flat[1]@flat[1].T),torch.tensor(flat[0]@flat[1].T)]
    diag=np.stack([y[0,:,j]@y[1,:,j].T for j in range(p)])
    s,e=energies(diag,tr,query,w)
    truth=y[:,query]-y[:,tr].mean(axis=1,keepdims=True)
    err=y[:,query]-np.einsum('bvt,btpg->bvpg',w,y[:,tr])
    np.testing.assert_allclose(s,np.sum(truth[0]*truth[1],axis=-1),atol=1e-10)
    np.testing.assert_allclose(e,np.sum(err[0]*err[1],axis=-1),atol=1e-10)
    for direction in range(2):
        coords,info=response_coordinates(grams,tr,p,direction,device='cpu')
        sp,ep=coordinate_energies(coords,tr,query,w)
        assert info['rank']==g
        np.testing.assert_allclose(sp[:,:,-1],s,atol=1e-8)
        np.testing.assert_allclose(ep[:,:,-1],e,atol=1e-8)
        a=y[direction,tr]-y[direction,tr].mean(axis=0,keepdims=True)
        _,_,v=np.linalg.svd(a.reshape(-1,g),full_matrices=False)
        q=v[:4].T;proj=q@q.T
        pred=np.einsum('bvt,btpg->bvpg',w,y[:,tr])-y[:,tr].mean(axis=1,keepdims=True)
        projected=np.einsum('bvpg,gh->bvph',pred,proj)
        efull=truth-projected
        recovered=s-np.sum(efull[0]*efull[1],axis=-1)
        index=RANKS.index(4)
        np.testing.assert_allclose(recovered,sp[:,:,index]-ep[:,:,index],atol=1e-8)

def test_held_response_mutation_invariance():
    rng=np.random.default_rng(11);n=15;tr=np.arange(12);names=[str(j) for j in range(n)]
    x=rng.normal(size=(2,n,26));y=rng.normal(size=(2,n,3,12))
    def diag(a):return np.stack([a[0,:,j]@a[1,:,j].T for j in range(3)])
    before,_,_=tune(x,diag(y),tr,names)
    mutant=y.copy();mutant[:,12:]=rng.normal(size=mutant[:,12:].shape)*1000
    after,_,_=tune(x,diag(mutant),tr,names)
    assert before==after

def test_input_measurement_and_fine_signal_controls():
    rng=np.random.default_rng(12);x=rng.normal(size=(1,20,30));x=np.concatenate([x,x],axis=0)
    full,retained=input_fidelity(x,np.arange(15),np.arange(15,20))
    np.testing.assert_allclose(retained[:,-1],full)
    assert np.all(retained[:,0] <= retained[:,3]+1e-10)
    # A prediction can contain recoverable signal only in a fine direction.
    q,_=np.linalg.qr(rng.normal(size=(30,30)))
    target=rng.normal(size=(12,1))*q[:,20][None,:]
    coarse=target@q[:,:4]@q[:,:4].T
    fine=target@q[:,:24]@q[:,:24].T
    assert np.sum(coarse**2)<1e-20
    np.testing.assert_allclose(fine,target,atol=1e-10)

def test_query_rows_do_not_fit_input_basis():
    rng=np.random.default_rng(13);x=rng.normal(size=(12,30));tr=np.arange(10)
    first=representation(x,tr,4)
    x[10:]*=100
    second=representation(x,tr,4)
    np.testing.assert_allclose(first[tr],second[tr],atol=1e-12)

def test_input_gram_optimization_matches_direct_vectors():
    rng=np.random.default_rng(14);x=rng.normal(size=(12,30));tr=np.arange(10)
    for dim in (2,4,'span','full'):
        z=representation(x,tr,dim);k=z@z.T
        direct=k/(np.trace(k[np.ix_(tr,tr)])/len(tr))
        np.testing.assert_allclose(kernel(x,tr,dim,'linear',1),direct,atol=1e-10)

def test_anchor_error_scoring_and_hidden_target_invariance():
    from anchors import anchor_matrices,corrected_diagonal
    rng=np.random.default_rng(15);y=rng.normal(size=(2,3,93,20));pred=rng.normal(size=y.shape)
    errors=y-pred;h=np.einsum('vpg,vqg->vpq',errors[0],errors[1])
    for a in anchor_matrices():
        assert np.all(np.diag(a)==0)
        np.testing.assert_allclose(a.sum(axis=1),1)
        corrected=pred+.5*np.einsum('pq,bvqg->bvpg',a,y-pred)
        residual=y-corrected
        np.testing.assert_allclose(corrected_diagonal(h,a,.5),np.sum(residual[0]*residual[1],axis=-1),atol=1e-10)
        mutant=y.copy();mutant[:,:,7]+=1000
        corrected_mutant=pred+.5*np.einsum('pq,bvqg->bvpg',a,mutant-pred)
        np.testing.assert_allclose(corrected[:,:,7],corrected_mutant[:,:,7],atol=1e-10)


def test_biological_specificity_decomposition_matches_direct_vectors():
    rng=np.random.default_rng(16);n,p,g=10,7,13
    y=rng.normal(size=(2,n,p,g));tr=np.arange(8);query=np.array([8,9])
    w=rng.normal(size=(2,2,8));w+=(1-w.sum(axis=2,keepdims=True))/8
    truth=y[:,query]-y[:,tr].mean(axis=1,keepdims=True)
    error=y[:,query]-np.einsum('bvt,btpg->bvpg',w,y[:,tr])
    mean=y.mean(axis=2);gram=(mean[0]@mean[1].T)[None]
    s,e=energies(gram,tr,query,w)
    for full,coarse in ((truth,s),(error,e)):
        general=full.mean(axis=2,keepdims=True)
        interaction=full-general
        direct_general=p*np.sum(general[0]*general[1],axis=(1,2))
        direct_interaction=np.sum(interaction[0]*interaction[1],axis=(1,2))
        np.testing.assert_allclose(coarse[:,0]*p,direct_general,atol=1e-10)
        np.testing.assert_allclose(direct_general+direct_interaction,np.sum(full[0]*full[1],axis=(1,2)),atol=1e-10)
