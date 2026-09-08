import numpy as np
import pytest

from igc_virtual_cell.cgc_entrywise.estimators import select_parameters
from igc_virtual_cell.cgc_entrywise.core import affine_ridge_weights


def fixture():
    rng = np.random.default_rng(20260908)
    x = rng.normal(size=(1,93,4)) + np.einsum('cr,rpg->cpg',rng.normal(size=(50,2)),rng.normal(size=(2,93,4)))
    x += .1*rng.normal(size=x.shape)
    orders = np.stack([np.roll(np.arange(1,50),-4*i) for i in range(8)])
    sentinels = np.stack([rng.permutation(93) for _ in range(8)])
    return rng,x,orders,sentinels,np.arange(93)%5


def select(x, orders, sentinels, folds, m=4, sequence=0):
    flat=x.reshape(4650,4)
    full=(flat@flat.T).reshape(50,93,50,93)
    matched=np.einsum('cpg,dpg->pcd',x,x)
    return select_parameters(matched,full,x.sum(axis=2).reshape(-1),0,m,4,orders,sentinels,folds,support_sequence=sequence)


def test_nonidentical_support_pool_is_rejected():
    _,x,orders,sentinels,folds=fixture()
    with pytest.raises(ValueError,match='BUDGET_LEAKAGE'):
        select(x,orders,sentinels,folds,sequence=None)


def test_all_unobserved_responses_leave_tuning_and_prediction_unchanged():
    rng,x,orders,sentinels,folds=fixture()
    a=select(x,orders,sentinels,folds)
    allowed=orders[0,:4]
    changed=x.copy()
    outside=np.setdiff1d(np.arange(50),allowed)
    # Includes every target-context outcome: none may enter reference CV.
    changed[outside]=1e6*rng.normal(size=changed[outside].shape)
    b=select(changed,orders,sentinels,folds)
    assert a == b
    sentinel=sentinels[0,:4]; hidden=int(sentinels[0,-1])
    source=x[allowed][:,sentinel].reshape(4,-1); target=x[0,sentinel].reshape(-1)
    wa=affine_ridge_weights(source@source.T,source@target,a.ridge)
    wb=affine_ridge_weights(source@source.T,source@target,b.ridge)
    np.testing.assert_array_equal(wa@x[allowed,hidden],wb@x[allowed,hidden])


def test_target_context_cannot_enter_reference_support():
    _,x,orders,sentinels,folds=fixture(); orders[0,0]=0
    with pytest.raises(ValueError,match='target-contaminated'):
        select(x,orders,sentinels,folds)


def test_identical_support_pool_remains_allowed():
    _,x,orders,sentinels,folds=fixture()
    # All 49 reference contexts are observed, so pooling is budget-valid.
    result=select(x,orders,sentinels,folds,m=49,sequence=None)
    assert result.pseudo_targets == 40
