from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *

def interval(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return [float(x) for x in np.quantile(v,[.025,.975])] if len(v) else [None,None]

def bootstrap_counts(draws=10000):
    rng=np.random.default_rng(CONFIG['seed']+1)
    contexts=rng.multinomial(50,np.full(50,1/50),size=draws)
    result=np.zeros((draws,50,93),dtype=np.int16)
    for c in range(50):
        for n in np.unique(contexts[:,c]):
            if n:
                take=np.flatnonzero(contexts[:,c]==n)
                result[take,c]=rng.multinomial(int(n)*93,np.full(93,1/93),size=len(take))
    return contexts,result.reshape(draws,-1)

def simultaneous(point,samples):
    p=np.asarray(point).ravel();b=np.asarray(samples).reshape(len(samples),-1)
    good=np.all(np.isfinite(b),axis=1)
    b=b[good];scale=np.std(b,axis=0,ddof=1)
    t=np.max(np.abs((b-p)/np.maximum(scale,1e-12)),axis=1)
    q=float(np.quantile(t,.95))
    return (p-q*scale).reshape(np.shape(point)),(p+q*scale).reshape(np.shape(point)),q,int(good.sum())

def main():
    out=HERE/'outputs';a=np.load(out/'contributions.npz')
    S=a['S'];E=a['E'];full=S[:,:,-1].sum()
    s=S.sum(axis=(0,1));e=E.sum(axis=(2,3))
    F=s/full;g=np.where(s>0,1-e/s,np.nan);U=(s-e)/full
    cc,bw=bootstrap_counts()
    sb=bw@S.reshape(4650,-1)
    eb=(bw@E.transpose(2,3,0,1,4).reshape(4650,-1)).reshape(10000,len(MODELS),len(DIMS),len(RANKS))
    denominator=sb[:,-1]
    fb=np.where(denominator[:,None]>0,sb/denominator[:,None],np.nan)
    gb=np.where(sb[:,None,None,:]>0,1-eb/sb[:,None,None,:],np.nan)
    ub=np.where(denominator[:,None,None,None]>0,(sb[:,None,None,:]-eb)/denominator[:,None,None,None],np.nan)
    inp=a['input_captured'].sum(axis=0)/a['input_S'].sum()
    ib=(cc@a['input_captured'])/(cc@a['input_S'])[:,None]
    rows=[]
    for mi,m in enumerate(MODELS):
        for di,d in enumerate(DIMS):
            for ri,r in enumerate(RANKS):
                row={'model':m,'input':d,'output':r,'F':F[ri],'g':g[mi,di,ri],'U':U[mi,di,ri]}
                for metric,sample in [('F',fb[:,ri]),('g',gb[:,mi,di,ri]),('U',ub[:,mi,di,ri])]:
                    lo,hi=interval(sample);row[metric+'_lower95']=lo;row[metric+'_upper95']=hi
                rows.append(row)
    pd.DataFrame(rows).to_csv(out/'joint_resolution_results.csv',index=False)
    pd.DataFrame([{'input':d,'F_input':inp[j],'lower95':interval(ib[:,j])[0],'upper95':interval(ib[:,j])[1]} for j,d in enumerate(DIMS)]).to_csv(out/'input_measurement_fidelity.csv',index=False)
    mi=MODELS.index('rbf');di=DIMS.index(16);ri=RANKS.index(64)
    primary={
      'input_full_minus_PCA16_recovery':{'estimate':float(g[mi,-1,-1]-g[mi,di,-1]),'interval95':interval(gb[:,mi,-1,-1]-gb[:,mi,di,-1])},
      'input_measurement_fraction_omitted_by_PCA16':{'estimate':float(1-inp[di]),'interval95':interval(1-ib[:,di])},
      'PCA16_full_response_recovery':{'estimate':float(g[mi,di,-1]),'interval95':interval(gb[:,mi,di,-1])},
      'output_full_minus_rank64_U':{'estimate':float(U[mi,-1,-1]-U[mi,-1,ri]),'interval95':interval(ub[:,mi,-1,-1]-ub[:,mi,-1,ri])},
      'output_truth_fraction_outside_rank64':{'estimate':float(1-F[ri]),'interval95':interval(1-fb[:,ri])},
      'rank64_U':{'estimate':float(U[mi,-1,ri]),'interval95':interval(ub[:,mi,-1,ri])}}
    primary['input_operational_support']=bool(primary['input_full_minus_PCA16_recovery']['interval95'][1]<.02 and primary['input_measurement_fraction_omitted_by_PCA16']['interval95'][0]>0 and primary['PCA16_full_response_recovery']['interval95'][0]>0)
    primary['output_operational_support']=bool(primary['output_full_minus_rank64_U']['interval95'][1]<.02 and primary['output_truth_fraction_outside_rank64']['interval95'][0]>.1 and primary['rank64_U']['interval95'][0]>0)
    pdiff=U-U[:,:,-1,None];bdiff=ub-ub[:,:,:,-1,None]
    lo,hi,q,valid=simultaneous(pdiff,bdiff)
    contrast_rows=[]
    for mm,m in enumerate(MODELS):
        for dd,d in enumerate(DIMS):
            for rr,r in enumerate(RANKS):
                contrast_rows.append({'model':m,'input':d,'output':r,'U_minus_full':pdiff[mm,dd,rr],'simultaneous_lower95':lo[mm,dd,rr],'simultaneous_upper95':hi[mm,dd,rr]})
    pd.DataFrame(contrast_rows).to_csv(out/'output_contrasts_simultaneous.csv',index=False)
    policy=[]
    for mm,m in enumerate(MODELS):
        selected=(a['selected_S'][mm]-a['selected_E'][mm]).reshape(-1)
        ref=a['selected_full_S'][mm]-a['selected_full_E'][mm]
        pb=(bw@selected)/denominator
        difference=(bw@(selected-ref.reshape(-1)))/denominator
        policy.append({'model':m,'selected_output_U':float(selected.sum()/full),'interval95':interval(pb),'selected_minus_same_model_full_output':float((selected.sum()-ref.sum())/full),'difference_interval95':interval(difference)})
    write_json(out/'primary_results.json',{'primary':primary,'policy':policy,'simultaneous_family':{'q':q,'valid_draws':valid,'comparisons':pdiff.size},'bootstrap':'conditional on trained folds and fitted bases; 10000 context/intervention paired draws'})
    np.savez_compressed(out/'bootstrap_summaries.npz',F=fb,g=gb,U=ub,input_F=ib)
    # Outcome-blind random rank64 projections use exactly the same saved predictions.
    rnd=np.load(HERE/'cache/random_output64.npy',mmap_mode='r');random_rows=[]
    outer=json.loads((out/'runtime.json').read_text())['outer_folds']
    for j in range(len(rnd)):
        sr=er=0.0
        for query in outer:
            query=np.array(query);tr=np.setdiff1d(np.arange(50),query)
            coords=rnd[j];truth=coords[:,query]-coords[:,tr].mean(axis=1,keepdims=True)
            w=a['weights'][mi,-1][:,query][:,:,tr]
            error=coords[:,query]-np.einsum('bvt,btpr->bvpr',w,coords[:,tr])
            sr+=np.sum(truth[0]*truth[1]);er+=np.sum(error[0]*error[1])
        random_rows.append({'seed_index':j,**ratio(sr,er,full)})
    pd.DataFrame(random_rows).to_csv(out/'random_rank64_control.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    xx=np.arange(len(DIMS));labels=list(map(str,DIMS))
    ax[0,0].plot(xx,inp,'o-',color='#286a8a');ax[0,0].set(xticks=xx,xticklabels=labels,ylabel='Retained cross-plate baseline signal',title='A  Measured input structure')
    for mm,m in enumerate(MODELS):
        ax[0,1].plot(xx,g[mm,:,-1],'o-',label=m)
    ax[0,1].axhline(0,color='gray',lw=.7);ax[0,1].set(xticks=xx,xticklabels=labels,ylabel='Full-response context-specific recovery g',title='B  Predictive gain from input detail');ax[0,1].legend()
    rr=np.arange(len(RANKS));rl=list(map(str,RANKS))
    ax[0,2].plot(rr,F,'o-',color='#286a8a');ax[0,2].set(xticks=rr,xticklabels=rl,ylabel='Retained response signal F',title='C  Measured response structure')
    for dd,label in [(DIMS.index(16),'PCA16 input'),(len(DIMS)-1,'Full RNA input')]:
        ax[1,0].plot(rr,g[mi,dd],'o-',label=label);ax[1,1].plot(rr,U[mi,dd],'o-',label=label)
    for aa in [ax[1,0],ax[1,1]]:aa.axhline(0,color='gray',lw=.7);aa.set_xticks(rr,rl);aa.legend()
    ax[1,0].set(ylabel='Recovery within output g',title='D  Prediction at each output resolution')
    ax[1,1].set(ylabel='Recovery normalized to full response U',title='E  How much full response is recovered?')
    im=ax[1,2].imshow(U[mi],aspect='auto',cmap='viridis');ax[1,2].set(xticks=rr,xticklabels=rl,yticks=xx,yticklabels=labels,xlabel='Output rank',ylabel='Input representation',title='F  Joint input-output recovery');fig.colorbar(im,ax=ax[1,2],label='U')
    for aa in ax.flat:aa.tick_params(axis='x',labelrotation=50)
    fig.suptitle('Tahoe: strictly held contexts, paired plates, training-only representations',fontsize=15)
    fig.savefig(out/'joint_resolution.png',dpi=240);fig.savefig(out/'joint_resolution.svg');plt.close(fig)
    print(json.dumps({'primary':primary,'policy':policy},indent=2))
if __name__=='__main__':main()
