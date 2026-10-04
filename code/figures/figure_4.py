from common import *
# Figure 4: two broad primary curves with small application and representation checks.
t=read('Main_Figure_6');sf=read('Supplementary_Figure_5');fig=plt.figure(figsize=(7.0866141732,6.1))
a=panel(fig,[.105,.69,.205,.20],'a','Patient-specific drug ranking')
q=t[(t.panel=='e')&(t.metric=='kendall_tau_b')];a.plot([0,1],q.value,color=LIGHT,lw=2);a.scatter([0,1],q.value,color=[GRAY,TEAL],s=30)
a.set_xticks([0,1],['Population','Individual']);a.set_xlim(-.35,1.35);a.set_ylim(.53,.665);a.set_ylabel('Mean Kendall τ')
for i,v in enumerate(q.value):a.annotate(f'{v:.3f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
b=panel(fig,[.105,.36,.205,.20],'b','Positive functional benefit')
q=t[(t.panel=='e')&(t.metric.isin(['PG_macro','g_func']))];err(b,q,col=TEAL);b.axhline(0,color=GRAY,lw=.6);b.set_xticks([0,1],['Ranking\ngain','Functional\nrecovery']);b.set_xlim(-.5,1.5);b.set_ylabel('Gain over reference');b.set_ylim(-.02,.30)
curve=[]
for rect,metric,letter,ttl,color in [([.47,.69,.495,.20],'PG_macro','c','Compact inputs retain ranking gain',BLUE),([.47,.36,.495,.20],'g_func','d','Compact inputs retain functional recovery',TEAL)]:
    ax=panel(fig,rect,letter,ttl);curve.append(ax);q=t[(t.panel=='b')&(t.metric==metric)];pc=q[q.object_id.str.startswith('PCA')].copy();pc['k']=pc.object_id.str[3:].astype(int);pc=pc.sort_values('k')
    assert pc.ci_low.notna().all() and pc.ci_high.notna().all()
    ax.fill_between(pc.k,pc.ci_low,pc.ci_high,color=color,alpha=.16,lw=0);ax.plot(pc.k,pc.value,'o-',color=color,ms=3.5,lw=1)
    full=q[q.object_id=='FULL_RNA_REOPT'].iloc[0];ax.axhline(full.value,color=CORAL,ls='--',lw=1)
    ax.set_xlim(1,33);ax.set_xticks([2,8,16,24,32]);ax.set_xlabel('Baseline RNA components');ax.set_ylabel('Ranking gain' if metric=='PG_macro' else 'Functional recovery')
    ax.annotate('Full RNA: 19,421 genes',xy=(.98,full.value),xycoords=ax.get_yaxis_transform(),xytext=(0,-4),textcoords='offset points',color=CORAL,ha='right',va='top',fontsize=6.5)
e=panel(fig,[.105,.075,.265,.12],'e','14 pathways versus 14 PCs')
q=sf[(sf.panel=='b')&sf.object_id.isin(['PROGENY14','PCA14','FULL_RNA'])].set_index('object_id').loc[['PROGENY14','PCA14','FULL_RNA']]
e.bar(range(3),q.value,color=[CORAL,BLUE,GRAY],width=.52);e.set_xticks(range(3),['14 pathways','14 PCs','Full RNA\nreference']);e.set_ylim(0,.17);e.set_ylabel('Functional recovery');e.set_yticks([0,.1])
f=panel(fig,[.55,.075,.415,.12],'f','Residual RNA restores useful information')
q=sf[(sf.panel=='d')&sf.object_id.str.match(r'^COARSE_R\d+$')].copy();q['k']=q.object_id.str[8:].astype(int);q=q.sort_values('k')
f.plot(q.k,q.value,'o-',color=TEAL,ms=4,lw=1);f.axhline(0,color=GRAY,lw=.6);f.set_xticks([0,4,8,16,32]);f.set_xlabel('RNA components beyond pathway span');f.set_ylabel('Functional recovery');f.set_ylim(-.02,.145)
for ax in [a,b,*curve,e,f]:
    title=ax.get_title(loc='left');ax.set_title('',loc='left');ax.set_title(title,loc='center',pad=7)
fig.set_size_inches(7.0866141732,5.15)
for ax,rect in zip([a,b,*curve,e,f],[[0.1, 0.7, 0.245, 0.21], [0.1, 0.385, 0.245, 0.2], [0.46, 0.7, 0.51, 0.21], [0.46, 0.385, 0.51, 0.2], [0.1, 0.09, 0.295, 0.14], [0.515, 0.09, 0.455, 0.14]]):ax.set_position(rect)
save(fig,'Figure_4',[a,b,*curve,e,f],['a','b','c','d','e','f'],rows=[['a','c'],['b','d'],['e','f']],cols=[['a','b'],['c','d']])

