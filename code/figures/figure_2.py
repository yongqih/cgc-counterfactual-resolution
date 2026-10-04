from common import *
# Frozen recovery data and the author's original vector schematic.
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
exec((HERE/'author_panel_a.py').read_text(encoding='utf-8'))
ec=read('Main_Figure_3');WPT=180/25.4*72;HPT=396
fig=plt.figure(figsize=(180/25.4,HPT/72))
a_height=(.415*WPT/201*113)
a_rect=[.055,251/HPT,.415,a_height/HPT]
a=panel(fig,a_rect,'a','Strict hide-first design');a.axis('off')
b=panel(fig,[.59,271/HPT,.313,(251+a_height-271)/HPT],'b','Recovery across measured support')
grid=ec[(ec.panel=='b')&(ec.k>0)].pivot(index='m',columns='k',values='value').sort_index().sort_index(axis=1)
assert grid.shape==(9,9) and grid.to_numpy().min()<0<grid.to_numpy().max()
cmap=LinearSegmentedColormap.from_list('signed_recovery',['#B84E5A','#FAFAF9','#3D78A8'])
norm=TwoSlopeNorm(vmin=float(grid.to_numpy().min()),vcenter=0,vmax=float(grid.to_numpy().max()))
im=b.pcolormesh(np.arange(10)-.5,np.arange(10)-.5,grid.to_numpy(),cmap=cmap,norm=norm,shading='flat',rasterized=False,linewidth=0)
b.set_xlim(-.5,8.5);b.set_ylim(8.5,-.5)
b.set_xticks(range(9),[str(int(x)) for x in grid.columns]);b.set_yticks(range(9),[str(int(x)) for x in grid.index]);b.set_xlabel('Target sentinels, k',labelpad=2);b.set_ylabel('Reference contexts, m',labelpad=2)
for m,k,col in [(40,4,GOLD),(49,92,CORAL)]:
    b.add_patch(plt.Rectangle((list(grid.columns).index(k)-.49,list(grid.index).index(m)-.49),.98,.98,fill=False,ec=col,lw=1.1))
cbax=fig.add_axes([.923,271/HPT,.013,(251+a_height-271)/HPT]);cb=fig.colorbar(im,cax=cbax,ticks=[-.10,-.05,0,.05,.10]);cb.solids.set_rasterized(False)
cbax.tick_params(labelsize=6);cbax.set_ylabel('Recovery g',fontsize=6.5,labelpad=2)
c=panel(fig,[.105,114/HPT,.86,104/HPT],'c','Detectable recovery requires extensive empirical support')
front=ec[(ec.panel=='c')&(ec.kind=='frontier')].copy()
pareto=front[front.flag.astype(str).str.lower().eq('true')]
assert len(front)==90 and len(pareto)==39 and (front.ci_low>=-.45).all()
c.scatter(front.x*100,front.value,s=7,color=GRAY,alpha=.22,linewidths=0,zorder=2)
c.vlines(pareto.x*100,pareto.ci_low,pareto.value,color=GRAY,alpha=.32,lw=.55,zorder=1)
c.scatter(pareto.x*100,pareto.value,s=11,color=GRAY,alpha=.72,linewidths=0,zorder=3)
c.axhline(0,color=GRAY,ls='--',lw=.6);c.set_xlim(0,101);c.set_ylim(-.45,.22);c.set_xticks([0,20,40,60,80,100]);c.set_yticks([-.4,-.2,0,.1]);c.set_xlabel('Measured perturbation entries (%)',labelpad=2);c.set_ylabel('Context-specific recovery g')
for m,k,col,txt in [(40,4,GOLD,'First detection\n80.09% measured'),(49,92,CORAL,'All-but-one')]:
    z=front[(front.m==m)&(front.k==k)].iloc[0]
    c.vlines(z.x*100,z.ci_low,z.value,color=col,lw=1,zorder=4)
    c.plot(z.x*100,z.ci_low,marker='_',color=col,ms=5,mew=1,zorder=5)
    c.plot(z.x*100,z.value,'o',color=col,ms=4.4,zorder=5)
    c.annotate(txt,(z.x*100,z.value),xytext=(-3,7) if k==92 else (0,7),textcoords='offset points',ha='right' if k==92 else 'center',fontsize=6.3,color=DARK)
d=panel(fig,[.105,22/HPT,.25,54/HPT],'d','All-but-one remains context-limited')
dd=ec[(ec.panel=='d')&(ec.kind=='all_but_one_summary')].set_index('label').loc[['Context-specific','Full response']]
d.plot([0,1],dd.value,'-',color=LIGHT,lw=1.5,zorder=1)
d.scatter([0,1],dd.value,color=[GRAY,TEAL],s=23,zorder=4)
lower=float(dd.iloc[0].ci_low);value=float(dd.iloc[0].value)
d.vlines(0,lower,value,color=CORAL,lw=1,zorder=2);d.plot(0,lower,marker='_',color=CORAL,ms=5,mew=1,zorder=3)
d.set_xticks([0,1],['Context-specific','Full response']);d.set_xlim(-.4,1.4);d.set_ylim(0,.84);d.set_ylabel('Recovery g');d.set_yticks([0,.4,.8])
for i,v in enumerate(dd.value):d.annotate(f'{v:.3f}',(i,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=6.5)
e=panel(fig,[.56,22/HPT,.405,54/HPT],'e','Matched completion models')
q=ec[ec.panel=='e'].copy();yy=[3,2,1,0];labs=['Affine Ridge','Additive','Low-rank','Low-rank − additive']
for pos,lab,(_,row),col in zip(yy,labs,q.iterrows(),[BLUE,BLUE,TEAL,CORAL]):
    e.errorbar(row.value,pos,xerr=[[row.value-row.ci_low],[row.ci_high-row.value]],fmt='o',color=col,ms=3.5,capsize=2,lw=.8)
e.set_yticks(yy,labs);e.set_ylim(-.5,3.5);e.set_xlim(-.15,.48);e.axvline(0,color=GRAY,ls='--',lw=.6);e.set_xticks([-.1,0,.2,.4]);e.set_xlabel('Context-specific recovery g',labelpad=2)
for ax in [a,b,c,d,e]:
    subtitle=ax.get_title(loc='left');ax.set_title('',loc='left');ax.set_title(subtitle,loc='center',pad=7)
save(fig,'Figure_2',[a,b,c,d,e],list('abcde'),rows=[['d','e']])
insert_author_panel('Figure_2',a_rect,WPT,HPT)
