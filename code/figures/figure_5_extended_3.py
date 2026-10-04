from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
OUT=ROOT
import os
FIG=Path(os.environ['CGC_FIGURE_OUTPUT']);QA=FIG/'qa'
FIG.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
from layout import require_matplotlib_panel_alignment
BLUE='#718BAC';TEAL='#65A39E';CORAL='#D27D6D';GRAY='#979FAA';DARK='#292D33'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','text.color':DARK,'axes.labelcolor':DARK})
def panel(fig,rect,label,title,labelx=None):
 ax=fig.add_axes(rect);ax.set_title(title,loc='center',pad=8)
 fig.text(rect[0]-.055 if labelx is None else labelx,rect[1]+rect[3]+8/(72*fig.get_figheight()),label,fontsize=10,fontweight='bold',va='bottom')
 return ax
def save(fig,name,axes,ids,rows,cols=[]):
 exemptions=[{'panels':g,'checks':['panel-width','horizontal-gutter'],'reason':'Deliberate unequal-width evidence hierarchy; categorical y labels require additional gutter.'} for g in rows]
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),overlay_svg=QA/(name+'.alignment.svg'),axes=axes,panel_ids=ids,row_groups=rows,column_groups=cols,exemptions=exemptions,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.svg'))
 fig.savefig(FIG/(name+'.png'),dpi=600);fig.savefig(FIG/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(QA/(name+'.preview.png'),dpi=300);plt.close(fig)
def herror(ax,x,y,lo,hi,**kw):
 assert np.isfinite([x,y,lo,hi]).all() and lo<=x<=hi
 ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],capsize=2,lw=.8,**kw)
lar=ROOT/'experiments/baseline_sufficiency_data_audit_20261002/larry'
ct=ROOT/'experiments/celltag_information_resolution_20261001/outputs_six_resolution'
closure=ROOT/'experiments/larry_information_scale_20261003'
clones=pd.read_csv(OUT/'source_data/Figure_5_LARRY_clone_estimates.csv');agg=pd.read_csv(OUT/'source_data/Figure_5_LARRY_aggregate_estimates.csv')
spec=pd.read_csv(OUT/'source_data/Figure_5_CellTag_specificity.csv');contr=pd.read_csv(OUT/'source_data/Figure_5_CellTag_paired_contrasts.csv');s=pd.read_csv(OUT/'source_data/Figure_5_CellTag_summary.csv')
models=[('LR',BLUE,'o',-.10),('RF',TEAL,'s',.10)]
conds=['RNA','ATAC','RNA_ATAC','RNA_unrelated_ATAC']
handles=[Line2D([],[],color=c,marker=m,ls='none',label=n) for n,c,m,_ in models]
fig=plt.figure(figsize=(7.0866141732,6.5))
a=panel(fig,[.095,.913,.865,.03],'a','LARRY · One early record, two realized outcomes',.04);a.axis('off')
for x,t in [(.14,'Day 2 · clonal RNA'),(.51,'Split culture'),(.87,'Day 6 · two wells')]:a.text(x,.3,t,ha='center',va='center',transform=a.transAxes)
for x1,x2 in [(.31,.41),(.64,.73)]:a.annotate('',xy=(x2,.3),xytext=(x1,.3),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GRAY,'lw':.9})
b=panel(fig,[.095,.586,.235,.232],'b','Clone 1978 · descendant fates',.04)
lib=pd.read_csv(OUT/'source_data/Figure_5_library_composition.csv');lib=lib[lib.clone_index==1978].sort_values(['Well','Library'])
assert lib.n_cells.tolist()==[10,7,17,16]
xs=np.array([0,.85,2.1,2.95]);bottom=np.zeros(4)
for col,color,label in [('n_Neutrophil',BLUE,'Neutrophil'),('n_Monocyte',CORAL,'Monocyte'),('n_Undifferentiated',GRAY,'Undifferentiated')]:
 v=lib[col].to_numpy()/lib.n_cells.to_numpy();b.bar(xs,v,bottom=bottom,width=.64,color=color,label=label);bottom+=v
assert np.allclose(bottom,1)
b.set_xticks(xs,['A','B','A','B']);b.set_ylim(0,1.15);b.set_yticks([0,.5,1]);b.set_ylabel('Cell fraction');b.set_xlim(-.6,3.55)
for x,n in zip(xs,lib.n_cells):b.text(x,1.04,str(n),ha='center',fontsize=6.5)
for x,t in [(.25,'Well 1'),(.75,'Well 2')]:b.text(x,-.19,t,transform=b.transAxes,ha='center')
fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in [(BLUE,'Neutrophil'),(CORAL,'Monocyte'),(GRAY,'Undifferentiated')]],loc='center',bbox_to_anchor=(.38,.87),ncol=3,fontsize=6.5,handlelength=1,columnspacing=1.5)
c=panel(fig,[.435,.586,.30,.232],'c','Shared-record error · 21 clones',.373)
q=clones[clones.variant=='raw'].sort_values(['culture_family','clone_index']);pos=0;xt=[];xl=[]
for family,color,label in [('LK',BLUE,'LK\n11 clones'),('LK_second',TEAL,'LK second\n8 clones'),('LSK',GRAY,'LSK\n2 clones')]:
 g=q[q.culture_family==family];xx=np.arange(len(g))+pos
 for x,(_,r) in zip(xx,g.iterrows()):
  col=CORAL if r.clone_index==1978 else color
  c.errorbar(x,r.estimate,yerr=[[r.estimate-r.ci95_low],[r.ci95_high-r.estimate]],fmt='o',color=col,ms=3,capsize=1.3,lw=.75)
  if r.clone_index==1978:c.text(x,r.ci95_high+2,'1978',ha='center',fontsize=6.5,color=CORAL)
 xt.append(xx.mean());xl.append(label);pos=xx[-1]+2.3
c.axhline(0,color=GRAY,lw=.65);c.set_xticks(xt,xl);c.tick_params(axis='x',length=0,pad=4);c.set_xlim(-1,pos-.25);c.set_ylim(-12,61);c.set_yticks([0,20,40,60]);c.set_ylabel('Squared RNA error')
assert q.ci95_low.min()>-12 and q.ci95_high.max()<61
c.text(.99,.92,'19/21 positive',ha='right',transform=c.transAxes,fontsize=6.5)
d=panel(fig,[.865,.586,.095,.232],'d','Error floor',.782)
rows=[('raw','clone_equal','normalized_floor','All 21'),('raw','exclude1978','normalized_floor','No 1978'),('barcode','clone_equal','normalized_floor','Barcode'),('reference','clone_equal','normalized_floor','Reference'),('raw','clone_equal','well_centered_normalized_floor','Well shift')]
for j,(v,ag,me,t) in enumerate(rows):
 r=agg[(agg.variant==v)&(agg.aggregation==ag)&(agg.metric==me)].iloc[0];herror(d,r.estimate*100,4-j,r.ci95_low*100,r.ci95_high*100,fmt='o',color=TEAL if j==0 else GRAY,ms=4 if j==0 else 3)
d.set_yticks(range(4,-1,-1),[r[3] for r in rows]);d.set_ylim(-.65,4.65);d.set_xlim(0,15);d.set_xticks([0,10]);d.set_xlabel('Variation (%)');d.spines['left'].set_visible(False);d.tick_params(axis='y',length=0)
e=panel(fig,[.095,.415,.865,.03],'e','CellTag-multi · Information and supported specificity',.04);e.axis('off')
for x,t in [(.14,'Early RNA / ATAC'),(.50,'Calibrate fate sets'),(.87,'Test on held-out clones')]:e.text(x,.3,t,ha='center',va='center',transform=e.transAxes)
for x1,x2 in [(.31,.39),(.65,.72)]:e.annotate('',xy=(x2,.3),xytext=(x1,.3),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GRAY,'lw':.9})
f=panel(fig,[.20,.09,.295,.235],'f','More informative inputs',.04)
conditions=['PRIOR']+conds;ys=np.arange(4,-1,-1)
for model,color,marker,off in models:
 q=spec[(spec.model==model)&(spec.nominal_coverage==.9)].set_index('condition').loc[conditions]
 f.errorbar(q.set_size,ys-off,xerr=[q.set_size-q.set_size_ci95_low,q.set_size_ci95_high-q.set_size],ls='none',marker=marker,color=color,ms=3.6,capsize=1.8,lw=.8)
 for y,v in zip(ys,q.coverage):f.text(.555 if model=='LR' else .616,y,f'{100*v:.1f}',ha='center',va='center',transform=matplotlib.transforms.blended_transform_factory(fig.transFigure,f.transData),color=color,fontsize=6.5)
f.set_yticks(ys,['Frequency only','RNA','ATAC','RNA + ATAC','RNA +\nshuffled ATAC']);f.set_ylim(-.6,4.6);f.set_xlim(1.55,3.5);f.set_xticks([2,2.5,3,3.5]);f.set_xlabel('Mean candidate fates · nominal 90%')
fig.text(.5855,.342,'Coverage (%)',ha='center',fontsize=6.5);fig.text(.555,.318,'LR',ha='center',fontsize=6.5,color=BLUE);fig.text(.616,.318,'RF',ha='center',fontsize=6.5,color=TEAL)
g=panel(fig,[.765,.09,.195,.235],'g','Reliability criterion',.70)
for model,color,marker,off in models:
 r=s[(s.model==model)&(s.condition=='RNA_ATAC')].iloc[0]
 g.plot(np.array([0,1])+off*.32,[r.size_90,r.size_cc90],marker=marker,color=color,ms=4,lw=.8)
g.set_xticks([0,1],['Across\nclones','Within\neach fate']);g.set_xlim(-.3,1.3);g.set_ylim(1,6.25);g.set_yticks([2,4,6]);g.set_ylabel('Mean candidate fates');g.text(.03,.92,'RNA + ATAC',ha='left',transform=g.transAxes,fontsize=6.5)
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.39,.373),ncol=2,fontsize=6.5,handlelength=1,columnspacing=2)
save(fig,'Figure_5',[a,b,c,d,e,f,g],list('abcdefg'),[['b','c','d'],['f','g']],[['a','e']])

# Merge the complementary lineage / calibration checks into a single ED figure.
cl=pd.read_csv(OUT/'source_data/Extended_Data_3_class_results.csv');fig=plt.figure(figsize=(7.0866141732,6.25))
a=panel(fig,[.15,.71,.28,.175],'a','LARRY · culture families',.04)
for v,color,marker,off in [('raw',BLUE,'o',.11),('reference',TEAL,'s',-.11)]:
 for y,family in zip([2,1,0],['LK','LK_second','LSK']):
  r=agg[(agg.variant==v)&(agg.aggregation=='family_'+family)&(agg.metric=='floor')].iloc[0];herror(a,r.estimate,y+off,r.ci95_low,r.ci95_high,fmt=marker,color=color,ms=4)
a.set_yticks([2,1,0],['LK · 11','LK second · 8','LSK · 2']);a.set_ylim(-.6,2.6);a.set_xlim(-2,12);a.set_xticks([0,5,10]);a.axvline(0,color=GRAY,ls='--',lw=.7);a.set_xlabel('Squared RNA error')
a.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='none',label='Raw'),Line2D([],[],color=TEAL,marker='s',ls='none',label='Reference')],loc='center',bbox_to_anchor=(.5,1.48),ncol=2,fontsize=6.5,handlelength=1)
b=panel(fig,[.615,.71,.345,.175],'b','CellTag · fate support',.54)
b.bar(range(6),[117,25,12,6,3,2],color=GRAY,width=.55);b.set_xticks(range(6),['M','N','P','E/M','B/E/M','L/D']);b.set_ylabel('Clones');b.set_ylim(0,130)
for x,v in enumerate([117,25,12,6,3,2]):b.text(x,v+3,str(v),ha='center',fontsize=6.5)
c=panel(fig,[.15,.395,.28,.19],'c','Probability scores',.04)
for model,color,marker,off in models:
 q=s[s.model==model].set_index('condition').loc[conds];c.plot(np.arange(4)+off,q.brier,ls='none',marker=marker,color=color,ms=4)
c.set_xticks(range(4),['RNA','ATAC','Joint','Shuffled']);c.set_ylabel('Brier score');c.set_ylim(.185,.225)
d=panel(fig,[.725,.395,.235,.19],'d','Matched-input set-size gain',.54)
for model,color,marker,off in models:
 for j,(comp,level) in enumerate([('RNA',80),('RNA',90),('RNA_unrelated_ATAC',80),('RNA_unrelated_ATAC',90)]):
  r=contr[(contr.model==model)&(contr.a=='RNA_ATAC')&(contr.b==comp)&(contr.metric==f'size_{level}')].iloc[0];herror(d,r.delta,3-j-off*1.4,r.ci95_low,r.ci95_high,fmt=marker,color=color,ms=3.8)
d.set_yticks([3,2,1,0],['RNA · 80%','RNA · 90%','Shuffled · 80%','Shuffled · 90%']);d.set_ylim(-.6,3.6);d.set_xlim(-.45,.025);d.set_xticks([-.4,-.2,0]);d.axvline(0,color=GRAY,ls='--',lw=.7);d.set_xlabel('Matched − comparator')
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.54,.64),ncol=2,fontsize=6.5,handlelength=1,columnspacing=2)
e=panel(fig,[.15,.09,.38,.18],'e','Coverage within each fate',.04)
for cond,color,marker,off,label in [('RNA',GRAY,'o',-.05,'RNA'),('RNA_ATAC',TEAL,'s',.05,'RNA + ATAC')]:
 q=cl[(cl.model=='RF')&(cl.condition==cond)].sort_values('y');e.plot(np.arange(6)+off,q.covered_90*100,ls='none',marker=marker,color=color,ms=4,label=label)
e.set_xticks(range(6),['M','N','P','E/M','B/E/M','L/D']);e.set_ylabel('Coverage (%) · RF');e.set_ylim(-5,110);e.axhline(90,color=GRAY,lw=.7,ls='--');e.legend(fontsize=6.5,loc='lower left')
f=panel(fig,[.725,.09,.235,.18],'f','Fate-conditional sets',.615)
for model,color,marker,off in models:
 q=s[s.model==model].set_index('condition').loc[conds];f.plot(np.arange(4)+off,q.size_cc90,ls='none',marker=marker,color=color,ms=4)
f.set_xticks(range(4),['RNA','ATAC','Joint','Shuffled']);f.set_ylim(5.6,6.02);f.set_yticks([5.6,5.8,6]);f.set_ylabel('Mean candidate fates')
save(fig,'Extended_Data_3',[a,b,c,d,e,f],list('abcdef'),[['a','b'],['c','d'],['e','f']],[['a','c'],['d','f']])
print('Rendered Figure 5 and its supporting figures.')
