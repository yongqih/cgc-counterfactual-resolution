from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
OUT = H.parents[1]
import os
FIG=Path(os.environ['CGC_FIGURE_OUTPUT']); QA=FIG/'qa'
FIG.mkdir(exist_ok=True); QA.mkdir(exist_ok=True)
from layout import require_matplotlib_panel_alignment
BLUE='#718BAC'; CORAL='#D27D6D'; GRAY='#979FAA'; DARK='#292D33'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'xtick.labelsize':6.5,'ytick.labelsize':6.5,'xtick.major.width':.6,
    'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
    'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','text.color':DARK,'axes.labelcolor':DARK})

def panel(fig, rect, label, title, labelx):
    ax=fig.add_axes(rect)
    ax.set_title(title,loc='center',pad=8)
    fig.text(labelx,rect[1]+rect[3]+8/(72*fig.get_figheight()),label,
             fontsize=10,fontweight='bold',va='bottom')
    return ax

def recovery(ax, q, pooled, labels):
    assert len(q)==5 and np.isfinite(q.value).all()
    ax.barh(range(5,0,-1),q.value,
            color=[BLUE if x>0 else CORAL for x in q.value],height=.58)
    assert pooled.ci_low <= pooled.value <= pooled.ci_high
    ax.errorbar(pooled.value,0,
                xerr=[[pooled.value-pooled.ci_low],[pooled.ci_high-pooled.value]],
                fmt='D',color=DARK,ms=4,capsize=2,lw=.8)
    ax.set_yticks(range(5,-1,-1),labels+['Pooled'])
    ax.set_xlim(-2,1.05); ax.set_xticks([-2,-1,0,1]); ax.set_ylim(-.5,5.55)
    ax.set_xlabel('Recovery g · ST-SE')
    ax.axvline(0,color=GRAY,lw=.6)
    assert min(q.value.min(),pooled.ci_low)>-2
    assert max(q.value.max(),pooled.ci_high)<1.05

st=pd.read_csv(OUT/'source_data/Extended_Data_1_cytokine.csv')
drug=pd.read_csv(OUT/'source_data/Extended_Data_1_drug.csv')
fig=plt.figure(figsize=(7.0866141732,5.05))
a=panel(fig,[.105,.63,.235,.245],'a','Cytokine response direction',.04)
q=st[(st.panel=='a')&(st.kind=='delta_pearson')].sort_values('value')
assert len(q)==2
a.bar([0,1],q.value,color=[GRAY,BLUE],width=.55)
a.set_xticks([0,1],['Mean','State']); a.set_ylabel('Δ-Pearson'); a.set_ylim(0,.6)
b=panel(fig,[.505,.63,.455,.245],'b','Cytokine context geometry',.43)
q=st[st.kind=='context_distance_pair']; assert len(q)==900
b.scatter(q.x,q.value,s=4,alpha=.3,color=BLUE)
b.set_xlim(0, q.x.max()*1.05)
b.set_ylim(0, q.value.max()*1.05)
identity_max=min(b.get_xlim()[1], b.get_ylim()[1])
b.plot([0,identity_max],[0,identity_max],color=GRAY,ls='--',lw=.8,zorder=1)
b.text(22,24.2,'y = x',color=GRAY,fontsize=6.5,ha='center')
b.set_xlabel('Observed context distance'); b.set_ylabel('Predicted context distance')
b.text(.06,.86,'Spearman ρ = 0.892',transform=b.transAxes)
c=panel(fig,[.105,.165,.235,.265],'c','Cytokine context recovery',.04)
q=st[st.kind=='context_g']; pooled=st[st.kind=='operator_g'].iloc[0]
assert q.label.tolist()==['B_Intermediate_Memory','B_Naive','Plasmablast','CD4_Memory','CD14_Mono']
recovery(c,q,pooled,['B mem.','B naive','Plasmablast','CD4 mem.','CD14 mono'])
d=panel(fig,[.505,.165,.235,.265],'d','Tahoe context recovery',.43)
q=drug[drug.kind=='context_g'].sort_values('value',ascending=False)
recovery(d,q,drug[drug.kind=='pooled_g'].iloc[0],q.label.tolist())
e=panel(fig,[.86,.165,.10,.265],'e','Sensitivity',.79)
q=drug[(drug.panel=='b')&drug.ci_low.notna()]; assert len(q)==2
e.errorbar([0,1],q.value,yerr=[q.value-q.ci_low,q.ci_high-q.value],
            fmt='o',color=GRAY,ms=4,capsize=2,lw=.9)
e.set_xticks([0,1],['ST-SE','ST-HVG'],rotation=45,rotation_mode='anchor',ha='right')
e.set_xlim(-.6,1.6)
e.set_ylabel('Pooled recovery g'); e.axhline(0,color=GRAY,lw=.6); e.set_ylim(-1.5,.5)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=QA/'Extended_Data_1.alignment.json',
    overlay_svg=QA/'Extended_Data_1.alignment.svg',axes=[a,b,c,d,e],panel_ids=list('abcde'),
    row_groups=[['a','b'],['c','d','e']],column_groups=[['a','c']],
    exemptions=[{'panels':['a','b'],'checks':['panel-width','horizontal-gutter'],
                 'reason':'The context-geometry scatter is the wider evidence panel.'},
                {'panels':['e'],'checks':['panel-width','horizontal-gutter'],
                 'reason':'Two sensitivity estimates need less width than five-context recovery panels.'}],
    tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(FIG/'Extended_Data_1.pdf')
fig.savefig(FIG/'Extended_Data_1.svg')
fig.savefig(FIG/'Extended_Data_1.png',dpi=600)
fig.savefig(FIG/'Extended_Data_1.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(QA/'Extended_Data_1.preview.png',dpi=300)
plt.close(fig)
print('Exported Extended Data Figure 1 from frozen source values.')

# Extended Data Figure 2: preserve all plotted values and panel geometry; center titles.
TEAL='#65A39E'
e7=pd.read_csv(OUT/'source_data/Extended_Data_2_observed_scaling.csv',low_memory=False)
fid=pd.read_csv(OUT/'source_data/Extended_Data_2_pathways.csv',low_memory=False)
joint=pd.read_csv(OUT/'source_data/Extended_Data_2_resolution_grid.csv')
sup=json.loads((OUT/'source_data/Extended_Data_2_gene_selection.json').read_text(encoding='utf-8'))
fig=plt.figure(figsize=(7.0866141732,4.35))
def edpanel(rect,letter,title):
    ax=fig.add_axes(rect)
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-21,6),
                textcoords='offset points',fontsize=10,fontweight='bold',
                ha='left',va='bottom',annotation_clip=False)
    ax.set_title(title,loc='center',pad=7)
    return ax
a=edpanel([.1,.61,.3,.27],'a','Observed support scaling')
for p,metric,color,label in [('a','pooled_g',GRAY,'Genes'),('b','pooled_g_pathway',TEAL,'Pathways')]:
    q=e7[e7.panel==p].sort_values('m'); assert len(q)>0
    a.plot(q.m,q[metric],'o-',color=color,ms=3,label=label)
    a.fill_between(q.m,q.bootstrap_lower_95,q.bootstrap_upper_95,color=color,alpha=.14,lw=0)
a.set_xlabel('Reference contexts'); a.set_ylabel('Recovery g'); a.legend(fontsize=6.5)
b=edpanel([.54,.61,.43,.27],'b','Pathway choice changes recovery')
q=fid[fid.panel=='e'].sort_values('pathway_count'); assert len(q)==14
b.plot(q.pathway_count,q['median'],'o-',color=TEAL,ms=3)
b.fill_between(q.pathway_count,q.q25,q.q75,color=TEAL,alpha=.15,lw=0)
b.set_xlabel('Pathways in subset'); b.set_ylabel('Recovery g')
c=edpanel([.1,.15,.545,.29],'c','Full output can retain more full-target utility')
q=joint[(joint.model=='rbf')&(joint.input.astype(str)=='full')].copy()
q['rank']=pd.to_numeric(q.output,errors='coerce'); q=q[q['rank'].notna()].sort_values('rank')
assert np.all(q['rank'].to_numpy() > 0), 'Log-axis output ranks must be strictly positive'
c.plot(q['rank'],q.U,'o-',color=BLUE,ms=3)
v=joint[(joint.model=='rbf')&(joint.input.astype(str)=='full')&(joint.output.astype(str)=='full')].iloc[0].U
c.axhline(v,color=CORAL,ls='--',lw=.9,label='Full output')
c.set_xscale('log',base=2)
c.set_xticks([1,4,16,64,256,1024],['1','4','16','64','256','1,024'])
c.set_xlabel('Output response components'); c.set_ylabel('Full-gene utility U')
c.legend(fontsize=6.5,loc='lower right')
d=edpanel([.8,.15,.17,.29],'d','Gene selection\nloses utility')
q=pd.DataFrame(sup['contrasts']); q=q[q.contrast=='support_minus_full']
ci=np.array(q.simultaneous95.tolist()); assert len(q)==2
d.errorbar(range(2),q.estimate,yerr=[q.estimate-ci[:,0],ci[:,1]-q.estimate],
            fmt='o',color=CORAL,capsize=2,ms=4)
d.set_xticks(range(2),['Linear','RBF']); d.set_xlim(-.5,1.5)
d.axhline(0,color=GRAY,lw=.6); d.set_ylim(-.08,.01); d.set_ylabel('Selected − full output, ΔU')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=QA/'Extended_Data_2.alignment.json',
    overlay_svg=QA/'Extended_Data_2.alignment.svg',axes=[a,b,c,d],panel_ids=list('abcd'),
    row_groups=[['a','b'],['c','d']],tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(FIG/'Extended_Data_2.pdf')
fig.savefig(FIG/'Extended_Data_2.svg')
fig.savefig(FIG/'Extended_Data_2.png',dpi=600)
fig.savefig(FIG/'Extended_Data_2.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(QA/'Extended_Data_2.preview.png',dpi=300)
plt.close(fig)
print('Centered all Extended Data Figure 2 panel titles.')
