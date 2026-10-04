from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
import os
FIG=Path(os.environ['CGC_FIGURE_OUTPUT']);QA=FIG/'qa'
FIG.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
from layout import require_matplotlib_panel_alignment
BLUE='#718BAC';TEAL='#65A39E';CORAL='#D27D6D';GRAY='#979FAA';LIGHT='#E3E7EC';DARK='#292D33';GOLD='#C5A055'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,
 'axes.titlesize':8,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'xtick.labelsize':6.5,'ytick.labelsize':6.5,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','text.color':DARK,'axes.labelcolor':DARK})

DATA_MAP={
 'Extended_Data_Figure_2':'Supplementary_Figure_2_integrity.csv',
 'Extended_Data_Figure_3':'Supplementary_Figure_2_interaction.csv',
 'Extended_Data_Figure_5':'Supplementary_Figure_3_LCL.csv',
 'Main_Figure_4':'Supplementary_Figure_3_span.csv',
 'Extended_Data_Figure_8':'Supplementary_Figure_4_application.csv',
 'Main_Figure_6':'Supplementary_Figure_4_random_controls.csv'}
def read(name):return pd.read_csv(PACKAGE/'source_data'/DATA_MAP[name],low_memory=False)
def panel(fig,rect,letter,title):
    ax=fig.add_axes(rect)
    if letter:ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-21,6),textcoords='offset points',fontsize=10,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
    ax.set_title(title,loc='center',pad=7)
    return ax
def save(fig,name,axes,ids,rows=None,cols=None,exemptions=()):
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),overlay_svg=QA/(name+'.alignment.svg'),
        axes=axes,panel_ids=ids,row_groups=rows or [],column_groups=cols or [],exemptions=exemptions,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    title_checks=[]
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax,letter in zip(axes,ids):
        assert ax.get_title() and not ax.get_title(loc='left')
        title_bbox=ax.title.get_window_extent(renderer);axis_bbox=ax.get_window_extent(renderer)
        delta=abs((title_bbox.x0+title_bbox.x1-axis_bbox.x0-axis_bbox.x1)/2)*72/fig.dpi
        assert delta<0.05,(name,letter,delta)
        title_checks.append({'panel':letter,'title':ax.get_title(),'center_offset_pt':delta})
    (QA/(name+'.titles.json')).write_text(json.dumps(title_checks,indent=2),encoding='utf-8')
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.svg'))
    fig.savefig(FIG/(name+'.png'),dpi=600)
    fig.savefig(FIG/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(QA/(name+'.preview.png'),dpi=300)
    plt.close(fig)
lcl=read('Main_Figure_4')
random_controls=read('Main_Figure_6')
pdo=pd.read_csv(PACKAGE/'source_data/Figure_4_PDO.csv',low_memory=False)
t=pd.concat([random_controls,pdo[pdo.panel=='b']],ignore_index=True)

# Same construction, counts, donor partitions and information set as the original Supplementary Figure 1.
from matplotlib.patches import FancyBboxPatch
def box(ax,x,y,w,h,txt,color,fs=7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.018',facecolor=color,edgecolor='none',transform=ax.transAxes))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,transform=ax.transAxes,linespacing=1.3)
fig=plt.figure(figsize=(7.0866141732,3.85))
a=panel(fig,[.055,.64,.91,.24],'a','Responses are constructed at pseudobulk resolution');a.axis('off')
for x,w,txt,color in [(0,.19,'Guide-assigned\ncells','#F1F3F5'),(.25,.23,'Donor × state × run × guide\npseudobulk','#E5EDF4'),(.54,.20,'Matched-NTC\nresponse','#DFEFEC'),(.80,.20,'Context-specific\noperator','#F3E7E2')]:
    box(a,x,.45,w,.46,txt,color,6.8)
for start,end in [(.195,.245),(.485,.535),(.745,.795)]:a.annotate('',xy=(end,.68),xytext=(start,.68),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GRAY,'lw':.8})
a.text(.5,.19,r'$Y_{p,c}=X_c+\Delta_{p,c}$     $\Delta_{p,c}=\bar{\Delta}_p+\Gamma_{p,c}$     $\Delta_{p,d,s,r}=\mathrm{PB}_{p,d,s,r}-\mathrm{NTC}_{d,s,r}$',transform=a.transAxes,ha='center',fontsize=10,color=DARK)
a.text(.5,-.055,'Cells construct pseudobulks; downstream inference uses aggregated response vectors.',transform=a.transAxes,ha='center',fontsize=6.8,color=DARK)
b=panel(fig,[.055,.11,.27,.395],'b','Frozen construction universe');b.axis('off')
for x,txt in [(0,'4\ndonors'),(.345,'3\nstates'),(.69,'2\nruns')]:box(b,x,.70,.30,.27,txt,'#E5EDF4',8)
for y,n,txt,color in [(.49,9082,'strict trans genes',TEAL),(.29,3186,'truth targets',GOLD),(.09,9386,'evaluation targets',BLUE)]:
    b.add_patch(plt.Rectangle((0,y),n/10000,.10,color=color,alpha=.7,ec='none',transform=b.transAxes))
    b.text(.015,y+.05,f'{n:,}',transform=b.transAxes,va='center',fontsize=6.8,fontweight='bold')
    b.text(.30,y+.05,txt,transform=b.transAxes,va='center',fontsize=6.8)
c=panel(fig,[.38,.11,.275,.395],'c','Replicate-disjoint\ndonor partitions');c.axis('off')
for y,k,l,r in [(.70,1,'CE0006864\nCE0008162','CE0008678\nCE0010866'),(.40,2,'CE0006864\nCE0008678','CE0008162\nCE0010866'),(.10,3,'CE0006864\nCE0010866','CE0008162\nCE0008678')]:
    c.text(.015,y+.11,str(k),va='center',ha='center',transform=c.transAxes,fontsize=7)
    box(c,.10,y,.39,.22,l,'#DFEFEC',6.5);box(c,.61,y,.39,.22,r,'#F3E7E2',6.5)
    c.annotate('',xy=(.60,y+.11),xytext=(.50,y+.11),xycoords='axes fraction',arrowprops={'arrowstyle':'<->','color':GRAY,'lw':.7})
c.text(.52,-.075,'NTC matched within donor × state × run\nEqual guide weighting · three states',ha='center',va='top',transform=c.transAxes,fontsize=6.3,linespacing=1.2)
d=panel(fig,[.71,.11,.255,.395],'d','Strict LODO information set');d.axis('off')
box(d,0,.55,1,.42,'Allowed\nHeld-out donor NTC\nTraining-donor responses\nTraining-derived basis','#DFEFEC',6.8)
box(d,0,.07,1,.42,'Forbidden\nHeld-out perturbation outcomes\nResponse-derived preprocessing\nOutcome-based tuning','#F3E7E2',6.8)
d.text(.5,-.025,'4 outer folds\nCovered interventions; unseen donor context\nTruth reliability and evaluation stay separate',ha='center',va='top',transform=d.transAxes,fontsize=6.3,linespacing=1.2)
save(fig,'Supplementary_Figure_1',[a,b,c,d],list('abcd'),rows=[['b','c','d']],exemptions=[{'panels':['b','c','d'],'checks':['panel-width','horizontal-gutter'],'reason':'Three explanatory columns have distinct text densities; their final edges and heights align.'}])

import textwrap
ORANGE=CORAL; RED=CORAL
plt.rcParams.update({'axes.prop_cycle':matplotlib.cycler(color=[BLUE,CORAL]),'lines.markersize':3.5,'lines.linewidth':1,'legend.fontsize':6.5})
def layout(n=4):
    f=plt.figure(figsize=(7.0866141732,4.05))
    return f,[f.add_axes(r) for r in [[.105,.59,.37,.28],[.60,.59,.37,.28],[.105,.13,.37,.28],[.60,.13,.37,.28]]]
def title(ax,letter,txt):
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-21,6),textcoords='offset points',fontsize=10,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
    ax.set_title(textwrap.fill(txt,width=33),loc='center',pad=7)
def bar(ax,labels,values,colors=None):
    ax.bar(np.arange(len(labels)),values,color=colors or BLUE,width=.6)
    ax.set_xticks(np.arange(len(labels)),labels);ax.axhline(0,color=GRAY,lw=.6)
def points(ax,labels,values,lo=None,hi=None,color=BLUE):
    x=np.arange(len(labels));y=np.asarray(values,float)
    if lo is None:ax.plot(x,y,'o',color=color)
    else:ax.errorbar(x,y,yerr=[y-np.asarray(lo),np.asarray(hi)-y],fmt='o',color=color,capsize=2)
    ax.set_xticks(x,labels);ax.set_xlim(-.6,len(labels)-.4);ax.axhline(0,color=GRAY,lw=.6)
def info(ax,txt):
    ax.axis('off');ax.text(.02,.95,txt,va='top',ha='left',transform=ax.transAxes,fontsize=7,linespacing=1.25)
def support_save(f,name):save(f,name,f.axes,list('abcd'),rows=[['a','b'],['c','d']],cols=[['a','c'],['b','d']])
e2=read('Extended_Data_Figure_2');e3=read('Extended_Data_Figure_3');f,a=layout(4)
title(a[0],'a','Hidden-target integrity checks');info(a[0],'Five target-integrity challenges\n× four fitted-object checks\n\n20 / 20 unchanged\n\nSupport, sentinel identities, tuning\nand predictions remain target-excluding.')
title(a[1],'b','Shared-response positive control');z=e2[e2.panel=='b'];bar(a[1],['Observed recovery','Null upper bound'],z.value,[BLUE,GRAY]);a[1].set_ylabel('Recovery g')
title(a[2],'c','Synthetic response calibration');z=e2[e2.panel=='c'];a[2].plot(z.x*100,z.value,'o-',color=BLUE);a[2].set_xlabel('Measured matrix entries (%)');a[2].set_ylabel('Recovery g')
title(a[3],'d','Known rank-4 interaction control');z=e3[e3.kind=='synthetic_rank4'];bar(a[3],['Additive','CP interaction'],z.value,[GRAY,BLUE]);a[3].set_ylabel('Recovery g');a[3].set_ylim(-.08,1.12)
support_save(f,'Supplementary_Figure_2')

e5=read('Extended_Data_Figure_5');f,a=layout(4)
title(a[0],'a','Independent RNA reprocessing')
for group,z in e5[e5.panel=='b'].groupby('group',sort=False):a[0].plot(range(len(z)),z.value,'o-',label='Post-SVA' if 'Original' in group else 'ARCHS4')
a[0].set_xticks(range(4),['Full genes','16 PCs','8 PCs','4 PCs']);a[0].set_ylabel('Held-out R²');a[0].legend(fontsize=6.5)
title(a[1],'b','Within-ancestry full-gene prediction')
for gi,(group,z) in enumerate(e5[e5.panel=='c'].groupby('group',sort=False)):a[1].plot(np.arange(len(z))+.08*gi,z.value,'o-',label=group)
a[1].set_xticks(range(3),['Ridge','PCA Ridge','RBF']);a[1].set_ylabel('Held-out full-gene R²');a[1].axhline(0,color=GRAY,lw=.6);a[1].legend()
title(a[2],'c','Directional sensitivity, incomplete recovery')
for group,z in e5[e5.kind=='semisynthetic_signal_sweep'].groupby('label'):a[2].plot(z.x,z['median'],'o-',label='RBF' if 'rbf' in group else group)
a[2].axhline(.25,color=RED,ls='--',label='Prespecified success threshold');a[2].set_xlabel('Encoded baseline dependence');a[2].set_ylabel('Held-out R²');a[2].set_ylim(-.015,.285);a[2].legend(fontsize=6.5)
title(a[3],'d','Training span transfers little response detail');z=lcl[lcl.panel=='e'];bar(a[3],['256 PCs','Full training span'],z.value*100,[BLUE,ORANGE]);a[3].set_ylabel('Held-out response fidelity (%)');a[3].set_ylim(0,8);a[3].text(.5,7.4,'Training variance retained: 99.3% / 100%',ha='center',fontsize=6.5)
support_save(f,'Supplementary_Figure_3')

e8=read('Extended_Data_Figure_8');f,a=layout(4)
title(a[0],'a','Top-k drug overlap');z=e8[e8.panel=='a'].groupby('k')[['model_overlap_fraction','population_overlap_fraction']].mean();a[0].plot(z.index,z.model_overlap_fraction,'o-',color=BLUE,label='Personalized');a[0].plot(z.index,z.population_overlap_fraction,'o-',color=GRAY,label='Population');a[0].set_xlabel('Top k drugs');a[0].set_ylabel('Mean overlap');a[0].set_xticks([1,3,5]);a[0].legend(fontsize=6.5)
title(a[1],'b','Patient-level benefit is heterogeneous');z=e8[e8.panel=='d'].sort_values('PG_i');bar(a[1],['']*len(z),z.PG_i,[BLUE if x>0 else GRAY for x in z.PG_i]);a[1].set_xlabel('52 held-out patients, sorted');a[1].set_ylabel('Ranking gain');a[1].set_xticks([])
title(a[2],'c','Secondary expression platform');z=e8[(e8.panel=='e')&(e8.row_type=='simultaneous_inference')];points(a[2],['HTA2.0: 50 patients'],z.estimate,z.simultaneous_95_ci_lower,z.simultaneous_95_ci_upper);a[2].set_ylabel('Mean ranking gain');a[2].set_xlim(-.6,.6)
title(a[3],'d','Dimension-matched random controls');z=t[(t.panel=='d')&(t.metric=='g_func')];groups=[]
for k in [16,24,32]:
    zz=z[z.label.astype(str).str.contains('k='+str(k))]
    if len(zz)!=100:zz=z[z.object_id.astype(str).str.contains('k'+str(k)+'_')]
    groups.append(zz.value.to_numpy())
if not all(len(x)==100 for x in groups):
    # Stable export labels encode the dimension in the object identifier.
    zz=z[z.object_id.astype(str).str.contains('random')]
    groups=[zz.iloc[i*100:(i+1)*100].value.to_numpy() for i in range(3)]
assert all(len(x)==100 for x in groups),z[['object_id','label']].head().to_dict()
a[3].boxplot(groups,positions=[0,1,2],widths=.4,showfliers=False);pp=t[(t.panel=='b')&(t.metric=='g_func')].set_index('object_id').loc[['PCA16','PCA24','PCA32']];a[3].plot([0,1,2],pp.value,'o',color=BLUE,label='RNA PCs');a[3].set_xticks([0,1,2],['16','24','32']);a[3].set_xlabel('Representation dimensions');a[3].set_ylabel('Functional recovery');a[3].legend(fontsize=6.5)
support_save(f,'Supplementary_Figure_4')

