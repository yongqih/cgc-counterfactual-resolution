from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from layout import require_matplotlib_panel_alignment
HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[1]
FIG=Path(os.environ['CGC_FIGURE_OUTPUT']);QA=FIG/'qa'
FIG.mkdir(parents=True,exist_ok=True);QA.mkdir(exist_ok=True)
TABLES={
'Main_Figure_1':['Figure_1.csv'],
'Main_Figure_3':['Figure_2.csv'],
'Main_Figure_4':['Figure_3_LCL.csv','Figure_3_reproducibility.csv','Supplementary_Figure_3_span.csv'],
'Main_Figure_6':['Figure_3_Tahoe.csv','Figure_4_PDO.csv','Supplementary_Figure_4_random_controls.csv'],
'Extended_Data_Figure_6':['Figure_3_fidelity.csv','Extended_Data_2_pathways.csv'],
'Supplementary_Figure_5':['Figure_4_representation.csv']}
def read(name):return pd.concat([pd.read_csv(RELEASE/'source_data'/f,low_memory=False) for f in TABLES[name]],ignore_index=True)
BLUE='#718BAC';TEAL='#65A39E';CORAL='#D27D6D';GRAY='#979FAA';LIGHT='#E3E7EC';DARK='#292D33';GOLD='#C5A055'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,
 'axes.titlesize':8,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'xtick.labelsize':6.5,'ytick.labelsize':6.5,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','text.color':DARK,'axes.labelcolor':DARK})
def panel(fig,rect,letter,title):
    ax=fig.add_axes(rect)
    if letter:ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-21,6),textcoords='offset points',fontsize=10,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
    ax.set_title(title,loc='left',pad=7);return ax
def save(fig,name,axes,ids,rows=None,cols=None,exemptions=()):
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),overlay_svg=QA/(name+'.alignment.svg'),
        axes=axes,panel_ids=ids,row_groups=rows or [],column_groups=cols or [],exemptions=exemptions,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.svg'))
    fig.savefig(FIG/(name+'.png'),dpi=600);fig.savefig(FIG/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
def err(ax,q,x=None,col=BLUE):
    vals=np.asarray(q.value);x=np.arange(len(q)) if x is None else x
    ax.errorbar(x,vals,yerr=[vals-q.ci_low.to_numpy(),q.ci_high.to_numpy()-vals],fmt='o',color=col,ms=4,capsize=2,lw=.9)
def names(ax):ax.set_xticks(range(3),['Ridge','Bilinear','MLP']);ax.set_xlim(-.55,2.55)
def summary(ax,s,kind,ci=True):
    q=s[s.kind==kind].set_index('model').loc[['Ridge','Bilinear','MLP']]
    if ci:err(ax,q,col=CORAL if kind=='operator_alpha' else BLUE)
    else:ax.plot(range(3),q.value,'o',color=GRAY,ms=4)
    names(ax)

