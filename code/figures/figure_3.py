from common import *
# Seven-panel Figure 3: two separate author schematics anchor two experiments.
lcl=read('Main_Figure_4');t=read('Main_Figure_6');fid=read('Extended_Data_Figure_6')
fig=plt.figure(figsize=(7.0866141732,5.7))
FIGURE_3_SHIFT_LEFT=0.03304327375352775
a=panel(fig,[.075,.79,.43,.14],'a','LCL paired-response prediction');a.axis('off')
b=panel(fig,[.61,.79,.35,.14],'b','Responses are reproducible')
q=lcl[lcl.kind=='lcl_truth_correlation'];assert len(q)==24
vals=q.value.to_numpy();b.scatter(vals,np.linspace(-.10,.10,24),s=13,color=BLUE,alpha=.8,zorder=3)
glob=lcl[lcl.kind=='global_truth_reliability'].iloc[0]
b.errorbar(glob.value,.72,xerr=[[glob.value-glob.ci_low],[glob.ci_high-glob.value]],fmt='D',color=TEAL,ms=4.5,capsize=2,lw=1)
b.set_yticks([0,.72],['24 LCLs','Global']);b.set_ylim(-.35,1.2);b.set_xlim(0,1);b.set_xticks([0,.5,1]);b.set_xlabel('Replicate response r')
b.text(.98,.92,f'{glob.value:.3f} [{glob.ci_low:.3f}, {glob.ci_high:.3f}]',transform=b.transAxes,ha='right',fontsize=6.5)
b.spines['left'].set_visible(True);b.tick_params(axis='y',length=2.5)
c=panel(fig,[.18,.49,.275,.205],'c','Full-gene prediction stays weak')
q=lcl[lcl.panel=='c'].iloc[[3,0,1,2,4,5]].copy();yy=np.arange(6)[::-1]
c.errorbar(q.value,yy,xerr=[q.value-q.ci_low,q.ci_high-q.value],fmt='o',ms=4,color=GRAY,lw=1,capsize=2)
c.set_yticks(yy,['RBF Ridge','Ridge','PCA Ridge','Pilot MLP','Boosting','Deep MLP']);c.axvline(0,color=GRAY,ls='--',lw=.7);c.set_xlim(-.0075,.0048);c.set_xticks([-.006,-.003,0,.003]);c.set_xlabel('Full-gene R²')
d=panel(fig,[.61,.49,.35,.205],'d','Selected programs recover more signal')
q=lcl[lcl.panel=='d'].set_index('label').loc[['Full gene vector','PC1-16','PC1-8','PC1-4']]
d.plot(range(4),q.value,'o-',color=TEAL,ms=5,lw=1.2);d.scatter([0],[q.iloc[0].value],color=GRAY,s=26,zorder=4)
d.set_xticks(range(4),['Full genes','16 PCs','8 PCs','4 PCs']);d.set_ylabel('R² in scored coordinates');d.set_ylim(-.015,.215);d.set_xlim(-.25,3.3)
for i,v in enumerate(q.value):d.annotate(f'{v:.3f}',(i,v),xytext=(9,-2) if i==0 else (0,8),textcoords='offset points',ha='left' if i==0 else 'center',fontsize=7)
e=panel(fig,[.1175,.19,.40,.17],'e','Same prediction, two resolutions');e.axis('off')
f=panel(fig,[.61,.075,.35,.285],'f','Fixed pathway readouts improve recovery')
q=t[(t.panel=='a')&(t.object_id=='resolution_point')];f.plot([0,1],q.value,color=TEAL,lw=1.5,zorder=1);f.scatter([0,1],q.value,color=[GRAY,TEAL],s=42,zorder=3)
f.set_xticks([0,1],['25,695 genes','5 pathways']);f.set_xlim(-.3,1.3);f.set_ylim(0,.34);f.set_ylabel('Tahoe recovery g');f.set_yticks([0,.1,.2,.3])
for i,v in enumerate(q.value):f.annotate(f'{v:.3f}',(i,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8,fontweight='bold')
f.text(.5,.12,'Paired Δg = 0.162\n95% interval [0.074, 0.250]',transform=f.transAxes,ha='center',fontsize=7,linespacing=1.4)
g=panel(fig,[.18,.075,.275,.055],'g','Full-response fidelity')
q=fid[(fid.panel=='f')&(fid.metric=='fidelity')].iloc[0];vv=[q.predefined_value*100,q.complete_14_pathway_value*100]
g.barh([1,0],[100,100],color=LIGHT,height=.56);g.barh([1,0],vv,color=TEAL,height=.56)
for i,v in zip([1,0],vv):g.text(v+3,i,f'{v:.2f}% retained',ha='left',va='center',fontsize=6.5)
g.set_yticks([1,0],['5 pathways','14 pathways']);g.set_xlim(0,100);g.set_ylim(-.55,1.55);g.set_xticks([0,50,100]);g.set_xlabel('Full-gene reproducible signal (%)')
for ax in [a,b,c,d,e,f,g]:
    title=ax.get_title(loc='left');ax.set_title('',loc='left');ax.set_title(title,loc='center',pad=7)
for ax,target_x in [(a,.10),(c,.10),(e,.10),(g,.10),(b,.565),(d,.565),(f,.565)]:
    letter=ax.texts[0]
    letter.set_position(((target_x-ax.get_position().x0)*fig.get_figwidth()*72,6))
# Uniform translation balances outer margins while preserving internal geometry.
for ax in [a,b,c,d,e,f,g]:
    rect=ax.get_position()
    ax.set_position([rect.x0-FIGURE_3_SHIFT_LEFT,rect.y0,rect.width,rect.height])
# The lower-right hero spans the two supporting panels on its left. Their
# unequal plot widths leave room for g's category labels; outer span edges align.
fig.canvas.draw()
span_checks={'lower_block_top_pt':abs(e.get_position().y1-f.get_position().y1)*fig.get_figheight()*72,
             'lower_block_bottom_pt':abs(g.get_position().y0-f.get_position().y0)*fig.get_figheight()*72,
             'top_row_title_anchor_pt':abs(a.title.get_window_extent().y1-b.title.get_window_extent().y1)*72/fig.dpi,
             'left_lower_title_center_pt':abs(e.title.get_window_extent().x0+e.title.get_window_extent().width/2-g.title.get_window_extent().x0-g.title.get_window_extent().width/2)*72/fig.dpi}
assert all(delta<=1.5 for delta in span_checks.values())
(QA/'Figure_3.span_alignment.json').write_text(json.dumps({'verdict':'PASS','tolerance_pt':1.5,'deviations':span_checks},indent=2),encoding='utf-8')
save(fig,'Figure_3',[a,b,c,d,e,f,g],list('abcdefg'),rows=[['a','b'],['c','d']],cols=[['c','g'],['b','d','f']],exemptions=[{'panels':['a','c','d','e','f','g'],'checks':['panel-width'],'reason':'Separate original schematics and a large primary Tahoe recovery panel, with compact supporting fidelity and model comparisons.'},{'panels':['b','d','f'],'checks':['vertical-gutter'],'reason':'Retain the approved larger gap between the LCL and Tahoe experiments; this revision changes horizontal alignment only.'}])
exec((HERE/'figure3_author_panels.py').read_text(encoding='utf-8'))
insert_figure3_artwork([('Figure_3a_author.svg',[.075-FIGURE_3_SHIFT_LEFT,.79,.43,.14]),('Figure_3e_author.svg',[.11-FIGURE_3_SHIFT_LEFT,.193,.415,.163])],7.0866141732*72,5.7*72)
