from common import *
# Figure 1: the state/response comparison dominates; truth checks form a compact rail.
s=read('Main_Figure_1');fig=plt.figure(figsize=(7.0866141732,5.35))
a=panel(fig,[.095,.785,.255,.095],'a','Reliable response structure')
q=s[s.kind=='factorial_share'];vals=q.relative_share.to_numpy()*100
a.barh([2,1,0],vals,color=[BLUE,TEAL,CORAL],height=.62);a.set_yticks([2,1,0],['Donor','Stim.','Interaction']);a.set_xlim(0,65);a.set_xticks([0,25,50]);a.set_xlabel('Positive energy share (%)',labelpad=2)
for y,v in zip([2,1,0],vals):a.text(v+1,y,f'{v:.1f}',va='center',fontsize=6.5)
b=panel(fig,[.095,.415,.255,.20],'b','Measured differences exceed null')
q=s[s.kind=='truth_distance']
for i,g in enumerate(q.label.unique()):
    v=q[q.label==g].value.to_numpy()*1000;b.scatter(i+np.array([-.075,0,.075]),v,s=15,color=TEAL,zorder=3);b.hlines(v.mean(),i-.22,i+.22,color=DARK,lw=1)
q=s[s.kind=='truth_summary'].copy();q[['value','ci_low','ci_high']]*=1000;err(b,q,x=[3],col=GOLD)
null95=s[s.kind=='truth_null'].iloc[0].value*1000
b.axhline(null95,color=GRAY,ls='--',lw=.8)
b.annotate('Shuffle null (95th percentile)',xy=(.98,null95),xycoords=b.get_yaxis_transform(),xytext=(0,4),textcoords='offset points',ha='right',va='bottom',fontsize=6.3,color=DARK)
b.set_xticks(range(4),['Rest–8 h','Rest–48 h','8–48 h','Overall'],rotation=45,ha='right',rotation_mode='anchor');b.set_ylim(0,1.12);b.set_xlim(-.5,3.5);b.set_ylabel('Operator distance (×0.001)')
cs=panel(fig,[.53,.415,.165,.465],'c','State similarity\nr ≈ 0.989')
ca=panel(fig,[.8,.415,.165,.465],None,'Response amplitude\nα ≈ 0.0065')
for i,m in enumerate(['Ridge','Bilinear','MLP']):
    v=s[(s.kind=='state_pearson_donor')&(s.model==m)].value.to_numpy();cs.scatter(i+np.linspace(-.12,.12,4),v,s=14,color=BLUE,alpha=.75);cs.hlines(v.mean(),i-.25,i+.25,color=DARK,lw=1.3)
names(cs);cs.set_ylim(.980,.995);cs.set_yticks([.980,.985,.990,.995]);cs.set_ylabel('State Pearson r (truncated axis)')
summary(ca,s,'operator_alpha');ca.set_ylim(0,.008);ca.set_yticks([0,.002,.004,.006,.008]);ca.set_ylabel('Truth-aligned response amplitude')
dc=panel(fig,[.095,.09,.19,.16],'d','Response alignment');summary(dc,s,'operator_cosine');dc.set_ylabel('Operator cosine');dc.set_ylim(0,.014);dc.set_yticks([0,.005,.010])
dr=panel(fig,[.415,.09,.19,.16],None,'Donor adaptation');summary(dr,s,'adaptation_r2',False);dr.axhline(0,color=GRAY,ls='--',lw=.7);dr.set_ylim(-.0046,.0005);dr.set_yticks([-.004,-.002,0]);dr.set_ylabel('Adaptation R²')
e=panel(fig,[.735,.09,.23,.16],'e','Shared rule dominates')
q=s[s.kind=='shared_prediction_energy_fraction'].set_index('model').loc[['Ridge','Bilinear','MLP']]
e.barh(range(3),q.value*100,color=BLUE,height=.55);e.set_yticks(range(3),q.index);e.set_xlim(0,100);e.set_xticks([0,50,100]);e.set_xlabel('Prediction energy (%)')
for i,v in enumerate(q.value):e.text(96,i,'>99.99%' if v>.99999 else f'{v*100:.2f}%',ha='right',va='center',fontsize=6.5,color='white')
for ax in [a,b,cs,ca,dc,dr,e]:
    subtitle=ax.get_title(loc='left');ax.set_title('',loc='left');ax.set_title(subtitle,loc='center',pad=7)
fig.set_size_inches(7.0866141732,4.8)
for ax,rect in zip([a,b,cs,ca,dc,dr,e],[[0.095, 0.8, 0.29, 0.095], [0.095, 0.44, 0.29, 0.2], [0.51, 0.44, 0.185, 0.455], [0.79, 0.44, 0.175, 0.455], [0.095, 0.085, 0.21, 0.21], [0.425, 0.085, 0.21, 0.21], [0.755, 0.085, 0.21, 0.21]]):ax.set_position(rect)
save(fig,'Figure_1',[a,b,cs,ca,dc,dr,e],['a','b','c1','c2','d1','d2','e'],rows=[['c1','c2'],['d1','d2','e']],cols=[['a','b']],
 exemptions=[{'panels':['e'],'checks':['panel-width'],'reason':'Shared-energy support panel has its own categorical labels.'}])

