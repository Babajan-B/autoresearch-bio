import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np, pandas as pd, json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve
from statsmodels.stats.proportion import proportion_confint
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'analysis.py')).read().split('rows=[]')[0])
T=pd.read_csv(OUT+'metrics_ci.csv'); D=pd.read_csv(OUT+'delong.csv'); A=pd.read_csv(OUT+'ablation_deltas.csv')
X=pd.read_csv(OUT+'external_metrics.csv')
cal=dict((r,(s,i)) for r,s,i in json.load(open(OUT+'test_calib.json')))
def g(run,split,k): return T[(T.run==run)&(T.split==split)][k].iloc[0]
VAL,TST='#0072B2','#D55E00'
which=sys.argv[1:] or ['2','3','4','S2','S3']
# ============ Main Fig 2: robustness
if '2' in which:
    fig,axs=plt.subplots(2,2,figsize=(174*MM,120*MM));plt.subplots_adjust(left=0.09,right=0.98,top=0.93,bottom=0.17,wspace=0.32,hspace=0.55)
    ax=axs[0,0];seeds=[('ET_seed7','7'),('ET_seed42','42'),('ET_seed123','123')]
    for split,off,mk,c in [('validation',-0.12,'o',VAL),('test',0.12,'s',TST)]:
        for i,(r,sd) in enumerate(seeds):
            v=g(r,split,'auroc');lo=g(r,split,'auroc_lo');hi=min(1,g(r,split,'auroc_hi'))
            ax.errorbar(i+off,v,yerr=[[v-lo],[hi-v]],fmt=mk,color=c,mfc='white' if split=='validation' else c,ms=4,capsize=2,lw=0.9)
    ax.set_xticks(range(3));ax.set_xticklabels(['7*','42','123']);ax.set_xlabel('Random seed (Extra Trees, k = 2,000)');ax.set_ylabel('AUROC');ax.set_ylim(0.89,1.0);ax.set_xlim(-0.5,2.5)
    panel(ax,'a')
    ks=['500','1,000','2,000','5,000'];rk=['ET_k500','ET_k1000','ET_k2000','ET_k5000'];xx=np.arange(4)
    for ax,k,yl,ylab,L in [(axs[0,1],'auroc',(0.89,1.0),'AUROC','b'),(axs[1,0],'mcc',(0.55,0.9),'MCC','c')]:
        for split,mk,c in [('validation','o',VAL),('test','s',TST)]:
            v=np.array([g(r,split,k) for r in rk]);lo=np.array([g(r,split,k+'_lo') for r in rk]);hi=np.minimum(1,[g(r,split,k+'_hi') for r in rk])
            ax.fill_between(xx,lo,hi,color=c,alpha=0.12,lw=0);ax.plot(xx,v,marker=mk,color=c,ms=3.8,mfc='white' if split=='validation' else c)
        ax.set_xticks(xx);ax.set_xticklabels(ks);ax.set_xlabel('Genes retained, k (Extra Trees, seed 42)');ax.set_ylabel(ylab);ax.set_ylim(*yl);panel(ax,L)
    ax=axs[1,1];mets=[('auroc','AUROC'),('macro_f1','Macro-F1'),('mcc','MCC'),('brier','Brier')];yy=np.arange(4)[::-1]
    for off,ab,c,mk in [(0.15,'LR_abl_ESR1','#CC79A7','D'),(-0.15,'LR_abl_lum','#009E73','^')]:
        for j,(m,ml) in enumerate(mets):
            r=A[(A.ablation==ab)&(A.metric==m)].iloc[0]
            ax.errorbar(r.delta,yy[j]+off,xerr=[[r.delta-r.lo],[r.hi-r.delta]],fmt=mk,color=c,ms=3.5,capsize=2,lw=0.9)
    ax.axvline(0,color='#999',lw=0.7,ls='--');ax.set_yticks(yy);ax.set_yticklabels([m[1] for m in mets]);ax.set_xlabel('Change vs intact baseline (test)');ax.set_xlim(-0.06,0.03);ax.set_xticks([-0.06,-0.04,-0.02,0,0.02]);ax.set_ylim(-0.6,3.6)
    panel(ax,'d',x=-0.2)
    h1=[Line2D([],[],marker='o',color=VAL,mfc='white',label='Validation'),Line2D([],[],marker='s',color=TST,label='Test')]
    h2=[Line2D([],[],marker='D',ls='',color='#CC79A7',label='− ESR1, PGR'),Line2D([],[],marker='^',ls='',color='#009E73',label='− AGR3')]
    fig.legend(handles=h1,loc='lower center',bbox_to_anchor=(0.28,0.0),ncol=2,fontsize=7)
    fig.legend(handles=h2,loc='lower center',bbox_to_anchor=(0.77,0.0),ncol=2,fontsize=7)
    save(fig,'M_Fig2')
# ============ Main Fig 3: SHAP
if '3' in which:
    R=ROOT
    S={'LR':R+'reports/baseline/GSE96058_ER_primary','ET7':R+'reports/runs/extra_trees_seed_7','ABL':R+'reports/runs/ablation_esr1_pgr_erbb2'}
    canon={'ESR1','PGR','ERBB2'};lum={'AGR3','CA12','SLC39A6'}
    FW,FH=174,150;fig=plt.figure(figsize=(FW*MM,FH*MM))
    def AX(l,b,w,h): return fig.add_axes([l/FW,b/FH,w/FW,h/FH])
    for i,(k,tit) in enumerate([('LR','Logistic baseline'),('ET7','Extra Trees seed 7'),('ABL','Baseline − ESR1, PGR')]):
        ax=AX(22+i*58,84,32,56);s_=pd.read_csv(S[k]+'/shap/top_genes_by_shap.csv').head(12);v=s_.mean_abs_shap/s_.mean_abs_shap.max()
        cols=['#D55E00' if gname in canon else ('#E69F00' if gname in lum else '#56B4E9') for gname in s_.gene]
        ax.barh(np.arange(12)[::-1],v,color=cols,edgecolor='#333333',lw=0.4,height=0.7)
        ax.set_yticks(np.arange(12)[::-1]);ax.set_yticklabels(s_.gene,fontsize=6.8,style='italic');ax.set_xlim(0,1.05);ax.set_xticks([0,0.5,1]);ax.set_xlabel('Relative mean |SHAP|',fontsize=7)
        ax.set_title(tit,fontsize=7.5);panel(ax,'abc'[i],x=-0.6,y=1.02)
    fig.legend(handles=[Patch(fc='#D55E00',ec='#333',lw=0.4,label='ESR1 / PGR'),Patch(fc='#E69F00',ec='#333',lw=0.4,label='AGR3 / CA12 / SLC39A6'),Patch(fc='#56B4E9',ec='#333',lw=0.4,label='Other')],loc='center',bbox_to_anchor=(0.5,72/FH),ncol=3,fontsize=7)
    RS=[('LR','LR',R+'reports/baseline/GSE96058_ER_primary'),('A1','LR-1',R+'reports/runs/ablation_esr1_pgr_erbb2'),('A2','LR-2',R+'reports/runs/ablation_luminal_markers'),
        ('E7','ET7',R+'reports/runs/extra_trees_seed_7'),('E42','ET42',R+'reports/runs/extra_trees_seed_42'),('E123','ET123',R+'reports/runs/extra_trees_seed_123'),('RF','RF',R+'reports/runs/random_forest_gse96058_er')]
    sets=[set(pd.read_csv(p+'/shap/top_genes_by_shap.csv').gene.head(50)) for _,_,p in RS]
    M=np.array([[len(x&y)/len(x|y) for y in sets] for x in sets])
    ax=AX(62,12,52,52);im=ax.imshow(M,cmap='Blues',vmin=0,vmax=1)
    for i in range(7):
        for j in range(7): ax.text(j,i,f'{M[i,j]:.2f}'.replace('0.','.').replace('1.00','1'),ha='center',va='center',fontsize=6.2,color='white' if M[i,j]>0.6 else '#222')
    ax.set_xticks(range(7));ax.set_yticks(range(7));ax.set_xticklabels([r[1] for r in RS],rotation=45,ha='right',fontsize=6.8);ax.set_yticklabels([r[1] for r in RS],fontsize=6.8);ax.tick_params(length=0)
    for s2 in ax.spines.values(): s2.set_visible(False)
    cax=AX(117,12,2.5,52);cb=fig.colorbar(im,cax=cax);cb.set_label('Jaccard index (top-50 genes)',fontsize=7);cb.ax.tick_params(labelsize=6.5)
    panel(ax,'d',x=-0.42,y=1.03)
    save(fig,'M_Fig3')
# ============ Main Fig 4: external
if '4' in which:
    fig,axs=plt.subplots(1,3,figsize=(174*MM,68*MM));plt.subplots_adjust(left=0.07,right=0.99,top=0.9,bottom=0.3,wspace=0.42)
    EX=[('LR_baseline','Logistic baseline','#0072B2'),('ET_seed7','Extra Trees seed 7','#D55E00')]
    ax=axs[0]
    for run,lab,col in EX:
        d=pd.read_csv(f'{OUT}ext_{run}.csv');f,t,_=roc_curve(d.er_path,d.probability_class_1);ax.plot(f,t,color=col)
    ax.plot([0,1],[0,1],color='#BBB',lw=0.7,ls='--');ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.set_xlim(-0.01,1.01);ax.set_ylim(-0.01,1.02)
    for i,(run,lab,col) in enumerate(EX):
        r=X[(X.run==run)&(X.endpoint=='ER pathology consensus')].iloc[0]
        ax.text(0.97,0.2-0.11*i,f'AUROC {r.auroc:.3f}',transform=ax.transAxes,ha='right',fontsize=7,color=col)
    panel(ax,'a',x=-0.22,y=1.04)
    ax=axs[1]
    for i,(run,lab,col) in enumerate(EX):
        d=pd.read_csv(f'{OUT}ext_{run}.csv');rs=np.random.RandomState(i)
        for yv,xoff in [(0,-0.18),(1,0.18)]:
            pp=d.probability_class_1[d.er_path==yv].values;xj=i+xoff+rs.uniform(-0.08,0.08,len(pp))
            ax.scatter(xj,pp,s=3,color=col,alpha=0.5,lw=0)
            ax.boxplot(pp,positions=[i+xoff],widths=0.22,showfliers=False,medianprops=dict(color='k',lw=0.9),boxprops=dict(lw=0.7),whiskerprops=dict(lw=0.7),capprops=dict(lw=0.7))
    ax.axhline(0.7,color='#A50026',lw=0.7,ls='--')
    ax.set_xticks([-0.18,0.18,0.82,1.18]);ax.set_xticklabels(['ER−','ER+','ER−','ER+']);ax.set_xlim(-0.5,1.5)
    ax.set_ylabel('Predicted P(ER+)');ax.set_ylim(-0.03,1.03);panel(ax,'b',x=-0.3,y=1.04)
    ax=axs[2];w=0.36
    for j,(run,lab,col) in enumerate(EX):
        for i,ep in enumerate(['PAM50 proxy','ER pathology consensus']):
            r=X[(X.run==run)&(X.endpoint==ep)].iloc[0];x=i+(j-0.5)*w
            ax.bar(x,r.brier,w*0.9,color=col,edgecolor='#333',lw=0.4,hatch='' if j==0 else '///')
            ax.errorbar(x,r.brier,yerr=[[r.brier-r.brier_lo_b],[r.brier_hi_b-r.brier]],color='k',capsize=2,lw=0.8)
    ax.set_xticks([0,1]);ax.set_xticklabels(['PAM50\nproxy','ER\npathology']);ax.set_ylabel('Brier score');ax.set_ylim(0,0.09);panel(ax,'c',x=-0.33,y=1.04)
    fig.legend(handles=[Patch(fc='#0072B2',ec='#333',lw=0.4,label='Logistic baseline'),Patch(fc='#D55E00',ec='#333',lw=0.4,hatch='///',label='Extra Trees seed 7 (promoted)'),Line2D([],[],color='#A50026',ls='--',lw=0.8,label='Locked threshold (0.70)')],loc='lower center',bbox_to_anchor=(0.5,0.0),ncol=3,fontsize=7)
    save(fig,'M_Fig4')
MAIN=[('LR_baseline','Logistic baseline','#0072B2','-'),('ET_seed7','Extra Trees seed 7','#D55E00','-'),('RF','Random forest','#009E73','--'),('HGB','Hist. gradient boosting','#CC79A7',':')]
# ============ Supp Fig S2: discrimination/calibration
if 'S2' in which:
    fig,axs=plt.subplots(2,2,figsize=(174*MM,140*MM));plt.subplots_adjust(left=0.09,right=0.98,top=0.95,bottom=0.14,wspace=0.35,hspace=0.42)
    ax=axs[0,0];axin=ax.inset_axes([0.45,0.12,0.5,0.5])
    for run,lab,col,ls in MAIN:
        d=load(run,'test');f,t,_=roc_curve(d.y_true,d.probability_class_1)
        ax.plot(f,t,color=col,ls=ls);axin.plot(f,t,color=col,ls=ls,lw=1.0)
        tn,fp,fn,tp=[g(run,'test',k) for k in ['tn','fp','fn','tp']];axin.plot(fp/(fp+tn),tp/(tp+fn),'o',ms=3.2,mfc='white',mec=col,mew=0.9)
    ax.plot([0,1],[0,1],color='#BBB',lw=0.7,ls='--');ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.set_xlim(-0.01,1.01);ax.set_ylim(-0.01,1.01)
    axin.set_xlim(0,0.3);axin.set_ylim(0.85,1.0);axin.set_xticks([0,0.1,0.2,0.3]);axin.set_yticks([0.85,0.9,0.95,1.0]);axin.tick_params(labelsize=6)
    for s_ in ['top','right']: axin.spines[s_].set_visible(True)
    panel(ax,'a')
    ax=axs[0,1]
    for run,lab,col,ls in MAIN:
        d=load(run,'test');pr,rc,_=precision_recall_curve(1-d.y_true,1-d.probability_class_1);ax.plot(rc,pr,color=col,ls=ls,drawstyle='steps-post')
    ax.axhline(48/615,color='#999',lw=0.7,ls='--');ax.set_xlabel('Recall (ER-negative)');ax.set_ylabel('Precision (ER-negative)');ax.set_xlim(0,1.01);ax.set_ylim(0,1.05);panel(ax,'b')
    ax=axs[1,0];bins=np.array([0,0.2,0.4,0.6,0.8,0.9,1.0001])
    for run,lab,col,ls in MAIN:
        d=load(run,'test');y=d.y_true.values;p=d.probability_class_1.values;idx=np.digitize(p,bins)-1;xs,ys=[],[]
        for b in range(len(bins)-1):
            m=idx==b
            if m.sum()>=3: xs.append(p[m].mean());ys.append(y[m].mean())
        ax.plot(xs,ys,color=col,ls=ls,marker='o',ms=2.8,lw=1.0)
    ax.plot([0,1],[0,1],color='#BBB',lw=0.7,ls='--');ax.set_xlabel('Mean predicted P(ER+)');ax.set_ylabel('Observed ER+ fraction');ax.set_xlim(0,1.01);ax.set_ylim(0,1.05);panel(ax,'c')
    ax=axs[1,1];yy=np.arange(4)[::-1]
    for j,(run,lab,col,ls) in enumerate(MAIN):
        tn,fp,fn,tp=[int(g(run,'test',k)) for k in ['tn','fp','fn','tp']]
        for off,(k,n,mk) in zip([0.16,-0.16],[(fp,48,'s'),(fn,567,'o')]):
            lo,hi=proportion_confint(k,n,method='wilson');r=k/n
            ax.errorbar(100*r,yy[j]+off,xerr=[[100*(r-lo)],[100*(hi-r)]],fmt=mk,color=col,mfc=col if mk=='o' else 'white',ms=3.8,capsize=2,lw=0.9)
    ax.set_yticks(yy);ax.set_yticklabels(['LR','ET s7','RF','HGB']);ax.set_xlabel('Error rate at locked threshold (%)');ax.set_xlim(0,30);ax.set_ylim(-0.6,3.6);panel(ax,'d')
    h=[Line2D([],[],color=c,ls=l,label=n) for _,n,c,l in MAIN]+[Line2D([],[],marker='s',ls='',mfc='white',mec='k',color='k',label='ER− called ER+'),Line2D([],[],marker='o',ls='',color='k',label='ER+ called ER−')]
    fig.legend(handles=h,loc='lower center',bbox_to_anchor=(0.5,0.0),ncol=3,fontsize=7)
    save(fig,'S_Fig2')
# ============ Supp Fig S3: locked vs post hoc
if 'S3' in which:
    RUNS4=[('ET_seed7','ET seed 7, k 2,000'),('ET_k500','ET seed 42, k 500'),('ET_seed123','ET seed 123, k 2,000'),('ET_seed42','ET seed 42, k 2,000'),
       ('ET_k5000','ET seed 42, k 5,000'),('ET_k1000','ET seed 42, k 1,000'),('RF','Random forest'),('LR_baseline','Logistic baseline'),('HGB','Hist. grad. boosting')]
    RUNS4=sorted(RUNS4,key=lambda r:-g(r[0],'validation','auroc'))
    fig=plt.figure(figsize=(174*MM,120*MM));gs=fig.add_gridspec(2,3,height_ratios=[1.6,1],left=0.2,right=0.9,top=0.93,bottom=0.1,wspace=0.12,hspace=0.6)
    axv=fig.add_subplot(gs[0,0]);axt=fig.add_subplot(gs[0,1],sharey=axv);axm=fig.add_subplot(gs[0,2],sharey=axv);yy=np.arange(len(RUNS4))[::-1]
    for j,(run,lab) in enumerate(RUNS4):
        col='#D55E00' if run=='ET_seed7' else '#444444'
        for ax,split,k in [(axv,'validation','auroc'),(axt,'test','auroc'),(axm,'test','mcc')]:
            v=g(run,split,k);lo=g(run,split,k+'_lo');hi=min(1,g(run,split,k+'_hi'))
            ax.errorbar(v,yy[j],xerr=[[v-lo],[hi-v]],fmt='D' if run=='ET_seed7' else 'o',color=col,ms=3.6,capsize=1.8,lw=0.9)
    axv.set_yticks(yy);axv.set_yticklabels([r[1] for r in RUNS4],fontsize=7)
    for ax in (axt,axm): plt.setp(ax.get_yticklabels(),visible=False)
    axv.set_xlim(0.85,1.0);axt.set_xlim(0.96,1.0);axm.set_xlim(0.62,0.88);axv.set_xticks([0.86,0.9,0.94,0.98]);axt.set_xticks([0.96,0.98,1.0]);axm.set_xticks([0.65,0.75,0.85])
    axv.set_xlabel('Validation AUROC');axt.set_xlabel('Test AUROC');axm.set_xlabel('Test MCC')
    axv.set_title('Read by the rule',fontsize=7.5);axt.set_title('Not read',fontsize=7.5);axm.set_title('Not read',fontsize=7.5)
    for ax in (axv,axt,axm): ax.axhspan(yy[0]-0.5,yy[0]+0.5,color='#D55E00',alpha=0.08,lw=0)
    for ax,k in [(axt,'auroc'),(axm,'mcc')]:
        b=max(RUNS4,key=lambda r:g(r[0],'test',k))[0];j=[r[0] for r in RUNS4].index(b);ax.plot(g(b,'test',k),yy[j],'o',ms=8,mfc='none',mec='#A50026',mew=1.0)
    panel(axv,'a',x=-0.8,y=1.05)
    axd=fig.add_subplot(gs[1,:])
    CMP=[('RF','Random forest'),('HGB','Hist. grad. boosting'),('ET_k5000','ET k 5,000'),('ET_k500','ET k 500'),('LR_baseline','Logistic baseline')];yy2=np.arange(len(CMP))[::-1]
    for j,(a,lab) in enumerate(CMP):
        if a=='LR_baseline': r=D[(D.a=='ET_seed7')&(D.b=='LR_baseline')&(D.split=='test')].iloc[0];diff,lo,hi,p=-r['diff'],-r['hi'],-r['lo'],r['p']
        else: r=D[(D.a==a)&(D.b=='ET_seed7')&(D.split=='test')].iloc[0];diff,lo,hi,p=r['diff'],r['lo'],r['hi'],r['p']
        axd.errorbar(diff,yy2[j],xerr=[[diff-lo],[hi-diff]],fmt='o',color='#333',ms=3.4,capsize=2,lw=0.9)
        axd.text(1.02,yy2[j],f'P = {p:.2f}',transform=axd.get_yaxis_transform(),fontsize=7,va='center')
    axd.axvline(0,color='#999',lw=0.7,ls='--');axd.set_yticks(yy2);axd.set_yticklabels([c[1] for c in CMP],fontsize=7);axd.set_xlim(-0.013,0.013)
    axd.set_xlabel('Difference in test AUROC vs ET seed 7 (DeLong 95% CI)');axd.set_xticks([-0.01,-0.005,0,0.005,0.01]);panel(axd,'b',x=-0.235,y=1.05)
    fig.subplots_adjust(right=0.9)
    save(fig,'S_Fig3')
