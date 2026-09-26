import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common_box import *
from matplotlib.patches import Rectangle, Arc
NAVY='#1F3A5F';ORG='#B35900'
fig,ax=canvas(174,132)
def band(y,h,label,ec,fc,ls='-',lock=False):
    ax.add_patch(FancyBboxPatch((3,y),168,h,boxstyle='round,pad=0,rounding_size=2',fc=fc,ec=ec,lw=0.9,ls=ls))
    tx=7
    if lock:
        ax.add_patch(Rectangle((6.3,y+h-6.3),3.2,2.5,fc=ec,ec=ec,lw=0.4));ax.add_patch(Arc((7.9,y+h-3.8),2.2,2.8,theta1=0,theta2=180,ec=ec,lw=0.9));tx=11.5
    ax.text(tx,y+h-5,label,fontsize=8,fontweight='bold',color=ec,va='center')
bx=[7,48.5,90,131.5];bw=36
rows=[(98,'a   Locked preparation',NAVY,'#EEF2F7','-',True,[('Source data','GSE96058'),('Cohort','n = 3,073'),('Split','60/20/20, seed 42'),('Gene scope','training variance')]),
      (56,'b   Iterative training',ORG,'#FFFFFF',(0,(4,2)),False,[('Baseline','logistic regression'),('Candidates','RF, ET, HGB'),('Ablations','declared in advance'),('Artefacts','predictions, SHAP')]),
      (14,'c   Locked evaluation',NAVY,'#EEF2F7','-',True,[('Threshold','validation only'),('Metrics','AUROC, F1, Brier, MCC'),('Promotion','validation AUROC'),('Audit log','results_log.json')])]
for y,lab,ec,fc,ls,lock,B in rows:
    band(y,32,lab,ec,fc,ls,lock)
    yb=y+9 if lab.startswith('b') else y+5
    for i,(h,b) in enumerate(B):
        box(ax,bx[i],yb,bw,15,h,b,ec=ec)
        if i<3: arrow(ax,bx[i]+bw+0.3,yb+7.5,bx[i+1]-0.3,yb+7.5)
arrow(ax,87,98,87,88.2,c=NAVY);arrow(ax,87,56,87,46.2)
ax.plot([bx[3]+bw/2,bx[3]+bw/2,bx[1]+bw/2],[65,61.5,61.5],color=ORG,lw=0.9);arrow(ax,bx[1]+bw/2,61.5,bx[1]+bw/2,64.9,c=ORG)
ax.text(bx[2]+bw/2,60.8,'next run',fontsize=6.8,color=ORG,ha='center',va='top',style='italic')
ax.text(87,16,'test metrics logged but never read by the rule',fontsize=6.8,color='#A50026',ha='center',style='italic')
ax.add_patch(FancyBboxPatch((3,2),168,9,boxstyle='round,pad=0,rounding_size=2',fc='#F7F7F7',ec='#555',lw=0.8))
ax.text(7,6.5,'d   External check',fontsize=8,fontweight='bold',color='#333',va='center')
ax.text(60,6.5,'GSE81538: frozen models applied, outside the audit log',fontsize=7,va='center',color='#222')
save(fig,'S_Fig1')
