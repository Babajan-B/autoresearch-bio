import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common_box import *
EXC='#8C2D04'
fig,ax=canvas(174,100)
ax.text(2,97,'a',fontsize=10,fontweight='bold',va='top');ax.text(8,96.6,'GSE96058 (primary cohort)',fontsize=8,fontweight='bold',va='top')
ax.text(122,97,'b',fontsize=10,fontweight='bold',va='top');ax.text(128,96.6,'GSE81538 (companion cohort)',fontsize=8,fontweight='bold',va='top')
cx=40;bw=44
box(ax,cx-bw/2,79,bw,11,'SCAN-B RNA-seq','3,409 profiles')
arrow(ax,cx,79,cx,67)
ax.plot([cx,cx+30],[73,73],color='#333',lw=0.8);box(ax,cx+30,69,34,8,'Excluded','136 technical replicates',ec=EXC,hs=7,fs=6.8)
box(ax,cx-bw/2,56,bw,11,'Primary tumours','n = 3,273')
arrow(ax,cx,56,cx,44)
ax.plot([cx,cx+30],[50,50],color='#333',lw=0.8);box(ax,cx+30,46,34,8,'Excluded','200 without binary ER label',ec=EXC,hs=7,fs=6.8)
box(ax,cx-bw/2,33,bw,11,'Analytic cohort','n = 3,073 (92.2% ER+)')
xs=[2,40,78];sw=34
for x in xs: arrow(ax,cx,33,x+sw/2,21)
S=[('Training','n = 1,843','ER− 145 · ER+ 1,698','#0072B2','model fitting'),('Validation','n = 615','ER− 48 · ER+ 567','#E69F00','threshold, promotion'),('Test','n = 615','ER− 48 · ER+ 567','#555555','reporting only')]
for x,(h,n,e,c,role) in zip(xs,S):
    box(ax,x,8,sw,13,h,n+'\n'+e,ec=c,lw=1.1)
    ax.text(x+sw/2,4.5,role,ha='center',va='center',fontsize=6.8,style='italic',color='#444')
ax.plot([118,118],[4,94],color='#BBBBBB',lw=0.6,ls=(0,(3,2)))
rx=146;rw=44;ra=134
box(ax,rx-rw/2,79,rw,11,'SCAN-B RNA-seq','405 tumours')
arrow(ax,ra,79,ra,67)
ax.plot([ra,ra+6],[73,73],color='#333',lw=0.8);box(ax,ra+6,69,32,8,'Excluded','87 other PAM50 types',ec=EXC,hs=7,fs=6.8)
box(ax,rx-rw/2,56,rw,11,'Companion cohort','n = 318')
arrow(ax,rx,56,rx,44)
box(ax,rx-rw/2,24,rw,20,'Endpoints (neg / pos)','PAM50 proxy: 57 / 261\nER pathology: 57 / 261\n4 tumours discordant')
ax.text(rx,17,'frozen models applied;\nno retraining',ha='center',va='center',fontsize=6.8,style='italic',color='#444',linespacing=1.3)
save(fig,'M_Fig1')
