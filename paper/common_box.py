import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
def canvas(W,H):
    fig=plt.figure(figsize=(W*MM,H*MM)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off'); return fig,ax
def box(ax,x,y,w,h,head,body='',ec='#1F3A5F',fc='white',fs=7,hs=7.5,lw=0.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1',fc=fc,ec=ec,lw=lw))
    n=len(body.split('\n')) if body else 0
    # vertical layout: header then body lines, block centered
    lh=fs*0.3528*1.35  # mm per body line
    hh=hs*0.3528*1.25
    tot=hh+(n*lh if n else 0)
    top=y+h/2+tot/2
    ax.text(x+w/2,top-hh/2,head,ha='center',va='center',fontsize=hs,fontweight='bold',color='#111')
    if n: ax.text(x+w/2,top-hh-n*lh/2,body,ha='center',va='center',fontsize=fs,linespacing=1.35,color='#222')
def arrow(ax,x1,y1,x2,y2,c='#333333',lw=0.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,color=c,lw=lw,shrinkA=0,shrinkB=0))
