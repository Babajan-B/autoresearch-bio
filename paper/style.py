import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
 'font.size':7.5,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.8,
 'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7,'xtick.major.size':3,'ytick.major.size':3,
 'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.facecolor':'white',
 'axes.titleweight':'normal','mathtext.default':'regular'})
MM=1/25.4
C={'LR':'#0072B2','ET7':'#D55E00','RF':'#009E73','HGB':'#CC79A7','ET42':'#E69F00','ET123':'#56B4E9','grey':'#7F7F7F','dark':'#222222'}
def panel(ax,letter,x=-0.16,y=1.04):
    ax.text(x,y,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='left')
def check_overlaps(fig,name,pad=0.5):
    fig.canvas.draw();r=fig.canvas.get_renderer()
    from matplotlib.text import Text
    T=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_alpha()!=0]
    B=[]
    for t in T:
        try: bb=t.get_window_extent(r)
        except Exception: continue
        if bb.width<1: continue
        B.append((t,bb))
    W,H=fig.bbox.width,fig.bbox.height;issues=[]
    import re as _re
    for t,bb in B:
        if _re.fullmatch(r'[−\-]?[0-9.,]+',t.get_text().strip()): continue
        if bb.x0<-1 or bb.y0<-1 or bb.x1>W+1 or bb.y1>H+1: issues.append(('OUTSIDE',t.get_text()[:40]))
    for i in range(len(B)):
        for j in range(i+1,len(B)):
            a,b=B[i][1],B[j][1]
            if a.x0<b.x1-pad and b.x0<a.x1-pad and a.y0<b.y1-pad and b.y0<a.y1-pad:
                issues.append(('OVERLAP',B[i][0].get_text()[:35],B[j][0].get_text()[:35]))
    print(name,'text issues:',len(issues))
    for x in issues: print('   ',x)
def save(fig,name,dpi=600):
    check_overlaps(fig,name)
    import os
    from PIL import Image
    base=OUT+'figs/'+name
    fig.savefig(base+'.pdf')
    fig.savefig(base+'.png',dpi=dpi)
    im=Image.open(base+'.png').convert('RGB');im.save(base+'.tif',dpi=(dpi,dpi),compression='tiff_lzw')
    im.save(base+'_preview.png') if False else None
    print('saved',name,im.size)
