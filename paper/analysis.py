import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss, confusion_matrix
from scipy import stats
R=ROOT
RUNS={
 'LR_baseline':'reports/baseline/GSE96058_ER_primary',
 'ET_seed7':'reports/runs/extra_trees_seed_7',
 'ET_seed42':'reports/runs/extra_trees_seed_42',
 'ET_seed123':'reports/runs/extra_trees_seed_123',
 'RF':'reports/runs/random_forest_gse96058_er',
 'HGB':'reports/runs/hist_gradient_boosting_gse96058_er',
 'ET_k500':'reports/runs/extra_trees_topk_500',
 'ET_k1000':'reports/runs/extra_trees_topk_1000',
 'ET_k2000':'reports/runs/extra_trees_topk_2000',
 'ET_k5000':'reports/runs/extra_trees_topk_5000',
 'LR_abl_ESR1':'reports/runs/ablation_esr1_pgr_erbb2',
 'LR_abl_lum':'reports/runs/ablation_luminal_markers',
}
def load(run,split):
    d=pd.read_csv(R+RUNS[run]+f'/evaluation/{split}_predictions.csv').sort_values('sample_id').reset_index(drop=True)
    return d
def summ(run): return json.load(open(R+RUNS[run]+'/run_summary.json'))
# DeLong
def compute_midrank(x):
    J=np.argsort(x);Z=x[J];N=len(x);T=np.zeros(N);i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N);T2[J]=T;return T2
def delong(y,p1,p2):
    y=np.asarray(y);order=np.argsort(-y);y=y[order]
    preds=np.vstack([p1,p2])[:,order];m=int(y.sum());n=len(y)-m
    pos=preds[:,:m];neg=preds[:,m:];k=2
    tx=np.array([compute_midrank(pos[r]) for r in range(k)])
    ty=np.array([compute_midrank(neg[r]) for r in range(k)])
    tz=np.array([compute_midrank(preds[r]) for r in range(k)])
    aucs=tz[:,:m].sum(1)/m/n-(m+1.)/2./n
    v01=(tz[:,:m]-tx)/n; v10=1.-(tz[:,m:]-ty)/m
    S=np.cov(v01)/m+np.cov(v10)/n
    diff=aucs[0]-aucs[1]; var=S[0,0]+S[1,1]-2*S[0,1]
    z=diff/np.sqrt(var); p=2*stats.norm.sf(abs(z))
    se1=np.sqrt(S[0,0])
    return aucs, diff, p, np.sqrt(var)
def delong_ci(y,p):
    a,_,_,_=delong(y,p,p*0+np.random.RandomState(0).rand(len(p)))
    # single-model variance
    y=np.asarray(y);order=np.argsort(-y);ys=y[order];ps=np.asarray(p)[order];m=int(ys.sum());n=len(ys)-m
    tx=compute_midrank(ps[:m]);ty=compute_midrank(ps[m:]);tz=compute_midrank(ps)
    auc=tz[:m].sum()/m/n-(m+1.)/2./n
    v01=(tz[:m]-tx)/n;v10=1.-(tz[m:]-ty)/m
    se=np.sqrt(np.var(v01,ddof=1)/m+np.var(v10,ddof=1)/n)
    lo,hi=auc-1.96*se,min(1,auc+1.96*se)
    return auc,lo,hi

from scipy.stats import rankdata
def _auc(y,p):
    r=rankdata(p);m=y.sum();n=len(y)-m
    return (r[y==1].sum()-m*(m+1)/2)/(m*n)
def _ap(y,p):
    o=np.argsort(-p,kind='mergesort');ys=y[o];tp=np.cumsum(ys);prec=tp/np.arange(1,len(y)+1)
    return (prec*ys).sum()/ys.sum()
def metrics(y,p,thr,full=True):
    yp=(p>=thr).astype(int)
    tp=int(((yp==1)&(y==1)).sum());tn=int(((yp==0)&(y==0)).sum());fp=int(((yp==1)&(y==0)).sum());fn=int(((yp==0)&(y==1)).sum())
    f1p=2*tp/(2*tp+fp+fn) if tp else 0;f1n=2*tn/(2*tn+fn+fp) if tn else 0
    den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));mcc=(tp*tn-fp*fn)/den if den>0 else 0
    return dict(auroc=_auc(y,p),prauc_neg=_ap(1-y,1-p),macro_f1=(f1p+f1n)/2,mcc=mcc,brier=float(np.mean((p-y)**2)),tn=tn,fp=fp,fn=fn,tp=tp)
def boot(y,p,thr,B=2000,seed=2026):
    rs=np.random.RandomState(seed);i0=np.where(y==0)[0];i1=np.where(y==1)[0];out=[]
    for b in range(B):
        idx=np.concatenate([rs.choice(i0,len(i0)),rs.choice(i1,len(i1))])
        out.append(metrics(y[idx],p[idx],thr))
    return pd.DataFrame(out)
rows=[]
for run in RUNS:
    s=summ(run)
    for split in ['validation','test']:
        d=load(run,split);y=d.y_true.values;p=d.probability_class_1.values
        thr=s['test_metrics']['threshold']; thr=round(thr,2)
        m=metrics(y,p,thr);bt=boot(y,p,thr,B=2000)
        a,lo,hi=delong_ci(y,p)
        row=dict(run=run,split=split,threshold=thr,n=len(y),features=s.get('feature_count'),**m,auroc_lo=lo,auroc_hi=hi)
        for k in ['prauc_neg','macro_f1','mcc','brier']:
            row[k+'_lo'],row[k+'_hi']=np.percentile(bt[k],[2.5,97.5])
        # sanity vs stored
        st=s['validation_metrics' if split=='validation' else 'test_metrics']
        row["stored_auroc"]=st["auroc"];row["stored_mcc"]=st["mcc"];row["stored_mf1"]=st["macro_f1"];row["stored_brier"]=st["brier_score"]
        rows.append(row)
T=pd.DataFrame(rows);T.to_csv(OUT+'metrics_ci.csv',index=False)
pd.set_option('display.width',250);print(T[['run','split','threshold','auroc','auroc_lo','auroc_hi','prauc_neg','macro_f1','mcc','mcc_lo','mcc_hi','brier','tn','fp','fn','tp']].round(4).to_string())
# DeLong comparisons on test
comp=[]
for a,b in [('ET_seed7','LR_baseline'),('RF','ET_seed7'),('HGB','ET_seed7'),('RF','LR_baseline'),('HGB','LR_baseline'),('LR_abl_ESR1','LR_baseline'),('LR_abl_lum','LR_baseline'),('ET_k5000','ET_seed7'),('ET_k500','ET_seed7')]:
    for split in ['validation','test']:
        d1=load(a,split);d2=load(b,split);assert (d1.sample_id==d2.sample_id).all()
        aucs,diff,p,se=delong(d1.y_true.values,d1.probability_class_1.values,d2.probability_class_1.values)
        comp.append(dict(a=a,b=b,split=split,auc_a=aucs[0],auc_b=aucs[1],diff=diff,lo=diff-1.96*se,hi=diff+1.96*se,p=p))
C=pd.DataFrame(comp);C.to_csv(OUT+'delong.csv',index=False);print(C.round(4).to_string())
# paired bootstrap deltas for ablations on test (MCC, macroF1)
rs=np.random.RandomState(7);out=[]
for ab in ['LR_abl_ESR1','LR_abl_lum']:
    d0=load('LR_baseline','test');d1=load(ab,'test');y=d0.y_true.values
    i0=np.where(y==0)[0];i1=np.where(y==1)[0];D=[]
    for b in range(2000):
        idx=np.concatenate([rs.choice(i0,len(i0)),rs.choice(i1,len(i1))])
        m0=metrics(y[idx],d0.probability_class_1.values[idx],0.7);m1=metrics(y[idx],d1.probability_class_1.values[idx],0.7)
        D.append({k:m1[k]-m0[k] for k in ['auroc','macro_f1','mcc','brier']})
    D=pd.DataFrame(D);m0=metrics(y,d0.probability_class_1.values,0.7);m1=metrics(y,d1.probability_class_1.values,0.7)
    for k in D: out.append(dict(ablation=ab,metric=k,delta=m1[k]-m0[k],lo=np.percentile(D[k],2.5),hi=np.percentile(D[k],97.5)))
A=pd.DataFrame(out);A.to_csv(OUT+'ablation_deltas.csv',index=False);print(A.round(4))
