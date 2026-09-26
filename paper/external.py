import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import numpy as np,pandas as pd,json
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'analysis.py')).read().split('rows=[]')[0])
import statsmodels.api as sm
def calib(y,p):
    p=np.clip(p,1e-4,1-1e-4);lp=np.log(p/(1-p))
    m=sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial()).fit()
    slope=m.params[1]
    mi=sm.GLM(y,np.ones_like(lp),family=sm.families.Binomial(),offset=lp).fit()
    return slope,mi.params[0]
ph=pd.read_csv(R+'data/processed/GSE81538_PAM50_proxy_external/phenotype.csv.gz').rename(columns={'Unnamed: 0':'sample_id'})
ph['er_path']=(ph['er consensus']>0).astype(int)
out=[]
E={'LR_baseline':'reports/external_validation/baseline_to_gse81538','ET_seed7':'reports/external_validation/extra_trees_seed_7_to_gse81538'}
for run,path in E.items():
    d=pd.read_csv(R+path+'/evaluation/external_predictions.csv').merge(ph[['sample_id','er_path','er consensus','pam50 subtype']],on='sample_id')
    d.to_csv(f'{OUT}ext_{run}.csv',index=False)
    for lab in ['y_true','er_path']:
        y=d[lab].values;p=d.probability_class_1.values
        m=metrics(y,p,0.7);bt=boot(y,p,0.7,B=2000);a,lo,hi=delong_ci(y,p);s,i=calib(y,p)
        r=dict(run=run,endpoint='PAM50 proxy' if lab=='y_true' else 'ER pathology consensus',n=len(y),neg=int((y==0).sum()),**m,auroc_lo=lo,auroc_hi=hi,cal_slope=s,cal_int=i)
        for k in ['macro_f1','mcc','brier','auroc']: r[k+'_lo_b'],r[k+'_hi_b']=np.percentile(bt[k],[2.5,97.5])
        out.append(r)
X=pd.DataFrame(out);X.to_csv(OUT+'external_metrics.csv',index=False);print(X.round(4).T.to_string())
print(pd.crosstab(ph['er_path'],ph['pam50 subtype']))
# internal test calibration
cal=[]
for run in ['LR_baseline','ET_seed7','RF','HGB']:
    d=load(run,'test');s,i=calib(d.y_true.values,d.probability_class_1.values);cal.append((run,s,i))
print(cal); json.dump(cal,open(OUT+'test_calib.json','w'))
# brier paired bootstrap ext diff
a=pd.read_csv(OUT+'ext_LR_baseline.csv').sort_values('sample_id');b=pd.read_csv(OUT+'ext_ET_seed7.csv').sort_values('sample_id')
y=a.er_path.values;rs=np.random.RandomState(1);D=[]
for k in range(2000):
    idx=rs.choice(len(y),len(y));D.append(np.mean((b.probability_class_1.values[idx]-y[idx])**2)-np.mean((a.probability_class_1.values[idx]-y[idx])**2))
print('brier diff ET-LR path',np.mean((b.probability_class_1-y)**2)-np.mean((a.probability_class_1-y)**2),np.percentile(D,[2.5,97.5]))
print(a[a.er_path!=a.y_true][['sample_id','er consensus','pam50 subtype','probability_class_1']]);print(b[b.er_path!=b.y_true][['sample_id','probability_class_1']])
print(b.probability_class_1.describe(), a.probability_class_1.describe())
