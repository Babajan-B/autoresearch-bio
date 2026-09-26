import os
from pathlib import Path
ROOT=str(Path(os.environ.get("AUTORESEARCH_ROOT",Path(__file__).resolve().parents[1])))+"/"
OUT=str(Path(os.environ.get("PAPER_OUT",Path(ROOT)/"reports"/"paper")))+"/"
os.makedirs(OUT+"figs",exist_ok=True)
import pandas as pd,numpy as np,json
from scipy import stats
R=ROOT
ph=pd.read_csv(R+'data/processed/GSE96058_ER_primary/phenotype.csv.gz').rename(columns={'Unnamed: 0':'sid'})
sp=json.load(open(R+'data/processed/GSE96058_ER_primary/splits.json'))
print({k:len(v) for k,v in sp.items()})
m={s:k for k,v in sp.items() for s in v}
ph['split']=ph.sid.map(m);print(ph.split.value_counts(dropna=False))
order=['train','validation','test'] if 'validation' in sp else ['train','val','test']
rows=[]
def cont(col,lab):
    g=[pd.to_numeric(ph.loc[ph.split==s,col],errors='coerce').dropna() for s in order];a=pd.to_numeric(ph[col],errors='coerce').dropna()
    f=lambda x:f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})" if lab!='Tumour size, mm' else f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})"
    p=stats.kruskal(*g).pvalue;rows.append([lab+', median (IQR)',f(a)]+[f(x) for x in g]+[f"{p:.2f}"])
def cat(col,lab,levels=None,names=None):
    v=ph[col].where(ph[col].notna(),'Missing').astype(str)
    levels=levels or sorted(v.unique())
    ct=pd.crosstab(v,ph.split).reindex(columns=order)
    sub=ct.drop(index=[l for l in ['Missing'] if l in ct.index])
    p=stats.chi2_contingency(sub.values)[1] if sub.shape[0]>1 else np.nan
    rows.append([lab,'','','','',f"{p:.2f}"])
    for i,l in enumerate(levels):
        if l not in ct.index: continue
        tot=(v==l).sum();nm=names[i] if names else l
        rows.append(['  '+nm,f"{tot} ({100*tot/len(v):.1f})"]+[f"{ct.loc[l,s]} ({100*ct.loc[l,s]/ct[s].sum():.1f})" for s in order]+[''])
rows.append(['Samples, n',str(len(ph))]+[str((ph.split==s).sum()) for s in order]+[''])
cat('er status','ER status (label), n (%)',['0','1'],['ER-negative','ER-positive'])
cont('age at diagnosis','Age at diagnosis, years')
cont('tumor size','Tumour size, mm')
cat('lymph node group','Lymph node status, n (%)',None)
cat('nhg','Nottingham histological grade, n (%)',None)
cat('pgr status','PgR status, n (%)',['0.0','1.0','Missing'],['Negative','Positive','Missing'])
cat('her2 status','HER2 status, n (%)',['0.0','1.0','Missing'],['Negative','Positive','Missing'])
cat('pam50 subtype','PAM50 subtype, n (%)',['LumA','LumB','Her2','Basal','Normal','Missing'])
T=pd.DataFrame(rows,columns=['Characteristic','All','Training','Validation','Test','P'])
print(T.to_string());T.to_csv(OUT+'table1.csv',index=False)
print(ph['lymph node group'].value_counts(dropna=False),ph.nhg.value_counts(dropna=False), ph['pam50 subtype'].value_counts(dropna=False))
