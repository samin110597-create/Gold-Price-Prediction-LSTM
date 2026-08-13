import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multi_asset_forecast_v2 import ASSETS as BASE_ASSETS, daily, load_factors, features

DATA=Path('data'); DATA.mkdir(exist_ok=True)
OUTPUT={'gold':'live_forecast.json','silver':'silver_forecast.json'}
H=(1,5,20)

def mods():
    return {
      'Logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.25,max_iter=2200,class_weight='balanced',random_state=42))]),
      'Extra Trees':ExtraTreesClassifier(n_estimators=110,max_depth=7,min_samples_leaf=12,max_features=.7,class_weight='balanced',random_state=42,n_jobs=-1),
      'Gradient Boost':HistGradientBoostingClassifier(max_iter=110,max_depth=3,learning_rate=.04,l2_regularization=.5,random_state=42),
    }

def origins(n,h,maxn=42):
    pool=np.arange(900+h,n-h-1,12 if h<=5 else 15)
    return pool[-maxn:].tolist() if len(pool)>maxn else pool.tolist()

def one(target,x,h):
    c=target.Close.astype(float);fwd=c.shift(-h)/c-1;vol=c.pct_change().rolling(20).std()*np.sqrt(h);thr=np.maximum(vol*.55,.0025*np.sqrt(h))
    y=np.where(fwd>thr,2,np.where(fwd<-thr,0,1)).astype(float);y[fwd.isna().values]=np.nan
    d=x.copy();d['_y']=y;d=d.dropna();X=d.drop(columns=['_y']);Y=d['_y'].astype(int);os=origins(len(d),h)
    if len(os)<30:raise RuntimeError(f'insufficient origins: {len(os)} from {len(d)} aligned rows')
    yy=[];pp=[];pred=[];baseline=[];M=mods()
    for o in os:
        te=o-h;xt,yt=X.iloc[:te],Y.iloc[:te];xp=X.iloc[[o]];probs=[]
        for tpl in M.values():
            m=clone(tpl);m.fit(xt,yt);p=m.predict_proba(xp)[0];z=np.zeros(3);z[m.classes_.astype(int)]=p;probs.append(z)
        q=np.mean(probs,axis=0);pp.append(q);pred.append(int(np.argmax(q)));yy.append(int(Y.iloc[o]));baseline.append(int(yt.value_counts().idxmax()))
    yy=np.asarray(yy);pp=np.asarray(pp);pred=np.asarray(pred);baseline=np.asarray(baseline)
    acc=float(accuracy_score(yy,pred));base=float(accuracy_score(yy,baseline));bal=float(balanced_accuracy_score(yy,pred));oh=np.eye(3)[yy];brier=float(np.mean(np.sum((pp-oh)**2,axis=1)))
    conf=np.max(pp,axis=1);active=(pred!=1)&(conf>=.55);active_n=int(active.sum());active_acc=float(np.mean(pred[active]==yy[active])) if active_n else 0.0;active_base=float(max(np.mean(yy[active]==0),np.mean(yy[active]==2),np.mean(yy[active]==1))) if active_n else 0.0;edge=active_acc-active_base
    valid=x.notna().all(axis=1)&np.isfinite(y);xt=x.loc[valid];yt=np.asarray(y)[valid.values].astype(int);latest=x.dropna().iloc[[-1]];probs=[];details=[]
    for name,tpl in M.items():
        m=clone(tpl);m.fit(xt,yt);p=m.predict_proba(latest)[0];z=np.zeros(3);z[m.classes_.astype(int)]=p;probs.append(z);details.append({'name':name,'down':round(z[0],4),'no_edge':round(z[1],4),'up':round(z[2],4)})
    q=np.mean(probs,axis=0);passflag=bool((acc-base)>=.02 and bal>=.36 and active_n>=18 and edge>=.02);maxdir=max(q[0],q[2]);signal='NO EDGE'
    if maxdir>=.55 and q[1]<=.45:signal='UP' if q[2]>q[0] else 'DOWN'
    return {'horizon_days':h,'probability_down':round(float(q[0]),4),'probability_no_edge':round(float(q[1]),4),'probability_up':round(float(q[2]),4),'signal':signal,'pass':passflag,'oos_accuracy':round(acc,4),'majority_baseline':round(base,4),'balanced_accuracy':round(bal,4),'active_direction_accuracy':round(active_acc,4),'active_baseline':round(active_base,4),'active_edge':round(edge,4),'active_signals':active_n,'walkforward_origins':len(yy),'multiclass_brier':round(brier,4),'validation':'V4 purged expanding-origin 3-state meaningful-move classifier using proven V2 aligned data pipeline','models':details}

def run(asset,cfg):
    t=daily(cfg['ticker']);fac,meta=load_factors(t.index,cfg['factors']);fac=fac[[c for c in fac.columns if meta.get(c,{}).get('available')]];x=features(t,fac);rows=[one(t,x,h) for h in H]
    payload={'status':'ok','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'definition':'UP/DOWN require a volatility-adjusted meaningful move; otherwise NO EDGE','horizons':rows};(DATA/f'{asset}_meaningful_probability_v4.json').write_text(json.dumps(payload,indent=2));fp=DATA/OUTPUT[asset];f=json.loads(fp.read_text());f['meaningful_probability_v4']=rows;fp.write_text(json.dumps(f,indent=2));print(asset,[(r['horizon_days'],r['signal'],r['pass']) for r in rows])

def main():
    for asset,cfg in BASE_ASSETS.items():
        try:run(asset,cfg)
        except Exception as e:
            p={'status':'unavailable','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'reason':str(e),'horizons':[]};(DATA/f'{asset}_meaningful_probability_v4.json').write_text(json.dumps(p,indent=2));print(asset,e)
if __name__=='__main__':main()
