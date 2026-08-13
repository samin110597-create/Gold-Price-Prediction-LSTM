import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multi_asset_forecast_v2 import ASSETS as BASE_ASSETS, daily, load_factors, features

DATA=Path('data'); DATA.mkdir(exist_ok=True)
HORIZONS={'1 Day':1,'1 Week':5}
FRED={'real_yield_10y':'DFII10','broad_usd':'DTWEXBGS'}

def fred_series(series_id):
    try:
        d=pd.read_csv(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}')
        if d.shape[1]<2:return pd.Series(dtype=float)
        d=d.iloc[:,:2];d.columns=['date','value'];d['date']=pd.to_datetime(d['date'],errors='coerce');d['value']=pd.to_numeric(d['value'],errors='coerce')
        s=d.dropna().set_index('date')['value'].sort_index();s.index=s.index.normalize();return s
    except Exception:
        return pd.Series(dtype=float)

def specialist_features(target,fac):
    x=features(target,fac).copy()
    for name,sid in FRED.items():
        s=fred_series(sid)
        if s.empty:continue
        z=s.reindex(x.index).ffill().shift(1)
        if z.notna().sum()<700:continue
        x[name]=z;x[f'{name}_chg5']=z.diff(5);x[f'{name}_chg20']=z.diff(20);x[f'{name}_z60']=(z-z.rolling(60).mean())/z.rolling(60).std().replace(0,np.nan)
    # Drop genuinely unusable optional columns rather than destroying all aligned history.
    min_nonnull=max(650,int(len(x)*.45));x=x[[c for c in x.columns if x[c].notna().sum()>=min_nonnull]]
    return x.replace([np.inf,-np.inf],np.nan)

def models():
    return {'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=12.0))]),'Extra Trees':ExtraTreesRegressor(n_estimators=120,max_depth=7,min_samples_leaf=10,max_features=.7,random_state=42,n_jobs=-1),'Gradient Boost':HistGradientBoostingRegressor(max_iter=110,max_depth=3,learning_rate=.04,l2_regularization=.5,random_state=42)}

def origin_positions(n,h,min_train=900,max_origins=48):
    end=n-h-1
    if end<=min_train+h:return []
    pool=np.arange(min_train+h,end+1,10 if h==1 else 12)
    return (pool[-max_origins:] if len(pool)>max_origins else pool).tolist()

def fit_eval(target,x,h):
    c=target.Close.astype(float);y=c.shift(-h)/c-1;df=x.copy();df['_y']=y;df=df.dropna()
    if len(df)<1050:raise RuntimeError(f'insufficient aligned history: {len(df)}')
    X=df.drop(columns=['_y']);Y=df['_y'].astype(float);M=models();os=origin_positions(len(df),h)
    if len(os)<30:raise RuntimeError(f'insufficient origins: {len(os)} from {len(df)} rows')
    actual=[];pred=[];base=[];store={k:[] for k in M}
    for o in os:
        te=o-h;xt,yt=X.iloc[:te],Y.iloc[:te];xp=X.iloc[[o]];parts=[]
        for name,tpl in M.items():
            m=clone(tpl);m.fit(xt,yt);p=float(m.predict(xp)[0]);parts.append(p);store[name].append(p)
        actual.append(float(Y.iloc[o]));pred.append(float(np.mean(parts)));base.append(float(np.median(yt.tail(min(756,len(yt))))))
    actual=np.asarray(actual);pred=np.asarray(pred);base=np.asarray(base);mae=float(np.mean(np.abs(actual-pred)));bmae=float(np.mean(np.abs(actual-base)));skill=float(1-mae/bmae) if bmae>0 else 0.;dacc=float(np.mean(np.sign(pred)==np.sign(actual)));bdacc=float(np.mean(np.sign(base)==np.sign(actual)));edge=dacc-bdacc;res=actual-pred
    latest=x.dropna().iloc[[-1]];valid=x.notna().all(axis=1)&y.notna();xt=x.loc[valid];yt=y.loc[valid];parts=[];detail=[]
    for name,tpl in M.items():
        m=clone(tpl);m.fit(xt,yt);p=float(m.predict(latest)[0]);parts.append(p);arr=np.asarray(store[name]);detail.append({'name':name,'projected_return_pct':round(p*100,3),'oos_mae_pct':round(float(np.mean(np.abs(actual-arr)))*100,3)})
    lr=float(np.mean(parts));last=float(c.iloc[-1]);price=last*(1+lr);q20,q80=np.quantile(res,[.2,.8]);q10,q90=np.quantile(res,[.1,.9]);focus=sorted([last*(1+lr+q20),last*(1+lr+q80)]);risk=sorted([last*(1+lr+q10),last*(1+lr+q90)]);rawp=float(np.mean(lr+res>0));trust=float(np.clip((max(skill,0)+max(edge,0))/.16,0,1));pup=float(np.clip(.5+(rawp-.5)*(.35+.65*trust),.2,.8));minreq=40 if h==1 else 36;passed=bool(edge>=.03 and skill>=.03 and len(actual)>=minreq)
    return {'predicted_price':round(price,2),'model_price':round(price,2),'raw_ml_price':round(price,2),'projected_return_pct':round(lr*100,3),'probability_up':round(pup,4),'probability_down':round(1-pup,4),'confidence':'Moderate' if passed else 'Low','confidence_score':55 if passed else 35,'forecast_status':'Validated' if passed else 'Fail','tight_model_zone':[round(focus[0],2),round(focus[1],2)],'focus_zone':[round(focus[0],2),round(focus[1],2)],'risk_zone':[round(risk[0],2),round(risk[1],2)],'backtest_directional_accuracy':round(dacc,4),'baseline_directional_accuracy':round(bdacc,4),'directional_edge':round(edge,4),'backtest_mae_pct':round(mae*100,3),'baseline_mae_pct':round(bmae*100,3),'mae_skill_vs_baseline':round(skill,4),'walkforward_origins':len(actual),'validation':'V4 purged expanding-origin horizon specialist using proven V2 aligned market features; optional FRED inputs lagged one day; live formula equals audited formula','model_version':'V4 horizon specialist','models':detail,'v4_pass':passed}

def score(r):return .55*float(r.get('mae_skill_vs_baseline') or 0)+.45*float(r.get('directional_edge') or 0)

def run(asset,cfg):
    target=daily(cfg['ticker']);fac,meta=load_factors(target.index,cfg['factors']);fac=fac[[c for c in fac.columns if meta.get(c,{}).get('available')]];x=specialist_features(target,fac);pp=DATA/f'{asset}_projections.json';p=json.loads(pp.read_text());by={r.get('horizon'):r for r in p.get('projections',[])};out={'status':'ok','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'horizons':[]}
    for label,h in HORIZONS.items():
        ch=fit_eval(target,x,h);inc=by.get(label);iscore=score(inc or {});cscore=score(ch);incpass=bool((inc or {}).get('directional_edge',0)>=.02 and (inc or {}).get('mae_skill_vs_baseline',0)>=.02);promote=bool(ch['v4_pass'] and (not incpass or cscore>=iscore+.01));out['horizons'].append({'horizon':label,'challenger':ch,'incumbent_score':round(iscore,4),'challenger_score':round(cscore,4),'selected_as_champion':promote,'promotion_rule':'challenger pass + combined OOS score at least 1 point above passing incumbent, or incumbent fails'})
        if promote and inc is not None:
            steps=inc.get('steps',h);inc.clear();inc.update(ch);inc['horizon']=label;inc['steps']=steps;inc['selected_model']='V4 horizon specialist';inc['promotion_note']='Promoted only after fixed OOS direction + MAE safeguards.'
    pp.write_text(json.dumps(p,indent=2));(DATA/f'{asset}_horizon_specialist_v4.json').write_text(json.dumps(out,indent=2));print(asset,[(r['horizon'],r['selected_as_champion']) for r in out['horizons']])

def main():
    for asset,cfg in BASE_ASSETS.items():
        try:run(asset,cfg)
        except Exception as e:
            q={'status':'unavailable','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'reason':str(e),'horizons':[]};(DATA/f'{asset}_horizon_specialist_v4.json').write_text(json.dumps(q,indent=2));print(asset,'unavailable',e)
if __name__=='__main__':main()
