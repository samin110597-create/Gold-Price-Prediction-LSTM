import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path('data')
DATA.mkdir(exist_ok=True)

ASSETS = {
    'gold': {'ticker': 'GC=F', 'factors': ['SI=F','DX-Y.NYB','^TNX','^VIX','TLT','SPY','HG=F','CL=F']},
    'silver': {'ticker': 'SI=F', 'factors': ['GC=F','DX-Y.NYB','^TNX','^VIX','TLT','SPY','HG=F','CL=F']},
}
HORIZONS = {'1 Day': 1, '1 Week': 5}
FRED = {'real_yield_10y': 'DFII10', 'broad_usd': 'DTWEXBGS'}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def norm_index(df):
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    out = df.copy(); out.index = idx.normalize()
    return out[~out.index.duplicated(keep='last')].sort_index()


def yf_daily(ticker):
    df = yf.download(ticker, start='2005-01-01', interval='1d', auto_adjust=False, progress=False, threads=False)
    df = norm_index(flatten(df))
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=['Close']).copy()


def fred_series(series_id):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    try:
        d = pd.read_csv(url); d.columns = ['date', 'value']
        d['date'] = pd.to_datetime(d['date'], errors='coerce'); d['value'] = pd.to_numeric(d['value'], errors='coerce')
        s = d.dropna().set_index('date')['value'].sort_index(); s.index = s.index.normalize(); return s
    except Exception:
        return pd.Series(dtype=float)


def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)


def atr(df, n=14):
    pc=df.Close.shift(1); tr=pd.concat([(df.High-df.Low),(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()


def build_frame(target, factor_map, fred_map):
    c=target.Close.astype(float); x=pd.DataFrame(index=target.index)
    for n in (1,2,5,10,20,60,120): x[f'ret_{n}']=c.pct_change(n)
    for n in (20,50,100,200):
        e=c.ewm(span=n,adjust=False).mean(); x[f'ema_gap_{n}']=c/e-1
    x['rsi14']=(rsi(c)-50)/50; x['atr14_pct']=atr(target)/c; x['rv20']=c.pct_change().rolling(20).std(); x['rv60']=c.pct_change().rolling(60).std()
    x['z20']=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan); x['z60']=(c-c.rolling(60).mean())/c.rolling(60).std().replace(0,np.nan)
    x['range20']=(c-target.Low.rolling(20).min())/(target.High.rolling(20).max()-target.Low.rolling(20).min()).replace(0,np.nan)
    x['range60']=(c-target.Low.rolling(60).min())/(target.High.rolling(60).max()-target.Low.rolling(60).min()).replace(0,np.nan); x['drawdown60']=c/c.rolling(60).max()-1
    for name,s in factor_map.items():
        z=s.reindex(x.index).ffill(limit=5)
        for n in (1,5,20,60): x[f'{name}_ret_{n}']=z.pct_change(n)
        x[f'{name}_z60']=(z-z.rolling(60).mean())/z.rolling(60).std().replace(0,np.nan); x[f'{name}_ema20']=z/z.ewm(span=20,adjust=False).mean()-1
    for name,s in fred_map.items():
        z=s.reindex(x.index).ffill().shift(1)
        x[name]=z; x[f'{name}_chg5']=z.diff(5); x[f'{name}_chg20']=z.diff(20); x[f'{name}_z60']=(z-z.rolling(60).mean())/z.rolling(60).std().replace(0,np.nan)
    if 'DX-Y_NYB' in factor_map:
        dxy=factor_map['DX-Y_NYB'].reindex(x.index).ffill(limit=5); x['target_dxy_corr60']=c.pct_change().rolling(60).corr(dxy.pct_change())
    if '^TNX' in factor_map:
        yy=factor_map['^TNX'].reindex(x.index).ffill(limit=5); x['yield_delta5']=yy.diff(5); x['yield_delta20']=yy.diff(20)
    return x.replace([np.inf,-np.inf],np.nan)


def models():
    return {
        'Ridge': Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=12.0))]),
        'Extra Trees': ExtraTreesRegressor(n_estimators=180,max_depth=7,min_samples_leaf=10,max_features=.7,random_state=42,n_jobs=-1),
        'Gradient Boost': HistGradientBoostingRegressor(max_iter=140,max_depth=3,learning_rate=.04,l2_regularization=.5,random_state=42),
    }


def origin_positions(n,h,min_train=1000,max_origins=54):
    end=n-h-1
    if end<=min_train+h: return []
    pool=np.arange(min_train+h,end+1,10 if h==1 else 12)
    if len(pool)>max_origins: pool=pool[-max_origins:]
    return pool.tolist()


def fit_eval(target,x,h):
    c=target.Close.astype(float); y=c.shift(-h)/c-1
    df=x.copy(); df['_y']=y; df=df.dropna()
    if len(df)<1150: raise RuntimeError(f'insufficient aligned history: {len(df)}')
    X=df.drop(columns=['_y']); Y=df['_y'].astype(float); mods=models(); origins=origin_positions(len(df),h)
    if len(origins)<30: raise RuntimeError(f'insufficient walk-forward origins: {len(origins)}')
    actual=[]; pred=[]; base_pred=[]; model_pred_store={k:[] for k in mods}
    for o in origins:
        train_end=o-h
        if train_end<900: continue
        xt,yt=X.iloc[:train_end],Y.iloc[:train_end]; xp=X.iloc[[o]]; ps=[]
        for name,tpl in mods.items():
            m=clone(tpl); m.fit(xt,yt); pv=float(m.predict(xp)[0]); ps.append(pv); model_pred_store[name].append(pv)
        actual.append(float(Y.iloc[o])); pred.append(float(np.mean(ps))); base_pred.append(float(np.median(yt.tail(min(756,len(yt))))))
    actual=np.asarray(actual); pred=np.asarray(pred); base_pred=np.asarray(base_pred)
    mae=float(np.mean(np.abs(actual-pred))); bmae=float(np.mean(np.abs(actual-base_pred))); skill=float(1-mae/bmae) if bmae>0 else 0.0
    dacc=float(np.mean(np.sign(pred)==np.sign(actual))); bdacc=float(np.mean(np.sign(base_pred)==np.sign(actual))); edge=dacc-bdacc; residual=actual-pred
    latest=x.dropna().iloc[[-1]]; valid=x.notna().all(axis=1)&y.notna(); Xtrain,Ytrain=x.loc[valid],y.loc[valid]; live_parts=[]; detail=[]
    for name,tpl in mods.items():
        m=clone(tpl); m.fit(Xtrain,Ytrain); pp=float(m.predict(latest)[0]); live_parts.append(pp); arr=np.asarray(model_pred_store[name]); mm=float(np.mean(np.abs(actual-arr))) if len(arr)==len(actual) else np.nan
        detail.append({'name':name,'projected_return_pct':round(pp*100,3),'oos_mae_pct':round(mm*100,3) if np.isfinite(mm) else None})
    live_ret=float(np.mean(live_parts)); last=float(target.Close.iloc[-1]); live_price=last*(1+live_ret)
    q20,q80=np.quantile(residual,[.20,.80]); q10,q90=np.quantile(residual,[.10,.90]); focus=[last*(1+live_ret+q20),last*(1+live_ret+q80)]; risk=[last*(1+live_ret+q10),last*(1+live_ret+q90)]
    if focus[0]>focus[1]: focus=focus[::-1]
    if risk[0]>risk[1]: risk=risk[::-1]
    p_up_raw=float(np.mean(live_ret+residual>0)); trust=float(np.clip((max(skill,0)+max(edge,0))/.16,0,1)); p_up=float(np.clip(.5+(p_up_raw-.5)*(.35+.65*trust),.20,.80))
    min_req=40 if h==1 else 36; pass_flag=bool(edge>=.03 and skill>=.03 and len(actual)>=min_req); confidence='Moderate' if pass_flag else 'Low'
    return {'predicted_price':round(live_price,2),'model_price':round(live_price,2),'raw_ml_price':round(live_price,2),'projected_return_pct':round(live_ret*100,3),'probability_up':round(p_up,4),'probability_down':round(1-p_up,4),'confidence':confidence,'confidence_score':55 if pass_flag else 35,'forecast_status':'Validated' if pass_flag else 'Fail','tight_model_zone':[round(focus[0],2),round(focus[1],2)],'focus_zone':[round(focus[0],2),round(focus[1],2)],'risk_zone':[round(risk[0],2),round(risk[1],2)],'backtest_directional_accuracy':round(dacc,4),'baseline_directional_accuracy':round(bdacc,4),'directional_edge':round(edge,4),'backtest_mae_pct':round(mae*100,3),'baseline_mae_pct':round(bmae*100,3),'mae_skill_vs_baseline':round(skill,4),'walkforward_origins':int(len(actual)),'validation':'V4 purged expanding-origin specialist; fixed equal-weight ensemble; live formula equals audited formula','model_version':'V4 horizon specialist','models':detail,'v4_pass':pass_flag}


def combined_score(row):
    return .55*float(row.get('mae_skill_vs_baseline') or 0)+.45*float(row.get('directional_edge') or 0)


def run_asset(asset,cfg):
    target=yf_daily(cfg['ticker'])
    if target.empty: raise RuntimeError(f'no target data for {asset}')
    fmap={}
    for ticker in cfg['factors']:
        d=yf_daily(ticker)
        if not d.empty: fmap[ticker.replace('.','_')]=d.Close.astype(float)
    fred_map={name:fred_series(sid) for name,sid in FRED.items()}; x=build_frame(target,fmap,fred_map)
    proj_path=DATA/f'{asset}_projections.json'; projections=json.loads(proj_path.read_text()); by_h={r.get('horizon'):r for r in projections.get('projections',[])}
    result={'status':'ok','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'horizons':[]}
    for label,h in HORIZONS.items():
        challenger=fit_eval(target,x,h); incumbent=by_h.get(label); inc_score=combined_score(incumbent or {}); ch_score=combined_score(challenger)
        incumbent_pass=bool((incumbent or {}).get('directional_edge',0)>=.02 and (incumbent or {}).get('mae_skill_vs_baseline',0)>=.02)
        promote=bool(challenger['v4_pass'] and (not incumbent_pass or ch_score>=inc_score+.01))
        result['horizons'].append({'horizon':label,'challenger':challenger,'incumbent_score':round(inc_score,4),'challenger_score':round(ch_score,4),'selected_as_champion':promote,'promotion_rule':'challenger pass + combined OOS score at least 1 point above passing incumbent, or incumbent fails'})
        if promote and incumbent is not None:
            keep_steps=incumbent.get('steps',h); incumbent.clear(); incumbent.update(challenger); incumbent['horizon']=label; incumbent['steps']=keep_steps; incumbent['selected_model']='V4 horizon specialist'; incumbent['promotion_note']='Promoted only after fixed OOS direction + MAE safeguards.'
    proj_path.write_text(json.dumps(projections,indent=2)); (DATA/f'{asset}_horizon_specialist_v4.json').write_text(json.dumps(result,indent=2)); print(asset,[(r['horizon'],r['selected_as_champion']) for r in result['horizons']])


def main():
    for asset,cfg in ASSETS.items():
        try: run_asset(asset,cfg)
        except Exception as e:
            payload={'status':'unavailable','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'reason':str(e),'horizons':[]}; (DATA/f'{asset}_horizon_specialist_v4.json').write_text(json.dumps(payload,indent=2)); print(asset,'unavailable',e)

if __name__=='__main__': main()
