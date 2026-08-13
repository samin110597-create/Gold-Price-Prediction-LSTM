import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path('data')
ASSETS = {'gold': 'GC=F', 'silver': 'SI=F'}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker):
    for period in ('2y', '1y'):
        try:
            df = flatten(yf.download(ticker, period=period, interval='1h', auto_adjust=False,
                                     progress=False, threads=False))
            if not df.empty:
                return df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        except Exception:
            pass
    raise RuntimeError(f'No hourly data for {ticker}')


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def features(df):
    c = df.Close.astype(float); h = df.High.astype(float); l = df.Low.astype(float)
    x = pd.DataFrame(index=df.index)
    for n in (1,2,4,8,24,48,120): x[f'ret{n}'] = c.pct_change(n)
    for span in (8,21,55): x[f'ema{span}_gap'] = c/c.ewm(span=span,adjust=False).mean()-1
    x['rsi14'] = (rsi(c)-50)/50
    x['vol20'] = c.pct_change().rolling(20).std()
    x['vol60'] = c.pct_change().rolling(60).std()
    x['z20'] = (c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    x['range20'] = (c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()).replace(0,np.nan)-.5
    x['slope10'] = (c/c.shift(10)-1)/10
    x['slope30'] = (c/c.shift(30)-1)/30
    return x.replace([np.inf,-np.inf],np.nan)


def templates():
    return {
        'Ridge': Pipeline([('scale', StandardScaler()), ('model', Ridge(alpha=10.0))]),
        'Random Forest': RandomForestRegressor(n_estimators=160,max_depth=6,min_samples_leaf=12,max_features='sqrt',random_state=42,n_jobs=-1),
        'Gradient Boost': HistGradientBoostingRegressor(max_iter=130,max_depth=3,learning_rate=.04,l2_regularization=.45,random_state=42),
    }


def evaluate(X, y, maxn):
    mods = templates(); origins = list(range(1604, len(X), 24))[-maxn:]
    rows=[]
    for o in origins:
        tr=o-4
        if tr < 700: continue
        Xtr,ytr=X.iloc[:tr],y.iloc[:tr]
        preds={}
        try:
            for name,tpl in mods.items():
                m=clone(tpl); m.fit(Xtr,ytr); preds[name]=float(m.predict(X.iloc[[o]])[0])
        except Exception:
            continue
        rows.append({'actual':float(y.iloc[o]),'baseline':float(ytr.tail(min(756,len(ytr))).median()),'preds':preds})
    if len(rows) < 20: raise RuntimeError('Too few hourly origins')
    actual=np.array([r['actual'] for r in rows]); base=np.array([r['baseline'] for r in rows])
    bmae=float(mean_absolute_error(actual,base)); bdir=float(np.mean((base>=0)==(actual>=0)))
    weights=[]; names=list(mods)
    for name in names:
        p=np.array([r['preds'][name] for r in rows]); mae=float(mean_absolute_error(actual,p)); d=float(np.mean((p>=0)==(actual>=0)))
        skill=1-mae/max(bmae,1e-9); edge=d-bdir
        weights.append(max(.08,1+np.clip(skill,-.25,.25)*2.2+np.clip(edge,-.15,.15)*2.5))
    w=np.array(weights); w/=w.sum()
    ens=np.array([sum(r['preds'][n]*ww for n,ww in zip(names,w)) for r in rows])
    mae=float(mean_absolute_error(actual,ens)); dacc=float(np.mean((ens>=0)==(actual>=0)))
    skill=1-mae/max(bmae,1e-9); edge=dacc-bdir; resid=actual-ens
    return {'origins':len(rows),'directional_accuracy':dacc,'baseline_directional_accuracy':bdir,
            'directional_edge':edge,'mae_pct':mae*100,'baseline_mae_pct':bmae*100,
            'mae_skill_vs_baseline':skill,'resid':resid,'weights':w,'names':names,'pass':bool(edge>=.02 and skill>=.02)}


def run(asset,ticker):
    df=download(ticker); x=features(df); y=df.Close.shift(-4)/df.Close-1
    valid=x.notna().all(axis=1)&y.notna(); X=x.loc[valid]; Y=y.loc[valid]
    recent=evaluate(X,Y,28); stable=evaluate(X,Y,96)
    mods=templates(); latest=x.dropna().iloc[[-1]]; cur=[]; detail=[]
    for name,ww in zip(recent['names'],recent['weights']):
        m=clone(mods[name]); m.fit(X,Y); ret=float(m.predict(latest)[0]); cur.append(ret)
        detail.append({'name':name,'projected_return_pct':round(ret*100,3),'weight':round(float(ww),4)})
    pret=float(np.dot(np.array(cur),recent['weights'])); last=float(df.Close.iloc[-1]); price=last*(1+pret)
    resid=recent['resid']; q=float(np.quantile(np.abs(resid),.60)); focus=[last*(1+pret-q),last*(1+pret+q)]
    rawp=float(np.mean(pret+resid>0)); pup=float(np.clip(rawp,.15,.85))
    payload={
        'status':'ok','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'label':'Recent 4H specialist restored from pre-stabilization model',
        'predicted_price':round(price,2),'probability_up':round(pup,4),'probability_down':round(1-pup,4),
        'focus_zone':[round(focus[0],2),round(focus[1],2)],'models':detail,
        'recent_28':{k:(round(v,4) if isinstance(v,float) else v) for k,v in recent.items() if k not in ('resid','weights','names')},
        'stable_96':{k:(round(v,4) if isinstance(v,float) else v) for k,v in stable.items() if k not in ('resid','weights','names')},
        'overall_status':'PASS' if recent['pass'] and stable['pass'] else ('RECENT PASS / STABILITY FAIL' if recent['pass'] else 'FAIL'),
        'note':'The 28-origin test reproduces the earlier 4H timeframe. The 96-origin test is kept as a separate stability check so recent strength is not mistaken for long-run robustness.'
    }
    (DATA/f'{asset}_4h_specialist.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(asset,payload['overall_status'])


def main():
    for asset,ticker in ASSETS.items():
        try: run(asset,ticker)
        except Exception as e:
            (DATA/f'{asset}_4h_specialist.json').write_text(json.dumps({'status':'unavailable','asset':asset,'reason':str(e)},indent=2),encoding='utf-8')

if __name__=='__main__': main()
