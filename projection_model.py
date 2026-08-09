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

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS = {
    "gold": {"ticker": "GC=F", "name": "COMEX Gold Futures", "output": "gold_projections.json"},
    "silver": {"ticker": "SI=F", "name": "COMEX Silver Futures", "output": "silver_projections.json"},
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker, period, interval):
    df = flatten(yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                             progress=False, threads=False))
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No data returned for {ticker} {interval}")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).copy()


def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan); return 100-100/(1+rs)


def make_features(df, intraday=False):
    c=df["Close"].astype(float); h=df["High"].astype(float); l=df["Low"].astype(float)
    x=pd.DataFrame(index=df.index)
    periods=(1,2,4,8,24,48,120) if intraday else (1,2,5,10,20,60,120)
    for n in periods:
        x[f"ret{n}"]=c.pct_change(n)
    for span in ((8,21,55) if intraday else (10,20,50,200)):
        ema=c.ewm(span=span,adjust=False).mean(); x[f"ema{span}_gap"]=c/ema-1
    x["rsi14"]=(rsi(c)-50)/50
    x["vol20"]=c.pct_change().rolling(20).std()
    x["vol60"]=c.pct_change().rolling(60).std()
    x["z20"]=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    x["range20"]=(c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()).replace(0,np.nan)-.5
    x["slope10"]=(c/c.shift(10)-1)/10
    x["slope30"]=(c/c.shift(30)-1)/30
    return x.replace([np.inf,-np.inf],np.nan)


def model_templates():
    return {
        "Ridge": Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=8.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=260,max_depth=6,min_samples_leaf=10,
                                                max_features="sqrt",random_state=42,n_jobs=-1),
        "Gradient Boost": HistGradientBoostingRegressor(max_iter=180,max_depth=3,learning_rate=.045,
                                                         l2_regularization=.35,random_state=42),
    }


def project(df, steps, label, intraday=False):
    feats=make_features(df,intraday=intraday)
    forward=df["Close"].shift(-steps)/df["Close"]-1
    valid=feats.notna().all(axis=1)&forward.notna()
    X=feats.loc[valid]; y=forward.loc[valid]
    min_rows=450 if intraday else 700
    if len(X)<min_rows:
        raise RuntimeError(f"Insufficient data for {label}")
    holdout=min(252 if not intraday else 300,max(120,len(X)//5)); split=len(X)-holdout
    Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
    latest=feats.dropna().iloc[[-1]]
    rows=[]; weights=[]
    for name,tpl in model_templates().items():
        m=clone(tpl)
        try:
            m.fit(Xtr,ytr); pred=m.predict(Xte); mae=float(mean_absolute_error(yte,pred)); direction=float(np.mean((pred>=0)==(yte.values>=0)))
            m.fit(X,y); now=float(m.predict(latest)[0])
            rows.append((name,now,mae,direction))
        except Exception:
            pass
    if len(rows)<2:
        raise RuntimeError(f"Projection ensemble failed for {label}")
    median_abs=float(np.median(np.abs(yte.values)))
    for _,_,mae,dacc in rows:
        err_penalty=max(.15,1-mae/max(median_abs,1e-6)); skill=max(.2,dacc)
        weights.append(err_penalty*skill)
    w=np.array(weights,dtype=float); w=w/w.sum()
    projected_return=float(sum(r[1]*ww for r,ww in zip(rows,w)))
    dacc=float(sum(r[3]*ww for r,ww in zip(rows,w)))
    mae=float(sum(r[2]*ww for r,ww in zip(rows,w)))
    last=float(df["Close"].iloc[-1]); model_price=last*(1+projected_return)

    residuals=[]
    for _,tpl in model_templates().items():
        try:
            m=clone(tpl); m.fit(Xtr,ytr); residuals.extend((yte.values-m.predict(Xte)).tolist())
        except Exception:
            pass
    residuals=np.array(residuals,dtype=float)
    q25,q75=np.quantile(residuals,[.25,.75]) if len(residuals)>50 else (-mae,mae)
    tight_low=last*(1+projected_return+q25); tight_high=last*(1+projected_return+q75)

    directions=[r[1]>=0 for r in rows]; majority=sum(directions)>=len(directions)/2
    agreement=sum(d==majority for d in directions)/len(directions)
    if dacc>=.58 and agreement>=.67: conf="High"
    elif dacc>=.54 and agreement>=.67: conf="Moderate"
    else: conf="Low"
    direction_sign=1.0 if projected_return>=0 else -1.0
    skill_strength=max(0.0,dacc-.5)*1.4
    magnitude_strength=min(.10,abs(float(np.tanh(projected_return/max(mae,1e-5))))*.10)
    p_up=float(np.clip(.5+direction_sign*(skill_strength+magnitude_strength),.25,.75))

    detail=[]
    for (name,ret,em,da),ww in zip(rows,w):
        detail.append({"name":name,"projected_return_pct":round(ret*100,3),"projected_price":round(last*(1+ret),2),
                       "backtest_mae_pct":round(em*100,3),"directional_accuracy":round(da,4),"weight":round(float(ww),4)})
    return {"horizon":label,"steps":steps,"model_price":round(model_price,2),"projected_return_pct":round(projected_return*100,3),
            "probability_up":round(p_up,4),"probability_down":round(1-p_up,4),"confidence":conf,
            "tight_model_zone":[round(float(tight_low),2),round(float(tight_high),2)],
            "backtest_directional_accuracy":round(dacc,4),"backtest_mae_pct":round(mae*100,3),
            "model_agreement":round(float(agreement),4),"models":detail}


def run_asset(key,cfg):
    daily=download(cfg["ticker"],"10y","1d")
    hourly=download(cfg["ticker"],"2y","1h")
    projections=[
        project(hourly,4,"4 Hours",intraday=True),
        project(daily,1,"1 Day"),
        project(daily,5,"1 Week"),
        project(daily,252,"1 Year"),
    ]
    payload={"status":"ok","asset":key,"symbol":cfg["ticker"],"instrument":cfg["name"],
             "updated_utc":datetime.now(timezone.utc).isoformat(),"latest_price":round(float(hourly["Close"].iloc[-1]),2),
             "projections":projections,
             "note":"Model price is an ensemble regression estimate. Tight model zones come from recent holdout residuals and are not guaranteed trading ranges. The 1-year estimate has structurally higher uncertainty than 4H/1D/1W."}
    (OUT_DIR/cfg["output"]).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(key,[(p["horizon"],p["model_price"],p["confidence"]) for p in projections])


def main():
    for key,cfg in ASSETS.items():
        run_asset(key,cfg)

if __name__=="__main__": main()
