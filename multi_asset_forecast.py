import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 5, 20)

ASSETS = {
    "gold": {
        "ticker": "GC=F",
        "instrument": "COMEX Gold Futures",
        "output": "live_forecast.json",
        "factors": {
            "silver": ("SI=F", "Silver Futures"),
            "dxy": ("DX-Y.NYB", "US Dollar Index"),
            "us10y": ("^TNX", "US 10Y Yield"),
            "vix": ("^VIX", "VIX"),
            "spy": ("SPY", "S&P 500 ETF"),
            "tlt": ("TLT", "Long Treasury ETF"),
        },
    },
    "silver": {
        "ticker": "SI=F",
        "instrument": "COMEX Silver Futures",
        "output": "silver_forecast.json",
        "factors": {
            "gold": ("GC=F", "Gold Futures"),
            "dxy": ("DX-Y.NYB", "US Dollar Index"),
            "us10y": ("^TNX", "US 10Y Yield"),
            "vix": ("^VIX", "VIX"),
            "copper": ("HG=F", "Copper Futures"),
            "spy": ("SPY", "S&P 500 ETF"),
        },
    },
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def normalize_index(df):
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_daily(ticker, period="10y"):
    df = flatten(yf.download(ticker, period=period, interval="1d", auto_adjust=False,
                             progress=False, threads=False))
    df = normalize_index(df)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    return df.dropna(subset=needed).copy()


def download_intraday(ticker):
    df = flatten(yf.download(ticker, period="10d", interval="1h", auto_adjust=False,
                             progress=False, threads=False))
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["Close"]).copy()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]),
                    (df["High"]-pc).abs(),
                    (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def load_factors(target_index, factor_defs):
    factors = pd.DataFrame(index=target_index)
    meta = {}
    for key, (ticker, name) in factor_defs.items():
        df = download_daily(ticker)
        if df.empty:
            meta[key] = {"ticker": ticker, "name": name, "available": False}
            continue
        s = df["Close"].astype(float).reindex(target_index).ffill(limit=3)
        factors[key] = s
        meta[key] = {"ticker": ticker, "name": name, "available": bool(s.notna().sum() > 400)}
    return factors, meta


def make_features(target, factors):
    c = target["Close"].astype(float)
    x = pd.DataFrame(index=target.index)
    for n in (1, 5, 20, 60):
        x[f"ret{n}"] = c.pct_change(n)
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    x["ema20_gap"] = c/e20 - 1
    x["ema50_gap"] = c/e50 - 1
    x["ema200_gap"] = c/e200 - 1
    x["ema20_50"] = e20/e50 - 1
    x["ema50_200"] = e50/e200 - 1
    x["rsi14"] = (rsi(c)-50)/50
    x["vol20"] = c.pct_change().rolling(20).std()
    x["vol60"] = c.pct_change().rolling(60).std()
    x["atr14_pct"] = atr(target)/c
    x["z20"] = (c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    hi20, lo20 = target["High"].rolling(20).max(), target["Low"].rolling(20).min()
    x["range_pos20"] = (c-lo20)/(hi20-lo20).replace(0,np.nan)-0.5
    x["drawdown60"] = c/c.rolling(60).max()-1

    for key in factors.columns:
        s = factors[key].astype(float)
        for n in (1, 5, 20):
            x[f"{key}_ret{n}"] = s.pct_change(n)
        x[f"{key}_z60"] = (s-s.rolling(60).mean())/s.rolling(60).std().replace(0,np.nan)
        x[f"{key}_ema20_gap"] = s/s.ewm(span=20,adjust=False).mean()-1

    if "dxy" in factors:
        x["dxy_corr60"] = c.pct_change().rolling(60).corr(factors["dxy"].pct_change())
    if "us10y" in factors:
        x["yield_change5"] = factors["us10y"].diff(5)
        x["yield_change20"] = factors["us10y"].diff(20)
    return x.replace([np.inf,-np.inf],np.nan)


def models():
    return {
        "Logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
            C=.35,max_iter=3000,class_weight="balanced",random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=240,max_depth=5,min_samples_leaf=12,
            max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=-1),
        "Gradient Boost": HistGradientBoostingClassifier(max_iter=150,max_depth=3,learning_rate=.045,
            l2_regularization=.25,random_state=42),
        "Regime KNN": Pipeline([("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=45,weights="distance"))]),
    }


def forecast_one(target, feats, horizon):
    forward = target["Close"].shift(-horizon)/target["Close"]-1
    y = (forward>0).astype(int)
    valid = feats.notna().all(axis=1) & forward.notna()
    X, Y = feats.loc[valid], y.loc[valid]
    if len(X) < 650 or Y.nunique()<2:
        raise RuntimeError(f"Insufficient clean history for {horizon}d")
    holdout = min(252,max(140,len(X)//5)); split=len(X)-holdout
    Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=Y.iloc[:split],Y.iloc[split:]
    latest=feats.dropna().iloc[[-1]]
    base=float(max(yte.mean(),1-yte.mean()))
    results=[]
    for name,template in models().items():
        m=clone(template)
        try:
            m.fit(Xtr,ytr); tp=m.predict_proba(Xte)[:,1]
            acc=float(accuracy_score(yte,tp>=.5)); bs=float(brier_score_loss(yte,tp))
            m.fit(X,Y); p=float(m.predict_proba(latest)[:,1][0])
            results.append((name,p,acc,bs))
        except Exception:
            pass
    if len(results)<2: raise RuntimeError("Too few ensemble models completed")
    weights=[]
    for _,_,acc,bs in results:
        skill=max(-.03,min(.08,acc-base)); calibration=max(0,.25-bs)
        weights.append(max(.10,1+skill*8+calibration*1.5))
    weights=np.array(weights); weights=weights/weights.sum()
    raw=float(sum(r[1]*w for r,w in zip(results,weights)))
    acc=float(sum(r[2]*w for r,w in zip(results,weights)))
    bs=float(sum(r[3]*w for r,w in zip(results,weights)))
    votes=[r[1]>=.5 for r in results]; majority=sum(votes)>=len(votes)/2
    agreement=float(sum(v==majority for v in votes)/len(votes))
    edge=acc-base; reliability=float(np.clip((edge+.015)/.07,.12,1))
    p=.5+(raw-.5)*(.30+.70*reliability); p=float(np.clip(p,.30,.70))
    hist=(target["Close"]/target["Close"].shift(horizon)-1).dropna().tail(1260)
    q20,q50,q80=[float(v) for v in hist.quantile([.2,.5,.8]).values]
    last=float(target["Close"].iloc[-1]); vol=float(target["Close"].pct_change().tail(60).std())
    tilt=(p-.5)*2*vol*np.sqrt(horizon)*.40
    lo,mid,hi=[last*(1+q+tilt) for q in (q20,q50,q80)]
    if p>=.58: signal="Bullish bias"
    elif p<=.42: signal="Bearish bias"
    else: signal="Neutral / no strong edge"
    distance=abs(p-.5)
    if edge>=.055 and distance>=.12 and agreement>=.75: confidence="High"
    elif edge>=.02 and distance>=.07 and agreement>=.65: confidence="Moderate"
    else: confidence="Low"
    detail=[]
    for (name,pp,aa,bb),w in zip(results,weights):
        detail.append({"name":name,"probability_up":round(pp,4),"backtest_accuracy":round(aa,4),
                       "brier_score":round(bb,4),"weight":round(float(w),4)})
    return {"horizon_days":horizon,"probability_up":round(p,4),"probability_down":round(1-p,4),
            "signal":signal,"confidence":confidence,"forecast_mid":round(mid,2),
            "range_20_80":[round(lo,2),round(hi,2)],"backtest_accuracy":round(acc,4),
            "naive_baseline":round(base,4),"backtest_edge":round(edge,4),"brier_score":round(bs,4),
            "model_agreement":round(agreement,4),"test_observations":holdout,"models":detail}


def factor_impact(asset_key,key,s,target_close):
    s=s.dropna(); g=target_close.dropna()
    if len(s)<25: return "Unavailable","Not enough recent data"
    r20=float(s.iloc[-1]/s.iloc[-21]-1); target20=float(g.iloc[-1]/g.iloc[-21]-1)
    if key=="dxy":
        return ("Supportive","Dollar weakened over 20 trading days") if r20<=-.007 else (("Headwind","Dollar strengthened over 20 trading days") if r20>=.007 else ("Neutral","Dollar trend is mixed"))
    if key=="us10y":
        d=float(s.iloc[-1]-s.iloc[-21])
        return ("Supportive","10Y yield fell") if d<=-.10 else (("Headwind","10Y yield rose") if d>=.10 else ("Neutral","10Y yield is range-bound"))
    if key=="vix":
        level=float(s.iloc[-1])
        return ("Risk-off","Equity volatility is elevated") if level>=28 else (("Risk-on","Equity volatility is low") if level<=15 else ("Neutral","Equity volatility is moderate"))
    if key=="silver":
        rel=r20-target20
        return ("Supportive","Silver is rising and outperforming gold") if r20>0 and rel>.01 else (("Headwind","Silver is weak") if r20<-.02 else ("Neutral","Silver breadth is mixed"))
    if key=="gold":
        return ("Supportive","Gold trend supports precious metals") if r20>.015 else (("Headwind","Gold trend is weak") if r20<-.015 else ("Neutral","Gold is range-bound"))
    if key=="copper":
        return ("Supportive","Copper strength supports industrial-metals demand") if r20>.025 else (("Headwind","Copper weakness signals softer cyclical demand") if r20<-.025 else ("Neutral","Copper trend is mixed"))
    if key=="spy":
        return ("Risk-on","Equity trend is constructive") if r20>.025 else (("Risk-off","Equity trend is weak") if r20<-.04 else ("Neutral","Equity trend is mixed"))
    if key=="tlt":
        return ("Supportive","Long-duration bonds are firm") if r20>.015 else (("Headwind","Long-duration bonds are weak") if r20<-.015 else ("Neutral","Bond trend is mixed"))
    return "Neutral","Mixed signal"


def build_summary(asset_key, forecasts, factor_rows, levels, regime):
    f5=next(f for f in forecasts if f["horizon_days"]==5)
    supportive=[f["name"] for f in factor_rows if f["impact"]=="Supportive"]
    headwinds=[f["name"] for f in factor_rows if f["impact"]=="Headwind"]
    p=f5["probability_up"]
    bias="Bullish" if p>=.58 else ("Bearish" if p<=.42 else "Neutral")
    conviction=f5["confidence"]
    if supportive and headwinds:
        read=f"Supportive: {', '.join(supportive[:2])}. Headwinds: {', '.join(headwinds[:2])}."
    elif supportive:
        read=f"Supportive backdrop led by {', '.join(supportive[:3])}."
    elif headwinds:
        read=f"Macro headwinds led by {', '.join(headwinds[:3])}."
    else:
        read="Cross-market factors are mixed with no dominant macro edge."
    trigger=(f"A sustained move above {levels['resistance_20d']:.2f} would strengthen upside confirmation; "
             f"below {levels['support_20d']:.2f} would weaken the setup.")
    risk=("Model agreement/backtest edge is weak, so confidence should stay limited." if f5["confidence"]=="Low"
          else "Macro reversals and volatility expansion can invalidate the statistical setup quickly.")
    return {"bias":bias,"conviction":conviction,"model_view":read,"trigger":trigger,"risk":risk,
            "regime":regime["overall"]}


def run_asset(asset_key,cfg):
    target=download_daily(cfg["ticker"])
    if len(target)<700: raise RuntimeError(f"Too little {asset_key} history")
    factors,meta=load_factors(target.index,cfg["factors"])
    factors=factors[[c for c in factors.columns if meta[c]["available"]]]
    feats=make_features(target,factors)
    forecasts=[forecast_one(target,feats,h) for h in HORIZONS]
    c=target["Close"].astype(float); intraday=download_intraday(cfg["ticker"])
    latest=float(intraday["Close"].iloc[-1]) if not intraday.empty else float(c.iloc[-1])
    prev=float(c.iloc[-2]); change=(latest/prev-1)*100
    e20=float(c.ewm(span=20,adjust=False).mean().iloc[-1]); e50=float(c.ewm(span=50,adjust=False).mean().iloc[-1]); e200=float(c.ewm(span=200,adjust=False).mean().iloc[-1])
    r=float(rsi(c).iloc[-1]); atrd=float(atr(target).iloc[-1]); atrpct=atrd/float(c.iloc[-1])*100
    vol20=c.pct_change().rolling(20).std(); histvol=vol20.dropna().tail(756)
    volpct=float((histvol<=vol20.iloc[-1]).mean()) if len(histvol) else .5
    volreg="High" if volpct>=.75 else ("Low" if volpct<=.25 else "Normal")
    if e20>e50>e200: trend="Strong uptrend"
    elif e20<e50<e200: trend="Strong downtrend"
    elif e20>e50: trend="Uptrend / transitional"
    elif e20<e50: trend="Downtrend / transitional"
    else: trend="Mixed / transitional"
    factor_rows=[]; macro_score=0
    for key in factors.columns:
        impact,reason=factor_impact(asset_key,key,factors[key],c)
        if impact=="Supportive": macro_score+=1
        elif impact=="Headwind": macro_score-=1
        s=factors[key].dropna(); r5=(s.iloc[-1]/s.iloc[-6]-1)*100 if len(s)>6 else np.nan; r20=(s.iloc[-1]/s.iloc[-21]-1)*100 if len(s)>21 else np.nan
        factor_rows.append({"key":key,"name":cfg["factors"][key][1],"symbol":cfg["factors"][key][0],
                            "value":round(float(s.iloc[-1]),3),"change_5d_pct":round(float(r5),2),
                            "change_20d_pct":round(float(r20),2),"impact":impact,"reason":reason})
    macro="Supportive" if macro_score>=2 else ("Headwind" if macro_score<=-2 else "Mixed")
    f5=next(f for f in forecasts if f["horizon_days"]==5)
    if f5["probability_up"]>=.58 and macro_score>=0: overall="Bullish"
    elif f5["probability_up"]<=.42 and macro_score<=0: overall="Bearish"
    else: overall="Neutral / mixed"
    regime={"overall":overall,"trend":trend,"macro":macro,"macro_score":macro_score,"volatility":volreg,"volatility_percentile":round(volpct,4)}
    close_ref=float(c.iloc[-1]); low20=float(target["Low"].tail(20).min()); high20=float(target["High"].tail(20).max()); low60=float(target["Low"].tail(60).min()); high60=float(target["High"].tail(60).max())
    levels={"daily_close_reference":round(close_ref,2),"support_20d":round(low20,2),"resistance_20d":round(high20,2),
            "support_60d":round(low60,2),"resistance_60d":round(high60,2),"atr14_dollars":round(atrd,2),
            "expected_move_1d":[round(close_ref-atrd,2),round(close_ref+atrd,2)],
            "expected_move_5d":[round(close_ref-atrd*np.sqrt(5),2),round(close_ref+atrd*np.sqrt(5),2)]}
    summary=build_summary(asset_key,forecasts,factor_rows,levels,regime)
    series=[{"date":pd.Timestamp(i).strftime("%Y-%m-%d"),"close":round(float(row["Close"]),2)} for i,row in target.tail(260).iterrows()]
    updated=datetime.now(timezone.utc).isoformat()
    market_time=pd.Timestamp(intraday.index[-1]).isoformat() if not intraday.empty else pd.Timestamp(target.index[-1]).isoformat()
    payload={"status":"ok","asset":asset_key,"symbol":cfg["ticker"],"instrument":cfg["instrument"],"updated_utc":updated,
             "latest_market_time":market_time,"latest_price":round(latest,2),"change_pct":round(change,3),"guru_summary":summary,
             "regime":regime,"indicators":{"ema20":round(e20,2),"ema50":round(e50,2),"ema200":round(e200,2),"rsi14":round(r,2),"atr14_pct":round(atrpct,2)},
             "levels":levels,"factors":factor_rows,"forecasts":forecasts,"series":series,
             "methodology":"Four-model ensemble using price momentum, trend, volatility and cross-market factors. Probabilities are shrunk toward 50% when recent holdout skill is weak.",
             "disclaimer":"Research/education only. Market data may be delayed. Forecasts are probabilistic and can be wrong; this is not trading or investment advice."}
    (OUT_DIR/cfg["output"]).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(asset_key, latest, [(f["horizon_days"],f["probability_up"],f["confidence"]) for f in forecasts])


def main():
    for key,cfg in ASSETS.items():
        run_asset(key,cfg)

if __name__=="__main__":
    main()
