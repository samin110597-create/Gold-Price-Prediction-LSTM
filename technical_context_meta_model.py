import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
DATA.mkdir(exist_ok=True)

ASSETS = {
    "gold": {
        "ticker": "GC=F",
        "output": "gold_technical_context.json",
        "factors": {
            "silver": "SI=F", "usd": "DX-Y.NYB", "yield10": "^TNX", "vix": "^VIX",
            "sp500": "SPY", "bonds": "TLT", "copper": "HG=F", "oil": "CL=F",
        },
    },
    "silver": {
        "ticker": "SI=F",
        "output": "silver_technical_context.json",
        "factors": {
            "gold": "GC=F", "usd": "DX-Y.NYB", "yield10": "^TNX", "vix": "^VIX",
            "sp500": "SPY", "bonds": "TLT", "copper": "HG=F", "oil": "CL=F",
        },
    },
}
HORIZONS = (1, 5, 20)


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download_one(ticker):
    df = flatten(yf.download(ticker, start="2008-01-01", interval="1d", auto_adjust=False,
                             progress=False, threads=False))
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No daily data for {ticker}")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy(); df.index = idx
    return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def target_features(df):
    c, o, h, l = [df[k].astype(float) for k in ("Close", "Open", "High", "Low")]
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    x = pd.DataFrame(index=df.index)
    for n in (1, 2, 5, 10, 20, 60):
        x[f"ret{n}"] = c.pct_change(n)
    e9 = c.ewm(span=9, adjust=False).mean(); e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean(); e200 = c.ewm(span=200, adjust=False).mean()
    for name, e in (("9", e9), ("20", e20), ("50", e50), ("200", e200)):
        x[f"ema_gap{name}"] = c/e - 1
    x["ema20_50"] = e20/e50 - 1
    x["ema50_200"] = e50/e200 - 1
    rr = rsi(c); x["rsi"] = (rr-50)/50; x["rsi_d5"] = rr.diff(5)/100
    macd = c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean()
    ms = macd.ewm(span=9, adjust=False).mean(); mh = macd-ms
    x["macd_hist"] = mh/c; x["macd_accel"] = mh.diff(3)/c
    a = atr(df); x["atr_pct"] = a/c
    x["vol20"] = c.pct_change().rolling(20).std(); x["vol60"] = c.pct_change().rolling(60).std()
    x["vol_ratio"] = x["vol20"]/x["vol60"].replace(0, np.nan)
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(); upper = mid+2*sd; lower = mid-2*sd
    x["bb_pct"] = (c-lower)/(upper-lower).replace(0, np.nan)-.5
    x["range20"] = (c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()).replace(0, np.nan)-.5
    x["range60"] = (c-l.rolling(60).min())/(h.rolling(60).max()-l.rolling(60).min()).replace(0, np.nan)-.5
    x["body"] = (c-o)/c
    x["upper_wick"] = (h-pd.concat([o,c], axis=1).max(axis=1))/c
    x["lower_wick"] = (pd.concat([o,c], axis=1).min(axis=1)-l)/c
    x["z20"] = (c-c.rolling(20).mean())/c.rolling(20).std().replace(0, np.nan)
    if v.abs().sum() > 0:
        x["volume_ratio"] = v/v.rolling(20).mean().replace(0, np.nan)
        obv = (np.sign(c.diff()).fillna(0)*v).cumsum()
        x["obv_slope"] = obv.diff(10)/obv.abs().rolling(60).max().replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def base_direction(df):
    c = df["Close"].astype(float)
    e9 = c.ewm(span=9, adjust=False).mean(); e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean(); e200 = c.ewm(span=200, adjust=False).mean()
    rr = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean()
    ms = macd.ewm(span=9, adjust=False).mean(); mh = macd-ms
    high20 = df["High"].shift(1).rolling(20).max(); low20 = df["Low"].shift(1).rolling(20).min()
    score = pd.Series(0.0, index=df.index)
    score += np.where(c>e20, 1, -1)
    score += np.where(e20>e50, 1, -1)
    score += np.where(e50>e200, .7, -.7)
    score += np.where(macd>ms, .8, -.8)
    score += np.where(mh>mh.shift(2), .4, -.4)
    score += np.where(rr>=55, .5, np.where(rr<=45, -.5, 0))
    score += np.where(c>high20, .8, np.where(c<low20, -.8, 0))
    direction = pd.Series(np.where(score>=1.5, 1, np.where(score<=-1.5, -1, 0)), index=df.index)
    return direction, score


def factor_features(target_index, factor_dfs):
    x = pd.DataFrame(index=target_index)
    for name, df in factor_dfs.items():
        s = df["Close"].astype(float).reindex(target_index).ffill(limit=3)
        for n in (1, 5, 20, 60):
            x[f"{name}_ret{n}"] = s.pct_change(n)
        x[f"{name}_z60"] = (s-s.rolling(60).mean())/s.rolling(60).std().replace(0, np.nan)
        x[f"{name}_vol20"] = s.pct_change().rolling(20).std()
    if "silver" in factor_dfs:
        other = factor_dfs["silver"]["Close"].astype(float).reindex(target_index).ffill(limit=3)
        # target close added by caller as ratio feature where relevant
        x["related_metal_mom20"] = other.pct_change(20)
    if "gold" in factor_dfs:
        other = factor_dfs["gold"]["Close"].astype(float).reindex(target_index).ffill(limit=3)
        x["related_metal_mom20"] = other.pct_change(20)
    return x.replace([np.inf, -np.inf], np.nan)


def build_panel(cfg):
    target = download_one(cfg["ticker"])
    factors = {}
    errors = []
    for name, ticker in cfg["factors"].items():
        try:
            factors[name] = download_one(ticker)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    tf = target_features(target)
    ff = factor_features(target.index, factors)
    c = target["Close"].astype(float)
    if "silver" in factors:
        s = factors["silver"]["Close"].astype(float).reindex(target.index).ffill(limit=3)
        ff["gold_silver_ratio_mom20"] = (c/s).pct_change(20)
    if "gold" in factors:
        g = factors["gold"]["Close"].astype(float).reindex(target.index).ffill(limit=3)
        ff["silver_gold_ratio_mom20"] = (c/g).pct_change(20)
    X = pd.concat([tf, ff], axis=1)
    # Remove factors with insufficient history rather than losing the whole panel.
    X = X.dropna(axis=1, thresh=max(500, int(len(X)*0.55)))
    direction, score = base_direction(target)
    return target, X, direction, score, errors


def models():
    return {
        "Logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=.12, max_iter=2500, class_weight="balanced", random_state=42))]),
        "Extra Trees": ExtraTreesClassifier(n_estimators=160, max_depth=6, min_samples_leaf=16, max_features="sqrt", class_weight="balanced", random_state=43, n_jobs=-1),
        "Gradient Boost": HistGradientBoostingClassifier(max_iter=120, max_depth=3, learning_rate=.035, l2_regularization=1.0, random_state=44),
    }


def folds(n, h, min_train=1300, block=126, max_folds=6):
    avail = n-min_train-h
    if avail < block*2: return []
    k = min(max_folds, avail//block)
    first = n-k*block
    return [(ts-h, ts, min(n, ts+block)) for ts in [first+i*block for i in range(k)] if ts-h>=min_train]


def choose_threshold(p, y, baseline_orientation):
    best = None
    for th in (.56,.58,.60,.62,.64,.66,.68,.70):
        long = p>=th; flip = p<=1-th; active = long|flip
        if active.sum()<25 or active.mean()<.12: continue
        correct = np.where(long[active], y[active]==1, y[active]==0)
        acc = float(np.mean(correct))
        if baseline_orientation == 1:
            base_acc = float(np.mean(y[active]==1))
        else:
            base_acc = float(np.mean(y[active]==0))
        lift = acc-base_acc
        score = lift + min(float(active.mean()), .45)*.04
        row = (score, th, acc, base_acc, lift, float(active.mean()), int(active.sum()))
        if best is None or row[0]>best[0]: best=row
    return best


def horizon_model(target, X0, direction0, h):
    fwd = target["Close"].shift(-h)/target["Close"]-1
    candidate = direction0 != 0
    meta_y = ((fwd*direction0)>0).astype(int)
    valid = candidate & X0.notna().all(axis=1) & fwd.notna()
    X = X0.loc[valid]; y = meta_y.loc[valid].astype(int); d = direction0.loc[valid].astype(int)
    dates = X.index
    if len(X)<1600: raise RuntimeError(f"Insufficient candidate history {h}D: {len(X)}")
    fs = folds(len(X), h)
    if len(fs)<4: raise RuntimeError(f"Too few context folds {h}D")
    templates=models(); rows=[]; all_decisions=[]; thresholds=[]
    for train_end, test_start, test_end in fs:
        cal_len=min(252,max(126,train_end//5)); cal_start=train_end-cal_len; fit_end=cal_start-h
        if fit_end<900: continue
        Xfit,yfit=X.iloc[:fit_end],y.iloc[:fit_end]
        Xcal,ycal=X.iloc[cal_start:train_end],y.iloc[cal_start:train_end]
        Xtest,ytest=X.iloc[test_start:test_end],y.iloc[test_start:test_end]
        calps=[]; testps=[]
        for tpl in templates.values():
            try:
                m=clone(tpl);m.fit(Xfit,yfit);calps.append(m.predict_proba(Xcal)[:,1]);testps.append(m.predict_proba(Xtest)[:,1])
            except Exception: pass
        if len(calps)<3: continue
        pcal=np.mean(np.vstack(calps),axis=0); ptest=np.mean(np.vstack(testps),axis=0)
        orientation=1 if float(ycal.mean())>=.5 else 0
        chosen=choose_threshold(pcal,ycal.values,orientation)
        if chosen is None:
            th=.70
        else:
            th=float(chosen[1])
        thresholds.append(th)
        long=ptest>=th; flip=ptest<=1-th; active=long|flip
        yy=ytest.values
        base_test_correct=(yy==orientation)
        base_acc=float(np.mean(base_test_correct))
        if active.any():
            correct=np.where(long[active],yy[active]==1,yy[active]==0)
            acc=float(np.mean(correct)); lift=acc-base_acc
        else: acc=np.nan;lift=np.nan
        rows.append({"threshold":round(th,3),"coverage":round(float(active.mean()),4),"signals":int(active.sum()),"accuracy":round(acc,4) if np.isfinite(acc) else None,"base_test_accuracy":round(base_acc,4),"lift":round(lift,4) if np.isfinite(lift) else None})
        for j in range(len(yy)):
            if active[j]:
                all_decisions.append((int(yy[j]), 1 if long[j] else 0, orientation))
    if len(all_decisions)<80: raise RuntimeError(f"Too few OOS meta decisions {h}D")
    arr=np.asarray(all_decisions,int); actual=arr[:,0]; chosen_orientation=arr[:,1]; base_orientation=arr[:,2]
    acc=float(np.mean(actual==chosen_orientation)); base=float(np.mean(actual==base_orientation)); lift=acc-base
    coverage=float(sum(r["signals"] for r in rows)/max(1,sum(126 for _ in rows)))
    # Current model trained on all historical candidates.
    current_ps=[]
    latest=X0.dropna().iloc[[-1]]
    for tpl in templates.values():
        try:
            m=clone(tpl);m.fit(X,y);current_ps.append(float(m.predict_proba(latest)[:,1][0]))
        except Exception: pass
    pnow=float(np.mean(current_ps)) if current_ps else .5
    thnow=float(np.median(thresholds)) if thresholds else .66
    current_base=int(direction0.loc[latest.index[0]]) if latest.index[0] in direction0.index else 0
    audit_pass=bool(lift>=.03 and acc>=.56 and len(all_decisions)>=100 and coverage>=.12)
    if audit_pass and current_base!=0 and pnow>=thnow:
        signal="Bullish" if current_base>0 else "Bearish"; active_now=True; mode="CONFIRM"
    elif audit_pass and current_base!=0 and pnow<=1-thnow:
        signal="Bearish" if current_base>0 else "Bullish"; active_now=True; mode="INVERT"
    else:
        signal="NO EDGE / ABSTAIN";active_now=False;mode="ABSTAIN"
    return {"horizon_days":h,"signal":signal,"active_signal":active_now,"mode":mode,"meta_probability_base_correct":round(pnow,4),"decision_threshold":round(thnow,3),"oos_accuracy":round(acc,4),"base_signal_accuracy":round(base,4),"lift_vs_base":round(lift,4),"coverage":round(coverage,4),"oos_signals":len(all_decisions),"audit_pass":audit_pass,"validation":"nested chronological meta-label walk-forward; cross-market context; threshold chosen only on earlier calibration slice","folds":rows}


def run_asset(asset,cfg):
    target,X,direction,score,errors=build_panel(cfg)
    rows=[]
    for h in HORIZONS:
        try: rows.append(horizon_model(target,X,direction,h))
        except Exception as exc: rows.append({"horizon_days":h,"signal":"NO EDGE / UNAVAILABLE","active_signal":False,"audit_pass":False,"error":str(exc)})
    payload={"status":"ok","asset":asset,"symbol":cfg["ticker"],"updated_utc":datetime.now(timezone.utc).isoformat(),"method":"Technical setup + cross-market meta model. Stage 1 creates chart direction; stage 2 confirms, rejects or inverts it using market context.","horizons":rows,"factor_errors":errors,"note":"This challenger is not promoted unless it improves unseen walk-forward accuracy versus the base technical setup."}
    (DATA/cfg["output"]).write_text(json.dumps(payload,indent=2),encoding="utf-8")


def main():
    for asset,cfg in ASSETS.items(): run_asset(asset,cfg)

if __name__=="__main__": main()
