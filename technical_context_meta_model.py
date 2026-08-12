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

DATA=Path("data");DATA.mkdir(exist_ok=True)
ASSETS={
 "gold":{"ticker":"GC=F","output":"gold_technical_context.json","factors":{"silver":"SI=F","usd":"DX-Y.NYB","yield10":"^TNX","vix":"^VIX","sp500":"SPY","bonds":"TLT","copper":"HG=F","oil":"CL=F"}},
 "silver":{"ticker":"SI=F","output":"silver_technical_context.json","factors":{"gold":"GC=F","usd":"DX-Y.NYB","yield10":"^TNX","vix":"^VIX","sp500":"SPY","bonds":"TLT","copper":"HG=F","oil":"CL=F"}},
}
HORIZONS=(1,5,20)

def flatten(df):
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0] for c in df.columns]
    return df

def dl(ticker):
    df=flatten(yf.download(ticker,start="2008-01-01",interval="1d",auto_adjust=False,progress=False,threads=False))
    if df.empty or "Close" not in df.columns:raise RuntimeError(f"No daily data for {ticker}")
    idx=pd.to_datetime(df.index)
    if getattr(idx,"tz",None) is not None:idx=idx.tz_localize(None)
    df=df.copy();df.index=idx
    return df.dropna(subset=["Open","High","Low","Close"]).sort_index()

def rsi(s,n=14):
    d=s.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-100/(1+rs)
def atr(df,n=14):
    pc=df.Close.shift(1);tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False).mean()

def target_features(df):
    c,o,h,l=[df[k].astype(float) for k in ("Close","Open","High","Low")];v=df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.,index=df.index);x=pd.DataFrame(index=df.index)
    for n in (1,2,5,10,20,60):x[f"ret{n}"]=c.pct_change(n)
    e9=c.ewm(span=9,adjust=False).mean();e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();e200=c.ewm(span=200,adjust=False).mean()
    for name,e in (("9",e9),("20",e20),("50",e50),("200",e200)):x[f"ema_gap{name}"]=c/e-1
    x["ema20_50"]=e20/e50-1;x["ema50_200"]=e50/e200-1;rr=rsi(c);x["rsi"]=(rr-50)/50;x["rsi_d5"]=rr.diff(5)/100
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();ms=m.ewm(span=9,adjust=False).mean();mh=m-ms;x["macd_hist"]=mh/c;x["macd_accel"]=mh.diff(3)/c
    a=atr(df);x["atr_pct"]=a/c;x["vol20"]=c.pct_change().rolling(20).std();x["vol60"]=c.pct_change().rolling(60).std();x["vol_ratio"]=x["vol20"]/x["vol60"].replace(0,np.nan)
    mid=c.rolling(20).mean();sd=c.rolling(20).std();up=mid+2*sd;lo=mid-2*sd;x["bb_pct"]=(c-lo)/(up-lo).replace(0,np.nan)-.5
    x["range20"]=(c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()).replace(0,np.nan)-.5;x["range60"]=(c-l.rolling(60).min())/(h.rolling(60).max()-l.rolling(60).min()).replace(0,np.nan)-.5
    x["body"]=(c-o)/c;x["upper_wick"]=(h-pd.concat([o,c],axis=1).max(axis=1))/c;x["lower_wick"]=(pd.concat([o,c],axis=1).min(axis=1)-l)/c;x["z20"]=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    if v.abs().sum()>0:
        x["volume_ratio"]=v/v.rolling(20).mean().replace(0,np.nan);obv=(np.sign(c.diff()).fillna(0)*v).cumsum();x["obv_slope"]=obv.diff(10)/obv.abs().rolling(60).max().replace(0,np.nan)
    return x.replace([np.inf,-np.inf],np.nan)

def base_direction(df):
    c=df.Close.astype(float);e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();e200=c.ewm(span=200,adjust=False).mean();rr=rsi(c);m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();ms=m.ewm(span=9,adjust=False).mean();mh=m-ms;hi=df.High.shift(1).rolling(20).max();lo=df.Low.shift(1).rolling(20).min();s=pd.Series(0.,index=df.index);s+=np.where(c>e20,1,-1);s+=np.where(e20>e50,1,-1);s+=np.where(e50>e200,.7,-.7);s+=np.where(m>ms,.8,-.8);s+=np.where(mh>mh.shift(2),.4,-.4);s+=np.where(rr>=55,.5,np.where(rr<=45,-.5,0));s+=np.where(c>hi,.8,np.where(c<lo,-.8,0));return pd.Series(np.where(s>=1.5,1,np.where(s<=-1.5,-1,0)),index=df.index),s

def factor_features(index,factors):
    x=pd.DataFrame(index=index)
    for name,df in factors.items():
        s=df.Close.astype(float).reindex(index).ffill(limit=3)
        for n in (1,5,20,60):x[f"{name}_ret{n}"]=s.pct_change(n)
        x[f"{name}_z60"]=(s-s.rolling(60).mean())/s.rolling(60).std().replace(0,np.nan);x[f"{name}_vol20"]=s.pct_change().rolling(20).std()
    return x.replace([np.inf,-np.inf],np.nan)

def build_panel(cfg):
    target=dl(cfg["ticker"]);factors={};errors=[]
    for name,t in cfg["factors"].items():
        try:factors[name]=dl(t)
        except Exception as e:errors.append(f"{name}: {e}")
    X=pd.concat([target_features(target),factor_features(target.index,factors)],axis=1);X=X.dropna(axis=1,thresh=max(500,int(len(X)*.55)))
    d,score=base_direction(target);return target,X,d,score,errors

def models():
    return {"Logistic":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.12,max_iter=2500,class_weight="balanced",random_state=42))]),"Extra Trees":ExtraTreesClassifier(n_estimators=150,max_depth=6,min_samples_leaf=16,max_features="sqrt",class_weight="balanced",random_state=43,n_jobs=-1),"Gradient Boost":HistGradientBoostingClassifier(max_iter=110,max_depth=3,learning_rate=.035,l2_regularization=1.,random_state=44)}

def outer_folds(n,h,min_train=1300,block=126,max_folds=6):
    available=n-min_train-h
    if available<block*2:return []
    k=min(max_folds,available//block);first=n-k*block;out=[]
    for i in range(k):
        ts=first+i*block;te=min(n,ts+block);train_end=ts-h
        if train_end>=min_train:out.append((train_end,ts,te))
    return out

def threshold(p,y,orientation):
    best=None
    for th in (.56,.58,.60,.62,.64,.66,.68,.70):
        confirm=p>=th;invert=p<=1-th;active=confirm|invert
        if active.sum()<20 or active.mean()<.12:continue
        correct=np.where(confirm[active],y[active]==1,y[active]==0);acc=float(np.mean(correct));base=float(np.mean(y[active]==orientation));lift=acc-base;row=(lift+min(float(active.mean()),.45)*.04,th,acc,base,lift,float(active.mean()),int(active.sum()))
        if best is None or row[0]>best[0]:best=row
    return best

def horizon_model(target,X0,direction0,h):
    fwd=target.Close.shift(-h)/target.Close-1;meta=((fwd*direction0)>0).astype(int);full_valid=X0.notna().all(axis=1)&fwd.notna();X=X0.loc[full_valid];d=direction0.loc[full_valid].astype(int);y=meta.loc[full_valid].astype(int)
    if len(X)<1700:raise RuntimeError(f"Insufficient full timeline {h}D: {len(X)}")
    fs=outer_folds(len(X),h);mods=models();decisions=[];foldrows=[];thresholds=[];candidate_test_total=0
    for train_end,test_start,test_end in fs:
        cal_end=train_end;cal_start=cal_end-252;fit_end=cal_start-h
        if fit_end<900:continue
        fitmask=(d.iloc[:fit_end]!=0).values;calmask=(d.iloc[cal_start:cal_end]!=0).values;testmask=(d.iloc[test_start:test_end]!=0).values
        Xfit=X.iloc[:fit_end].iloc[fitmask];yfit=y.iloc[:fit_end].iloc[fitmask];Xcal=X.iloc[cal_start:cal_end].iloc[calmask];ycal=y.iloc[cal_start:cal_end].iloc[calmask];Xtest=X.iloc[test_start:test_end].iloc[testmask];ytest=y.iloc[test_start:test_end].iloc[testmask]
        if len(Xfit)<700 or len(Xcal)<40 or len(Xtest)<10:continue
        cp=[];tp=[]
        for tpl in mods.values():
            try:m=clone(tpl);m.fit(Xfit,yfit);cp.append(m.predict_proba(Xcal)[:,1]);tp.append(m.predict_proba(Xtest)[:,1])
            except Exception:pass
        if len(cp)<3:continue
        pcal=np.mean(np.vstack(cp),axis=0);ptest=np.mean(np.vstack(tp),axis=0);orientation=1 if float(ycal.mean())>=.5 else 0;ch=threshold(pcal,ycal.values,orientation);th=float(ch[1]) if ch else .70;thresholds.append(th);confirm=ptest>=th;invert=ptest<=1-th;active=confirm|invert;yy=ytest.values;candidate_test_total+=len(yy);base_acc=float(np.mean(yy==orientation))
        if active.any():
            correct=np.where(confirm[active],yy[active]==1,yy[active]==0);acc=float(np.mean(correct));lift=acc-base_acc
            for val,conf,inv in zip(yy[active],confirm[active],invert[active]):decisions.append((int(val),1 if conf else 0,orientation))
        else:acc=np.nan;lift=np.nan
        foldrows.append({"threshold":round(th,3),"candidate_setups":len(yy),"coverage":round(float(active.mean()),4),"signals":int(active.sum()),"accuracy":round(acc,4) if np.isfinite(acc) else None,"base_test_accuracy":round(base_acc,4),"lift":round(lift,4) if np.isfinite(lift) else None})
    if len(decisions)<80:raise RuntimeError(f"Too few OOS context decisions {h}D: {len(decisions)}")
    a=np.asarray(decisions,int);acc=float(np.mean(a[:,0]==a[:,1]));base=float(np.mean(a[:,0]==a[:,2]));lift=acc-base;coverage=len(decisions)/max(1,candidate_test_total);valid_lifts=[r["lift"] for r in foldrows if r.get("lift") is not None];positive_share=float(np.mean(np.asarray(valid_lifts)>0)) if valid_lifts else 0
    cand=d!=0;Xtrain=X.loc[cand];ytrain=y.loc[cand];latest=X0.dropna().iloc[[-1]];ps=[]
    for tpl in mods.values():
        try:m=clone(tpl);m.fit(Xtrain,ytrain);ps.append(float(m.predict_proba(latest)[:,1][0]))
        except Exception:pass
    pnow=float(np.mean(ps)) if ps else .5;thnow=float(np.median(thresholds)) if thresholds else .66;current_base=int(direction0.reindex(latest.index).fillna(0).iloc[0]);audit=bool(lift>=.03 and acc>=.56 and len(decisions)>=100 and coverage>=.12 and positive_share>=.5)
    if audit and current_base!=0 and pnow>=thnow:signal="Bullish" if current_base>0 else "Bearish";active_now=True;mode="CONFIRM"
    elif audit and current_base!=0 and pnow<=1-thnow:signal="Bearish" if current_base>0 else "Bullish";active_now=True;mode="INVERT"
    else:signal="NO EDGE / ABSTAIN";active_now=False;mode="ABSTAIN"
    return {"horizon_days":h,"signal":signal,"active_signal":active_now,"mode":mode,"meta_probability_base_correct":round(pnow,4),"decision_threshold":round(thnow,3),"oos_accuracy":round(acc,4),"base_signal_accuracy":round(base,4),"lift_vs_base":round(lift,4),"coverage":round(coverage,4),"oos_signals":len(decisions),"fold_positive_share":round(positive_share,4),"audit_pass":audit,"validation":"nested chronological meta-label walk-forward; purge measured on full trading-day timeline; threshold chosen only on earlier calibration slice","folds":foldrows}

def run_asset(asset,cfg):
    target,X,direction,score,errors=build_panel(cfg);rows=[]
    for h in HORIZONS:
        try:rows.append(horizon_model(target,X,direction,h))
        except Exception as e:rows.append({"horizon_days":h,"signal":"NO EDGE / UNAVAILABLE","active_signal":False,"audit_pass":False,"error":str(e)})
    (DATA/cfg["output"]).write_text(json.dumps({"status":"ok","asset":asset,"symbol":cfg["ticker"],"updated_utc":datetime.now(timezone.utc).isoformat(),"method":"Technical setup plus cross-market meta model. Stage 1 creates direction; stage 2 confirms, rejects or inverts it using market context.","horizons":rows,"factor_errors":errors,"note":"Purge is based on actual trading-day timeline. Challenger is never promoted without OOS lift and fold consistency."},indent=2))
def main():
    for asset,cfg in ASSETS.items():run_asset(asset,cfg)
if __name__=="__main__":main()
