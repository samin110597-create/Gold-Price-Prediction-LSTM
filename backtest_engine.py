# audited champion-challenger validation
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

DATA=Path("data"); ASSETS={"gold":"GC=F","silver":"SI=F"}
def flatten(df):
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0] for c in df.columns]
    return df
def download(t):
    df=flatten(yf.download(t,period="5y",interval="1d",auto_adjust=False,progress=False,threads=False))
    if df.empty:raise RuntimeError(f"No data for {t}")
    return df.dropna(subset=["Open","High","Low","Close"]).copy()
def rsi(s,n=14):
    d=s.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-100/(1+rs)
def atr(df,n=14):
    pc=df.Close.shift(1);tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(df,n=14):
    up=df.High.diff();down=-df.Low.diff();pdm=pd.Series(np.where((up>down)&(up>0),up,0.),index=df.index);mdm=pd.Series(np.where((down>up)&(down>0),down,0.),index=df.index);a=atr(df,n);pdi=100*pdm.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan);mdi=100*mdm.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan);dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan);return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi
def score_series(df):
    c,h,l=df.Close.astype(float),df.High.astype(float),df.Low.astype(float);e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();e200=c.ewm(span=200,adjust=False).mean();macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();ms=macd.ewm(span=9,adjust=False).mean();mh=macd-ms;rr=rsi(c);lo=l.rolling(14).min();hi=h.rolling(14).max();sk=100*(c-lo)/(hi-lo).replace(0,np.nan);sd=sk.rolling(3).mean();mid=c.rolling(20).mean();st=c.rolling(20).std();bu=mid+2*st;bl=mid-2*st;bp=(c-bl)/(bu-bl).replace(0,np.nan);ax,pdi,mdi=adx(df);s=pd.Series(0.,index=df.index);s+=np.where(c>e20,1,-1);s+=np.where(e20>e50,1,-1);s+=np.where(e50>e200,1,-1);s+=np.where(macd>ms,1,-1);s+=np.where(mh>mh.shift(1),.5,-.5);s+=np.select([(rr>=55)&(rr<=72),rr>=75,rr<=25,rr<45],[1,-.35,.35,-1],default=0);s+=np.select([(ax>=20)&(pdi>mdi),(ax>=20)&(mdi>pdi)],[1,-1],default=0);s+=np.select([(sk>sd)&(sk<80),(sk<sd)&(sk>20)],[.5,-.5],default=0);s+=np.select([(bp>=.55)&(bp<=.9),bp>1,(bp>=.1)&(bp<.45),bp<0],[.5,-.25,-.5,.25],default=0);return pd.Series(np.clip(50+s/7*50,0,100),index=df.index)
def technical_bt(df):
    score=score_series(df);c=df.Close.astype(float);out=[];start=max(250,len(df)-756)
    for h in (1,5,20):
        fwd=c.shift(-h)/c-1;s=score.iloc[start:];fr=fwd.iloc[start:];v=s.notna()&fr.notna();s,fr=s[v],fr[v];sig=np.where(s>=60,1,np.where(s<=40,-1,0));active=sig!=0
        if not active.any():out.append({"horizon_days":h,"signals":0,"coverage":0,"hit_rate":None,"edge":None});continue
        signed=fr.values[active]*sig[active];hit=float(np.mean(signed>0));pos=float(np.mean(fr.values[active]>0));base=max(pos,1-pos);out.append({"horizon_days":h,"signals":int(active.sum()),"coverage":round(float(np.mean(active)),4),"hit_rate":round(hit,4),"baseline_direction_rate":round(float(base),4),"edge":round(hit-base,4),"avg_signed_return_pct":round(float(np.mean(signed))*100,3)})
    return out
def load(p):return json.loads(p.read_text())
def maybe(p):
    try:return load(p) if p.exists() else None
    except Exception:return None

def technical_champions(old_rows,selective_payload,context_payload):
    sel={int(x["horizon_days"]):x for x in (selective_payload or {}).get("horizons",[]) if "horizon_days" in x};ctx={int(x["horizon_days"]):x for x in (context_payload or {}).get("horizons",[]) if "horizon_days" in x};out=[]
    for old in old_rows:
        h=int(old["horizon_days"]);candidates=[];oe=float(old.get("edge") or -9);candidates.append({"horizon_days":h,"source":"Deterministic confluence","accuracy":old.get("hit_rate"),"baseline":old.get("baseline_direction_rate"),"edge":oe,"coverage":old.get("coverage"),"signals":old.get("signals"),"current_signal":None,"pass":bool(oe>0)})
        s=sel.get(h)
        if s and s.get("selective_edge") is not None:
            e=float(s.get("selective_edge") or 0);candidates.append({"horizon_days":h,"source":"Selective nested technical model","accuracy":s.get("selective_accuracy"),"baseline":s.get("baseline_direction_rate"),"edge":e,"coverage":s.get("coverage"),"signals":s.get("oos_signals"),"current_signal":s.get("signal"),"pass":bool(s.get("audit_pass") and e>.02)})
        m=ctx.get(h)
        if m and m.get("lift_vs_base") is not None:
            e=float(m.get("lift_vs_base") or 0);candidates.append({"horizon_days":h,"source":"Technical + cross-market meta model","accuracy":m.get("oos_accuracy"),"baseline":m.get("base_signal_accuracy"),"edge":e,"coverage":m.get("coverage"),"signals":m.get("oos_signals"),"current_signal":m.get("signal"),"mode":m.get("mode"),"pass":bool(m.get("audit_pass") and e>=.03)})
        # Passing challengers outrank failing models; otherwise retain the best observed edge transparently.
        candidates.sort(key=lambda x:(1 if x.get("pass") else 0,float(x.get("edge") or -9),int(x.get("signals") or 0)),reverse=True);winner=candidates[0];winner["candidates"]=candidates;out.append(winner)
    return out

def audit(asset,ticker):
    forecast=load(DATA/("live_forecast.json" if asset=="gold" else "silver_forecast.json"));proj=load(DATA/f"{asset}_projections.json");old=technical_bt(download(ticker));sel=maybe(DATA/f"{asset}_technical_model.json");ctx=maybe(DATA/f"{asset}_technical_context.json");macro=maybe(DATA/f"{asset}_macro_1y.json")
    fr=[]
    for x in forecast.get("forecasts",[]):
        e=float(x.get("backtest_edge",0) or 0);b=float(x.get("brier_score",1) or 1);fr.append({"horizon_days":x.get("horizon_days"),"accuracy":x.get("backtest_accuracy"),"baseline":x.get("naive_baseline"),"edge":e,"brier_score":b,"oos_observations":x.get("test_observations"),"pass":bool(e>0 and b<.25)})
    pr=[]
    for x in proj.get("projections",[]):
        e=float(x.get("directional_edge",0) or 0);sk=float(x.get("mae_skill_vs_baseline",-9) or -9);z=x.get("tight_model_zone",[None,None]);mp=x.get("model_price");inside=bool(z and z[0] is not None and mp is not None and z[0]<=mp<=z[1]);pr.append({"horizon":x.get("horizon"),"selected_model":x.get("selected_model","Incumbent projection model"),"directional_accuracy":x.get("backtest_directional_accuracy"),"baseline_directional_accuracy":x.get("baseline_directional_accuracy"),"directional_edge":e,"mae_pct":x.get("backtest_mae_pct"),"baseline_mae_pct":x.get("baseline_mae_pct"),"mae_skill_vs_baseline":sk,"walkforward_origins":x.get("walkforward_origins"),"zone_contains_model_price":inside,"pass":bool(e>0 and sk>0 and inside)})
    champs=technical_champions(old,sel,ctx);tests=[x["pass"] for x in fr]+[x["pass"] for x in pr]+[x["pass"] for x in champs];rate=float(np.mean(tests));grade="PASS" if rate>=.70 else ("MIXED" if rate>=.40 else "FAIL")
    return {"audit_grade":grade,"component_pass_rate":round(rate,3),"probability_forecasts":fr,"price_projections":pr,"technical_signals_legacy":old,"technical_selective":(sel or {}).get("horizons",[]),"technical_context":(ctx or {}).get("horizons",[]),"technical_champions":champs,"macro_1y_challenger":macro,"interpretation":"PASS means most champion components beat simple OOS baselines. New technical and 1Y challengers can replace incumbents only when fixed walk-forward safeguards are met. FAIL means do not treat the whole system as predictive."}
def main():
    payload={"status":"ok","updated_utc":datetime.now(timezone.utc).isoformat(),"validation_method":"Purged walk-forward validation. Technical meta thresholds are chosen only on earlier calibration windows. 1Y macro model uses a 12-month label purge and lagged public macro/positioning inputs. Champion-challenger selection prevents a newer model from replacing a stronger incumbent without OOS evidence.","assets":{k:audit(k,t) for k,t in ASSETS.items()}};(DATA/"model_backtest.json").write_text(json.dumps(payload,indent=2));print({k:v["audit_grade"] for k,v in payload["assets"].items()})
if __name__=="__main__":main()
