import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data"); DATA.mkdir(exist_ok=True)
ASSETS = {
    "gold": {"ticker":"GC=F","projection_file":"gold_projections.json","output":"gold_macro_1y.json","commodity":"GOLD","cross":{"silver":"SI=F","copper":"HG=F","oil":"CL=F","sp500":"SPY","long_bonds":"TLT"}},
    "silver": {"ticker":"SI=F","projection_file":"silver_projections.json","output":"silver_macro_1y.json","commodity":"SILVER","cross":{"gold":"GC=F","copper":"HG=F","oil":"CL=F","sp500":"SPY","long_bonds":"TLT"}},
}
FRED={"real_yield_10y":"DFII10","breakeven_10y":"T10YIE","broad_usd":"DTWEXBGS","vix":"VIXCLS"}

def flatten(df):
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    return df

def month_index(idx):
    idx=pd.to_datetime(idx)
    if getattr(idx,"tz",None) is not None: idx=idx.tz_localize(None)
    return idx.to_period("M").to_timestamp("M")

def market_monthly(ticker):
    df=flatten(yf.download(ticker,start="2003-01-01",interval="1d",auto_adjust=False,progress=False,threads=False))
    if df.empty or "Close" not in df.columns: raise RuntimeError(f"No market data for {ticker}")
    s=df["Close"].astype(float).dropna(); s.index=pd.to_datetime(s.index)
    if getattr(s.index,"tz",None) is not None: s.index=s.index.tz_localize(None)
    s=s.resample("ME").last().dropna(); s.index=month_index(s.index)
    return s[~s.index.duplicated(keep="last")]

def fred_series(sid):
    df=pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    d,v=df.columns[:2]; df[d]=pd.to_datetime(df[d]); df[v]=pd.to_numeric(df[v],errors="coerce")
    s=df.dropna().set_index(d)[v].astype(float); s.index=month_index(s.index)
    return s.groupby(level=0).last().sort_index().shift(1)

def cftc_monthly(commodity):
    fields="report_date_as_yyyy_mm_dd,commodity_name,open_interest_all,m_money_positions_long_all,m_money_positions_short_all,prod_merc_positions_long,prod_merc_positions_short"
    params={"$select":fields,"$where":f"commodity_name='{commodity}'","$order":"report_date_as_yyyy_mm_dd ASC","$limit":5000}
    url="https://publicreporting.cftc.gov/resource/72hh-3qpy.csv?"+urlencode(params)
    df=pd.read_csv(url)
    if df.empty: raise RuntimeError(f"No CFTC data for {commodity}")
    d="report_date_as_yyyy_mm_dd"; df[d]=pd.to_datetime(df[d])
    for c in df.columns:
        if c not in (d,"commodity_name"): df[c]=pd.to_numeric(df[c],errors="coerce")
    oi=df["open_interest_all"].replace(0,np.nan)
    df["managed_net"]=(df["m_money_positions_long_all"]-df["m_money_positions_short_all"])/oi
    df["producer_net"]=(df["prod_merc_positions_long"]-df["prod_merc_positions_short"])/oi
    out=df.set_index(d)[["managed_net","producer_net"]].resample("ME").last(); out.index=month_index(out.index)
    return out.shift(1)

def build_panel(cfg):
    target=market_monthly(cfg["ticker"]); panel=pd.DataFrame({"target":target}); status={"fred":{},"cftc":False}
    for name,ticker in cfg["cross"].items():
        try: panel[name]=market_monthly(ticker).reindex(panel.index).ffill(limit=2)
        except Exception: pass
    for name,sid in FRED.items():
        try: panel[name]=fred_series(sid).reindex(panel.index).ffill(limit=2); status["fred"][name]=True
        except Exception: status["fred"][name]=False
    try:
        cot=cftc_monthly(cfg["commodity"]).reindex(panel.index)
        for c in cot.columns: panel[f"cftc_{c}"]=cot[c]
        status["cftc"]=True
    except Exception as exc: status["cftc_error"]=str(exc)
    return panel.sort_index(),status

def features(panel):
    p=panel["target"].astype(float); x=pd.DataFrame(index=panel.index)
    for n in (1,3,6,12): x[f"target_ret{n}"]=p.pct_change(n)
    x["momentum_12_1"]=p.shift(1)/p.shift(12)-1; x["vol6"]=p.pct_change().rolling(6).std(); x["vol12"]=p.pct_change().rolling(12).std()
    x["drawdown12"]=p/p.rolling(12).max()-1; x["z12"]=(p-p.rolling(12).mean())/p.rolling(12).std().replace(0,np.nan); x["z24"]=(p-p.rolling(24).mean())/p.rolling(24).std().replace(0,np.nan)
    for col in panel.columns:
        if col=="target": continue
        s=panel[col].astype(float)
        if col.startswith("cftc_"):
            avail=s.notna().astype(float); z=(s-s.rolling(36,min_periods=12).mean())/s.rolling(36,min_periods=12).std().replace(0,np.nan)
            x[f"{col}_level"]=s.fillna(0); x[f"{col}_z36"]=z.fillna(0); x[f"{col}_chg3"]=s.diff(3).fillna(0); x[f"{col}_available"]=avail
        elif col in ("real_yield_10y","breakeven_10y","vix"):
            x[f"{col}_level"]=s; x[f"{col}_chg3"]=s.diff(3); x[f"{col}_chg12"]=s.diff(12); x[f"{col}_z36"]=(s-s.rolling(36).mean())/s.rolling(36).std().replace(0,np.nan)
        else:
            for n in (1,3,6,12): x[f"{col}_ret{n}"]=s.pct_change(n)
            x[f"{col}_z36"]=(s-s.rolling(36).mean())/s.rolling(36).std().replace(0,np.nan)
    if "real_yield_10y" in panel and "breakeven_10y" in panel: x["real_minus_breakeven"]=panel["real_yield_10y"]-panel["breakeven_10y"]
    if "gold" in panel: x["silver_gold_ratio_mom6"]=(panel["target"]/panel["gold"]).pct_change(6)
    if "silver" in panel: x["gold_silver_ratio_mom6"]=(panel["target"]/panel["silver"]).pct_change(6)
    x=x.replace([np.inf,-np.inf],np.nan); x=x.dropna(axis=1,thresh=max(100,int(len(x)*.60)))
    return x

def templates():
    return {"Ridge":Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=25.0))]),"Elastic Net":Pipeline([("scale",StandardScaler()),("model",ElasticNet(alpha=.015,l1_ratio=.18,max_iter=5000,random_state=42))]),"Random Forest":RandomForestRegressor(n_estimators=180,max_depth=4,min_samples_leaf=7,max_features=.65,random_state=42,n_jobs=-1),"Gradient Boost":HistGradientBoostingRegressor(max_iter=110,max_depth=2,learning_rate=.035,l2_regularization=1.2,random_state=43)}

def dacc(pred,actual): return float(np.mean((np.asarray(pred)>=0)==(np.asarray(actual)>=0)))
def metrics(pred,actual,base):
    mae=float(mean_absolute_error(actual,pred)); bmae=float(mean_absolute_error(actual,base)); da=dacc(pred,actual); bda=dacc(base,actual)
    return {"mae":mae,"base_mae":bmae,"skill":1-mae/max(bmae,1e-9),"dacc":da,"base_dacc":bda,"edge":da-bda}

def walkforward(panel):
    X0=features(panel); fwd=panel["target"].shift(-12)/panel["target"]-1; valid=X0.notna().all(axis=1)&fwd.notna(); X,y=X0.loc[valid],fwd.loc[valid]
    if len(X)<145: raise RuntimeError(f"Insufficient usable monthly history: {len(X)}")
    mods=templates(); origins=list(range(108,len(X),3))[-48:]; rec=[]
    for origin in origins:
        train_end=origin-12
        if train_end<96: continue
        preds={}
        for name,tpl in mods.items():
            try: m=clone(tpl);m.fit(X.iloc[:train_end],y.iloc[:train_end]);preds[name]=float(m.predict(X.iloc[[origin]])[0])
            except Exception: pass
        if len(preds)>=3: rec.append({"date":str(X.index[origin].date()),"actual":float(y.iloc[origin]),"baseline":float(y.iloc[:train_end].tail(min(120,train_end)).median()),"preds":preds})
    if len(rec)<20: raise RuntimeError(f"Too few purged macro origins: {len(rec)}")
    actual=np.array([r["actual"] for r in rec]); base=np.array([r["baseline"] for r in rec]); names=[n for n in mods if all(n in r["preds"] for r in rec)]; raw=[]; mm={}
    for name in names:
        pred=np.array([r["preds"][name] for r in rec]); m=metrics(pred,actual,base); w=max(.05,1+np.clip(m["skill"],-.3,.3)*2.3+np.clip(m["edge"],-.2,.2)*1.8);mm[name]=dict(m,weight=w);raw.append(w)
    w=np.asarray(raw);w/=w.sum(); ens=np.array([sum(r["preds"][n]*ww for n,ww in zip(names,w)) for r in rec]); em=metrics(ens,actual,base); resid=actual-ens
    idx=np.arange(0,len(rec),4); non=metrics(ens[idx],actual[idx],base[idx]) if len(idx)>=6 else {"skill":-9,"edge":-9,"dacc":0,"base_dacc":0,"mae":9,"base_mae":9}
    latest=X0.dropna().iloc[[-1]]; current=[]; details=[]
    for name,ww in zip(names,w):
        m=clone(mods[name]);m.fit(X,y);ret=float(m.predict(latest)[0]);current.append(ret);q=mm[name];details.append({"name":name,"projected_return_pct":round(ret*100,2),"weight":round(float(ww),4),"oos_mae_skill":round(q["skill"],4),"oos_direction_accuracy":round(q["dacc"],4)})
    rawret=float(np.dot(current,w)); anchor=float(y.tail(min(120,len(y))).median()); trust=float(np.clip(.30+max(0,em["skill"])*1.15+max(0,em["edge"])*.75,.20,.70)); ret=trust*rawret+(1-trust)*anchor
    latest_price=float(panel["target"].dropna().iloc[-1]); price=latest_price*(1+ret); q50=float(np.quantile(np.abs(resid),.5));q80=float(np.quantile(np.abs(resid),.8));focus=[latest_price*(1+ret-q50),latest_price*(1+ret+q50)];risk=[latest_price*(1+ret-q80),latest_price*(1+ret+q80)]
    validated=bool(em["skill"]>.03 and em["edge"]>.01 and len(rec)>=20 and non["skill"]>0 and non["edge"]>=-.05); confidence="Moderate" if validated and em["skill"]>=.10 and em["edge"]>=.04 and len(rec)>=28 and non["skill"]>.02 else "Low"
    return {"horizon":"1 Year","model_price":round(price,2),"predicted_price":round(price,2),"projected_return_pct":round(ret*100,2),"raw_macro_return_pct":round(rawret*100,2),"historical_anchor_return_pct":round(anchor*100,2),"macro_model_trust":round(trust,3),"probability_up":round(float(np.mean((ret+resid)>0)),4),"confidence":confidence,"forecast_status":"Validated" if validated else "Estimate only","tight_model_zone":[round(focus[0],2),round(focus[1],2)],"focus_zone":[round(focus[0],2),round(focus[1],2)],"risk_zone":[round(risk[0],2),round(risk[1],2)],"backtest_directional_accuracy":round(em["dacc"],4),"baseline_directional_accuracy":round(em["base_dacc"],4),"directional_edge":round(em["edge"],4),"backtest_mae_pct":round(em["mae"]*100,2),"baseline_mae_pct":round(em["base_mae"]*100,2),"mae_skill_vs_baseline":round(em["skill"],4),"walkforward_origins":len(rec),"nonoverlap_origins":len(idx),"nonoverlap_mae_skill":round(non["skill"],4),"nonoverlap_directional_edge":round(non["edge"],4),"validation":"monthly purged expanding-origin walk-forward; 12-month label purge; FRED and CFTC inputs conservatively lagged; non-overlapping 12-month sanity subset","models":details}

def score(row): return 2*max(0,float(row.get("mae_skill_vs_baseline",0) or 0))+1.3*max(0,float(row.get("directional_edge",0) or 0))+min(.35,float(row.get("walkforward_origins",0) or 0)/120)
def maybe_promote(asset,cfg,row):
    path=DATA/cfg["projection_file"];payload=json.loads(path.read_text());ps=payload.get("projections",[]);i=next((i for i,x in enumerate(ps) if x.get("horizon")=="1 Year"),None)
    if i is None:return False,None
    old=ps[i];os,ns=score(old),score(row); skill=float(row.get("mae_skill_vs_baseline",0));edge=float(row.get("directional_edge",0));n=int(row.get("walkforward_origins",0));non_skill=float(row.get("nonoverlap_mae_skill",-9));non_edge=float(row.get("nonoverlap_directional_edge",-9));old_skill=float(old.get("mae_skill_vs_baseline",0) or 0);old_edge=float(old.get("directional_edge",0) or 0)
    promote=bool(n>=20 and skill>.03 and edge>=.01 and non_skill>0 and non_edge>=-.05 and ns>os+.04 and skill>=old_skill-.01 and edge>=old_edge-.02)
    if promote:
        new=dict(row);new["selected_model"]="Dedicated macro + positioning 1Y champion";new["champion_reason"]="Beat incumbent on pre-defined OOS score plus non-overlapping sanity check.";ps[i]=new;payload["projections"]=ps;payload["note"]=(payload.get("note","")+" 1Y uses champion-challenger selection and a non-overlapping sanity check.").strip();path.write_text(json.dumps(payload,indent=2))
    return promote,{"incumbent_score":round(os,4),"challenger_score":round(ns,4),"incumbent_skill":round(old_skill,4),"challenger_skill":round(skill,4),"incumbent_edge":round(old_edge,4),"challenger_edge":round(edge,4),"challenger_nonoverlap_skill":round(non_skill,4),"challenger_nonoverlap_edge":round(non_edge,4)}

def run_asset(asset,cfg):
    panel,status=build_panel(cfg);row=walkforward(panel);promoted,comparison=maybe_promote(asset,cfg,row);out={"status":"ok","asset":asset,"updated_utc":datetime.now(timezone.utc).isoformat(),"data_inputs":status,"macro_forecast":row,"selected_as_champion":promoted,"comparison":comparison,"method_note":"Dedicated 1Y challenger uses daily-resampled monthly market history, lagged FRED macro inputs, optional lagged CFTC positioning, 12-month purge and non-overlap sanity check."};(DATA/cfg["output"]).write_text(json.dumps(out,indent=2))
def main():
    for asset,cfg in ASSETS.items():run_asset(asset,cfg)
if __name__=="__main__":main()
