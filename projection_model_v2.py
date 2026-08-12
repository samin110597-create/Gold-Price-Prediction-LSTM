import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR=Path('data'); OUT_DIR.mkdir(parents=True,exist_ok=True)
ASSETS={'gold':{'ticker':'GC=F','name':'COMEX Gold Futures','output':'gold_projections.json'},'silver':{'ticker':'SI=F','name':'COMEX Silver Futures','output':'silver_projections.json'}}

def flatten(df):
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0] for c in df.columns]
    return df

def download(ticker,period,interval):
    periods=['2y','1y'] if interval=='1h' else [period]
    last_error=None
    for p in periods:
        try:
            df=flatten(yf.download(ticker,period=p,interval=interval,auto_adjust=False,progress=False,threads=False))
            if not df.empty and 'Close' in df.columns:
                df=df.dropna(subset=['Open','High','Low','Close']).copy()
                if not df.empty:return df
        except Exception as exc:last_error=exc
    raise RuntimeError(f'No data returned for {ticker} {interval}: {last_error}')

def rsi(s,n=14):
    d=s.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-100/(1+rs)

def features(df,intraday=False):
    c=df.Close.astype(float);h=df.High.astype(float);l=df.Low.astype(float);x=pd.DataFrame(index=df.index);periods=(1,2,4,8,24,48,120) if intraday else (1,2,5,10,20,60,120)
    for n in periods:x[f'ret{n}']=c.pct_change(n)
    for span in ((8,21,55) if intraday else (10,20,50,200)):x[f'ema{span}_gap']=c/c.ewm(span=span,adjust=False).mean()-1
    x['rsi14']=(rsi(c)-50)/50;x['vol20']=c.pct_change().rolling(20).std();x['vol60']=c.pct_change().rolling(60).std();x['z20']=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan);x['range20']=(c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()).replace(0,np.nan)-.5;x['slope10']=(c/c.shift(10)-1)/10;x['slope30']=(c/c.shift(30)-1)/30
    return x.replace([np.inf,-np.inf],np.nan)

def templates():
    return {'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10.0))]),'Random Forest':RandomForestRegressor(n_estimators=160,max_depth=6,min_samples_leaf=12,max_features='sqrt',random_state=42,n_jobs=-1),'Gradient Boost':HistGradientBoostingRegressor(max_iter=130,max_depth=3,learning_rate=.04,l2_regularization=.45,random_state=42)}

def origins(n,steps,intraday,label):
    if intraday:min_train,spacing,maxn=1600,24,28
    elif label=='1 Year':min_train,spacing,maxn=1260,42,16
    else:min_train,spacing,maxn=900,5,36
    return list(range(min_train+steps,n,spacing))[-maxn:]

def project(df,steps,label,intraday=False):
    x=features(df,intraday);fwd=df.Close.shift(-steps)/df.Close-1;valid=x.notna().all(axis=1)&fwd.notna();X,y=x.loc[valid],fwd.loc[valid];mods=templates();records=[]
    if len(X)<1000:raise RuntimeError(f'Insufficient data for {label}')
    for o in origins(len(X),steps,intraday,label):
        tr=o-steps
        if tr<700:continue
        Xtr,ytr=X.iloc[:tr],y.iloc[:tr];preds={};ok=True
        for name,tpl in mods.items():
            try:m=clone(tpl);m.fit(Xtr,ytr);preds[name]=float(m.predict(X.iloc[[o]])[0])
            except:ok=False;break
        if ok:records.append({'actual':float(y.iloc[o]),'baseline':float(ytr.tail(min(756,len(ytr))).median()),'preds':preds})
    if len(records)<12:raise RuntimeError(f'Too few purged walk-forward origins for {label}')
    actual=np.array([r['actual'] for r in records]);basepred=np.array([r['baseline'] for r in records]);base_mae=float(mean_absolute_error(actual,basepred));base_dir=float(np.mean((basepred>=0)==(actual>=0)));metrics={};raw=[]
    for name in mods:
        pred=np.array([r['preds'][name] for r in records]);mae=float(mean_absolute_error(actual,pred));dacc=float(np.mean((pred>=0)==(actual>=0)));skill=1-mae/max(base_mae,1e-9);edge=dacc-base_dir;weight=max(.08,1+np.clip(skill,-.25,.25)*2.2+np.clip(edge,-.15,.15)*2.5);metrics[name]=(mae,dacc,skill,edge,weight);raw.append(weight)
    w=np.array(raw);w/=w.sum();names=list(mods);ens=np.array([sum(r['preds'][n]*ww for n,ww in zip(names,w)) for r in records]);mae=float(mean_absolute_error(actual,ens));dacc=float(np.mean((ens>=0)==(actual>=0)));skill=1-mae/max(base_mae,1e-9);edge=dacc-base_dir;resid=actual-ens
    latest=x.dropna().iloc[[-1]];cur=[];detail=[]
    for name,tpl,ww in zip(names,[mods[n] for n in names],w):
        m=clone(tpl);m.fit(X,y);ret=float(m.predict(latest)[0]);cur.append(ret);mm=metrics[name];detail.append({'name':name,'projected_return_pct':round(ret*100,3),'backtest_mae_pct':round(mm[0]*100,3),'directional_accuracy':round(mm[1],4),'mae_skill_vs_baseline':round(mm[2],4),'weight':round(float(ww),4)})
    pret=float(np.dot(np.array(cur),w));last=float(df.Close.iloc[-1]);price=last*(1+pret);pup=float(np.clip(np.mean((pret+resid)>0),.15,.85));coverage=.50 if label=='1 Year' else .60;q=float(np.quantile(np.abs(resid),coverage));low=last*(1+pret-q);high=last*(1+pret+q);directions=[r>=0 for r in cur];maj=sum(directions)>=len(directions)/2;agree=float(sum(d==maj for d in directions)/len(directions))
    conf='High' if edge>=.06 and skill>=.10 and agree>=.67 and len(records)>=20 else ('Moderate' if edge>=.03 and skill>=.04 and agree>=.67 and len(records)>=16 else 'Low')
    if label=='1 Year':conf='Moderate' if (edge>=.03 and skill>=.10 and len(records)>=20) else 'Low'
    passed=bool(edge>0 and skill>0 and len(records)>=12)
    return {'horizon':label,'steps':steps,'predicted_price':round(price,2),'model_price':round(price,2),'raw_ml_price':round(price,2),'projected_return_pct':round(pret*100,3),'probability_up':round(pup,4),'probability_down':round(1-pup,4),'confidence':conf,'confidence_score':70 if conf=='High' else (58 if conf=='Moderate' else 38),'forecast_status':'Validated' if passed else 'Estimate only','tight_model_zone':[round(float(low),2),round(float(high),2)],'focus_zone':[round(float(low),2),round(float(high),2)],'risk_zone':[round(float(low),2),round(float(high),2)],'zone_target_coverage':coverage,'backtest_directional_accuracy':round(dacc,4),'baseline_directional_accuracy':round(base_dir,4),'directional_edge':round(edge,4),'backtest_mae_pct':round(mae*100,3),'baseline_mae_pct':round(base_mae*100,3),'mae_skill_vs_baseline':round(skill,4),'model_agreement':round(agree,4),'walkforward_origins':len(records),'validation':'purged expanding-origin walk-forward','model_version':'pre-calibration champion','models':detail}

def run_asset(key,cfg):
    d=download(cfg['ticker'],'10y','1d');h=download(cfg['ticker'],'2y','1h');ps=[project(h,4,'4 Hours',True),project(d,1,'1 Day'),project(d,5,'1 Week'),project(d,252,'1 Year')];payload={'status':'ok','asset':key,'symbol':cfg['ticker'],'instrument':cfg['name'],'updated_utc':datetime.now(timezone.utc).isoformat(),'latest_price':round(float(h.Close.iloc[-1]),2),'projections':ps,'model_version':'pre-calibration champion with guarded long-horizon confidence','note':'Restored stronger pre-calibration champion. Purged walk-forward validation; 1-year confidence is deliberately capped because samples are limited and regime-dependent.'};(OUT_DIR/cfg['output']).write_text(json.dumps(payload,indent=2),encoding='utf-8')

def main():
    for k,c in ASSETS.items():run_asset(k,c)
if __name__=='__main__':main()
