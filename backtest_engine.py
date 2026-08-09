# workflow trigger: audited V2 validation
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, yfinance as yf
DATA=Path('data');ASSETS={'gold':'GC=F','silver':'SI=F'}
def flatten(df):
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0] for c in df.columns]
    return df
def download(ticker):
    df=flatten(yf.download(ticker,period='5y',interval='1d',auto_adjust=False,progress=False,threads=False))
    if df.empty:raise RuntimeError(f'No data for {ticker}')
    return df.dropna(subset=['Open','High','Low','Close']).copy()
def rsi(s,n=14):
    d=s.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-100/(1+rs)
def atr(df,n=14):
    pc=df.Close.shift(1);tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(df,n=14):
    up=df.High.diff();down=-df.Low.diff();pdm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index);mdm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index);a=atr(df,n);pdi=100*pdm.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan);mdi=100*mdm.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan);dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan);return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi
def score_series(df):
    c,h,l=df.Close.astype(float),df.High.astype(float),df.Low.astype(float);e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();e200=c.ewm(span=200,adjust=False).mean();macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();ms=macd.ewm(span=9,adjust=False).mean();mh=macd-ms;rr=rsi(c);lo=l.rolling(14).min();hi=h.rolling(14).max();sk=100*(c-lo)/(hi-lo).replace(0,np.nan);sd=sk.rolling(3).mean();mid=c.rolling(20).mean();st=c.rolling(20).std();bu=mid+2*st;bl=mid-2*st;bp=(c-bl)/(bu-bl).replace(0,np.nan);ax,pdi,mdi=adx(df);s=pd.Series(0.0,index=df.index);s+=np.where(c>e20,1,-1);s+=np.where(e20>e50,1,-1);s+=np.where(e50>e200,1,-1);s+=np.where(macd>ms,1,-1);s+=np.where(mh>mh.shift(1),.5,-.5);s+=np.select([(rr>=55)&(rr<=72),rr>=75,rr<=25,rr<45],[1,-.35,.35,-1],default=0);s+=np.select([(ax>=20)&(pdi>mdi),(ax>=20)&(mdi>pdi)],[1,-1],default=0);s+=np.select([(sk>sd)&(sk<80),(sk<sd)&(sk>20)],[.5,-.5],default=0);s+=np.select([(bp>=.55)&(bp<=.9),bp>1,(bp>=.1)&(bp<.45),bp<0],[.5,-.25,-.5,.25],default=0);return pd.Series(np.clip(50+s/7*50,0,100),index=df.index)
def technical_bt(df):
    score=score_series(df);c=df.Close.astype(float);out=[];start=max(250,len(df)-756)
    for h in (1,5,20):
        fwd=c.shift(-h)/c-1;s=score.iloc[start:];fr=fwd.iloc[start:];v=s.notna()&fr.notna();s=s[v];fr=fr[v];sig=np.where(s>=60,1,np.where(s<=40,-1,0));active=sig!=0
        if not active.any():out.append({'horizon_days':h,'signals':0,'coverage':0,'hit_rate':None,'edge':None});continue
        signed=fr.values[active]*sig[active];hit=float(np.mean(signed>0));pos=float(np.mean(fr.values[active]>0));base=max(pos,1-pos);out.append({'horizon_days':h,'signals':int(active.sum()),'coverage':round(float(np.mean(active)),4),'hit_rate':round(hit,4),'baseline_direction_rate':round(float(base),4),'edge':round(hit-base,4),'avg_signed_return_pct':round(float(np.mean(signed))*100,3)})
    return out
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def audit(asset,ticker):
    f=load(DATA/('live_forecast.json' if asset=='gold' else 'silver_forecast.json'));p=load(DATA/f'{asset}_projections.json');t=technical_bt(download(ticker));fr=[];pr=[]
    for x in f['forecasts']:
        edge=float(x.get('backtest_edge',0));bs=float(x.get('brier_score',1));fr.append({'horizon_days':x['horizon_days'],'accuracy':x.get('backtest_accuracy'),'baseline':x.get('naive_baseline'),'edge':edge,'brier_score':bs,'oos_observations':x.get('test_observations'),'pass':edge>0 and bs<.25})
    for x in p['projections']:
        edge=float(x.get('directional_edge',0));skill=float(x.get('mae_skill_vs_baseline',-9));z=x.get('tight_model_zone',[None,None]);mp=x.get('model_price');zone=bool(z[0] is not None and z[0]<=mp<=z[1]);pr.append({'horizon':x['horizon'],'directional_accuracy':x.get('backtest_directional_accuracy'),'baseline_directional_accuracy':x.get('baseline_directional_accuracy'),'directional_edge':edge,'mae_pct':x.get('backtest_mae_pct'),'baseline_mae_pct':x.get('baseline_mae_pct'),'mae_skill_vs_baseline':skill,'walkforward_origins':x.get('walkforward_origins'),'zone_contains_model_price':zone,'pass':edge>0 and skill>0 and zone})
    tests=[x['pass'] for x in fr]+[x['pass'] for x in pr]+[bool(x.get('edge') is not None and x.get('edge',0)>0) for x in t];rate=float(np.mean(tests));grade='PASS' if rate>=.70 else ('MIXED' if rate>=.40 else 'FAIL');return {'audit_grade':grade,'component_pass_rate':round(rate,3),'probability_forecasts':fr,'price_projections':pr,'technical_signals':t,'interpretation':'PASS means most components beat simple out-of-sample baselines; MIXED means use only components with positive edge; FAIL means do not treat the system as predictive.'}
def main():
    payload={'status':'ok','updated_utc':datetime.now(timezone.utc).isoformat(),'validation_method':'Purged expanding-window/origin walk-forward for trained models; last ~3 years deterministic historical test for technical confluence. Future labels are excluded from each training window.','assets':{k:audit(k,t) for k,t in ASSETS.items()}};(DATA/'model_backtest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print({k:v['audit_grade'] for k,v in payload['assets'].items()})
if __name__=='__main__':main()
