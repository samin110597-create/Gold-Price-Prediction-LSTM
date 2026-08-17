import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

DATA=Path('data')
ASSETS={'gold':('GC=F',DATA/'gold_technicals.json'),'silver':('SI=F',DATA/'silver_technicals.json')}


def flatten(df):
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    return df


def load(ticker):
    df=flatten(yf.download(ticker,period='10y',interval='1d',auto_adjust=False,progress=False,threads=False))
    if df.empty: raise RuntimeError(f'No history for {ticker}')
    idx=pd.to_datetime(df.index)
    if getattr(idx,'tz',None) is not None: idx=idx.tz_localize(None)
    df=df.copy(); df.index=idx
    return df.dropna(subset=['High','Low','Close']).sort_index()


def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)


def build_events(df):
    c=df['Close'].astype(float); h=df['High'].astype(float); l=df['Low'].astype(float); vol=df.get('Volume',pd.Series(0,index=df.index)).fillna(0).astype(float)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean(); macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=macd.ewm(span=9,adjust=False).mean(); rr=rsi(c)
    phi=h.shift(1).rolling(20).max(); plo=l.shift(1).rolling(20).min(); vm=vol.rolling(20).mean()
    out={}
    def put(name,mask,direction):
        s=pd.Series(0,index=df.index,dtype=int); s.loc[mask.fillna(False)]=direction; out[name]=s
    put('EMA20/50 bullish cross',(e20>e50)&(e20.shift(1)<=e50.shift(1)),1)
    put('EMA20/50 bearish cross',(e20<e50)&(e20.shift(1)>=e50.shift(1)),-1)
    put('Golden cross',(e50>e200)&(e50.shift(1)<=e200.shift(1)),1)
    put('Death cross',(e50<e200)&(e50.shift(1)>=e200.shift(1)),-1)
    put('MACD bullish cross',(macd>ms)&(macd.shift(1)<=ms.shift(1)),1)
    put('MACD bearish cross',(macd<ms)&(macd.shift(1)>=ms.shift(1)),-1)
    put('20D breakout',c>phi,1)
    put('20D breakdown',c<plo,-1)
    put('RSI exited overbought',(rr<70)&(rr.shift(1)>=70),-1)
    put('RSI exited oversold',(rr>30)&(rr.shift(1)<=30),1)
    # Direction-neutral volume expansion is audited conditionally by the candle direction.
    spike=(vol>=1.8*vm)&(vm>0); s=pd.Series(0,index=df.index,dtype=int); s.loc[spike & (c>=c.shift(1))]=1; s.loc[spike & (c<c.shift(1))]=-1; out['Volume expansion']=s
    return out


def audit(df, events):
    c=df['Close'].astype(float); horizons=[1,5,20]; rows=[]
    for name,sig in events.items():
        for h in horizons:
            fwd=c.shift(-h)/c-1; mask=(sig!=0)&fwd.notna(); n=int(mask.sum())
            if n<8:
                rows.append({'signal':name,'horizon_days':h,'events':n,'hit_rate':None,'baseline':None,'edge':None,'avg_signed_return_pct':None,'status':'INSUFFICIENT'})
                continue
            y=(fwd[mask]>0).astype(int); direction=sig[mask]; hit=((np.sign(fwd[mask])==direction)).mean(); baseline=max(float(y.mean()),1-float(y.mean())); edge=float(hit)-baseline; signed=(fwd[mask]*direction).mean()*100
            # Descriptive fixed-rule grade: no parameter fitting, but still not equivalent to champion OOS validation.
            status='POSITIVE' if n>=30 and edge>=.02 and signed>0 else ('MIXED' if signed>0 else 'NEGATIVE')
            rows.append({'signal':name,'horizon_days':h,'events':n,'hit_rate':round(float(hit),4),'baseline':round(float(baseline),4),'edge':round(edge,4),'avg_signed_return_pct':round(float(signed),3),'status':status})
    return rows


def enrich(asset,ticker,path):
    payload=json.loads(path.read_text()); df=load(ticker); rows=audit(df,build_events(df)); payload['chart_signal_audit']={'method':'Fixed-rule historical event audit. Rules are pre-specified; results are descriptive and do not override the champion walk-forward model audit.','updated_utc':datetime.now(timezone.utc).isoformat(),'rows':rows}
    by={(r['signal'],r['horizon_days']):r for r in rows}
    for e in payload.get('chart_signal_events',[]):
        e['audit_1d']=by.get((e.get('name'),1)); e['audit_5d']=by.get((e.get('name'),5)); e['audit_20d']=by.get((e.get('name'),20))
    path.write_text(json.dumps(payload,indent=2)); print(asset,'audited',len(rows),'signal/horizon rows')


if __name__=='__main__':
    for asset,(ticker,path) in ASSETS.items():
        try: enrich(asset,ticker,path)
        except Exception as e: print(asset,'signal audit failed:',e)
