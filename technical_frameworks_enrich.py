import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

DATA=Path('data')
ASSETS={'gold':('GC=F','gold_technicals.json'),'silver':('SI=F','silver_technicals.json')}

def flatten(df):
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0] for c in df.columns]
    return df

def download(ticker):
    df=flatten(yf.download(ticker,period='5y',interval='1d',auto_adjust=False,progress=False,threads=False))
    if df.empty:raise RuntimeError(f'No data for {ticker}')
    return df.dropna(subset=['Open','High','Low','Close']).copy()

def atr(df,n=14):
    pc=df.Close.shift(1);tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def pivots(df,order=3):
    out=[];h=df.High.values;l=df.Low.values
    for i in range(order,len(df)-order):
        if h[i]>=np.nanmax(h[i-order:i+order+1]):out.append({'i':i,'type':'H','price':float(h[i]),'date':str(df.index[i].date())})
        if l[i]<=np.nanmin(l[i-order:i+order+1]):out.append({'i':i,'type':'L','price':float(l[i]),'date':str(df.index[i].date())})
    return sorted(out,key=lambda x:x['i'])

def zones(df,pv):
    last=float(df.Close.iloc[-1]);a=float(atr(df).iloc[-1]);tol=max(a*.55,last*.004)
    def side(kind,where):
        pts=[p for p in pv[-80:] if p['type']==kind and ((where=='support' and p['price']<last) or (where=='resistance' and p['price']>last))]
        cl=[]
        for p in pts:
            c=next((z for z in cl if abs(z['center']-p['price'])<=tol),None)
            if c is None:c={'center':p['price'],'vals':[]};cl.append(c)
            c['vals'].append(p['price']);c['center']=float(np.mean(c['vals']))
        rows=[]
        for c in cl:
            rows.append({'center':round(c['center'],2),'low':round(c['center']-tol*.65,2),'high':round(c['center']+tol*.65,2),'touches':len(c['vals']),'strength':'Strong' if len(c['vals'])>=3 else ('Medium' if len(c['vals'])==2 else 'Light')})
        rows.sort(key=lambda x:(-x['touches'],abs(x['center']-last)))
        return rows[:3]
    return {'supports':side('L','support'),'resistances':side('H','resistance')}

def fibonacci(df):
    w=df.tail(120);hi=float(w.High.max());lo=float(w.Low.min());hi_i=w.High.values.argmax();lo_i=w.Low.values.argmin();up=lo_i<hi_i;r=hi-lo
    retr={k:round(hi-r*v if up else lo+r*v,2) for k,v in {'23.6':.236,'38.2':.382,'50.0':.5,'61.8':.618,'78.6':.786}.items()}
    ext={k:round(hi+r*v if up else lo-r*v,2) for k,v in {'127.2':.272,'161.8':.618,'200.0':1.0}.items()}
    return {'swing_direction':'Up' if up else 'Down','swing_high':round(hi,2),'swing_low':round(lo,2),'retracements':retr,'extensions':ext}

def zigzag(df):
    c=df.Close.astype(float);a=float(atr(df).iloc[-1]);last=float(c.iloc[-1]);threshold=max(a/last*1.25,.018);start=max(0,len(df)-180);direction=0;ext=float(c.iloc[start]);ext_i=start;zz=[]
    for i in range(start+1,len(df)):
        p=float(c.iloc[i])
        if direction>=0:
            if p>ext:ext=p;ext_i=i
            if p/ext-1<=-threshold:zz.append({'type':'H','price':round(ext,2),'date':str(df.index[ext_i].date())});direction=-1;ext=p;ext_i=i
        if direction<=0:
            if p<ext:ext=p;ext_i=i
            if p/ext-1>=threshold:zz.append({'type':'L','price':round(ext,2),'date':str(df.index[ext_i].date())});direction=1;ext=p;ext_i=i
    return zz[-8:]

def elliott(zz):
    recent=zz[-6:];phase='No clean Elliott-style sequence';bias='Neutral';conf='Low';labels=[f'P{i+1}' for i in range(len(recent))]
    if len(recent)==6:
        t=''.join(x['type'] for x in recent);v=[x['price'] for x in recent]
        if t=='LHLHLH' and v[3]>v[1] and v[4]>v[2] and v[5]>v[3]:phase='Bullish 5-wave-style impulse candidate';bias='Bullish';conf='Medium';labels=['Start','Wave 1','Wave 2','Wave 3','Wave 4','Wave 5']
        elif t=='HLHLHL' and v[3]<v[1] and v[4]<v[2] and v[5]<v[3]:phase='Bearish 5-wave-style impulse candidate';bias='Bearish';conf='Medium';labels=['Start','Wave 1','Wave 2','Wave 3','Wave 4','Wave 5']
    pts=[dict(x,label=labels[i]) for i,x in enumerate(recent)]
    return {'phase':phase,'bias':bias,'confidence':conf,'points':pts,'note':'Interpretive swing framework; alternate wave counts are possible.'}

def wyckoff(df,volume_flow):
    c=df.Close.astype(float);e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();w=df.tail(60);last=float(c.iloc[-1]);hi=float(w.High.max());lo=float(w.Low.min());pos=(last-lo)/max(hi-lo,1e-9);prior=float(c.iloc[-61]/c.iloc[-120]-1) if len(c)>=120 else 0
    phase='Trading range / transition';bias='Neutral';conf='Low';evidence=[]
    if e20.iloc[-1]>e50.iloc[-1] and pos>.62:phase='Markup';bias='Bullish';conf='Medium';evidence=['Price is high in the 60-day range.','EMA20 is above EMA50.']
    elif e20.iloc[-1]<e50.iloc[-1] and pos<.38:phase='Markdown';bias='Bearish';conf='Medium';evidence=['Price is low in the 60-day range.','EMA20 is below EMA50.']
    elif prior<-.08:phase='Accumulation candidate';bias='Bullish-leaning';evidence=['A prior decline is followed by a range.','A breakout is still required.']
    elif prior>.08:phase='Distribution candidate';bias='Bearish-leaning';evidence=['A prior advance is followed by a range.','A breakdown is still required.']
    obv=(volume_flow or {}).get('obv_trend');
    if obv:evidence.append(f'OBV trend: {obv}.')
    return {'phase':phase,'bias':bias,'confidence':conf,'range_position':round(pos,3),'evidence':evidence,'note':'Rule-based Wyckoff-style context, not a definitive phase label.'}

def enrich(asset,ticker,file):
    path=DATA/file
    if not path.exists():return
    payload=json.loads(path.read_text())
    df=download(ticker);pv=pivots(df);zz=zigzag(df)
    payload['support_resistance']=zones(df,pv)
    payload['fibonacci_deep']=fibonacci(df)
    payload['elliott_wave']=elliott(zz)
    payload['wyckoff']=wyckoff(df,payload.get('volume_flow'))
    payload['frameworks_updated_utc']=datetime.now(timezone.utc).isoformat()
    payload['framework_note']='Support/resistance are zones. Elliott and Wyckoff are interpretive frameworks and are not guaranteed predictions.'
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')

def main():
    for asset,(ticker,file) in ASSETS.items():enrich(asset,ticker,file)
if __name__=='__main__':main()
