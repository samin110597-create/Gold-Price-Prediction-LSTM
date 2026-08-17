import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path('data')
ASSETS = {
    'gold': ('GC=F', DATA/'gold_technicals.json'),
    'silver': ('SI=F', DATA/'silver_technicals.json'),
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker):
    df = flatten(yf.download(ticker, period='2y', interval='1d', auto_adjust=False,
                             progress=False, threads=False))
    if df.empty:
        raise RuntimeError(f'No daily data for {ticker}')
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy(); df.index = idx
    return df.dropna(subset=['Open','High','Low','Close']).sort_index()


def atr(df, n=14):
    pc = df['Close'].shift(1)
    tr = pd.concat([(df['High']-df['Low']).abs(), (df['High']-pc).abs(), (df['Low']-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def pivots(s, order=3, mode='high'):
    v = s.values; out=[]
    for i in range(order, len(s)-order):
        w = v[i-order:i+order+1]
        if mode=='high' and np.isfinite(v[i]) and v[i] >= np.nanmax(w): out.append((i, s.index[i], float(v[i])))
        if mode=='low' and np.isfinite(v[i]) and v[i] <= np.nanmin(w): out.append((i, s.index[i], float(v[i])))
    return out


def quality(base, **checks):
    score = float(base)
    for _, val in checks.items(): score += float(val)
    return int(max(20, min(95, round(score))))


def pattern(name, bias, status, confidence, q, start, end, confirmation=None, invalidation=None, target=None, detail='', segments=None):
    return {
        'name':name, 'bias':bias, 'status':status, 'confidence':confidence,
        'structural_quality':q,
        'start':str(pd.Timestamp(start).date()) if start is not None else None,
        'end':str(pd.Timestamp(end).date()) if end is not None else None,
        'confirmation':round(float(confirmation),2) if confirmation is not None and np.isfinite(confirmation) else None,
        'invalidation':round(float(invalidation),2) if invalidation is not None and np.isfinite(invalidation) else None,
        'target':round(float(target),2) if target is not None and np.isfinite(target) else None,
        'detail':detail,
        'segments':segments or [],
    }


def advanced_patterns(df):
    df=df.tail(180).copy(); c=df['Close'].astype(float); h=df['High'].astype(float); l=df['Low'].astype(float)
    a=float(atr(df).iloc[-1]); last=float(c.iloc[-1]); hs=pivots(h,3,'high'); ls=pivots(l,3,'low'); pats=[]
    tol=max(a*.65,last*.006)

    # Double top / bottom with neckline confirmation.
    if len(hs)>=2:
        p1,p2=hs[-2],hs[-1]
        if 5 <= p2[0]-p1[0] <= 70 and abs(p2[2]-p1[2]) <= tol:
            between=l.iloc[p1[0]:p2[0]+1]
            neck=float(between.min()); confirmed=last < neck
            height=(p1[2]+p2[2])/2-neck
            q=quality(48, symmetry=max(0,18*(1-abs(p2[2]-p1[2])/tol)), spacing=min(12,(p2[0]-p1[0])/3), confirmation=12 if confirmed else 0)
            seg=[{'x1':str(p1[1].date()),'y1':p1[2],'x2':str(p2[1].date()),'y2':p2[2],'label':'tops'}]
            pats.append(pattern('Double top','Bearish','Confirmed' if confirmed else 'Candidate','High' if confirmed else 'Medium',q,p1[1],df.index[-1],neck,max(p1[2],p2[2])+a*.35,neck-height,'Two comparable swing highs; bearish only after neckline confirmation.',seg))
    if len(ls)>=2:
        p1,p2=ls[-2],ls[-1]
        if 5 <= p2[0]-p1[0] <= 70 and abs(p2[2]-p1[2]) <= tol:
            between=h.iloc[p1[0]:p2[0]+1]
            neck=float(between.max()); confirmed=last > neck
            height=neck-(p1[2]+p2[2])/2
            q=quality(48, symmetry=max(0,18*(1-abs(p2[2]-p1[2])/tol)), spacing=min(12,(p2[0]-p1[0])/3), confirmation=12 if confirmed else 0)
            seg=[{'x1':str(p1[1].date()),'y1':p1[2],'x2':str(p2[1].date()),'y2':p2[2],'label':'bottoms'}]
            pats.append(pattern('Double bottom','Bullish','Confirmed' if confirmed else 'Candidate','High' if confirmed else 'Medium',q,p1[1],df.index[-1],neck,min(p1[2],p2[2])-a*.35,neck+height,'Two comparable swing lows; bullish only after neckline confirmation.',seg))

    # Head & shoulders / inverse H&S using last three same-type pivots.
    if len(hs)>=3:
        s1,head,s2=hs[-3],hs[-2],hs[-1]
        if head[2] > max(s1[2],s2[2])+a*.55 and abs(s1[2]-s2[2]) <= tol*1.25:
            lows1=l.iloc[s1[0]:head[0]+1]; lows2=l.iloc[head[0]:s2[0]+1]
            n1=float(lows1.min()); n2=float(lows2.min()); neck=(n1+n2)/2; confirmed=last<neck
            q=quality(50, shoulders=max(0,18*(1-abs(s1[2]-s2[2])/(tol*1.25))), head=10, confirmation=12 if confirmed else 0)
            tgt=neck-(head[2]-neck)
            seg=[{'x1':str(s1[1].date()),'y1':s1[2],'x2':str(head[1].date()),'y2':head[2],'label':'left-head'}, {'x1':str(head[1].date()),'y1':head[2],'x2':str(s2[1].date()),'y2':s2[2],'label':'head-right'}]
            pats.append(pattern('Head & shoulders','Bearish','Confirmed' if confirmed else 'Candidate','High' if confirmed else 'Medium',q,s1[1],df.index[-1],neck,head[2]+a*.35,tgt,'Shoulders are similar and the head is higher; neckline break is required.',seg))
    if len(ls)>=3:
        s1,head,s2=ls[-3],ls[-2],ls[-1]
        if head[2] < min(s1[2],s2[2])-a*.55 and abs(s1[2]-s2[2]) <= tol*1.25:
            highs1=h.iloc[s1[0]:head[0]+1]; highs2=h.iloc[head[0]:s2[0]+1]
            n1=float(highs1.max()); n2=float(highs2.max()); neck=(n1+n2)/2; confirmed=last>neck
            q=quality(50, shoulders=max(0,18*(1-abs(s1[2]-s2[2])/(tol*1.25))), head=10, confirmation=12 if confirmed else 0)
            tgt=neck+(neck-head[2])
            seg=[{'x1':str(s1[1].date()),'y1':s1[2],'x2':str(head[1].date()),'y2':head[2],'label':'left-head'}, {'x1':str(head[1].date()),'y1':head[2],'x2':str(s2[1].date()),'y2':s2[2],'label':'head-right'}]
            pats.append(pattern('Inverse head & shoulders','Bullish','Confirmed' if confirmed else 'Candidate','High' if confirmed else 'Medium',q,s1[1],df.index[-1],neck,head[2]-a*.35,tgt,'Shoulders are similar and the head is lower; neckline break is required.',seg))

    # Trend-line formations on last 35 bars.
    w=df.tail(35); x=np.arange(len(w)); hi=w['High'].astype(float).values; lo=w['Low'].astype(float).values
    sh=np.polyfit(x,hi,1)[0]; sl=np.polyfit(x,lo,1)[0]
    upper0=float(np.polyval(np.polyfit(x,hi,1),0)); upperN=float(np.polyval(np.polyfit(x,hi,1),len(w)-1)); lower0=float(np.polyval(np.polyfit(x,lo,1),0)); lowerN=float(np.polyval(np.polyfit(x,lo,1),len(w)-1))
    width0=max(upper0-lower0,1e-9); widthN=max(upperN-lowerN,1e-9); contract=widthN/width0
    flat=max(a*.015,last*.00018)
    ptype=None; bias='Neutral'; conf='Medium'
    if contract < .72:
        if abs(sh)<=flat and sl>flat: ptype='Ascending triangle'; bias='Bullish'
        elif sh<-flat and abs(sl)<=flat: ptype='Descending triangle'; bias='Bearish'
        elif sh<-flat and sl>flat: ptype='Symmetrical triangle'; bias='Neutral'
        elif sh>flat and sl>flat and sl>sh*1.25: ptype='Rising wedge'; bias='Bearish'
        elif sh<-flat and sl<-flat and sh<sl*1.25: ptype='Falling wedge'; bias='Bullish'
    if ptype:
        up_break=last>upperN+a*.10; dn_break=last<lowerN-a*.10
        status='Confirmed' if (up_break or dn_break) else 'Candidate'
        actual_bias='Bullish' if up_break else ('Bearish' if dn_break else bias)
        confirm=upperN if actual_bias=='Bullish' else (lowerN if actual_bias=='Bearish' else None)
        inval=lowerN if actual_bias=='Bullish' else (upperN if actual_bias=='Bearish' else None)
        tgt=(last+width0) if actual_bias=='Bullish' else ((last-width0) if actual_bias=='Bearish' else None)
        q=quality(45, contraction=max(0,min(25,(1-contract)*60)), confirmation=12 if status=='Confirmed' else 0)
        seg=[{'x1':str(w.index[0].date()),'y1':upper0,'x2':str(w.index[-1].date()),'y2':upperN,'label':'upper trendline'}, {'x1':str(w.index[0].date()),'y1':lower0,'x2':str(w.index[-1].date()),'y2':lowerN,'label':'lower trendline'}]
        pats.append(pattern(ptype,actual_bias,status,'High' if status=='Confirmed' else conf,q,w.index[0],w.index[-1],confirm,inval,tgt,f'35-bar formation; range width contracted to {contract:.0%} of its starting width.',seg))

    # Flag candidate: impulse followed by smaller counter-trend consolidation.
    if len(df)>=30:
        impulse=df.iloc[-25:-10]; flag=df.iloc[-10:]
        imp_ret=float(impulse['Close'].iloc[-1]/impulse['Close'].iloc[0]-1)
        fx=np.arange(len(flag)); fsl=np.polyfit(fx,flag['Close'].astype(float).values,1)[0]/last
        flag_range=float((flag['High'].max()-flag['Low'].min())/last)
        if imp_ret>.055 and fsl<0 and abs(fsl)<abs(imp_ret)/10 and flag_range<abs(imp_ret)*.75:
            confirm=float(flag['High'].max()); confirmed=last>confirm
            q=quality(48, impulse=min(18,imp_ret*180), compact=max(0,12*(1-flag_range/max(abs(imp_ret),1e-6))), confirmation=12 if confirmed else 0)
            pats.append(pattern('Bull flag','Bullish','Confirmed' if confirmed else 'Candidate','High' if confirmed else 'Medium',q,impulse.index[0],df.index[-1],confirm,float(flag['Low'].min())-a*.2,confirm+abs(imp_ret)*last,'Strong prior advance followed by a smaller downward/sideways consolidation.'))
        elif imp_ret<-.055 and fsl>0 and abs(fsl)<abs(imp_ret)/10 and flag_range<abs(imp_ret)*.75:
            confirm=float(flag['Low'].min()); confirmed=last<confirm
            q=quality(48, impulse=min(18,abs(imp_ret)*180), compact=max(0,12*(1-flag_range/max(abs(imp_ret),1e-6))), confirmation=12 if confirmed else 0)
            pats.append(pattern('Bear flag','Bearish','Confirmed' if confirmed else 'Candidate','High' if confirmed else 'Medium',q,impulse.index[0],df.index[-1],confirm,float(flag['High'].max())+a*.2,confirm-abs(imp_ret)*last,'Strong prior decline followed by a smaller upward/sideways consolidation.'))

    # Sort confirmed first, then structural quality; keep chart readable.
    pats.sort(key=lambda p: (p['status']!='Confirmed', -p['structural_quality']))
    return pats[:8]


def divergences(df):
    c=df['Close'].astype(float); h=df['High'].astype(float); l=df['Low'].astype(float); rr=rsi(c); macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=macd.ewm(span=9,adjust=False).mean(); mh=macd-ms
    hs=pivots(h.tail(120),3,'high'); ls=pivots(l.tail(120),3,'low'); out=[]
    # pivot indices are local to tail(120); map by timestamps for indicator lookup
    if len(hs)>=2:
        a,b=hs[-2],hs[-1]; da,db=a[1],b[1]
        if b[2]>a[2] and rr.loc[db]<rr.loc[da]-3:
            out.append({'name':'Bearish RSI divergence','bias':'Bearish','status':'Active','date':str(db.date()),'price':round(b[2],2),'quality':70,'detail':'Price made a higher swing high while RSI made a lower high.'})
        if b[2]>a[2] and mh.loc[db]<mh.loc[da]:
            out.append({'name':'Bearish MACD divergence','bias':'Bearish','status':'Active','date':str(db.date()),'price':round(b[2],2),'quality':65,'detail':'Price made a higher swing high while MACD histogram weakened.'})
    if len(ls)>=2:
        a,b=ls[-2],ls[-1]; da,db=a[1],b[1]
        if b[2]<a[2] and rr.loc[db]>rr.loc[da]+3:
            out.append({'name':'Bullish RSI divergence','bias':'Bullish','status':'Active','date':str(db.date()),'price':round(b[2],2),'quality':70,'detail':'Price made a lower swing low while RSI made a higher low.'})
        if b[2]<a[2] and mh.loc[db]>mh.loc[da]:
            out.append({'name':'Bullish MACD divergence','bias':'Bullish','status':'Active','date':str(db.date()),'price':round(b[2],2),'quality':65,'detail':'Price made a lower swing low while MACD histogram improved.'})
    return out[-4:]


def signal_events(df):
    df=df.tail(180).copy(); c=df['Close'].astype(float); h=df['High'].astype(float); l=df['Low'].astype(float); vol=df.get('Volume',pd.Series(index=df.index,dtype=float)).astype(float)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean(); rr=rsi(c); macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=macd.ewm(span=9,adjust=False).mean()
    prior_hi=h.shift(1).rolling(20).max(); prior_lo=l.shift(1).rolling(20).min(); vm=vol.rolling(20).mean() if vol.notna().sum() else pd.Series(index=df.index,dtype=float)
    events=[]
    def add(i,name,bias,kind,detail):
        events.append({'date':str(df.index[i].date()),'price':round(float(c.iloc[i]),2),'name':name,'bias':bias,'kind':kind,'detail':detail})
    start=max(1,len(df)-70)
    for i in range(start,len(df)):
        if e20.iloc[i]>e50.iloc[i] and e20.iloc[i-1]<=e50.iloc[i-1]: add(i,'EMA20/50 bullish cross','Bullish','trend','EMA20 crossed above EMA50.')
        if e20.iloc[i]<e50.iloc[i] and e20.iloc[i-1]>=e50.iloc[i-1]: add(i,'EMA20/50 bearish cross','Bearish','trend','EMA20 crossed below EMA50.')
        if e50.iloc[i]>e200.iloc[i] and e50.iloc[i-1]<=e200.iloc[i-1]: add(i,'Golden cross','Bullish','trend','EMA50 crossed above EMA200.')
        if e50.iloc[i]<e200.iloc[i] and e50.iloc[i-1]>=e200.iloc[i-1]: add(i,'Death cross','Bearish','trend','EMA50 crossed below EMA200.')
        if macd.iloc[i]>ms.iloc[i] and macd.iloc[i-1]<=ms.iloc[i-1]: add(i,'MACD bullish cross','Bullish','momentum','MACD crossed above signal line.')
        if macd.iloc[i]<ms.iloc[i] and macd.iloc[i-1]>=ms.iloc[i-1]: add(i,'MACD bearish cross','Bearish','momentum','MACD crossed below signal line.')
        if np.isfinite(prior_hi.iloc[i]) and c.iloc[i]>prior_hi.iloc[i]: add(i,'20D breakout','Bullish','breakout','Close exceeded the prior 20-day high.')
        if np.isfinite(prior_lo.iloc[i]) and c.iloc[i]<prior_lo.iloc[i]: add(i,'20D breakdown','Bearish','breakout','Close fell below the prior 20-day low.')
        if rr.iloc[i]<70 and rr.iloc[i-1]>=70: add(i,'RSI exited overbought','Bearish','momentum','RSI crossed back below 70.')
        if rr.iloc[i]>30 and rr.iloc[i-1]<=30: add(i,'RSI exited oversold','Bullish','momentum','RSI crossed back above 30.')
        if len(vm) and np.isfinite(vm.iloc[i]) and vm.iloc[i]>0 and np.isfinite(vol.iloc[i]) and vol.iloc[i] >= 1.8*vm.iloc[i]: add(i,'Volume expansion','Neutral','volume',f'Volume reached {vol.iloc[i]/vm.iloc[i]:.1f}x its 20-day average.')
    # De-duplicate same event family and keep the most recent readable set.
    seen=set(); keep=[]
    for e in reversed(events):
        key=e['name']
        if key not in seen:
            keep.append(e); seen.add(key)
        if len(keep)>=9: break
    return list(reversed(keep))


def nearest_levels(payload, last):
    levels=[]
    sr=payload.get('support_resistance') or {}
    su=sorted(sr.get('supports') or [], key=lambda x:abs(float(x.get('center',0))-last))[:2]
    re=sorted(sr.get('resistances') or [], key=lambda x:abs(float(x.get('center',0))-last))[:2]
    for x in su: levels.append({'label':'Support','price':round(float(x['center']),2),'kind':'support','strength':x.get('strength')})
    for x in re: levels.append({'label':'Resistance','price':round(float(x['center']),2),'kind':'resistance','strength':x.get('strength')})
    conf=payload.get('confirmation') or {}
    if conf.get('bullish_above') is not None: levels.append({'label':'Bull confirm','price':round(float(conf['bullish_above']),2),'kind':'bull_trigger'})
    if conf.get('bearish_below') is not None: levels.append({'label':'Bear confirm','price':round(float(conf['bearish_below']),2),'kind':'bear_trigger'})
    return levels[:6]


def enrich(asset,ticker,path):
    payload=json.loads(path.read_text())
    raw=download(ticker)
    d=raw.tail(180).copy(); c=d['Close'].astype(float)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean(); mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bbu=mid+2*sd; bbl=mid-2*sd
    vol=d.get('Volume',pd.Series(0,index=d.index)).fillna(0).astype(float); typical=(d['High']+d['Low']+d['Close'])/3; vwap=(typical*vol).rolling(20).sum()/vol.rolling(20).sum().replace(0,np.nan)
    old={str(x.get('date')):x for x in payload.get('chart',[])}
    chart=[]
    for i,dt in enumerate(d.index):
        key=str(dt.date()); prev=old.get(key,{})
        chart.append({'date':key,'open':round(float(d['Open'].iloc[i]),2),'high':round(float(d['High'].iloc[i]),2),'low':round(float(d['Low'].iloc[i]),2),'close':round(float(c.iloc[i]),2),'volume':round(float(vol.iloc[i]),0) if np.isfinite(vol.iloc[i]) else None,'ema20':round(float(e20.iloc[i]),2),'ema50':round(float(e50.iloc[i]),2),'ema200':round(float(e200.iloc[i]),2),'bb_upper':round(float(bbu.iloc[i]),2) if np.isfinite(bbu.iloc[i]) else None,'bb_lower':round(float(bbl.iloc[i]),2) if np.isfinite(bbl.iloc[i]) else None,'vwap20':round(float(vwap.iloc[i]),2) if np.isfinite(vwap.iloc[i]) else prev.get('vwap20')})
    pats=advanced_patterns(d); divs=divergences(d); events=signal_events(d); last=float(c.iloc[-1])
    payload['chart']=chart
    payload['advanced_chart_patterns']=pats
    payload['divergence_signals']=divs
    payload['chart_signal_events']=events
    payload['chart_overlay']={
        'levels':nearest_levels(payload,last),
        'patterns':pats[:4],
        'divergences':divs,
        'events':events,
        'note':'Visual overlays are rule-based structural detections. Candidate patterns require confirmation; none is counted as predictive evidence unless its separate OOS audit passes.'
    }
    payload['overlay_updated_utc']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload,indent=2))
    print(asset, 'patterns', len(pats), 'divergences', len(divs), 'events', len(events))


if __name__=='__main__':
    for asset,(ticker,path) in ASSETS.items():
        try: enrich(asset,ticker,path)
        except Exception as e: print(asset,'overlay enrichment failed:',e)
