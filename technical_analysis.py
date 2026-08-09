import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

OUT_DIR = Path('data')
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS = {
    'gold': {'ticker': 'GC=F', 'name': 'Gold Futures', 'output': 'gold_technicals.json'},
    'silver': {'ticker': 'SI=F', 'name': 'Silver Futures', 'output': 'silver_technicals.json'},
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker, period='5y', interval='1d'):
    df = flatten(yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                             progress=False, threads=False))
    if df.empty:
        raise RuntimeError(f'No data for {ticker} {interval}')
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy(); df.index = idx
    return df.dropna(subset=['Open','High','Low','Close']).sort_index()


def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan); return 100-100/(1+rs)


def atr(df,n=14):
    pc=df['Close'].shift(1)
    tr=pd.concat([df['High']-df['Low'],(df['High']-pc).abs(),(df['Low']-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()


def adx(df,n=14):
    h,l=df['High'],df['Low']; up=h.diff(); down=-l.diff()
    pdm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index)
    mdm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index)
    a=atr(df,n)
    pdi=100*pdm.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    mdi=100*mdm.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi


def obv(df):
    vol=df.get('Volume',pd.Series(0,index=df.index)).fillna(0).astype(float)
    return (np.sign(df['Close'].diff()).fillna(0)*vol).cumsum()


def basic_snapshot(df):
    c=df['Close'].astype(float)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=m.ewm(span=9,adjust=False).mean()
    rr=rsi(c); ax,pdi,mdi=adx(df)
    if e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1]: trend='Strong bullish'
    elif e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1]: trend='Strong bearish'
    elif c.iloc[-1]>e20.iloc[-1] and e20.iloc[-1]>e50.iloc[-1]: trend='Bullish'
    elif c.iloc[-1]<e20.iloc[-1] and e20.iloc[-1]<e50.iloc[-1]: trend='Bearish'
    else: trend='Mixed'
    return {
        'trend':trend,'close':round(float(c.iloc[-1]),2),'ema20':round(float(e20.iloc[-1]),2),
        'ema50':round(float(e50.iloc[-1]),2),'ema200':round(float(e200.iloc[-1]),2),
        'rsi14':round(float(rr.iloc[-1]),1),'macd_bias':'Bullish' if m.iloc[-1]>ms.iloc[-1] else 'Bearish',
        'adx14':round(float(ax.iloc[-1]),1),'dmi_bias':'Bullish' if pdi.iloc[-1]>mdi.iloc[-1] else 'Bearish'
    }


def resample_ohlcv(df, rule):
    agg={'Open':'first','High':'max','Low':'min','Close':'last'}
    if 'Volume' in df.columns: agg['Volume']='sum'
    return df.resample(rule).agg(agg).dropna(subset=['Open','High','Low','Close'])


def multi_timeframe(daily, hourly):
    out={}
    try:
        h4=resample_ohlcv(hourly,'4h')
        out['4H']=basic_snapshot(h4)
    except Exception:
        out['4H']={'trend':'Unavailable'}
    out['1D']=basic_snapshot(daily)
    try:
        weekly=resample_ohlcv(daily,'W-FRI')
        out['1W']=basic_snapshot(weekly)
    except Exception:
        out['1W']={'trend':'Unavailable'}
    return out


def candle_patterns(df):
    if len(df)<4: return []
    a,b,c=df.iloc[-3],df.iloc[-2],df.iloc[-1]
    body=lambda x: abs(float(x.Close-x.Open)); rng=lambda x: max(float(x.High-x.Low),1e-9)
    upper=lambda x: float(x.High-max(x.Open,x.Close)); lower=lambda x: float(min(x.Open,x.Close)-x.Low)
    pats=[]
    def add(name,bias,meaning): pats.append({'name':name,'bias':bias,'meaning':meaning})
    if body(c)/rng(c)<.12: add('Doji','Neutral','Latest candle shows indecision.')
    if lower(c)>body(c)*2 and upper(c)<max(body(c),rng(c)*.15): add('Hammer / lower rejection','Bullish','Buyers rejected lower prices.')
    if upper(c)>body(c)*2 and lower(c)<max(body(c),rng(c)*.15): add('Shooting star / upper rejection','Bearish','Sellers rejected higher prices.')
    if c.Close>c.Open and b.Close<b.Open and c.Close>=b.Open and c.Open<=b.Close: add('Bullish engulfing','Bullish','Latest up candle fully covered the prior down candle.')
    if c.Close<c.Open and b.Close>b.Open and c.Open>=b.Close and c.Close<=b.Open: add('Bearish engulfing','Bearish','Latest down candle fully covered the prior up candle.')
    if c.High<b.High and c.Low>b.Low: add('Inside bar','Neutral','Price compressed inside the prior candle; breakout matters next.')
    if c.High>b.High and c.Low<b.Low: add('Outside bar','Neutral','Volatility expanded and both sides were tested.')
    if a.Close<a.Open and body(b)<body(a)*.55 and c.Close>c.Open and c.Close>(a.Open+a.Close)/2: add('Morning-star type reversal','Bullish','Three-candle sequence suggests downside rejection.')
    if a.Close>a.Open and body(b)<body(a)*.55 and c.Close<c.Open and c.Close<(a.Open+a.Close)/2: add('Evening-star type reversal','Bearish','Three-candle sequence suggests upside rejection.')
    if not pats: add('No major candle pattern','Neutral','No strong textbook candle pattern on the latest bar.')
    return pats[:5]


def swing_points(s, order=3, mode='high'):
    vals=s.values; pts=[]
    for i in range(order,len(s)-order):
        w=vals[i-order:i+order+1]
        if mode=='high' and vals[i]>=np.nanmax(w): pts.append((s.index[i],float(vals[i])))
        if mode=='low' and vals[i]<=np.nanmin(w): pts.append((s.index[i],float(vals[i])))
    return pts


def chart_patterns(df):
    h,l,c=df['High'].astype(float),df['Low'].astype(float),df['Close'].astype(float)
    a=float(atr(df).iloc[-1]); last=float(c.iloc[-1]); pats=[]
    def add(name,bias,confidence,meaning): pats.append({'name':name,'bias':bias,'confidence':confidence,'meaning':meaning})
    highs=swing_points(h.tail(120),3,'high'); lows=swing_points(l.tail(120),3,'low')
    if len(highs)>=2 and len(lows)>=2:
        hh=highs[-1][1]>highs[-2][1]; hl=lows[-1][1]>lows[-2][1]; lh=highs[-1][1]<highs[-2][1]; ll=lows[-1][1]<lows[-2][1]
        if hh and hl: add('Higher highs + higher lows','Bullish','High','Swing structure is trending upward.')
        elif lh and ll: add('Lower highs + lower lows','Bearish','High','Swing structure is trending downward.')
        else: add('Mixed swing structure','Neutral','Medium','Recent swing highs/lows disagree.')
        tol=max(a,last*.012)
        if abs(highs[-1][1]-highs[-2][1])<=tol and lows[-1][1]<min(highs[-1][1],highs[-2][1])-2*a: add('Possible double top','Bearish','Medium','Two similar swing highs; confirmation needs a break of support.')
        if abs(lows[-1][1]-lows[-2][1])<=tol and highs[-1][1]>max(lows[-1][1],lows[-2][1])+2*a: add('Possible double bottom','Bullish','Medium','Two similar swing lows; confirmation needs a break of resistance.')
    prior_hi=float(h.shift(1).rolling(20).max().iloc[-1]); prior_lo=float(l.shift(1).rolling(20).min().iloc[-1])
    if last>prior_hi: add('20-day breakout','Bullish','High','Price closed above the previous 20-day high.')
    elif last<prior_lo: add('20-day breakdown','Bearish','High','Price closed below the previous 20-day low.')
    width20=float((h.tail(20).max()-l.tail(20).min())/last); widths=((h.rolling(20).max()-l.rolling(20).min())/c).dropna().tail(252)
    if len(widths)>30 and float((widths<=width20).mean())<.20: add('Range compression','Neutral','Medium','Recent range is unusually tight; breakout risk is higher.')
    y=c.tail(30).values; x=np.arange(len(y)); slope=np.polyfit(x,y,1)[0] if len(y)>=10 else 0; slope_pct=slope/last*100
    if slope_pct>.08: add('Rising price channel','Bullish','Medium','30-day price slope is clearly upward.')
    elif slope_pct<-.08: add('Falling price channel','Bearish','Medium','30-day price slope is clearly downward.')
    return pats[:8]


def volume_profile(df,bins=24):
    vol=df.get('Volume',pd.Series(0,index=df.index)).fillna(0).astype(float).tail(90); tp=((df['High']+df['Low']+df['Close'])/3).tail(90).astype(float)
    mask=(vol>0)&tp.notna()
    if mask.sum()<10: return {'poc':None,'volume_state':'Unavailable','obv_trend':'Unavailable','volume_vs_20d':None}
    tp=tp[mask]; vol=vol[mask]; edges=np.linspace(float(tp.min()),float(tp.max()),bins+1); ids=np.clip(np.digitize(tp.values,edges)-1,0,bins-1); sums=np.zeros(bins)
    for i,vv in zip(ids,vol.values): sums[i]+=vv
    j=int(np.argmax(sums)); poc=(edges[j]+edges[j+1])/2; v20=float(df['Volume'].tail(20).mean()); v5=float(df['Volume'].tail(5).mean()); vr=v5/v20 if v20>0 else np.nan
    ob=obv(df); trend='Rising' if ob.iloc[-1]>ob.iloc[-10] else 'Falling'; state='Heavy' if np.isfinite(vr) and vr>1.25 else ('Light' if np.isfinite(vr) and vr<.75 else 'Normal')
    return {'poc':round(float(poc),2),'volume_state':state,'obv_trend':trend,'volume_vs_20d':round(float(vr),2) if np.isfinite(vr) else None}


def technicals(df,hourly):
    c,h,l,v=df['Close'].astype(float),df['High'].astype(float),df['Low'].astype(float),df.get('Volume',pd.Series(0,index=df.index)).fillna(0).astype(float)
    ema9=c.ewm(span=9,adjust=False).mean(); ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean(); ema200=c.ewm(span=200,adjust=False).mean(); sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean(); sma200=c.rolling(200).mean()
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); macd_sig=macd.ewm(span=9,adjust=False).mean(); macd_hist=macd-macd_sig; rr=rsi(c,14)
    low14=l.rolling(14).min(); high14=h.rolling(14).max(); stoch_k=100*(c-low14)/(high14-low14).replace(0,np.nan); stoch_d=stoch_k.rolling(3).mean()
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bb_u=mid+2*sd; bb_l=mid-2*sd; bb_pct=(c-bb_l)/(bb_u-bb_l).replace(0,np.nan); bb_width=(bb_u-bb_l)/mid.replace(0,np.nan)*100
    aa=atr(df,14); adx14,plus_di,minus_di=adx(df,14); ob=obv(df); ob_slope=(ob-ob.shift(10))/ob.abs().rolling(60).max().replace(0,np.nan); typical=(h+l+c)/3; rvwap=(typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    realized20=c.pct_change().rolling(20).std()*np.sqrt(252)*100; histvol=realized20.dropna().tail(756); vol_pct=float((histvol<=realized20.iloc[-1]).mean()) if len(histvol) else .5
    last=float(c.iloc[-1]); at=float(aa.iloc[-1]); score=0.; signals=[]
    def sig(name,state,pts,detail):
        nonlocal score; score+=pts; signals.append({'name':name,'state':state,'score':pts,'detail':detail})
    sig('Short trend','Bullish' if last>ema20.iloc[-1] else 'Bearish',1 if last>ema20.iloc[-1] else -1,'Price is above EMA20' if last>ema20.iloc[-1] else 'Price is below EMA20')
    sig('Medium trend','Bullish' if ema20.iloc[-1]>ema50.iloc[-1] else 'Bearish',1 if ema20.iloc[-1]>ema50.iloc[-1] else -1,'EMA20 is above EMA50' if ema20.iloc[-1]>ema50.iloc[-1] else 'EMA20 is below EMA50')
    sig('Long trend','Bullish' if ema50.iloc[-1]>ema200.iloc[-1] else 'Bearish',1 if ema50.iloc[-1]>ema200.iloc[-1] else -1,'EMA50 is above EMA200' if ema50.iloc[-1]>ema200.iloc[-1] else 'EMA50 is below EMA200')
    sig('MACD','Bullish' if macd.iloc[-1]>macd_sig.iloc[-1] else 'Bearish',1 if macd.iloc[-1]>macd_sig.iloc[-1] else -1,'MACD is above signal line' if macd.iloc[-1]>macd_sig.iloc[-1] else 'MACD is below signal line')
    sig('MACD momentum','Improving' if macd_hist.iloc[-1]>macd_hist.iloc[-2] else 'Fading',.5 if macd_hist.iloc[-1]>macd_hist.iloc[-2] else -.5,'MACD histogram is improving' if macd_hist.iloc[-1]>macd_hist.iloc[-2] else 'MACD histogram is fading')
    rv=float(rr.iloc[-1])
    if 55<=rv<=72: sig('RSI','Bullish',1,'RSI is in constructive momentum zone')
    elif rv>=75: sig('RSI','Overbought',-.35,'RSI is stretched / overbought')
    elif rv<=25: sig('RSI','Oversold',.35,'RSI is stretched / oversold')
    elif rv<45: sig('RSI','Bearish',-1,'RSI is below neutral momentum')
    else: sig('RSI','Neutral',0,'RSI is near neutral')
    ad=float(adx14.iloc[-1]); pdi=float(plus_di.iloc[-1]); mdi=float(minus_di.iloc[-1])
    if ad>=20 and pdi>mdi: sig('ADX / DMI','Bullish trend',1,'Trend strength is active with +DI leading')
    elif ad>=20 and mdi>pdi: sig('ADX / DMI','Bearish trend',-1,'Trend strength is active with -DI leading')
    else: sig('ADX / DMI','Weak trend',0,'ADX does not confirm a strong directional trend')
    sk=float(stoch_k.iloc[-1]); sdv=float(stoch_d.iloc[-1])
    if sk>sdv and sk<80: sig('Stochastic','Bullish',.5,'%K is above %D without extreme overbought')
    elif sk<sdv and sk>20: sig('Stochastic','Bearish',-.5,'%K is below %D')
    else: sig('Stochastic','Extreme',0,'Stochastic is at an extreme / mixed')
    bp=float(bb_pct.iloc[-1])
    if .55<=bp<=.9: sig('Bollinger position','Bullish',.5,'Price is in upper half of the band')
    elif bp>1: sig('Bollinger position','Extended',-.25,'Price is above upper band and extended')
    elif .1<=bp<.45: sig('Bollinger position','Bearish',-.5,'Price is in lower half of the band')
    elif bp<0: sig('Bollinger position','Oversold',.25,'Price is below lower band')
    else: sig('Bollinger position','Neutral',0,'Price is near band midpoint')
    if np.isfinite(ob_slope.iloc[-1]) and ob_slope.iloc[-1]>0: sig('Volume / OBV','Confirming',.5,'OBV trend is rising')
    elif np.isfinite(ob_slope.iloc[-1]) and ob_slope.iloc[-1]<0: sig('Volume / OBV','Diverging',-.5,'OBV trend is falling')
    if np.isfinite(rvwap.iloc[-1]): sig('20D VWAP','Bullish' if last>rvwap.iloc[-1] else 'Bearish',.5 if last>rvwap.iloc[-1] else -.5,'Price is above rolling VWAP' if last>rvwap.iloc[-1] else 'Price is below rolling VWAP')
    score_pct=float(np.clip(50+score/8*50,0,100)); bias='Strong Bullish' if score_pct>=72 else ('Bullish' if score_pct>=60 else ('Strong Bearish' if score_pct<=28 else ('Bearish' if score_pct<=40 else 'Neutral / mixed')))
    prev=df.iloc[-2]; pivot=float((prev.High+prev.Low+prev.Close)/3); r1=2*pivot-float(prev.Low); s1=2*pivot-float(prev.High); r2=pivot+(float(prev.High)-float(prev.Low)); s2=pivot-(float(prev.High)-float(prev.Low))
    swing_hi=float(h.tail(60).max()); swing_lo=float(l.tail(60).min()); span=swing_hi-swing_lo; fib={'23.6':swing_hi-.236*span,'38.2':swing_hi-.382*span,'50.0':swing_hi-.5*span,'61.8':swing_hi-.618*span,'78.6':swing_hi-.786*span}
    recent_hi=float(h.iloc[-11:-1].max()); recent_lo=float(l.iloc[-11:-1].min()); direction=1 if score_pct>=58 else (-1 if score_pct<=42 else 0); center1=last+direction*at*.55; width1=at*.28; center5=last+direction*at*1.15; width5=at*.45
    zone1=[center1-width1,center1+width1] if direction else [last-at*.45,last+at*.45]; zone5=[center5-width5,center5+width5] if direction else [last-at*.9,last+at*.9]
    structure='Higher-high / bullish pressure' if last>recent_hi else ('Lower-low / bearish pressure' if last<recent_lo else 'Inside recent 10-day structure'); confirmation={'bullish_above':round(max(recent_hi,r1),2),'bearish_below':round(min(recent_lo,s1),2)}
    narrative=[('Trend and momentum confluence currently lean bullish.' if score_pct>=60 else ('Trend and momentum confluence currently lean bearish.' if score_pct<=40 else 'Technicals are mixed, so confirmation matters more than prediction.')),f"ADX is {ad:.1f}; {'trend strength is meaningful' if ad>=20 else 'trend strength is weak'}.",f"RSI is {rv:.1f} and MACD momentum is {'improving' if macd_hist.iloc[-1]>macd_hist.iloc[-2] else 'fading'}." ]
    vp=volume_profile(df); mtf=multi_timeframe(df,hourly); cp=candle_patterns(df); patterns=chart_patterns(df)
    chart=[{'date':pd.Timestamp(i).strftime('%Y-%m-%d'),'close':round(float(c.loc[i]),2),'ema20':round(float(ema20.loc[i]),2),'ema50':round(float(ema50.loc[i]),2),'bb_upper':round(float(bb_u.loc[i]),2) if np.isfinite(bb_u.loc[i]) else None,'bb_lower':round(float(bb_l.loc[i]),2) if np.isfinite(bb_l.loc[i]) else None} for i in df.index[-180:]]
    return {'updated_utc':datetime.now(timezone.utc).isoformat(),'technical_score':round(score_pct,1),'technical_bias':bias,'raw_confluence':round(score,2),'market_structure':structure,'candlestick':cp[0]['name'] if cp else 'No clear pattern','candlestick_patterns':cp,'chart_patterns':patterns,'multi_timeframe':mtf,'indicators':{'ema9':round(float(ema9.iloc[-1]),2),'ema20':round(float(ema20.iloc[-1]),2),'ema50':round(float(ema50.iloc[-1]),2),'ema200':round(float(ema200.iloc[-1]),2),'sma20':round(float(sma20.iloc[-1]),2),'sma50':round(float(sma50.iloc[-1]),2),'sma200':round(float(sma200.iloc[-1]),2),'rsi14':round(rv,2),'macd':round(float(macd.iloc[-1]),3),'macd_signal':round(float(macd_sig.iloc[-1]),3),'macd_hist':round(float(macd_hist.iloc[-1]),3),'adx14':round(ad,2),'plus_di':round(pdi,2),'minus_di':round(mdi,2),'stoch_k':round(sk,2),'stoch_d':round(sdv,2),'bb_percent_b':round(bp,3),'bb_width_pct':round(float(bb_width.iloc[-1]),2),'bb_upper':round(float(bb_u.iloc[-1]),2),'bb_mid':round(float(mid.iloc[-1]),2),'bb_lower':round(float(bb_l.iloc[-1]),2),'atr14':round(at,2),'atr14_pct':round(at/last*100,2),'realized_vol20_pct':round(float(realized20.iloc[-1]),2),'volatility_percentile':round(vol_pct,3),'rolling_vwap20':round(float(rvwap.iloc[-1]),2) if np.isfinite(rvwap.iloc[-1]) else None},'volume_flow':vp,'signals':signals,'pivots':{'pivot':round(pivot,2),'r1':round(r1,2),'r2':round(r2,2),'s1':round(s1,2),'s2':round(s2,2)},'fibonacci_60d':{k:round(v,2) for k,v in fib.items()},'swing_60d':{'high':round(swing_hi,2),'low':round(swing_lo,2)},'confirmation':confirmation,'technical_path_zones':{'1_3d':[round(zone1[0],2),round(zone1[1],2)],'5_10d':[round(zone5[0],2),round(zone5[1],2)]},'narrative':narrative,'chart':chart,'note':'Full technical research layer. Pattern names are rule-based detections, not guarantees. Use confirmation levels and audited model horizons together.'}


def main():
    for key,cfg in ASSETS.items():
        daily=download(cfg['ticker'],'5y','1d')
        try: hourly=download(cfg['ticker'],'730d','1h')
        except Exception: hourly=download(cfg['ticker'],'60d','1h')
        payload={'status':'ok','asset':key,'symbol':cfg['ticker'],'instrument':cfg['name'],**technicals(daily,hourly)}
        (OUT_DIR/cfg['output']).write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(key,payload['technical_bias'],payload['technical_score'],[p['name'] for p in payload['chart_patterns'][:3]])

if __name__=='__main__': main()
