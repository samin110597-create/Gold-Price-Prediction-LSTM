import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS = {
    "gold": {"ticker": "GC=F", "name": "Gold Futures", "output": "gold_technicals.json"},
    "silver": {"ticker": "SI=F", "name": "Silver Futures", "output": "silver_technicals.json"},
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker):
    df = flatten(yf.download(ticker, period="5y", interval="1d", auto_adjust=False,
                             progress=False, threads=False))
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy(); df.index = idx
    return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff(); down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / a.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / a.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di


def obv(df):
    vol = df.get("Volume", pd.Series(0, index=df.index)).fillna(0).astype(float)
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * vol).cumsum()


def candle_pattern(df):
    if len(df) < 3: return "No clear pattern"
    a, b = df.iloc[-2], df.iloc[-1]
    body = abs(b.Close-b.Open); rng = max(b.High-b.Low, 1e-9)
    upper = b.High-max(b.Open,b.Close); lower=min(b.Open,b.Close)-b.Low
    if b.Close>b.Open and a.Close<a.Open and b.Close>=a.Open and b.Open<=a.Close: return "Bullish engulfing"
    if b.Close<a.Open and a.Close>a.Open and b.Open>=a.Close and b.Close<=a.Open: return "Bearish engulfing"
    if lower > body*2 and upper < body and body/rng < .4: return "Hammer / rejection low"
    if upper > body*2 and lower < body and body/rng < .4: return "Shooting-star / rejection high"
    if body/rng < .12: return "Doji / indecision"
    return "Bullish candle" if b.Close>b.Open else "Bearish candle"


def technicals(df):
    c,h,l,v = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float), df.get("Volume", pd.Series(0,index=df.index)).fillna(0).astype(float)
    ema9=c.ewm(span=9,adjust=False).mean(); ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean(); ema200=c.ewm(span=200,adjust=False).mean()
    macd= c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); macd_sig=macd.ewm(span=9,adjust=False).mean(); macd_hist=macd-macd_sig
    rr=rsi(c,14)
    low14=l.rolling(14).min(); high14=h.rolling(14).max(); stoch_k=100*(c-low14)/(high14-low14).replace(0,np.nan); stoch_d=stoch_k.rolling(3).mean()
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bb_u=mid+2*sd; bb_l=mid-2*sd; bb_pct=(c-bb_l)/(bb_u-bb_l).replace(0,np.nan)
    aa=atr(df,14); adx14,plus_di,minus_di=adx(df,14)
    ob=obv(df); ob_slope=(ob-ob.shift(10))/ob.abs().rolling(60).max().replace(0,np.nan)
    typical=(h+l+c)/3; rvwap=(typical*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)

    last=float(c.iloc[-1]); at=float(aa.iloc[-1]); score=0.0; signals=[]
    def sig(name, state, pts, detail):
        nonlocal score; score += pts; signals.append({"name":name,"state":state,"score":pts,"detail":detail})

    if last>ema20.iloc[-1]: sig("Short trend","Bullish",1,"Price is above EMA20")
    else: sig("Short trend","Bearish",-1,"Price is below EMA20")
    if ema20.iloc[-1]>ema50.iloc[-1]: sig("Medium trend","Bullish",1,"EMA20 is above EMA50")
    else: sig("Medium trend","Bearish",-1,"EMA20 is below EMA50")
    if ema50.iloc[-1]>ema200.iloc[-1]: sig("Long trend","Bullish",1,"EMA50 is above EMA200")
    else: sig("Long trend","Bearish",-1,"EMA50 is below EMA200")
    if macd.iloc[-1]>macd_sig.iloc[-1]: sig("MACD","Bullish",1,"MACD is above signal line")
    else: sig("MACD","Bearish",-1,"MACD is below signal line")
    if macd_hist.iloc[-1]>macd_hist.iloc[-2]: sig("MACD momentum","Improving",.5,"MACD histogram is improving")
    else: sig("MACD momentum","Fading",-.5,"MACD histogram is fading")
    rv=float(rr.iloc[-1])
    if 55<=rv<=72: sig("RSI","Bullish",1,"RSI is in constructive momentum zone")
    elif rv>=75: sig("RSI","Overbought",-.35,"RSI is stretched / overbought")
    elif rv<=25: sig("RSI","Oversold",.35,"RSI is stretched / oversold")
    elif rv<45: sig("RSI","Bearish",-1,"RSI is below neutral momentum")
    else: sig("RSI","Neutral",0,"RSI is near neutral")
    ad=float(adx14.iloc[-1]); pdi=float(plus_di.iloc[-1]); mdi=float(minus_di.iloc[-1])
    if ad>=20 and pdi>mdi: sig("ADX / DMI","Bullish trend",1,"Trend strength is active with +DI leading")
    elif ad>=20 and mdi>pdi: sig("ADX / DMI","Bearish trend",-1,"Trend strength is active with -DI leading")
    else: sig("ADX / DMI","Weak trend",0,"ADX does not confirm a strong directional trend")
    sk=float(stoch_k.iloc[-1]); sdv=float(stoch_d.iloc[-1])
    if sk>sdv and sk<80: sig("Stochastic","Bullish",.5,"%K is above %D without extreme overbought")
    elif sk<sdv and sk>20: sig("Stochastic","Bearish",-.5,"%K is below %D")
    else: sig("Stochastic","Extreme",0,"Stochastic is at an extreme / mixed")
    bp=float(bb_pct.iloc[-1])
    if .55<=bp<=.9: sig("Bollinger position","Bullish",.5,"Price is in upper half of the band")
    elif bp>1: sig("Bollinger position","Extended",-.25,"Price is above upper band and extended")
    elif .1<=bp<.45: sig("Bollinger position","Bearish",-.5,"Price is in lower half of the band")
    elif bp<0: sig("Bollinger position","Oversold",.25,"Price is below lower band")
    else: sig("Bollinger position","Neutral",0,"Price is near band midpoint")
    if np.isfinite(ob_slope.iloc[-1]) and ob_slope.iloc[-1]>0: sig("Volume / OBV","Confirming",.5,"OBV trend is rising")
    elif np.isfinite(ob_slope.iloc[-1]) and ob_slope.iloc[-1]<0: sig("Volume / OBV","Diverging",-.5,"OBV trend is falling")
    if np.isfinite(rvwap.iloc[-1]):
        if last>rvwap.iloc[-1]: sig("20D VWAP","Bullish",.5,"Price is above rolling VWAP")
        else: sig("20D VWAP","Bearish",-.5,"Price is below rolling VWAP")

    score_pct=float(np.clip(50 + score/8.0*50, 0, 100))
    if score_pct>=72: bias="Strong Bullish"
    elif score_pct>=60: bias="Bullish"
    elif score_pct<=28: bias="Strong Bearish"
    elif score_pct<=40: bias="Bearish"
    else: bias="Neutral / mixed"

    prev=df.iloc[-2]; pivot=float((prev.High+prev.Low+prev.Close)/3); r1=2*pivot-float(prev.Low); s1=2*pivot-float(prev.High); r2=pivot+(float(prev.High)-float(prev.Low)); s2=pivot-(float(prev.High)-float(prev.Low))
    swing_hi=float(h.tail(60).max()); swing_lo=float(l.tail(60).min()); span=swing_hi-swing_lo
    fib={"23.6":swing_hi-.236*span,"38.2":swing_hi-.382*span,"50.0":swing_hi-.5*span,"61.8":swing_hi-.618*span,"78.6":swing_hi-.786*span}
    recent_hi=float(h.iloc[-11:-1].max()); recent_lo=float(l.iloc[-11:-1].min())
    direction=1 if score_pct>=58 else (-1 if score_pct<=42 else 0)
    center1=last + direction*at*.55; width1=at*.28
    center5=last + direction*at*1.15; width5=at*.45
    zone1=[center1-width1,center1+width1]; zone5=[center5-width5,center5+width5]
    if direction==0:
        zone1=[last-at*.45,last+at*.45]; zone5=[last-at*.9,last+at*.9]

    structure = "Higher-high / bullish pressure" if last>recent_hi else ("Lower-low / bearish pressure" if last<recent_lo else "Inside recent 10-day structure")
    confirmation={"bullish_above":round(max(recent_hi,r1),2),"bearish_below":round(min(recent_lo,s1),2)}
    narrative=[]
    if score_pct>=60: narrative.append("Trend and momentum confluence currently lean bullish.")
    elif score_pct<=40: narrative.append("Trend and momentum confluence currently lean bearish.")
    else: narrative.append("Technicals are mixed, so breakout confirmation matters more than prediction.")
    narrative.append(f"ADX is {ad:.1f}; {'trend strength is meaningful' if ad>=20 else 'trend strength is weak'}.")
    narrative.append(f"RSI is {rv:.1f} and MACD histogram is {'improving' if macd_hist.iloc[-1]>macd_hist.iloc[-2] else 'fading'}.")

    chart=[]
    for i in df.index[-180:]:
        chart.append({"date":pd.Timestamp(i).strftime("%Y-%m-%d"),"close":round(float(c.loc[i]),2),"ema20":round(float(ema20.loc[i]),2),"ema50":round(float(ema50.loc[i]),2),"bb_upper":round(float(bb_u.loc[i]),2) if np.isfinite(bb_u.loc[i]) else None,"bb_lower":round(float(bb_l.loc[i]),2) if np.isfinite(bb_l.loc[i]) else None})

    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "technical_score": round(score_pct,1), "technical_bias": bias, "raw_confluence": round(score,2),
        "market_structure": structure, "candlestick": candle_pattern(df),
        "indicators": {"ema9":round(float(ema9.iloc[-1]),2),"ema20":round(float(ema20.iloc[-1]),2),"ema50":round(float(ema50.iloc[-1]),2),"ema200":round(float(ema200.iloc[-1]),2),"rsi14":round(rv,2),"macd":round(float(macd.iloc[-1]),3),"macd_signal":round(float(macd_sig.iloc[-1]),3),"macd_hist":round(float(macd_hist.iloc[-1]),3),"adx14":round(ad,2),"plus_di":round(pdi,2),"minus_di":round(mdi,2),"stoch_k":round(sk,2),"stoch_d":round(sdv,2),"bb_percent_b":round(bp,3),"bb_upper":round(float(bb_u.iloc[-1]),2),"bb_mid":round(float(mid.iloc[-1]),2),"bb_lower":round(float(bb_l.iloc[-1]),2),"atr14":round(at,2),"rolling_vwap20":round(float(rvwap.iloc[-1]),2) if np.isfinite(rvwap.iloc[-1]) else None},
        "signals": signals,
        "pivots": {"pivot":round(pivot,2),"r1":round(r1,2),"r2":round(r2,2),"s1":round(s1,2),"s2":round(s2,2)},
        "fibonacci_60d": {k:round(v,2) for k,v in fib.items()},
        "swing_60d": {"high":round(swing_hi,2),"low":round(swing_lo,2)},
        "confirmation": confirmation,
        "technical_path_zones": {"1_3d":[round(zone1[0],2),round(zone1[1],2)],"5_10d":[round(zone5[0],2),round(zone5[1],2)]},
        "narrative": narrative,
        "chart": chart,
        "note": "Technical path zones are tighter chart-based scenarios, not certainty bands. Use them together with the wider statistical range and confirmation levels."
    }


def main():
    for key,cfg in ASSETS.items():
        df=download(cfg["ticker"])
        payload={"status":"ok","asset":key,"symbol":cfg["ticker"],"instrument":cfg["name"],**technicals(df)}
        (OUT_DIR/cfg["output"]).write_text(json.dumps(payload,indent=2),encoding="utf-8")
        print(key,payload["technical_bias"],payload["technical_score"],payload["technical_path_zones"])

if __name__=="__main__": main()
