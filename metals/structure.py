"""Confirmed pivots, close-only structure breaks and staged reversals; no backdated signals."""
import numpy as np

def scan(frame, left=3, right=3, retain_all=False):
    pivots, events = [], []
    books = {scope:{"high":None, "low":None, "direction":0} for scope in ("internal","major")}
    last_same = {}
    watches = []
    failed_breaks = set()
    def emit(i, kind, direction, scope="major", **extra):
        event = {"time":frame.index[i].isoformat(), "kind":kind, "direction":direction, "scope":scope, **extra}
        events.append(event)
        return event
    for i in range(len(frame)):
        row = frame.iloc[i]
        atr = row.atr
        if not np.isfinite(atr) or atr <= 0:
            continue
        c, h, l = row.close, row.high, row.low
        for scope, book in books.items():
            for side, direction in (("high",1),("low",-1)):
                pivot = book[side]
                if not pivot or pivot["broken"]:
                    continue
                level = pivot["price"]
                crossed = c > level+.08*atr if direction==1 else c < level-.08*atr
                if crossed:
                    kind = "CHoCH" if book["direction"] == -direction else "BOS"
                    emit(i,kind,direction,scope,level=level,pivot_time=pivot["pivot_time"],pivot_confirmed_at=pivot["confirmed_at"],index=i)
                    pivot["broken"] = True
                    book["direction"] = direction
                    for watch in watches:
                        if scope=="major" and watch["direction"]==direction and i > watch["index"]:
                            if kind == "CHoCH" and watch["stage"]=="WATCH":
                                watch.update(stage="TRANSITION", choch=i)
                                emit(i,"REVERSAL_TRANSITION",direction,origin=watch["time"])
                            elif kind=="BOS" and watch["stage"]=="HIGHER_LOW" and i > watch["hl"]:
                                watch["stage"]="CONFIRMED"
                                emit(i,"REVERSAL_CONFIRMED",direction,origin=watch["time"])
                elif scope=="major":
                    wick = (h-max(row.open,c)) if direction==1 else (min(row.open,c)-l)
                    swept = h > level+.08*atr and c < level if direction==1 else l < level-.08*atr and c > level
                    if swept and wick > .25*atr and not any(w["pivot"]==pivot["confirmed_at"] and w["stage"] not in ("FAILED","EXPIRED") for w in watches):
                        event = emit(i,"SWEEP_RECLAIM",-direction,level=level,note="OHLCV liquidity proxy; reversal watch only")
                        watches.append({"index":i,"time":event["time"],"direction":-direction,"pivot":pivot["confirmed_at"],"stage":"WATCH","invalidation":float(l if direction==-1 else h)})
        for watch in watches:
            if watch["stage"] in ("FAILED","EXPIRED"):
                continue
            invalid = c < watch["invalidation"] if watch["direction"]==1 else c > watch["invalidation"]
            if invalid:
                watch["stage"]="FAILED"
                emit(i,"REVERSAL_FAILED",watch["direction"],origin=watch["time"])
            elif i-watch["index"] > 20 and watch["stage"] != "CONFIRMED":
                watch["stage"]="EXPIRED"
        for old in events[-30:]:
            if old["kind"] in ("BOS","CHoCH") and (old["time"],old["scope"],old["kind"]) not in failed_breaks and 0 < i-old.get("index",i) <= 5:
                if (c-old["level"])*old["direction"] < -.08*atr:
                    failed_breaks.add((old["time"],old["scope"],old["kind"]))
                    emit(i,"FAILED_BREAKOUT", -old["direction"],old["scope"],level=old["level"],break_time=old["time"])
                    break
        j = i-right
        if j < left:
            continue
        for side in ("high","low"):
            value = float(frame.iloc[j][side])
            values = frame[side].iloc[j-left:i+1]
            extreme = values.max() if side=="high" else values.min()
            if value != extreme or (values==value).sum()!=1:
                continue
            opposite = "low" if side=="high" else "high"
            previous = last_same.get(side)
            last_opposite = last_same.get(opposite)
            distance = abs(value-last_opposite["price"])/frame.atr.iloc[j] if last_opposite else 0
            separation = j-last_opposite["index"] if last_opposite else 0
            impulse = abs(value-c)/atr
            volume = frame.relative_volume.iloc[j]
            quality = min(100,25+min(30,distance*10)+min(25,impulse*10)+min(15,separation*2)+(5 if np.isfinite(volume) and volume>=1.5 else 0))
            major = quality>=65 and distance>=1.5 and separation>=4
            label = ("HH" if value>previous["price"] else "LH") if side=="high" and previous else ("HL" if value>previous["price"] else "LL") if previous else side.upper()
            pivot = {"index":j,"pivot_time":frame.index[j].isoformat(),"confirmed_at":frame.index[i].isoformat(),"price":value,"side":side,"label":label,"quality":round(quality,1),"major":major,"broken":False,"rsi":float(frame.rsi.iloc[j]),"macd_hist":float(frame.macd_hist.iloc[j])}
            pivots.append(pivot)
            books["internal"][side] = dict(pivot)
            if major:
                books["major"][side] = dict(pivot)
            if previous and j-previous["index"]>=5:
                pdiff = value-previous["price"]
                rdiff = pivot["rsi"]-previous["rsi"]
                bearish = side=="high" and pdiff>.1*atr and rdiff < -3
                bullish = side=="low" and pdiff<-.1*atr and rdiff > 3
                if bearish or bullish:
                    emit(i,"RSI_DIVERGENCE",-1 if bearish else 1,pivot_time=pivot["pivot_time"],first_pivot=previous["pivot_time"],note="Confirmed on this bar; RSI primary, MACD secondary",macd_agrees=bool((pivot["macd_hist"]-previous["macd_hist"])*(-1 if bearish else 1)>0))
            for watch in watches:
                favorable = (watch["direction"]==1 and side=="low" and label=="HL") or (watch["direction"]==-1 and side=="high" and label=="LH")
                if watch["stage"]=="TRANSITION" and j > watch["choch"] and favorable:
                    watch.update(stage="HIGHER_LOW",hl=i)
                    emit(i,"REVERSAL_PIVOT",watch["direction"],pivot_time=pivot["pivot_time"],origin=watch["time"])
            last_same[side] = pivot
        spread = h-l
        if np.isfinite(row.relative_volume) and row.relative_volume>=1.8 and spread>0:
            close_location = (c-l)/spread
            if close_location>.75 and c>=row.open and spread<atr:
                emit(i,"ABSORPTION_PROXY",1,note="High volume, narrow spread, close near high; no participant identity inferred")
            elif close_location<.25 and c<=row.open and spread<atr:
                emit(i,"DISTRIBUTION_PROXY",-1,note="High volume, narrow spread, close near low; OHLCV proxy")
    last = frame.iloc[-1]
    levels = []
    # Pivots are observed objectives, never an ATR forecast.
    for p in reversed(pivots):
        if not p["major"]:
            continue
        if any(abs(z["price"]-p["price"]) < .25*last.atr for z in levels):
            continue
        levels.append({"price":p["price"],"zone":[p["price"]-.25*last.atr,p["price"]+.25*last.atr],"basis":"Confirmed major swing","confirmed_at":p["confirmed_at"],"quality":p["quality"],"role":"support" if p["price"]<last.close else "resistance"})
        if len(levels)>=12:
            break
    return {"direction":books["major"]["direction"],"internal_direction":books["internal"]["direction"],"pivots":pivots if retain_all else pivots[-120:],"events":events if retain_all else events[-180:],"levels":sorted(levels,key=lambda x:x["price"]),"reversals":[{k:v for k,v in w.items() if k not in ("index","choch","hl")} for w in watches[-8:]],"note":"Pivot coordinates are historical; every signal is dated when its required right-hand bars close."}

def families(frame, structure):
    r = frame.iloc[-1]
    sign = lambda v: int(np.sign(v)) if np.isfinite(v) else 0
    votes = {"trend_structure":structure["direction"] or sign(r.ema20-r.ema50), "momentum":sign(r.rsi-50) if abs(r.rsi-50)>5 else 0, "flow":sign(r.cmf) if np.isfinite(r.cmf) and abs(r.cmf)>.05 else 0}
    score = 50+sum(votes.values())*50/3
    return {"score":round(score,1),"votes":votes,"note":"Three capped evidence families. Confluence score is not a probability.","regime":"TREND" if r.adx>=25 else "RANGE","volatility":"HIGH" if r.atr_percentile>.8 else "LOW" if r.atr_percentile<.2 else "NORMAL","extension":"OVERBOUGHT" if r.rsi>=72 else "OVERSOLD" if r.rsi<=28 else "NORMAL"}
