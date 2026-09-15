"""Causal technical playbooks and educational, fixed-level OHLC replay.

Rule detection is separate from entry confirmation, execution and validation.
No replay frequency is used as a live calibrated probability.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import math
import numpy as np
import pandas as pd

VERSION = "technical-playbooks-v1"
CATALOG = {
    "EMA_PULLBACK": ("Trend pullback", "Trend", "Price retests EMA20 while EMA20/50 and momentum agree, then closes back with the trend.", "Prefer an orderly pullback within a trend; confirm through the rejection candle.", "A close through the pullback extreme invalidates the idea; avoid buying an extended impulse."),
    "RANGE_BREAKOUT": ("20-bar range breakout", "Trend", "A completed close clears the prior 20-bar high or low by 0.10 ATR.", "Look for acceptance outside the range; a later retest can offer a better entry.", "A wick outside the range is not a close breakout. A fast return inside can trap breakout traders."),
    "BREAKOUT_RETEST": ("Breakout retest", "Trend", "Within eight bars of a range breakout, price revisits the original boundary and closes back on the breakout side.", "Use the retest extreme for invalidation and the frozen boundary to judge acceptance.", "Repeated closes back inside the old range weaken the continuation case."),
    "SQUEEZE_RELEASE": ("Volatility squeeze release", "Volatility", "Bollinger width was in the bottom fifth of its prior 60-bar history; price now closes outside the previous band.", "Compression identifies a condition; the completed breakout supplies direction.", "A squeeze alone is not bullish or bearish. Avoid assuming every expansion continues."),
    "RSI_RECLAIM": ("RSI momentum reclaim", "Momentum", "RSI crosses 50 in the direction of the EMA20/50 trend.", "Use this as momentum confirmation near a good location, not a reason to chase price.", "RSI and MACD describe related momentum; counting both as independent probability evidence exaggerates confidence."),
    "MACD_TURN": ("MACD momentum turn", "Momentum", "MACD histogram crosses zero in the direction of the EMA20/50 trend.", "Look for price to confirm through the signal candle with room to a known objective.", "MACD lags price and can alternate rapidly in a range."),
    "ENGULFING_REJECTION": ("Engulfing at a known level", "Reversal", "A candle body engulfs the previous opposite body near a prior swing or range boundary.", "The level supplies context; a subsequent close beyond the candle supplies confirmation.", "An engulfing body in the middle of a range is not enough. Candle direction alone does not establish a reversal."),
    "PIN_REJECTION": ("Wick rejection at a level", "Reversal", "A wick exceeds twice the body, price closes in the outer third, and a known level is nearby.", "Treat the wick as rejection evidence; wait for a later close through the signal candle.", "A wick can be liquidity probing during a continuing trend. Never treat it as automatic reversal proof."),
    "SWEEP_RECLAIM": ("Liquidity sweep and reclaim", "Reversal", "Price pierces a confirmed major swing and closes back across it.", "A sweep begins a watch. CHoCH and a later higher low/lower high strengthen the case.", "OHLCV cannot identify institutions or prove that stops caused the move."),
    "RSI_DIVERGENCE": ("Confirmed RSI divergence", "Reversal", "Price makes a new pivot extreme while RSI makes a weaker one; three right-hand bars confirm the pivot.", "The signal is known on the confirmation bar, not on the earlier pivot candle.", "Divergence can persist through several trend legs. Require price confirmation and clear invalidation."),
    "SWEEP_DIVERGENCE": ("Sweep plus RSI divergence", "Reversal", "A same-direction sweep and confirmed RSI divergence both become known within eight bars of each other.", "This combines location and momentum failure. A later structure change is stronger confirmation.", "The combination remains a reversal watch; its live probability is unvalidated."),
    "STRUCTURE_BREAK": ("Major BOS / CHoCH", "Trend", "A completed close breaks a confirmed major swing with the structure engine's ATR buffer.", "BOS usually supports continuation; CHoCH is an early change warning that needs follow-through.", "Small internal breaks and major breaks are different. A failed break is recorded separately."),
    "REVERSAL_CONFIRMED": ("Staged structure reversal", "Reversal", "A sweep is followed by CHoCH, then a higher low/lower high, then a later BOS.", "Use the new structure as the thesis and a fresh entry location as the execution decision.", "Structural confirmation can arrive after much of a move. Skip a poor reward/risk entry."),
    "FLOW_TURN": ("Price and money-flow turn", "Flow", "CMF crosses +0.05 or -0.05 while price is on the matching side of EMA20.", "Volume-weighted close location can support a price setup when source volume is available.", "CMF is a bar-volume proxy, not signed order flow or proof of institutional buying."),
    "CAPITULATION_WATCH": ("Selling-climax watch", "Flow", "A down candle spans at least 1.8 ATR, relative volume is at least 2.5, and its close is in the bottom quarter.", "Watch for a later reclaim of the entire signal candle. Heavy selling alone is not a buy signal.", "Capitulation and the start of a further sell-off can look similar. Never enter solely because volume is high."),
    "VWAP_RECLAIM": ("Session VWAP reclaim", "Flow", "Price crosses the current session's cumulative OHLCV VWAP approximation with EMA20/50 alignment.", "Use only intraday bars from a complete session history; the session resets at 18:00 New York.", "This is typical-price VWAP, not transaction VWAP. A missing constituent disables the session signal."),
    "INSIDE_BREAKOUT": ("Inside-bar breakout", "Volatility", "The previous candle sits inside its mother candle, then a close breaks the mother high or low.", "The mother range describes compression; wait for a completed close and enough room to a target.", "A break in a choppy range can fail quickly. Avoid entering after the objective has already traded."),
}


def catalog():
    return [{"kind":k,"name":v[0],"family":v[1],"rule":v[2],"use":v[3],"trap":v[4]} for k,v in CATALOG.items()]


def session_vwap(frame, timeframe):
    out = pd.Series(np.nan, index=frame.index)
    if timeframe not in ("15m", "1h", "4h"):
        return out
    previous_session = None
    total = volume = 0.0
    complete = False
    for t, row in frame.iterrows():
        local = pd.Timestamp(row.open_at).tz_convert("America/New_York")
        key = (local.tz_localize(None)-pd.Timedelta(hours=18)).date()
        if key != previous_session:
            total = volume = 0.0
            complete = local.hour == 18 and local.minute == 0
            previous_session = key
        elif row.gap_before:
            complete = False
        if not np.isfinite(row.volume) or row.volume < 0:
            complete = False
        if complete:
            total += (row.high+row.low+row.close)/3*row.volume
            volume += row.volume
            if volume > 0:
                out.loc[t] = total/volume
    return out


def detect(frame, structure, asset, timeframe, higher=None):
    """Return append-stable issues with levels derived only from then-known prices."""
    rows = list(frame.itertuples())
    times = {t.isoformat(): i for i,t in enumerate(frame.index)}
    events_by_bar, pivots_by_bar = defaultdict(list), defaultdict(list)
    for e in structure["events"]:
        if e["time"] in times:
            events_by_bar[times[e["time"]]].append(e)
    for p in structure["pivots"]:
        if p["confirmed_at"] in times:
            pivots_by_bar[times[p["confirmed_at"]]].append(p)
    vwap = session_vwap(frame, timeframe).to_numpy()
    width_cutoff = frame.bb_width.shift().rolling(60, min_periods=60).quantile(.2).to_numpy()
    if higher is not None:
        ht = pd.DataFrame({"direction":np.sign(higher.ema20-higher.ema50), "time":higher.index}, index=higher.index).reindex(frame.index, method="ffill")
    else:
        ht = None
    issues, known, last_fired, breaks, combinations = [], [], {}, [], {}
    for i,r in enumerate(rows):
        known.extend(pivots_by_bar[i])
        if i < 60 or not np.isfinite(r.atr) or r.atr <= 0 or not np.isfinite(r.ema50):
            continue
        p, mother = rows[i-1], rows[i-2]
        if any(x.gap_before or x.roll_gap_proxy for x in rows[i-2:i+1]):
            continue
        trend = int(np.sign(r.ema20-r.ema50))
        spread, body = r.high-r.low, abs(r.close-r.open)
        location = (r.close-r.low)/spread if spread>0 else .5
        nearby = lambda side, price: any(z["side"]==side and abs(z["price"]-price)<=.5*r.atr for z in known[-30:]) or abs(price-(r.donchian_low if side=="low" else r.donchian_high))<=.5*r.atr
        found = []
        if trend==1 and r.low<=r.ema20+.15*r.atr and r.close>r.ema20 and r.close>r.open and 45<r.rsi<72:
            found.append(("EMA_PULLBACK",1,"EMA20 rejection in an upward EMA20/50 trend"))
        if trend==-1 and r.high>=r.ema20-.15*r.atr and r.close<r.ema20 and r.close<r.open and 28<r.rsi<55:
            found.append(("EMA_PULLBACK",-1,"EMA20 rejection in a downward EMA20/50 trend"))
        for direction, boundary in ((1,r.donchian_high),(-1,r.donchian_low)):
            if (r.close-boundary)*direction>.1*r.atr and (p.close-boundary)*direction<=0:
                found.append(("RANGE_BREAKOUT",direction,"Close beyond the prior 20-bar range"))
                breaks.append({"i":i,"direction":direction,"level":boundary,"used":False})
        for b in breaks[-8:]:
            if b["used"] or not 1<=i-b["i"]<=8:
                continue
            d,level=b["direction"],b["level"]
            if r.low<=level+.15*r.atr and r.high>=level-.15*r.atr and (r.close-level)*d>.05*r.atr and (r.close-r.open)*d>0:
                found.append(("BREAKOUT_RETEST",d,"Retest of the original breakout boundary"))
                b["used"]=True
        if p.bb_width<=width_cutoff[i]:
            if r.close>p.bb_upper and p.close<=p.bb_upper:
                found.append(("SQUEEZE_RELEASE",1,"Close above the prior upper band after compression"))
            if r.close<p.bb_lower and p.close>=p.bb_lower:
                found.append(("SQUEEZE_RELEASE",-1,"Close below the prior lower band after compression"))
        for direction in (1,-1):
            if trend==direction and (r.rsi-50)*direction>0 and (p.rsi-50)*direction<=0:
                found.append(("RSI_RECLAIM",direction,"RSI crossed its 50 midpoint with the trend"))
            if trend==direction and r.macd_hist*direction>0 and p.macd_hist*direction<=0:
                found.append(("MACD_TURN",direction,"MACD histogram crossed zero with the trend"))
            if (r.cmf-.05*direction)*direction>0 and (p.cmf-.05*direction)*direction<=0 and (r.close-r.ema20)*direction>0:
                found.append(("FLOW_TURN",direction,"CMF crossed its directional threshold with price confirmation"))
            # Compare both closes with the VWAP known at their own completed bar.
            same_session=(pd.Timestamp(r.open_at).tz_convert("America/New_York").tz_localize(None)-pd.Timedelta(hours=18)).date()==(pd.Timestamp(p.open_at).tz_convert("America/New_York").tz_localize(None)-pd.Timedelta(hours=18)).date()
            if trend==direction and same_session and np.isfinite(vwap[i-1:i+1]).all() and (r.close-vwap[i])*direction>0 and (p.close-vwap[i-1])*direction<=0:
                found.append(("VWAP_RECLAIM",direction,"Cross of completed-bar session VWAP approximation"))
        if r.close>r.open and p.close<p.open and r.open<=p.close and r.close>=p.open and nearby("low",r.low):
            found.append(("ENGULFING_REJECTION",1,"Bullish body engulfing near a known low"))
        if r.close<r.open and p.close>p.open and r.open>=p.close and r.close<=p.open and nearby("high",r.high):
            found.append(("ENGULFING_REJECTION",-1,"Bearish body engulfing near a known high"))
        if spread>.5*r.atr and min(r.open,r.close)-r.low>max(2*body,.25*r.atr) and location>.67 and nearby("low",r.low):
            found.append(("PIN_REJECTION",1,"Long lower wick at a known level"))
        if spread>.5*r.atr and r.high-max(r.open,r.close)>max(2*body,.25*r.atr) and location<.33 and nearby("high",r.high):
            found.append(("PIN_REJECTION",-1,"Long upper wick at a known level"))
        if r.close<r.open and spread>=1.8*r.atr and r.relative_volume>=2.5 and location<.25:
            found.append(("CAPITULATION_WATCH",1,"Unusually wide, high-volume selling; reversal not yet established"))
        if p.high<mother.high and p.low>mother.low:
            if r.close>mother.high and p.close<=mother.high:
                found.append(("INSIDE_BREAKOUT",1,"Close above the inside-bar mother range"))
            if r.close<mother.low and p.close>=mother.low:
                found.append(("INSIDE_BREAKOUT",-1,"Close below the inside-bar mother range"))
        for e in events_by_bar[i]:
            kind=e["kind"]
            if kind in ("BOS","CHoCH") and e["scope"]=="major":
                found.append(("STRUCTURE_BREAK",e["direction"],"Major "+kind+" confirmed by a completed close"))
            elif kind in ("SWEEP_RECLAIM","RSI_DIVERGENCE","REVERSAL_CONFIRMED"):
                found.append((kind,e["direction"],CATALOG[kind][2]))
            if kind in ("SWEEP_RECLAIM","RSI_DIVERGENCE"):
                combinations[(kind,e["direction"])]=i
                other="RSI_DIVERGENCE" if kind=="SWEEP_RECLAIM" else "SWEEP_RECLAIM"
                if 0<=i-combinations.get((other,e["direction"]),-100)<=8:
                    found.append(("SWEEP_DIVERGENCE",e["direction"],"Both sweep and confirmed divergence are now known"))
        for kind,direction,evidence in found:
            if i-last_fired.get((kind,direction),-10)<3:
                continue
            last_fired[(kind,direction)]=i
            trigger=r.high+.05*r.atr if direction==1 else r.low-.05*r.atr
            stop=r.low-.1*r.atr if direction==1 else r.high+.1*r.atr
            candidates=[{"price":z["price"],"basis":"Confirmed swing","known_at":z["confirmed_at"]} for z in known[-60:] if z["side"]==("high" if direction==1 else "low")]
            candidates.extend({"price":price,"basis":basis,"known_at":r.Index.isoformat()} for price,basis in ((r.donchian_high if direction==1 else r.donchian_low,"Prior 20-bar extreme"),(r.ema20,"EMA20 frozen at signal"),(r.ema50,"EMA50 frozen at signal")))
            targets=[]
            for z in sorted(candidates,key=lambda x:x["price"],reverse=direction==-1):
                if np.isfinite(z["price"]) and (z["price"]-trigger)*direction>0 and all(abs(z["price"]-t["price"])>.05*r.atr for t in targets):
                    targets.append(z)
                if len(targets)==2:
                    break
            rr=[(z["price"]-trigger)*direction/abs(trigger-stop) for z in targets]
            votes={"trend":35 if trend==direction else 0,"momentum":25 if (r.rsi-50)*direction>0 else 0,"volume_flow":15 if np.isfinite(r.cmf) and r.cmf*direction>.05 else 0,"location":25 if rr and rr[0]>=1 else 0}
            score=sum(votes.values())
            higher_direction=int(ht.direction.iloc[i]) if ht is not None and np.isfinite(ht.direction.iloc[i]) else 0
            higher_time=ht.time.iloc[i].isoformat() if ht is not None and pd.notna(ht.time.iloc[i]) else None
            matches={"Adaptive":bool(score>=50 and rr and rr[0]>=1),"Strict":bool(score>=75 and higher_direction==direction and rr and rr[0]>=1.5)}
            identity=hashlib.sha256(f"{VERSION}|{asset}|{timeframe}|{kind}|{direction}|{r.Index.isoformat()}".encode()).hexdigest()[:24]
            issues.append({"id":identity,"version":VERSION,"asset":asset,"timeframe":timeframe,"kind":kind,"family":CATALOG[kind][1],"name":CATALOG[kind][0],"direction":direction,"time":r.Index.isoformat(),"price_at_signal":float(r.close),"trigger":float(trigger),"stop":float(stop),"targets":targets,"risk_reward":[float(x) for x in rr],"quality":score,"quality_components":votes,"mode_match":matches,"higher_direction":higher_direction,"higher_time":higher_time,"evidence":evidence,"probability":None,"confirmation_bars":3,"holding_bars":20,"plan_note":"Later completed close beyond trigger; next-bar open is a hypothetical entry. Levels stay fixed. No known objective means no mapped trade."})
    return issues


TERMINAL={"STOP TOUCHED","OBJECTIVE TOUCHED","EXPIRED","MISSED","INVALIDATED","POOR ENTRY","TIMEOUT"}


def replay(issue, frame, min_rr=1.0):
    """Identical rule for live observations and historical replay; never nearest-bar resolution."""
    origin=pd.Timestamp(issue["time"])
    if origin not in frame.index:
        return {"state":"UNAVAILABLE","reason":"Exact origin absent from source"}
    i=frame.index.get_loc(origin)
    if not issue["targets"]:
        return {"state":"UNMAPPED","reason":"No known reward-side objective at signal time"}
    direction,trigger,stop,target=issue["direction"],issue["trigger"],issue["stop"],issue["targets"][0]["price"]
    confirmation=None
    def stopped(r):
        return r.low<=stop if direction==1 else r.high>=stop
    def objective(r):
        return r.high>=target if direction==1 else r.low<=target
    for j in range(i+1,min(len(frame),i+1+issue["confirmation_bars"])):
        r=frame.iloc[j]
        if r.gap_before or r.roll_gap_proxy:
            return {"state":"UNAVAILABLE","reason":"Source gap interrupts the entry window"}
        if stopped(r):
            return {"state":"INVALIDATED","time":frame.index[j].isoformat(),"reason":"Invalidation traded before entry"}
        if objective(r):
            return {"state":"MISSED","time":frame.index[j].isoformat(),"reason":"Objective traded before a valid entry; do not chase"}
        if (r.close-trigger)*direction>0:
            confirmation=j
            break
    if confirmation is None:
        return {"state":"EXPIRED" if len(frame)-1-i>=issue["confirmation_bars"] else "WATCH"}
    if confirmation+1>=len(frame):
        return {"state":"CONFIRMED","time":frame.index[confirmation].isoformat(),"reason":"Completed close confirmed; next completed entry bar is not available"}
    entry_index=confirmation+1
    r=frame.iloc[entry_index]
    if r.gap_before or r.roll_gap_proxy:
        return {"state":"UNAVAILABLE","reason":"Missing entry bar"}
    entry=float(r.open)
    if (entry-stop)*direction<=0 or (target-entry)*direction<=0:
        return {"state":"MISSED","reason":"Opening price is beyond invalidation or objective"}
    risk=abs(entry-stop)
    rr=(target-entry)*direction/risk
    if rr<min_rr:
        return {"state":"POOR ENTRY","reason":"Opening reward/risk fails the selected mode"}
    result={"state":"TRIGGERED","entry_time":pd.Timestamp(r.open_at).isoformat(),"entry_bar_close":frame.index[entry_index].isoformat(),"confirmation_time":frame.index[confirmation].isoformat(),"entry":entry,"entry_rr":rr}
    for j in range(entry_index,min(len(frame),entry_index+issue["holding_bars"])):
        r=frame.iloc[j]
        if r.gap_before or r.roll_gap_proxy:
            return {**result,"state":"UNAVAILABLE","reason":"Source gap interrupts the outcome window"}
        hit_stop,hit_target=stopped(r),objective(r)
        if hit_stop or hit_target:
            exit_price=(min(stop,r.open) if direction==1 else max(stop,r.open)) if hit_stop else target
            return {**result,"state":"STOP TOUCHED" if hit_stop else "OBJECTIVE TOUCHED","time":frame.index[j].isoformat(),"exit":float(exit_price),"gross_r":float((exit_price-entry)*direction/risk),"ambiguous":bool(hit_stop and hit_target)}
    if len(frame)>=entry_index+issue["holding_bars"]:
        r=frame.iloc[entry_index+issue["holding_bars"]-1]
        return {**result,"state":"TIMEOUT","time":frame.index[entry_index+issue["holding_bars"]-1].isoformat(),"gross_r":float((r.close-entry)*direction/risk)}
    return result


def historical_audit(issues, frame):
    summaries={}
    cutoff=frame.index[int(len(frame)*.8)]
    for mode in ("Adaptive","Strict"):
        records,excluded,last_end=[],defaultdict(int),{}
        for issue in issues:
            if not issue["mode_match"][mode]:
                continue
            result=replay(issue,frame,1.5 if mode=="Strict" else 1.0)
            key=(issue["kind"],issue["direction"])
            if result["state"] not in ("STOP TOUCHED","OBJECTIVE TOUCHED","TIMEOUT"):
                excluded[result["state"]]+=1
                continue
            if pd.Timestamp(issue["time"])<cutoff<=pd.Timestamp(result["time"]):
                excluded["PARTITION_BOUNDARY"]+=1
                continue
            if issue["time"]<=last_end.get(key,""):
                excluded["OVERLAP"]+=1
                continue
            last_end[key]=result["time"]
            records.append({"id":issue["id"],"kind":issue["kind"],"direction":issue["direction"],"origin":issue["time"],"partition":"recent_holdout" if pd.Timestamp(issue["time"])>=cutoff else "earlier_history",**result})
        by_rule={}
        for kind in CATALOG:
            parts={}
            for partition in ("earlier_history","recent_holdout"):
                rs=[r for r in records if r["kind"]==kind and r["partition"]==partition]
                n=len(rs)
                hit=sum(r["state"]=="OBJECTIVE TOUCHED" for r in rs)
                p=hit/n if n else None
                ci=None
                if n:
                    z=1.96; center=(p+z*z/(2*n))/(1+z*z/n)
                    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                    ci=[max(0,center-half),min(1,center+half)]
                parts[partition]={"n":n,"tp1_first":p,"wilson95":ci,"mean_gross_r":float(np.mean([r["gross_r"] for r in rs])) if rs else None,"stops":sum(r["state"]=="STOP TOUCHED" for r in rs),"timeouts":sum(r["state"]=="TIMEOUT" for r in rs)}
            by_rule[kind]=parts
        summaries[mode]={"by_rule":by_rule,"records":records,"excluded":dict(excluded)}
    return {"version":VERSION,"status":"DESCRIPTIVE REPLAY — NOT CALIBRATED","note":"Fixed signal, later close confirmation, next-bar open, frozen TP1/stop. Stop-first on ambiguous bars; adverse stop gaps at open. No overlapping positions within a rule/direction. Final 20% shown separately; no parameter selection. Gross results exclude fees/slippage and are not a future probability.","cutoff":cutoff.isoformat(),"modes":summaries}


def workbench(issues, frame, ledger, asof, fresh, quote=None, higher_fresh=True):
    issued=ledger.setdefault("issued",{})
    observations=ledger.setdefault("observations",{})
    start=frame.index[max(0,len(frame)-100)].isoformat()
    current=[]
    for proposed in issues:
        if proposed["time"]<start:
            continue
        identity=proposed["id"]
        issued.setdefault(identity,{**deepcopy(proposed),"first_published":asof})
        fixed=issued[identity]
        prior=observations.get(identity,{})
        result=prior if prior.get("state") in TERMINAL else replay(fixed,frame)
        observations[identity]=result
        # Quote-only touches disable a new entry but do not fabricate a completed-bar outcome.
        quote_tested=False
        if quote and pd.Timestamp(quote["time"])>pd.Timestamp(fixed["time"]):
            d=fixed["direction"]
            quote_tested=(quote["price"]-fixed["stop"])*d<=0 or bool(fixed["targets"] and (quote["price"]-fixed["targets"][0]["price"])*d>=0)
        current.append({**fixed,"observation":result,"age_bars":int(len(frame)-1-frame.index.get_loc(pd.Timestamp(fixed["time"]))),"fresh":fresh,"mode_fresh":{"Adaptive":bool(fresh),"Strict":bool(fresh and higher_fresh)},"quote_tested":quote_tested,"watch_eligible":bool(fresh and not quote_tested and result["state"] in ("WATCH","CONFIRMED") and any(fixed["mode_match"].values()))})
    current.sort(key=lambda s:(s["time"],s["quality"]),reverse=True)
    return {"version":VERSION,"items":current[:36],"status_note":"Rule matches are conditional research plans, not calibrated buy/sell recommendations. Expired, touched and unmapped signals remain available for learning."}
