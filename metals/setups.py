"""Immutable conditional plans; a scored confluence does not establish barrier probability."""
import hashlib
import json
import numpy as np

def geometry(direction, zone, stop, targets):
    if direction not in (-1,1) or len(zone)!=2 or not all(np.isfinite(v) and v>0 for v in [*zone,stop,*targets]):
        return False, []
    lo, hi = zone
    if lo>hi or (direction==1 and stop>=lo) or (direction==-1 and stop<=hi):
        return False, []
    entry = hi if direction==1 else lo
    if not targets or any((t-entry)*direction<=0 for t in targets):
        return False, []
    if targets != sorted(set(targets),reverse=direction==-1):
        return False, []
    return True, [round((t-entry)*direction/abs(entry-stop),2) for t in targets]

def build(asset, mode, frames, analyses, forecasts, fresh):
    current = frames["1h"].close.iloc[-1]
    a = analyses["4h"]
    direction = a["structure"]["direction"]
    atr = frames["4h"].atr.iloc[-1]
    out = {"mode":mode,"action":"WAIT","direction":direction,"entry_zone":None,"stop":None,"targets":[],"risk_reward":[],"barrier_probability":None,"barrier_validation":"UNVALIDATED — current Strict/Adaptive recipe has no independent target-before-stop audit","reasons":[],"id":None}
    if direction==0:
        out["reasons"]=["Major structure has no confirmed direction"]
        return out
    levels = a["structure"]["levels"]
    supports = [z for z in levels if z["price"]<current]
    resistances = [z for z in levels if z["price"]>current]
    origins = supports if direction==1 else resistances
    objectives = resistances if direction==1 else supports
    if not origins or not objectives:
        out["reasons"]=["Insufficient confirmed swings for an entry and objective"]
        return out
    anchor = max(origins,key=lambda z:z["price"]) if direction==1 else min(origins,key=lambda z:z["price"])
    zone = anchor["zone"]
    stop = zone[0]-.15*atr if direction==1 else zone[1]+.15*atr
    reference = zone[1] if direction==1 else zone[0]
    targets = sorted({z["price"] for z in objectives if (z["price"]-reference)*direction>0},reverse=direction==-1)[:3]
    valid, rr = geometry(direction,zone,stop,targets)
    if not valid:
        out["reasons"]=["Incomplete or invalid risk map"]
        return out
    # Fixed identity uses the structural origin, not a moving ATR or latest quote.
    identity = [asset,mode,direction,anchor["confirmed_at"]]
    out.update(id=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24],entry_zone=zone,stop=stop,targets=targets,risk_reward=rr,trigger="Completed 1H rejection from entry zone in the indicated direction",invalidation_rule="Stop touch; same-bar target/stop ambiguity is resolved stop-first",target_basis="Observed confirmed major swings; conditional objectives, not model forecasts",action="WATCH")
    out["reasons"].append("Await a rejection from the fixed entry zone")
    if not fresh:
        out["action"]="WAIT"
        out["reasons"].append("One or more required timeframes lack the latest completed bar")
    if mode=="Strict":
        aligned = all(analyses[k]["structure"]["direction"]==direction for k in ("4h","1d","1w"))
        validated = all(forecasts[h]["status"]=="VALIDATED" for h in ("4H","1D","1W"))
        if not aligned or not validated:
            out["action"]="WAIT"
            out["reasons"].append("Strict requires 4H/daily/weekly structure alignment and validated forecasts")
    if rr[0]<1:
        out["action"]="WAIT"
        out["reasons"].append("First objective offers less than 1:1 reward/risk")
    # Executable entries require independent barrier validation. WATCH is intentionally not BUY/SELL.
    out["reasons"].append("No validated target-before-stop probability; no executable trade recommendation")
    return out

def freeze(ledger, proposed, asset, asof, frame):
    identity = proposed.get("id")
    if not identity:
        return proposed
    issued = ledger.setdefault("issued", {})
    if identity not in issued:
        issued[identity] = {"asset":asset,"issued_at":asof,**proposed}
    fixed = dict(issued[identity])
    bars = frame[frame.index > issued[identity]["issued_at"]]
    observations = ledger.setdefault("observations", {}).setdefault(identity,{"state":"UNTOUCHED","target_touches":[]})
    if observations["state"]!="INVALIDATED":
        direction = fixed["direction"]
        for t,r in bars.iterrows():
            stopped = r.low <= fixed["stop"] if direction==1 else r.high >= fixed["stop"]
            if stopped:
                observations.update(state="INVALIDATED",time=t.isoformat())
                break
            for n,target in enumerate(fixed["targets"]):
                touched = r.high>=target if direction==1 else r.low<=target
                if touched and n not in observations["target_touches"]:
                    observations["target_touches"].append(n)
                    observations.update(state="OBJECTIVE TOUCHED",time=t.isoformat())
    # Presentation gates can tighten with fresh evidence; historical levels never change.
    fixed["action"] = "WAIT" if observations["state"]!="UNTOUCHED" else proposed["action"]
    fixed["reasons"] = proposed["reasons"] + ([observations["state"]+"; await a new structure origin"] if observations["state"]!="UNTOUCHED" else [])
    fixed["observation"] = observations
    return fixed

def barrier(bars, direction, entry, stop, target):
    """Next-open execution helper for future recipe audits; conservative OHLC ambiguity."""
    mfe = mae = 0.0
    for t,r in bars.iterrows():
        mfe = max(mfe,(r.high-entry) if direction==1 else (entry-r.low))
        mae = max(mae,(entry-r.low) if direction==1 else (r.high-entry))
        hit_stop = r.low<=stop if direction==1 else r.high>=stop
        hit_target = r.high>=target if direction==1 else r.low<=target
        if hit_stop:
            exit_price = min(stop,r.open) if direction==1 else max(stop,r.open)
            return {"outcome":"STOP","exit":exit_price,"time":str(t),"ambiguous":bool(hit_target),"mfe_upper_bound":mfe,"mae_upper_bound":mae}
        if hit_target:
            return {"outcome":"TARGET","exit":target,"time":str(t),"ambiguous":False,"mfe":mfe,"mae":mae}
    return {"outcome":"TIMEOUT","exit":float(bars.close.iloc[-1]) if len(bars) else entry,"mfe":mfe,"mae":mae}
