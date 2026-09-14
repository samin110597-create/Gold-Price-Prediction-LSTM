"""Immutable forecast issues, separate exact-bar outcomes, no history truncation."""
import hashlib
import json
def issue(ledger, asset, horizon, forecast, run_id, commit, source_hash, issued_at):
    live = forecast.get("research")
    if not live:
        return
    key = hashlib.sha256(json.dumps([asset,horizon,forecast["recipe_version"],live["origin"]]).encode()).hexdigest()[:28]
    ledger.setdefault("issued",{}).setdefault(key,{"asset":asset,"horizon":horizon,"model_version":forecast["recipe_version"],"run_id":run_id,"commit":commit,"source_hash":source_hash,"issued_at":issued_at,"horizon_bars":forecast["horizon_bars"],"unit":forecast["unit"],"status_at_issue":forecast["status"],**live})
def resolve(ledger,frames):
    outcomes=ledger.setdefault("outcomes",{})
    for key,record in ledger.get("issued",{}).items():
        if key in outcomes:
            continue
        frame=frames[record["asset"]]["1h" if record["horizon"]=="4H" else "1d"]
        # Never score a nearest bar or a target already realized when issued.
        import pandas as pd
        origin=pd.Timestamp(record["origin"])
        if origin not in frame.index:
            continue
        i=frame.index.get_loc(origin)
        j=i+record["horizon_bars"]
        if j>=len(frame) or frame.gap_before.iloc[i+1:j+1].any() or frame.roll_gap_proxy.iloc[i+1:j+1].any():
            continue
        target_time=frame.index[j]
        if target_time<=pd.Timestamp(record["issued_at"]):
            continue
        actual=float(frame.close.iloc[j])
        outcomes[key]={"target_time":target_time.isoformat(),"actual_price":actual,"direction_correct":bool((record["price"]>record["reference_price"])==(actual>record["reference_price"])),"absolute_error_percent":100*abs(record["price"]-actual)/record["reference_price"],"inside_interval80":bool(record["interval80"][0]<=actual<=record["interval80"][1])}
