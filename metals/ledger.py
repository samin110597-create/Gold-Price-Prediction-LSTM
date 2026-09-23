"""Immutable forecast issues, separate exact-bar outcomes, no history truncation."""
import hashlib
import json
from datetime import datetime
def issue(ledger, asset, horizon, forecast, run_id, commit, source_hash, issued_at, channel="research"):
    live = forecast.get("research")
    if not live:
        return
    latency=(datetime.fromisoformat(issued_at)-datetime.fromisoformat(live["origin"])).total_seconds()
    eligible=channel=="production" and 0<=latency<=1800
    key = hashlib.sha256(json.dumps([asset,horizon,forecast.get("model_id",forecast["recipe_version"]),live["origin"]]).encode()).hexdigest()[:28]
    ledger.setdefault("issued",{}).setdefault(key,{"asset":asset,"horizon":horizon,"model_version":forecast["recipe_version"],"model_id":forecast.get("model_id",forecast["recipe_version"]),"channel":channel,"origin_latency_seconds":latency,"forward_eligible":eligible,"run_id":run_id,"commit":commit,"source_hash":source_hash,"issued_at":issued_at,"horizon_bars":forecast["horizon_bars"],"unit":forecast["unit"],"status_at_issue":forecast["status"],**live})
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

def forward_summary(ledger,asset,horizon,model_id):
    """Evaluate the exact deployed lineage, not a pool of changing model versions."""
    import numpy as np
    entries={k:r for k,r in ledger['issued'].items() if r['asset']==asset and r['horizon']==horizon
             and r.get('model_id')==model_id and r.get('forward_eligible') and r.get('channel')=='production'}
    resolved=[(r,ledger['outcomes'][k]) for k,r in entries.items() if k in ledger['outcomes']]
    result={'issued':len(entries),'resolved':len(resolved),'model_id':model_id,
            'note':'Exact model lineage, issued within 30 minutes of origin; original estimates remain fixed. Small samples are descriptive, not proof of skill.'}
    if resolved:
        actual=np.array([o['actual_price']/r['reference_price']-1 for r,o in resolved])
        predicted=np.array([r['price']/r['reference_price']-1 for r,o in resolved])
        error=np.mean(abs(actual-predicted)); baseline=np.mean(abs(actual))
        result.update(mae_percent=float(error*100),no_change_mae_percent=float(baseline*100),
            mae_skill=float(1-error/baseline) if baseline else None,
            directional_accuracy=float(np.mean(((predicted>0)==(actual>0))&(abs(predicted)>1e-6))),
            coverage80=float(np.mean([o['inside_interval80'] for r,o in resolved])))
    return result
