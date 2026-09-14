"""Descriptive audit of the exact causal event rules. Never a setup probability."""
from collections import defaultdict
import numpy as np
import pandas as pd
from metals.structure import scan

def evaluate_events(frame, horizon=5):
    events=scan(frame,retain_all=True)["events"]
    groups=defaultdict(list)
    last_origin={}
    for event in events:
        kind=event["kind"]+"/"+event["scope"]
        i=frame.index.get_loc(pd.Timestamp(event["time"]))
        if i+horizon>=len(frame) or i<60 or i<=last_origin.get(kind,-horizon)+horizon-1:
            continue
        if frame.gap_before.iloc[i+1:i+horizon+1].any() or frame.roll_gap_proxy.iloc[i+1:i+horizon+1].any():
            continue
        current=float(frame.close.iloc[i])
        future=float(frame.close.iloc[i+horizon])
        direction=event["direction"]
        previous=frame.close.iloc[:i+1].pct_change(horizon).dropna()
        prior=previous.iloc[-252:]
        baseline=float((prior>0).mean()) if direction==1 else float((prior<=0).mean())
        groups[kind].append({"origin":event["time"],"target_time":frame.index[i+horizon].isoformat(),"direction":direction,"hit":bool((future>current)==(direction==1)),"signed_return":float(direction*(future/current-1)),"prior_baseline":baseline,"regime":"TREND" if frame.adx.iloc[i]>=25 else "RANGE"})
        last_origin[kind]=i
    summaries={}
    for kind,records in groups.items():
        hits=np.asarray([r["hit"] for r in records],dtype=float)
        baselines=np.asarray([r["prior_baseline"] for r in records])
        summaries[kind]={"n":len(records),"hit_rate":float(hits.mean()),"baseline":float(baselines.mean()),"edge":float((hits-baselines).mean()),"mean_signed_return":float(np.mean([r["signed_return"] for r in records])),"status":"DESCRIPTIVE ONLY" if len(records)>=30 else "SMALL SAMPLE","regimes":{regime:{"n":sum(r["regime"]==regime for r in records),"hit_rate":float(np.mean([r["hit"] for r in records if r["regime"]==regime]))} for regime in ("TREND","RANGE") if any(r["regime"]==regime for r in records)}}
    return {"horizon_bars":horizon,"formula":"Direction at confirmed-event close versus close h completed sessions later; horizons do not overlap within an event family","promotion":"None. Multiple families are descriptive; this is not entry execution or target-before-stop validation.","summary":summaries,"records":dict(groups)}
