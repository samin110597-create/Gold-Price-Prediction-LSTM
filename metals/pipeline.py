"""Build both assets from one snapshot and publish only a completely validated bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from metals import VERSION
from metals.data import SYMBOLS, CONTEXT, snapshot, clean, aggregate, session_context, digest
from metals.indicators import compute
from metals.structure import scan, families
from metals.models import evaluate, features
from metals.setups import build, freeze, geometry
from metals.ledger import issue, resolve
from metals.signal_audit import evaluate_events
from metals.trading_signals import catalog, detect, historical_audit, workbench

def safe(value):
    if isinstance(value,dict):
        return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value,(np.bool_,)):
        return bool(value)
    if isinstance(value,(np.integer,)):
        return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value
def read(path,default):
    return json.loads(path.read_text()) if path.exists() else default
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(value),indent=None if path.name=="dashboard.json" else 2,allow_nan=False))
def price_context(structure,frame):
    majors=[p for p in structure["pivots"] if p["major"]]
    result={"fibonacci":[],"patterns":[],"elliott":"UNCONFIRMED — no validated wave count","wyckoff":"OHLCV phase interpretation unavailable; volume proxies shown separately"}
    if len(majors)>=2:
        a,b=majors[-2:]
        if a["side"]!=b["side"]:
            result["fibonacci"]=[{"ratio":r,"price":b["price"]-(b["price"]-a["price"])*r,"basis":"Latest confirmed major swing leg","known_at":b["confirmed_at"]} for r in (.236,.382,.5,.618,.786)]
    if len(majors)>=3:
        a,b,c=majors[-3:]
        tolerance=.5*frame.atr.iloc[-1]
        if a["side"]==c["side"] and a["side"]!=b["side"] and abs(a["price"]-c["price"])<=tolerance:
            direction=1 if c["side"]=="low" else -1
            confirmed=(frame.close.iloc[-1]-b["price"])*direction>.08*frame.atr.iloc[-1]
            result["patterns"].append({"name":"Double bottom" if direction==1 else "Double top","state":"CONFIRMED" if confirmed else "CANDIDATE","neckline":b["price"],"objective":b["price"]+direction*abs(b["price"]-(a["price"]+c["price"])/2) if confirmed else None,"basis":"Geometric measured move; unvalidated conditional objective, not expected price","known_at":c["confirmed_at"]})
    return result
def validate(payload):
    assert payload["schema_version"]==1
    assert set(payload["assets"])=={"gold","silver"}
    for asset,a in payload["assets"].items():
        assert a["run_id"]==payload["run_id"] and a["asof"]==payload["asof"]
        assert a["quote"]["price"]>0
        assert pd.Timestamp(a["quote"]["time"])<=pd.Timestamp(payload["asof"])
        assert set(a["timeframes"])=={"15m","1h","4h","1d","1w"}
        for tf,detail in a["timeframes"].items():
            assert pd.Timestamp(detail["last_completed"])<=pd.Timestamp(payload["asof"])
            times=[r["time"] for r in detail["chart"]]
            assert times==sorted(set(times))
            assert detail["chart"][-1]["time"]==detail["last_completed"]
            for e in detail["structure"]["events"]:
                assert pd.Timestamp(e["time"])<=pd.Timestamp(detail["last_completed"])
                if e.get("pivot_time"):
                    assert e["pivot_time"]<=e["time"]
            for signal in detail.get("trading_signals",{}).get("items",[]):
                assert signal["time"]<=detail["last_completed"]
                assert signal["probability"] is None
                assert signal["trigger"]>0 and signal["stop"]>0
                assert (signal["trigger"]-signal["stop"])*signal["direction"]>0
                assert all(t["known_at"]<=signal["time"] and (t["price"]-signal["trigger"])*signal["direction"]>0 for t in signal["targets"])
                if not signal["fresh"] or signal["quote_tested"]:
                    assert not signal["watch_eligible"]
        for setup in a["setups"].values():
            if setup["id"]:
                valid,rr=geometry(setup["direction"],setup["entry_zone"],setup["stop"],setup["targets"])
                assert valid and all(x>0 for x in rr)
                assert setup["barrier_probability"] is None
        for prediction in a["forecasts"].values():
            assert all(prediction["integrity"][k] for k in ("training_before_calibration","calibration_before_test","unique_origins"))
            assert (prediction["primary"] is not None)==(prediction["status"]=="VALIDATED")
            if prediction["primary"]:
                assert all(prediction["checks"].values())
    json.dumps(safe(payload),allow_nan=False)

def run(raw_folder,stage,history):
    raw_folder,stage,history=map(Path,(raw_folder,stage,history))
    manifest=read(raw_folder/"snapshot.json",None)
    if manifest is None:
        manifest=snapshot(raw_folder)
    asof=pd.Timestamp(manifest["asof"])
    commit=os.environ.get("GITHUB_SHA","local")
    channel="production" if os.environ.get("GITHUB_REF")=="refs/heads/master" and os.environ.get("GITHUB_EVENT_NAME")!="pull_request" else "research"
    run_id=asof.strftime("%Y%m%dT%H%M%SZ")+"-"+os.environ.get("GITHUB_RUN_ID","local")
    code_hash=hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path("metals").glob("*.py")))).hexdigest()
    payload={"schema_version":1,"version":VERSION,"run_id":run_id,"asof":asof.isoformat(),"commit":commit,"code_hash":code_hash,"source_hash":manifest["source_hash"],"assets":{},"context":{},"limitations":["Yahoo continuous futures; per-bar contract mapping and revision vintages unavailable.","No exchange order book, participant identities or validated event calendar.","OHLCV liquidity and absorption labels are proxies.","Legacy model files and their frozen baseline are research history; corrected metrics are a new evaluation lineage."]}
    all_frames={}
    setup_ledger=read(history/"setup_ledger.json",{"schema_version":1,"issued":{},"observations":{}})
    signal_ledger=read(history/"signal_ledger.json",{"schema_version":1,"issued":{},"observations":{}})
    payload["signal_catalog"]=catalog()
    forecast_ledger=read(history/"forward_ledger_v2.json",{"schema_version":2,"issued":{},"outcomes":{},"legacy_note":"Original forward_validation_ledger.json preserved separately; nearest-bar legacy scores are unverified."})
    for asset,symbol in SYMBOLS.items():
        print("Building",asset,flush=True)
        frames,reports,cals={},{},{}
        for interval,tf in [("1d","1d"),("60m","1h"),("15m","15m")]:
            raw=read(raw_folder/(asset+"_"+interval+".json"),None)
            frames[tf],reports[tf],cals[tf]=clean(raw,asset,interval,asof)
        frames["4h"],reports["4h"]=aggregate(frames["1h"],cals["1h"],asof)
        frames["1w"],reports["1w"]=aggregate(frames["1h"],cals["1d"],asof,weekly=True,daily=frames["1d"])
        frames={k:compute(v) for k,v in frames.items()}
        all_frames[asset]=frames
        print("DATA DIAGNOSTIC",asset,json.dumps({k:{"bars":len(v),"feature_complete":int(features(v).notna().all(axis=1).sum()),"gaps":int(v.gap_before.sum()),"rolls":int(v.roll_gap_proxy.sum()),"cmf_missing":int(v.cmf.isna().sum()),"quality":reports[k]} for k,v in frames.items()}),flush=True)
        analyses={}
        detected={}
        higher_map={"15m":"1h","1h":"4h","4h":"1d","1d":"1w","1w":None}
        columns=["open","high","low","close","volume","ema20","ema50","bb_upper","bb_lower","rsi","macd_hist"]
        for tf in ("15m","1h","4h","1d","1w"):
            frame=frames[tf]
            structure=scan(frame,retain_all=True)
            detected[tf]=detect(frame,structure,asset,tf,frames.get(higher_map[tf]))
            replay_report=historical_audit(detected[tf],frame)
            write(stage/"validation"/(asset+"_"+tf+"_playbooks.json"),replay_report)
            chart=[{"time":t.isoformat(),**{k:r[k] for k in columns}} for t,r in frame.iloc[-150:].iterrows()]
            structure["pivots"]=structure["pivots"][-40:]
            structure["events"]=structure["events"][-60:]
            analyses[tf]={"last_completed":frame.index[-1].isoformat(),"quality":reports[tf],"indicators":{k:v for k,v in frame.iloc[-1].items() if k not in ("open_at","gap_before","roll_gap_proxy")},"structure":structure,"evidence":families(frame,structure),"chart":chart,"context":price_context(structure,frame)}
            analyses[tf]["playbook_audit"]={"status":replay_report["status"],"note":replay_report["note"],"modes":{m:v["by_rule"] for m,v in replay_report["modes"].items()},"file":"data/validation/"+asset+"_"+tf+"_playbooks.json"}
        signal_audit=evaluate_events(frames["1d"])
        write(stage/"validation"/(asset+"_signals.json"),signal_audit)
        forecasts={}
        for h,tf,bars in [("4H","1h",4),("1D","1d",1),("1W","1d",5),("1M","1d",21)]:
            data_hash=hashlib.sha256(pd.util.hash_pandas_object(frames[tf],index=True).values.tobytes()).hexdigest()
            cache_key=digest([VERSION,code_hash,data_hash,bars])
            path=history/"validation"/(asset+"_"+h+".json")
            cached=read(path,{})
            result=cached if cached.get("cache_key")==cache_key else evaluate(frames[tf],bars,hourly=tf=="1h")
            result["checks"]["latest_completed_bar"] = reports[tf]["latest_expected_bar_present"]
            result["failed_gates"] = [k for k,v in result["checks"].items() if not v]
            result["status"] = "WAIT" if result["failed_gates"] else "VALIDATED"
            result["primary"] = result["research"] if result["status"]=="VALIDATED" else None
            result["cache_key"]=cache_key
            result["model_id"]=VERSION+":"+code_hash[:12]+":"+result["recipe_version"]
            result["asset"]=asset
            result["source_hash"]=manifest["source_hash"]
            result["run_id"]=run_id
            write(stage/"validation"/(asset+"_"+h+".json"),result)
            forecasts[h]={k:v for k,v in result.items() if k!="records"}
            forecasts[h]["evidence_file"]="data/validation/"+asset+"_"+h+".json"
            issue(forecast_ledger,asset,h,result,run_id,commit,manifest["source_hash"],asof.isoformat(),channel=channel)
            print(asset,h,result["status"],"OOS",result["oos"].get("n"),"holdout",result["holdout"].get("n"),"failed",result["failed_gates"],flush=True)
        quote_frame=frames["15m"]
        quote={"price":float(quote_frame.close.iloc[-1]),"time":quote_frame.index[-1].isoformat(),"basis":"Latest completed 15m close","stale":not reports["15m"]["latest_expected_bar_present"]}
        raw=read(raw_folder/(asset+"_60m.json"),{})["meta"]
        qt=pd.Timestamp(raw.get("regularMarketTime",0),unit="s",tz="UTC")
        qp=raw.get("regularMarketPrice")
        if qp and quote_frame.index[-1]<=qt<=asof:
            quote.update(price=float(qp),time=qt.isoformat(),basis="Yahoo timestamped quote")
        for tf in analyses:
            analyses[tf]["trading_signals"]=workbench(detected[tf],frames[tf],signal_ledger,asof.isoformat(),reports[tf]["latest_expected_bar_present"],quote)
        fresh=all(reports[k]["latest_expected_bar_present"] for k in ("15m","1h","4h","1d","1w"))
        setups={mode:freeze(setup_ledger,build(asset,mode,frames,analyses,forecasts,fresh if mode=="Strict" else all(reports[k]["latest_expected_bar_present"] for k in ("15m","1h","4h","1d"))),asset,asof.isoformat(),frames["15m"]) for mode in ("Strict","Adaptive")}
        payload["assets"][asset]={"symbol":symbol,"name":asset.title(),"run_id":run_id,"asof":asof.isoformat(),"quote":quote,"fresh":fresh,"timeframes":analyses,"setups":setups,"forecasts":forecasts,"session":session_context(frames["1h"],cals["1h"],asof),"signal_audit":signal_audit["summary"],"macro_forecast":{"status":"UNAVAILABLE","reason":"No corrected, vintage-safe long-horizon model has passed validation"}}
    for name in CONTEXT:
        raw=read(raw_folder/(name+"_1d.json"),None)
        item={"status":"UNAVAILABLE"}
        if raw:
            candidates=[(t,c) for t,c in zip(raw.get("timestamp",[]),raw["indicators"]["quote"][0]["close"]) if c is not None and pd.Timestamp(t,unit="s",tz="UTC")+pd.Timedelta(hours=24)<=asof]
            if candidates:
                t,c=candidates[-1]
                item={"status":"CONTEXT ONLY","value":c,"source_time":pd.Timestamp(t,unit="s",tz="UTC").isoformat(),"available_after":(pd.Timestamp(t,unit="s",tz="UTC")+pd.Timedelta(hours=24)).isoformat(),"note":"Conservative daily lag; no vintage proof; not a model feature"}
        payload["context"][name]=item
    aligned=pd.concat([all_frames["gold"]["1d"].close,all_frames["silver"]["1d"].close],axis=1,join="inner").dropna()
    payload["context"]["gold_silver_ratio"]={"value":float(aligned.iloc[-1,0]/aligned.iloc[-1,1]),"source_time":aligned.index[-1].isoformat(),"status":"CONTEXT ONLY"}
    payload["context"]["event_calendar"]={"status":"UNAVAILABLE","note":"No verified scheduled-release feed connected"}
    resolve(forecast_ledger,all_frames)
    for asset,a in payload["assets"].items():
        ids=[key for key,r in forecast_ledger["issued"].items() if r["asset"]==asset and r.get("channel")=="production" and r.get("forward_eligible")]
        resolved=[forecast_ledger["outcomes"][key] for key in ids if key in forecast_ledger["outcomes"]]
        a["forward"]={"issued":len(ids),"resolved":len(resolved),"directional_accuracy":sum(r["direction_correct"] for r in resolved)/len(resolved) if resolved else None,"note":"Corrected immutable ledger only. Legacy results are not mixed in."}
    validate(payload)
    write(stage/"dashboard.json",payload)
    write(stage/"setup_ledger.json",setup_ledger)
    write(stage/"signal_ledger.json",signal_ledger)
    write(stage/"forward_ledger_v2.json",forecast_ledger)
    write(stage/"run_manifest.json",{"run_id":run_id,"asof":asof.isoformat(),"commit":commit,"code_hash":code_hash,"snapshot":manifest,"files":{str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob("*.json") if p.name!="run_manifest.json"}})
    print("VALIDATED coherent staged payload",run_id,flush=True)
    return payload

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",default=".snapshot")
    parser.add_argument("--stage",default=".stage")
    parser.add_argument("--history",default="data")
    args=parser.parse_args()
    run(args.raw,args.stage,args.history)
