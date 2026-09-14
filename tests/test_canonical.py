import copy
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from metals.indicators import wilder, compute
from metals.data import schedule, aggregate, session_context
from metals.structure import scan
from metals.setups import geometry, freeze, barrier
from metals.ledger import issue, resolve
from metals.models import evaluate
from metals.pipeline import safe

def frame(n=300,seed=42):
    rng=np.random.default_rng(seed)
    close=100+np.cumsum(rng.normal(0,.8,n))
    f=pd.DataFrame({"open":close-.1,"high":close+1,"low":close-1,"close":close,"volume":rng.integers(100,300,n),"gap_before":False,"roll_gap_proxy":False},index=pd.date_range("2023-01-01",periods=n,freq="1h",tz="UTC"))
    f["open_at"]=f.index-pd.Timedelta(hours=1)
    return f

def test_wilder_seed():
    x=pd.Series([np.nan,1,2,3,6])
    result=wilder(x,3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3]==2
    assert result.iloc[4]==(2*2+6)/3

def test_rsi_boundaries():
    f=frame(80)
    for values,expected in [(np.arange(80)+100,100),(200-np.arange(80),0),(np.full(80,100),50)]:
        f["close"]=values
        f["open"]=f.close
        f["high"]=f.close+1
        f["low"]=f.close-1
        assert compute(f).rsi.iloc[-1]==expected

def test_atr_accounts_for_gap():
    f=frame(40)
    f[["open","close"]]=100
    f["high"],f["low"]=101,99
    f.iloc[-1,f.columns.get_indexer(["open","high","low","close"])]=[110,111,109,110]
    assert compute(f).atr.iloc[-1]>2.5

def test_gold_and_silver_frozen_geometry():
    assert not geometry(1,[4383.69,4374.8],4267.67,[4404.74,4436.29,5038.2])[0]
    assert not geometry(-1,[64.84,65.78],64.99,[63.1])[0]
    assert not geometry(1,[4600,4625.5],4267.67,[4404.74,5038.2])[0]

def test_worst_entry_and_target_order():
    assert geometry(1,[101,103],98,[106,110])==(True,[.6,1.4])
    assert geometry(-1,[97,99],102,[96,90])==(True,[.2,1.4])
    assert not geometry(-1,[97,99],102,[90,96])[0]
    assert not geometry(1,[101,103],98,[102])[0]
    assert not geometry(1,[101,103],float("nan"),[106])[0]

def test_prefix_invariance_of_events():
    f=compute(frame(340))
    earlier=scan(f.iloc[:280])
    later=scan(f)
    cutoff=f.index[279].isoformat()
    # A later outcome is a separate event, never a mutation of an earlier marker.
    assert earlier["events"]==[e for e in later["events"] if e["time"]<=cutoff]
    for e in later["events"]:
        if e.get("pivot_time"):
            assert e["time"]>e["pivot_time"]

def test_sweep_is_not_automatic_reversal():
    events=scan(compute(frame(500,seed=18)))["events"]
    for e in events:
        if e["kind"]=="REVERSAL_CONFIRMED":
            before=[x for x in events if x["time"]<e["time"] and x.get("origin")==e["origin"]]
            assert any(x["kind"]=="REVERSAL_TRANSITION" for x in before)
            assert any(x["kind"]=="REVERSAL_PIVOT" for x in before)

def test_session_4h_anchor_and_short_final_bar():
    cal=schedule("gold","2026-09-08","2026-09-08")
    row=cal.iloc[0]
    starts=pd.date_range(row.market_open,row.market_close,freq="1h",inclusive="left")
    f=frame(len(starts));f.index=starts+pd.Timedelta(hours=1);f["open_at"]=starts
    out,report=aggregate(f,cal,row.market_close)
    assert len(out)==6
    assert out.index[0]==row.market_open+pd.Timedelta(hours=4)
    assert out.index[-1]==row.market_close
    assert len(f.loc[(f.open_at>=out.open_at.iloc[-1])])==3
    forming,_=aggregate(f,cal,row.market_open+pd.Timedelta(hours=5))
    assert len(forming)==1

def test_dst_and_holiday_schedule():
    cal=schedule("gold","2026-03-06","2026-03-10")
    opens=cal.market_open.dt.tz_convert("America/New_York")
    assert (opens.dt.hour==18).all()
    assert cal.market_open.iloc[0].hour!=cal.market_open.iloc[-1].hour
    holiday=schedule("silver","2026-09-07","2026-09-07").iloc[0]
    assert holiday.market_close-holiday.market_open<pd.Timedelta(hours=23)

def test_missing_constituent_is_not_a_shortened_4h_bar():
    cal=schedule("gold","2026-09-08","2026-09-08")
    row=cal.iloc[0]
    starts=pd.date_range(row.market_open,row.market_close,freq="1h",inclusive="left")
    f=frame(len(starts));f.index=starts+pd.Timedelta(hours=1);f["open_at"]=starts
    out,report=aggregate(f.drop(f.index[1]),cal,row.market_close)
    assert len(out)==5
    assert report["incomplete_groups_rejected"]==1

def test_initial_balance_never_looks_ahead():
    cal=schedule("gold","2026-09-08","2026-09-08");row=cal.iloc[0]
    starts=pd.date_range(row.market_open,row.market_close,freq="1h",inclusive="left")
    f=frame(len(starts));f.index=starts+pd.Timedelta(hours=1);f["open_at"]=starts
    context=session_context(f,cal,starts[0]+pd.Timedelta(hours=2))
    assert context["initial_four_hour_high"] is None
    assert context["high"]==f.high.iloc[:2].max()
    context=session_context(f,cal,starts[0]+pd.Timedelta(hours=4))
    assert context["initial_four_hour_high"]==f.high.iloc[:4].max()

def test_stop_first_ambiguity_and_gap():
    f=frame(1);f["open"]=94;f["high"]=107;f["low"]=93;f["close"]=101
    result=barrier(f,1,100,95,105)
    assert result["outcome"]=="STOP" and result["ambiguous"]
    assert result["exit"]==94

def test_frozen_levels_never_move():
    ledger={}
    proposal={"id":"one","mode":"Adaptive","direction":1,"entry_zone":[98,99],"stop":95,"targets":[105],"risk_reward":[1.5],"action":"WATCH","reasons":[]}
    f=frame(3);asof=f.index[-1].isoformat()
    freeze(ledger,proposal,"gold",asof,f)
    old=copy.deepcopy(ledger["issued"])
    changed={**proposal,"stop":94,"targets":[110]}
    fixed=freeze(ledger,changed,"gold",asof,f)
    assert ledger["issued"]==old and fixed["stop"]==95 and fixed["targets"]==[105]

def test_exact_forward_resolution_and_immutable_issue():
    f=frame(10)
    forecast={"recipe_version":"1","horizon_bars":2,"unit":"bars","status":"WAIT","research":{"origin":f.index[2].isoformat(),"reference_price":100,"price":101,"interval80":[90,110]}}
    ledger={}
    issue(ledger,"gold","4H",forecast,"run","sha","hash",f.index[2].isoformat())
    before=copy.deepcopy(ledger["issued"])
    forecast["research"]["price"]=110
    issue(ledger,"gold","4H",forecast,"run2","sha2","hash2",f.index[2].isoformat())
    assert ledger["issued"]==before
    missing=f.drop(f.index[2])
    resolve(ledger,{"gold":{"1h":missing}})
    assert not ledger["outcomes"]
    resolve(ledger,{"gold":{"1h":f}})
    assert len(ledger["outcomes"])==1
    assert next(iter(ledger["outcomes"].values()))["target_time"]==f.index[4].isoformat()

def test_walk_forward_purge_and_global_nonoverlap():
    f=compute(frame(2300))
    result=evaluate(f,5,hourly=False)
    records=result["records"]
    assert len(records)>50
    assert all(result["integrity"][k] for k in ("training_before_calibration","calibration_before_test","unique_origins"))
    origins=[f.index.get_loc(pd.Timestamp(r["origin"])) for r in records]
    assert all(b-a>=5 for a,b in zip(origins,origins[1:]))
    assert result["primary"] is None or all(result["checks"].values())

def test_nonfinite_serialization():
    result=safe({"a":float("nan"),"b":np.float64("inf"),"c":np.bool_(True)})
    assert result=={"a":None,"b":None,"c":True}
    json.dumps(result,allow_nan=False)
