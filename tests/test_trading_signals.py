from copy import deepcopy
import numpy as np
import pandas as pd
from metals.indicators import compute
from metals.structure import scan
from metals.trading_signals import detect, replay, session_vwap, workbench, historical_audit, catalog


def candles(n=350):
    rng=np.random.default_rng(89)
    c=100+np.cumsum(rng.normal(0,.8,n))
    index=pd.date_range("2025-01-01",periods=n,freq="1h",tz="UTC")
    return pd.DataFrame({"open":c-.2,"high":c+1,"low":c-1,"close":c,"volume":rng.integers(100,400,n),"open_at":index-pd.Timedelta(hours=1),"gap_before":False,"roll_gap_proxy":False},index=index)


def example():
    f=candles(8)
    f[["open","high","low","close"]]=[101.,102.,100.,101.]
    s={"id":"example","kind":"EMA_PULLBACK","time":f.index[0].isoformat(),"direction":1,"trigger":102,"stop":98,"targets":[{"price":108,"basis":"Known swing","known_at":f.index[0].isoformat()}],"confirmation_bars":3,"holding_bars":3,"mode_match":{"Strict":True,"Adaptive":True},"quality":85}
    return f,s


def test_issues_are_prefix_invariant_and_levels_then_known():
    f=compute(candles())
    past=f.iloc[:270]
    a=detect(past,scan(past,retain_all=True),"gold","1h")
    b=detect(f,scan(f,retain_all=True),"gold","1h")
    assert len(a)>20
    assert a==[s for s in b if s["time"]<=past.index[-1].isoformat()]
    for s in b:
        assert all(t["known_at"]<=s["time"] and (t["price"]-s["trigger"])*s["direction"]>0 for t in s["targets"])
        assert (s["trigger"]-s["stop"])*s["direction"]>0
        assert s["probability"] is None


def test_higher_timeframe_uses_completed_past_only():
    f=compute(candles())
    higher=f.iloc[::4].copy()
    higher["ema20"]=90; higher["ema50"]=100
    boundary=higher.index[-10]
    higher.loc[boundary:,"ema20"]=110
    issues=detect(f,scan(f,retain_all=True),"gold","1h",higher)
    assert all(s["higher_time"]<=s["time"] for s in issues)
    assert all(s["higher_direction"]==-1 for s in issues if s["time"]<boundary.isoformat())


def test_requires_later_close_not_intrabar_wick():
    f,s=example()
    f.loc[f.index[1],"high"]=103
    assert replay(s,f.iloc[:2])["state"]=="WATCH"
    f.loc[f.index[1],"close"]=102.5
    assert replay(s,f.iloc[:2])["state"]=="CONFIRMED"


def test_target_before_entry_is_missed_not_a_win():
    f,s=example(); f.loc[f.index[1],["high","close"]]=[109,103]
    assert replay(s,f)["state"]=="MISSED"


def test_stop_first_ambiguity_and_adverse_gap():
    f,s=example(); f.loc[f.index[1],["high","close"]]=[104,103]
    f.loc[f.index[2],["open","high","low","close"]]=[103,104,100,103]
    f.loc[f.index[3],["open","high","low","close"]]=[97,110,96,99]
    result=replay(s,f)
    assert result["state"]=="STOP TOUCHED" and result["ambiguous"]
    assert result["exit"]==97 and result["gross_r"]<-1
    assert result["entry_time"]==f.open_at.iloc[2].isoformat()


def test_signal_levels_are_immutable_after_publication():
    f,s=example(); ledger={}
    first=workbench([s],f.iloc[:1],ledger,f.index[0].isoformat(),True)
    before=deepcopy(ledger["issued"])
    changed=deepcopy(s); changed["stop"]=97; changed["targets"][0]["price"]=112
    later=workbench([changed],f.iloc[:2],ledger,f.index[1].isoformat(),True)
    assert ledger["issued"]==before
    assert later["items"][0]["stop"]==first["items"][0]["stop"]==98


def test_stale_and_quote_tested_signals_cannot_be_current_watches():
    f,s=example(); ledger={}
    assert not workbench([s],f.iloc[:1],ledger,f.index[0].isoformat(),False)["items"][0]["watch_eligible"]
    quote={"time":f.index[1].isoformat(),"price":109}
    item=workbench([s],f.iloc[:2],ledger,f.index[1].isoformat(),True,quote)["items"][0]
    assert item["quote_tested"] and not item["watch_eligible"]


def test_exact_origin_and_gap_required():
    f,s=example()
    assert replay(s,f.iloc[1:])["state"]=="UNAVAILABLE"
    f.loc[f.index[1],"gap_before"]=True
    assert replay(s,f)["state"]=="UNAVAILABLE"


def test_session_vwap_reset_and_missing_history():
    f=candles(10)
    f.index=pd.date_range("2026-09-14T23:00Z",periods=10,freq="1h")
    f["open_at"]=f.index-pd.Timedelta(hours=1)
    v=session_vwap(f,"1h")
    assert v.notna().all()
    f.loc[f.index[4],"gap_before"]=True
    assert session_vwap(f,"1h").iloc[4:].isna().all()
    assert session_vwap(f.iloc[1:],"1h").isna().all()
    assert session_vwap(f,"1d").isna().all()


def test_catalog_and_historical_audit_are_explicitly_unvalidated():
    f=compute(candles(500))
    signals=detect(f,scan(f,retain_all=True),"gold","1h")
    result=historical_audit(signals,f)
    assert len(catalog())==17
    assert "NOT CALIBRATED" in result["status"]
    for mode in result["modes"].values():
        last={}
        for r in mode["records"]:
            key=(r["kind"],r["direction"])
            assert r["origin"]>last.get(key,"")
            last[key]=r["time"]
            if r["partition"]=="earlier_history":
                assert r["time"]<result["cutoff"]


def test_rejected_aggregate_bar_leaves_a_replay_gap():
    from metals.data import aggregate, schedule
    cal=schedule("gold","2026-09-08","2026-09-08")
    day=cal.iloc[0]
    starts=pd.date_range(day.market_open,day.market_close,freq="1h",inclusive="left")
    f=candles(len(starts));f.index=starts+pd.Timedelta(hours=1);f["open_at"]=starts
    f=f.drop(f.index[5])
    out,report=aggregate(f,cal,day.market_close)
    assert report["incomplete_groups_rejected"]==1
    assert not out.gap_before.iloc[0] and out.gap_before.iloc[1]
