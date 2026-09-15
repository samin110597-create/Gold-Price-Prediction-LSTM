"""One source snapshot, exchange sessions, completed bars, explicit quality reports."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests

SYMBOLS = {"gold": "GC=F", "silver": "SI=F"}
CONTEXT = {"dollar": "DX-Y.NYB", "treasury_10y": "^TNX", "equities": "^GSPC", "volatility": "^VIX", "oil": "CL=F"}
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def schedule(asset, start, end):
    return mcal.get_calendar("CMEGlobex_GC" if asset == "gold" else "CMEGlobex_SI").schedule(start_date=start, end_date=end)

def download(symbol, interval, days, asof):
    params = {"period1": int((asof-pd.Timedelta(days=days)).timestamp()), "period2": int(asof.timestamp()), "interval": interval, "events": "history"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/"+symbol, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
            response.raise_for_status()
            payload = response.json()["chart"]
            if payload.get("error") or not payload.get("result"):
                raise ValueError(str(payload.get("error")))
            return payload["result"][0]
        except Exception as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"{symbol}/{interval}: {error}")

def snapshot(folder, asof=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp(asof or pd.Timestamp.now(tz="UTC"))
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    tasks = [(a, i, s, d) for a,s in SYMBOLS.items() for i,d in [("1d", 22*366), ("60m", 729), ("15m", 59)]]
    tasks += [(name, "1d", s, 8*366) for name,s in CONTEXT.items()]
    manifest = {"asof": asof.isoformat(), "sources": {}, "errors": {}, "provider": "Yahoo Finance chart; continuous front-month futures; contract mapping unavailable"}
    def work(task):
        name, interval, symbol, days = task
        key = name+"_"+interval
        try:
            result = download(symbol, interval, days, asof)
            return key, result, None
        except Exception as exc:
            return key, None, str(exc)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for key, value, error in pool.map(work, tasks):
            if error:
                manifest["errors"][key] = error
            else:
                (folder/(key+".json")).write_text(json.dumps(value))
                manifest["sources"][key] = {"sha256": digest(value), "bars": len(value.get("timestamp", []))}
    required = {a+"_"+i for a in SYMBOLS for i in ("1d", "60m", "15m")}
    if not required.issubset(manifest["sources"]):
        raise RuntimeError("Essential sources incomplete: "+str(manifest["errors"]))
    manifest["source_hash"] = digest(manifest["sources"])
    (folder/"snapshot.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def clean(raw, asset, interval, asof):
    q = raw["indicators"]["quote"][0]
    frame = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=pd.to_datetime(raw["timestamp"], unit="s", utc=True))
    report = {"raw_bars": len(frame), "duplicates": int(frame.index.duplicated().sum())}
    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    missing = frame[["open","high","low","close"]].isna().any(axis=1)
    invalid = (~missing)&((~np.isfinite(frame[["open","high","low","close"]])).any(axis=1) | (frame[["open","high","low","close"]] <= 0).any(axis=1) | (frame.high < frame[["open","close","low"]].max(axis=1)) | (frame.low > frame[["open","close","high"]].min(axis=1)))
    report.update(missing_ohlc=int(missing.sum()), invalid_ohlc=int(invalid.sum()), missing_volume=int(frame.volume.isna().sum()))
    frame = frame.loc[~(missing|invalid)].copy()
    frame.loc[frame.volume < 0, "volume"] = np.nan
    cal = schedule(asset, frame.index.min().date()-timedelta(days=2), asof.date()+timedelta(days=10))
    if interval == "1d":
        by_date = {str(k.date()): v for k,v in cal.market_close.items()}
        opens_by_date = {str(k.date()): v for k,v in cal.market_open.items()}
        ends = pd.to_datetime([by_date.get(str(t.date()), pd.NaT) for t in frame.index], utc=True)
        frame["open_at"] = pd.to_datetime([opens_by_date.get(str(t.date()), pd.NaT) for t in frame.index], utc=True)
        frame.index = ends
        expected = pd.DatetimeIndex(cal.market_close)
    else:
        step = pd.Timedelta(minutes=15 if interval == "15m" else 60)
        starts, ends = [], []
        for row in cal.itertuples():
            for start in pd.date_range(row.market_open, row.market_close, freq=step, inclusive="left"):
                starts.append(start)
                ends.append(min(start+step, row.market_close))
        end_by_start = dict(zip(starts, ends))
        frame["open_at"] = frame.index
        frame.index = pd.to_datetime([end_by_start.get(t, pd.NaT) for t in frame.index], utc=True)
        expected = pd.DatetimeIndex(ends)
    report["outside_calendar"] = int(frame.index.isna().sum())
    frame = frame.loc[frame.index.notna()]
    report["forming_bars_excluded"] = int((frame.index > asof).sum())
    frame = frame.loc[frame.index <= asof]
    if len(frame) < 60:
        raise ValueError(f"{asset}/{interval}: insufficient completed data")
    expected = expected[(expected >= frame.index[0]) & (expected <= asof)]
    report["missing_expected_bars"] = len(expected.difference(frame.index))
    positions = expected.get_indexer(frame.index)
    frame["gap_before"] = np.r_[False, np.diff(positions) > 1]
    tr = pd.concat([frame.high-frame.low, (frame.high-frame.close.shift()).abs(), (frame.low-frame.close.shift()).abs()],axis=1).max(axis=1)
    prior_atr = tr.rolling(14).mean().shift()
    frame["roll_gap_proxy"] = ((frame.open-frame.close.shift()).abs() > 4*prior_atr) & ((frame.open/frame.close.shift()-1).abs() > .015)
    report["suspected_roll_or_price_gap_bars"] = int(frame.roll_gap_proxy.sum())
    report["last_completed"] = frame.index[-1].isoformat()
    report["expected_last_completed"] = expected[-1].isoformat() if len(expected) else None
    report["latest_expected_bar_present"] = not len(expected) or frame.index[-1] == expected[-1]
    report["roll_policy"] = "Raw continuous series; roll mapping unavailable. Gap proxies flag uncertainty; no synthetic adjustment."
    return frame, report, cal

def aggregate(hourly, calendar, asof, weekly=False, daily=None):
    source = daily if weekly else hourly
    rows = []
    if weekly:
        groups = {}
        for label, session in calendar.iterrows():
            key = str(label.to_period("W-FRI"))
            groups.setdefault(key, []).append(session.market_close)
        windows = [(min(times), max(times), times) for times in groups.values()]
    else:
        windows = []
        for row in calendar.itertuples():
            for start in pd.date_range(row.market_open, row.market_close, freq="4h", inclusive="left"):
                end = min(start+pd.Timedelta(hours=4), row.market_close)
                times = [min(t+pd.Timedelta(hours=1), end) for t in pd.date_range(start,end,freq="1h",inclusive="left")]
                windows.append((start,end,times))
    rejected = 0
    pending_gap = False
    expected_last = None
    for start,end,times in windows:
        if end > asof or end < source.index[0]:
            continue
        expected_last = end
        if not all(t in source.index for t in times):
            rejected += 1
            pending_gap = True
            continue
        part = source.loc[times]
        rows.append({"time":end, "open":part.open.iloc[0], "high":part.high.max(), "low":part.low.min(), "close":part.close.iloc[-1], "volume":part.volume.sum(min_count=len(part)), "open_at":part.open_at.iloc[0], "gap_before":bool(part.gap_before.any() or pending_gap), "roll_gap_proxy":bool(part.roll_gap_proxy.any())})
        pending_gap = False
    if not rows:
        raise ValueError("No complete aggregate bars")
    frame = pd.DataFrame(rows).set_index("time")
    return frame, {"bars":len(frame), "incomplete_groups_rejected":rejected, "last_completed":frame.index[-1].isoformat(), "expected_last_completed":expected_last.isoformat() if expected_last is not None else None, "latest_expected_bar_present":frame.index[-1] == expected_last, "anchor":"Exchange week ending Friday" if weekly else "18:00, 22:00, 02:00, 06:00, 10:00, 14:00 New York; final session bar may be shorter"}

def session_context(hourly, calendar, asof):
    selected = calendar[(calendar.market_open <= asof) & (calendar.market_close >= asof)]
    if selected.empty:
        return {"status":"MARKET CLOSED", "session_vwap":None}
    session = selected.iloc[0]
    part = hourly[(hourly.open_at >= session.market_open) & (hourly.index <= asof)]
    if part.empty:
        return {"status":"NO COMPLETED SESSION BARS", "session_vwap":None}
    full = part.open_at.iloc[0] == session.market_open and not part.gap_before.iloc[1:].any()
    volume = part.volume
    vwap = float((((part.high+part.low+part.close)/3)*volume).sum()/volume.sum()) if full and volume.notna().all() and volume.sum()>0 else None
    first = part.iloc[:4] if len(part)>=4 and full else None
    return {"status":"OPEN", "high":float(part.high.max()), "low":float(part.low.min()), "session_vwap":vwap, "vwap_note":"Completed hourly typical-price/volume approximation, reset at 18:00 New York. Not transaction VWAP.", "initial_four_hour_high":float(first.high.max()) if first is not None else None, "initial_four_hour_low":float(first.low.min()) if first is not None else None, "completed_bars":len(part)}
