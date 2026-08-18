import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from multi_asset_forecast_v2 import ASSETS

DATA=Path('data');DATA.mkdir(exist_ok=True);LEDGER=DATA/'forward_validation_ledger.json'


def flatten(df):
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0] for c in df.columns]
    return df


def history(ticker,interval):
    period='2y' if interval=='1h' else '10y'
    d=flatten(yf.download(ticker,period=period,interval=interval,auto_adjust=False,progress=False,threads=False))
    if d.empty or 'Close' not in d.columns:return pd.DataFrame()
    d=d.dropna(subset=['Close']).copy();idx=pd.to_datetime(d.index)
    if getattr(idx,'tz',None) is not None:idx=idx.tz_convert('UTC').tz_localize(None)
    d.index=idx;return d.sort_index()


def load(path):
    try:return json.loads(Path(path).read_text())
    except:return {}


def find_position(idx,issued):
    ts=pd.Timestamp(issued)
    if getattr(ts,'tzinfo',None) is not None:ts=ts.tz_convert('UTC').tz_localize(None)
    pos=int(idx.searchsorted(ts,side='left'))
    if pos>=len(idx):return None
    if pos>0 and abs(idx[pos]-ts)>abs(idx[pos-1]-ts):pos-=1
    return pos


def mature(entries,series_map):
    for e in entries:
        if e.get('resolved'):continue
        d=series_map.get((e['asset'],e['interval']))
        if d is None or d.empty:continue
        pos=find_position(d.index,e['issued_market_time'])
        if pos is None:continue
        target=pos+int(e['steps'])
        if target>=len(d):continue
        actual=float(d.Close.iloc[target]);start=float(e['start_price']);pred=float(e['predicted_price'])
        e['resolved']=True;e['resolved_market_time']=d.index[target].isoformat();e['actual_price']=round(actual,4)
        e['direction_correct']=bool((pred-start>=0)==(actual-start>=0));e['abs_error_pct']=round(abs(pred-actual)/max(abs(actual),1e-9)*100,4)


def summarize(entries):
    out={}
    for e in entries:
        if not e.get('resolved'):continue
        k=f"{e['asset']}|{e['horizon']}";g=out.setdefault(k,{'asset':e['asset'],'horizon':e['horizon'],'resolved':0,'direction_hits':0,'errors':[]})
        g['resolved']+=1;g['direction_hits']+=int(bool(e.get('direction_correct')));g['errors'].append(float(e.get('abs_error_pct') or 0))
    rows=[]
    for g in out.values():
        rows.append({'asset':g['asset'],'horizon':g['horizon'],'resolved_forecasts':g['resolved'],'forward_direction_accuracy':round(g['direction_hits']/g['resolved'],4),'forward_mae_pct':round(float(np.mean(g['errors'])),4)})
    return sorted(rows,key=lambda x:(x['asset'],x['horizon']))


def append_if_new(entries,entry):
    key=(entry['asset'],entry['horizon'],entry['issued_market_time'],entry['source'])
    if any((e.get('asset'),e.get('horizon'),e.get('issued_market_time'),e.get('source'))==key for e in entries):return
    entries.append(entry)


def main():
    payload=load(LEDGER);entries=payload.get('entries',[]);series={}
    for asset,cfg in ASSETS.items():
        series[(asset,'1h')]=history(cfg['ticker'],'1h');series[(asset,'1d')]=history(cfg['ticker'],'1d')
    mature(entries,series)
    now=datetime.now(timezone.utc).isoformat()
    for asset,cfg in ASSETS.items():
        h1=series[(asset,'1h')];d1=series[(asset,'1d')]
        v6=load(DATA/f'{asset}_future_price_action_v6.json');h4=load(DATA/f'{asset}_4h_specialist.json');macro=load(DATA/f'{asset}_macro_1y.json')
        if not h1.empty and h4.get('predicted_price') is not None:
            append_if_new(entries,{'asset':asset,'horizon':'4H','source':'Recent 4H specialist','interval':'1h','steps':4,'issued_utc':now,'issued_market_time':h1.index[-1].isoformat(),'start_price':round(float(h1.Close.iloc[-1]),4),'predicted_price':float(h4['predicted_price']),'probability_up':h4.get('probability_up'),'model_status':h4.get('overall_status'),'resolved':False})
        if not d1.empty:
            for z in v6.get('horizons',[]):
                append_if_new(entries,{'asset':asset,'horizon':z.get('horizon'),'source':'Future Price Action V6','interval':'1d','steps':int(z.get('steps') or 1),'issued_utc':now,'issued_market_time':d1.index[-1].isoformat(),'start_price':round(float(d1.Close.iloc[-1]),4),'predicted_price':float(z.get('predicted_price')),'probability_up':z.get('probability_up'),'model_status':z.get('status'),'regime':z.get('current_regime'),'resolved':False})
            mf=macro.get('macro_forecast',{})
            if mf.get('predicted_price') is not None:
                append_if_new(entries,{'asset':asset,'horizon':'1 Year','source':'Macro 1Y challenger','interval':'1d','steps':252,'issued_utc':now,'issued_market_time':d1.index[-1].isoformat(),'start_price':round(float(d1.Close.iloc[-1]),4),'predicted_price':float(mf['predicted_price']),'probability_up':mf.get('probability_up'),'model_status':'PASS' if macro.get('selected_as_champion') else 'FAIL','resolved':False})
    entries=entries[-3000:]
    out={'updated_utc':now,'method':'Predictions are logged before outcomes exist. A forecast is resolved only when the required future market observations become available. This is forward validation, not backtest tuning.','summary':summarize(entries),'entries':entries}
    LEDGER.write_text(json.dumps(out,indent=2));print('forward ledger',len(entries),'entries',sum(bool(e.get('resolved')) for e in entries),'resolved')

if __name__=='__main__':main()
