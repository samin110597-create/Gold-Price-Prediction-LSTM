"""Small current-market refresh, independent of expensive forecast evaluation.

No training, ledger mutations, or relabeling of issued forecasts. Quotes retain
instrument and source time; completed-bar analysis uses the canonical cleaner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from metals.data import SYMBOLS, download, clean, aggregate, session_context
from metals.indicators import compute
from metals.structure import scan
from metals.outlook import technical_brief, paths
from metals.providers import collect, merge_caches, public_summary, number, timestamp
from metals.pipeline import read, write


def latest_quote(raws, symbol, asof):
    candidates=[]
    for raw in raws:
        meta=raw.get('meta',{})
        if meta.get('symbol') != symbol:
            continue
        price=number(meta.get('regularMarketPrice'))
        time=timestamp(meta.get('regularMarketTime'), unit='s')
        if price and price>0 and time and pd.Timestamp(time)<=asof:
            candidates.append({'price':price,'time':time,'symbol':symbol,
                'basis':'Yahoo continuous futures · delayed provider quote'})
    if not candidates:
        raise ValueError('No timestamped matching-instrument quote')
    return max(candidates,key=lambda q:q['time'])


def build_asset(asset, asof):
    symbol=SYMBOLS[asset]
    tasks=[('60m',90),('1d',730)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        raws=list(pool.map(lambda task:download(symbol,*task,asof),tasks))
    quote=latest_quote(raws,symbol,asof)
    frames,reports,cals={},{},{}
    for (interval,_),raw in zip(tasks,raws):
        tf='1h' if interval=='60m' else '1d'
        frames[tf],reports[tf],cals[tf]=clean(raw,asset,interval,asof)
    frames['4h'],reports['4h']=aggregate(frames['1h'],cals['1h'],asof)
    analyses={}
    for tf,frame in frames.items():
        frame=compute(frame)
        if len(frame)<100:
            raise ValueError('Insufficient technical history')
        analyses[tf]={'last_completed':frame.index[-1].isoformat(),'quality':reports[tf],
            'indicators':frame.iloc[-1].to_dict(),'structure':scan(frame)}
    return {'asof':asof.isoformat(),'symbol':symbol,'quote':quote,
        'session':session_context(frames['1h'],cals['1h'],asof),
        'technical_brief':technical_brief(analyses),'price_paths':paths(analyses['4h'],quote),
        'history_note':'90 calendar days intraday / 730 daily. Completed bars only; rolling-window structure may differ from the full-history audit.'}


def run(history, output):
    history=Path(history); asof=pd.Timestamp.now(tz='UTC')
    old=read(history/'live.json',{})
    previous=merge_caches(read(history/'provider_cache.json',{}),old.get('external_data',{}),now=asof)
    # FRED initial-vintage history and Polygon previous-day context stay in full build.
    cache=collect(previous,now=asof,names={'finnhub','fmp','alpha_vantage'})
    payload={'schema_version':1,'asof':asof.isoformat(),'assets':{},'external_data':public_summary(cache),
        'requested_refresh_seconds':300,'errors':{},
        'note':'Best-effort five-minute publication; provider delays and GitHub scheduling can add latency. No guaranteed real-time feed.'}
    for asset in SYMBOLS:
        try:
            item=build_asset(asset,asof)
            prior=old.get('assets',{}).get(asset,{})
            if prior and pd.Timestamp(item['quote']['time'])<pd.Timestamp(prior['quote']['time']):
                raise ValueError('Source timestamp regressed')
            payload['assets'][asset]=item
        except Exception:
            # Never print raw provider exceptions (URLs may contain credentials).
            payload['errors'][asset]='Refresh failed; last observation retained with its original timestamp.'
            if asset in old.get('assets',{}):
                payload['assets'][asset]=old['assets'][asset]
    if not payload['assets']:
        raise RuntimeError('No usable metals observations; publication withheld')
    write(Path(output),payload)
    for asset,item in payload['assets'].items():
        print('CURRENT MARKET',asset,'source',item['quote']['time'],'analysis',item['asof'])
    for name,item in cache['providers'].items():
        print('QUOTE PROVIDER',name,item['status'],'checked',item['checked_at'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',default='data');p.add_argument('--output',default='data/live.json')
    a=p.parse_args();run(a.history,a.output)
