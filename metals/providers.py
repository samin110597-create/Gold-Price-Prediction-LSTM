"""Quota-aware, server-side provider adapters. Only whitelisted observations leave this module.

Never persist request URLs, headers, API keys, raw responses or exception messages.
Spot, commodity quotes, ETFs and continuous futures remain separate instruments.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
import pandas as pd
import requests

SPECS = {
    'finnhub': ('FINNHUB_API_KEY', 900, 'https://finnhub.io/docs/api/quote'),
    'fmp': ('FMP_API_KEY', 900, 'https://site.financialmodelingprep.com/developer/docs/stable/commodities-quote'),
    'fred': ('FRED_API_KEY', 86400, 'https://fred.stlouisfed.org/docs/api/fred/series_observations.html'),
    'polygon': ('POLYGON_API_KEY', 86400, 'https://massive.com/docs/rest/forex/aggregates/previous-day-bar'),
    'alpha_vantage': ('ALPHA_VANTAGE_API_KEY', 21600, 'https://www.alphavantage.co/documentation/'),
}
FRED_SERIES = {'DFII10': '10-year real Treasury yield', 'DGS10': '10-year Treasury yield',
               'T10YIE': '10-year inflation breakeven', 'DTWEXBGS': 'Broad trade-weighted US dollar'}

class ProviderError(Exception):
    pass

def number(value):
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None

def timestamp(value, unit=None):
    try:
        if value is None or value == '':
            return None
        t = pd.to_datetime(value, unit=unit, utc=True)
        return t.isoformat() if pd.notna(t) else None
    except (TypeError, ValueError, OverflowError):
        return None

def request(url, params=None, headers=None):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30, allow_redirects=False)
        if r.status_code in (401,403):
            raise ProviderError('ACCESS RESTRICTED')
        if r.status_code == 429:
            raise ProviderError('RATE LIMITED')
        if r.status_code != 200:
            raise ProviderError('HTTP '+str(r.status_code))
        value = r.json()
        if isinstance(value, dict):
            # Inspect errors for classification only. Never log their contents.
            error = next((value[k] for k in ('Error Message','error','Note','Information') if value.get(k)), None)
            if error:
                words = str(error).lower()
                if any(w in words for w in ('rate','limit','frequency','25 requests')):
                    raise ProviderError('RATE LIMITED')
                if any(w in words for w in ('premium','subscription','entitle','access','api key','apikey','token')):
                    raise ProviderError('ACCESS RESTRICTED')
                raise ProviderError('PROVIDER ERROR')
            if value.get('status') in ('ERROR','NOT_AUTHORIZED'):
                raise ProviderError('ACCESS RESTRICTED')
        return value
    except ProviderError:
        raise
    except (requests.RequestException, ValueError, TypeError):
        raise ProviderError('NETWORK OR RESPONSE ERROR') from None

def quote(asset, symbol, basis, price, source_time, now, **extra):
    p = number(price)
    if p is None or p <= 0:
        raise ProviderError('NO VALID OBSERVATION')
    t = timestamp(source_time)
    if t and pd.Timestamp(t) > now + pd.Timedelta(minutes=2):
        raise ProviderError('FUTURE TIMESTAMP REJECTED')
    return dict(asset=asset, symbol=symbol, basis=basis, price=p, source_time=t,
                retrieved_at=now.isoformat(), currency='USD', **extra)

def collect_provider(name, key, now):
    observations, histories, failures = [], {}, []
    def attempt(fn):
        try:
            fn()
        except ProviderError as exc:
            failures.append(str(exc))
        except (KeyError, IndexError, TypeError, ValueError):
            failures.append('UNRECOGNIZED RESPONSE')
    for asset, metal, etf, commodity, forex in [('gold','GOLD','GLD','GCUSD','C:XAUUSD'),('silver','SILVER','SLV','SIUSD','C:XAGUSD')]:
        def fetch():
            if name == 'finnhub':
                r=request('https://finnhub.io/api/v1/quote', {'symbol':etf}, {'X-Finnhub-Token':key})
                observations.append(quote(asset,etf,'ETF proxy · USD/share',r.get('c'),timestamp(r.get('t'),unit='s'),now))
            elif name == 'fmp':
                r=request('https://financialmodelingprep.com/stable/quote',{'symbol':commodity,'apikey':key})
                r=r[0] if isinstance(r,list) and r else {}
                observations.append(quote(asset,commodity,'FMP commodity quote · contract basis not verified',r.get('price'),timestamp(r.get('timestamp'),unit='s'),now))
            elif name == 'polygon':
                r=request('https://api.massive.com/v2/aggs/ticker/'+forex+'/prev',{'adjusted':'true','apiKey':key})
                r=r.get('results',[{}])[0]
                # t is the START of a previous-day aggregate, never a live quote timestamp.
                observations.append(quote(asset,forex,'Spot USD/oz · previous-day close',r.get('c'),None,now,period_start=timestamp(r.get('t'),unit='ms')))
            elif name == 'alpha_vantage':
                # Space sequential requests for free-tier per-minute limits.
                if asset == 'silver':
                    time.sleep(15)
                r=request('https://www.alphavantage.co/query',{'function':'GOLD_SILVER_SPOT','symbol':metal,'apikey':key})
                observations.append(quote(asset,metal,'Spot USD/oz',r.get('price'),r.get('timestamp'),now))
        if name != 'fred':
            attempt(fetch)
    if name == 'fred':
        for series, title in FRED_SERIES.items():
            def fetch_fred():
                # FRED permits at most 2000 vintage dates per JSON request.
                # Five calendar years contain fewer than 2000 daily release dates.
                items=[]
                start=pd.Timestamp('2003-01-01',tz='UTC')
                while start<=now:
                    end=min(start+pd.DateOffset(years=5)-pd.Timedelta(days=1),now)
                    r=request('https://api.stlouisfed.org/fred/series/observations',{
                        'series_id':series,'api_key':key,'file_type':'json','observation_start':'2003-01-01',
                        'realtime_start':start.date().isoformat(),'realtime_end':end.date().isoformat(),
                        'output_type':4,'limit':100000})
                    items.extend(r.get('observations',[]))
                    start=end.normalize()+pd.Timedelta(days=1)
                rows=[]
                for item in items:
                    value=number(item.get('value'))
                    date=timestamp(item.get('date'))
                    released=timestamp(item.get('realtime_start'))
                    if value is None or not date or not released:
                        continue
                    # Two full UTC days cover date-only US publication timing. No revised-value fill.
                    available=(pd.Timestamp(released)+pd.Timedelta(days=2))
                    if pd.Timestamp(date)<=pd.Timestamp(released) and available<=now:
                        rows.append({'date':date,'initial_release':released,'available_at':available.isoformat(),'value':value})
                if not rows:
                    raise ProviderError('NO VINTAGE-SAFE OBSERVATIONS')
                # Keep the earliest returned release for each observation across chunks.
                rows=sorted(rows,key=lambda x:x['initial_release'])
                unique={}
                for row in rows:
                    unique.setdefault(row['date'],row)
                rows=sorted(unique.values(),key=lambda x:(x['available_at'],x['date']))
                histories[series]=rows
                last=max(rows,key=lambda x:x['date'])
                observations.append({'symbol':series,'basis':title,'value':last['value'],'source_time':last['date'],
                    'available_at':last['available_at'],'retrieved_at':now.isoformat(),'units':'index' if series=='DTWEXBGS' else 'percent',
                    'initial_release_only':True})
            attempt(fetch_fred)
    status='CONNECTED' if observations and not failures else 'PARTIAL' if observations else (failures[0] if failures else 'NO OBSERVATIONS')
    return {'status':status,'observations':observations,'histories':histories,'errors':sorted(set(failures))}

def collect(previous=None, now=None, environ=None):
    now=pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    environ=os.environ if environ is None else environ
    previous=previous or {}
    result={'schema_version':2,'asof':now.isoformat(),'providers':{}}
    for name,(env,ttl,docs) in SPECS.items():
        prior=previous.get('providers',{}).get(name,{})
        fetched=timestamp(prior.get('checked_at'))
        key=environ.get(env,'').strip()
        if not key:
            entry={'status':'KEY NOT CONFIGURED','observations':[],'histories':{},'errors':[], 'checked_at':now.isoformat()}
        elif fetched and 0<=(now-pd.Timestamp(fetched)).total_seconds()<ttl and prior.get('status')!='KEY NOT CONFIGURED' and (name!='fred' or prior.get('adapter_version')==2):
            entry=dict(prior)
        else:
            entry=collect_provider(name,key,now)
            entry['checked_at']=now.isoformat()
            # Retain explicitly old observations during outages, never relabel them current.
            if not entry['observations'] and prior.get('observations'):
                entry['observations']=prior['observations']
                entry['histories']=prior.get('histories',{})
                entry['retained_previous']=True
        entry.update(documentation=docs,refresh_seconds=ttl,secret_name=env,adapter_version=2)
        result['providers'][name]=entry
    return result

def public_summary(cache):
    return {'asof':cache['asof'],'providers':{k:{x:y for x,y in v.items() if x not in ('histories','secret_name')} for k,v in cache['providers'].items()},
            'note':'Provider timestamps and instrument bases are distinct. API access does not imply streaming or an exchange real-time entitlement. Keys stay in GitHub Actions.'}

def macro_features(index, cache):
    """As-of join to initial-release values whose conservative availability predates the origin."""
    out=pd.DataFrame(index=index)
    histories=cache.get('providers',{}).get('fred',{}).get('histories',{})
    for series,rows in histories.items():
        if series not in FRED_SERIES or not rows:
            continue
        table=pd.DataFrame(rows)
        table['available_at']=pd.to_datetime(table.available_at,utc=True)
        table['date']=pd.to_datetime(table.date,utc=True)
        # A late initial release of old data must not displace a more recent observation.
        table=table.sort_values(['available_at','date']).drop_duplicates('available_at',keep='last')
        table=table[table.date.eq(table.date.cummax())]
        joined=pd.merge_asof(pd.DataFrame({'origin':index}),table.sort_values('available_at'),left_on='origin',right_on='available_at',direction='backward')
        values=pd.Series(joined.value.to_numpy(),index=index)
        age=(pd.Series(index,index=index)-pd.Series(joined.date.to_numpy(),index=index)).dt.total_seconds()/86400
        out['macro_'+series]=values.where(age<=14)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='provider-audit.json')
    parser.add_argument('--history',default='data/provider_cache.json')
    args=parser.parse_args()
    prior=json.loads(Path(args.history).read_text()) if Path(args.history).exists() else {}
    result=collect(prior)
    Path(args.output).write_text(json.dumps(result,allow_nan=False))
    for name,item in public_summary(result)['providers'].items():
        print('PROVIDER',name,json.dumps(item,allow_nan=False),flush=True)
