"""Observable market drivers and dated editorial context, never forecast overrides.

Daily comparisons are five OBSERVATIONS apart, not five calendar days. Revised
context is explicitly excluded from model features and historical backtests.
"""
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
import pandas as pd
import requests

FRED = {'DFII10': 'Real interest rates', 'T10YIE': 'Inflation expectations'}
YAHOO = {'dollar': ('US dollar', 'DX-Y.NYB'), 'treasury_10y': ('Nominal Treasury yield', '^TNX'),
         'equities': ('Equity risk appetite', '^GSPC'), 'volatility': ('Market stress', '^VIX'), 'oil': ('Energy / inflation pressure', 'CL=F')}
FEED = 'https://www.federalreserve.gov/feeds/press_monetary.xml'

def valid_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def dated_change(rows, now, max_age=5):
    """Reject future observations/availability, duplicate dates and invalid values."""
    now = pd.Timestamp(now)
    clean = {}
    for r in rows:
        try:
            date, available = pd.Timestamp(r['date']), pd.Timestamp(r['available_at'])
            if not valid_number(r['value']) or pd.isna(date) or pd.isna(available):
                continue
            if date <= available <= now:
                clean[date] = r
        except (KeyError, ValueError, TypeError):
            continue
    rows = [clean[k] for k in sorted(clean)]
    if not rows:
        return {'status': 'UNAVAILABLE', 'value': None, 'change': None}
    last = rows[-1]; prev = rows[-6] if len(rows) >= 6 else None
    stale = (now-pd.Timestamp(last['date'])).total_seconds() > max_age*86400
    return {'status': 'STALE' if stale else 'OBSERVED · LAGGED', 'value': last['value'],
            'source_time': last['date'], 'available_at': last['available_at'],
            'comparison_time': prev['date'] if prev else None,
            'change': last['value']-prev['value'] if prev else None,
            'change_percent': (last['value']/prev['value']-1)*100 if prev and prev['value'] else None,
            'comparison': 'Five available observations earlier; dates shown'}

def observations(raw):
    if not raw:
        return []
    return [{'date': pd.Timestamp(t,unit='s',tz='UTC').isoformat(),
             'available_at': (pd.Timestamp(t,unit='s',tz='UTC')+pd.Timedelta(hours=24)).isoformat(), 'value': c}
            for t,c in zip(raw.get('timestamp',[]),raw.get('indicators',{}).get('quote',[{}])[0].get('close',[]))]

def parse_releases(content, now):
    rows=[]; now=pd.Timestamp(now)
    root=ET.fromstring(content)
    for item in root.findall('./channel/item'):
        title=item.findtext('title') or ''; url=item.findtext('link') or ''
        try:
            date=pd.to_datetime(item.findtext('pubDate'),utc=True)
            if pd.isna(date) or date>now or (now-date).days>90:
                continue
        except (ValueError,TypeError):
            continue
        parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.hostname not in ('www.federalreserve.gov','federalreserve.gov'):
            continue
        rows.append({'title':title[:300],'url':url,'published_at':date.isoformat()})
    return sorted(rows,key=lambda r:r['published_at'],reverse=True)[:4]

def releases(previous, now):
    now=pd.Timestamp(now); previous=previous or {}
    checked=pd.to_datetime(previous.get('checked_at'),utc=True)
    if checked is not None and not pd.isna(checked) and 0 <= (now-checked).total_seconds()<14400:
        return previous
    try:
        r=requests.get(FEED,timeout=12,headers={'User-Agent':'MetalsResearch/1.0'},allow_redirects=False)
        r.raise_for_status()
        if len(r.content)>2000000:
            raise ValueError('response too large')
        items=parse_releases(r.content,now)
        return {'status':'AVAILABLE' if items else 'NO RECENT RELEASES','checked_at':now.isoformat(),'items':items,'source':FEED}
    except (requests.RequestException,ValueError,ET.ParseError):
        return {'status':'UNAVAILABLE','checked_at':now.isoformat(),'items':[], 'source':FEED}

def context_factors(raw_folder, cache, context, now):
    factors=[]
    for key,(name,symbol) in YAHOO.items():
        file=Path(raw_folder)/(key+'_1d.json')
        raw=json.loads(file.read_text()) if file.exists() else None
        factors.append(dict(dated_change(observations(raw),now),key=key,name=name,
                            source='Yahoo Finance · daily proxy',url='https://finance.yahoo.com/quote/'+symbol+'/',
                            units='percent yield' if key=='treasury_10y' else 'USD/barrel' if key=='oil' else 'index'))
    for series,name in FRED.items():
        fred=cache.get('providers',{}).get('fred',{})
        history=fred.get('histories',{}).get(series,[])
        revised=not bool(history)
        if revised:
            history=fred.get('context_histories',{}).get(series,[])
        # Initial releases only, or a current observation with NO invented trend.
        rows=[{'date':r['date'],'available_at':r['available_at'],'value':r['value']} for r in history]
        obs=context.get('fred_'+series,{})
        if not rows and obs.get('source_time') and obs.get('available_at'):
            rows=[{'date':obs['source_time'],'available_at':obs['available_at'],'value':obs['value']}]
        factors.append(dict(dated_change(rows,now),key=series,name=name,units='percent',
                            source='FRED · latest vintage, context only' if revised else 'FRED · initial-release observations',url='https://fred.stlouisfed.org/series/'+series))
    return factors

def explain(f, asset):
    f=dict(f); key=f['key']; change=f.get('change'); pct=f.get('change_percent')
    valid=f['status']=='OBSERVED · LAGGED' and change is not None
    impact='UNAVAILABLE' if f['status'] in ('UNAVAILABLE','STALE') else 'DIRECTION UNKNOWN'
    if key in ('dollar','DFII10','treasury_10y'):
        if valid:
            impact='HEADWIND' if change>0 else 'TAILWIND' if change<0 else 'NEUTRAL'
        meaning={'dollar':'A rising dollar generally makes dollar-priced metals more expensive for foreign buyers; a falling dollar eases that pressure.',
                 'DFII10':'Higher inflation-adjusted bond yields increase the cost of holding a metal that pays no interest. Falling real yields tend to help.',
                 'treasury_10y':'Higher nominal yields can compete with metals, but real yields are the stronger measure. Do not count both as independent signals.'}[key]
    elif key=='T10YIE':
        impact='MIXED' if valid else impact
        meaning='Higher market-implied inflation can encourage hedging, but tighter policy and rising real yields can outweigh it. Breakevens include risk and liquidity premiums; they are not CPI or a pure inflation forecast.'
    elif key=='oil':
        impact='MIXED' if valid else impact
        meaning='Rising oil can increase inflation and safe-haven demand, but can also lift rate expectations and production costs. For silver, weaker industrial activity is an additional risk.'
    elif key=='equities':
        impact=('HEADWIND' if change<0 else 'TAILWIND' if change>0 else 'NEUTRAL') if valid and asset=='silver' else 'MIXED' if valid else impact
        meaning='Equity weakness can signal growth concerns and forced selling. Gold may attract safety demand; silver also faces industrial-demand risk. The index does not measure physical metal demand.'
    else:
        impact='MIXED' if valid else impact
        meaning='Rising VIX signals equity-market stress, which may attract gold buyers but can also trigger cash raising and metal sales. It is not a geopolitical-risk measure.'
    movement=(f'{change:+.3f} percentage points' if key in ('DFII10','T10YIE','treasury_10y') else f'{pct:+.2f}%') if valid else 'Trend unavailable'
    f.update(impact=impact,movement=movement,meaning=meaning,
             caveat='Mechanism-based interpretation, not measured causation or a quantified price contribution.')
    return f

def report_cards(asset, now):
    path=Path(__file__).with_name('market_reports.json')
    reports=json.loads(path.read_text())
    result=[]; now=pd.Timestamp(now)
    for r in reports:
        if asset not in r['assets'] or now<pd.Timestamp(r['verified_at']):
            continue
        expired=now>=pd.Timestamp(r['review_after'])
        result.append({**r,'status':'REVIEW DUE · BACKGROUND ONLY' if expired else 'DATED REPORT · NOT LIVE',
                       'expired':expired})
    return result

def build_brief(asset, a, factors, now):
    rows=a['technical_brief']['rows']; relevant=[r for r in rows if r['timeframe'] in ('4H','1D')]
    aligned=len(relevant)==2 and all(r['fresh'] for r in relevant) and len({r['bias'] for r in relevant})==1
    bias=relevant[0]['bias'] if aligned else 'MIXED'
    view={'BEARISH':'Downward pressure remains the technical base case; a bounce is not yet a confirmed reversal.',
          'BULLISH':'Upward pressure remains the technical base case; pullbacks can still interrupt the trend.',
          'MIXED':'The evidence does not establish a clear direction. Treat the next move as a two-sided scenario.'}[bias]
    interpreted=[explain(f,asset) for f in factors]
    headwinds=[f['name'] for f in interpreted if f['impact']=='HEADWIND']
    tailwinds=[f['name'] for f in interpreted if f['impact']=='TAILWIND']
    daily=next((r for r in rows if r['timeframe']=='1D'),{})
    return {'asof':str(now),'bias':bias,'summary':view,'horizon':'Near-term technical view: next 1–5 trading sessions; timing is not validated.',
            'headwinds':headwinds,'tailwinds':tailwinds,'factors':interpreted,'reports':report_cards(asset,now),
            'daily_support':daily.get('support'),'daily_resistance':daily.get('resistance'),
            'difference':'Gold has direct reserve-manager demand; silver has greater exposure to manufacturing and substitution in industrial uses. Gold buying is not evidence that central banks are buying silver.',
            'limitations':['No complete real-time global-news feed or verified event calendar.',
                           'Current CFTC positioning, exchange inventories, lease rates and regional physical premiums are not connected.',
                           'Current silver ETF flows and current country-by-country reserve purchases are unavailable.',
                           'Market-factor explanations do not adjust model targets or establish causal dollar impacts.'],
            'note':'Technical direction and statistical forecasts can disagree. The model must earn an accuracy claim on unseen outcomes; extra decimal places do not make a prediction more accurate.'}
