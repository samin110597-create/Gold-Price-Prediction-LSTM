import json
from pathlib import Path
from datetime import datetime, timezone

DATA=Path('data')
FILES=[DATA/'gold_technicals.json',DATA/'silver_technicals.json']

def fnum(x):
    try:
        v=float(x)
        return v if v==v else None
    except Exception:
        return None

def rr(x):
    return round(float(x),2) if x is not None else None

def enrich(path):
    if not path.exists(): return
    d=json.loads(path.read_text(encoding='utf-8'))
    rows=d.get('chart') or []
    if len(rows)<22: return
    last=fnum(rows[-1].get('close'))
    atr=fnum((d.get('indicators') or {}).get('atr14')) or (last*.015 if last else 1.0)
    prev20=rows[-21:-1]
    highs=[fnum(r.get('high')) for r in prev20]; highs=[x for x in highs if x is not None]
    lows=[fnum(r.get('low')) for r in prev20]; lows=[x for x in lows if x is not None]
    if not highs or not lows or last is None: return
    hi=max(highs); lo=min(lows); height=max(hi-lo,atr)
    mid=(hi+lo)/2
    pats=d.get('advanced_chart_patterns') or []

    # Fill missing targets on genuine advanced patterns from their own geometry.
    for p in pats:
        if p.get('target') is not None or p.get('targets'):
            p.setdefault('target_type','Measured move')
            continue
        bias=str(p.get('bias') or '').lower(); confirm=fnum(p.get('confirmation'))
        start=p.get('start'); subset=rows
        if start:
            subset=[r for r in rows if str(r.get('date',''))>=str(start)] or rows[-20:]
        sh=[fnum(r.get('high')) for r in subset]; sh=[x for x in sh if x is not None]
        sl=[fnum(r.get('low')) for r in subset]; sl=[x for x in sl if x is not None]
        ph=max(sh)-min(sl) if sh and sl else height
        ph=max(ph,atr)
        if 'bull' in bias:
            base=confirm if confirm is not None else last
            p['target']=rr(base+ph); p['target_type']='Measured move'
            p['target_basis']='Pattern height projected upward from confirmation.'
        elif 'bear' in bias:
            base=confirm if confirm is not None else last
            p['target']=rr(base-ph); p['target_type']='Measured move'
            p['target_basis']='Pattern height projected downward from confirmation.'
        else:
            p['targets']=[rr(hi+height),rr(lo-height)]
            p['target_type']='Up / Down measured move'
            p['target_basis']='Prior range height projected from either breakout boundary.'

    # If the advanced scanner has no current formation, create a defensible measured
    # range-breakout structure from the already-detected 20-day breakout/breakdown.
    if not pats:
        basic=d.get('chart_patterns') or []
        names=' | '.join(str(x.get('name','')) for x in basic).lower()
        if '20-day breakout' in names or last>hi:
            pats=[{
                'name':'20-day range breakout','bias':'Bullish','status':'Confirmed','confidence':'High',
                'structural_quality':76,'start':str(prev20[0].get('date')),'end':str(rows[-1].get('date')),
                'confirmation':rr(hi),'invalidation':rr(max(mid,hi-.5*atr)),
                'target':rr(hi+height),'target_type':'Measured move',
                'target_basis':'Previous 20-day high-low range projected above the breakout level.',
                'detail':'Confirmed close above the previous 20-day range. Target is a structural measured move, not a model forecast.',
                'segments':[]
            }]
        elif '20-day breakdown' in names or last<lo:
            pats=[{
                'name':'20-day range breakdown','bias':'Bearish','status':'Confirmed','confidence':'High',
                'structural_quality':76,'start':str(prev20[0].get('date')),'end':str(rows[-1].get('date')),
                'confirmation':rr(lo),'invalidation':rr(min(mid,lo+.5*atr)),
                'target':rr(lo-height),'target_type':'Measured move',
                'target_basis':'Previous 20-day high-low range projected below the breakdown level.',
                'detail':'Confirmed close below the previous 20-day range. Target is a structural measured move, not a model forecast.',
                'segments':[]
            }]
        else:
            # A trend is not a textbook measured-move pattern; show an objective and label it honestly.
            bull=any('bull' in str(x.get('bias','')).lower() for x in basic)
            bear=any('bear' in str(x.get('bias','')).lower() for x in basic)
            if bull and not bear:
                pats=[{'name':'Trend continuation structure','bias':'Bullish','status':'Detected','confidence':'Medium','structural_quality':58,
                       'start':str(prev20[0].get('date')),'end':str(rows[-1].get('date')),'confirmation':rr(hi),'invalidation':rr(mid),
                       'target':rr(last+.5*height),'target_type':'Structure objective','target_basis':'Half of the recent 20-day range projected from current price.',
                       'detail':'No clean textbook formation is active; this is a structure objective, not a measured-pattern forecast.','segments':[]}]
            elif bear and not bull:
                pats=[{'name':'Trend continuation structure','bias':'Bearish','status':'Detected','confidence':'Medium','structural_quality':58,
                       'start':str(prev20[0].get('date')),'end':str(rows[-1].get('date')),'confirmation':rr(lo),'invalidation':rr(mid),
                       'target':rr(last-.5*height),'target_type':'Structure objective','target_basis':'Half of the recent 20-day range projected from current price.',
                       'detail':'No clean textbook formation is active; this is a structure objective, not a measured-pattern forecast.','segments':[]}]
    d['advanced_chart_patterns']=pats
    ov=d.get('chart_overlay') or {}
    ov['patterns']=pats
    d['chart_overlay']=ov
    d['measured_move_updated_utc']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(d,indent=2),encoding='utf-8')
    print(path.name,[(p.get('name'),p.get('target'),p.get('targets')) for p in pats[:4]])

for p in FILES:
    enrich(p)
