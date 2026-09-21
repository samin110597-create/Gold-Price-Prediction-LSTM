"""Dated forecasts and conditional structure paths, with no invented trade probabilities."""
import pandas as pd
from metals.data import schedule

def target_time(asset, origin, horizon, hourly=False):
    origin=pd.Timestamp(origin)
    cal=schedule(asset,origin.date()-pd.Timedelta(days=2),origin.date()+pd.Timedelta(days=max(60,horizon*3)))
    if hourly:
        ends=[min(start+pd.Timedelta(hours=1),row.market_close) for row in cal.itertuples()
              for start in pd.date_range(row.market_open,row.market_close,freq='1h',inclusive='left')]
    else:
        ends=list(cal.market_close)
    future=[t for t in ends if t>origin]
    return future[horizon-1].isoformat()

def describe(result, asset, hourly, asof):
    r=result.get('research')
    if not r:
        return {'direction':'UNAVAILABLE','target_time':None}
    target=target_time(asset,r['origin'],result['horizon_bars'],hourly)
    error=result.get('holdout',{}).get('mae_percent')
    movement=r['return']
    flat_threshold=max(1e-6,(error or 0)/100*.1)
    direction='UPWARD' if movement>flat_threshold else 'DOWNWARD' if movement< -flat_threshold else 'FLAT'
    crosses=r['interval80'][0]<=r['reference_price']<=r['interval80'][1]
    return {'target_time':target,'direction':direction,'expired':pd.Timestamp(target)<=pd.Timestamp(asof),
            'reference_price':r['reference_price'],'expected_change':r['price']-r['reference_price'],
            'recent_mean_absolute_error_dollars':r['reference_price']*error/100 if error is not None else None,
            'move_to_error_ratio':abs(movement)/(error/100) if error else None,
            'flat_threshold_return':flat_threshold,
            'range_contains_no_change':crosses,
            'strength':'UNCERTAIN DIRECTION' if crosses else 'ONE-SIDED MODEL RANGE',
            'interpretation':'Point estimate is a model scenario, not a promised price or an entry order. Flat means the predicted move is less than one tenth of recent mean error (minimum 0.0001%). The model range is a historical residual band; its measured coverage is shown.'}

def paths(detail, quote):
    """Nearest two already-known structure zones in each direction. No fabricated ATR objectives."""
    levels=detail['structure'].get('levels',[])
    price=quote['price']
    above=sorted([v for v in levels if min(v['zone'])>price],key=lambda v:min(v['zone']))
    below=sorted([v for v in levels if max(v['zone'])<price],key=lambda v:max(v['zone']),reverse=True)
    bull_trigger=above[0]['zone'][1] if above else None
    bear_trigger=below[0]['zone'][0] if below else None
    # The edge of an overlapping neighbouring zone is not a valid objective.
    bull_objective=next((v['zone'][0] for v in above[1:] if v['zone'][0]>bull_trigger),None)
    bear_objective=next((v['zone'][1] for v in below[1:] if v['zone'][1]<bear_trigger),None)
    return {'timeframe':'4H','known_at':detail['last_completed'],'reference_price':price,
            'bull':{'trigger':bull_trigger,'objective':bull_objective,
                    'rule':'A completed 4H close above resistance, followed by a retest that holds, supports continuation toward the next known zone.'},
            'bear':{'trigger':bear_trigger,'objective':bear_objective,
                    'rule':'A completed 4H close below support, followed by a failed reclaim, supports continuation toward the next known zone.'},
            'base':{'lower':below[0]['zone'][1] if below else None,'upper':above[0]['zone'][0] if above else None,
                    'rule':'Between the nearest confirmed zones, direction is unresolved. Wait for a completed-bar break or a confirmed reversal.'},
            'note':'Conditional chart scenarios, not probability forecasts or published trades. A missing next zone means no defensible structural objective is available.'}
