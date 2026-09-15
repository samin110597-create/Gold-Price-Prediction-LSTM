import numpy as np
import pandas as pd
from metals.models import evaluate, paired_comparison
from metals.indicators import compute
from test_canonical import frame


def test_scaled_recipe_is_causal_and_purges_holdout_boundary():
    f=compute(frame(2600))
    out=evaluate(f,5,recipe="volatility_scaled")
    boundary=pd.Timestamp(out['data_evidence']['holdout_start'])
    assert out['records']
    for r in out['records']:
        assert r['fit_end']<r['calibration_start']<=r['calibration_end']<r['test_start']<=r['origin']<r['target_time']
        if r['partition']=='walk_forward':
            assert pd.Timestamp(r['target_time'])<boundary
    assert out['recipe_version']=='2.0'
    assert 'oos_vs_no_change' in out['checks']
    assert out['primary'] is None or all(out['checks'].values())


def test_future_volatility_cannot_change_earlier_predictions():
    f=compute(frame(2600))
    before=evaluate(f,1,recipe='volatility_scaled')
    changed=f.copy()
    changed.iloc[-100:,changed.columns.get_loc('close')]*=1.7
    changed.iloc[-100:,changed.columns.get_loc('atr')]*=4
    after=evaluate(changed,1,recipe='volatility_scaled')
    cutoff=f.index[-100].isoformat()
    a={r['origin']:r for r in before['records'] if r['target_time']<cutoff}
    b={r['origin']:r for r in after['records'] if r['target_time']<cutoff}
    assert a and a==b


def test_monthly_calibration_uses_longer_complete_history():
    f=compute(frame(4000))
    f.loc[f.index[::89],'gap_before']=True
    out=evaluate(f,21,recipe='volatility_scaled')
    assert out['research'] is not None
    assert out['research']['calibration_n']>=20
    assert out['data_evidence']['live_fit_n']>=378
    assert out['research']['calibration_end']<out['research']['origin']
    assert out['holdout']['n']>=12


def test_recipe_comparison_requires_exact_matching_partitions_and_targets():
    f=compute(frame(1800))
    a=evaluate(f,1,recipe='volatility_scaled')
    b=evaluate(f,1)
    pairs=paired_comparison(a,b)
    assert pairs['holdout']['n']>0
    b['records']=[{**r,'target_time':'2099-01-01T00:00:00+00:00'} for r in b['records']]
    assert paired_comparison(a,b)['holdout']['n']==0


def test_delayed_provider_candle_is_not_completed_by_wall_clock_alone():
    from metals.data import clean,schedule
    cal=schedule('gold','2026-09-14','2026-09-15')
    starts=pd.DatetimeIndex([t for row in cal.itertuples() for t in pd.date_range(row.market_open,row.market_close,freq='15min',inclusive='left')])[:120]
    prices=(100+np.arange(len(starts))*.01).tolist()
    quote_time=starts[-1]+pd.Timedelta(minutes=7)
    raw={'timestamp':[int(t.timestamp()) for t in starts],'meta':{'regularMarketTime':int(quote_time.timestamp())},'indicators':{'quote':[{'open':prices,'high':[p+1 for p in prices],'low':[p-1 for p in prices],'close':prices,'volume':[100]*len(starts)}]}}
    asof=starts[-1]+pd.Timedelta(minutes=17)
    out,report,_=clean(raw,'gold','15m',asof)
    assert out.index[-1]==starts[-1]
    assert report['provider_pending_bars_excluded']==1
    assert not report['latest_expected_bar_present']
    assert pd.Timestamp(report['provider_available_through'])==quote_time
