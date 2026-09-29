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


def test_robust_model_is_causal_and_shrinks_small_sample_probabilities():
    f=compute(frame(1800))
    before=evaluate(f,1,recipe='robust_recent')
    assert before['recipe_version']=='5.0'
    assert len({r['fit_end'] for r in before['records'] if r['partition']=='holdout'})>1
    assert all(before['integrity'][k] for k in ('training_before_calibration','calibration_before_test','unique_origins'))
    assert all(0<r['selection']['probability_weight']<1 for r in before['records'])
    changed=f.copy()
    changed.iloc[-60:,changed.columns.get_loc('close')]*=2
    after=evaluate(changed,1,recipe='robust_recent')
    cutoff=f.index[-60].isoformat()
    assert [r for r in before['records'] if r['target_time']<cutoff]==[r for r in after['records'] if r['target_time']<cutoff]
    assert before['primary'] is None or all(before['checks'].values())


def test_bias_correction_is_causal_with_three_disjoint_calibration_blocks():
    f=compute(frame(2200))
    before=evaluate(f,1,recipe='bias_corrected')
    assert before['recipe_version']=='6.0'
    assert before['records'] and before['research']
    assert all(before['integrity'][k] for k in ('training_before_calibration','calibration_before_test','unique_origins'))
    selection=before['research']['selection']
    assert selection['bias_n']>=20 and selection['selection_n']>=20 and selection['interval_n']>=20
    assert selection['bias_end']<selection['selection_end']<before['research']['calibration_end']<before['research']['origin']
    changed=f.copy()
    changed.iloc[-60:,changed.columns.get_loc('close')]*=2
    after=evaluate(changed,1,recipe='bias_corrected')
    cutoff=f.index[-60].isoformat()
    assert [r for r in before['records'] if r['target_time']<cutoff]==[r for r in after['records'] if r['target_time']<cutoff]


def test_interval_outcomes_cannot_select_bias_correction():
    from metals.models import features,train
    f=compute(frame(1000))
    x=features(f)
    y=f.close.shift(-1)/f.close-1
    scale=f.atr/f.close
    fit=np.arange(250,700)
    cal=np.arange(710,890)
    before=train(x,y,fit,cal,scale,adaptive=True,debiased=True)
    y2=y.copy()
    y2.iloc[cal[-60:]]+=.5
    after=train(x,y2,fit,cal,scale,adaptive=True,debiased=True)
    assert before[-1]==after[-1]
    assert after[3]>before[3] and after[4]>before[4]


def test_bias_selection_reduces_known_past_bias_and_is_bounded():
    from metals.models import select_bias_correction
    raw=np.ones(100)
    actual=np.full(100,.06)
    scale=np.full(100,.1)
    s=select_bias_correction(raw,actual,scale)
    corrected=(s['ridge_weight']*raw+s['normalized_offset'])*scale
    assert np.mean(abs(actual-corrected))<np.mean(abs(actual-raw*scale))
    assert abs(s['normalized_offset'])<=.5*50/(50+50)
    neutral=select_bias_correction(raw,np.zeros(100),scale)
    assert neutral['selected_index']==0


def test_error_diagnostics_distinguish_bias_and_large_misses():
    from metals.models import metrics
    records=[]
    for actual,predicted in [(0.,.01),(.01,.03),(-.02,.05)]:
        records.append(dict(actual_return=actual,predicted_return=predicted,probability_up=.5,
            baseline_return=.001,prior_up=.5,lower_return=-.1,upper_return=.1,atr_fraction=.01))
    m=metrics(records)
    assert np.isclose(m['bias_percent'],(1+2+7)/3)
    assert m['rmse_percent']>m['mae_percent']
    assert m['p90_absolute_error_percent']>m['mae_percent']


def test_lower_average_error_cannot_hide_worse_severe_errors():
    from metals.models import promotion_checks
    good=dict(n=120,mae_improvement=.1,current_p90_error_percent=3.,previous_p90_error_percent=4.,
              current_rmse_percent=2.,previous_rmse_percent=2.5,current_brier=.24,previous_brier=.25,
              current_coverage80=.8)
    pair={p:dict(good) for p in ('walk_forward','holdout')}
    assert all(promotion_checks(pair).values())
    pair['holdout']['current_p90_error_percent']=4.1
    assert not promotion_checks(pair)['holdout_tail_error']
    pair['holdout']['current_rmse_percent']=2.6
    assert not promotion_checks(pair)['holdout_rmse']
    assert not any(promotion_checks({p:{'n':0} for p in pair}).values())
