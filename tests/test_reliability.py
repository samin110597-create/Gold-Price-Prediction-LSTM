import numpy as np
from copy import deepcopy
from metals.reliability import error_evidence, interval_score
from metals.models import select_bias_correction, evaluate, promotion_checks
from metals.indicators import compute
from metals.ledger import forward_summary
from test_canonical import frame


def test_paired_evidence_refuses_small_samples_and_detects_worse_forecasts():
    assert not error_evidence([.01]*29,[0]*29)['skill_established']
    strong=error_evidence([.02]*60,[.019]*60)
    assert strong['skill_established'] and strong['mae_gain_ci95'][0]>0
    worse=error_evidence([.02]*60,[.06]*60)
    assert not worse['skill_established'] and worse['mae_gain_ci95'][1]<0
    assert error_evidence([],[])['n']==0


def test_interval_score_penalizes_huge_bands_and_missed_tails():
    assert interval_score([0],[-.1],[.1])[0] < interval_score([0],[-10],[10])[0]
    assert np.isclose(interval_score([.2],[-.1],[.1])[0],1.2)


def test_guarded_selector_rejects_a_gain_that_reverses_in_later_subperiod():
    # Stable earlier returns, then a collapse in the later selection subperiod.
    raw=np.ones(120); scale=np.ones(120); actual=np.r_[np.full(90,.8),np.full(30,.05)]
    unguarded=select_bias_correction(raw,actual,scale)
    guarded=select_bias_correction(raw,actual,scale,guarded=True)
    assert unguarded['ridge_weight']>0
    assert guarded['selected_index']==0
    stable=select_bias_correction(raw,np.full(120,.8),scale,guarded=True)
    assert stable['ridge_weight']>0


def test_new_recipe_does_not_learn_from_future_outcomes():
    f=compute(frame(2200))
    before=evaluate(f,1,recipe='guarded_shrinkage')
    changed=f.copy(); changed.iloc[-60:,changed.columns.get_loc('close')]*=2
    after=evaluate(changed,1,recipe='guarded_shrinkage')
    cutoff=f.index[-60].isoformat()
    assert before['recipe_version']=='7.0' and before['research']
    assert [r for r in before['records'] if r['target_time']<cutoff]==[r for r in after['records'] if r['target_time']<cutoff]


def test_forward_confidence_does_not_count_overlapping_bets_as_independent():
    import pandas as pd
    start=pd.Timestamp('2026-09-01T00:00:00Z'); ledger={'issued':{},'outcomes':{}}
    for i in range(40):
        ledger['issued'][str(i)]={'asset':'gold','horizon':'4H','model_id':'m','channel':'production','forward_eligible':True,'origin':(start+pd.Timedelta(hours=i)).isoformat(),'reference_price':100,'price':101}
        ledger['outcomes'][str(i)]={'target_time':(start+pd.Timedelta(hours=i+4)).isoformat(),'actual_price':101,'inside_interval80':True}
    before=deepcopy(ledger)
    s=forward_summary(ledger,'gold','4H','m')
    assert s['resolved']==40 and s['nonoverlapping_resolved']==10
    assert not s['independent_evidence']['skill_established']
    assert ledger==before


def test_better_average_error_is_not_enough_to_promote_without_confidence():
    row=dict(n=120,mae_improvement=.1,current_p90_error_percent=1.,previous_p90_error_percent=2.,current_rmse_percent=1.,previous_rmse_percent=2.,current_brier=.2,previous_brier=.25,current_coverage80=.8,current_interval_score80_percent=2.,previous_interval_score80_percent=3.,paired_error_evidence={'skill_established':False})
    checks=promotion_checks({p:row for p in ('walk_forward','holdout')})
    assert checks['holdout_price_improvement'] and not checks['holdout_paired_confidence']


def test_shadow_forecasts_remain_separate_from_production():
    from metals.ledger import issue
    from datetime import datetime, timezone
    at=datetime(2026,10,18,tzinfo=timezone.utc).isoformat()
    f={'recipe_version':'7.0','model_id':'candidate','horizon_bars':4,'unit':'bars','status':'RESEARCH','research':{'origin':at,'reference_price':100,'price':101,'interval80':[90,110]}}
    ledger={}
    issue(ledger,'gold','4H',f,'run','commit','hash',at,channel='shadow')
    record=next(iter(ledger['issued'].values()))
    assert record['forward_eligible'] and record['channel']=='shadow'
    assert forward_summary(ledger,'gold','4H','candidate')['issued']==0
    assert forward_summary(ledger,'gold','4H','candidate',channel='shadow')['issued']==1
