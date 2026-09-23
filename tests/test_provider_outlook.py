import json
import pandas as pd
import numpy as np
import requests
import pytest
from metals import providers
from metals.outlook import target_time, describe, paths
from metals.models import evaluate
from metals.indicators import compute
from test_canonical import frame

def test_secrets_and_raw_error_text_never_leave_adapter(monkeypatch):
    monkeypatch.setattr(providers.time,'sleep',lambda _:None)
    key='TEST-CREDENTIAL-DO-NOT-PUBLISH'
    class Response:
        status_code=200
        def json(self):
            return {'error':'invalid api key '+key}
    monkeypatch.setattr(providers.requests,'get',lambda *a,**kw:Response())
    data=providers.collect(now='2026-09-16T13:00Z',environ={spec[0]:key for spec in providers.SPECS.values()})
    assert key not in json.dumps(data)
    assert all(v['status']=='ACCESS RESTRICTED' for v in data['providers'].values())
    def failed(*args,**kwargs):
        raise requests.ConnectionError('https://example.invalid/?key='+key)
    monkeypatch.setattr(providers.requests,'get',failed)
    assert key not in json.dumps(providers.collect(environ={'FMP_API_KEY':key}))

def test_quota_cache_does_not_repeat_requests_or_mark_old_data_new(monkeypatch):
    calls=[]
    def fetch(name,key,now):
        calls.append(name)
        return {'status':'CONNECTED','observations':[{'retrieved_at':now.isoformat()}],'histories':{},'errors':[]}
    monkeypatch.setattr(providers,'collect_provider',fetch)
    first=providers.collect(now='2026-09-16T00:00Z',environ={'ALPHA_VANTAGE_API_KEY':'test'})
    again=providers.collect(first,now='2026-09-16T05:59Z',environ={'ALPHA_VANTAGE_API_KEY':'test'})
    assert calls==['alpha_vantage']
    assert again['providers']['alpha_vantage']['checked_at']==first['providers']['alpha_vantage']['checked_at']
    providers.collect(again,now='2026-09-16T06:00Z',environ={'ALPHA_VANTAGE_API_KEY':'test'})
    assert len(calls)==2

def test_fred_values_cannot_enter_before_release_or_after_stale_cutoff():
    index=pd.date_range('2026-01-01',periods=25,tz='UTC')
    rows=[{'date':'2026-01-01T00:00Z','available_at':'2026-01-05T00:00Z','value':2.1},
          {'date':'2026-01-02T00:00Z','available_at':'2026-01-10T00:00Z','value':2.2}]
    cache={'providers':{'fred':{'histories':{'DFII10':rows}}}}
    values=providers.macro_features(index,cache).macro_DFII10
    assert values.iloc[:4].isna().all()
    assert (values.iloc[4:9]==2.1).all()
    assert values.iloc[9]==2.2
    rows[-1]['value']=999
    changed=providers.macro_features(index,cache).macro_DFII10
    pd.testing.assert_series_equal(values.iloc[:9],changed.iloc[:9])
    assert changed.iloc[-1:].isna().all()

def test_previous_day_spot_bar_is_not_a_live_quote(monkeypatch):
    monkeypatch.setattr(providers,'request',lambda *a,**k:{'results':[{'c':4000,'t':1757908800000}]})
    result=providers.collect_provider('polygon','test',pd.Timestamp('2026-09-16T00:00Z'))
    assert all(o['source_time'] is None and o['period_start'] and 'previous-day' in o['basis'] for o in result['observations'])

def test_provider_rejects_future_or_invalid_quotes():
    now=pd.Timestamp('2026-01-01T00:00Z')
    for price,date in [(0,None),(float('inf'),None),(100,'2030-01-01')]:
        with pytest.raises(providers.ProviderError):
            providers.quote('gold','GOLD','spot',price,date,now)

def test_forecast_targets_follow_sessions_across_weekend():
    assert target_time('gold','2026-09-18T21:00Z',1)=='2026-09-21T21:00:00+00:00'
    assert target_time('silver','2026-09-18T20:00Z',4,True)=='2026-09-21T01:00:00+00:00'

def test_direction_is_uncertain_when_range_spans_both_sides():
    result={'horizon_bars':1,'research':{'origin':'2026-09-15T21:00Z','reference_price':100,'price':101,'return':.01,'interval80':[97,105]},'holdout':{'mae_percent':2}}
    out=describe(result,'gold',False,'2026-09-16T12:00Z')
    assert out['direction']=='UPWARD' and out['strength']=='UNCERTAIN DIRECTION'
    assert out['recent_mean_absolute_error_dollars']==2
    assert not out['expired']
    assert describe(result,'gold',False,'2026-09-17T12:00Z')['expired']

def test_paths_only_use_already_confirmed_structure_levels():
    detail={'last_completed':'2026-09-16T10:00Z','structure':{'levels':[{'zone':[90,92]},{'zone':[95,97]},{'zone':[103,105]},{'zone':[108,110]}]}}
    out=paths(detail,{'price':100})
    assert out['bull']['trigger']==105 and out['bull']['objective']==108
    assert out['bear']['trigger']==95 and out['bear']['objective']==92
    assert paths({'last_completed':'2026-09-16','structure':{'levels':[]}}, {'price':100})['bull']['objective'] is None

def test_adaptive_selector_never_sees_future_labels():
    f=compute(frame(2300))
    before=evaluate(f,1,recipe='history_selected')
    altered=f.copy()
    altered.iloc[-100:,altered.columns.get_loc('close')]*=1.7
    after=evaluate(altered,1,recipe='history_selected')
    cutoff=f.index[-100].isoformat()
    a={r['origin']:r for r in before['records'] if r['target_time']<cutoff}
    b={r['origin']:r for r in after['records'] if r['target_time']<cutoff}
    assert a and a==b
    assert all(r['selection']['interval_n']>=10 for r in before['records'])
    assert before['research']['selection']['selection_end']<before['research']['origin']

def test_overlapping_zones_cannot_produce_wrong_side_objectives():
    zones=[[103,108],[105,110],[112,117],[93,98],[91,96],[84,89]]
    detail={'last_completed':'2026-09-18T21:00Z','structure':{'levels':[{'zone':z} for z in zones]}}
    out=paths(detail,{'price':100})
    assert out['bull']['trigger']==108 and out['bull']['objective']==112
    assert out['bear']['trigger']==93 and out['bear']['objective']==89
    detail['structure']['levels']=[{'zone':z} for z in zones if z not in ([112,117],[84,89])]
    out=paths(detail,{'price':100})
    assert out['bull']['objective'] is None and out['bear']['objective'] is None

def test_negligible_predicted_move_is_flat_relative_to_measured_error():
    result={'horizon_bars':1,'research':{'origin':'2026-09-18T21:00Z','reference_price':4424.9,'price':4424.76,'return':-.14/4424.9,'interval80':[4345,4520]},'holdout':{'mae_percent':1.3}}
    assert describe(result,'gold',False,'2026-09-20T15:00Z')['direction']=='FLAT'


def test_fred_requests_split_vintages_and_keep_earliest_release(monkeypatch):
    calls=[]
    def fetch(url,params):
        calls.append(params)
        assert (pd.Timestamp(params['realtime_end'])-pd.Timestamp(params['realtime_start'])).days<2000
        return {'observations':[{'date':'2020-01-01','realtime_start':'2020-01-03','value':'1.5'},
                                {'date':'2020-01-01','realtime_start':'2021-01-03','value':'9.9'}]}
    monkeypatch.setattr(providers,'request',fetch)
    r=providers.collect_provider('fred','test',pd.Timestamp('2026-09-23T12:00Z'))
    assert r['status']=='CONNECTED' and len(r['observations'])==4 and len(calls)==20
    assert all(v['value']==1.5 for v in r['observations'])


def test_recent_technical_forecasts_are_causal():
    f=compute(frame(2300)); before=evaluate(f,1,recipe='recent_technical')
    changed=f.copy(); changed.iloc[-100:,changed.columns.get_loc('close')]*=1.7
    after=evaluate(changed,1,recipe='recent_technical')
    cutoff=f.index[-100].isoformat()
    a=[r for r in before['records'] if r['target_time']<cutoff]
    b=[r for r in after['records'] if r['target_time']<cutoff]
    assert a and a==b
    assert before['recipe_version']=='4.0'
    assert before['integrity']['recency_half_life_bars']==252
    assert before['research']['selection']['selection_end']<before['research']['origin']


def test_forward_summary_excludes_other_models_and_neutral_hits():
    from metals.ledger import forward_summary
    r={'asset':'gold','horizon':'1D','model_id':'current','forward_eligible':True,'channel':'production','price':100,'reference_price':100}
    o={'actual_price':99,'inside_interval80':True}
    ledger={'issued':{'a':r,'b':{**r,'model_id':'old'}},'outcomes':{'a':o,'b':o}}
    s=forward_summary(ledger,'gold','1D','current')
    assert s['issued']==s['resolved']==1
    assert s['directional_accuracy']==0 and s['mae_skill']==0

def test_technical_brief_requires_structure_and_ema_agreement():
    from metals.outlook import technical_brief
    indicators={'close':110,'ema20':105,'ema50':100,'rsi':60,'macd_hist':1,'atr':2,'adx':28}
    detail={'indicators':indicators,'last_completed':'2026-09-23T10:00Z','quality':{'latest_expected_bar_present':True},
            'structure':{'direction':1,'levels':[{'zone':[99,101]},{'zone':[115,117]}]}}
    analyses={tf:detail for tf in ('1h','4h','1d')}
    b=technical_brief(analyses)
    assert b['alignment']=='BULLISH' and b['rows'][0]['extension_warning']
    assert b['rows'][0]['support']==101 and b['rows'][0]['resistance']==115
    analyses['1d']={**detail,'structure':{'direction':-1,'levels':[]}}
    assert technical_brief(analyses)['alignment']=='MIXED'
