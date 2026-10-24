import copy
import pandas as pd
from metals.market_brief import dated_change, explain, parse_releases, report_cards, build_brief

def rows():
    return [{'date':d.isoformat(),'available_at':(d+pd.Timedelta(days=1)).isoformat(),'value':float(i+100)}
            for i,d in enumerate(pd.date_range('2026-10-01',periods=10,tz='UTC'))]

def test_context_uses_only_available_observations_and_actual_comparison_dates():
    r=rows(); result=dated_change(r,'2026-10-08T00:00Z')
    assert result['value']==106 and result['change']==5
    assert result['comparison_time'].startswith('2026-10-02')
    r[-1]['value']=9999
    assert dated_change(r,'2026-10-08T00:00Z')==result
    assert dated_change(r,'2026-11-01T00:00Z')['status']=='STALE'

def test_stale_missing_and_single_values_cannot_create_direction():
    for r in ([],rows()[:1],rows()):
        f={**dated_change(r,'2026-11-01T00:00Z'),'key':'dollar'}
        assert explain(f,'gold')['impact']=='UNAVAILABLE'
    f={**dated_change(rows()[:1],'2026-10-02T00:00Z'),'key':'DFII10'}
    assert explain(f,'gold')['impact']=='DIRECTION UNKNOWN'
    assert dated_change([{'date':'2026-10-01','available_at':'2026-10-02','value':float('nan')}],'2026-10-03T00:00Z')['status']=='UNAVAILABLE'

def test_factor_interpretation_is_asset_specific_not_a_fake_probability():
    f={'key':'equities','status':'OBSERVED · LAGGED','change':-5,'change_percent':-1}
    assert explain(f,'silver')['impact']=='HEADWIND'
    assert explain(f,'gold')['impact']=='MIXED'
    assert explain({**f,'key':'dollar'},'gold')['impact']=='TAILWIND'
    assert explain({**f,'key':'oil'},'silver')['impact']=='MIXED'

def test_rss_rejects_future_undated_and_untrusted_links():
    def item(date,url):
        return f'<item><title>Policy update</title><link>{url}</link><pubDate>{date}</pubDate></item>'
    xml='<rss><channel>'+''.join([item('22 Oct 2026 12:00:00 GMT','https://www.federalreserve.gov/new.htm'),item('25 Oct 2026 12:00:00 GMT','https://www.federalreserve.gov/future.htm'),item('22 Oct 2026 12:00:00 GMT','javascript:alert(1)'),item('','https://www.federalreserve.gov/undated.htm')])+'</channel></rss>'
    result=parse_releases(xml,'2026-10-23T12:00Z')
    assert len(result)==1 and result[0]['url'].endswith('/new.htm')

def test_reports_are_not_backfilled_and_expire_without_changing_source_dates():
    assert report_cards('gold','2026-10-22T12:00Z')==[]
    recent=report_cards('silver','2026-10-23T14:00Z')
    later=report_cards('silver','2027-01-01T00:00Z')
    assert recent and all(not r['expired'] for r in recent)
    assert all(r['expired'] for r in later)
    assert recent[0]['verified_at']==later[0]['verified_at']

def test_brief_never_changes_forecasts_and_demands_timeframe_agreement():
    a={'technical_brief':{'rows':[{'timeframe':t,'fresh':True,'bias':'BEARISH'} for t in ('4H','1D')]},'forecasts':{'test':{'price':100}}}
    original=copy.deepcopy(a)
    assert build_brief('gold',a,[],'2026-10-23T14:00Z')['bias']=='BEARISH'
    assert a==original
    a['technical_brief']['rows'][1]['fresh']=False
    assert build_brief('gold',a,[],'2026-10-23T14:00Z')['bias']=='MIXED'

def test_revised_fred_context_never_becomes_a_model_feature(monkeypatch,tmp_path):
    from metals import providers
    from metals.market_brief import context_factors
    def fetch(url,params):
        if url.endswith('vintagedates'):
            raise providers.ProviderError('VINTAGE REQUEST REJECTED')
        return {'observations':[{'date':r['date'][:10],'value':str(r['value'])} for r in rows()][::-1]}
    monkeypatch.setattr(providers,'request',fetch)
    now=pd.Timestamp('2026-10-11T12:00Z')
    result=providers.collect_provider('fred','test',now)
    assert result['histories']=={} and len(result['context_histories']['DFII10'])==10
    cache={'asof':now.isoformat(),'providers':{'fred':result}}
    assert providers.macro_features(pd.date_range('2026-10-01',periods=20,tz='UTC'),cache).empty
    assert 'context_histories' not in providers.public_summary(cache)['providers']['fred']
    factors=context_factors(tmp_path,cache,{},now)
    real=next(f for f in factors if f['key']=='DFII10')
    assert real['change']==5 and 'latest vintage' in real['source']
    earlier=context_factors(tmp_path,cache,{},now-pd.Timedelta(hours=1))
    assert next(f for f in earlier if f['key']=='DFII10')['status']=='UNAVAILABLE'
