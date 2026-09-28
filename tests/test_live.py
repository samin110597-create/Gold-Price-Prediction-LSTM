import pandas as pd
import pytest
from metals.live import latest_quote
from metals.providers import merge_caches, collect

NOW=pd.Timestamp('2026-09-28T10:00:00Z')
def raw(symbol='GC=F',price=4000,seconds=0):
    return {'meta':{'symbol':symbol,'regularMarketPrice':price,'regularMarketTime':int((NOW+pd.Timedelta(seconds=seconds)).timestamp())}}

def test_quote_requires_matching_instrument_finite_price_and_nonfuture_time():
    q=latest_quote([raw(seconds=-600),raw(seconds=-60),raw('GLD',price=400),raw(seconds=60)],'GC=F',NOW)
    assert q['time']==(NOW-pd.Timedelta(seconds=60)).isoformat()
    for bad in [raw('GLD'),raw(price=float('nan')),raw(seconds=60),raw(price=-1)]:
        with pytest.raises(ValueError): latest_quote([bad],'GC=F',NOW)

def test_shared_provider_cache_uses_latest_check_without_retimestamping():
    old={'checked_at':'2026-09-28T09:45:00+00:00','status':'CONNECTED','observations':[{'source_time':'2026-09-28T09:30:00Z'}]}
    new={**old,'checked_at':'2026-09-28T09:59:00+00:00'}
    merged=merge_caches({'providers':{'fmp':old}},{'providers':{'fmp':new}},now=NOW)
    out=collect(merged,now=NOW,environ={'FMP_API_KEY':'test'},names={'fmp'})
    assert list(out['providers'])==['fmp']
    assert out['providers']['fmp']['checked_at']==new['checked_at']
    assert out['providers']['fmp']['observations']==old['observations']
    future={**new,'checked_at':'2099-01-01T00:00:00Z'}
    assert merge_caches(merged,{'providers':{'fmp':future}},now=NOW)['providers']['fmp']==new
