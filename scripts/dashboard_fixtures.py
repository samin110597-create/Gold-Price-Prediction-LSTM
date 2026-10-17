"""Browser-only synthetic feed cases; never written to production data."""
import json
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

DURATIONS = {'15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
EXPECTED = {
    'fresh': 'PRICE + TECHNICALS · SAME REFRESH',
    'closed': 'MARKET CLOSED — LAST REPORTED DATA',
    'stale': 'STALE / UNAVAILABLE — WAIT',
    'gap': 'LATEST COMPLETED BAR MISSING',
    'unavailable': 'UNAVAILABLE — WAIT',
}


def install_spot_fixture(page, scenario):
    """Replace only public browser feeds; keep real staged model/manifest checks."""
    now = int(datetime.now(timezone.utc).timestamp()) - 1
    stamp = now - (172800 if scenario in ('closed', 'stale') else 0)

    def iso(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    def route_feed(route):
        url = urlparse(route.request.url)
        if scenario == 'unavailable':
            route.fulfill(status=503, content_type='application/json', body='{}')
            return
        if url.hostname == 'api.gold-api.com':
            symbol = url.path.rsplit('/', 1)[-1]
            body = {'symbol': symbol, 'currency': 'USD', 'price': 200 if symbol == 'XAU' else 30, 'updatedAt': iso(stamp)}
        elif url.path.endswith('/latest'):
            body = {symbol: {'symbol': symbol, 'timestamp': iso(stamp),
                    'source': 'MetaTrader 5 (TEST FIXTURE)', 'bid': price-.1,
                    'ask': price+.1, 'mid': price, 'marketState': 'closed' if scenario == 'closed' else 'open',
                    'stale': scenario == 'stale'} for symbol, price in [('XAUUSD', 200), ('XAGUSD', 30)]}
        else:
            symbol = url.path.split('/')[2]
            tf = parse_qs(url.query)['interval'][0]
            duration = DURATIONS[tf]
            end = stamp // duration * duration
            if scenario == 'gap':
                end -= 3 * duration
            base = 180 if symbol == 'XAUUSD' else 20
            body = {'symbol': symbol, 'interval': tf, 'bars': [
                {'openTime': iso(end-(90-i)*duration), 'open': base+i*.1,
                 'close': base+i*.1+.05, 'high': base+i*.1+.2,
                 'low': base+i*.1-.1, 'isOpen': False} for i in range(90)]}
        route.fulfill(status=200, content_type='application/json', body=json.dumps(body))

    for pattern in ('https://biquote.io/api/**', 'https://api.gold-api.com/price/**'):
        page.unroute(pattern)
        page.route(pattern, route_feed)


def check_spot_panel(page, scenario=None):
    """A live feed may degrade; it must never silently claim current data."""
    page.wait_for_function("!spotBusy && !spotTechnicalBusy", timeout=45000)
    status = page.locator('#technical-status').inner_text()
    assert status in EXPECTED.values(), f'Unexpected technical status: {status}'
    if scenario:
        assert status == EXPECTED[scenario], f'{scenario}: {status}'
    state = page.evaluate("""() => {
      const s=spotTechnicalState[asset];
      return {present:!!s,error:!!spotTechnicalErrors[asset],now:Date.now(),
        quote:s?.quote,asof:s?.asof,missing:s?.frames[spotTechnicalFrame]?.missingLatest};
    }""")
    if not state['present']:
        assert state['error'] and status == EXPECTED['unavailable']
        assert page.locator('[data-live-technical-frame]').count() == 0
        assert 'No futures history is substituted' in page.locator('#spot-technicals').inner_text()
    else:
        assert page.locator('[data-live-technical-frame]').count() == 4
        quote = state['quote']
        age = state['now'] - datetime.fromisoformat(quote['time'].replace('Z', '+00:00')).timestamp()*1000
        calculation_age = state['now'] - datetime.fromisoformat(state['asof'].replace('Z', '+00:00')).timestamp()*1000
        expected = (EXPECTED['stale'] if state['error'] else EXPECTED['closed'] if quote['marketState'] == 'closed'
                    else EXPECTED['stale'] if quote['providerStale'] or age > 1200000 or calculation_age > 1200000
                    else EXPECTED['gap'] if state['missing'] else EXPECTED['fresh'])
        assert status == expected, f'Freshness mismatch: {status}, expected {expected}'
        for tf in DURATIONS:
            page.locator(f'[data-spot-tf="{tf}"]').click()
            assert page.locator(f'[data-spot-tf="{tf}"]').get_attribute('aria-pressed') == 'true'
            assert page.locator('#spot-chart').evaluate('(c)=>c.width>0 && c.height>0')
    return {'status': status, 'has_candles': state['present'], 'source_time': (state.get('quote') or {}).get('time')}
