import json
from datetime import datetime, timezone
from pathlib import Path

import regime_intraday_model as rim

DATA = Path('data')
ASSETS = {
    'gold': ('gold_projections.json', 'gold_regime_intraday.json'),
    'silver': ('silver_projections.json', 'silver_regime_intraday.json'),
}


def safe_download(ticker, period, interval):
    periods = ('2y', '1y') if interval == '1h' else (period,)
    last_error = None
    for p in periods:
        try:
            df = rim.flatten(rim.yf.download(
                ticker,
                period=p,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            ))
            if not df.empty and 'Close' in df.columns:
                clean = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
                if not clean.empty:
                    return clean
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'No usable data for {ticker} {interval}: {last_error}')


def write_fallback(reason):
    for asset, (projection_name, output_name) in ASSETS.items():
        p = DATA / projection_name
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding='utf-8'))
        p4 = next((x for x in data.get('projections', []) if x.get('horizon') == '4 Hours'), None)
        if not p4:
            continue
        out = {
            'status': 'fallback',
            'asset': asset,
            'updated_utc': datetime.now(timezone.utc).isoformat(),
            'current_regime': 'Regime model unavailable',
            'selected_4h_model': 'Normal 4H model fallback',
            'comparison': {
                'all_market': {
                    'directional_accuracy': p4.get('backtest_directional_accuracy'),
                    'directional_edge': p4.get('directional_edge'),
                    'mae_skill': p4.get('mae_skill_vs_baseline'),
                    'origins': p4.get('walkforward_origins'),
                },
                'regime_specific': None,
            },
            'enhanced_4h': p4,
            'fallback_reason': str(reason),
        }
        (DATA / output_name).write_text(json.dumps(out, indent=2), encoding='utf-8')


rim.download = safe_download

if __name__ == '__main__':
    try:
        rim.main()
    except Exception as exc:
        print(f'Regime model failed; using normal 4H projection fallback: {exc}')
        write_fallback(exc)
