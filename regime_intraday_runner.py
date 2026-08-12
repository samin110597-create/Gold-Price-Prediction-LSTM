import json
from datetime import datetime, timezone
from pathlib import Path

import regime_intraday_model as rim

DATA = Path('data')
PROJECTION_FILES = {
    'gold': DATA / 'gold_projections.json',
    'silver': DATA / 'silver_projections.json',
}
REGIME_FILES = {
    'gold': DATA / 'gold_regime_intraday.json',
    'silver': DATA / 'silver_regime_intraday.json',
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


def fallback_regime(asset, projection):
    p4 = next((x for x in projection.get('projections', []) if x.get('horizon') == '4 Hours'), {})
    return {
        'status': 'fallback',
        'asset': asset,
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'current_regime': 'Regime comparison unavailable',
        'selected_4h_model': 'Calibrated main 4H model',
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
        'note': 'Regime engine was unavailable, so this panel mirrors the calibrated main 4H projection.',
    }


def main():
    rim.download = safe_download
    saved = {}
    parsed = {}
    for asset, path in PROJECTION_FILES.items():
        if path.exists():
            saved[asset] = path.read_text(encoding='utf-8')
            parsed[asset] = json.loads(saved[asset])

    try:
        rim.main()
    except Exception as exc:
        print(f'Regime model warning: {exc}')
        for asset, projection in parsed.items():
            REGIME_FILES[asset].write_text(json.dumps(fallback_regime(asset, projection), indent=2), encoding='utf-8')
    finally:
        # The regime engine is a comparison layer. Never let it overwrite the calibrated official projections.
        for asset, text in saved.items():
            PROJECTION_FILES[asset].write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
