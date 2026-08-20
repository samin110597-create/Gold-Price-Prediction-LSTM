import json
from pathlib import Path
from datetime import datetime, timezone

DATA = Path('data')


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return {}


def f(x, d=None):
    try:
        return float(x)
    except Exception:
        return d


def forward_map(asset, ledger):
    return {str(x.get('horizon')): x for x in ledger.get('summary', []) if x.get('asset') == asset}


def probability_lean(p):
    if p is None:
        return 'NEUTRAL'
    if p >= 0.55:
        return 'UP'
    if p <= 0.45:
        return 'DOWN'
    return 'NEUTRAL'


def price_lean(current, target, horizon):
    if current is None or target is None or current <= 0:
        return 'RANGE', None
    move = (target / current - 1.0) * 100.0
    thresholds = {
        '4H': 0.05,
        '1 Day': 0.10,
        '1 Week': 0.25,
        '1 Month': 0.50,
        '1Y': 1.00,
    }
    th = thresholds.get(horizon, 0.10)
    if move > th:
        return 'UP', move
    if move < -th:
        return 'DOWN', move
    return 'RANGE', move


def technical_lean(tech, horizon):
    mt = tech.get('multi_timeframe') or {}
    key = {'4H': '4H', '1 Day': '1D', '1 Week': '1W', '1 Month': '1W'}.get(horizon)
    label = str((mt.get(key) or {}).get('trend') or tech.get('technical_bias') or 'Neutral')
    if 'bull' in label.lower():
        return 'UP', label
    if 'bear' in label.lower():
        return 'DOWN', label
    return 'NEUTRAL', label


def trust_label(audit_ok, live_samples=0, live_accuracy=None, stability_ok=None):
    if audit_ok:
        if live_samples >= 20 and live_accuracy is not None and live_accuracy < 0.50:
            return 'MIXED'
        if stability_ok is False:
            return 'MIXED'
        return 'VALIDATED'
    if live_samples >= 20 and live_accuracy is not None and live_accuracy >= 0.60:
        return 'MIXED'
    return 'LOW TRUST'


def alignment(price_dir, prob_dir, tech_dir):
    dirs = [x for x in (price_dir, prob_dir, tech_dir) if x in ('UP', 'DOWN')]
    if not dirs:
        return 'MIXED'
    up = dirs.count('UP')
    down = dirs.count('DOWN')
    if up >= 2:
        return 'BULLISH CONFLUENCE'
    if down >= 2:
        return 'BEARISH CONFLUENCE'
    return 'MIXED'


def run(asset):
    v6 = load(DATA / f'{asset}_future_price_action_v6.json')
    h4 = load(DATA / f'{asset}_4h_specialist.json')
    macro = load(DATA / f'{asset}_macro_1y.json')
    tech = load(DATA / f'{asset}_technicals.json')
    ledger = load(DATA / 'forward_validation_ledger.json')
    live = load(DATA / ('live_forecast.json' if asset == 'gold' else 'silver_forecast.json'))
    fw = forward_map(asset, ledger)

    current = f(live.get('latest_price'))
    if current is None:
        current = f(((tech.get('multi_timeframe') or {}).get('1D') or {}).get('close'))

    out = {
        'status': 'ok',
        'asset': asset,
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'model_version': 'Directional Readout V7.1',
        'method': 'Always show the raw price-action lean and target. Validation changes the trust label; it never erases the forecast. Probability, technical structure, historical audit and live-forward evidence are displayed separately.',
        'current_price': current,
        'horizons': [],
    }

    # 4H
    p = f(h4.get('probability_up'))
    target = f(h4.get('predicted_price'))
    price_dir, move = price_lean(current, target, '4H')
    prob_dir = probability_lean(p)
    tech_dir, tech_label = technical_lean(tech, '4H')
    status = str(h4.get('overall_status') or '')
    audit_ok = status.upper() == 'PASS' or ('PASS' in status.upper() and 'FAIL' not in status.upper())
    fm = fw.get('4H', {})
    n = int(fm.get('resolved_forecasts') or 0)
    fa = f(fm.get('forward_direction_accuracy'))
    trust = trust_label(audit_ok, n, fa)
    out['horizons'].append({
        'horizon': '4H',
        'call': price_dir,
        'price_direction': price_dir,
        'predicted_price': target,
        'projected_return_pct': move,
        'probability_up': p,
        'probability_down': None if p is None else 1 - p,
        'probability_lean': prob_dir,
        'technical_lean': tech_dir,
        'technical_label': tech_label,
        'alignment': alignment(price_dir, prob_dir, tech_dir),
        'trust': trust,
        'audit_ok': audit_ok,
        'audit_status': status or 'FAIL',
        'forward_samples': n,
        'forward_direction_accuracy': fa,
        'scenario_20_50_80': h4.get('focus_zone'),
        'reason': 'Raw 4H price direction remains visible. Historical audit and live-forward record affect trust, not whether the forecast is shown.',
    })

    # 1D / 1W / 1M from V6
    for z in v6.get('horizons', []):
        label = z.get('horizon')
        target = f(z.get('predicted_price'))
        price_dir, move = price_lean(current, target, label)
        p = f(z.get('probability_up'))
        prob_dir = probability_lean(p)
        tech_dir, tech_label = technical_lean(tech, label)
        audit_ok = bool(z.get('pass'))
        stable = (f(z.get('recent_directional_edge'), -1) >= 0 and f(z.get('recent_mae_skill'), -1) >= 0)
        fm = fw.get(label, {})
        n = int(fm.get('resolved_forecasts') or 0)
        fa = f(fm.get('forward_direction_accuracy'))
        trust = trust_label(audit_ok, n, fa, stable)
        out['horizons'].append({
            'horizon': label,
            'call': price_dir,
            'price_direction': price_dir,
            'predicted_price': target,
            'projected_return_pct': move,
            'probability_up': p,
            'probability_down': None if p is None else 1 - p,
            'probability_lean': prob_dir,
            'technical_lean': tech_dir,
            'technical_label': tech_label,
            'alignment': alignment(price_dir, prob_dir, tech_dir),
            'trust': trust,
            'audit_ok': audit_ok,
            'audit_status': 'PASS' if audit_ok else 'FAIL',
            'stability_ok': stable,
            'forward_samples': n,
            'forward_direction_accuracy': fa,
            'scenario_20_50_80': z.get('scenario_20_50_80'),
            'risk_10_90': z.get('risk_10_90'),
            'directional_accuracy': z.get('directional_accuracy'),
            'baseline_directional_accuracy': z.get('baseline_directional_accuracy'),
            'directional_edge': z.get('directional_edge'),
            'mae_skill_vs_baseline': z.get('mae_skill_vs_baseline'),
            'recent_directional_edge': z.get('recent_directional_edge'),
            'recent_mae_skill': z.get('recent_mae_skill'),
            'reason': 'Direction comes from the current V6 point-price path. Probability, technical structure and validation are shown independently so disagreement is visible instead of suppressed.',
        })

    # 1Y macro challenger: always show the directional lean, but trust stays separate.
    mf = macro.get('macro_forecast') or {}
    p = f(mf.get('probability_up'))
    target = f(mf.get('predicted_price') if mf.get('predicted_price') is not None else mf.get('model_price'))
    price_dir, move = price_lean(current, target, '1Y')
    prob_dir = probability_lean(p)
    tech_dir, tech_label = technical_lean(tech, '1 Month')
    selected = bool(macro.get('selected_as_champion'))
    trust = 'VALIDATED' if selected else 'LOW TRUST'
    out['horizons'].append({
        'horizon': '1Y',
        'call': price_dir,
        'price_direction': price_dir,
        'predicted_price': target,
        'projected_return_pct': move,
        'probability_up': p,
        'probability_down': None if p is None else 1 - p,
        'probability_lean': prob_dir,
        'technical_lean': tech_dir,
        'technical_label': tech_label,
        'alignment': alignment(price_dir, prob_dir, tech_dir),
        'trust': trust,
        'audit_ok': selected,
        'audit_status': 'PROMOTED' if selected else 'CHALLENGER FAIL',
        'directional_accuracy': mf.get('backtest_directional_accuracy'),
        'directional_edge': mf.get('directional_edge'),
        'mae_skill_vs_baseline': mf.get('mae_skill_vs_baseline'),
        'reason': 'The macro target and directional lean stay visible even when the challenger is not promoted. Promotion only controls predictive trust.',
    })

    out['validated_horizons'] = sum(1 for x in out['horizons'] if x.get('trust') == 'VALIDATED')
    out['mixed_horizons'] = sum(1 for x in out['horizons'] if x.get('trust') == 'MIXED')
    out['low_trust_horizons'] = sum(1 for x in out['horizons'] if x.get('trust') == 'LOW TRUST')
    out['technical_context'] = {
        'score': tech.get('technical_score'),
        'bias': tech.get('technical_bias'),
        'market_structure': tech.get('market_structure'),
    }

    (DATA / f'{asset}_precision_gate_v7.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(asset, [(x['horizon'], x['call'], x['trust']) for x in out['horizons']])


for asset in ('gold', 'silver'):
    run(asset)
