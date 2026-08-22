import json
from pathlib import Path
from datetime import datetime, timezone

DATA = Path('data')


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return {}


def num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def trend_sign(label):
    s = str(label or '').lower()
    if 'bull' in s:
        return 1
    if 'bear' in s:
        return -1
    return 0


def recent_signal_edge(tech, horizon_key='audit_5d'):
    events = tech.get('chart_signal_events') or []
    chart = tech.get('chart') or []
    if not chart:
        return {'edge': 0.0, 'events': 0, 'details': []}
    cutoff = str(chart[max(0, len(chart)-12)].get('date') or '')
    vals, details = [], []
    for e in events:
        if str(e.get('date') or '') < cutoff:
            continue
        bias = trend_sign(e.get('bias'))
        audit = e.get(horizon_key) or {}
        edge = num(audit.get('edge'))
        if bias == 0 or edge is None:
            continue
        signed = bias * edge
        vals.append(signed)
        details.append({
            'signal': e.get('name'), 'bias': e.get('bias'), 'events': audit.get('events'),
            'hit_rate': audit.get('hit_rate'), 'baseline': audit.get('baseline'),
            'edge': edge, 'signed_directional_edge': signed, 'status': audit.get('status')
        })
    return {'edge': sum(vals)/len(vals) if vals else 0.0, 'events': len(vals), 'details': details[-6:]}


def active_pattern(tech, preferred_bias=None):
    ranked = []
    for p in tech.get('advanced_chart_patterns') or []:
        bias = str(p.get('bias') or '')
        if preferred_bias and preferred_bias.lower() not in bias.lower():
            continue
        ranked.append(((1 if str(p.get('status')).lower() == 'confirmed' else 0,
                        num(p.get('structural_quality'), 0),
                        num(p.get('target'), -1) if num(p.get('target')) is not None else -1), p))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1] if ranked else {}


def build(asset):
    tech = load(DATA / f'{asset}_technicals.json')
    readout = load(DATA / f'{asset}_precision_gate_v7.json')
    live = load(DATA / ('live_forecast.json' if asset == 'gold' else 'silver_forecast.json'))

    current = num(live.get('latest_price'))
    if current is None:
        current = num(((tech.get('multi_timeframe') or {}).get('1D') or {}).get('close'))

    mt = tech.get('multi_timeframe') or {}
    ind = tech.get('indicators') or {}
    vf = tech.get('volume_flow') or {}
    conf = tech.get('confirmation') or {}
    fib = tech.get('fibonacci_60d') or {}
    swing = tech.get('swing_60d') or {}
    paths = tech.get('technical_path_zones') or {}

    score = 50.0
    reasons = []
    tech_score = num(tech.get('technical_score'), 50.0)
    score += (tech_score - 50.0) * 0.35
    reasons.append(f'Technical confluence {tech_score:.1f}/100')

    for key, weight in [('4H', 7), ('1D', 7), ('1W', 7)]:
        sign = trend_sign((mt.get(key) or {}).get('trend'))
        score += sign * weight
        if sign:
            reasons.append(f'{key} trend {"bullish" if sign > 0 else "bearish"}')

    structure = str(tech.get('market_structure') or '')
    if 'higher-high' in structure.lower() or 'bullish' in structure.lower():
        score += 8; reasons.append('Higher-high / bullish market structure')
    elif 'lower-low' in structure.lower() or 'bearish' in structure.lower():
        score -= 8; reasons.append('Lower-low / bearish market structure')

    macd_bias = str((mt.get('1D') or {}).get('macd_bias') or '').lower()
    if macd_bias.startswith('bull'):
        score += 5; reasons.append('Daily MACD bullish')
    elif macd_bias.startswith('bear'):
        score -= 5; reasons.append('Daily MACD bearish')

    adx, pdi, mdi = num(ind.get('adx14'), 0), num(ind.get('plus_di'), 0), num(ind.get('minus_di'), 0)
    if adx >= 25 and pdi > mdi:
        score += 5; reasons.append(f'ADX {adx:.1f} with +DI leading')
    elif adx >= 25 and mdi > pdi:
        score -= 5; reasons.append(f'ADX {adx:.1f} with -DI leading')

    obv = str(vf.get('obv_trend') or '').lower()
    if 'rising' in obv:
        score += 3; reasons.append('OBV rising')
    elif 'fall' in obv:
        score -= 3; reasons.append('OBV falling')

    volx = num(vf.get('volume_vs_20d'), 1.0)
    if volx >= 1.5:
        score += 3 if score >= 50 else -3
        reasons.append(f'Volume {volx:.2f}x 20D')

    vwap = num(ind.get('rolling_vwap20'))
    if current and vwap:
        if current > vwap:
            score += 3; reasons.append('Price above 20D VWAP')
        else:
            score -= 3; reasons.append('Price below 20D VWAP')

    sig = recent_signal_edge(tech, 'audit_5d')
    score += clamp(sig['edge'] * 100.0, -10, 10)
    if sig['events']:
        reasons.append(f'Recent audited 5D signal edge {sig["edge"]*100:+.1f}pp')

    rsi, stoch, bbp = num(ind.get('rsi14')), num(ind.get('stoch_k')), num(ind.get('bb_percent_b'))
    extension_flags = []
    if rsi is not None and rsi >= 72:
        score -= 8; extension_flags.append(f'RSI {rsi:.1f}')
    if stoch is not None and stoch >= 90:
        score -= 5; extension_flags.append(f'Stoch {stoch:.1f}')
    if bbp is not None and bbp >= 1.0:
        score -= 5; extension_flags.append('above upper Bollinger band')
    if rsi is not None and rsi <= 28:
        score += 8; extension_flags.append(f'RSI {rsi:.1f} oversold')
    if stoch is not None and stoch <= 10:
        score += 5; extension_flags.append(f'Stoch {stoch:.1f} oversold')

    score = clamp(score)
    if score >= 62:
        bias, preferred = 'BULLISH', 'Bullish'
    elif score <= 38:
        bias, preferred = 'BEARISH', 'Bearish'
    else:
        bias, preferred = 'NEUTRAL', None

    pattern = active_pattern(tech, preferred)
    atr = num(ind.get('atr14'), 0)
    ema20_4h = num((mt.get('4H') or {}).get('ema20'))
    ema50_4h = num((mt.get('4H') or {}).get('ema50'))
    bull, bear = num(conf.get('bullish_above')), num(conf.get('bearish_below'))
    fib38, sh, sl = num(fib.get('38.2')), num(swing.get('high')), num(swing.get('low'))
    t13, t510 = paths.get('1_3d') or [], paths.get('5_10d') or []

    trigger = retest = invalidation = None
    targets, location = [], 'WAIT'

    if bias == 'BULLISH' and current:
        if bull and current <= bull * 1.002:
            trigger = bull
        elif sh and sh > current and (sh/current - 1) <= 0.04:
            trigger = sh
        else:
            trigger = current
        if ema20_4h and atr:
            lo, hi = ema20_4h - 0.25*atr, ema20_4h + 0.25*atr
            retest = [round(min(lo, hi), 2), round(min(current, max(lo, hi)), 2)]
        candidates = [num(pattern.get('invalidation')), ema50_4h, fib38, (bull - 0.5*atr) if bull and atr else None]
        candidates = [x for x in candidates if x is not None and x < current]
        invalidation = max(candidates) if candidates else bear
        for x in [num(t13[-1]) if t13 else None, num(t510[-1]) if t510 else None, num(pattern.get('target')), sh]:
            if x is not None and x > current and all(abs(x-y) > max(0.01, current*0.002) for y in targets):
                targets.append(x)
        location = 'WAIT — BULLISH BUT EXTENDED' if len(extension_flags) >= 2 else ('BULLISH SETUP ACTIVE' if bull and current >= bull else 'BULLISH SETUP PENDING CONFIRMATION')

    elif bias == 'BEARISH' and current:
        if bear and current >= bear * 0.998:
            trigger = bear
        elif sl and sl < current and (1 - sl/current) <= 0.04:
            trigger = sl
        else:
            trigger = current
        if ema20_4h and atr:
            lo, hi = ema20_4h - 0.25*atr, ema20_4h + 0.25*atr
            retest = [round(max(current, min(lo, hi)), 2), round(max(lo, hi), 2)]
        candidates = [num(pattern.get('invalidation')), ema50_4h, fib38, (bear + 0.5*atr) if bear and atr else None]
        candidates = [x for x in candidates if x is not None and x > current]
        invalidation = min(candidates) if candidates else bull
        for x in [num(t13[0]) if t13 else None, num(t510[0]) if t510 else None, num(pattern.get('target')), sl]:
            if x is not None and x < current and all(abs(x-y) > max(0.01, current*0.002) for y in targets):
                targets.append(x)
        location = 'WAIT — BEARISH BUT EXTENDED' if len(extension_flags) >= 2 else ('BEARISH SETUP ACTIVE' if bear and current <= bear else 'BEARISH SETUP PENDING CONFIRMATION')
    else:
        location = 'WAIT — NO CLEAR DIRECTIONAL SETUP'

    targets = targets[:3]
    rr = []
    if trigger is not None and invalidation is not None:
        risk = abs(trigger - invalidation)
        if risk > 0:
            for t in targets:
                reward = (t-trigger) if bias == 'BULLISH' else (trigger-t)
                rr.append(round(reward/risk, 2))

    if bias in ('BULLISH','BEARISH') and rr and rr[min(1, len(rr)-1)] < 1.0:
        location = f'WAIT — {bias} BIAS, POOR LOCATION/RISK-REWARD'

    trust_map = {str(x.get('horizon')): x for x in (readout.get('horizons') or [])}
    out = {
        'status': 'ok', 'asset': asset, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'model_version': 'Market Setup V8',
        'purpose': 'Action-oriented market research. Bias/location/trigger/invalidation are shown separately from statistical validation; this is not personalized trading advice.',
        'current_price': current, 'market_bias': bias, 'setup_state': location,
        'evidence_score': round(score, 1),
        'score_note': 'Evidence score is an explainable confluence score, not a calibrated probability.',
        'trigger': round(trigger, 2) if trigger is not None else None,
        'retest_zone': retest, 'invalidation': round(invalidation, 2) if invalidation is not None else None,
        'targets': [round(x, 2) for x in targets], 'risk_reward': rr,
        'extension_flags': extension_flags, 'technical_score': tech_score,
        'technical_bias': tech.get('technical_bias'), 'market_structure': tech.get('market_structure'),
        'rsi14': rsi, 'adx14': adx, 'plus_di': pdi, 'minus_di': mdi,
        'macd_hist': num(ind.get('macd_hist')), 'volume_vs_20d': volx,
        'obv_trend': vf.get('obv_trend'), 'vwap20': vwap,
        'bullish_confirmation': bull, 'bearish_confirmation': bear,
        'active_pattern': pattern, 'recent_signal_edge_5d': sig,
        'statistical_trust': {
            '4H': (trust_map.get('4H') or {}).get('trust'),
            '1W': (trust_map.get('1 Week') or {}).get('trust'),
            '1Y': (trust_map.get('1Y') or {}).get('trust')
        },
        'evidence': reasons,
    }
    (DATA / f'{asset}_actionable_setup_v8.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(asset, bias, location, round(score,1), 'trigger', trigger, 'targets', targets, 'rr', rr)


for asset in ('gold','silver'):
    build(asset)
