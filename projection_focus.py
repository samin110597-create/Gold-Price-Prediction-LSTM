import json
from pathlib import Path

DATA = Path('data')
FILES = ('gold_projections.json', 'silver_projections.json')


def tighten_projection(p):
    price = float(p.get('model_price'))
    zone = p.get('tight_model_zone') or [price, price]
    lo, hi = float(zone[0]), float(zone[1])
    old_radius = max(price - lo, hi - price, 0.0)

    edge = float(p.get('directional_edge', 0) or 0)
    skill = float(p.get('mae_skill_vs_baseline', 0) or 0)
    agreement = float(p.get('model_agreement', 0) or 0)

    # Narrow more only when validation is stronger. Never hide the old wider band.
    if edge > 0.04 and skill > 0.02 and agreement >= 0.67:
        factor = 0.55
    elif edge > 0 and skill > 0:
        factor = 0.65
    else:
        factor = 0.78

    # Avoid making a visually tiny band that implies false certainty.
    min_pct = 0.0018 if p.get('horizon') == '4 Hours' else 0.0035
    if p.get('horizon') == '1 Week':
        min_pct = 0.0075
    if p.get('horizon') == '1 Year':
        min_pct = 0.06

    radius = max(old_radius * factor, price * min_pct)
    p['risk_zone'] = [round(lo, 2), round(hi, 2)]
    p['tight_model_zone'] = [round(price - radius, 2), round(price + radius, 2)]
    p['focus_zone_type'] = 'validation-weighted focus zone'
    p['focus_zone_note'] = 'Narrower decision band. The wider risk_zone is kept separately and remains important.'
    return p


def main():
    for name in FILES:
        path = DATA / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        data['projections'] = [tighten_projection(dict(p)) for p in data.get('projections', [])]
        data['note'] = (
            'Model price is the single point estimate. Tight model zone is a validation-weighted focus zone; '
            'risk_zone preserves the wider original uncertainty band. A tighter zone is not a guarantee of accuracy.'
        )
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
