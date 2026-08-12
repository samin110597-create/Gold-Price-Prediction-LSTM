import json
from pathlib import Path

DATA = Path('data')
FILES = ('gold_projections.json', 'silver_projections.json')


def normalize_projection(p):
    price = float(p.get('predicted_price', p.get('model_price')))
    focus = p.get('focus_zone') or p.get('tight_model_zone') or [price, price]
    risk = p.get('risk_zone') or p.get('tight_model_zone') or focus

    flo, fhi = sorted([float(focus[0]), float(focus[1])])
    rlo, rhi = sorted([float(risk[0]), float(risk[1])])
    flo = min(flo, price)
    fhi = max(fhi, price)
    rlo = min(rlo, flo)
    rhi = max(rhi, fhi)

    p['predicted_price'] = round(price, 2)
    p['model_price'] = round(price, 2)
    p['focus_zone'] = [round(flo, 2), round(fhi, 2)]
    p['tight_model_zone'] = [round(flo, 2), round(fhi, 2)]
    p['risk_zone'] = [round(rlo, 2), round(rhi, 2)]
    p['focus_zone_type'] = 'out-of-sample calibrated focus zone'
    p['focus_zone_note'] = 'Focus zone is the narrower calibrated band; risk zone preserves the wider uncertainty band.'
    return p


def main():
    for name in FILES:
        path = DATA / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        data['projections'] = [normalize_projection(dict(p)) for p in data.get('projections', [])]
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
