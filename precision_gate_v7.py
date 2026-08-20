import json
from pathlib import Path
from datetime import datetime, timezone

DATA=Path('data')

def load(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8'))
    except:return {}

def f(x,d=None):
    try:return float(x)
    except:return d

def forward_map(asset,ledger):
    return {str(x.get('horizon')):x for x in ledger.get('summary',[]) if x.get('asset')==asset}

def call_from_prob(p):
    if p is None:return 'NO EDGE'
    if p>=.58:return 'UP'
    if p<=.42:return 'DOWN'
    return 'NO EDGE'

def run(asset):
    v6=load(DATA/f'{asset}_future_price_action_v6.json')
    h4=load(DATA/f'{asset}_4h_specialist.json')
    macro=load(DATA/f'{asset}_macro_1y.json')
    tech=load(DATA/f'{asset}_technicals.json')
    ledger=load(DATA/'forward_validation_ledger.json')
    fw=forward_map(asset,ledger)
    out={'status':'ok','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),
         'model_version':'Precision Gate V7','method':'Selective release gate. It does not improve raw model predictions; it improves precision of published directional calls by abstaining unless predeclared evidence thresholds are met.','horizons':[]}

    # 4H: require separated probability, recent-method pass, and enough live evidence.
    p=f(h4.get('probability_up'))
    raw=call_from_prob(p); status=str(h4.get('overall_status') or '')
    fm=fw.get('4H',{}); n=int(fm.get('resolved_forecasts') or 0); fa=f(fm.get('forward_direction_accuracy'))
    live_ok=(n>=20 and fa is not None and fa>=.60)
    audit_ok=('PASS' in status.upper())
    release=(raw!='NO EDGE' and audit_ok and live_ok)
    out['horizons'].append({'horizon':'4H','call':raw if release else 'NO EDGE','watch_direction':raw,'probability_up':p,
        'release':release,'audit_ok':audit_ok,'forward_ok':live_ok,'forward_samples':n,'forward_direction_accuracy':fa,
        'reason':'Released only with separated probability + 4H audit pass + >=20 live forecasts at >=60% direction accuracy.'})

    # Daily/weekly/monthly: V6 must PASS first. Forward history becomes an extra gate once mature.
    for z in v6.get('horizons',[]):
        label=z.get('horizon'); p=f(z.get('probability_up')); raw=call_from_prob(p); audit_ok=bool(z.get('pass'))
        fm=fw.get(label,{}); n=int(fm.get('resolved_forecasts') or 0); fa=f(fm.get('forward_direction_accuracy'))
        forward_ok=True if n<20 else (fa is not None and fa>=.55)
        stable=(f(z.get('recent_directional_edge'),-1)>=0 and f(z.get('recent_mae_skill'),-1)>=0)
        release=(raw!='NO EDGE' and audit_ok and stable and forward_ok)
        out['horizons'].append({'horizon':label,'call':raw if release else 'NO EDGE','watch_direction':raw,'probability_up':p,
            'release':release,'audit_ok':audit_ok,'stability_ok':stable,'forward_ok':forward_ok,'forward_samples':n,'forward_direction_accuracy':fa,
            'scenario_20_50_80':z.get('scenario_20_50_80'),'risk_10_90':z.get('risk_10_90'),
            'directional_edge':z.get('directional_edge'),'mae_skill_vs_baseline':z.get('mae_skill_vs_baseline'),
            'reason':'Released only when V6 PASS + recent stability + probability outside 42–58% no-edge band; once 20 live outcomes exist, live direction must be >=55%.'})

    # 1Y: only release when the exact-formula macro challenger is actually selected/promoted.
    mf=macro.get('macro_forecast') or {}; p=f(mf.get('probability_up')); raw=call_from_prob(p)
    selected=bool(macro.get('selected_as_champion')); release=(raw!='NO EDGE' and selected)
    out['horizons'].append({'horizon':'1Y','call':raw if release else 'NO EDGE','watch_direction':raw,'probability_up':p,
        'release':release,'audit_ok':selected,'predicted_price':mf.get('predicted_price'),
        'reason':'1Y direction is released only if the exact-formula macro challenger earns champion promotion.'})

    calls=[x for x in out['horizons'] if x['release']]
    out['released_calls']=len(calls);out['abstained_horizons']=len(out['horizons'])-len(calls)
    out['technical_context']={'score':tech.get('technical_score'),'bias':tech.get('technical_bias'),'market_structure':tech.get('market_structure')}
    (DATA/f'{asset}_precision_gate_v7.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(asset,[(x['horizon'],x['call'],x['release']) for x in out['horizons']])

for asset in ('gold','silver'):
    run(asset)
