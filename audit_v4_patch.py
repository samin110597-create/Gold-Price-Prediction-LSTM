import json
from pathlib import Path

DATA=Path('data')

def load(name):
    p=DATA/name
    try:return json.loads(p.read_text()) if p.exists() else {}
    except:return {}

def grade(rate):
    return 'PASS' if rate>=.70 else ('MIXED' if rate>=.40 else 'FAIL')

def main():
    p=DATA/'model_backtest.json'
    if not p.exists():return
    data=json.loads(p.read_text())
    for asset,a in (data.get('assets') or {}).items():
        legacy=list(a.get('probability_forecasts') or [])
        v4=load(f'{asset}_meaningful_probability_v4.json')
        by={int(x.get('horizon_days')):x for x in (v4.get('horizons') or []) if x.get('horizon_days') is not None}
        champs=[]
        for old in legacy:
            h=int(old.get('horizon_days'))
            z=by.get(h)
            if z and z.get('pass'):
                champs.append({'horizon_days':h,'source':'V4 meaningful-move probability','task':'volatility-adjusted UP / DOWN / NO EDGE','probability_up':z.get('probability_up'),'probability_down':z.get('probability_down'),'probability_no_edge':z.get('probability_no_edge'),'signal':z.get('signal'),'accuracy':z.get('oos_accuracy'),'baseline':z.get('majority_baseline'),'edge':round(float(z.get('oos_accuracy') or 0)-float(z.get('majority_baseline') or 0),4),'balanced_accuracy':z.get('balanced_accuracy'),'active_direction_accuracy':z.get('active_direction_accuracy'),'active_edge':z.get('active_edge'),'oos_observations':z.get('walkforward_origins'),'pass':True,'validation':z.get('validation')})
            else:
                q=dict(old);q['source']='Legacy binary UP/DOWN probability';champs.append(q)
        a['probability_forecasts_legacy']=legacy
        a['meaningful_probability_v4']=(v4.get('horizons') or [])
        a['probability_forecasts']=champs
        tests=[bool(x.get('pass')) for x in champs]+[bool(x.get('pass')) for x in (a.get('price_projections') or [])]+[bool(x.get('pass')) for x in (a.get('technical_champions') or [])]
        rate=sum(tests)/len(tests) if tests else 0.0
        a['component_pass_rate']=round(rate,3);a['audit_grade']=grade(rate)
    data['validation_method']=str(data.get('validation_method',''))+' V4 probability champion uses a separate volatility-adjusted 3-state task and can replace the legacy probability component only after its own frozen walk-forward pass rule is met.'
    p.write_text(json.dumps(data,indent=2))
    print({k:(v.get('audit_grade'),v.get('component_pass_rate')) for k,v in (data.get('assets') or {}).items()})
if __name__=='__main__':main()
