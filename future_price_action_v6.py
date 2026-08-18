import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multi_asset_forecast_v2 import ASSETS as BASE_ASSETS, daily, load_factors, features

DATA = Path('data'); DATA.mkdir(exist_ok=True)
HORIZONS = {'1 Day': 1, '1 Week': 5, '1 Month': 20}
FRED = {'real_yield_10y':'DFII10','breakeven_10y':'T10YIE','broad_usd':'DTWEXBGS'}


def fred_series(series_id):
    try:
        d = pd.read_csv(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}')
        d = d.iloc[:, :2]; d.columns = ['date','value']
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d['value'] = pd.to_numeric(d['value'], errors='coerce')
        s = d.dropna().set_index('date')['value'].sort_index(); s.index = s.index.normalize()
        return s
    except Exception:
        return pd.Series(dtype=float)


def enhanced_features(target, fac):
    x = features(target, fac).copy()
    c = target.Close.astype(float)
    e20 = c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    vol20 = c.pct_change().rolling(20).std(); vol60 = c.pct_change().rolling(60).std()
    x['trend_stack'] = ((c>e20).astype(int) + (e20>e50).astype(int) + (e50>e200).astype(int))/3
    x['trend_accel'] = (e20/e20.shift(10)-1) - (e50/e50.shift(10)-1)
    x['vol_ratio'] = vol20/vol60.replace(0,np.nan)
    x['gap_from_20d_high'] = c/target.High.rolling(20).max()-1
    x['gap_from_20d_low'] = c/target.Low.rolling(20).min()-1
    if 'Volume' in target.columns:
        v=target.Volume.astype(float); x['volume_ratio20']=v/v.rolling(20).mean().replace(0,np.nan); x['volume_z60']=(v-v.rolling(60).mean())/v.rolling(60).std().replace(0,np.nan)
    for name,sid in FRED.items():
        s=fred_series(sid)
        if s.empty: continue
        z=s.reindex(x.index).ffill().shift(1)
        if z.notna().sum()<650: continue
        x[name]=z; x[f'{name}_chg5']=z.diff(5); x[f'{name}_chg20']=z.diff(20); x[f'{name}_z60']=(z-z.rolling(60).mean())/z.rolling(60).std().replace(0,np.nan)
    min_nonnull=max(650,int(len(x)*.45)); x=x[[col for col in x.columns if x[col].notna().sum()>=min_nonnull]]
    return x.replace([np.inf,-np.inf],np.nan)


def regime_series(target):
    c=target.Close.astype(float); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    vol=c.pct_change().rolling(20).std(); med=vol.rolling(252,min_periods=100).median()
    out=[]
    for i in range(len(c)):
        trend='BULL' if c.iloc[i]>e20.iloc[i]>e50.iloc[i] else ('BEAR' if c.iloc[i]<e20.iloc[i]<e50.iloc[i] else 'RANGE')
        vr=vol.iloc[i]/med.iloc[i] if i<len(med) and np.isfinite(med.iloc[i]) and med.iloc[i]>0 else 1.0
        v='HIGHVOL' if vr>=1.25 else ('LOWVOL' if vr<=.80 else 'MIDVOL')
        out.append(f'{trend}_{v}')
    return pd.Series(out,index=target.index,name='regime')


def models():
    return {
        'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=12.0))]),
        'Extra Trees':ExtraTreesRegressor(n_estimators=180,max_depth=7,min_samples_leaf=10,max_features=.72,random_state=42,n_jobs=-1),
        'Gradient Boost':HistGradientBoostingRegressor(max_iter=140,max_depth=3,learning_rate=.04,l2_regularization=.55,random_state=42),
    }


def origin_positions(n,h,max_origins=80):
    min_train=1000; spacing=8 if h==1 else (10 if h==5 else 15); end=n-h-1
    if end<=min_train+h:return []
    pool=np.arange(min_train+h,end+1,spacing)
    return (pool[-max_origins:] if len(pool)>max_origins else pool).tolist()


def dynamic_weights(records, names, regime):
    if len(records)<12:return np.ones(len(names))/len(names)
    same=[r for r in records if r['regime']==regime]
    use=same if len(same)>=8 else records[-32:]
    q=[]
    for name in names:
        ae=np.array([abs(r['actual']-r['model_preds'][name]) for r in use]); da=np.array([(r['model_preds'][name]>=0)==(r['actual']>=0) for r in use],dtype=float)
        mae=float(ae.mean()) if len(ae) else 1.; acc=float(da.mean()) if len(da) else .5
        q.append((.70+.60*acc)/max(mae,1e-5))
    w=np.array(q,dtype=float); return w/w.sum()


def residual_pool(records, regime):
    same=[r['actual']-r['pred'] for r in records if r['regime']==regime]
    if len(same)>=12:return np.asarray(same,dtype=float)
    return np.asarray([r['actual']-r['pred'] for r in records[-48:]],dtype=float)


def fit_eval(target,x,h,label):
    c=target.Close.astype(float); y=c.shift(-h)/c-1; rg=regime_series(target)
    df=x.copy(); df['_y']=y; df['_regime']=rg; df=df.dropna()
    if len(df)<1150:raise RuntimeError(f'insufficient aligned history: {len(df)}')
    X=df.drop(columns=['_y','_regime']); Y=df['_y'].astype(float); R=df['_regime']; M=models(); names=list(M)
    origins=origin_positions(len(df),h); records=[]
    for o in origins:
        tr=o-h
        if tr<850:continue
        xt,yt=X.iloc[:tr],Y.iloc[:tr]; xp=X.iloc[[o]]; rnow=str(R.iloc[o]); mp={}
        for name,tpl in M.items():
            m=clone(tpl);m.fit(xt,yt);mp[name]=float(m.predict(xp)[0])
        w=dynamic_weights(records,names,rnow); pred=float(sum(mp[n]*ww for n,ww in zip(names,w)))
        base=float(np.median(yt.tail(min(756,len(yt)))))
        pool=residual_pool(records,rnow); interval=None
        if len(pool)>=12:
            q20,q80=np.quantile(pool,[.2,.8]);interval=(pred+q20,pred+q80)
        records.append({'actual':float(Y.iloc[o]),'pred':pred,'baseline':base,'regime':rnow,'model_preds':mp,'weights':w.tolist(),'interval':interval})
    if len(records)<40:raise RuntimeError(f'too few origins for {label}: {len(records)}')
    actual=np.array([r['actual'] for r in records]); pred=np.array([r['pred'] for r in records]); base=np.array([r['baseline'] for r in records])
    mae=float(np.mean(np.abs(actual-pred))); bmae=float(np.mean(np.abs(actual-base))); skill=float(1-mae/max(bmae,1e-9))
    dacc=float(np.mean((pred>=0)==(actual>=0))); bdacc=float(np.mean((base>=0)==(actual>=0))); edge=dacc-bdacc
    third=max(16,len(records)//3); ra=actual[-third:];rp=pred[-third:];rb=base[-third:]
    recent_skill=float(1-np.mean(np.abs(ra-rp))/max(np.mean(np.abs(ra-rb)),1e-9)); recent_edge=float(np.mean((rp>=0)==(ra>=0))-np.mean((rb>=0)==(ra>=0)))
    iv=[(r['interval'][0]<=r['actual']<=r['interval'][1]) for r in records if r['interval'] is not None]; coverage=float(np.mean(iv)) if iv else None
    current_regime=str(regime_series(target).iloc[-1]); latest=x.dropna().iloc[[-1]]; valid=x.notna().all(axis=1)&y.notna();xt=x.loc[valid];yt=y.loc[valid]
    mp={};details=[]
    for name,tpl in M.items():
        m=clone(tpl);m.fit(xt,yt);p=float(m.predict(latest)[0]);mp[name]=p;details.append({'name':name,'projected_return_pct':round(p*100,3)})
    w=dynamic_weights(records,names,current_regime); lr=float(sum(mp[n]*ww for n,ww in zip(names,w))); last=float(c.iloc[-1]); price=last*(1+lr)
    pool=residual_pool(records,current_regime)
    if len(pool)<12:pool=actual-pred
    q10,q20,q50,q80,q90=np.quantile(pool,[.1,.2,.5,.8,.9])
    downside=last*(1+lr+q20); basecase=last*(1+lr+q50); upside=last*(1+lr+q80); risklo=last*(1+lr+q10); riskhi=last*(1+lr+q90)
    rawp=float(np.mean(lr+pool>0)); strength=float(np.clip((max(edge,0)+max(skill,0))/.12,0,1)*np.clip(len(pool)/30,0,1)); pup=float(np.clip(.5+(rawp-.5)*(.30+.70*strength),.20,.80))
    path='UP' if pup>=.57 and lr>0 else ('DOWN' if pup<=.43 and lr<0 else 'RANGE / MIXED')
    minreq=56 if h==1 else 48
    stable=recent_edge>=0 and recent_skill>=0
    calibrated=(coverage is None or .45<=coverage<=.75)
    passed=bool(edge>=.03 and skill>=.03 and len(records)>=minreq and stable and calibrated)
    for d,ww in zip(details,w):d['live_weight']=round(float(ww),4)
    return {
        'horizon':label,'steps':h,'current_regime':current_regime,'dominant_path':path,'predicted_price':round(price,2),'base_case_price':round(basecase,2),
        'projected_return_pct':round(lr*100,3),'probability_up':round(pup,4),'probability_down':round(1-pup,4),
        'scenario_20_50_80':[round(downside,2),round(basecase,2),round(upside,2)],'risk_10_90':[round(risklo,2),round(riskhi,2)],
        'directional_accuracy':round(dacc,4),'baseline_directional_accuracy':round(bdacc,4),'directional_edge':round(edge,4),
        'mae_pct':round(mae*100,3),'baseline_mae_pct':round(bmae*100,3),'mae_skill_vs_baseline':round(skill,4),
        'recent_directional_edge':round(recent_edge,4),'recent_mae_skill':round(recent_skill,4),'interval_60_coverage':round(coverage,4) if coverage is not None else None,
        'walkforward_origins':len(records),'regime_residual_samples':int(len(pool)),'pass':passed,'status':'PASS' if passed else 'FAIL',
        'models':details,'validation':'Purged chronological walk-forward. Ensemble weights at each origin use only prior OOS errors, preferring the same market regime. Probability and scenario bands use prior OOS residuals; no future-origin information is used.'
    }


def score(r):return .55*float(r.get('mae_skill_vs_baseline') or 0)+.45*float(r.get('directional_edge') or 0)


def load_json(path):
    try:return json.loads(Path(path).read_text())
    except:return {}


def run(asset,cfg):
    target=daily(cfg['ticker']); fac,meta=load_factors(target.index,cfg['factors']); fac=fac[[c for c in fac.columns if meta.get(c,{}).get('available')]]; x=enhanced_features(target,fac)
    out={'status':'ok','asset':asset,'symbol':cfg['ticker'],'updated_utc':datetime.now(timezone.utc).isoformat(),'model_version':'Future Price Action V6','horizons':[],'promotions':[]}
    pp=DATA/f'{asset}_projections.json'; proj=load_json(pp); by={r.get('horizon'):r for r in proj.get('projections',[])}
    for label,h in HORIZONS.items():
        ch=fit_eval(target,x,h,label); out['horizons'].append(ch)
        if label in ('1 Day','1 Week'):
            inc=by.get(label,{}); iscore=score(inc); cscore=score(ch); incpass=bool(inc.get('directional_edge',0)>=.02 and inc.get('mae_skill_vs_baseline',0)>=.02)
            promote=bool(ch['pass'] and (not incpass or cscore>=iscore+.015));out['promotions'].append({'horizon':label,'selected_as_champion':promote,'incumbent_score':round(iscore,4),'challenger_score':round(cscore,4)})
            if promote and inc:
                old_steps=inc.get('steps',h);inc.clear();inc.update({'horizon':label,'steps':old_steps,'predicted_price':ch['predicted_price'],'model_price':ch['predicted_price'],'raw_ml_price':ch['predicted_price'],'projected_return_pct':ch['projected_return_pct'],'probability_up':ch['probability_up'],'probability_down':ch['probability_down'],'confidence':'Moderate','confidence_score':58,'forecast_status':'PASS','tight_model_zone':[ch['scenario_20_50_80'][0],ch['scenario_20_50_80'][2]],'focus_zone':[ch['scenario_20_50_80'][0],ch['scenario_20_50_80'][2]],'risk_zone':ch['risk_10_90'],'backtest_directional_accuracy':ch['directional_accuracy'],'baseline_directional_accuracy':ch['baseline_directional_accuracy'],'directional_edge':ch['directional_edge'],'backtest_mae_pct':ch['mae_pct'],'baseline_mae_pct':ch['baseline_mae_pct'],'mae_skill_vs_baseline':ch['mae_skill_vs_baseline'],'recent_half_directional_edge':ch['recent_directional_edge'],'recent_half_mae_skill':ch['recent_mae_skill'],'walkforward_origins':ch['walkforward_origins'],'selected_model':'Future Price Action V6','validation':ch['validation'],'model_version':'Future Price Action V6'})
    if pp.exists():pp.write_text(json.dumps(proj,indent=2))
    h4=load_json(DATA/f'{asset}_4h_specialist.json'); macro=load_json(DATA/f'{asset}_macro_1y.json')
    out['path_summary']={'4H':{'price':h4.get('predicted_price'),'probability_up':h4.get('probability_up'),'status':h4.get('overall_status')},'1D':next((z for z in out['horizons'] if z['horizon']=='1 Day'),{}),'1W':next((z for z in out['horizons'] if z['horizon']=='1 Week'),{}),'1M':next((z for z in out['horizons'] if z['horizon']=='1 Month'),{}),'1Y':{'price':macro.get('macro_forecast',{}).get('predicted_price'),'probability_up':macro.get('macro_forecast',{}).get('probability_up'),'status':'PASS' if macro.get('selected_as_champion') else 'FAIL'}}
    (DATA/f'{asset}_future_price_action_v6.json').write_text(json.dumps(out,indent=2));print(asset,[(z['horizon'],z['status'],z['dominant_path']) for z in out['horizons']])


def main():
    for asset,cfg in BASE_ASSETS.items():
        try:run(asset,cfg)
        except Exception as e:
            (DATA/f'{asset}_future_price_action_v6.json').write_text(json.dumps({'status':'unavailable','asset':asset,'updated_utc':datetime.now(timezone.utc).isoformat(),'reason':str(e),'horizons':[]},indent=2));print(asset,'unavailable',e)

if __name__=='__main__':main()
