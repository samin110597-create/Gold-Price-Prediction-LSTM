import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path('data')
DATA.mkdir(exist_ok=True)
ASSETS = {
    'gold': {'ticker':'GC=F','projection':'gold_projections.json','output':'gold_regime_intraday.json'},
    'silver': {'ticker':'SI=F','projection':'silver_projections.json','output':'silver_regime_intraday.json'},
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker, period, interval):
    df = flatten(yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                             progress=False, threads=False))
    if df.empty or 'Close' not in df.columns:
        raise RuntimeError(f'No data for {ticker} {interval}')
    return df.dropna(subset=['Open','High','Low','Close']).copy()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def atr(df, n=14):
    pc = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    up = df.High.diff(); down = -df.Low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1/n,adjust=False).mean() / a.replace(0,np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n,adjust=False).mean() / a.replace(0,np.nan)
    dx = 100 * (plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()


def daily_regimes(target, vix, spy):
    c = target.Close.astype(float)
    e20 = c.ewm(span=20,adjust=False).mean(); e50 = c.ewm(span=50,adjust=False).mean()
    a = atr(target,14)/c
    ax = adx(target,14)
    a_pct = a.rolling(756,min_periods=180).rank(pct=True)
    vix_s = vix.Close.astype(float).reindex(target.index).ffill(limit=3)
    spy_s = spy.Close.astype(float).reindex(target.index).ffill(limit=3)
    spy20 = spy_s.pct_change(20)
    out = pd.Series('Range', index=target.index, dtype='object')
    trend = (ax >= 24) & ((e20/e50-1).abs() >= .008)
    highvol = (a_pct >= .75)
    riskoff = (vix_s >= 25) & (spy20 <= -.03)
    out.loc[trend] = 'Trend'
    out.loc[highvol] = 'High Volatility'
    out.loc[riskoff] = 'Risk-Off'
    return out


def session_features(h):
    df = h.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, 'tz', None) is None:
        idx = idx.tz_localize('UTC')
    try:
        local = idx.tz_convert('America/New_York')
    except Exception:
        local = idx
    df['_local_date'] = pd.Index(local.date)
    df['_local_hour'] = local.hour
    tp = (df.High + df.Low + df.Close) / 3
    vol = df.get('Volume', pd.Series(0.0,index=df.index)).fillna(0).astype(float)
    pv = tp * vol
    grp = df['_local_date']
    cumvol = vol.groupby(grp).cumsum().replace(0,np.nan)
    svwap = pv.groupby(grp).cumsum() / cumvol
    daily_high = df.High.groupby(grp).transform('max')
    daily_low = df.Low.groupby(grp).transform('min')
    sess = df.groupby('_local_date').agg(high=('High','max'), low=('Low','min'), close=('Close','last'))
    prev_hi = sess.high.shift(1); prev_lo=sess.low.shift(1); prev_cl=sess.close.shift(1)
    map_hi = grp.map(prev_hi); map_lo=grp.map(prev_lo); map_cl=grp.map(prev_cl)
    pos = df.groupby('_local_date').cumcount()
    first4_hi = df.High.where(pos < 4).groupby(grp).transform('max')
    first4_lo = df.Low.where(pos < 4).groupby(grp).transform('min')
    first4_hi = first4_hi.groupby(grp).ffill(); first4_lo = first4_lo.groupby(grp).ffill()
    return pd.DataFrame({
        'session_vwap': svwap,
        'day_range_pos': (df.Close-daily_low)/(daily_high-daily_low).replace(0,np.nan)-.5,
        'prev_high_gap': df.Close/map_hi.astype(float)-1,
        'prev_low_gap': df.Close/map_lo.astype(float)-1,
        'prev_close_gap': df.Close/map_cl.astype(float)-1,
        'opening_high_gap': df.Close/first4_hi-1,
        'opening_low_gap': df.Close/first4_lo-1,
        'hour_sin': np.sin(2*np.pi*df['_local_hour']/24),
        'hour_cos': np.cos(2*np.pi*df['_local_hour']/24),
    }, index=df.index)


def intraday_features(h):
    c=h.Close.astype(float); hi=h.High.astype(float); lo=h.Low.astype(float)
    v=h.get('Volume',pd.Series(0.0,index=h.index)).fillna(0).astype(float)
    x=pd.DataFrame(index=h.index)
    for n in (1,2,4,8,12,24,48,120): x[f'ret{n}']=c.pct_change(n)
    for span in (8,21,55,144): x[f'ema{span}_gap']=c/c.ewm(span=span,adjust=False).mean()-1
    x['rsi14']=(rsi(c)-50)/50
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=macd.ewm(span=9,adjust=False).mean()
    x['macd_gap']=(macd-ms)/c
    x['atr14_pct']=atr(h,14)/c
    x['vol12']=c.pct_change().rolling(12).std(); x['vol48']=c.pct_change().rolling(48).std()
    x['vol_expansion']=x['vol12']/x['vol48'].replace(0,np.nan)-1
    x['range24']=(c-lo.rolling(24).min())/(hi.rolling(24).max()-lo.rolling(24).min()).replace(0,np.nan)-.5
    x['volume_ratio20']=v/v.rolling(20).mean().replace(0,np.nan)
    x['volume_accel']=v.rolling(4).mean()/v.rolling(24).mean().replace(0,np.nan)-1
    sf=session_features(h)
    x=x.join(sf)
    x['session_vwap_gap']=c/x['session_vwap']-1
    return x.drop(columns=['session_vwap']).replace([np.inf,-np.inf],np.nan)


def model_templates():
    return {
        'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10.0))]),
        'Random Forest':RandomForestRegressor(n_estimators=180,max_depth=6,min_samples_leaf=12,max_features='sqrt',random_state=42,n_jobs=-1),
        'Gradient Boost':HistGradientBoostingRegressor(max_iter=150,max_depth=3,learning_rate=.04,l2_regularization=.5,random_state=42),
    }


def hourly_regime_map(hourly, daily_labels):
    idx=pd.to_datetime(hourly.index)
    naive=idx.tz_localize(None) if getattr(idx,'tz',None) is not None else idx
    dates=pd.Index(naive.normalize())
    shifted=daily_labels.shift(1)
    return pd.Series(dates.map(shifted), index=hourly.index).ffill()


def walkforward_eval(X, y, labels, current_regime, use_regime):
    mods=model_templates(); rec=[]
    candidates=list(range(max(1800,len(X)-1400), len(X), 24))[-40:]
    for o in candidates:
        if o >= len(X) or labels.iloc[o] != current_regime: continue
        train_end=o-4
        if train_end < 1200: continue
        mask=np.arange(len(X)) < train_end
        if use_regime: mask = mask & (labels.values == current_regime)
        idx=np.where(mask)[0]
        if len(idx) < 700: continue
        preds={}; ok=True
        for name,tpl in mods.items():
            try:
                m=clone(tpl);m.fit(X.iloc[idx],y.iloc[idx]);preds[name]=float(m.predict(X.iloc[[o]])[0])
            except Exception:
                ok=False;break
        if ok: rec.append({'actual':float(y.iloc[o]),'preds':preds,'baseline':float(y.iloc[idx].tail(756).median())})
    if len(rec) < 12: return None
    actual=np.array([r['actual'] for r in rec]); base=np.array([r['baseline'] for r in rec])
    base_mae=float(mean_absolute_error(actual,base));base_dir=float(np.mean((base>=0)==(actual>=0)))
    names=list(mods); weights=[]
    for name in names:
        pr=np.array([r['preds'][name] for r in rec]);mae=float(mean_absolute_error(actual,pr));dacc=float(np.mean((pr>=0)==(actual>=0)));skill=1-mae/max(base_mae,1e-9);edge=dacc-base_dir
        weights.append(max(.08,1+np.clip(skill,-.25,.25)*2.2+np.clip(edge,-.15,.15)*2.5))
    w=np.array(weights);w/=w.sum();ens=np.array([sum(r['preds'][n]*ww for n,ww in zip(names,w)) for r in rec]);mae=float(mean_absolute_error(actual,ens));dacc=float(np.mean((ens>=0)==(actual>=0)));skill=1-mae/max(base_mae,1e-9);edge=dacc-base_dir
    return {'mae':mae,'dacc':dacc,'skill':skill,'edge':edge,'base_mae':base_mae,'base_dir':base_dir,'records':rec,'weights':w}


def current_projection(X,y,labels,current_regime,eval_generic,eval_regime):
    chosen='Regime-specific' if eval_regime and (eval_regime['edge'] > eval_generic['edge']+.01 or eval_regime['skill'] > eval_generic['skill']+.02) else 'All-market enriched'
    ev=eval_regime if chosen=='Regime-specific' else eval_generic
    mask=np.ones(len(X),dtype=bool)
    if chosen=='Regime-specific': mask=(labels.values==current_regime)
    idx=np.where(mask)[0];mods=model_templates();cur=[];names=list(mods)
    for name,tpl in mods.items():
        m=clone(tpl);m.fit(X.iloc[idx],y.iloc[idx]);cur.append(float(m.predict(X.iloc[[-1]])[0]))
    w=ev['weights'];pret=float(np.dot(cur,w));last_idx=X.index[-1]
    resid=np.array([r['actual']-sum(r['preds'][n]*ww for n,ww in zip(names,w)) for r in ev['records']]);q=float(np.quantile(np.abs(resid),.60));pup=float(np.clip(np.mean((pret+resid)>0),.15,.85))
    return chosen,ev,pret,q,pup,cur,last_idx


def build_asset(key,cfg):
    daily=download(cfg['ticker'],'10y','1d'); hourly=download(cfg['ticker'],'730d','1h')
    vix=download('^VIX','10y','1d'); spy=download('SPY','10y','1d')
    dlabels=daily_regimes(daily,vix,spy); current_regime=str(dlabels.iloc[-1])
    feats=intraday_features(hourly); fwd=hourly.Close.shift(-4)/hourly.Close-1; labels=hourly_regime_map(hourly,dlabels)
    valid=feats.notna().all(axis=1)&fwd.notna()&labels.notna();X=feats.loc[valid];y=fwd.loc[valid];labels=labels.loc[valid]
    if len(X)<2200: raise RuntimeError('Not enough enriched hourly history')
    generic=walkforward_eval(X,y,labels,current_regime,False)
    if generic is None: raise RuntimeError('Generic 4H walk-forward unavailable')
    regime=walkforward_eval(X,y,labels,current_regime,True)
    chosen,ev,pret,q,pup,cur,last_index=current_projection(X,y,labels,current_regime,generic,regime)
    last=float(hourly.Close.loc[last_index]);price=last*(1+pret);low=last*(1+pret-q);high=last*(1+pret+q)
    directions=[r>=0 for r in cur];maj=sum(directions)>=len(directions)/2;agree=sum(d==maj for d in directions)/len(directions)
    conf='High' if ev['edge']>=.06 and ev['skill']>=.08 and agree>=.67 and len(ev['records'])>=20 else ('Moderate' if ev['edge']>=.03 and ev['skill']>=.02 and agree>=.67 and len(ev['records'])>=16 else 'Low')
    enhanced={'horizon':'4 Hours','steps':4,'model_price':round(price,2),'projected_return_pct':round(pret*100,3),'probability_up':round(pup,4),'probability_down':round(1-pup,4),'confidence':conf,'tight_model_zone':[round(low,2),round(high,2)],'zone_target_coverage':.60,'backtest_directional_accuracy':round(ev['dacc'],4),'baseline_directional_accuracy':round(ev['base_dir'],4),'directional_edge':round(ev['edge'],4),'backtest_mae_pct':round(ev['mae']*100,3),'baseline_mae_pct':round(ev['base_mae']*100,3),'mae_skill_vs_baseline':round(ev['skill'],4),'model_agreement':round(float(agree),4),'walkforward_origins':len(ev['records']),'validation':'purged hourly walk-forward with intraday/session features','selected_model':chosen,'market_regime':current_regime,'models':[{'name':n,'projected_return_pct':round(r*100,3),'weight':round(float(w),4)} for n,r,w in zip(model_templates().keys(),cur,ev['weights'])]}
    comparison={'all_market':{'directional_accuracy':round(generic['dacc'],4),'directional_edge':round(generic['edge'],4),'mae_skill':round(generic['skill'],4),'origins':len(generic['records'])},'regime_specific':None if regime is None else {'directional_accuracy':round(regime['dacc'],4),'directional_edge':round(regime['edge'],4),'mae_skill':round(regime['skill'],4),'origins':len(regime['records'])}}
    pp=DATA/cfg['projection'];payload=json.loads(pp.read_text(encoding='utf-8'))
    payload['projections']=[enhanced if x.get('horizon')=='4 Hours' else x for x in payload.get('projections',[])]
    payload['note']='4H now uses enriched intraday/session data and chooses regime-specific training only when walk-forward tests improve. 1D/1W/1Y remain purged walk-forward projections.'
    pp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    out={'status':'ok','asset':key,'updated_utc':datetime.now(timezone.utc).isoformat(),'current_regime':current_regime,'selected_4h_model':chosen,'comparison':comparison,'enhanced_4h':enhanced,'feature_explanation':{'session_vwap':'where price sits versus today\'s volume-weighted average','previous_day_levels':'distance from previous session high/low/close','opening_range':'distance from early-session range','volume_acceleration':'whether recent volume is increasing versus the day','volatility_expansion':'whether short-term movement is expanding versus normal','hourly_momentum':'hourly RSI/MACD/EMA structure'}}
    (DATA/cfg['output']).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(key,current_regime,chosen,enhanced['model_price'],enhanced['directional_edge'],enhanced['mae_skill_vs_baseline'])


def main():
    for k,c in ASSETS.items(): build_asset(k,c)

if __name__=='__main__': main()
