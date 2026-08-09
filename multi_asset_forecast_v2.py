import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR=Path('data'); OUT_DIR.mkdir(parents=True,exist_ok=True); HORIZONS=(1,5,20)
ASSETS={
'gold':{'ticker':'GC=F','instrument':'COMEX Gold Futures','output':'live_forecast.json','factors':{'silver':('SI=F','Silver Futures'),'dxy':('DX-Y.NYB','US Dollar Index'),'us10y':('^TNX','US 10Y Yield'),'vix':('^VIX','VIX'),'spy':('SPY','S&P 500 ETF'),'tlt':('TLT','Long Treasury ETF')}},
'silver':{'ticker':'SI=F','instrument':'COMEX Silver Futures','output':'silver_forecast.json','factors':{'gold':('GC=F','Gold Futures'),'dxy':('DX-Y.NYB','US Dollar Index'),'us10y':('^TNX','US 10Y Yield'),'vix':('^VIX','VIX'),'copper':('HG=F','Copper Futures'),'spy':('SPY','S&P 500 ETF')}}}

def flatten(df):
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    return df

def norm(df):
    if df.empty:return df
    idx=pd.to_datetime(df.index)
    if getattr(idx,'tz',None) is not None:idx=idx.tz_localize(None)
    df=df.copy();df.index=idx.normalize();return df[~df.index.duplicated(keep='last')].sort_index()

def daily(ticker,period='10y'):
    df=norm(flatten(yf.download(ticker,period=period,interval='1d',auto_adjust=False,progress=False,threads=False)))
    if df.empty or 'Close' not in df.columns:return pd.DataFrame()
    return df.dropna(subset=[c for c in ('Open','High','Low','Close') if c in df.columns]).copy()

def intraday(ticker):
    df=flatten(yf.download(ticker,period='10d',interval='1h',auto_adjust=False,progress=False,threads=False))
    return df.dropna(subset=['Close']).copy() if not df.empty and 'Close' in df.columns else pd.DataFrame()

def rsi(s,n=14):
    d=s.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();rs=up/dn.replace(0,np.nan);return 100-100/(1+rs)

def atr(df,n=14):
    pc=df.Close.shift(1);tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1);return tr.ewm(alpha=1/n,adjust=False).mean()

def load_factors(index,defs):
    f=pd.DataFrame(index=index);meta={}
    for key,(ticker,name) in defs.items():
        d=daily(ticker)
        if d.empty:meta[key]={'available':False};continue
        s=d.Close.astype(float).reindex(index).ffill(limit=3);f[key]=s;meta[key]={'available':bool(s.notna().sum()>500)}
    return f,meta

def features(target,factors):
    c=target.Close.astype(float);x=pd.DataFrame(index=target.index)
    for n in (1,2,5,10,20,60):x[f'ret{n}']=c.pct_change(n)
    e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean();e200=c.ewm(span=200,adjust=False).mean()
    x['ema20_gap']=c/e20-1;x['ema50_gap']=c/e50-1;x['ema200_gap']=c/e200-1;x['ema20_50']=e20/e50-1;x['ema50_200']=e50/e200-1
    x['rsi14']=(rsi(c)-50)/50;x['vol20']=c.pct_change().rolling(20).std();x['vol60']=c.pct_change().rolling(60).std();x['atr14_pct']=atr(target)/c
    x['z20']=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan);hi=target.High.rolling(20).max();lo=target.Low.rolling(20).min();x['range_pos20']=(c-lo)/(hi-lo).replace(0,np.nan)-.5;x['drawdown60']=c/c.rolling(60).max()-1
    for key in factors.columns:
        s=factors[key].astype(float)
        for n in (1,5,20):x[f'{key}_ret{n}']=s.pct_change(n)
        x[f'{key}_z60']=(s-s.rolling(60).mean())/s.rolling(60).std().replace(0,np.nan);x[f'{key}_ema20_gap']=s/s.ewm(span=20,adjust=False).mean()-1
    if 'dxy' in factors:x['dxy_corr60']=c.pct_change().rolling(60).corr(factors.dxy.pct_change())
    if 'us10y' in factors:x['yield_change5']=factors.us10y.diff(5);x['yield_change20']=factors.us10y.diff(20)
    return x.replace([np.inf,-np.inf],np.nan)

def templates():
    return {'Logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.3,max_iter=3000,class_weight='balanced',random_state=42))]),'Random Forest':RandomForestClassifier(n_estimators=220,max_depth=5,min_samples_leaf=14,max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=-1),'Gradient Boost':HistGradientBoostingClassifier(max_iter=140,max_depth=3,learning_rate=.04,l2_regularization=.35,random_state=42)}

def folds(n,h,min_train=900,block=126,max_folds=4):
    available=max(0,n-min_train-h);k=min(max_folds,max(2,available//block)) if available>=block*2 else 0
    if k<2:return []
    first=n-k*block;out=[]
    for i in range(k):
        ts=first+i*block;te=min(n,ts+block);tr=ts-h
        if tr>=min_train:out.append((tr,ts,te))
    return out

def forecast_one(target,x,h):
    fwd=target.Close.shift(-h)/target.Close-1;y=(fwd>0).astype(int);valid=x.notna().all(axis=1)&fwd.notna();X,Y=x.loc[valid],y.loc[valid]
    if len(X)<1100:raise RuntimeError(f'Insufficient history {h}d')
    fs=folds(len(X),h);latest=x.dropna().iloc[[-1]];mods=templates();oos={n:{'y':[],'p':[]} for n in mods}
    for tr,ts,te in fs:
        for name,tpl in mods.items():
            try:
                m=clone(tpl);m.fit(X.iloc[:tr],Y.iloc[:tr]);p=m.predict_proba(X.iloc[ts:te])[:,1];oos[name]['y']+=Y.iloc[ts:te].astype(int).tolist();oos[name]['p']+=p.tolist()
            except:pass
    rows=[]
    for name,tpl in mods.items():
        yy=np.array(oos[name]['y']);pp=np.array(oos[name]['p'])
        if len(yy)<200:continue
        acc=float(accuracy_score(yy,pp>=.5));bs=float(brier_score_loss(yy,pp));base=float(max(yy.mean(),1-yy.mean()));m=clone(tpl);m.fit(X,Y);now=float(m.predict_proba(latest)[:,1][0]);rows.append((name,now,acc,bs,base,len(yy)))
    if len(rows)<2:raise RuntimeError('Too few walk-forward models')
    w=np.array([max(.08,1+np.clip(a-ba,-.03,.08)*9+np.clip(.25-b,-.05,.08)*3) for _,_,a,b,ba,_ in rows]);w/=w.sum();raw=float(sum(r[1]*ww for r,ww in zip(rows,w)));acc=float(sum(r[2]*ww for r,ww in zip(rows,w)));bs=float(sum(r[3]*ww for r,ww in zip(rows,w)));base=float(sum(r[4]*ww for r,ww in zip(rows,w)));edge=acc-base
    votes=[r[1]>=.5 for r in rows];maj=sum(votes)>=len(votes)/2;agree=float(sum(v==maj for v in votes)/len(votes));rel=float(np.clip((edge+.01)/.06,0,1));p=float(np.clip(.5+(raw-.5)*(.25+.75*rel),.32,.68))
    hist=(target.Close/target.Close.shift(h)-1).dropna().tail(1260);q20,q50,q80=hist.quantile([.2,.5,.8]).values;last=float(target.Close.iloc[-1]);vol=float(target.Close.pct_change().tail(60).std());tilt=(p-.5)*2*vol*np.sqrt(h)*.35;lo,mid,hi=[last*(1+q+tilt) for q in (q20,q50,q80)]
    sig='Bullish bias' if p>=.58 else ('Bearish bias' if p<=.42 else 'Neutral / no strong edge');dist=abs(p-.5);n=min(r[5] for r in rows)
    conf='High' if edge>=.05 and dist>=.10 and agree>=.67 and bs<.245 and n>=300 else ('Moderate' if edge>=.02 and dist>=.06 and agree>=.67 and bs<.255 and n>=250 else 'Low')
    detail=[{'name':r[0],'probability_up':round(r[1],4),'walkforward_accuracy':round(r[2],4),'walkforward_baseline':round(r[4],4),'brier_score':round(r[3],4),'weight':round(float(ww),4),'oos_observations':int(r[5])} for r,ww in zip(rows,w)]
    return {'horizon_days':h,'probability_up':round(p,4),'probability_down':round(1-p,4),'signal':sig,'confidence':conf,'forecast_mid':round(mid,2),'range_20_80':[round(lo,2),round(hi,2)],'backtest_accuracy':round(acc,4),'naive_baseline':round(base,4),'backtest_edge':round(edge,4),'brier_score':round(bs,4),'model_agreement':round(agree,4),'test_observations':int(n),'validation':'purged expanding-window walk-forward','models':detail}

def impact(key,s,target):
    s=s.dropna();r20=float(s.iloc[-1]/s.iloc[-21]-1);t20=float(target.iloc[-1]/target.iloc[-21]-1)
    if key=='dxy':return ('Supportive','Dollar weakened') if r20<=-.007 else (('Headwind','Dollar strengthened') if r20>=.007 else ('Neutral','Dollar mixed'))
    if key=='us10y':
        d=float(s.iloc[-1]-s.iloc[-21]);return ('Supportive','10Y yield fell') if d<=-.10 else (('Headwind','10Y yield rose') if d>=.10 else ('Neutral','10Y yield range-bound'))
    if key=='vix':return ('Risk-off','Volatility elevated') if float(s.iloc[-1])>=28 else (('Risk-on','Volatility low') if float(s.iloc[-1])<=15 else ('Neutral','Volatility moderate'))
    if key=='silver':return ('Supportive','Silver outperforming gold') if r20>0 and r20-t20>.01 else (('Headwind','Silver weak') if r20<-.02 else ('Neutral','Silver mixed'))
    if key=='gold':return ('Supportive','Gold trend supportive') if r20>.015 else (('Headwind','Gold weak') if r20<-.015 else ('Neutral','Gold range-bound'))
    if key=='copper':return ('Supportive','Copper strong') if r20>.025 else (('Headwind','Copper weak') if r20<-.025 else ('Neutral','Copper mixed'))
    if key=='spy':return ('Risk-on','Equities constructive') if r20>.025 else (('Risk-off','Equities weak') if r20<-.04 else ('Neutral','Equities mixed'))
    if key=='tlt':return ('Supportive','Long bonds firm') if r20>.015 else (('Headwind','Long bonds weak') if r20<-.015 else ('Neutral','Bonds mixed'))
    return 'Neutral','Mixed'

def run_asset(key,cfg):
    t=daily(cfg['ticker']);fac,meta=load_factors(t.index,cfg['factors']);fac=fac[[c for c in fac.columns if meta[c]['available']]];x=features(t,fac);fc=[forecast_one(t,x,h) for h in HORIZONS];c=t.Close.astype(float);intra=intraday(cfg['ticker']);latest=float(intra.Close.iloc[-1]) if not intra.empty else float(c.iloc[-1]);prev=float(c.iloc[-2]);change=(latest/prev-1)*100
    e20=float(c.ewm(span=20,adjust=False).mean().iloc[-1]);e50=float(c.ewm(span=50,adjust=False).mean().iloc[-1]);e200=float(c.ewm(span=200,adjust=False).mean().iloc[-1]);rv=float(rsi(c).iloc[-1]);at=float(atr(t).iloc[-1]);vol=c.pct_change().rolling(20).std();hist=vol.dropna().tail(756);vp=float((hist<=vol.iloc[-1]).mean());vr='High' if vp>=.75 else ('Low' if vp<=.25 else 'Normal');trend='Strong uptrend' if e20>e50>e200 else ('Strong downtrend' if e20<e50<e200 else ('Uptrend / transitional' if e20>e50 else 'Downtrend / transitional'))
    fr=[];ms=0
    for k in fac.columns:
        imp,reason=impact(k,fac[k],c);ms+=1 if imp=='Supportive' else (-1 if imp=='Headwind' else 0);s=fac[k].dropna();fr.append({'key':k,'name':cfg['factors'][k][1],'symbol':cfg['factors'][k][0],'value':round(float(s.iloc[-1]),3),'change_5d_pct':round(float((s.iloc[-1]/s.iloc[-6]-1)*100),2),'change_20d_pct':round(float((s.iloc[-1]/s.iloc[-21]-1)*100),2),'impact':imp,'reason':reason})
    macro='Supportive' if ms>=2 else ('Headwind' if ms<=-2 else 'Mixed');f5=next(f for f in fc if f['horizon_days']==5);overall='Bullish' if f5['probability_up']>=.58 and ms>=0 else ('Bearish' if f5['probability_up']<=.42 and ms<=0 else 'Neutral / mixed');reg={'overall':overall,'trend':trend,'macro':macro,'macro_score':ms,'volatility':vr,'volatility_percentile':round(vp,4)};ref=float(c.iloc[-1]);levels={'daily_close_reference':round(ref,2),'support_20d':round(float(t.Low.tail(20).min()),2),'resistance_20d':round(float(t.High.tail(20).max()),2),'support_60d':round(float(t.Low.tail(60).min()),2),'resistance_60d':round(float(t.High.tail(60).max()),2),'atr14_dollars':round(at,2),'expected_move_1d':[round(ref-at,2),round(ref+at,2)],'expected_move_5d':[round(ref-at*np.sqrt(5),2),round(ref+at*np.sqrt(5),2)]}
    sup=[f['name'] for f in fr if f['impact']=='Supportive'];head=[f['name'] for f in fr if f['impact']=='Headwind'];p=f5['probability_up'];bias='Bullish' if p>=.58 else ('Bearish' if p<=.42 else 'Neutral');read=f"Supportive: {', '.join(sup[:2])}. Headwinds: {', '.join(head[:2])}." if sup and head else (f"Supportive backdrop led by {', '.join(sup[:3])}." if sup else (f"Macro headwinds led by {', '.join(head[:3])}." if head else 'Cross-market factors are mixed.'));summary={'bias':bias,'conviction':f5['confidence'],'model_view':read,'trigger':f"Above {levels['resistance_20d']:.2f} strengthens upside; below {levels['support_20d']:.2f} weakens the setup.",'risk':'Walk-forward edge is weak, so confidence should stay limited.' if f5['confidence']=='Low' else 'Macro reversals can invalidate the setup.','regime':overall}
    series=[{'date':pd.Timestamp(i).strftime('%Y-%m-%d'),'close':round(float(r.Close),2)} for i,r in t.tail(260).iterrows()];mt=pd.Timestamp(intra.index[-1]).isoformat() if not intra.empty else pd.Timestamp(t.index[-1]).isoformat();payload={'status':'ok','asset':key,'symbol':cfg['ticker'],'instrument':cfg['instrument'],'updated_utc':datetime.now(timezone.utc).isoformat(),'latest_market_time':mt,'latest_price':round(latest,2),'change_pct':round(change,3),'guru_summary':summary,'regime':reg,'indicators':{'ema20':round(e20,2),'ema50':round(e50,2),'ema200':round(e200,2),'rsi14':round(rv,2),'atr14_pct':round(at/ref*100,2)},'levels':levels,'factors':fr,'forecasts':fc,'series':series,'methodology':'Purged expanding-window walk-forward ensemble using momentum, trend, volatility and cross-market factors.','disclaimer':'Research/education only; probabilistic and can be wrong.'};(OUT_DIR/cfg['output']).write_text(json.dumps(payload,indent=2),encoding='utf-8')

def main():
    for k,c in ASSETS.items():run_asset(k,c)
if __name__=='__main__':main()
