"""Fixed-recipe purged walk-forward evaluation. Probabilities are distinct from evidence scores."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, log_loss

FEATURES = ["return1","return5","return20","trend","rsi","atr","range_position","flow","gap_fraction"]
GATES = {"min_oos":100,"edge":.03,"mae_skill":.03,"min_holdout":12,"confidence_lower":0,"brier_skill":0}
def features(f):
    x = pd.DataFrame(index=f.index)
    x["return1"],x["return5"],x["return20"] = (f.close.pct_change(k) for k in (1,5,20))
    x["trend"] = (f.ema20-f.ema50)/f.close
    x["rsi"] = (f.rsi-50)/50
    x["atr"] = f.atr/f.close
    x["range_position"] = (f.close-f.donchian_low)/(f.donchian_high-f.donchian_low).replace(0,np.nan)-.5
    x["flow"] = f.cmf
    x["gap_fraction"] = f.gap_before.astype(float).rolling(20,min_periods=20).mean()
    return x.replace([np.inf,-np.inf],np.nan)

def metrics(records):
    if not records:
        return {"n":0}
    y,p,prob,base,prior,lo,hi,atr = (np.asarray([r[k] for r in records],dtype=float) for k in ("actual_return","predicted_return","probability_up","baseline_return","prior_up","lower_return","upper_return","atr_fraction"))
    labels = y>0
    wins = ((p>0)==labels).astype(float)
    baseline_wins = ((base>0)==labels).astype(float)
    mae = np.mean(abs(y-p))
    bmae = np.mean(abs(y-base))
    brier = np.mean((prob-labels)**2)
    base_brier = np.mean((prior-labels)**2)
    difference = wins-baseline_wins
    rng = np.random.default_rng(739)
    block = max(2,int(len(y)**(1/3)))
    bootstrap=[]
    for _ in range(600):
        starts = rng.integers(0,len(y),int(np.ceil(len(y)/block)))
        idx = np.concatenate([(start+np.arange(block))%len(y) for start in starts])[:len(y)]
        bootstrap.append(float(difference[idx].mean()))
    bins=[]
    for left in (0,.2,.4,.6,.8):
        mask=(prob>=left)&(prob<left+.2 if left<.8 else prob<=1)
        if mask.any():
            bins.append({"from":left,"to":left+.2,"n":int(mask.sum()),"mean_probability":float(prob[mask].mean()),"observed_up":float(labels[mask].mean())})
    selective=(np.maximum(prob,1-prob)>=.6)&((prob>=.5)==(p>0))
    return {"n":len(y),"directional_accuracy":float(wins.mean()),"baseline_accuracy":float(baseline_wins.mean()),"edge":float(difference.mean()),"edge_ci95":[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))],"balanced_accuracy":float(balanced_accuracy_score(labels,p>0)),"mcc":float(matthews_corrcoef(labels,p>0)),"mae_percent":float(mae*100),"baseline_mae_percent":float(bmae*100),"mae_skill":float(1-mae/bmae) if bmae else None,"zero_change_mae_percent":float(abs(y).mean()*100),"atr_normalized_mae":float(np.mean(abs(y-p)/atr)),"brier":float(brier),"baseline_brier":float(base_brier),"brier_skill":float(1-brier/base_brier) if base_brier else None,"log_loss":float(log_loss(labels,prob,labels=[False,True])),"coverage80":float(np.mean((y>=lo)&(y<=hi))),"mean_interval_width_percent":float(np.mean(hi-lo)*100),"calibration_bins":bins,"selective_n":int(selective.sum()),"selective_coverage":float(selective.mean()),"selective_accuracy":float(wins[selective].mean()) if selective.any() else None}

def train(x,y,fit,cal):
    reg = make_pipeline(StandardScaler(),Ridge(alpha=20))
    clf = make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=500))
    reg.fit(x.iloc[fit],y.iloc[fit])
    clf.fit(x.iloc[fit],(y.iloc[fit]>0).astype(int))
    raw = clf.decision_function(x.iloc[cal]).reshape(-1,1)
    calibration = LogisticRegression(C=1,max_iter=300).fit(raw,(y.iloc[cal]>0).astype(int))
    residual = y.iloc[cal].to_numpy()-reg.predict(x.iloc[cal])
    low,high = np.quantile(residual,[.1,.9])
    return reg,clf,calibration,float(low),float(high)

def evaluate(frame, horizon, hourly=False):
    x = features(frame)
    y = frame.close.shift(-horizon)/frame.close-1
    n = len(frame)
    pos = np.arange(n)
    # A suspect gap anywhere in a feature/target dependency excludes that origin.
    bad = (frame.gap_before | frame.roll_gap_proxy).astype(int)
    # Historical missing bars are known at the origin and represented explicitly.
    # Indicators use observed completed bars, never invented fills. Roll dependencies remain excluded.
    past_bad = frame.roll_gap_proxy.astype(int).rolling(61,min_periods=1).max().astype(bool)
    future_bad = bad.iloc[::-1].rolling(horizon+1,min_periods=1).max().iloc[::-1].astype(bool)
    valid_x = x.notna().all(axis=1) & ~past_bad
    valid = valid_x & y.notna() & ~future_bad
    minfit = 1500 if hourly else 756
    calspan = max(504 if hourly else 252,30*horizon)
    holdspan = 60*23 if hourly else 252
    blockspan = 504 if hourly else 126
    hold_start = n-holdspan-horizon
    start = minfit+calspan+2*horizon
    records=[]
    def indices(start_test):
        cal_start = start_test-calspan-horizon
        fit = pos[(pos<cal_start-horizon)&valid.to_numpy()]
        cal = pos[(pos>=cal_start)&(pos<start_test-horizon)&(pos%horizon==0)&valid.to_numpy()]
        return fit,cal
    for test_start in list(range(start,max(start,hold_start),blockspan))+[max(start,hold_start)]:
        if test_start>=n-horizon:
            continue
        end = min(test_start+blockspan,hold_start) if test_start<hold_start else n-horizon
        test = pos[(pos>=test_start)&(pos<end)&(pos%horizon==0)&valid.to_numpy()]
        fit,cal = indices(test_start)
        if len(fit)<minfit//2 or len(cal)<20 or len(test)==0 or y.iloc[fit].gt(0).nunique()<2 or y.iloc[cal].gt(0).nunique()<2:
            continue
        reg,clf,calibrator,low,high = train(x,y,fit,cal)
        predictions = reg.predict(x.iloc[test])
        probabilities = calibrator.predict_proba(clf.decision_function(x.iloc[test]).reshape(-1,1))[:,1]
        baseline = float(y.iloc[fit].median())
        prior = float((y.iloc[fit]>0).mean())
        for i,p,prob in zip(test,predictions,probabilities):
            records.append({"origin":frame.index[i].isoformat(),"target_time":frame.index[i+horizon].isoformat(),"partition":"holdout" if test_start>=hold_start else "walk_forward","fit_end":frame.index[fit[-1]+horizon].isoformat(),"calibration_start":frame.index[cal[0]].isoformat(),"calibration_end":frame.index[cal[-1]+horizon].isoformat(),"test_start":frame.index[test[0]].isoformat(),"actual_return":float(y.iloc[i]),"predicted_return":float(p),"probability_up":float(prob),"baseline_return":baseline,"prior_up":prior,"lower_return":float(p+low),"upper_return":float(p+high),"atr_fraction":float(frame.atr.iloc[i]/frame.close.iloc[i])})
    oos = metrics([r for r in records if r["partition"]=="walk_forward"])
    holdout = metrics([r for r in records if r["partition"]=="holdout"])
    live = None
    ood = True
    fit,cal = indices(n-1)
    if len(cal)>=20 and len(fit)>=minfit//2 and valid_x.iloc[-1] and y.iloc[cal].gt(0).nunique()==2 and y.iloc[fit].gt(0).nunique()==2:
        reg,clf,calibrator,low,high = train(x,y,fit,cal)
        p = float(reg.predict(x.iloc[[-1]])[0])
        probability = float(calibrator.predict_proba(clf.decision_function(x.iloc[[-1]]).reshape(-1,1))[0,1])
        scaler=reg[0]
        dist = np.max(abs(scaler.transform(x.iloc[fit])),axis=1)
        live_dist=float(np.max(abs(scaler.transform(x.iloc[[-1]]))))
        ood=live_dist>max(4,float(np.quantile(dist,.995)))
        price=float(frame.close.iloc[-1])
        live={"origin":frame.index[-1].isoformat(),"reference_price":price,"price":price*(1+p),"return":p,"probability_up":probability,"interval80":[price*(1+p+low),price*(1+p+high)],"feature_coefficients":dict(zip(FEATURES,map(float,reg[1].coef_))),"calibration_n":len(cal),"ood_distance":live_dist}
    checks = {"oos_sample":oos.get("n",0)>=100,"oos_edge":oos.get("edge",-1)>=.03,"edge_confidence":oos.get("edge_ci95",[-1])[0]>0,"oos_mae":(oos.get("mae_skill") or -1)>=.03,"oos_brier":(oos.get("brier_skill") or -1)>0,"holdout_sample":holdout.get("n",0)>=12,"holdout_edge":holdout.get("edge",-1)>0,"holdout_mae":(holdout.get("mae_skill") or -1)>0,"holdout_brier":(holdout.get("brier_skill") or -1)>0,"interval_coverage":.68<=holdout.get("coverage80",0)<=.9,"in_distribution":not ood,"live_available":live is not None,"positive_ordered_prices":live is not None and 0<live["interval80"][0]<live["interval80"][1] and live["price"]>0}
    status="VALIDATED" if all(checks.values()) else "WAIT"
    return {"status":status,"horizon_bars":horizon,"unit":"completed exchange hourly bars" if hourly else "completed trading sessions","model":"Fixed Ridge return + Logistic/Platt direction; no ensemble selection","recipe_version":"1.1","primary":live if status=="VALIDATED" else None,"research":live,"oos":oos,"holdout":holdout,"checks":checks,"failed_gates":[k for k,v in checks.items() if not v],"thresholds":GATES,"records":records,"integrity":{"training_before_calibration":all(r["fit_end"]<r["calibration_start"] for r in records),"calibration_before_test":all(r["calibration_end"]<r["test_start"] for r in records),"unique_origins":len({r["origin"] for r in records})==len(records),"overlap_policy":"Origins are globally spaced by the horizon; training/calibration boundaries purge immature labels","holdout":"Final 252 daily / 1380 hourly bars; same fixed recipe, no tuning on holdout","data_note":"Latest Yahoo continuous history. Roll mapping and historical data vintages unavailable; future target gaps and past roll dependencies excluded; known historical gaps are an explicit feature."}}
