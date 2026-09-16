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
    wins = (((p>0)==labels) & (abs(p)>1e-6)).astype(float)
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
    return {"n":len(y),"directional_accuracy":float(wins.mean()),"direction_coverage":float(np.mean(abs(p)>1e-6)),"neutral_policy":"Predictions within 0.0001% of zero are abstentions and do not count as directional hits","baseline_accuracy":float(baseline_wins.mean()),"edge":float(difference.mean()),"edge_ci95":[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))],"balanced_accuracy":float(balanced_accuracy_score(labels,p>0)),"mcc":float(matthews_corrcoef(labels,p>0)),"mae_percent":float(mae*100),"baseline_mae_percent":float(bmae*100),"mae_skill":float(1-mae/bmae) if bmae else None,"zero_change_mae_percent":float(abs(y).mean()*100),"atr_normalized_mae":float(np.mean(abs(y-p)/atr)),"brier":float(brier),"baseline_brier":float(base_brier),"brier_skill":float(1-brier/base_brier) if base_brier else None,"log_loss":float(log_loss(labels,prob,labels=[False,True])),"coverage80":float(np.mean((y>=lo)&(y<=hi))),"mean_interval_width_percent":float(np.mean(hi-lo)*100),"calibration_bins":bins,"selective_n":int(selective.sum()),"selective_coverage":float(selective.mean()),"selective_accuracy":float(wins[selective].mean()) if selective.any() else None}

def train(x,y,fit,cal,scale=None,adaptive=False):
    reg = make_pipeline(StandardScaler(),Ridge(alpha=20))
    clf = make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=500))
    target=y if scale is None else y/scale
    reg.fit(x.iloc[fit],target.iloc[fit])
    clf.fit(x.iloc[fit],(y.iloc[fit]>0).astype(int))
    selector=None
    if adaptive:
        # Fixed candidate set. Select only on the FIRST half of past calibration labels.
        # The later half remains untouched until probability/interval calibration.
        selection,cal=np.array_split(cal,2)
        drift=float(y.iloc[fit].median())
        raw_prediction=reg.predict(x.iloc[selection])*scale.iloc[selection].to_numpy()
        candidates=[(0.,0.),(0.,drift),(.25,0.),(.5,0.),(1.,0.)]
        losses=[float(np.mean(abs(y.iloc[selection].to_numpy()-(w*raw_prediction+b)))) for w,b in candidates]
        best=int(np.argmin(losses))
        weight,offset=candidates[best]
        selector={'ridge_weight':weight,'return_offset':offset,'selection_n':len(selection),
                  'selection_end_position':int(selection[-1]),'interval_n':len(cal),
                  'candidate_mae_percent':[v*100 for v in losses],
                  'candidates':['no change','past median return','25% Ridge','50% Ridge','Ridge']}
    raw = clf.decision_function(x.iloc[cal]).reshape(-1,1)
    calibration = LogisticRegression(C=1,max_iter=300).fit(raw,(y.iloc[cal]>0).astype(int)) if y.iloc[cal].gt(0).nunique()==2 else None
    predicted=reg.predict(x.iloc[cal])
    if selector:
        predicted=selector['ridge_weight']*predicted+selector['return_offset']/scale.iloc[cal].to_numpy()
    residual = target.iloc[cal].to_numpy()-predicted
    low,high = np.quantile(residual,[.1,.9])
    return reg,clf,calibration,float(low),float(high),selector

def predict(reg,clf,calibrator,x,scale,selector,prior):
    p=reg.predict(x)*scale
    if selector:
        p=selector['ridge_weight']*p+selector['return_offset']
    prob=calibrator.predict_proba(clf.decision_function(x).reshape(-1,1))[:,1] if calibrator is not None else np.repeat(prior,len(x))
    return p,prob

def evaluate(frame, horizon, hourly=False, recipe="legacy"):
    if recipe not in ("legacy","volatility_scaled","history_selected"):
        raise ValueError("Unknown fixed recipe")
    scaled=recipe!="legacy"
    adaptive=recipe=="history_selected"
    x = features(frame)
    if adaptive:
        for key in sorted(k for k in frame if k.startswith('macro_')):
            x[key]=frame[key]
            x[key+'_change20']=frame[key].diff(20)
    scale=(frame.atr/frame.close*np.sqrt(horizon)).clip(lower=1e-6) if scaled else pd.Series(1.0,index=frame.index)
    if scaled:
        # Every normalizer is available at the forecast origin, never at the target.
        volatility=(frame.atr/frame.close).clip(lower=1e-6)
        for key,period in (("return1",1),("return5",5),("return20",20),("trend",1)):
            x[key]=x[key]/(volatility*np.sqrt(period))
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
    calspan = max(504 if hourly else 252,(60 if scaled else 30)*horizon)
    holdspan = max(60*23 if hourly else 252,30*horizon) if scaled else (60*23 if hourly else 252)
    blockspan = 504 if hourly else 126
    hold_start = n-holdspan-horizon
    start = minfit+calspan+2*horizon
    records=[]
    def indices(start_test):
        cal_start = start_test-calspan-horizon
        if scaled:
            # Extend by sample availability only, without inspecting outcomes or scores.
            while cal_start>minfit+horizon:
                available=pos[(pos>=cal_start)&(pos<start_test-horizon)&(pos%horizon==0)&valid.to_numpy()]
                if len(available)>=30:
                    break
                cal_start=max(minfit+horizon,cal_start-max(126,10*horizon))
        fit = pos[(pos<cal_start-horizon)&valid.to_numpy()]
        cal = pos[(pos>=cal_start)&(pos<start_test-horizon)&(pos%horizon==0)&valid.to_numpy()]
        return fit,cal
    for test_start in list(range(start,max(start,hold_start),blockspan))+[max(start,hold_start)]:
        if test_start>=n-horizon:
            continue
        end = min(test_start+blockspan,hold_start) if test_start<hold_start else n-horizon
        test = pos[(pos>=test_start)&(pos<end)&(pos%horizon==0)&valid.to_numpy()]
        if scaled and test_start<hold_start:
            test=test[test+horizon<hold_start]
        fit,cal = indices(test_start)
        if len(fit)<minfit//2 or len(cal)<20 or len(test)==0 or y.iloc[fit].gt(0).nunique()<2 or y.iloc[cal].gt(0).nunique()<2:
            continue
        reg,clf,calibrator,low,high,selector = train(x,y,fit,cal,scale if scaled else None,adaptive=adaptive)
        predictions,probabilities = predict(reg,clf,calibrator,x.iloc[test],scale.iloc[test].to_numpy(),selector,float((y.iloc[fit]>0).mean()))
        baseline = float(y.iloc[fit].median())
        prior = float((y.iloc[fit]>0).mean())
        for i,p,prob in zip(test,predictions,probabilities):
            records.append({"origin":frame.index[i].isoformat(),"target_time":frame.index[i+horizon].isoformat(),"partition":"holdout" if test_start>=hold_start else "walk_forward","fit_end":frame.index[fit[-1]+horizon].isoformat(),"calibration_start":frame.index[cal[0]].isoformat(),"calibration_end":frame.index[cal[-1]+horizon].isoformat(),"test_start":frame.index[test[0]].isoformat(),"selection":selector,"actual_return":float(y.iloc[i]),"predicted_return":float(p),"probability_up":float(prob),"baseline_return":baseline,"prior_up":prior,"lower_return":float(p+low*scale.iloc[i]),"upper_return":float(p+high*scale.iloc[i]),"atr_fraction":float(frame.atr.iloc[i]/frame.close.iloc[i])})
    oos = metrics([r for r in records if r["partition"]=="walk_forward"])
    holdout = metrics([r for r in records if r["partition"]=="holdout"])
    live = None
    ood = True
    fit,cal = indices(n-1)
    if len(cal)>=20 and len(fit)>=minfit//2 and valid_x.iloc[-1] and y.iloc[cal].gt(0).nunique()==2 and y.iloc[fit].gt(0).nunique()==2:
        reg,clf,calibrator,low,high,selector = train(x,y,fit,cal,scale if scaled else None,adaptive=adaptive)
        ps,probs=predict(reg,clf,calibrator,x.iloc[[-1]],scale.iloc[[-1]].to_numpy(),selector,float((y.iloc[fit]>0).mean()))
        p,probability=float(ps[0]),float(probs[0])
        scaler=reg[0]
        dist = np.max(abs(scaler.transform(x.iloc[fit])),axis=1)
        live_dist=float(np.max(abs(scaler.transform(x.iloc[[-1]]))))
        ood=live_dist>max(4,float(np.quantile(dist,.995)))
        price=float(frame.close.iloc[-1])
        live={"origin":frame.index[-1].isoformat(),"reference_price":price,"price":price*(1+p),"return":p,"probability_up":probability,"interval80":[price*(1+p+low*scale.iloc[-1]),price*(1+p+high*scale.iloc[-1])],"feature_coefficients":dict(zip(x.columns,map(float,reg[1].coef_))),"calibration_n":len(cal),"calibration_start":frame.index[cal[0]].isoformat(),"calibration_end":frame.index[cal[-1]+horizon].isoformat(),"ood_distance":live_dist,"selection":selector}
    checks = {"oos_sample":oos.get("n",0)>=100,"oos_edge":oos.get("edge",-1)>=.03,"edge_confidence":oos.get("edge_ci95",[-1])[0]>0,"oos_mae":(oos.get("mae_skill") or -1)>=.03,"oos_brier":(oos.get("brier_skill") or -1)>0,"holdout_sample":holdout.get("n",0)>=12,"holdout_edge":holdout.get("edge",-1)>0,"holdout_mae":(holdout.get("mae_skill") or -1)>0,"holdout_brier":(holdout.get("brier_skill") or -1)>0,"interval_coverage":.68<=holdout.get("coverage80",0)<=.9,"in_distribution":not ood,"live_available":live is not None,"positive_ordered_prices":live is not None and 0<live["interval80"][0]<live["interval80"][1] and live["price"]>0}
    if scaled:
        checks["oos_vs_no_change"]=oos.get("mae_percent",float("inf"))<.97*oos.get("zero_change_mae_percent",0)
        checks["holdout_vs_no_change"]=holdout.get("mae_percent",float("inf"))<holdout.get("zero_change_mae_percent",0)
    status="VALIDATED" if all(checks.values()) else "WAIT"
    result = {"status":status,"horizon_bars":horizon,"unit":"completed exchange hourly bars" if hourly else "completed trading sessions","model":"Fixed Ridge return + Logistic/Platt direction; no ensemble selection","recipe_version":"1.1","primary":live if status=="VALIDATED" else None,"research":live,"oos":oos,"holdout":holdout,"checks":checks,"failed_gates":[k for k,v in checks.items() if not v],"thresholds":GATES,"records":records,"integrity":{"training_before_calibration":all(r["fit_end"]<r["calibration_start"] for r in records),"calibration_before_test":all(r["calibration_end"]<r["test_start"] for r in records),"unique_origins":len({r["origin"] for r in records})==len(records),"overlap_policy":"Origins are globally spaced by the horizon; training/calibration boundaries purge immature labels","holdout":"Final 252 daily / 1380 hourly bars; same fixed recipe, no tuning on holdout","data_note":"Latest Yahoo continuous history. Roll mapping and historical data vintages unavailable; future target gaps and past roll dependencies excluded; known historical gaps are an explicit feature."}}

    result["data_evidence"]={"completed_bars":n,"eligible_nonoverlapping_targets":int((valid & (pos%horizon==0)).sum()),"feature_complete_origins":int(valid_x.sum()),"excluded_target_or_feature_origins":int((~valid & y.notna()).sum()),"live_fit_n":len(fit),"live_calibration_n":len(cal),"holdout_start":frame.index[max(0,hold_start)].isoformat(),"last_completed":frame.index[-1].isoformat()}
    if scaled:
        result.update(model="Volatility-scaled Ridge return + Logistic/Platt direction; fixed recipe",recipe_version="2.0")
        result["integrity"].update(holdout=f"Final {holdspan} bars; walk-forward targets purged before this boundary; no model selection on these results",data_note="Latest Yahoo continuous history, not point-in-time vintages. Missing target windows and roll dependencies excluded. This revised recipe requires prospective confirmation; historical results are research evidence.",normalization="ATR at each origin scales returns and interval residuals; no target-period volatility is used",calibration="Non-overlapping completed labels; monthly initial window 1260 bars, extended using sample counts only")
    if adaptive:
        result.update(model="Past-only selection of no-change, historical drift and damped Ridge; release-lagged FRED features when available",recipe_version="3.0")
        result['integrity']['selection']='Fixed candidates selected by MAE on first half of pre-test calibration; second half calibrates intervals and probabilities. No future or final-holdout scores select the recipe.'
        result['integrity']['macro_features']=[k for k in x if k.startswith('macro_')]
        result['integrity']['macro_timing']='FRED initial-release values only; available no earlier than release date + 2 UTC days; no current snapshots inserted into history.'
        if live and live.get('selection'):
            live['selection']['selection_end']=frame.index[live['selection']['selection_end_position']+horizon].isoformat()
    return result


def paired_comparison(current,previous):
    """Compare errors only where both fixed recipes made a forecast for the same target."""
    old={(r["origin"],r["target_time"]):r for r in previous["records"]}
    result={"reference_recipe":previous["recipe_version"],"reference_model":previous["model"],"selection":"Neither recipe is selected or tuned using this comparison"}
    for partition in ("walk_forward","holdout"):
        pairs=[(r,old[(r["origin"],r["target_time"])]) for r in current["records"] if r["partition"]==partition and (r["origin"],r["target_time"]) in old and old[(r["origin"],r["target_time"])]["partition"]==partition]
        if not pairs:
            result[partition]={"n":0}
            continue
        a,b=map(list,zip(*pairs))
        ma,mb=metrics(a),metrics(b)
        result[partition]={"n":len(pairs),"current_mae_percent":ma["mae_percent"],"previous_mae_percent":mb["mae_percent"],"mae_improvement":1-ma["mae_percent"]/mb["mae_percent"] if mb["mae_percent"] else None,"current_coverage80":ma["coverage80"],"previous_coverage80":mb["coverage80"],"current_brier":ma["brier"],"previous_brier":mb["brier"]}
    return result
