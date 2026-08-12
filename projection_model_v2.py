import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "gold": {"ticker": "GC=F", "name": "COMEX Gold Futures", "output": "gold_projections.json"},
    "silver": {"ticker": "SI=F", "name": "COMEX Silver Futures", "output": "silver_projections.json"},
}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def download(ticker, period, interval):
    periods = [period]
    if interval == "1h":
        periods = ["2y", "1y"]
    last_err = None
    for p in periods:
        try:
            df = flatten(yf.download(ticker, period=p, interval=interval, auto_adjust=False, progress=False, threads=False))
            if not df.empty and "Close" in df.columns:
                df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
                if not df.empty:
                    return df
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"No usable data returned for {ticker} {interval}: {last_err}")


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def features(df, intraday=False):
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    x = pd.DataFrame(index=df.index)
    periods = (1, 2, 4, 8, 12, 24, 48, 120) if intraday else (1, 2, 5, 10, 20, 60, 120, 252)
    for n in periods:
        x[f"ret{n}"] = c.pct_change(n)
    spans = (8, 21, 55, 144) if intraday else (10, 20, 50, 100, 200)
    for span in spans:
        e = c.ewm(span=span, adjust=False).mean()
        x[f"ema{span}_gap"] = c / e - 1
        x[f"ema{span}_slope"] = e / e.shift(5 if intraday else 10) - 1
    x["rsi14"] = (rsi(c) - 50) / 50
    x["atr14_pct"] = atr(df) / c
    x["vol20"] = c.pct_change().rolling(20).std()
    x["vol60"] = c.pct_change().rolling(60).std()
    x["vol_ratio"] = x["vol20"] / x["vol60"].replace(0, np.nan)
    x["z20"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    x["z60"] = (c - c.rolling(60).mean()) / c.rolling(60).std().replace(0, np.nan)
    x["range20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan) - 0.5
    x["range60"] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min()).replace(0, np.nan) - 0.5
    x["body_pct"] = (c - o) / c
    x["upper_wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    x["lower_wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    x["macd_gap"] = (macd - macd_sig) / c
    if v.abs().sum() > 0:
        x["volume_ratio20"] = v / v.rolling(20).mean().replace(0, np.nan)
        x["volume_accel"] = v.rolling(4 if intraday else 5).mean() / v.rolling(24 if intraday else 20).mean().replace(0, np.nan) - 1
    return x.replace([np.inf, -np.inf], np.nan)


def templates(intraday=False):
    min_leaf = 14 if intraday else 12
    return {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=12.0 if intraday else 10.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=180, max_depth=6, min_samples_leaf=min_leaf, max_features="sqrt", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=180, max_depth=7, min_samples_leaf=min_leaf, max_features="sqrt", random_state=43, n_jobs=-1),
        "Gradient Boost": HistGradientBoostingRegressor(max_iter=140, max_depth=3, learning_rate=0.04, l2_regularization=0.6, random_state=42),
    }


def origins(n, steps, intraday, label):
    if intraday:
        min_train, spacing, maxn = 1500, 24, 28
    elif label == "1 Year":
        min_train, spacing, maxn = 1260, 42, 15
    else:
        min_train, spacing, maxn = 900, 5, 36
    return list(range(min_train + steps, n, spacing))[-maxn:]


def safe_direction_accuracy(pred, actual):
    return float(np.mean((pred >= 0) == (actual >= 0)))


def trust_score(skill, edge, agreement, n, label):
    score = 0.25
    score += np.clip(skill, -0.10, 0.20) * 1.8
    score += np.clip(edge, -0.10, 0.15) * 1.4
    score += max(0.0, agreement - 0.5) * 0.35
    score += min(0.10, n / 500.0)
    cap = 0.55 if label == "1 Year" else 0.82
    return float(np.clip(score, 0.15, cap))


def project(df, steps, label, intraday=False):
    x = features(df, intraday)
    fwd = df["Close"].shift(-steps) / df["Close"] - 1
    valid = x.notna().all(axis=1) & fwd.notna()
    X, y = x.loc[valid], fwd.loc[valid]
    mods = templates(intraday)
    records = []
    if len(X) < 1000:
        raise RuntimeError(f"Insufficient data for {label}")
    for o in origins(len(X), steps, intraday, label):
        tr = o - steps
        if tr < 700:
            continue
        Xtr, ytr = X.iloc[:tr], y.iloc[:tr]
        preds = {}
        for name, tpl in mods.items():
            try:
                m = clone(tpl)
                m.fit(Xtr, ytr)
                preds[name] = float(m.predict(X.iloc[[o]])[0])
            except Exception:
                continue
        if len(preds) >= 3:
            baseline = float(ytr.tail(min(756, len(ytr))).median())
            records.append({"actual": float(y.iloc[o]), "baseline": baseline, "preds": preds})
    if len(records) < 12:
        raise RuntimeError(f"Too few purged walk-forward origins for {label}")
    actual = np.array([r["actual"] for r in records], dtype=float)
    basepred = np.array([r["baseline"] for r in records], dtype=float)
    base_mae = float(mean_absolute_error(actual, basepred))
    base_dir = safe_direction_accuracy(basepred, actual)
    model_names = [name for name in mods if all(name in r["preds"] for r in records)]
    if len(model_names) < 3:
        raise RuntimeError(f"Too few complete ensemble models for {label}")
    metrics = {}
    raw_weights = []
    for name in model_names:
        pred = np.array([r["preds"][name] for r in records])
        mae = float(mean_absolute_error(actual, pred))
        dacc = safe_direction_accuracy(pred, actual)
        skill = 1 - mae / max(base_mae, 1e-9)
        edge = dacc - base_dir
        weight = max(0.08, 1 + np.clip(skill, -0.25, 0.25) * 2.6 + np.clip(edge, -0.15, 0.15) * 2.8)
        metrics[name] = (mae, dacc, skill, edge, weight)
        raw_weights.append(weight)
    w = np.asarray(raw_weights, dtype=float)
    w /= w.sum()
    ens = np.array([sum(r["preds"][n] * ww for n, ww in zip(model_names, w)) for r in records], dtype=float)
    raw_mae = float(mean_absolute_error(actual, ens))
    raw_dacc = safe_direction_accuracy(ens, actual)
    raw_skill = 1 - raw_mae / max(base_mae, 1e-9)
    raw_edge = raw_dacc - base_dir
    latest = x.dropna().iloc[[-1]]
    current_returns = []
    details = []
    directions = []
    for name, ww in zip(model_names, w):
        m = clone(mods[name])
        m.fit(X, y)
        ret = float(m.predict(latest)[0])
        current_returns.append(ret)
        directions.append(ret >= 0)
        mm = metrics[name]
        details.append({"name": name, "projected_return_pct": round(ret * 100, 3), "backtest_mae_pct": round(mm[0] * 100, 3), "directional_accuracy": round(mm[1], 4), "mae_skill_vs_baseline": round(mm[2], 4), "weight": round(float(ww), 4)})
    majority = sum(directions) >= len(directions) / 2
    agreement = float(sum(d == majority for d in directions) / len(directions))
    raw_current_return = float(np.dot(np.asarray(current_returns), w))
    baseline_now = float(y.tail(min(756, len(y))).median())
    trust = trust_score(raw_skill, raw_edge, agreement, len(records), label)
    cal_oos = trust * ens + (1 - trust) * basepred
    cal_resid_prebias = actual - cal_oos
    bias = float(np.median(cal_resid_prebias))
    bias_cap = float(np.quantile(np.abs(cal_resid_prebias), 0.40))
    bias = float(np.clip(bias, -bias_cap, bias_cap) * 0.5)
    cal_oos = cal_oos + bias
    cal_mae = float(mean_absolute_error(actual, cal_oos))
    cal_dacc = safe_direction_accuracy(cal_oos, actual)
    cal_skill = 1 - cal_mae / max(base_mae, 1e-9)
    cal_edge = cal_dacc - base_dir
    resid = actual - cal_oos
    calibrated_return = trust * raw_current_return + (1 - trust) * baseline_now + bias
    last = float(df["Close"].iloc[-1])
    raw_price = last * (1 + raw_current_return)
    predicted_price = last * (1 + calibrated_return)
    q_focus = float(np.quantile(np.abs(resid), 0.40))
    q_risk = float(np.quantile(np.abs(resid), 0.80))
    focus_low = min(last * (1 + calibrated_return - q_focus), predicted_price)
    focus_high = max(last * (1 + calibrated_return + q_focus), predicted_price)
    risk_low = min(last * (1 + calibrated_return - q_risk), focus_low)
    risk_high = max(last * (1 + calibrated_return + q_risk), focus_high)
    pup = float(np.clip(np.mean((calibrated_return + resid) > 0), 0.15, 0.85))
    confidence_score = 45 + np.clip(cal_skill, -0.15, 0.20) * 120 + np.clip(cal_edge, -0.15, 0.15) * 100 + max(0, agreement - 0.5) * 30 + min(8, len(records) / 6)
    if label == "1 Year":
        confidence_score -= 12
    confidence_score = int(np.clip(round(confidence_score), 10, 90))
    confidence = "High" if confidence_score >= 70 else ("Moderate" if confidence_score >= 55 else "Low")
    validated = bool(cal_skill > 0 and cal_edge >= 0.02 and len(records) >= (14 if label == "1 Year" else 24))
    return {
        "horizon": label,
        "steps": steps,
        "predicted_price": round(predicted_price, 2),
        "model_price": round(predicted_price, 2),
        "raw_ml_price": round(raw_price, 2),
        "projected_return_pct": round(calibrated_return * 100, 3),
        "raw_ml_return_pct": round(raw_current_return * 100, 3),
        "baseline_return_pct": round(baseline_now * 100, 3),
        "ml_trust": round(trust, 3),
        "bias_correction_pct": round(bias * 100, 3),
        "probability_up": round(pup, 4),
        "probability_down": round(1 - pup, 4),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "forecast_status": "Validated" if validated else "Estimate only",
        "focus_zone": [round(float(focus_low), 2), round(float(focus_high), 2)],
        "risk_zone": [round(float(risk_low), 2), round(float(risk_high), 2)],
        "tight_model_zone": [round(float(focus_low), 2), round(float(focus_high), 2)],
        "zone_target_coverage": 0.40,
        "backtest_directional_accuracy": round(cal_dacc, 4),
        "baseline_directional_accuracy": round(base_dir, 4),
        "directional_edge": round(cal_edge, 4),
        "backtest_mae_pct": round(cal_mae * 100, 3),
        "baseline_mae_pct": round(base_mae * 100, 3),
        "mae_skill_vs_baseline": round(cal_skill, 4),
        "raw_ml_mae_skill_vs_baseline": round(raw_skill, 4),
        "model_agreement": round(agreement, 4),
        "walkforward_origins": len(records),
        "validation": "purged expanding-origin walk-forward with OOS shrinkage calibration",
        "models": details,
    }


def run_asset(key, cfg):
    daily = download(cfg["ticker"], "10y", "1d")
    hourly = download(cfg["ticker"], "2y", "1h")
    projections = [project(hourly, 4, "4 Hours", True), project(daily, 1, "1 Day"), project(daily, 5, "1 Week"), project(daily, 252, "1 Year")]
    payload = {"status": "ok", "asset": key, "symbol": cfg["ticker"], "instrument": cfg["name"], "updated_utc": datetime.now(timezone.utc).isoformat(), "latest_price": round(float(hourly["Close"].iloc[-1]), 2), "projections": projections, "note": "Calibrated price model. The point forecast is shrunk toward a simple baseline when walk-forward evidence is weak. Focus zones are narrower OOS-error bands; risk zones preserve wider uncertainty. Research/education only."}
    (OUT_DIR / cfg["output"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    for key, cfg in ASSETS.items():
        run_asset(key, cfg)


if __name__ == "__main__":
    main()
