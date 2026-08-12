import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
DATA.mkdir(exist_ok=True)

ASSETS = {
    "gold": {"ticker": "GC=F", "output": "gold_technical_model.json"},
    "silver": {"ticker": "SI=F", "output": "silver_technical_model.json"},
}
HORIZONS = (1, 5, 20)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def download(ticker):
    df = flatten(yf.download(ticker, start="2008-01-01", interval="1d", auto_adjust=False,
                             progress=False, threads=False))
    if df.empty:
        raise RuntimeError(f"No daily data for {ticker}")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / a.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, mdi

def technical_features(df):
    c = df["Close"].astype(float)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    x = pd.DataFrame(index=df.index)

    for n in (1, 2, 3, 5, 10, 20, 40, 60):
        x[f"ret_{n}"] = c.pct_change(n)

    for span in (9, 20, 50, 100, 200):
        e = c.ewm(span=span, adjust=False).mean()
        x[f"ema_gap_{span}"] = c / e - 1
        x[f"ema_slope_{span}"] = e / e.shift(10) - 1

    rr = rsi(c)
    x["rsi14"] = (rr - 50) / 50
    x["rsi_change5"] = rr.diff(5) / 100

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig
    x["macd_gap"] = macd_hist / c
    x["macd_accel"] = macd_hist.diff(3) / c

    ax, pdi, mdi = adx(df)
    x["adx14"] = ax / 100
    x["dmi_spread"] = (pdi - mdi) / 100

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    x["stoch14"] = (stoch - 50) / 50
    x["stoch_change3"] = stoch.diff(3) / 100

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    upper = mid + 2 * sd
    lower = mid - 2 * sd
    x["bb_percent_b"] = (c - lower) / (upper - lower).replace(0, np.nan) - 0.5
    x["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

    a = atr(df)
    x["atr_pct"] = a / c
    x["vol20"] = c.pct_change().rolling(20).std()
    x["vol60"] = c.pct_change().rolling(60).std()
    x["vol_ratio"] = x["vol20"] / x["vol60"].replace(0, np.nan)

    for n in (10, 20, 60):
        hi = h.rolling(n).max()
        lo = l.rolling(n).min()
        x[f"range_pos_{n}"] = (c - lo) / (hi - lo).replace(0, np.nan) - 0.5

    x["body_pct"] = (c - o) / c
    x["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    x["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c

    x["z20"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    x["z60"] = (c - c.rolling(60).mean()) / c.rolling(60).std().replace(0, np.nan)
    x["drawdown60"] = c / c.rolling(60).max() - 1
    x["drawdown252"] = c / c.rolling(252).max() - 1

    if v.abs().sum() > 0:
        x["volume_ratio20"] = v / v.rolling(20).mean().replace(0, np.nan)
        x["volume_accel"] = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan) - 1
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        x["obv_slope10"] = obv.diff(10) / obv.abs().rolling(60).max().replace(0, np.nan)

    x["trend_regime"] = np.sign((c.ewm(span=20, adjust=False).mean() /
                                 c.ewm(span=50, adjust=False).mean()) - 1)
    x["high_vol_regime"] = (x["vol_ratio"] > 1.15).astype(float)

    return x.replace([np.inf, -np.inf], np.nan)

def models():
    return {
        "Logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.18, max_iter=3000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=140, max_depth=5, min_samples_leaf=18, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=140, max_depth=6, min_samples_leaf=18, max_features="sqrt",
            class_weight="balanced", random_state=43, n_jobs=-1
        ),
        "Gradient Boost": HistGradientBoostingClassifier(
            max_iter=110, max_depth=3, learning_rate=0.035, l2_regularization=0.8, random_state=44
        ),
    }

def outer_folds(n, h, min_train=1200, block=126, max_folds=6):
    available = n - min_train - h
    if available < block * 2:
        return []
    k = min(max_folds, available // block)
    first = n - k * block
    out = []
    for i in range(k):
        ts = first + i * block
        te = min(n, ts + block)
        train_end = ts - h
        if train_end >= min_train:
            out.append((train_end, ts, te))
    return out

def fit_platt(raw_p, y):
    raw_p = np.asarray(raw_p, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 60:
        return None
    z = np.log(np.clip(raw_p, 1e-4, 1 - 1e-4) / np.clip(1 - raw_p, 1e-4, 1 - 1e-4)).reshape(-1, 1)
    m = LogisticRegression(C=0.5, max_iter=1000)
    m.fit(z, y)
    return m

def apply_platt(model, p):
    p = np.asarray(p, dtype=float)
    if model is None:
        return p
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1 - 1e-4)).reshape(-1, 1)
    return model.predict_proba(z)[:, 1]

def threshold_from_calibration(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    best = None
    for th in (0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66):
        active = (p >= th) | (p <= 1 - th)
        n = int(active.sum())
        coverage = float(active.mean())
        if n < 45 or coverage < 0.14:
            continue
        pred = (p[active] >= 0.5).astype(int)
        yy = y[active]
        acc = float(accuracy_score(yy, pred))
        base = float(max(yy.mean(), 1 - yy.mean()))
        edge = acc - base
        score = edge + min(coverage, 0.45) * 0.06
        row = (score, th, acc, base, edge, coverage, n)
        if best is None or row[0] > best[0]:
            best = row
    if best is None:
        return 0.66
    return float(best[1])

def horizon_model(df, h):
    X0 = technical_features(df)
    fwd = df["Close"].shift(-h) / df["Close"] - 1
    y0 = (fwd > 0).astype(int)
    valid = X0.notna().all(axis=1) & fwd.notna()
    X, y = X0.loc[valid], y0.loc[valid]

    if len(X) < 1500:
        raise RuntimeError(f"Insufficient technical history for {h}D")

    folds = outer_folds(len(X), h)
    if len(folds) < 3:
        raise RuntimeError(f"Too few walk-forward folds for {h}D")

    templates = models()
    all_y, all_p, all_active = [], [], []
    fold_rows = []
    thresholds = []

    for train_end, test_start, test_end in folds:
        cal_len = min(252, max(126, train_end // 5))
        cal_start = train_end - cal_len
        fit_end = cal_start - h
        if fit_end < 900:
            continue

        Xfit, yfit = X.iloc[:fit_end], y.iloc[:fit_end]
        Xcal, ycal = X.iloc[cal_start:train_end], y.iloc[cal_start:train_end]
        Xtest, ytest = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        cal_probs, test_probs = [], []
        for name, tpl in templates.items():
            try:
                m = clone(tpl)
                m.fit(Xfit, yfit)
                cal_probs.append(m.predict_proba(Xcal)[:, 1])
                test_probs.append(m.predict_proba(Xtest)[:, 1])
            except Exception:
                continue

        if len(cal_probs) < 3:
            continue

        raw_cal = np.mean(np.vstack(cal_probs), axis=0)
        raw_test = np.mean(np.vstack(test_probs), axis=0)
        platt = fit_platt(raw_cal, ycal.values)
        pcal = apply_platt(platt, raw_cal)
        ptest = apply_platt(platt, raw_test)
        th = threshold_from_calibration(pcal, ycal.values)
        thresholds.append(th)

        active = (ptest >= th) | (ptest <= 1 - th)
        yy = ytest.values.astype(int)
        all_y.extend(yy.tolist())
        all_p.extend(ptest.tolist())
        all_active.extend(active.tolist())

        if active.any():
            pred = (ptest[active] >= 0.5).astype(int)
            ysel = yy[active]
            acc = float(accuracy_score(ysel, pred))
            base = float(max(ysel.mean(), 1 - ysel.mean()))
            edge = acc - base
        else:
            acc = base = edge = np.nan

        fold_rows.append({
            "threshold": round(th, 3),
            "coverage": round(float(active.mean()), 4),
            "signals": int(active.sum()),
            "accuracy": round(acc, 4) if np.isfinite(acc) else None,
            "baseline": round(base, 4) if np.isfinite(base) else None,
            "edge": round(edge, 4) if np.isfinite(edge) else None,
        })

    yy = np.asarray(all_y, dtype=int)
    pp = np.asarray(all_p, dtype=float)
    active = np.asarray(all_active, dtype=bool)
    if len(yy) < 300 or active.sum() < 70:
        raise RuntimeError(f"Too few OOS observations/signals for {h}D")

    pred_active = (pp[active] >= 0.5).astype(int)
    y_active = yy[active]
    selective_acc = float(accuracy_score(y_active, pred_active))
    baseline = float(max(y_active.mean(), 1 - y_active.mean()))
    edge = selective_acc - baseline
    coverage = float(active.mean())
    brier_all = float(brier_score_loss(yy, pp))

    latest = X0.dropna().iloc[[-1]]
    current_raw = []
    detail = []
    for name, tpl in templates.items():
        try:
            m = clone(tpl)
            m.fit(X, y)
            now = float(m.predict_proba(latest)[:, 1][0])
            current_raw.append(now)
            detail.append({"name": name, "probability_up": round(now, 4)})
        except Exception:
            pass

    if len(current_raw) < 3:
        raise RuntimeError("Too few current technical models")

    pooled_platt = fit_platt(pp, yy)
    current_p = float(apply_platt(pooled_platt, [float(np.mean(current_raw))])[0])
    th_now = float(np.median(thresholds)) if thresholds else 0.62

    audited = bool(edge > 0.02 and selective_acc >= 0.55 and active.sum() >= 100 and brier_all < 0.255)
    active_now = bool(audited and (current_p >= th_now or current_p <= 1 - th_now))
    signal = "Bullish" if active_now and current_p >= 0.5 else ("Bearish" if active_now else "NO EDGE / ABSTAIN")

    return {
        "horizon_days": h,
        "probability_up": round(current_p, 4),
        "probability_down": round(1 - current_p, 4),
        "signal": signal,
        "active_signal": active_now,
        "decision_threshold": round(th_now, 3),
        "selective_accuracy": round(selective_acc, 4),
        "baseline_direction_rate": round(baseline, 4),
        "selective_edge": round(edge, 4),
        "coverage": round(coverage, 4),
        "oos_signals": int(active.sum()),
        "oos_observations": int(len(yy)),
        "brier_all": round(brier_all, 4),
        "audit_pass": audited,
        "validation": "nested chronological walk-forward; threshold chosen only on earlier calibration data",
        "folds": fold_rows,
        "models": detail,
    }

def run_asset(asset, cfg):
    df = download(cfg["ticker"])
    rows = [horizon_model(df, h) for h in HORIZONS]
    payload = {
        "status": "ok",
        "asset": asset,
        "symbol": cfg["ticker"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Selective technical model. It abstains when confidence is below a threshold selected on earlier data.",
        "horizons": rows,
        "note": "Accuracy is measured only on active signals; coverage shows how often the model was willing to make a call. No-edge states are intentional.",
    }
    (DATA / cfg["output"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main():
    for asset, cfg in ASSETS.items():
        run_asset(asset, cfg)

if __name__ == "__main__":
    main()
