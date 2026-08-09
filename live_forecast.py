import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

TICKER = "GC=F"
OUT = Path("data/live_forecast.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    c = df["Close"].astype(float)
    x["ret1"] = c.pct_change(1)
    x["ret5"] = c.pct_change(5)
    x["ret20"] = c.pct_change(20)
    x["ema20_gap"] = c / c.ewm(span=20, adjust=False).mean() - 1
    x["ema50_gap"] = c / c.ewm(span=50, adjust=False).mean() - 1
    x["ema20_50"] = c.ewm(span=20, adjust=False).mean() / c.ewm(span=50, adjust=False).mean() - 1
    x["rsi14"] = (rsi(c, 14) - 50) / 50
    x["vol20"] = c.pct_change().rolling(20).std()
    x["atr14_pct"] = atr(df, 14) / c
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    x["z20"] = (c - ma20) / sd20.replace(0, np.nan)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def forecast_one(df: pd.DataFrame, feats: pd.DataFrame, horizon: int):
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    y = (future_ret > 0).astype(int)
    train = feats.copy()
    valid = train.notna().all(axis=1) & future_ret.notna()
    X = train.loc[valid]
    Y = y.loc[valid]

    if len(X) < 350 or Y.nunique() < 2:
        raise RuntimeError("Not enough market history to fit forecast model")

    holdout = min(252, max(100, len(X)//5))
    split = len(X) - holdout
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.6, max_iter=3000, class_weight="balanced")),
    ])
    model.fit(X.iloc[:split], Y.iloc[:split])
    test_prob = model.predict_proba(X.iloc[split:])[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    acc = float(accuracy_score(Y.iloc[split:], test_pred))
    base = float(max(Y.iloc[split:].mean(), 1 - Y.iloc[split:].mean()))
    brier = float(brier_score_loss(Y.iloc[split:], test_prob))

    model.fit(X, Y)
    latest_x = feats.dropna().iloc[[-1]]
    raw_p = float(model.predict_proba(latest_x)[:, 1][0])

    # Shrink probability toward 50% when the holdout did not beat a naive baseline.
    edge = max(0.0, acc - base)
    reliability = min(1.0, edge / 0.08)
    p_up = 0.5 + (raw_p - 0.5) * (0.35 + 0.65 * reliability)
    p_up = float(np.clip(p_up, 0.38, 0.62))

    hist_h = (df["Close"] / df["Close"].shift(horizon) - 1).dropna().tail(756)
    q20, q50, q80 = [float(v) for v in hist_h.quantile([0.20, 0.50, 0.80]).values]
    # modest model tilt, not a deterministic target
    vol = float(df["Close"].pct_change().tail(60).std())
    tilt = (p_up - 0.5) * 2 * vol * np.sqrt(horizon) * 0.45
    last = float(df["Close"].iloc[-1])
    lo = last * (1 + q20 + tilt)
    mid = last * (1 + q50 + tilt)
    hi = last * (1 + q80 + tilt)

    if p_up >= 0.56:
        signal = "Bullish bias"
    elif p_up <= 0.44:
        signal = "Bearish bias"
    else:
        signal = "Neutral / no strong edge"

    if acc >= max(base + 0.04, 0.56):
        confidence = "Higher"
    elif acc >= max(base + 0.015, 0.53):
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "horizon_days": horizon,
        "probability_up": round(p_up, 4),
        "signal": signal,
        "confidence": confidence,
        "forecast_mid": round(mid, 2),
        "range_20_80": [round(lo, 2), round(hi, 2)],
        "backtest_accuracy": round(acc, 4),
        "naive_baseline": round(base, 4),
        "brier_score": round(brier, 4),
        "test_observations": int(holdout),
    }


def main():
    daily = flatten(yf.download(TICKER, period="8y", interval="1d", auto_adjust=False, progress=False, threads=False))
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(daily) < 500:
        raise RuntimeError("Gold data download returned too few rows")

    intraday = flatten(yf.download(TICKER, period="10d", interval="1h", auto_adjust=False, progress=False, threads=False))
    intraday = intraday.dropna(subset=["Close"]) if not intraday.empty else intraday

    feats = make_features(daily)
    forecasts = [forecast_one(daily, feats, h) for h in (1, 5, 20)]

    daily_close = daily["Close"].astype(float)
    latest_daily = float(daily_close.iloc[-1])
    latest_price = float(intraday["Close"].iloc[-1]) if not intraday.empty else latest_daily
    previous = float(daily_close.iloc[-2])
    change_pct = (latest_price / previous - 1) * 100

    ema20 = float(daily_close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(daily_close.ewm(span=50, adjust=False).mean().iloc[-1])
    rsi14 = float(rsi(daily_close, 14).iloc[-1])
    atr_pct = float((atr(daily, 14).iloc[-1] / latest_daily) * 100)

    series = []
    for idx, row in daily.tail(260).iterrows():
        series.append({
            "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
        })

    updated = datetime.now(timezone.utc).isoformat()
    if not intraday.empty:
        try:
            latest_market_time = pd.Timestamp(intraday.index[-1]).isoformat()
        except Exception:
            latest_market_time = updated
    else:
        latest_market_time = pd.Timestamp(daily.index[-1]).isoformat()

    payload = {
        "status": "ok",
        "symbol": TICKER,
        "instrument": "COMEX Gold Futures",
        "updated_utc": updated,
        "latest_market_time": latest_market_time,
        "latest_price": round(latest_price, 2),
        "change_pct": round(change_pct, 3),
        "indicators": {
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "rsi14": round(rsi14, 2),
            "atr14_pct": round(atr_pct, 2),
            "trend": "Uptrend" if ema20 > ema50 else "Downtrend",
        },
        "forecasts": forecasts,
        "series": series,
        "methodology": "Logistic-regression ensemble-style feature model using momentum, EMA trend, RSI, volatility, ATR and z-score. Probabilities are shrunk toward 50% when recent holdout performance is weak. Price bands use historical forward-return distributions with a small probability tilt.",
        "disclaimer": "Research/education only. Market data may be delayed. Forecasts are probabilistic and can be wrong; they are not trading or investment advice.",
    }

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"latest_price": payload["latest_price"], "forecasts": forecasts}, indent=2))


if __name__ == "__main__":
    main()
