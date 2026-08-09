import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("data/live_forecast.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

TARGET = "GC=F"
FACTOR_TICKERS = {
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "vix": "^VIX",
    "spy": "SPY",
    "tlt": "TLT",
}

HORIZONS = (1, 5, 20)


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_daily(ticker: str, period: str = "10y") -> pd.DataFrame:
    df = flatten(
        yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )
    df = normalize_index(df)
    if "Close" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["Close"]).copy()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - pc).abs(),
            (df["Low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def aligned_factor_closes(gold_index: pd.Index) -> tuple[pd.DataFrame, dict]:
    closes = pd.DataFrame(index=gold_index)
    availability = {}
    for key, ticker in FACTOR_TICKERS.items():
        df = download_daily(ticker)
        if df.empty:
            availability[key] = False
            continue
        s = df["Close"].astype(float).rename(key)
        closes[key] = s.reindex(gold_index).ffill(limit=3)
        availability[key] = bool(closes[key].notna().sum() > 300)
    return closes, availability


def make_features(gold: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    c = gold["Close"].astype(float)
    x = pd.DataFrame(index=gold.index)

    x["gold_ret1"] = c.pct_change(1)
    x["gold_ret5"] = c.pct_change(5)
    x["gold_ret20"] = c.pct_change(20)
    x["gold_ret60"] = c.pct_change(60)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    x["gold_ema20_gap"] = c / ema20 - 1
    x["gold_ema50_gap"] = c / ema50 - 1
    x["gold_ema200_gap"] = c / ema200 - 1
    x["gold_ema20_50"] = ema20 / ema50 - 1
    x["gold_ema50_200"] = ema50 / ema200 - 1
    x["gold_rsi14"] = (rsi(c, 14) - 50) / 50
    x["gold_vol20"] = c.pct_change().rolling(20).std()
    x["gold_vol60"] = c.pct_change().rolling(60).std()
    x["gold_atr14_pct"] = atr(gold, 14) / c
    x["gold_z20"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    hi20 = gold["High"].rolling(20).max()
    lo20 = gold["Low"].rolling(20).min()
    x["gold_range_pos20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan) - 0.5
    x["gold_drawdown60"] = c / c.rolling(60).max() - 1

    for key in factors.columns:
        s = factors[key].astype(float)
        x[f"{key}_ret1"] = s.pct_change(1)
        x[f"{key}_ret5"] = s.pct_change(5)
        x[f"{key}_ret20"] = s.pct_change(20)
        x[f"{key}_z60"] = (s - s.rolling(60).mean()) / s.rolling(60).std().replace(0, np.nan)
        x[f"{key}_ema20_gap"] = s / s.ewm(span=20, adjust=False).mean() - 1

    if "silver" in factors:
        ratio = c / factors["silver"].replace(0, np.nan)
        x["gold_silver_ratio_ret20"] = ratio.pct_change(20)
        x["gold_silver_ratio_z60"] = (
            (ratio - ratio.rolling(60).mean())
            / ratio.rolling(60).std().replace(0, np.nan)
        )

    if "dxy" in factors:
        x["gold_dxy_corr60"] = c.pct_change().rolling(60).corr(factors["dxy"].pct_change())
    if "us10y" in factors:
        x["yield_change5"] = factors["us10y"].diff(5)
        x["yield_change20"] = factors["us10y"].diff(20)

    return x.replace([np.inf, -np.inf], np.nan)


def model_templates():
    return {
        "Logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.35,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=260,
            max_depth=5,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boost": HistGradientBoostingClassifier(
            max_iter=160,
            max_depth=3,
            learning_rate=0.045,
            l2_regularization=0.25,
            random_state=42,
        ),
        "Regime KNN": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=45, weights="distance", p=2)),
            ]
        ),
    }


def confidence_label(prob_up: float, accuracy: float, baseline: float, agreement: float) -> str:
    edge = accuracy - baseline
    distance = abs(prob_up - 0.5)
    if edge >= 0.055 and distance >= 0.12 and agreement >= 0.75:
        return "High"
    if edge >= 0.02 and distance >= 0.07 and agreement >= 0.65:
        return "Moderate"
    return "Low"


def forecast_one(gold: pd.DataFrame, feats: pd.DataFrame, horizon: int) -> dict:
    future_ret = gold["Close"].shift(-horizon) / gold["Close"] - 1
    y = (future_ret > 0).astype(int)

    valid = feats.notna().all(axis=1) & future_ret.notna()
    X = feats.loc[valid]
    Y = y.loc[valid]

    if len(X) < 650 or Y.nunique() < 2:
        raise RuntimeError(f"Not enough clean multi-factor history for {horizon}d model")

    holdout = min(252, max(140, len(X) // 5))
    split = len(X) - holdout
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = Y.iloc[:split], Y.iloc[split:]

    latest_x = feats.dropna().iloc[[-1]]
    models = model_templates()
    diagnostics = []
    current_probs = []

    base = float(max(y_test.mean(), 1 - y_test.mean()))

    for name, template in models.items():
        model = clone(template)
        try:
            model.fit(X_train, y_train)
            test_prob = model.predict_proba(X_test)[:, 1]
            pred = (test_prob >= 0.5).astype(int)
            acc = float(accuracy_score(y_test, pred))
            brier = float(brier_score_loss(y_test, test_prob))

            model.fit(X, Y)
            p_now = float(model.predict_proba(latest_x)[:, 1][0])
            current_probs.append((name, p_now, acc, brier))
        except Exception:
            continue

    if len(current_probs) < 2:
        raise RuntimeError(f"Too few ensemble models completed for {horizon}d horizon")

    weights = []
    for _, _, acc, brier in current_probs:
        skill = max(-0.03, min(0.08, acc - base))
        calibration = max(0.0, 0.25 - brier)
        weights.append(max(0.10, 1.0 + skill * 8.0 + calibration * 1.5))
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    raw_prob = float(
        np.sum(np.array([p for _, p, _, _ in current_probs], dtype=float) * weights)
    )
    weighted_acc = float(
        np.sum(np.array([a for _, _, a, _ in current_probs], dtype=float) * weights)
    )
    weighted_brier = float(
        np.sum(np.array([b for _, _, _, b in current_probs], dtype=float) * weights)
    )

    direction_votes = [p >= 0.5 for _, p, _, _ in current_probs]
    majority = sum(direction_votes) >= (len(direction_votes) / 2)
    agreement = float(sum(v == majority for v in direction_votes) / len(direction_votes))

    edge = weighted_acc - base
    reliability = float(np.clip((edge + 0.015) / 0.07, 0.12, 1.0))
    prob_up = 0.5 + (raw_prob - 0.5) * (0.30 + 0.70 * reliability)
    prob_up = float(np.clip(prob_up, 0.30, 0.70))

    historical = (gold["Close"] / gold["Close"].shift(horizon) - 1).dropna().tail(1260)
    q20, q50, q80 = [float(v) for v in historical.quantile([0.20, 0.50, 0.80]).values]
    last_daily = float(gold["Close"].iloc[-1])
    vol = float(gold["Close"].pct_change().tail(60).std())
    tilt = (prob_up - 0.5) * 2 * vol * np.sqrt(horizon) * 0.40
    low = last_daily * (1 + q20 + tilt)
    mid = last_daily * (1 + q50 + tilt)
    high = last_daily * (1 + q80 + tilt)

    if prob_up >= 0.58:
        signal = "Bullish bias"
    elif prob_up <= 0.42:
        signal = "Bearish bias"
    else:
        signal = "Neutral / no strong edge"

    confidence = confidence_label(prob_up, weighted_acc, base, agreement)

    for (name, p_now, acc, brier), weight in zip(current_probs, weights):
        diagnostics.append(
            {
                "name": name,
                "probability_up": round(p_now, 4),
                "backtest_accuracy": round(acc, 4),
                "brier_score": round(brier, 4),
                "weight": round(float(weight), 4),
            }
        )

    return {
        "horizon_days": horizon,
        "probability_up": round(prob_up, 4),
        "probability_down": round(1 - prob_up, 4),
        "signal": signal,
        "confidence": confidence,
        "forecast_mid": round(mid, 2),
        "range_20_80": [round(low, 2), round(high, 2)],
        "backtest_accuracy": round(weighted_acc, 4),
        "naive_baseline": round(base, 4),
        "backtest_edge": round(weighted_acc - base, 4),
        "brier_score": round(weighted_brier, 4),
        "model_agreement": round(agreement, 4),
        "test_observations": int(holdout),
        "models": diagnostics,
    }


def factor_impact(key: str, series: pd.Series, gold: pd.Series) -> tuple[str, str]:
    s = series.dropna()
    if len(s) < 25:
        return "Unavailable", "Not enough recent data"
    ret20 = float(s.iloc[-1] / s.iloc[-21] - 1)
    gold20 = float(gold.iloc[-1] / gold.iloc[-21] - 1)

    if key == "dxy":
        if ret20 <= -0.007:
            return "Supportive", "Dollar has weakened over 20 trading days"
        if ret20 >= 0.007:
            return "Headwind", "Dollar has strengthened over 20 trading days"
        return "Neutral", "Dollar trend is mixed"

    if key == "us10y":
        delta20 = float(s.iloc[-1] - s.iloc[-21])
        if delta20 <= -0.10:
            return "Supportive", "10Y yield has fallen"
        if delta20 >= 0.10:
            return "Headwind", "10Y yield has risen"
        return "Neutral", "10Y yield is range-bound"

    if key == "silver":
        relative = ret20 - gold20
        if ret20 > 0 and relative > 0.01:
            return "Supportive", "Silver is rising and outperforming gold"
        if ret20 < -0.02:
            return "Headwind", "Silver is weak, reducing metals breadth"
        return "Neutral", "Silver breadth is mixed"

    if key == "tlt":
        if ret20 >= 0.015:
            return "Supportive", "Long-duration bonds are firm"
        if ret20 <= -0.015:
            return "Headwind", "Long-duration bonds are weak"
        return "Neutral", "Bond trend is mixed"

    if key == "vix":
        level = float(s.iloc[-1])
        if level >= 28:
            return "Risk-off", "Elevated equity volatility / stress"
        if level <= 15:
            return "Risk-on", "Low equity-volatility regime"
        return "Neutral", "Equity volatility is mid-range"

    if key == "spy":
        if ret20 <= -0.04:
            return "Risk-off", "Equities have weakened materially"
        if ret20 >= 0.04:
            return "Risk-on", "Equities have strong momentum"
        return "Neutral", "Equity trend is mixed"

    return "Neutral", "Mixed"


def factor_snapshot(factors: pd.DataFrame, gold_close: pd.Series) -> list[dict]:
    labels = {
        "dxy": ("US Dollar Index", "DX-Y.NYB"),
        "us10y": ("US 10Y Yield", "^TNX"),
        "silver": ("Silver Futures", "SI=F"),
        "vix": ("VIX", "^VIX"),
        "spy": ("S&P 500 ETF", "SPY"),
        "tlt": ("20+Y Treasury ETF", "TLT"),
    }
    output = []
    for key in ("dxy", "us10y", "silver", "vix", "spy", "tlt"):
        if key not in factors.columns:
            continue
        s = factors[key].dropna()
        if len(s) < 25:
            continue
        impact, reason = factor_impact(key, s, gold_close)
        ret5 = float(s.iloc[-1] / s.iloc[-6] - 1)
        ret20 = float(s.iloc[-1] / s.iloc[-21] - 1)
        name, symbol = labels[key]
        output.append(
            {
                "key": key,
                "name": name,
                "symbol": symbol,
                "value": round(float(s.iloc[-1]), 3),
                "change_5d_pct": round(ret5 * 100, 2),
                "change_20d_pct": round(ret20 * 100, 2),
                "impact": impact,
                "reason": reason,
            }
        )
    return output


def build_regime(gold: pd.DataFrame, factors: pd.DataFrame, snapshots: list[dict]) -> dict:
    c = gold["Close"].astype(float)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    last = float(c.iloc[-1])

    if last > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
        trend = "Strong uptrend"
        trend_score = 2
    elif last > ema50.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        trend = "Uptrend"
        trend_score = 1
    elif last < ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
        trend = "Strong downtrend"
        trend_score = -2
    elif last < ema50.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        trend = "Downtrend"
        trend_score = -1
    else:
        trend = "Mixed / transitional"
        trend_score = 0

    atr_pct_series = atr(gold, 14) / c
    current_atr_pct = float(atr_pct_series.iloc[-1])
    ref = atr_pct_series.dropna().tail(252)
    vol_percentile = float((ref <= current_atr_pct).mean()) if len(ref) else 0.5
    if vol_percentile >= 0.80:
        volatility = "High"
    elif vol_percentile <= 0.30:
        volatility = "Low"
    else:
        volatility = "Normal"

    directional = {"Supportive": 1, "Headwind": -1}
    macro_score = sum(directional.get(x["impact"], 0) for x in snapshots)
    if macro_score >= 2:
        macro = "Supportive"
    elif macro_score <= -2:
        macro = "Headwind"
    else:
        macro = "Mixed"

    if trend_score >= 1 and macro_score >= 1:
        overall = "Bullish"
    elif trend_score <= -1 and macro_score <= -1:
        overall = "Bearish"
    else:
        overall = "Neutral / mixed"

    return {
        "overall": overall,
        "trend": trend,
        "macro": macro,
        "macro_score": int(macro_score),
        "volatility": volatility,
        "volatility_percentile": round(vol_percentile, 4),
    }


def build_levels(gold: pd.DataFrame) -> dict:
    c = gold["Close"].astype(float)
    last = float(c.iloc[-1])
    atr_abs = float(atr(gold, 14).iloc[-1])
    return {
        "daily_close_reference": round(last, 2),
        "support_20d": round(float(gold["Low"].tail(20).min()), 2),
        "resistance_20d": round(float(gold["High"].tail(20).max()), 2),
        "support_60d": round(float(gold["Low"].tail(60).min()), 2),
        "resistance_60d": round(float(gold["High"].tail(60).max()), 2),
        "atr14_dollars": round(atr_abs, 2),
        "expected_move_1d": [round(last - atr_abs, 2), round(last + atr_abs, 2)],
        "expected_move_5d": [
            round(last - atr_abs * np.sqrt(5), 2),
            round(last + atr_abs * np.sqrt(5), 2),
        ],
    }


def main():
    gold = download_daily(TARGET)
    required = {"Open", "High", "Low", "Close"}
    if gold.empty or not required.issubset(gold.columns):
        raise RuntimeError("Gold daily data download failed")
    gold = gold.dropna(subset=list(required)).copy()
    if len(gold) < 800:
        raise RuntimeError("Gold history returned too few observations")

    factors, availability = aligned_factor_closes(gold.index)
    usable_cols = [c for c in factors.columns if factors[c].notna().sum() > 700]
    factors = factors[usable_cols].copy()

    feats = make_features(gold, factors)
    forecasts = [forecast_one(gold, feats, h) for h in HORIZONS]

    intraday = flatten(
        yf.download(
            TARGET,
            period="10d",
            interval="1h",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )
    intraday = (
        intraday.dropna(subset=["Close"])
        if not intraday.empty and "Close" in intraday.columns
        else pd.DataFrame()
    )

    close = gold["Close"].astype(float)
    latest_daily = float(close.iloc[-1])
    latest_price = float(intraday["Close"].iloc[-1]) if not intraday.empty else latest_daily
    previous = float(close.iloc[-2])
    change_pct = (latest_price / previous - 1) * 100

    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])
    atr_pct = float((atr(gold, 14).iloc[-1] / latest_daily) * 100)

    snapshots = factor_snapshot(factors, close)
    regime = build_regime(gold, factors, snapshots)
    levels = build_levels(gold)

    series = [
        {"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for idx, row in gold.tail(300).iterrows()
    ]

    updated = datetime.now(timezone.utc).isoformat()
    if not intraday.empty:
        latest_market_time = pd.Timestamp(intraday.index[-1]).isoformat()
    else:
        latest_market_time = pd.Timestamp(gold.index[-1]).isoformat()

    payload = {
        "status": "ok",
        "symbol": TARGET,
        "instrument": "COMEX Gold Futures",
        "updated_utc": updated,
        "latest_market_time": latest_market_time,
        "latest_price": round(latest_price, 2),
        "change_pct": round(change_pct, 3),
        "regime": regime,
        "indicators": {
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "rsi14": round(rsi14, 2),
            "atr14_pct": round(atr_pct, 2),
        },
        "levels": levels,
        "factors": snapshots,
        "factor_availability": availability,
        "forecasts": forecasts,
        "series": series,
        "methodology": (
            "Multi-factor ensemble using gold trend/momentum/volatility plus cross-asset context "
            "from the US dollar, Treasury yields, silver, VIX, equities and long-duration bonds. "
            "Four classifiers are combined and probabilities are shrunk toward 50% when recent "
            "time-ordered holdout performance is weak. Forecast price bands use historical forward-return distributions."
        ),
        "disclaimer": (
            "Research/education only. Market data may be delayed. Forecasts are probabilistic, "
            "can fail during news shocks and are not trading or investment advice."
        ),
    }

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "latest_price": payload["latest_price"],
                "regime": payload["regime"],
                "forecasts": [
                    {
                        "horizon": x["horizon_days"],
                        "p_up": x["probability_up"],
                        "confidence": x["confidence"],
                        "edge": x["backtest_edge"],
                    }
                    for x in forecasts
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
