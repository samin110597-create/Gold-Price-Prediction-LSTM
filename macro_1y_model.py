import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
DATA.mkdir(exist_ok=True)

ASSETS = {
    "gold": {
        "ticker": "GC=F",
        "projection_file": "gold_projections.json",
        "output": "gold_macro_1y.json",
        "cross": {"silver": "SI=F", "copper": "HG=F", "oil": "CL=F", "sp500": "SPY", "long_bonds": "TLT"},
    },
    "silver": {
        "ticker": "SI=F",
        "projection_file": "silver_projections.json",
        "output": "silver_macro_1y.json",
        "cross": {"gold": "GC=F", "copper": "HG=F", "oil": "CL=F", "sp500": "SPY", "long_bonds": "TLT"},
    },
}

FRED = {
    "real_yield_10y": "DFII10",
    "breakeven_10y": "T10YIE",
    "broad_usd": "DTWEXBGS",
    "vix": "VIXCLS",
}

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def month_index(idx):
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.to_period("M").to_timestamp("M")

def market_monthly(ticker):
    df = flatten(yf.download(ticker, start="2003-01-01", interval="1mo",
                             auto_adjust=False, progress=False, threads=False))
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No monthly market data for {ticker}")
    s = df["Close"].astype(float).dropna().copy()
    s.index = month_index(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()

def fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.dropna().set_index(date_col)[val_col].astype(float)
    s.index = month_index(s.index)
    s = s.groupby(level=0).last().sort_index().shift(1)
    return s

def build_panel(cfg):
    target = market_monthly(cfg["ticker"])
    panel = pd.DataFrame({"target": target})

    for name, ticker in cfg["cross"].items():
        try:
            panel[name] = market_monthly(ticker).reindex(panel.index).ffill(limit=2)
        except Exception:
            pass

    fred_ok = {}
    for name, sid in FRED.items():
        try:
            panel[name] = fred_series(sid).reindex(panel.index).ffill(limit=2)
            fred_ok[name] = True
        except Exception:
            fred_ok[name] = False

    return panel.sort_index(), fred_ok

def features(panel):
    p = panel["target"].astype(float)
    x = pd.DataFrame(index=panel.index)

    for n in (1, 3, 6, 12):
        x[f"target_ret{n}"] = p.pct_change(n)
    x["momentum_12_1"] = p.shift(1) / p.shift(12) - 1
    x["vol6"] = p.pct_change().rolling(6).std()
    x["vol12"] = p.pct_change().rolling(12).std()
    x["drawdown12"] = p / p.rolling(12).max() - 1
    x["z12"] = (p - p.rolling(12).mean()) / p.rolling(12).std().replace(0, np.nan)
    x["z24"] = (p - p.rolling(24).mean()) / p.rolling(24).std().replace(0, np.nan)

    for col in panel.columns:
        if col == "target":
            continue
        s = panel[col].astype(float)
        if col in ("real_yield_10y", "breakeven_10y", "vix"):
            x[f"{col}_level"] = s
            x[f"{col}_chg3"] = s.diff(3)
            x[f"{col}_chg12"] = s.diff(12)
            x[f"{col}_z36"] = (s - s.rolling(36).mean()) / s.rolling(36).std().replace(0, np.nan)
        else:
            for n in (1, 3, 6, 12):
                x[f"{col}_ret{n}"] = s.pct_change(n)
            x[f"{col}_z36"] = (s - s.rolling(36).mean()) / s.rolling(36).std().replace(0, np.nan)

    if "real_yield_10y" in panel and "breakeven_10y" in panel:
        x["real_minus_breakeven"] = panel["real_yield_10y"] - panel["breakeven_10y"]

    if "gold" in panel:
        x["silver_gold_ratio_mom6"] = (panel["target"] / panel["gold"]).pct_change(6)
    if "silver" in panel:
        x["gold_silver_ratio_mom6"] = (panel["target"] / panel["silver"]).pct_change(6)

    return x.replace([np.inf, -np.inf], np.nan)

def model_templates():
    return {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=28.0))]),
        "Elastic Net": Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=0.018, l1_ratio=0.20, max_iter=5000, random_state=42))]),
        "Random Forest": RandomForestRegressor(
            n_estimators=220, max_depth=4, min_samples_leaf=8, max_features=0.65,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boost": HistGradientBoostingRegressor(
            max_iter=120, max_depth=2, learning_rate=0.035, l2_regularization=1.2, random_state=43
        ),
    }

def direction_accuracy(pred, actual):
    return float(np.mean((np.asarray(pred) >= 0) == (np.asarray(actual) >= 0)))

def walkforward(panel):
    X0 = features(panel)
    fwd = panel["target"].shift(-12) / panel["target"] - 1
    valid = X0.notna().all(axis=1) & fwd.notna()
    X, y = X0.loc[valid], fwd.loc[valid]

    if len(X) < 170:
        raise RuntimeError("Insufficient monthly history for macro 1Y model")

    templates = model_templates()
    origins = list(range(132, len(X), 3))[-52:]
    records = []

    for origin in origins:
        train_end = origin - 12
        if train_end < 120:
            continue
        preds = {}
        for name, tpl in templates.items():
            try:
                m = clone(tpl)
                m.fit(X.iloc[:train_end], y.iloc[:train_end])
                preds[name] = float(m.predict(X.iloc[[origin]])[0])
            except Exception:
                continue
        if len(preds) >= 3:
            baseline = float(y.iloc[:train_end].tail(min(120, train_end)).median())
            records.append({
                "date": str(X.index[origin].date()),
                "actual": float(y.iloc[origin]),
                "baseline": baseline,
                "preds": preds,
            })

    if len(records) < 24:
        raise RuntimeError("Too few purged macro walk-forward origins")

    actual = np.array([r["actual"] for r in records])
    base = np.array([r["baseline"] for r in records])
    base_mae = float(mean_absolute_error(actual, base))
    base_dir = direction_accuracy(base, actual)

    complete_names = [n for n in templates if all(n in r["preds"] for r in records)]
    metrics = {}
    raw_w = []
    for name in complete_names:
        pred = np.array([r["preds"][name] for r in records])
        mae = float(mean_absolute_error(actual, pred))
        dacc = direction_accuracy(pred, actual)
        skill = 1 - mae / max(base_mae, 1e-9)
        edge = dacc - base_dir
        weight = max(0.05, 1 + np.clip(skill, -0.30, 0.30) * 2.3 + np.clip(edge, -0.20, 0.20) * 1.8)
        metrics[name] = {"mae": mae, "dacc": dacc, "skill": skill, "edge": edge, "weight": weight}
        raw_w.append(weight)

    w = np.asarray(raw_w, dtype=float)
    w /= w.sum()
    ensemble = np.array([
        sum(r["preds"][name] * ww for name, ww in zip(complete_names, w))
        for r in records
    ])

    ens_mae = float(mean_absolute_error(actual, ensemble))
    ens_dir = direction_accuracy(ensemble, actual)
    ens_skill = 1 - ens_mae / max(base_mae, 1e-9)
    ens_edge = ens_dir - base_dir
    residual = actual - ensemble

    latest_x = X0.dropna().iloc[[-1]]
    current = []
    model_detail = []
    for name, ww in zip(complete_names, w):
        m = clone(templates[name])
        m.fit(X, y)
        ret = float(m.predict(latest_x)[0])
        current.append(ret)
        mm = metrics[name]
        model_detail.append({
            "name": name,
            "projected_return_pct": round(ret * 100, 2),
            "weight": round(float(ww), 4),
            "oos_mae_pct": round(mm["mae"] * 100, 2),
            "oos_direction_accuracy": round(mm["dacc"], 4),
            "oos_mae_skill": round(mm["skill"], 4),
        })

    raw_current = float(np.dot(current, w))
    historical_anchor = float(y.tail(min(120, len(y))).median())
    trust = float(np.clip(0.35 + max(0, ens_skill) * 1.2 + max(0, ens_edge) * 0.8, 0.25, 0.72))
    current_return = trust * raw_current + (1 - trust) * historical_anchor

    latest_price = float(panel["target"].dropna().iloc[-1])
    predicted_price = latest_price * (1 + current_return)

    q50 = float(np.quantile(np.abs(residual), 0.50))
    q80 = float(np.quantile(np.abs(residual), 0.80))
    focus = [latest_price * (1 + current_return - q50), latest_price * (1 + current_return + q50)]
    risk = [latest_price * (1 + current_return - q80), latest_price * (1 + current_return + q80)]

    validated = bool(ens_skill > 0.03 and ens_edge > 0.01 and len(records) >= 24)
    confidence = "Moderate" if ens_skill >= 0.12 and ens_edge >= 0.05 and len(records) >= 36 else "Low"

    return {
        "horizon": "1 Year",
        "model_price": round(predicted_price, 2),
        "predicted_price": round(predicted_price, 2),
        "projected_return_pct": round(current_return * 100, 2),
        "raw_macro_return_pct": round(raw_current * 100, 2),
        "historical_anchor_return_pct": round(historical_anchor * 100, 2),
        "macro_model_trust": round(trust, 3),
        "probability_up": round(float(np.mean((current_return + residual) > 0)), 4),
        "confidence": confidence,
        "forecast_status": "Validated" if validated else "Estimate only",
        "tight_model_zone": [round(focus[0], 2), round(focus[1], 2)],
        "focus_zone": [round(focus[0], 2), round(focus[1], 2)],
        "risk_zone": [round(risk[0], 2), round(risk[1], 2)],
        "backtest_directional_accuracy": round(ens_dir, 4),
        "baseline_directional_accuracy": round(base_dir, 4),
        "directional_edge": round(ens_edge, 4),
        "backtest_mae_pct": round(ens_mae * 100, 2),
        "baseline_mae_pct": round(base_mae * 100, 2),
        "mae_skill_vs_baseline": round(ens_skill, 4),
        "walkforward_origins": len(records),
        "validation": "monthly purged expanding-origin walk-forward; 12-month label purge; FRED macro series lagged one month",
        "models": model_detail,
    }

def score(row):
    return (
        2.0 * max(0.0, float(row.get("mae_skill_vs_baseline", 0) or 0))
        + 1.3 * max(0.0, float(row.get("directional_edge", 0) or 0))
        + min(0.35, float(row.get("walkforward_origins", 0) or 0) / 120.0)
    )

def maybe_promote(asset, cfg, macro_row):
    path = DATA / cfg["projection_file"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    projections = payload.get("projections", [])
    idx = next((i for i, x in enumerate(projections) if x.get("horizon") == "1 Year"), None)
    if idx is None:
        return False, None

    champion = projections[idx]
    old_score = score(champion)
    new_score = score(macro_row)

    new_skill = float(macro_row.get("mae_skill_vs_baseline", 0) or 0)
    new_edge = float(macro_row.get("directional_edge", 0) or 0)
    new_n = int(macro_row.get("walkforward_origins", 0) or 0)
    old_skill = float(champion.get("mae_skill_vs_baseline", 0) or 0)
    old_edge = float(champion.get("directional_edge", 0) or 0)

    promote = bool(
        new_n >= 24
        and new_skill > 0.03
        and new_edge >= 0.01
        and new_score > old_score + 0.04
        and new_skill >= old_skill - 0.01
        and new_edge >= old_edge - 0.02
    )

    if promote:
        row = dict(macro_row)
        row["selected_model"] = "Dedicated macro 1Y champion"
        row["champion_reason"] = "Macro model beat the incumbent on pre-defined OOS score and minimum skill/edge safeguards."
        projections[idx] = row
        payload["projections"] = projections
        payload["note"] = (payload.get("note", "") + " 1Y is selected by champion-challenger comparison; the macro challenger can replace the incumbent only when OOS safeguards are met.").strip()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return promote, {
        "incumbent_score": round(old_score, 4),
        "challenger_score": round(new_score, 4),
        "incumbent_skill": round(old_skill, 4),
        "challenger_skill": round(new_skill, 4),
        "incumbent_edge": round(old_edge, 4),
        "challenger_edge": round(new_edge, 4),
    }

def run_asset(asset, cfg):
    panel, fred_ok = build_panel(cfg)
    macro = walkforward(panel)
    promoted, comparison = maybe_promote(asset, cfg, macro)
    out = {
        "status": "ok",
        "asset": asset,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "fred_inputs_available": fred_ok,
        "macro_forecast": macro,
        "selected_as_champion": promoted,
        "comparison": comparison,
        "method_note": "Dedicated 1Y model uses monthly macro/cross-market inputs and cannot replace the incumbent unless it wins the pre-defined OOS champion test.",
    }
    (DATA / cfg["output"]).write_text(json.dumps(out, indent=2), encoding="utf-8")

def main():
    for asset, cfg in ASSETS.items():
        run_asset(asset, cfg)

if __name__ == "__main__":
    main()
