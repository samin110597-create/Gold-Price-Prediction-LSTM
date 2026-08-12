import json
from pathlib import Path
from urllib.parse import urlencode

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
        "commodity": "GOLD",
        "cross": {
            "silver": "SI=F",
            "copper": "HG=F",
            "oil": "CL=F",
            "sp500": "SPY",
            "long_bonds": "TLT",
        },
    },
    "silver": {
        "ticker": "SI=F",
        "projection_file": "silver_projections.json",
        "output": "silver_macro_1y.json",
        "commodity": "SILVER",
        "cross": {
            "gold": "GC=F",
            "copper": "HG=F",
            "oil": "CL=F",
            "sp500": "SPY",
            "long_bonds": "TLT",
        },
    },
}

FRED = {
    "real_yield_10y": "DFII10",
    "breakeven_10y": "T10YIE",
    "broad_usd": "DTWEXBGS",
    "vix": "VIXCLS",
}

MODEL_TRUST = 0.50


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
    df = flatten(
        yf.download(
            ticker,
            start="1995-01-01",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No market data for {ticker}")
    s = df["Close"].astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s.resample("ME").last().dropna()
    s.index = month_index(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def fred_series(series_id):
    df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    date_col, value_col = df.columns[:2]
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    s = df.dropna().set_index(date_col)[value_col].astype(float)
    s.index = month_index(s.index)
    # Conservative one-month lag: only information already available before the forecast month.
    return s.groupby(level=0).last().sort_index().shift(1)


def cftc_monthly(commodity):
    fields = (
        "report_date_as_yyyy_mm_dd,commodity_name,open_interest_all,"
        "m_money_positions_long_all,m_money_positions_short_all,"
        "prod_merc_positions_long,prod_merc_positions_short"
    )
    params = {
        "$select": fields,
        "$where": f"commodity_name='{commodity}'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 5000,
    }
    url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv?" + urlencode(params)
    df = pd.read_csv(url)
    if df.empty:
        raise RuntimeError(f"No CFTC data for {commodity}")
    date_col = "report_date_as_yyyy_mm_dd"
    df[date_col] = pd.to_datetime(df[date_col])
    for col in df.columns:
        if col not in (date_col, "commodity_name"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    oi = df["open_interest_all"].replace(0, np.nan)
    df["managed_net"] = (
        df["m_money_positions_long_all"] - df["m_money_positions_short_all"]
    ) / oi
    df["producer_net"] = (
        df["prod_merc_positions_long"] - df["prod_merc_positions_short"]
    ) / oi
    out = (
        df.set_index(date_col)[["managed_net", "producer_net"]]
        .resample("ME")
        .last()
    )
    out.index = month_index(out.index)
    # Extra-conservative one-month lag for weekly positioning.
    return out.shift(1)


def build_panel(cfg):
    target = market_monthly(cfg["ticker"])
    panel = pd.DataFrame({"target": target})
    status = {"fred": {}, "cftc": False}

    for name, ticker in cfg["cross"].items():
        try:
            panel[name] = market_monthly(ticker).reindex(panel.index).ffill(limit=2)
        except Exception as exc:
            status[f"{name}_error"] = str(exc)

    for name, sid in FRED.items():
        try:
            panel[name] = fred_series(sid).reindex(panel.index).ffill(limit=2)
            status["fred"][name] = True
        except Exception as exc:
            status["fred"][name] = False
            status[f"{name}_error"] = str(exc)

    try:
        cot = cftc_monthly(cfg["commodity"]).reindex(panel.index)
        for col in cot.columns:
            panel[f"cftc_{col}"] = cot[col]
        status["cftc"] = True
    except Exception as exc:
        status["cftc_error"] = str(exc)

    return panel.sort_index(), status


def zscore(s, window=36, min_periods=12):
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mean) / std


def features(panel):
    p = panel["target"].astype(float)
    x = pd.DataFrame(index=panel.index)

    for n in (1, 3, 6, 12):
        x[f"target_ret{n}"] = p.pct_change(n)
    x["momentum_12_1"] = p.shift(1) / p.shift(12) - 1
    x["vol6"] = p.pct_change().rolling(6).std()
    x["vol12"] = p.pct_change().rolling(12).std()
    x["drawdown12"] = p / p.rolling(12).max() - 1
    x["z12"] = zscore(p, 12, 12)
    x["z24"] = zscore(p, 24, 18)

    for col in panel.columns:
        if col == "target":
            continue
        s = panel[col].astype(float)
        available = s.notna().astype(float)
        x[f"{col}_available"] = available

        if col.startswith("cftc_"):
            x[f"{col}_level"] = s.fillna(0.0)
            x[f"{col}_z36"] = zscore(s).fillna(0.0)
            x[f"{col}_chg3"] = s.diff(3).fillna(0.0)
        elif col in ("real_yield_10y", "breakeven_10y", "vix"):
            x[f"{col}_level"] = s.fillna(0.0)
            x[f"{col}_chg3"] = s.diff(3).fillna(0.0)
            x[f"{col}_chg12"] = s.diff(12).fillna(0.0)
            x[f"{col}_z36"] = zscore(s).fillna(0.0)
        else:
            for n in (1, 3, 6, 12):
                x[f"{col}_ret{n}"] = s.pct_change(n).fillna(0.0)
            x[f"{col}_z36"] = zscore(s).fillna(0.0)

    if "real_yield_10y" in panel and "breakeven_10y" in panel:
        both = panel["real_yield_10y"].notna() & panel["breakeven_10y"].notna()
        x["real_minus_breakeven"] = (
            panel["real_yield_10y"] - panel["breakeven_10y"]
        ).where(both, 0.0)
        x["real_minus_breakeven_available"] = both.astype(float)

    if "gold" in panel:
        ratio = panel["target"] / panel["gold"]
        x["silver_gold_ratio_mom6"] = ratio.pct_change(6).fillna(0.0)
    if "silver" in panel:
        ratio = panel["target"] / panel["silver"]
        x["gold_silver_ratio_mom6"] = ratio.pct_change(6).fillna(0.0)

    return x.replace([np.inf, -np.inf], np.nan)


def templates():
    return {
        "Ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=25.0))]
        ),
        "Elastic Net": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.015,
                        l1_ratio=0.18,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=4,
            min_samples_leaf=7,
            max_features=0.65,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boost": HistGradientBoostingRegressor(
            max_iter=110,
            max_depth=2,
            learning_rate=0.035,
            l2_regularization=1.2,
            random_state=43,
        ),
    }


def direction_accuracy(pred, actual):
    return float(
        np.mean((np.asarray(pred) >= 0) == (np.asarray(actual) >= 0))
    )


def metric_block(pred, actual, baseline):
    mae = float(mean_absolute_error(actual, pred))
    baseline_mae = float(mean_absolute_error(actual, baseline))
    dacc = direction_accuracy(pred, actual)
    baseline_dacc = direction_accuracy(baseline, actual)
    return {
        "mae": mae,
        "baseline_mae": baseline_mae,
        "skill": 1 - mae / max(baseline_mae, 1e-9),
        "dacc": dacc,
        "baseline_dacc": baseline_dacc,
        "edge": dacc - baseline_dacc,
    }


def walkforward(panel):
    X0 = features(panel)
    fwd = panel["target"].shift(-12) / panel["target"] - 1
    valid = X0.notna().all(axis=1) & fwd.notna()
    X = X0.loc[valid]
    y = fwd.loc[valid]

    if len(X) < 130:
        raise RuntimeError(f"Insufficient usable monthly history: {len(X)}")

    model_map = templates()
    # Six years of training + a full 12-month label purge before each origin.
    origins = list(range(84, len(X), 3))[-64:]
    records = []

    for origin in origins:
        train_end = origin - 12
        if train_end < 72:
            continue

        preds = {}
        for name, tpl in model_map.items():
            try:
                model = clone(tpl)
                model.fit(X.iloc[:train_end], y.iloc[:train_end])
                preds[name] = float(model.predict(X.iloc[[origin]])[0])
            except Exception:
                continue

        if len(preds) >= 3:
            baseline = float(
                y.iloc[:train_end].tail(min(120, train_end)).median()
            )
            records.append(
                {
                    "date": str(X.index[origin].date()),
                    "actual": float(y.iloc[origin]),
                    "baseline": baseline,
                    "preds": preds,
                }
            )

    if len(records) < 28:
        raise RuntimeError(f"Too few purged macro origins: {len(records)}")

    names = [
        name
        for name in model_map
        if all(name in record["preds"] for record in records)
    ]
    if len(names) < 3:
        raise RuntimeError("Too few complete macro models")

    actual = np.array([r["actual"] for r in records], dtype=float)
    baseline = np.array([r["baseline"] for r in records], dtype=float)
    raw_ensemble = np.array(
        [np.mean([r["preds"][name] for name in names]) for r in records],
        dtype=float,
    )

    # This is the exact formula published live: fixed 50% model / 50% historical
    # anchor. The same formula is evaluated here out-of-sample.
    final_ensemble = MODEL_TRUST * raw_ensemble + (1 - MODEL_TRUST) * baseline
    ensemble_metrics = metric_block(final_ensemble, actual, baseline)
    residual = actual - final_ensemble

    # Origins are three months apart; every fourth origin is 12 months apart.
    non_idx = np.arange(0, len(records), 4)
    if len(non_idx) < 8:
        raise RuntimeError(
            f"Too few non-overlapping 12-month origins: {len(non_idx)}"
        )
    non_metrics = metric_block(
        final_ensemble[non_idx], actual[non_idx], baseline[non_idx]
    )

    model_details = []
    for name in names:
        pred = np.array([r["preds"][name] for r in records], dtype=float)
        m = metric_block(pred, actual, baseline)
        model_details.append(
            {
                "name": name,
                "weight": round(1 / len(names), 4),
                "oos_mae_skill": round(m["skill"], 4),
                "oos_direction_accuracy": round(m["dacc"], 4),
            }
        )

    latest_x = X0.dropna().iloc[[-1]]
    current_preds = []
    for name in names:
        model = clone(model_map[name])
        model.fit(X, y)
        current = float(model.predict(latest_x)[0])
        current_preds.append(current)
        for detail in model_details:
            if detail["name"] == name:
                detail["projected_return_pct"] = round(current * 100, 2)

    raw_current = float(np.mean(current_preds))
    anchor = float(y.tail(min(120, len(y))).median())
    current_return = MODEL_TRUST * raw_current + (1 - MODEL_TRUST) * anchor

    latest_price = float(panel["target"].dropna().iloc[-1])
    predicted_price = latest_price * (1 + current_return)

    q50 = float(np.quantile(np.abs(residual), 0.50))
    q80 = float(np.quantile(np.abs(residual), 0.80))
    focus = [
        latest_price * (1 + current_return - q50),
        latest_price * (1 + current_return + q50),
    ]
    risk = [
        latest_price * (1 + current_return - q80),
        latest_price * (1 + current_return + q80),
    ]

    # Probability is empirical, then shrunk toward 50% because the truly
    # independent annual sample is much smaller than the overlapping OOS sample.
    p_all = float(np.mean((current_return + residual) > 0))
    p_non = float(np.mean((current_return + residual[non_idx]) > 0))
    p_raw = 0.5 * p_all + 0.5 * p_non
    probability_reliability = min(0.75, len(non_idx) / 16.0)
    probability_up = float(
        np.clip(
            0.5 + (p_raw - 0.5) * probability_reliability,
            0.20,
            0.80,
        )
    )

    validated = bool(
        ensemble_metrics["skill"] > 0.03
        and ensemble_metrics["edge"] > 0.01
        and len(records) >= 28
        and non_metrics["skill"] > 0
        and non_metrics["edge"] >= 0
    )

    confidence = "Low"
    if (
        validated
        and len(non_idx) >= 12
        and ensemble_metrics["skill"] >= 0.12
        and ensemble_metrics["edge"] >= 0.04
        and non_metrics["skill"] >= 0.08
        and non_metrics["edge"] >= 0.04
    ):
        confidence = "Moderate"

    return {
        "horizon": "1 Year",
        "model_price": round(predicted_price, 2),
        "predicted_price": round(predicted_price, 2),
        "projected_return_pct": round(current_return * 100, 2),
        "raw_macro_return_pct": round(raw_current * 100, 2),
        "historical_anchor_return_pct": round(anchor * 100, 2),
        "macro_model_trust": MODEL_TRUST,
        "probability_up": round(probability_up, 4),
        "probability_method": (
            "empirical OOS residual probability, blended with non-overlapping "
            "annual sample and shrunk toward 50%"
        ),
        "confidence": confidence,
        "forecast_status": "Validated" if validated else "Estimate only",
        "tight_model_zone": [round(focus[0], 2), round(focus[1], 2)],
        "focus_zone": [round(focus[0], 2), round(focus[1], 2)],
        "risk_zone": [round(risk[0], 2), round(risk[1], 2)],
        "backtest_directional_accuracy": round(
            ensemble_metrics["dacc"], 4
        ),
        "baseline_directional_accuracy": round(
            ensemble_metrics["baseline_dacc"], 4
        ),
        "directional_edge": round(ensemble_metrics["edge"], 4),
        "backtest_mae_pct": round(ensemble_metrics["mae"] * 100, 2),
        "baseline_mae_pct": round(
            ensemble_metrics["baseline_mae"] * 100, 2
        ),
        "mae_skill_vs_baseline": round(ensemble_metrics["skill"], 4),
        "walkforward_origins": len(records),
        "nonoverlap_origins": len(non_idx),
        "nonoverlap_mae_skill": round(non_metrics["skill"], 4),
        "nonoverlap_directional_edge": round(non_metrics["edge"], 4),
        "audited_live_formula": True,
        "validation": (
            "monthly purged expanding-origin walk-forward; 12-month label purge; "
            "fixed 50/50 macro-model/historical-anchor formula audited OOS; "
            "FRED and CFTC inputs lagged; non-overlapping 12-month sanity subset"
        ),
        "models": model_details,
    }


def score(row):
    return (
        2.0 * max(0.0, float(row.get("mae_skill_vs_baseline", 0) or 0))
        + 1.3 * max(0.0, float(row.get("directional_edge", 0) or 0))
        + min(
            0.35,
            float(row.get("walkforward_origins", 0) or 0) / 120.0,
        )
    )


def maybe_promote(asset, cfg, row):
    path = DATA / cfg["projection_file"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    projections = payload.get("projections", [])
    idx = next(
        (
            i
            for i, item in enumerate(projections)
            if item.get("horizon") == "1 Year"
        ),
        None,
    )
    if idx is None:
        return False, None

    incumbent = projections[idx]
    incumbent_score = score(incumbent)
    challenger_score = score(row)

    skill = float(row.get("mae_skill_vs_baseline", 0) or 0)
    edge = float(row.get("directional_edge", 0) or 0)
    n = int(row.get("walkforward_origins", 0) or 0)
    non_skill = float(row.get("nonoverlap_mae_skill", -9))
    non_edge = float(row.get("nonoverlap_directional_edge", -9))
    old_skill = float(incumbent.get("mae_skill_vs_baseline", 0) or 0)
    old_edge = float(incumbent.get("directional_edge", 0) or 0)

    promote = bool(
        n >= 28
        and skill > 0.03
        and edge >= 0.01
        and non_skill > 0
        and non_edge >= 0
        and challenger_score > incumbent_score + 0.04
        and skill >= old_skill - 0.01
        and edge >= old_edge - 0.02
    )

    if promote:
        new_row = dict(row)
        new_row["selected_model"] = "Dedicated macro + positioning 1Y champion"
        new_row["champion_reason"] = (
            "Beat incumbent using the exact published OOS formula plus the "
            "non-overlapping annual sanity check."
        )
        projections[idx] = new_row
        payload["projections"] = projections
        payload["note"] = (
            payload.get("note", "")
            + " 1Y uses an audited fixed-formula champion-challenger process "
            "with a non-overlapping annual sanity check."
        ).strip()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return promote, {
        "incumbent_score": round(incumbent_score, 4),
        "challenger_score": round(challenger_score, 4),
        "incumbent_skill": round(old_skill, 4),
        "challenger_skill": round(skill, 4),
        "incumbent_edge": round(old_edge, 4),
        "challenger_edge": round(edge, 4),
        "challenger_nonoverlap_skill": round(non_skill, 4),
        "challenger_nonoverlap_edge": round(non_edge, 4),
        "audited_live_formula": True,
    }
