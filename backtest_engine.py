# audited champion-challenger validation
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path("data")
ASSETS = {"gold": "GC=F", "silver": "SI=F"}


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def as_float(value, default=0.0):
    return default if value is None else float(value)


def as_int(value, default=0):
    return default if value is None else int(value)


def download(ticker):
    df = flatten(
        yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).copy()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df.Close.shift(1)
    tr = pd.concat(
        [df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df.High.diff()
    down = -df.Low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def score_series(df):
    c = df.Close.astype(float)
    h = df.High.astype(float)
    l = df.Low.astype(float)
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ms = macd.ewm(span=9, adjust=False).mean()
    mh = macd - ms
    rr = rsi(c)
    lo = l.rolling(14).min()
    hi = h.rolling(14).max()
    sk = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    sd = sk.rolling(3).mean()
    mid = c.rolling(20).mean()
    st = c.rolling(20).std()
    bu = mid + 2 * st
    bl = mid - 2 * st
    bp = (c - bl) / (bu - bl).replace(0, np.nan)
    ax, pdi, mdi = adx(df)

    s = pd.Series(0.0, index=df.index)
    s += np.where(c > e20, 1, -1)
    s += np.where(e20 > e50, 1, -1)
    s += np.where(e50 > e200, 1, -1)
    s += np.where(macd > ms, 1, -1)
    s += np.where(mh > mh.shift(1), 0.5, -0.5)
    s += np.select(
        [(rr >= 55) & (rr <= 72), rr >= 75, rr <= 25, rr < 45],
        [1, -0.35, 0.35, -1],
        default=0,
    )
    s += np.select(
        [(ax >= 20) & (pdi > mdi), (ax >= 20) & (mdi > pdi)],
        [1, -1],
        default=0,
    )
    s += np.select(
        [(sk > sd) & (sk < 80), (sk < sd) & (sk > 20)],
        [0.5, -0.5],
        default=0,
    )
    s += np.select(
        [(bp >= 0.55) & (bp <= 0.9), bp > 1, (bp >= 0.1) & (bp < 0.45), bp < 0],
        [0.5, -0.25, -0.5, 0.25],
        default=0,
    )
    return pd.Series(np.clip(50 + s / 7 * 50, 0, 100), index=df.index)


def technical_bt(df):
    score = score_series(df)
    c = df.Close.astype(float)
    out = []
    start = max(250, len(df) - 756)

    for h in (1, 5, 20):
        fwd = c.shift(-h) / c - 1
        s = score.iloc[start:]
        fr = fwd.iloc[start:]
        valid = s.notna() & fr.notna()
        s, fr = s[valid], fr[valid]
        sig = np.where(s >= 60, 1, np.where(s <= 40, -1, 0))
        active = sig != 0

        if not active.any():
            out.append(
                {
                    "horizon_days": h,
                    "signals": 0,
                    "coverage": 0,
                    "hit_rate": None,
                    "baseline_direction_rate": None,
                    "edge": None,
                }
            )
            continue

        signed = fr.values[active] * sig[active]
        hit = float(np.mean(signed > 0))
        pos = float(np.mean(fr.values[active] > 0))
        base = max(pos, 1 - pos)
        out.append(
            {
                "horizon_days": h,
                "signals": int(active.sum()),
                "coverage": round(float(np.mean(active)), 4),
                "hit_rate": round(hit, 4),
                "baseline_direction_rate": round(base, 4),
                "edge": round(hit - base, 4),
                "avg_signed_return_pct": round(float(np.mean(signed)) * 100, 3),
            }
        )
    return out


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def maybe(path):
    try:
        return load(path) if path.exists() else None
    except Exception:
        return None


def technical_champions(old_rows, selective_payload, context_payload):
    selective = {
        int(x["horizon_days"]): x
        for x in (selective_payload or {}).get("horizons", [])
        if "horizon_days" in x
    }
    context = {
        int(x["horizon_days"]): x
        for x in (context_payload or {}).get("horizons", [])
        if "horizon_days" in x
    }
    out = []

    for old in old_rows:
        h = int(old["horizon_days"])
        candidates = []
        old_edge = as_float(old.get("edge"), -9.0)
        candidates.append(
            {
                "horizon_days": h,
                "source": "Deterministic confluence",
                "accuracy": old.get("hit_rate"),
                "baseline": old.get("baseline_direction_rate"),
                "edge": old_edge,
                "coverage": old.get("coverage"),
                "signals": old.get("signals"),
                "current_signal": None,
                "pass": bool(old_edge > 0),
            }
        )

        s = selective.get(h)
        if s and s.get("selective_edge") is not None:
            edge = as_float(s.get("selective_edge"))
            candidates.append(
                {
                    "horizon_days": h,
                    "source": "Selective nested technical model",
                    "accuracy": s.get("selective_accuracy"),
                    "baseline": s.get("baseline_direction_rate"),
                    "edge": edge,
                    "coverage": s.get("coverage"),
                    "signals": s.get("oos_signals"),
                    "current_signal": s.get("signal"),
                    "pass": bool(s.get("audit_pass") and edge > 0.02),
                }
            )

        m = context.get(h)
        if m and m.get("lift_vs_base") is not None:
            edge = as_float(m.get("lift_vs_base"))
            candidates.append(
                {
                    "horizon_days": h,
                    "source": "Technical + cross-market meta model",
                    "accuracy": m.get("oos_accuracy"),
                    "baseline": m.get("base_signal_accuracy"),
                    "edge": edge,
                    "coverage": m.get("coverage"),
                    "signals": m.get("oos_signals"),
                    "current_signal": m.get("signal"),
                    "mode": m.get("mode"),
                    "pass": bool(m.get("audit_pass") and edge >= 0.03),
                }
            )

        candidates.sort(
            key=lambda x: (
                1 if x.get("pass") else 0,
                as_float(x.get("edge"), -9.0),
                as_int(x.get("signals")),
            ),
            reverse=True,
        )

        # Copy the winner and its candidates. Never embed the winner's list inside
        # the winner object itself; that creates a circular JSON reference.
        winner = dict(candidates[0])
        winner["candidates"] = [dict(candidate) for candidate in candidates]
        out.append(winner)

    return out


def audit(asset, ticker):
    forecast = load(DATA / ("live_forecast.json" if asset == "gold" else "silver_forecast.json"))
    projections = load(DATA / f"{asset}_projections.json")
    legacy_technical = technical_bt(download(ticker))
    selective = maybe(DATA / f"{asset}_technical_model.json")
    context = maybe(DATA / f"{asset}_technical_context.json")
    macro = maybe(DATA / f"{asset}_macro_1y.json")

    probability_rows = []
    for row in forecast.get("forecasts", []):
        edge = as_float(row.get("backtest_edge"))
        brier = as_float(row.get("brier_score"), 1.0)
        probability_rows.append(
            {
                "horizon_days": row.get("horizon_days"),
                "accuracy": row.get("backtest_accuracy"),
                "baseline": row.get("naive_baseline"),
                "edge": edge,
                "brier_score": brier,
                "oos_observations": row.get("test_observations"),
                "pass": bool(edge > 0 and brier < 0.25),
            }
        )

    price_rows = []
    for row in projections.get("projections", []):
        edge = as_float(row.get("directional_edge"))
        skill = as_float(row.get("mae_skill_vs_baseline"), -9.0)
        zone = row.get("tight_model_zone") or [None, None]
        model_price = row.get("model_price")
        inside = bool(
            len(zone) >= 2
            and zone[0] is not None
            and zone[1] is not None
            and model_price is not None
            and zone[0] <= model_price <= zone[1]
        )
        price_rows.append(
            {
                "horizon": row.get("horizon"),
                "selected_model": row.get("selected_model", "Incumbent projection model"),
                "directional_accuracy": row.get("backtest_directional_accuracy"),
                "baseline_directional_accuracy": row.get("baseline_directional_accuracy"),
                "directional_edge": edge,
                "mae_pct": row.get("backtest_mae_pct"),
                "baseline_mae_pct": row.get("baseline_mae_pct"),
                "mae_skill_vs_baseline": skill,
                "walkforward_origins": row.get("walkforward_origins"),
                "zone_contains_model_price": inside,
                "pass": bool(edge > 0 and skill > 0 and inside),
            }
        )

    champions = technical_champions(legacy_technical, selective, context)
    tests = (
        [x["pass"] for x in probability_rows]
        + [x["pass"] for x in price_rows]
        + [x["pass"] for x in champions]
    )
    rate = float(np.mean(tests)) if tests else 0.0
    grade = "PASS" if rate >= 0.70 else ("MIXED" if rate >= 0.40 else "FAIL")

    return {
        "audit_grade": grade,
        "component_pass_rate": round(rate, 3),
        "probability_forecasts": probability_rows,
        "price_projections": price_rows,
        "technical_signals_legacy": legacy_technical,
        "technical_selective": (selective or {}).get("horizons", []),
        "technical_context": (context or {}).get("horizons", []),
        "technical_champions": champions,
        "macro_1y_challenger": macro,
        "interpretation": (
            "PASS means most champion components beat simple OOS baselines. "
            "New technical and 1Y challengers can replace incumbents only when fixed "
            "walk-forward safeguards are met. FAIL means do not treat the whole system as predictive."
        ),
    }


def main():
    payload = {
        "status": "ok",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "validation_method": (
            "Purged walk-forward validation. Technical meta thresholds are chosen only on earlier "
            "calibration windows. 1Y macro model uses a 12-month label purge and lagged public "
            "macro/positioning inputs. Champion-challenger selection prevents a newer model from "
            "replacing a stronger incumbent without OOS evidence."
        ),
        "assets": {key: audit(key, ticker) for key, ticker in ASSETS.items()},
    }
    (DATA / "model_backtest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print({key: value["audit_grade"] for key, value in payload["assets"].items()})


if __name__ == "__main__":
    main()
