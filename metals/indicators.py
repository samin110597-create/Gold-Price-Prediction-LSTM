"""Causal indicators. All windows include only the current completed bar and its past."""
import numpy as np
import pandas as pd

def wilder(s, n=14):
    values = np.asarray(s, dtype=float)
    out = np.full(len(values), np.nan)
    count, total, previous = 0, 0.0, np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            count, total, previous = 0, 0.0, np.nan
            continue
        if not np.isfinite(previous):
            count += 1
            total += value
            if count == n:
                previous = total / n
                out[i] = previous
        else:
            previous = (previous * (n - 1) + value) / n
            out[i] = previous
    return pd.Series(out, index=s.index)

def compute(frame):
    f = frame.copy()
    c, h, l, v = (f[k] for k in ("close", "high", "low", "volume"))
    for n in (9, 20, 50, 100, 200):
        f[f"ema{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    for n in (50, 200):
        f[f"sma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain, loss = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    f.loc[(gain == 0) & (loss > 0), "rsi"] = 0
    f.loc[(gain == 0) & (loss == 0), "rsi"] = 50
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = wilder(tr)
    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    plus.iloc[0] = minus.iloc[0] = np.nan
    f["plus_di"] = 100 * wilder(plus) / f.atr
    f["minus_di"] = 100 * wilder(minus) / f.atr
    denom = (f.plus_di + f.minus_di).replace(0, np.nan)
    dx = 100 * (f.plus_di - f.minus_di).abs() / denom
    dx = dx.where(denom.notna(), 0).where(f.plus_di.notna())
    f["adx"] = wilder(dx)
    f["macd"] = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd_signal"] = f.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_hist"] = f.macd-f.macd_signal
    f["bb_mid"] = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    f["bb_upper"], f["bb_lower"] = f.bb_mid + 2*sd, f.bb_mid - 2*sd
    f["bb_width"] = 4*sd / f.bb_mid
    f["bb_percent_b"] = (c-f.bb_lower) / (4*sd).replace(0, np.nan)
    f["donchian_high"] = h.shift().rolling(20).max()
    f["donchian_low"] = l.shift().rolling(20).min()
    lo, hi = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"] = 100*(c-lo)/(hi-lo).replace(0, np.nan)
    f["stoch_d"] = f.stoch_k.rolling(3).mean()
    typical = (h+l+c)/3
    valid_v = v.where(v > 0)
    money = typical * valid_v
    mf_up = money.where(typical.diff() > 0, 0).where(valid_v.notna()).rolling(14).sum()
    mf_dn = money.where(typical.diff() < 0, 0).where(valid_v.notna()).rolling(14).sum()
    f["mfi"] = 100-100/(1+mf_up/mf_dn.replace(0, np.nan))
    f.loc[(mf_dn == 0) & (mf_up > 0), "mfi"] = 100
    f["cmf"] = (((2*c-h-l)/(h-l).replace(0, np.nan))*valid_v).rolling(20).sum()/valid_v.rolling(20).sum()
    f["obv"] = (np.sign(c.diff()).fillna(0)*v).cumsum()
    f["relative_volume"] = valid_v / valid_v.shift().rolling(20).mean()
    f["rolling_vwap20"] = (typical*valid_v).rolling(20).sum()/valid_v.rolling(20).sum()
    f["realized_volatility20"] = c.pct_change().rolling(20).std()
    f["atr_percentile"] = f.atr.rolling(252, min_periods=60).rank(pct=True)
    return f
