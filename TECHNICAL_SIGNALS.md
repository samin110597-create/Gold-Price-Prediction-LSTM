# Technical signals for trading study

The technical workbench adds 17 fixed-rule playbooks to both metals and all five chart timeframes. It is separate from the forecast model. Detecting a technical condition does not establish a profitable strategy or calibrated probability.

## How to use the page

1. Choose gold or silver, then a timeframe. Use 4H/daily for swing context; 15m/1H are shorter timing views, not automatically swing trades.
2. Open Trading signals. All recent signals includes failed and expired examples. Current qualifying watches requires fresh data, untouched levels, a current watch/confirmation and the selected mode's criteria.
3. Read the fixed close-confirmation trigger, invalidation, TP1/TP2 and reference reward/risk. No known reward-side objective means no mapped trade.
4. A later completed close beyond the trigger, within three bars, confirms the rule. Replay enters at the next bar's open only if the stop/objective remain valid and reward/risk still meets the mode. A target reached before entry is MISSED, not a win.
5. Use Show on chart, hover/click markers, or Study this signal. Learn signals explains detection, interpretation and the common trap. Historical replay shows older and recent samples separately.

## Rules

| Family | Playbooks |
|---|---|
| Trend | EMA20 pullback, 20-bar range breakout, breakout retest, major BOS/CHoCH |
| Momentum | RSI 50 reclaim, MACD histogram turn |
| Volatility | Bollinger squeeze release, inside-bar breakout |
| Reversal | Engulfing at a known level, wick rejection, major liquidity sweep, confirmed RSI divergence, sweep plus divergence, staged structure reversal |
| Flow | CMF/price turn, selling-climax watch, session VWAP reclaim |

Strict requires quality ≥75/100, aligned higher-timeframe EMA20/50 direction and TP1 ≥1.5R. Adaptive requires quality ≥50 and TP1 ≥1R. Quality uses capped trend (35), momentum (25), volume/flow (15) and location (25) components. It is not a percent chance of success. Weekly Strict has no higher timeframe in this application and therefore does not qualify.

All parameters are fixed research choices, not optimised thresholds. Candle/pivot signals are dated when knowable. Higher-timeframe values come only from bars completed by the signal time. Trigger, stop and objectives are frozen; observations are stored separately in data/signal_ledger.json. The first publication timestamp is distinct from historical detection time.

## Targets, status and limitations

Targets use then-known confirmed swings, the prior 20-bar extreme and EMA20/50 frozen at detection, in reward-side order. Unavailable TP2 stays blank. Trigger/stop buffers use signal ATR; targets do not come from arbitrary ATR projections. A later price touch never moves a target or stop to preserve the idea.

WATCH means awaiting a later completed close. CONFIRMED means the rule's close condition was met; it is not a trade order. TRIGGERED and subsequent target/stop/timeout states describe hypothetical historical replay, not brokerage fills. After three bars without confirmation the watch expires. The outcome window is 20 bars. Current eligibility checks completed-bar extremes and the timestamped quote; forming-bar extremes are not independently scored.

OHLCV cannot establish institutional ownership, true order flow or transaction VWAP. A high-volume selling climax remains a reversal watch until later price confirmation. A sweep plus RSI divergence is stronger descriptive context, not proven probability.

## Replay evidence

The same fixed detection and replay functions power the displayed observations and historical audit. Confirmation is a later close, entry is the next open, adverse stop gaps fill at the open, and candles touching both TP1 and stop are scored stop-first. Insufficient entry reward/risk, missed objectives, missing data and overlapping rule/direction positions are excluded and counted. Each rule/direction allows only one overlapping position; different rules are correlated and must not be pooled as independent evidence.

The final 20% of each timeframe's source history is reported separately. Cases crossing that boundary are purged. Tables show sample size, TP1-first frequency, Wilson 95% interval and mean gross R. Fees, slippage and contract roll execution are not modelled. These are descriptive replay results, not calibrated future probabilities; no forecast gate is relaxed and no live signal probability is invented.

Full records: data/validation/{metal}_{timeframe}_playbooks.json. Tests cover append-only timing, then-known levels, completed higher timeframes, close confirmation, target-before-entry handling, stop-first ambiguity, gaps, immutable publication, stale quotes and VWAP availability.

## Indicator background

RSI measures momentum and can remain extreme through a strong trend; divergence alone is not confirmation. See [Fidelity's RSI guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/rsi). MACD is a moving-average momentum tool; see [Fidelity's MACD guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd). Bollinger bands describe volatility around an average; see [Fidelity's Bollinger Bands guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands). These references explain the underlying indicators, not this application's custom thresholds or trade results.
