# Corrected validation protocol

The pre-change evidence in research/baseline-2026-09-14 is unchanged. Corrected evaluations are a new lineage. Existing projection ensembles used full-outcome weights, and the legacy ledger used nearest-bar resolution; those figures cannot be treated as comparable validated champions.

## Exact production recipe
- Ridge(alpha=20) for returns; LogisticRegression(C=0.1) for direction.
- Separate LogisticRegression(C=1) calibrates decision functions on an earlier disjoint calibration window.
- Nine fixed features: 1/5/20-bar returns, EMA20–50 spread, RSI, ATR fraction, prior Donchian range position, CMF and the observed 20-bar gap fraction.
- No grid search, ensemble weight selection or probability inversion.
- Expanding training, purged disjoint calibration, sequential test blocks. Origins globally spaced by the horizon.
- Final holdout: 252 daily / 1380 hourly bars, same fixed recipe, no holdout tuning.
- Live formula repeats the same procedure on matured history. Calibration-residual quantiles define an 80% interval; no post-evaluation widening.

## Gates
All must pass: at least 100 walk-forward origins; ≥3 percentage-point directional edge over earlier-training median-return direction; paired block-bootstrap 95% edge lower bound >0; MAE skill ≥3%; positive Brier skill against earlier-training direction prior; at least 12 holdout origins; positive holdout edge, MAE skill and Brier skill; holdout interval coverage 68–90%; live features in distribution.

No threshold is lowered to obtain a signal. The fixed 0.60 selective probability slice is descriptive and cannot promote a model. Records contain label-maturity, calibration, origin and target timestamps.

## Reporting
Each horizon file contains accuracy, balanced accuracy, MCC, paired edge interval, MAE, zero-change and training-median baselines, ATR-normalized error, Brier/log loss, calibration bins, interval coverage/width, and selective coverage. Holdout metrics remain separate.

No predictive improvement is claimed merely because correctness defects were fixed. A fair improvement claim requires the same completed source history, horizons, origin calendar and recipe definitions. Frozen published figures remain historical context.

## Technical and setup limits
BOS/CHoCH, divergence and liquidity proxies are causal descriptive features. The legacy generic signal audit does not validate these exact rules. Strict/Adaptive target-before-stop probability is null until an independent recipe audit establishes evidence. WATCH is a conditional map, not an executable recommendation.

The barrier helper resolves ambiguous candles stop-first and adverse stop gaps at the opening price. Regression tests verify mechanics; they are not historical edge evidence. Trading-performance claims would also require fees, slippage and contract-roll treatment.

## Verification
The workflow produces regression JUnit XML, timestamped evaluations, a browser report and desktop/mobile screenshots for each asset. Its final master step checks the exact deployed run ID. Consult the Actions conclusion; this document itself is not a passing test result.

## Source-quality correction (recipe 1.1)
The first mechanical validation run exposed zero-volume/flat-bar CMF propagation and an overly broad historical-gap exclusion. Known zero volume is preserved, a zero-range bar contributes zero CMF multiplier, and actual missing volume remains unavailable. Known historical gaps become a fixed explicit feature; missing target bars and 61-bar suspected-roll dependencies still exclude scoring. No performance threshold changed. Earlier recipe 1.0 results are not mixed into recipe 1.1 evidence.

Forward issues carry model-code identity and production/research channel. Pre-release branch evaluations are preserved but excluded from the displayed production-forward score. OHLC barrier excursion values are upper bounds because intrabar event order is unknown.

Production forward scores include only forecasts issued within 30 minutes of their completed reference bar. Older reference origins remain recorded with their latency but cannot inflate the forward score. Forecast horizons are measured from the explicitly displayed completed reference bar, not from a later quote.

## Recorded release evaluation

Snapshot 20260914T184818Z-34882952760; engine commit 66735d168d0d44c2f20f46f786a6fdb74b564edd. These are the actual corrected results, not a baseline-comparable improvement claim.

| Metal | Horizon | OOS origins | OOS hit rate | Baseline hit rate | MAE skill | Holdout origins | Primary |
|---|---|---:|---:|---:|---:|---:|---|
| gold | 4H | 1935 | 49.72% | 51.94% | -0.70% | 342 | WAIT |
| gold | 1D | 3711 | 51.36% | 52.82% | -0.48% | 238 | WAIT |
| gold | 1W | 592 | 53.38% | 55.24% | -0.38% | 44 | WAIT |
| gold | 1M | 0 | Unavailable | Unavailable | Unavailable | 0 | WAIT |
| silver | 4H | 1905 | 49.76% | 51.29% | -1.07% | 340 | WAIT |
| silver | 1D | 3261 | 51.52% | 51.89% | -0.37% | 238 | WAIT |
| silver | 1W | 524 | 49.43% | 50.00% | -2.72% | 44 | WAIT |
| silver | 1M | 0 | Unavailable | Unavailable | Unavailable | 0 | WAIT |

Monthly calibration has insufficient eligible contiguous target windows after source-gap exclusions. No monthly estimate or probability is fabricated.

Branch verification: 21 regression tests passed, full pipeline schema/timing/geometry checks passed, and both assets passed desktop/mobile interactions across both modes and five chart timeframes without JavaScript errors. Evidence run: https://github.com/samin110597-create/Gold-Price-Prediction-LSTM/actions/runs/34882952760 . Deployed verification is a separate master-workflow step.
