# Forecast reliability upgrade — 2026-10-18

This is a research and risk-control upgrade, not a claim of institutional-grade
market data, profitable trading, or proven future accuracy.

## What the audit found

The incumbent often has higher price error than an unchanged-price forecast.
Many nominal live forecasts overlap the same price movement. Coverage alone can
also reward an unnecessarily wide range. These issues require explicit evidence,
not more confident target prices.

## Fixed challenger 7

Retain the volatility-scaled Ridge model and three disjoint past calibration
blocks used in recipe 6. Split the selection block chronologically into two
subperiods. A damped or bounded-bias directional candidate is eligible only if it
beats no change by at least 5% in BOTH subperiods. Otherwise its point estimate is
no change. Choose the eligible candidate with the lowest whole-selection MAE.
The final calibration block still has no role in candidate selection. Model
refits only use matured labels; a future-outcome perturbation test protects this.
This additional safeguard can also miss real regime changes. It is a challenger,
not a guaranteed improvement. Recipe 2 remains the incumbent unless all research
promotion criteria pass for the individual metal and horizon.

## Stronger evidence

- Paired moving-block bootstrap confidence intervals compare absolute errors on
  identical outcomes, including the no-change benchmark. Fixed seed, 600
  resamples, block length max(2, floor(n^(1/3))), minimum 30 observations.
- An apparent reduction is insufficient: both earlier and recent partitions
  require a positive lower confidence bound for promotion. Bounds are exploratory,
  not corrected for repeated research or multiple comparisons. They do not prove
  prospective skill.
- Nominal 80% interval score = width + 10 times lower-tail shortfall + 10 times
  upper-tail overshoot. This penalizes both excessive width and missed moves.
  A challenger must not worsen it, alongside existing MAE, RMSE, tail error,
  probability and coverage requirements.
- Live reporting retains every immutable issue and all-resolved descriptive
  metrics, but counts non-overlapping forecast windows separately. Selection is
  chronological, never based on which forecasts won. Only the non-overlapping
  subset supplies the live error confidence interval. Non-overlap alone does
  not imply independence; block resampling also addresses some dependence.
- Neutral forecasts no longer count as correct down calls in newly resolved
  ledger entries. Old issued forecasts and recorded outcomes are not rewritten.

Challenger predictions are also frozen in a separate shadow channel before
outcomes, with their own non-overlapping evidence. They are never pooled with
active-model results. Local experiments cannot enter production/shadow evidence.

## Interpretation and limits

The dashboard explicitly reports whether lower error versus unchanged price is
established and shows the uncertainty in that estimate. WAIT remains appropriate
when samples or improvement are insufficient. These additions do not lower gates
to manufacture more signals. Historical analysis uses the provider's latest
continuous-futures history, not point-in-time vintages or a mapped tradable
contract. Spot technicals remain separate from futures forecasts. Trading costs,
execution, order-book liquidity and a verified event calendar are not modeled.
Real prospective performance must accumulate after release before accuracy claims.
