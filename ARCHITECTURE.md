# Canonical architecture

Production: one source snapshot → completed exchange-session bars → causal indicators → confirmed structure → fixed-recipe evaluation → immutable issues → validated coherent bundle → static dashboard.

## Data
metals/data.py downloads each essential symbol/interval once. Both metals share an as-of time. Sources are hashed and retained in Actions evidence. Missing essential sources abort. Bad OHLC, duplicate timestamps, missing volume, forming bars, missing expected bars and suspected roll gaps are reported separately. No invented fill bars.

CME GC/SI calendars handle New York DST, the 18:00 anchor, maintenance and holidays. A 4H aggregate requires every expected hourly constituent; the final bar follows the session close and may be shorter. Weekly bars require every expected daily session. Quote and analysis timestamps are separate.

## Indicators and structure
metals/indicators.py centralizes Wilder-seeded indicators and qualified rolling volume measures. Nine fixed dimensionless features feed models. Known historical missing bars are represented by a gap-fraction feature; indicators use observed completed bars. No prices are filled. Future target gaps and suspected past roll dependencies exclude affected origins. Revised macro series are not predictive features.

metals/structure.py processes the complete available history chronologically. Pivots require three right-hand bars. Major pivots require displacement, separation and structural quality. Signals carry pivot and confirmation times. Buffered closes create breaks; wick sweeps create watches. Reversals require later CHoCH, later HL/LH, then later BOS. Failed breaks create new events without rewriting the original signal.

## Forecasts
Fixed Ridge returns and Logistic direction with separate Platt calibration. Every fold purges training and calibration labels before the next partition. Origins are globally horizon-spaced. The final year of daily data / 60 trading days of hourly data is held out under the same recipe. All gates must pass; research and primary forecasts are separate.

## Publication
metals/pipeline.py stages both assets and all eight validation files. A manifest records code, source and run hashes. The UI reads only dashboard.json; cards cannot mix independently refreshed files.

The active workflow runs regression tests, full evaluation, desktop/mobile browser checks, and one data commit. Older writers are manual no-ops. The existing repository and GitHub Pages hosting are retained. The final master step checks the deployed run ID and both asset views.

## History
Setup issuance levels never move. Observations separately record stop/target touches. Forecast issues never change; outcomes use the exact completed origin plus the declared trading-bar horizon. Legacy history is preserved separately and cannot qualify a corrected model.
