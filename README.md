# Gold & Silver Metals Intelligence

Live application: https://samin110597-create.github.io/Gold-Price-Prediction-LSTM/

The deployed application is a causal market-research terminal for gold (GC=F) and silver (SI=F). The original LSTM notebook and the V2–V8 experiments remain as research history. They are not the current production forecasting engine.

## Terminal
- One timestamped snapshot for both metals, with 15m, 1H, exchange-session 4H, daily and completed weekly bars.
- Wilder-seeded RSI/ATR/ADX; moving averages, momentum, volatility and qualified volume measures.
- Internal and major pivots, BOS/CHoCH, divergence at confirmation time, failed breaks and staged reversal watches. Liquidity/absorption/distribution labels are OHLCV proxies.
- Strict and Adaptive conditional setup maps with immutable entry zones, stops and observed swing objectives. Confluence is not probability. No BUY/SELL is issued without independently validated target-before-stop evidence.
- Fixed-recipe purged walk-forward 4H/1D/1W/1M forecasts, calibrated directional probabilities, baseline comparisons and an untouched final evaluation period. Weak estimates remain research-only; the primary forecast says WAIT.
- Exact-rule 5-session descriptive event audits for BOS/CHoCH, divergence, reversal and flow proxies; these do not qualify trade setups.
- A new immutable forecast issue ledger with separate exact-bar outcomes. Legacy forward history remains separate.

## Run
Python 3.12:
~~~bash
pip install -r requirements-canonical.txt
python -m pytest tests/test_canonical.py -q
python -m metals.pipeline --raw .snapshot --stage .stage --history data
~~~

Both metals must pass schema, timing, geometry and model-integrity checks before the staged bundle is published. Missing essential sources abort the build.

Production forward scoring also excludes issues more than 30 minutes after their completed reference origin.

The single scheduled workflow is Canonical Metals Test Build Publish. It tests, builds, exercises desktop/mobile gold and silver views, commits one coherent bundle, and verifies the existing GitHub Pages site. Six old writers are retired; legacy HTML patchers are no longer run.

## Evidence
- [AUDIT.md](AUDIT.md): defects and dispositions.
- [ARCHITECTURE.md](ARCHITECTURE.md): calculation boundaries.
- [VALIDATION_BASELINE.md](VALIDATION_BASELINE.md): unchanged frozen pre-change metrics.
- [VALIDATION_REPORT.md](VALIDATION_REPORT.md): corrected protocol and limits.
- data/validation: full timestamped origins, metrics, gates and research estimates for each asset/horizon.
- data/run_manifest.json: run, commit, code/source hashes and bundle checksums.
- data/forward_ledger_v2.json and data/setup_ledger.json: immutable issues and separate outcomes.
- Actions artifacts: regression XML, browser report, four screenshots and raw source snapshots, retained 30 days.

## Known limits
Yahoo continuous futures do not provide verified per-bar contract mappings or historical revision vintages. Roll-gap detection is a proxy, not back-adjustment. Session VWAP is an OHLCV approximation. A final session 4H bar can be three hours or shorter on holidays.

External daily prices are lagged by 24 hours and are context only. A verified release calendar, vintage-safe long-horizon macro model, validated Elliott/Wyckoff phase model, and independently validated Strict/Adaptive barrier probabilities are unavailable. These are explicitly withheld.

Correctness improvements do not establish better predictive accuracy. No profitable-trading or out-of-sample performance improvement is claimed.
