# Gold & Silver Metals Intelligence

Live application: https://samin110597-create.github.io/Gold-Price-Prediction-LSTM/

The deployed application is a causal market-research terminal for gold (GC=F) and silver (SI=F). The original LSTM notebook and the V2–V8 experiments remain as research history. They are not the current production forecasting engine.

## Terminal
- One timestamped snapshot for both metals, with 15m, 1H, exchange-session 4H, daily and completed weekly bars.
- Wilder-seeded RSI/ATR/ADX; moving averages, momentum, volatility and qualified volume measures.
- Internal and major pivots, BOS/CHoCH, divergence at confirmation time, failed breaks and staged reversal watches. Liquidity/absorption/distribution labels are OHLCV proxies.
- Strict and Adaptive conditional setup maps with immutable entry zones, stops and observed swing objectives. Confluence is not probability. No BUY/SELL is issued without independently validated target-before-stop evidence.
- Fixed-recipe purged walk-forward 4H/1D/1W/1M research estimates, probability calibration and explicit baseline comparisons. Revised recipes use a separate recent evaluation period; this is research evidence, not a claim of an untouched dataset or a profitable edge. Unvalidated estimates are labeled prominently.
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

## Technical signals and education

The technical workbench adds 17 causal playbooks with fixed triggers, stops and available TP1/TP2, Strict/Adaptive filters, chart markers, individual lessons and separate historical replay evidence. See [TECHNICAL_SIGNALS.md](TECHNICAL_SIGNALS.md) for the exact interpretation and limitations. Replay hit rates are not live calibrated probabilities.

## Forecast evidence and data freshness

The page displays actual observed reference prices separately from model estimates, estimated 80% ranges, sample counts, error rates, interval coverage and failed validation gates. Recipe 2.0 scales returns and calibration residuals by volatility known at each origin. Monthly calibration starts with 1,260 daily bars and extends backward based on available complete sample counts, never based on performance. Monthly recent evaluation uses 630 daily bars. Missing target windows remain excluded; a small sample does not become validated just because an estimate exists.

The previous recipe is evaluated on the same source history. Paired comparisons use only identical origin, target and partition tuples. The revised model improves recent interval coverage in the release audit, but point errors and direction do not consistently beat the baseline. The interface exposes that limitation. Recipe 2.0 remains the default. The 3.0 challenger described below can replace a research estimate only under explicit promotion criteria; a research promotion never bypasses the trading-validation gates.

Data comes from Yahoo Finance GC=F and SI=F continuous futures. It is **real provider data delivered as delayed snapshots**, not a streaming exchange feed. The observed quote timestamp can lag the build; the page displays quote age, snapshot age and last successful page check. Builds are scheduled at minutes 12, 27, 42 and 57 of each hour. GitHub scheduling may delay publication. The visible page polls the manifest every 60 seconds, fetches changed bundles, verifies the dashboard SHA-256 and matching run metadata, and preserves the last verified snapshot if a refresh fails. Snapshots older than 30 minutes pause current-watch eligibility. Reloading cannot eliminate provider delay.

Background: [Yahoo exchange data policies](https://help.yahoo.com/kb/SLN2310.html), [GitHub scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule), [chronological evaluation and gaps](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Delayed intraday candles are excluded using the provider quote timestamp, even when their nominal end is already behind the wall clock. A completed session is accepted after a conservative 30-minute finalization allowance. The quality report records the provider cutoff and excluded pending bars.


## Dated outlook and additional providers

Each forecast now gives a target exchange date/time, upward/downward/flat bias, estimated future price, residual range, current reference and recent mean error. Targets follow exchange sessions and exclude weekends and session breaks. An elapsed target or stale source prevents a current validated label. Bull/base/bear paths use only confirmed 4H structure zones, with a completed close and retest/reclaim rule. Missing objectives remain unavailable.

Five server-side adapters use repository Actions secrets; keys never enter browser code, JSON, logs or artifacts. Canonical names are `FINNHUB_API_KEY`, `FMP_API_KEY`, `FRED_API_KEY`, `POLYGON_API_KEY`, `ALPHA_VANTAGE_API_KEY`; common aliases are accepted in the workflow. Access is tested rather than assumed. The initial repository audit on September 16 returned **KEY NOT CONFIGURED** for all five bindings. The integrations are ready, but their live access cannot be certified until credentials are available to this repository workflow. Keys in another repository or an unselected GitHub Environment are not automatically available here.

| Provider | Requested data | Minimum refresh | Instrument / limitation |
| --- | --- | --- | --- |
| Finnhub | GLD, SLV quotes | 15 minutes | ETF shares, separate from metals futures |
| FMP | GCUSD, SIUSD commodity quotes | 15 minutes | Contract basis must remain distinct; free-plan entitlement tested |
| FRED | DFII10, DGS10, T10YIE, DTWEXBGS initial-release history | 24 hours | Daily macro, never a live metals quote |
| Polygon / Massive | C:XAUUSD, C:XAGUSD previous-day bars | 24 hours | Spot previous-day close; period start is not a live timestamp |
| Alpha Vantage | GOLD, SILVER spot quotes | 6 hours | At most eight scheduled requests/day here; 25/day free budget may be shared with other usage |

Finnhub/FMP use at most 192 scheduled calls/day each in this workflow. Failures are cached too, preventing retry storms. Access denials and rate limits remain explicit. Previous observations retained during outages keep their original timestamps. Primary chart, technicals and forecasts remain GC=F/SI=F; snapshots of spot and ETF prices are never spliced into historical futures bars. FRED features use initial releases and a two-UTC-day availability delay; stale macro observations are excluded. No latest revised macro series is backfilled into historical model features.

Recipe 3.0 is a predeclared challenger: no-change, historical median, 25%, 50%, or 100% of the volatility-scaled Ridge prediction. Each historical block selects its candidate by absolute error on the first half of already completed calibration labels. The second half calibrates intervals and probabilities. Neutral point forecasts count as directional abstentions, not correct bearish predictions. Promotion requires at least 5% lower exact-matched MAE in both historical partitions, 100 earlier and 12 recent samples, no worse Brier score, and 68–90% interval coverage. Failed candidates remain visible in the evaluation and are not promoted. Historical research performance still requires prospective confirmation.

API references: [Finnhub](https://finnhub.io/docs/api/quote), [FMP](https://site.financialmodelingprep.com/developer/docs/stable/commodities-quote), [FRED initial releases](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [Polygon/Massive previous-day bars](https://massive.com/docs/rest/forex/aggregates/previous-day-bar), [Alpha Vantage](https://www.alphavantage.co/documentation/).

### September 23 accuracy and technical update

- Recipe 4.0 is a fixed recency-weighted technical challenger: ATR-scaled Ridge,
  calibrated logistic direction, ADX/DI, MACD/ATR, volatility rank and band width.
  Training weights halve every 252 daily / 1500 hourly bars. Damping is selected
  only from past calibration observations; subsequent observations calibrate ranges.
- Research promotion still requires at least 5% lower matched-origin error in both
  historical partitions, sufficient cases, no worse Brier score and acceptable range
  coverage. Failed challengers do not replace the deployed recipe. These historical
  comparisons are research selection, not independent proof of a trading edge.
- Forecast cards report prospective results for their exact model lineage, keeping
  historical versions and neutral predictions from inflating current-model accuracy.
- The current technical brief summarizes completed 1H/4H/daily bars with explicit
  timestamps, structure/EMA agreement, momentum, ATR extension and known levels.
  Conditional 4H scenarios show known TP1/TP2 zones and failed-break invalidation;
  these are not automatically triggered trades or guaranteed price forecasts.
- FRED initial-release history is requested in five-year real-time windows to remain
  within its 2000-vintage JSON limit. The earliest release per observation is retained.
  Alpha Vantage metal requests are spaced 15 seconds apart; daily quotas still apply.
