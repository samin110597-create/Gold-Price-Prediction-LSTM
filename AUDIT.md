# Gold & Silver Intelligence audit

Baseline: `ebc895e988c3f07a18cee6d2b8802b6a60fc6eda`, captured 2026-09-14 before implementation. Exact input JSON and SHA-256 manifest are in `research/baseline-2026-09-14/`. This is an ongoing engineering audit; baseline metrics have not been independently reproduced yet.

## Confirmed production defects

1. V8 gold retest zone `[4383.69, 4374.8]` is reversed. Targets 4404.74 and 4436.29 are below long trigger 4625.5; displayed R:R is negative. Silver stop 64.99 intersects entry zone `[64.84, 65.78]`.
2. `recent_signal_edge` multiplies negative historical edge by bearish direction and awards bullish points. Descriptive underperformance does not establish an inverse strategy.
3. Technical 4H bars use generic calendar resampling after timezone removal. Forming daily/weekly/intraday bars are not excluded.
4. Multiple downloads in one run can produce different reference prices/times. Several enrichers catch failures and leave old outputs in place. Fast workflows use different concurrency groups and publish only subsets.
5. Divergence markers are placed on the pivot date although the pivot requires three future bars. Current fixed-rule signal audit does not audit divergence or BOS/CHoCH at all; it must not be interpreted as validating them. Zigzag threshold uses the last ATR for the whole historical window.
6. Projection V2 and the restored 4H specialist derive ensemble weights from all evaluated outcomes, then evaluate the weighted ensemble on those same outcomes. These are contaminated ensemble estimates.
7. V6 runner makes origins maturity-safe, but raw V6 entry point does not enforce the same spacing. Macro features use latest revised FRED histories with approximate lags; no vintage proof.
8. Forward ledger uses nearest timestamp matching, can resolve forming bars, truncates to the latest 3000 entries, and lacks immutable model/run/commit provenance. Legacy records must be preserved and distinguished from corrected records.
9. V7.1 always emits raw direction and point forecasts even for FAIL. Its `trust_label` can lift FAIL to MIXED based on only 20 forward outcomes.
10. Wilder RSI uses recursive EWM initialization and returns NaN for zero losses; ATR/ADX implementations are duplicated. The technical score counts multiple correlated trend/momentum indicators.
11. Daily volume-weighted price is occasionally labelled VWAP without the rolling qualifier. It is not session VWAP or exchange transaction VWAP.
12. Six overlapping Actions workflows and nine dashboard patches mutate shared `index.html`; there is no coherent snapshot manifest or acceptance test. Original README describes LSTM work which is not the deployed pipeline.

## Model status

No model is promoted by this audit. Existing published statuses and metrics are frozen in VALIDATION_BASELINE.md. Correctness changes require a new evaluation lineage; they are not evidence of higher predictive accuracy.


## Canonical release dispositions
Six legacy writers and nine HTML patchers are retired from production. V2–V8 scripts and frozen history remain research-only.

- Invalid zones, wrong-sided targets and intersecting stops are rejected; new issued levels are immutable.
- No negative historical signal edge is inverted into positive confluence. Three capped families replace correlated-indicator stacking.
- Exchange-calendar completion replaces generic resampling; forming/incomplete bars are excluded.
- Pivots have separate pivot and confirmation times. Later failure is a new event. Session ranges cannot look ahead.
- Fixed formulas and purged evaluation replace contaminated ensemble promotion.
- Every primary forecast must pass all gates; a small legacy forward sample cannot lift its status.
- Wilder initialization and zero-loss RSI are centralized and corrected. Rolling/session VWAP approximations are explicitly labelled.
- Exact completed-origin ledger resolution replaces nearest matching. No corrected history truncation.
- Gold/silver desktop/mobile interactions are checked before publication and again on the deployed run.

Remaining unavailable: roll mapping/vintages, verified release calendar, corrected long-horizon macro forecast, validated Elliott/Wyckoff phase model and independently validated setup barrier probabilities. No predictive-accuracy improvement or institutional participant detection is claimed.
