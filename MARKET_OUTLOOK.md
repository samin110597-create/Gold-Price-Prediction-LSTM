# Gold and silver outlook

The visible outlook separates three distinct products:

- A near-term technical bias requires fresh completed 4H and daily structure/EMA agreement. It is not a calibrated forecast. Support, resistance and reversal conditions come from existing observed zones; missing levels are not fabricated.
- The four dated statistical forecasts show the existing central estimate, historical residual band, measured coverage and recent average error. Failed validation remains conspicuous. Neither current spot prices nor narrative macro opinions move issued targets.
- A market-factor interpretation explains measured changes and dated source reports. Dollar, nominal yields, real yields, breakevens, equities, VIX and oil use their own source dates. Changes span five available observations. Yahoo daily context uses a conservative 24-hour lag. Observations older than five calendar days lose their directional interpretation. Correlated factors are not summed into a probability.

FRED's revised historical context, when initial-release access fails, is kept in `context_histories`. It never enters `macro_features`, training or historical evaluations. Availability is the retrieval time; this history is only contemporary explanatory context.

`metals/market_reports.json` contains concise, manually researched summaries, source URLs, publication dates, observation periods, verification times and explicit review deadlines. These are not an automatic news service. Reports are excluded before verification time and labelled background-only after the review deadline. Source publication dates with no verified time remain dates, not invented timestamps. The official Fed RSS feed is checked at most every four hours; publication dates and HTTPS hostnames are checked. Headlines do not imply a policy surprise.

No claim is made to cover every market factor. Missing live positioning, inventories, regional premiums, lease rates, current silver ETF flows and a verified event calendar are explicitly disclosed. The new narrative and UI do not establish better forecasting accuracy. Exact arrival times for structural levels and causal dollar impacts are unavailable.

Verification includes timing/expiry tests, causal exclusion of future context, stale-data behavior, asset-specific interpretation, headline URL filtering, HTML escaping, model immutability and browser checks on gold/silver and mobile.
