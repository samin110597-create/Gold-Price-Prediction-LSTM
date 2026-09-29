# Gold and silver forecast error controls

Recipe 6 is a research challenger. Each metal and horizon is evaluated separately
against deployed recipe 2 on identical forecast origins, target dates and partitions.
The thresholds are fixed before the new production snapshot is fetched.

The challenger retains volatility-scaled Ridge features, refits sequentially, and
splits past calibration labels into three disjoint chronological blocks:

1. Estimate median residual corrections, bounded to half an origin volatility unit
   and shrunk by n/(n+50).
2. Select no change or 25%, 50%, 100% Ridge, with or without correction. A directional
   candidate must beat no-change MAE by at least 5% on this past block.
3. Calibrate probability and interval residuals independently. Interval endpoints
   use outward finite-sample order statistics. Coverage remains empirical, not
   guaranteed in changing markets.

At least 60 nonoverlapping past calibration labels are required. Sparse monthly
histories can therefore have no eligible challenger forecast; missing evidence
is not a successful result. All target labels must mature before training,
selection or calibration can use them. Recent evaluation uses sequential refits,
not an untouched holdout. Previously examined histories are research evidence.

Promotion requires at least 100 earlier and 12 recent matched forecasts, at least
5% lower MAE in both partitions, no worse Brier score, RMSE or 90th-percentile
absolute error, and 68–90% coverage of the nominal 80% interval. Existing trading
validation gates remain additional requirements. Repeated research comparisons
cannot establish prospective accuracy.

The site reports signed error (forecast minus actual), average absolute error and
90th-percentile absolute error for the active model. Positive bias means systematic
overprediction. Errors are returns relative to each forecast's origin price;
prices, ranges and issued forecasts are never rebased after issuance.

Validation: unit tests include future-outcome perturbation, independent interval
calibration, bounded correction, zero-return abstention and signed/tail errors.
The deployment workflow additionally fetches a current snapshot and tests the
rendered gold/silver pages, mobile layout and published data checksums.
