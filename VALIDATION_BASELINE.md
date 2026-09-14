# Frozen validation baseline

Captured from production commit `ebc895e988c3f07a18cee6d2b8802b6a60fc6eda`. These are the **previous system’s reported metrics**, not newly verified OOS results. All JSON files are preserved with hashes in `research/baseline-2026-09-14/`.

| Metal | Horizon/model | N | Direction accuracy | Baseline accuracy | Direction edge | MAE skill | Published status |
|---|---|---:|---:|---:|---:|---:|---|
| gold | 4H recent_28 | 28 | 42.86% | 46.43% | -3.57% | 1.00% | FAIL |
| gold | 4H stable_96 | 96 | 42.71% | 61.46% | -18.75% | -3.20% | FAIL |
| gold | V6 1 Day | 72 | 55.56% | 62.50% | -6.94% | 2.81% | FAIL |
| gold | V6 1 Week | 72 | 55.56% | 66.67% | -11.11% | -3.96% | FAIL |
| gold | V6 1 Month | 59 | 61.02% | 57.63% | 3.39% | -4.04% | FAIL |
| silver | 4H recent_28 | 28 | 60.71% | 50.00% | 10.71% | 3.16% | PASS |
| silver | 4H stable_96 | 96 | 44.79% | 55.21% | -10.42% | -1.70% | FAIL |
| silver | V6 1 Day | 72 | 61.11% | 59.72% | 1.39% | 7.16% | FAIL |
| silver | V6 1 Week | 72 | 47.22% | 62.50% | -15.28% | -6.37% | FAIL |
| silver | V6 1 Month | 46 | 45.65% | 47.83% | -2.17% | -15.51% | FAIL |

## Legacy forward record

Legacy issuance/maturity semantics are under audit. These samples overlap and cannot be treated as independent or as proof of calibrated probabilities.

| Metal | Horizon | Resolved | Direction accuracy | MAE % |
|---|---|---:|---:|---:|
| gold | 1 Day | 23 | 56.52% | 1.2541 |
| gold | 1 Week | 17 | 41.18% | 3.1676 |
| gold | 4H | 182 | 57.14% | 0.4663 |
| silver | 1 Day | 23 | 65.22% | 1.9712 |
| silver | 1 Week | 17 | 52.94% | 3.0176 |
| silver | 4H | 182 | 52.20% | 0.7513 |

## Before/after policy

Baseline statuses are frozen. Any changed features, bar semantics, model weights or targets start a new model version. Revised results are not directly comparable unless evaluated on identical origins and outcomes. A holdout becomes research data after inspection; do not retune against it. No accuracy improvement is claimed yet.
