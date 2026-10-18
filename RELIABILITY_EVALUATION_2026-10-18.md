# Challenger 7 release evaluation

Snapshot: 2026-10-18T17:37:49.016513+00:00 (UTC). Source hash: `51c5ec5dd563752bb9fb575de59061c2313864a61b4bbd63c9c1fa6721a1c72e`.

Same-date recent comparisons with incumbent recipe 2. Positive reduction means lower mean absolute error. This is historical research, not a prospective accuracy claim.

| Metal | Horizon | Matched recent cases | MAE reduction | Promoted |
|---|---|---:|---:|---|
| gold | 4H | 344 | 0.92% | No |
| gold | 1D | 238 | 0.73% | No |
| gold | 1W | 43 | 1.48% | No |
| gold | 1M | 13 | -4.47% | No |
| silver | 4H | 344 | 0.36% | No |
| silver | 1D | 238 | 0.12% | No |
| silver | 1W | 43 | 1.74% | No |
| silver | 1M | 3 | -70.09% | No |

No candidate met every required promotion criterion. The incumbent remains active and the challenger is tracked separately. The monthly samples are particularly weak; silver has only three matched recent cases. Narrower-looking targets or improved average error alone are not adequate evidence.

The deployment adds stricter error-confidence gates, proper interval scoring and non-overlapping forward evidence. It does not claim improved live forecast accuracy. Full per-origin source and evaluation records are produced by the workflow artifacts.
