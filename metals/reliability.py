"""Paired error evidence and interval scores; no forecast fitting or future inputs."""
import numpy as np


def error_evidence(actual, predicted, baseline=None):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    baseline = np.zeros_like(actual) if baseline is None else np.asarray(baseline, float)
    if not len(actual):
        return {'n': 0, 'skill_established': False}
    gain = abs(actual-baseline)-abs(actual-predicted)
    result = {'n': len(actual), 'mae_gain_percentage_points': float(gain.mean()*100),
              'mae_gain_ci95': None, 'skill_established': False,
              'method': '600 paired circular moving-block resamples; exploratory 95% interval, not a multiple-testing-adjusted guarantee'}
    if len(actual) >= 30:
        rng = np.random.default_rng(1907)
        block = max(2, int(len(actual)**(1/3)))
        starts = rng.integers(0, len(actual), (600, int(np.ceil(len(actual)/block))))
        indices = ((starts[:, :, None]+np.arange(block)) % len(actual)).reshape(600, -1)[:, :len(actual)]
        ci = np.quantile(gain[indices].mean(axis=1)*100, [.025, .975])
        result.update(mae_gain_ci95=list(map(float, ci)), skill_established=bool(ci[0] > 0))
    return result


def interval_score(actual, lower, upper, alpha=.2):
    """Proper interval score: penalize both width and observations outside the band."""
    y, lo, hi = map(lambda x: np.asarray(x, float), (actual, lower, upper))
    if np.any(lo > hi):
        raise ValueError('Reversed interval')
    return (hi-lo)+(2/alpha)*np.maximum(lo-y, 0)+(2/alpha)*np.maximum(y-hi, 0)
