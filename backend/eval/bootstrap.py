"""Paired bootstrap confidence intervals over questions (plan section 9.4).

Both systems are scored on the same questions; each resample draws questions with
replacement and takes the mean difference (system B minus system A). A difference is called
real only if the 95% interval excludes zero.
"""

import random
from collections.abc import Sequence


def paired_bootstrap(a: Sequence[float], b: Sequence[float], resamples: int = 1000, seed: int = 13,
                     alpha: float = 0.05) -> dict:
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs scores for the same questions")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "diff": None, "low": None, "high": None, "real": False}
    diffs = [y - x for x, y in pairs]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low = means[int(resamples * alpha / 2)]
    high = means[min(resamples - 1, int(resamples * (1 - alpha / 2)))]
    return {"n": n, "diff": observed, "low": low, "high": high, "real": low > 0 or high < 0}


def bootstrap_mean(values: Sequence[float], resamples: int = 1000, seed: int = 13, alpha: float = 0.05) -> dict:
    xs = [v for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    rng = random.Random(seed)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {"n": n, "mean": sum(xs) / n, "low": means[int(resamples * alpha / 2)],
            "high": means[min(resamples - 1, int(resamples * (1 - alpha / 2)))]}
