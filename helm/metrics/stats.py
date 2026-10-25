"""Small statistics helpers: bootstrap confidence intervals over episodes, mean +/- SE over seeds, Pearson r."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Nonparametric bootstrap of the mean: returns ``(mean, lo, hi)``."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mean_se(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
