"""Correlation / error metrics used for both spot-level and gene-level evaluation."""

import numpy as np
from scipy import stats as scipy_stats


def safe_pearson(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    r, _ = scipy_stats.spearmanr(a, b)
    return float(r)


def safe_rmse(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))
