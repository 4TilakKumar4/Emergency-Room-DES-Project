"""
analysis_utils.py — Shared statistical utilities for the IE 7215 ED simulation.

Single source of truth for confidence interval calculation.  Previously ci95()
was defined independently in four files, three of which used the asymptotic
z = 1.96 approximation.  At the DOE replication budget of 30 runs the correct
t-quantile (t₀.₀₂₅,₂₉ = 2.045) is 4% wider, so the z version underreports
uncertainty for small-n experiments.

All simulation files import from here so any future change (e.g., switching
to BCa bootstrap intervals) is made in one place.
"""

import math
import pandas as pd
from scipy import stats


def ci(series: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    Return (mean, half_width) of the (1 - alpha) t-interval.

    Uses the t-distribution with (n-1) degrees of freedom — correct for any
    sample size.  At n = 100 replications the result is indistinguishable from
    the z = 1.96 approximation (t₀.₀₂₅,₉₉ = 1.984); at n = 30 (DOE budget)
    the difference is ~4%.

    Parameters
    ----------
    series : pd.Series
        Per-replication scalar observations (e.g., mean doctor wait per rep).
    alpha  : float
        Significance level.  Default 0.05 gives a 95 % CI.

    Returns
    -------
    (mean, half_width) both as floats.
    """
    n  = len(series)
    m  = float(series.mean())
    s  = float(series.std(ddof=1))
    t  = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    hw = t * s / math.sqrt(n)
    return m, hw


# Backward-compatible alias used by older call sites
ci95 = ci
