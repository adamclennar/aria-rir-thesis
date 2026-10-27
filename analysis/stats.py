"""Shared statistics and metric definitions for the analyses beyond the reproduction.

Imported by the other analysis scripts, not run on its own.

Primary T20 measure (README, "Scoring"): the mean |relative T20 error| in the
500 Hz and 1 kHz octave bands, the bands of ISO 3382-1's mid-frequency
reverberation time (T_mid). Beside it, the band groups that carry other
perceptual attributes: low (125 + 250 Hz, warmth, as in Beranek's bass ratio)
and high (2 + 4 kHz, brilliance); 8 kHz on its own; and all seven bands, the
paper's measure. A conclusion drawn on T_mid holds for T_mid: where the groups
disagree, the analyses report it. The comparison with the paper (RQ1,
reproduction.py) uses all seven bands, as the paper does.

Every comparison is PAIRED: the same take (or room and voice) under two
conditions. paired() gives the median paired difference, a bootstrap 95% CI of
that median (10 000 resamples), the number of pairs that went down, and exact
two-sided sign and Wilcoxon signed-rank tests. Takes from one room share its
acoustics, so pairs are not fully independent: per-room values should be
shown beside any headline.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
BAND_SETS = {
    "T_mid": (500, 1000),                 # primary: ISO 3382-1 mid-frequency bands
    "low": (125, 250),                    # warmth (Beranek's bass ratio)
    "high": (2000, 4000),                 # brilliance
    "8k": (8000,),
    "all": BANDS,                         # the paper's measure
}
PRIMARY = "T_mid"
SEED = 1


def num(x) -> float:
    """float(x), with '', 'NaN' and None as nan."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def t20_err(row: dict, bands, col: str = "t20Err_{}") -> float:
    """Mean |relative T20 error| (%) over `bands`, from one result row whose
    per-band signed errors are in columns col.format(band)."""
    v = [abs(num(row.get(col.format(b)))) for b in bands]
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else math.nan


def paired(d, n_boot: int = 10_000) -> dict | None:
    """Paired differences d (condition A minus condition B, one per pair).

    Returns n, the median and its bootstrap 95% CI, `down` (pairs where A was
    lower, i.e. better for an error), and exact two-sided sign and Wilcoxon
    p-values; None if there are no finite differences. Zero differences are
    dropped from the tests, as both tests require.
    """
    d = np.asarray([x for x in d if x is not None and math.isfinite(x)], float)
    n = d.size
    if n == 0:
        return None
    rng = np.random.default_rng(SEED)
    boot = np.median(rng.choice(d, size=(n_boot, n), replace=True), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = d[d != 0]
    p_sign = sps.binomtest(int((nz > 0).sum()), nz.size, 0.5).pvalue if nz.size else 1.0
    p_wil = sps.wilcoxon(nz, alternative="two-sided", method="exact").pvalue if nz.size > 1 else 1.0
    return dict(n=n, median=float(np.median(d)), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                down=int((d < 0).sum()), p_sign=float(p_sign), p_wilcoxon=float(p_wil))


def fmt_paired(r: dict | None, unit: str) -> str:
    """One line: median [95% CI], pairs down / n, sign and Wilcoxon p (* for p < 0.05)."""
    if r is None:
        return "--"
    star = "*" if r["p_wilcoxon"] < 0.05 else " "
    return (f"{r['median']:+6.2f} {unit} [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  down {r['down']}/{r['n']}  "
            f"sign p={r['p_sign']:.3f}  Wilcoxon p={r['p_wilcoxon']:.3f} {star}")
