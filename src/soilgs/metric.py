"""Competition metric (log-weighted EMD between cumulative curves) and curve utilities.

Curves are cumulative % passing at the 11 sieves in `SIEVES` (last one is always 100).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

SIEVES = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
LOG_SIEVES = np.log10(SIEVES)
WEIGHTS = np.diff(LOG_SIEVES)  # one weight per interval, ~0.5 each
# Extension used when a curve has to be evaluated outside the sieve range (shifts, quantiles).
LOG_D_MIN, LOG_D_MAX = np.log10(0.0005), np.log10(200.0)


def emd_per_sample(y_true, y_pred):
    """Left-point rule: sum_i |F_true(d_i) - F_pred(d_i)| * (log10 d_{i+1} - log10 d_i)."""
    y_true, y_pred = np.atleast_2d(y_true), np.atleast_2d(y_pred)
    return (np.abs(y_true - y_pred)[:, :-1] * WEIGHTS).sum(axis=1)


def emd(y_true, y_pred):
    return float(emd_per_sample(y_true, y_pred).mean())


def emd_trapezoid_per_sample(y_true, y_pred):
    """Trapezoid variant of the same integral (used by one public notebook) — for comparison."""
    err = np.abs(np.atleast_2d(y_true) - np.atleast_2d(y_pred))
    return ((err[:, :-1] + err[:, 1:]) / 2 * WEIGHTS).sum(axis=1)


def emd_per_interval(y_true, y_pred):
    """(n_samples, 10) matrix whose row sums are `emd_per_sample`."""
    return np.abs(np.atleast_2d(y_true) - np.atleast_2d(y_pred))[:, :-1] * WEIGHTS


def project_valid(curves):
    """Closest (L2) valid curve: within [0, 100], non-decreasing, 100 at 200 mm."""
    curves = np.clip(np.atleast_2d(np.asarray(curves, dtype=float)), 0, 100)
    iso = IsotonicRegression(y_min=0, y_max=100)
    out = np.stack([iso.fit_transform(np.arange(len(c)), c) for c in curves])
    out[:, -1] = 100.0
    return out


def _extended(curve):
    """Curve on log10(d) with anchors 0 % at 0.0005 mm and 100 % at 200 mm."""
    x = np.concatenate([[LOG_D_MIN], LOG_SIEVES])
    y = np.concatenate([[0.0], np.maximum.accumulate(curve)])
    return x, y


def curve_at(curve, d_mm):
    """Evaluate a cumulative curve at arbitrary diameters (linear in log10 d)."""
    x, y = _extended(curve)
    return np.interp(np.log10(d_mm), x, y, left=0.0, right=100.0)


def shift_curve(curve, factor):
    """Curve that a size estimate wrong by `factor` would produce (factor 2 = grains read 2x larger).

    F'(d) = F(d / factor): every grain appears `factor` times bigger, the curve moves right.
    """
    return curve_at(curve, SIEVES / factor)


def d_percentile(curve, pct):
    """Diameter (mm) at which `pct` % passes, e.g. D10, D50, D60. Interpolated in log10 d."""
    x, y = _extended(curve)
    y = y + np.arange(len(y)) * 1e-9  # strictly increasing for interp
    return float(10 ** np.interp(pct, y, x))


def d_errors(y_true, y_pred, pcts=(10, 50, 60)):
    """Absolute error in log10(mm) of D-percentiles (complementary metric). Shape (n, len(pcts))."""
    y_true, y_pred = np.atleast_2d(y_true), np.atleast_2d(y_pred)
    return np.array([[abs(np.log10(d_percentile(t, p)) - np.log10(d_percentile(q, p))) for p in pcts]
                     for t, q in zip(y_true, y_pred)])
