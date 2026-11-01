"""Extrapolation guard (H6c) for samples coarser than anything in training.

A one-feature model maps the texture-scale centroid (mm-indexed DoG energy centroid, averaged over a
sample's 50 mm tiles) to the curve's *quantiles in log-diameter* with a linear ridge, so it extrapolates
linearly instead of shrinking towards the training mean. It is blended into a main prediction with weight
w = clip((c - c_max_train) / SPAN, 0, 1): inside the training range nothing changes.
Idea adapted from the public notebook "soil-psd-physics-not-deep-learning" (its "extrapolation anchor").
"""
import numpy as np
from sklearn.linear_model import Ridge

from .metric import COLS, LOG_D_MAX, LOG_D_MIN, LOG_SIEVES, project_valid

LEVELS = np.array([2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98], dtype=float)
SPAN = 0.5


def curve_to_quantiles(curve):
    y = np.maximum.accumulate(np.concatenate([[0.0], curve])) + np.arange(len(curve) + 1) * 1e-6
    return np.interp(LEVELS, y, np.concatenate([[LOG_D_MIN], LOG_SIEVES]))


def quantiles_to_curve(q):
    x = np.maximum.accumulate(np.concatenate([[LOG_D_MIN], np.sort(q), [LOG_D_MAX]])) + np.arange(len(q) + 2) * 1e-6
    return np.interp(LOG_SIEVES, x, np.concatenate([[0.0], LEVELS, [100.0]]))


class Guard:
    """two_sided=False: only samples coarser than training (H6c); True: also finer than training (H6d)."""

    def __init__(self, two_sided=False):
        self.two_sided = two_sided

    def fit(self, centroid, curves):
        self.c_max, self.c_min = float(np.max(centroid)), float(np.min(centroid))
        Q = np.array([curve_to_quantiles(c) for c in curves])
        self.model = Ridge(alpha=0.1).fit(np.asarray(centroid).reshape(-1, 1), Q)
        return self

    def weight(self, centroid):
        c = np.asarray(centroid, dtype=float)
        w = np.clip((c - self.c_max) / SPAN, 0, 1)
        return np.maximum(w, np.clip((self.c_min - c) / SPAN, 0, 1)) if self.two_sided else w

    def apply(self, centroid, main_curves):
        """Blend; returns (curves, weights)."""
        centroid, main_curves = np.atleast_1d(centroid), np.atleast_2d(main_curves)
        anchor = np.array([quantiles_to_curve(q) for q in self.model.predict(centroid.reshape(-1, 1))])
        w = self.weight(centroid)[:, None]
        return project_valid((1 - w) * main_curves + w * anchor), w.ravel()


