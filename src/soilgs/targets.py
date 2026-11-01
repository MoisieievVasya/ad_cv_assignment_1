"""Target encodings. Models predict an encoded vector; `decode` always returns a valid curve.

- direct:    cumulative % at the first 10 sieves.
- logit:     logit of cumulative fraction (saturates at 0 / 100 naturally).
- sqrt_mass: sqrt of the mass fraction in each of the 11 bins; decoded by squaring, renormalising
             to 100 % and cumulating, so monotonicity and the 100 % end point hold by construction.
             (Best encoding in the public notebook "cv-psgsdi-dinov2-features-sqrt-mass-ridge".)
"""
import numpy as np

from .metric import project_valid

EPS = 0.5  # % clipping for logit


def encode(y, mode):
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if mode == "direct":
        return y[:, :-1]
    if mode == "logit":
        p = np.clip(y[:, :-1], EPS, 100 - EPS) / 100
        return np.log(p / (1 - p))
    if mode == "sqrt_mass":
        mass = np.diff(np.column_stack([np.zeros(len(y)), y]), axis=1)
        return np.sqrt(np.clip(mass, 0, None) / 100)
    raise ValueError(mode)


def decode(z, mode):
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if mode == "direct":
        return project_valid(np.column_stack([z, np.full(len(z), 100.0)]))
    if mode == "logit":
        return project_valid(np.column_stack([100 / (1 + np.exp(-z)), np.full(len(z), 100.0)]))
    if mode == "sqrt_mass":
        mass = np.square(np.clip(z, 0, None)) + 1e-8
        return np.cumsum(100 * mass / mass.sum(axis=1, keepdims=True), axis=1)
    raise ValueError(mode)


def dim(mode):
    return 11 if mode == "sqrt_mass" else 10
