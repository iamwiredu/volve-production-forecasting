"""Arps decline-curve analysis: the physics-based baseline."""
import numpy as np
from scipy.optimize import curve_fit


def hyperbolic(t, qi, di, b):
    """Arps hyperbolic rate. b -> 0 is exponential, b = 1 is harmonic."""
    b = np.maximum(b, 1e-6)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def fit(t, q):
    """Fit qi, Di (1/time unit of t) and b to rate data.

    b is bounded to [0, 2]; values above 1 are common in transient flow but
    should be treated with care for long-term forecasts.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    p0 = (q[0], 0.05, 0.5)
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, 2.0])
    params, _ = curve_fit(hyperbolic, t, q, p0=p0, bounds=bounds, maxfev=20000)
    return dict(zip(("qi", "di", "b"), params))


def predict(t, params):
    return hyperbolic(np.asarray(t, dtype=float), params["qi"], params["di"], params["b"])
