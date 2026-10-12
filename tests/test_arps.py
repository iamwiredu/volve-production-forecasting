import numpy as np

from volve import arps, metrics


def test_fit_recovers_known_parameters():
    t = np.arange(0, 60)
    true = {"qi": 1200.0, "di": 0.08, "b": 0.6}
    q = arps.predict(t, true)
    fitted = arps.fit(t, q)
    assert abs(fitted["qi"] - true["qi"]) / true["qi"] < 1e-3
    assert abs(fitted["di"] - true["di"]) / true["di"] < 1e-2
    assert abs(fitted["b"] - true["b"]) < 1e-2


def test_fit_with_noise_forecasts_holdout():
    rng = np.random.default_rng(0)
    t = np.arange(0, 72)
    true = {"qi": 900.0, "di": 0.05, "b": 0.4}
    q = arps.predict(t, true) * rng.normal(1.0, 0.03, t.size)
    train, test = slice(0, 54), slice(54, None)
    fitted = arps.fit(t[train], q[train])
    assert metrics.mape(q[test], arps.predict(t[test], fitted)) < 10


def test_small_b_matches_exponential():
    t = np.arange(0, 24)
    q = arps.hyperbolic(t, 500.0, 0.1, 0.0)
    assert np.allclose(q, 500.0 * np.exp(-0.1 * t), rtol=1e-4)
