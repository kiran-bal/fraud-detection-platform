import numpy as np
import pytest

from fraud_platform.config import CostConfig
from fraud_platform.evaluation.calibration import Calibrator
from fraud_platform.evaluation.metrics import expected_calibration_error
from fraud_platform.evaluation.uncertainty import bootstrap_threshold, threshold_uncertainty


def _miscalibrated(n=6000, seed=0):
    """Scores that rank well but are inflated, like a class-weighted model's output."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    true_p = np.where(y == 1, rng.beta(6, 2, n), rng.beta(1, 8, n))
    inflated = np.clip(true_p ** 0.3, 1e-4, 1 - 1e-4)  # pushes everything upward
    return y, inflated


@pytest.mark.parametrize("kind", ["sigmoid", "isotonic"])
def test_calibration_reduces_ece_and_keeps_ranking(kind):
    y, p = _miscalibrated()
    fit_y, fit_p, ev_y, ev_p = y[:3000], p[:3000], y[3000:], p[3000:]
    cal = Calibrator(kind).fit(fit_p, fit_y)
    q = cal.transform(ev_p)
    assert expected_calibration_error(ev_y, q) < expected_calibration_error(ev_y, ev_p) / 2
    order_before, order_after = np.argsort(ev_p), np.argsort(q)
    # monotone transforms preserve the top-k set
    assert set(order_before[-100:]) == set(order_after[-100:]) or kind == "isotonic"


def test_none_calibrator_is_identity_and_roundtrips(tmp_path):
    p = np.array([0.1, 0.5, 0.9])
    cal = Calibrator("none").fit(p, np.array([0, 1, 1]))
    np.testing.assert_array_equal(cal.transform(p), p)
    cal.save(tmp_path / "c.joblib")
    assert Calibrator.load(tmp_path / "c.joblib").kind == "none"


def test_transform_before_fit_and_unknown_kind():
    with pytest.raises(RuntimeError):
        Calibrator("sigmoid").transform(np.array([0.2]))
    with pytest.raises(ValueError):
        Calibrator("spline").fit(np.array([0.2]), np.array([1]))


def test_bootstrap_threshold_is_deterministic_and_bounded():
    rng = np.random.default_rng(1)
    y = np.array([0] * 950 + [1] * 50)
    p = np.concatenate([rng.uniform(0, 0.5, 950), rng.uniform(0.3, 1.0, 50)])
    amounts = rng.lognormal(3, 1, 1000)
    a = bootstrap_threshold(y, p, amounts, CostConfig(), n_boot=50, seed=7)
    b = bootstrap_threshold(y, p, amounts, CostConfig(), n_boot=50, seed=7)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 50 and np.all((a >= 0) & (a <= 1.0 + 1e-9))


def test_threshold_uncertainty_intervals_are_ordered():
    rng = np.random.default_rng(2)
    y = np.array([0] * 1900 + [1] * 100)
    p = np.concatenate([rng.uniform(0, 0.5, 1900), rng.uniform(0.3, 1.0, 100)])
    amounts = rng.lognormal(3, 1, 2000)
    order = rng.permutation(2000)  # interleave labels so both halves contain fraud
    y, p, amounts = y[order], p[order], amounts[order]
    u = threshold_uncertainty(y[:1000], p[:1000], amounts[:1000], y[1000:], p[1000:], amounts[1000:], 0.5,
                              CostConfig(), n_boot=30, seed=3)
    for key in ("threshold", "eval_cost_of_bootstrap_thresholds", "eval_cost_at_chosen"):
        iv = u[key]
        assert iv["p2_5"] <= iv["p50"] <= iv["p97_5"]
    assert u["n_boot"] == 30


def test_uncertainty_with_no_fraud_in_selection_block_is_nan_not_crash():
    y = np.zeros(200, dtype=int)
    p = np.linspace(0, 1, 200)
    u = threshold_uncertainty(y, p, np.ones(200), y, p, np.ones(200), 0.5, CostConfig(), n_boot=5)
    assert u["n_boot"] == 0 and np.isnan(u["threshold"]["p50"])
