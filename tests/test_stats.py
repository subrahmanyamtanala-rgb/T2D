import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from t2d.stats import (
    calibration_intercept_slope,
    corrected_paired_ttest,
    corrected_t_interval,
    delong_ci,
    delong_test,
    holm,
    net_benefit,
    threshold_for_sensitivity,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    a = y + rng.normal(0, 0.8, 300)
    b = y + rng.normal(0, 1.2, 300)
    return y, a, b


def test_delong_auc_matches_sklearn(scores):
    y, a, _ = scores
    auc, lo, hi = delong_ci(y, a)
    assert auc == pytest.approx(roc_auc_score(y, a))
    assert lo < auc < hi


def test_delong_test_detects_better_score(scores):
    y, a, b = scores
    diff, _, p = delong_test(y, a, b)
    assert diff == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b))
    assert p < 0.05
    assert delong_test(y, a, a)[2] == 1.0


def test_corrected_interval_is_wider_than_naive():
    s = np.random.default_rng(1).normal(0.8, 0.03, 50)
    mean, lo, hi = corrected_t_interval(s, n_train=540, n_test=60)
    naive = 1.96 * s.std(ddof=1) / np.sqrt(50)
    assert mean == pytest.approx(s.mean())
    assert (hi - lo) / 2 > naive


def test_corrected_paired_ttest_zero_difference():
    s = np.linspace(0.7, 0.9, 20)
    d, t, p = corrected_paired_ttest(s, s, 540, 60)
    assert d == 0 and p == 1.0


def test_holm_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_calibration_of_true_probabilities():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.uniform(size=p.size) < p
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05


def test_net_benefit_treat_all_and_threshold():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.2, 0.1])
    nb, treat_all = net_benefit(y, p, [0.5])
    assert nb[0] == pytest.approx(2 / 5)
    assert treat_all[0] == pytest.approx(2 / 5 - 3 / 5)
    assert threshold_for_sensitivity(y, p, 1.0) == 0.6
    assert threshold_for_sensitivity(y, p, 0.5) == 0.9
