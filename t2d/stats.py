"""Statistical tools for the revision: corrected CV inference, DeLong, bootstrap, calibration, DCA."""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def corrected_t_interval(scores, n_train, n_test, alpha=0.05):
    """Mean and CI of k-fold x r-repeat scores with the Nadeau-Bengio variance correction.

    Fold scores from overlapping training sets are positively correlated, so the
    naive standard error is too small. The corrected variance is
    (1/J + n_test/n_train) * s^2 with J = number of fold scores.
    """
    scores = np.asarray(scores, dtype=float)
    J = len(scores)
    mean = scores.mean()
    se = np.sqrt((1 / J + n_test / n_train) * scores.var(ddof=1))
    h = stats.t.ppf(1 - alpha / 2, J - 1) * se
    return mean, mean - h, mean + h


def corrected_paired_ttest(a, b, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) on paired fold scores a - b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    J = len(d)
    var = (1 / J + n_test / n_train) * d.var(ddof=1)
    if var == 0:
        return d.mean(), 0.0, 1.0
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return d.mean(), t, p


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y, scores):
    """Fast DeLong (Sun & Xu, 2014): AUCs and their covariance for k score vectors."""
    y = np.asarray(y)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y, score, alpha=0.05):
    aucs, cov = _delong_components(y, np.asarray(score, dtype=float)[None, :])
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return aucs[0], max(aucs[0] - z * se, 0.0), min(aucs[0] + z * se, 1.0)


def delong_test(y, score_a, score_b):
    """Two-sided DeLong test for two correlated ROC-AUCs on the same patients."""
    aucs, cov = _delong_components(y, np.vstack([score_a, score_b]).astype(float))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return diff, 0.0, 1.0
    z = diff / np.sqrt(var)
    return diff, z, 2 * stats.norm.sf(abs(z))


def bootstrap_ci(y, proba, metric_fn, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of every metric returned by metric_fn(y, proba) -> dict."""
    rng = np.random.default_rng(seed)
    y, proba = np.asarray(y), np.asarray(proba)
    point = metric_fn(y, proba)
    draws = {k: [] for k in point}
    n = len(y)
    while len(next(iter(draws.values()))) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue  # a resample needs both classes
        for k, v in metric_fn(y[idx], proba[idx]).items():
            draws[k].append(v)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (point[k], np.percentile(draws[k], lo), np.percentile(draws[k], hi)) for k in point}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, proba):
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope."""
    y = np.asarray(y, dtype=float)
    lp = _logit(proba)
    # Slope: logistic regression of y on logit(p) (Newton-Raphson, no penalty).
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    slope = beta[1]
    # Intercept: logistic regression with logit(p) as fixed offset.
    a = 0.0
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(a + lp)))
        step = (y - mu).sum() / (mu * (1 - mu)).sum()
        a += step
        if abs(step) < 1e-10:
            break
    return a, slope


def net_benefit(y, proba, thresholds):
    """Decision-curve net benefit of 'treat if p >= t' versus treat-all, per threshold."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    n = len(y)
    prev = y.mean()
    model, treat_all = [], []
    for t in thresholds:
        pred = proba >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        w = t / (1 - t)
        model.append(tp / n - fp / n * w)
        treat_all.append(prev - (1 - prev) * w)
    return np.array(model), np.array(treat_all)


def threshold_for_sensitivity(y, proba, target):
    """Largest threshold whose sensitivity on (y, proba) is at least `target`."""
    y, proba = np.asarray(y), np.asarray(proba)
    pos = np.sort(proba[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    return pos[k - 1]


def auc(y, proba):
    return roc_auc_score(y, proba)
