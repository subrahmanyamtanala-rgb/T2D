"""Metrics and leakage-free cross-validated model comparison."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


def classification_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _fit_score(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    return classification_metrics(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1])


def compare_models(preprocess, selector, classifiers, X, y, n_splits=10, n_repeats=1, random_state=42, log=print):
    """Cross-validate every classifier with and without RFECV.

    For each outer fold the preprocessing and the RFECV selector are fitted on
    the training part only, then the selected subset is shared by all
    classifiers in that fold. This is a *common* feature-selection protocol: the
    subset is optimised with the selector's own ranking estimator (a random
    forest by default), not separately for each classifier. It is not
    equivalent to cross-validating a learner-specific RFECV pipeline per
    classifier, but it runs RFECV once per fold instead of once per
    (fold, classifier).

    Returns (per-fold results DataFrame, Counter of how often each feature was
    selected, number of folds).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows, selection_counts = [], Counter()
    n_folds = n_splits * n_repeats
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        prep = clone(preprocess)
        Z_tr, Z_te = prep.fit_transform(X_tr, y_tr), prep.transform(X_te)

        rfecv = clone(selector).fit(Z_tr, y_tr)
        kept = list(rfecv.get_feature_names_out())
        selection_counts.update(kept)
        S_tr, S_te = Z_tr[kept], Z_te[kept]
        log(f"  fold {fold:>2}/{n_folds}: RFECV kept {len(kept)}/{Z_tr.shape[1]} features")

        for name, clf in classifiers.items():
            for use_rfecv, (A_tr, A_te) in ((False, (Z_tr, Z_te)), (True, (S_tr, S_te))):
                scores = _fit_score(clone(clf), A_tr, y_tr, A_te, y_te)
                rows.append({"model": name, "rfecv": use_rfecv, "fold": fold, "n_features": A_tr.shape[1], **scores})
    return pd.DataFrame(rows), selection_counts, n_folds


def summarize(results):
    """Mean ± std of every metric per (model, rfecv) configuration, best ROC-AUC first."""
    metrics = [c for c in results.columns if c not in ("model", "rfecv", "fold")]
    agg = results.groupby(["model", "rfecv"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.sort_values("roc_auc_mean", ascending=False).reset_index()


def format_summary(summary, metrics=("accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc")):
    out = summary[["model", "rfecv"]].copy()
    out["n_features"] = summary["n_features_mean"].round(1)
    for m in metrics:
        out[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"].fillna(0))]
    return out


def preprocessing_only(pipeline: Pipeline):
    """The fitted-or-unfitted steps of a pipeline that come before RFECV."""
    names = [n for n, _ in pipeline.steps]
    stop = names.index("rfecv") if "rfecv" in names else len(names) - 1
    return Pipeline(pipeline.steps[:stop]).set_output(transform="pandas")


def selection_frequency(counts, n_folds):
    freq = pd.Series({k: v / n_folds for k, v in counts.items()}, name="selection_frequency")
    return freq.sort_values(ascending=False).rename_axis("feature").reset_index()


def stack_meta_weights(stacking):
    """Meta-learner coefficients per base learner (positive-class probability column)."""
    coef = np.ravel(stacking.final_estimator_.coef_)
    names = [n for n, _ in stacking.estimators]
    # With stack_method="predict_proba" on a binary task sklearn keeps one column per learner.
    return pd.Series(coef[: len(names)], index=names, name="meta_coefficient").sort_values(ascending=False)
