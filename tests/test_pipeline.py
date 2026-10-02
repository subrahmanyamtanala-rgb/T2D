import numpy as np
import pandas as pd
import pytest

from t2d.data import FEATURES, load_dataset
from t2d.evaluation import classification_metrics, compare_models, preprocessing_only, summarize
from t2d.features import ClinicalFeatureEngineer
from t2d.models import base_learners, build_pipeline, build_stacking, selected_features
from t2d.predict import predict


@pytest.fixture(scope="module")
def data():
    X, y = load_dataset()
    # Small stratified subset keeps the tests fast.
    idx = y.groupby(y).sample(n=90, random_state=0).index
    return X.loc[idx], y.loc[idx]


def fast_stack():
    return build_stacking(["lr", "knn", "dt"], cv_folds=3)


def fast_rfecv():
    return dict(estimator="lr", cv_folds=3, min_features=2)


def test_load_dataset_marks_impossible_zeros_missing():
    X, y = load_dataset()
    assert X.shape == (768, 8)
    assert set(y.unique()) == {0, 1}
    assert X["Glucose"].isna().sum() > 0
    assert (X[["Glucose", "BMI", "Insulin"]] == 0).sum().sum() == 0
    assert (X["Pregnancies"] == 0).any()  # zero is valid here


def test_feature_engineer_adds_named_columns():
    X, _ = load_dataset()
    fe = ClinicalFeatureEngineer().fit(X.fillna(X.median()))
    Z = fe.transform(X.fillna(X.median()))
    assert list(Z.columns) == list(fe.get_feature_names_out())
    assert Z.shape[1] == 16
    assert not Z.isna().any().any()
    off = ClinicalFeatureEngineer(enabled=False).fit(X)
    assert list(off.get_feature_names_out()) == FEATURES


def test_rfecv_stacking_pipeline_fits_and_selects_subset(data):
    X, y = data
    pipe = build_pipeline(fast_stack(), rfecv_kwargs=fast_rfecv()).fit(X, y)
    kept = selected_features(pipe)
    assert 2 <= len(kept) <= 16
    assert set(kept) <= set(pipe.named_steps["engineer"].get_feature_names_out())
    proba = pipe.predict_proba(X)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
    assert classification_metrics(y, pipe.predict(X), proba)["roc_auc"] > 0.7


def test_pipeline_handles_missing_values(data):
    X, y = data
    pipe = build_pipeline(fast_stack(), rfecv_kwargs=fast_rfecv()).fit(X, y)
    row = X.iloc[[0]].copy()
    row.loc[:, ["Insulin", "SkinThickness"]] = np.nan
    assert pipe.predict_proba(row).shape == (1, 2)


def test_compare_models_reports_both_configurations(data):
    X, y = data
    pipe = build_pipeline(fast_stack(), rfecv_kwargs=fast_rfecv())
    learners = base_learners()
    results, counts, n_folds = compare_models(
        preprocessing_only(pipe),
        pipe.named_steps["rfecv"],
        {"lr": learners["lr"], "nb": learners["nb"]},
        X,
        y,
        n_splits=3,
        log=lambda *_: None,
    )
    assert n_folds == 3
    assert len(results) == 2 * 2 * 3
    assert set(results["rfecv"]) == {True, False}
    assert max(counts.values()) <= n_folds
    summary = summarize(results)
    assert {"roc_auc_mean", "roc_auc_std"} <= set(summary.columns)


def test_predict_converts_zeros_and_thresholds(data):
    X, y = data
    pipe = build_pipeline(fast_stack(), rfecv_kwargs=fast_rfecv()).fit(X, y)
    patients = pd.DataFrame([[2, 150, 70, 0, 0, 33.0, 0.5, 45]], columns=FEATURES)
    out = predict(pipe, patients, threshold=0.0)
    assert out["t2dm_prediction"].iloc[0] == 1
    assert 0 <= out["t2dm_probability"].iloc[0] <= 1


def test_unknown_names_rejected():
    with pytest.raises(ValueError):
        build_stacking(["nope"])
    with pytest.raises(ValueError):
        build_pipeline(fast_stack(), rfecv_kwargs={"estimator": "nope"})
