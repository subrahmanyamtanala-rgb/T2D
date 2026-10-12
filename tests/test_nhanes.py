import numpy as np
import pandas as pd

from t2d.data import load_named
from t2d.nhanes import _at_resolution, from_cdc


def test_resolution_filter_drops_imputed_floats():
    s = pd.Series([115.0, 115.550255536627, 6.41, 12.4657667844523])
    assert _at_resolution(s, 0).tolist() == [True, False, False, False]
    assert _at_resolution(s, 2).tolist() == [True, False, True, False]


def test_nhanes_cohort_is_adult_and_hba1c_defined():
    df = pd.read_csv("data/nhanes_2013_2014_ogtt.csv")
    assert (df.Age >= 21).all()
    assert df.SEQN.is_unique
    assert ((df.HbA1c >= 5.7).astype(int) == df.Outcome).all()
    X, y = load_named("nhanes")
    assert list(X.columns) == ["Sex", "Age", "BMI", "BloodPressure", "Glucose", "Insulin"]
    assert not (X[["BMI", "BloodPressure", "Glucose", "Insulin"]] == 0).any().any()
    assert len(y) == 1815 and y.sum() == 551
    assert X.BloodPressure.dropna().between(20, 150).all()


def test_nhanes_cohort_rebuilds_from_official_files():
    rebuilt = from_cdc("data/nhanes_cdc")
    pd.testing.assert_frame_equal(rebuilt, pd.read_csv("data/nhanes_2013_2014_ogtt.csv"), check_dtype=False)
