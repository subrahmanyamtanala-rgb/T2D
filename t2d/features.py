"""Clinically motivated feature engineering, applied after imputation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# (name, required inputs, formula). A feature is added only when all its inputs are present, so
# the same transformer serves PIMA (8 inputs -> 16 candidates) and NHANES (6 inputs -> 13).
DERIVED = [
    ("Glucose_BMI", {"Glucose", "BMI"}, lambda X: X["Glucose"] * X["BMI"] / 100.0),
    ("Glucose_Age", {"Glucose", "Age"}, lambda X: X["Glucose"] * X["Age"] / 100.0),
    # HOMA-IR *surrogate*: PIMA has 2-h post-load values, not the fasting values HOMA-IR requires.
    ("HOMA_IR_surrogate", {"Glucose", "Insulin"}, lambda X: X["Glucose"] * X["Insulin"] / 405.0),
    ("Insulin_Glucose_Ratio", {"Glucose", "Insulin"}, lambda X: X["Insulin"] / X["Glucose"].clip(lower=1.0)),
    ("BMI_Age", {"BMI", "Age"}, lambda X: X["BMI"] * X["Age"] / 100.0),
    ("Obese", {"BMI"}, lambda X: (X["BMI"] >= 30).astype(float)),
    ("IGT_2h", {"Glucose"}, lambda X: (X["Glucose"] >= 140).astype(float)),
    ("Pedigree_Age", {"DiabetesPedigreeFunction", "Age"}, lambda X: X["DiabetesPedigreeFunction"] * X["Age"]),
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived T2DM risk features to the raw clinical measurements.

    The extra features give RFECV a richer (and partly redundant) candidate set
    to prune, which is where recursive elimination is most useful.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        if not self.enabled:
            return X
        has = set(X.columns)
        for name, needs, fn in DERIVED:
            if needs <= has:
                X[name] = fn(X)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.enabled:
            has = set(names)
            names += [name for name, needs, _ in DERIVED if needs <= has]
        return np.asarray(names, dtype=object)
