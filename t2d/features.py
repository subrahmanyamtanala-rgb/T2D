"""Clinically motivated feature engineering, applied after imputation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived T2DM risk features to the raw PIMA measurements.

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
        X["Glucose_BMI"] = X["Glucose"] * X["BMI"] / 100.0
        X["Glucose_Age"] = X["Glucose"] * X["Age"] / 100.0
        # HOMA-IR surrogate (glucose mg/dL * insulin uU/mL / 405).
        X["HOMA_IR"] = X["Glucose"] * X["Insulin"] / 405.0
        X["Insulin_Glucose_Ratio"] = X["Insulin"] / X["Glucose"].clip(lower=1.0)
        X["BMI_Age"] = X["BMI"] * X["Age"] / 100.0
        X["Obese"] = (X["BMI"] >= 30).astype(float)
        X["Hyperglycemic"] = (X["Glucose"] >= 140).astype(float)
        X["Pedigree_Age"] = X["DiabetesPedigreeFunction"] * X["Age"]
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.enabled:
            names += [
                "Glucose_BMI",
                "Glucose_Age",
                "HOMA_IR",
                "Insulin_Glucose_Ratio",
                "BMI_Age",
                "Obese",
                "Hyperglycemic",
                "Pedigree_Age",
            ]
        return np.asarray(names, dtype=object)
