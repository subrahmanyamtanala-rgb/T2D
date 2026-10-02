"""Dataset loading and cleaning for the PIMA Indians Diabetes dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"
TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# Physiologically impossible zeros in these columns encode missing values.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path=DEFAULT_DATA_PATH):
    """Return (X, y) with impossible zeros replaced by NaN.

    Imputation is deliberately *not* done here: it happens inside the model
    pipeline so that it is fitted on training folds only (no data leakage).
    """
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    X = df[FEATURES].astype(float).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    y = df[TARGET].astype(int)
    return X, y
