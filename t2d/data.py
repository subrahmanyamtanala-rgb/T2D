"""Dataset loading and cleaning: PIMA Indians Diabetes (primary) and NHANES 2013-2014 (external check)."""

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


NHANES_PATH = Path(__file__).resolve().parent.parent / "data" / "nhanes_2013_2014_ogtt.csv"
DATASETS = {
    "pima": {"path": DEFAULT_DATA_PATH, "features": FEATURES, "zero_as_missing": ZERO_AS_MISSING},
    # Built by t2d.nhanes; outcome is HbA1c-defined dysglycaemia (>= 5.7 %).
    "nhanes": {
        "path": NHANES_PATH,
        "features": ["Sex", "Age", "BMI", "BloodPressure", "Glucose", "Insulin"],
        "zero_as_missing": ["BMI", "BloodPressure", "Glucose", "Insulin"],
    },
}


def load_named(name="pima"):
    """(X, y) for a registered dataset, with impossible zeros recoded as missing."""
    spec = DATASETS[name]
    df = pd.read_csv(spec["path"])
    X = df[spec["features"]].astype(float).copy()
    zeros = spec["zero_as_missing"]
    X[zeros] = X[zeros].replace(0, np.nan)
    return X, df[TARGET].astype(int)


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
