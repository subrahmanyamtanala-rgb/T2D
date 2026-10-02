"""Score new patients with a trained RFECV + stacking model.

Usage:
    python -m t2d.predict --model results/stacking_rfecv_model.joblib --input patients.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from t2d.data import FEATURES, ZERO_AS_MISSING


def predict(model, patients: pd.DataFrame, threshold=0.5):
    X = patients[FEATURES].astype(float).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    proba = model.predict_proba(X)[:, 1]
    out = patients.copy()
    out["t2dm_probability"] = proba.round(4)
    out["t2dm_prediction"] = (proba >= threshold).astype(int)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("results/stacking_rfecv_model.joblib"))
    p.add_argument("--input", type=Path, required=True, help="CSV with the 8 PIMA feature columns")
    p.add_argument("--output", type=Path, help="where to write predictions (default: stdout)")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    result = predict(joblib.load(args.model), pd.read_csv(args.input), args.threshold)
    if args.output:
        result.to_csv(args.output, index=False)
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
