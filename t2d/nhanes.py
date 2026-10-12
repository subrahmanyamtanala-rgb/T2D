"""Build the NHANES 2013-2014 external-check cohort.

Cohort: adults aged >= 21 years in the NHANES 2013-2014 OGTT subsample with valid
2-h OGTT glucose, fasting insulin and HbA1c. Outcome: dysglycaemia, i.e.
HbA1c >= 5.7 % (ADA prediabetes-or-diabetes threshold). HbA1c is an independent
laboratory measurement, so unlike the PIMA outcome it is not defined by the
2-h glucose predictor.

Two sources produce the same columns:

* ``from_cdc``    the official CDC SAS transport files (DEMO_H, BMX_H, BPX_H,
                  OGTT_H, INS_H, GHB_H) downloaded from https://wwwn.cdc.gov/nchs/nhanes/.
                  This is the authoritative route.
* ``from_mirror`` the processed copy in github.com/ChintanaL/diabetes-prediction-
                  using-nhanes-dataset, which the analysis environment could reach.
                  That copy contains values imputed by its author; only values at
                  NHANES' published measurement resolution (glucose: integer,
                  insulin: 2 decimals, HbA1c and BMI: 1 decimal, BP: integer) are
                  kept, everything else is set to missing and re-imputed inside the
                  validation folds.

    python -m t2d.nhanes --mirror <repo_dir>     # writes data/nhanes_2013_2014_ogtt.csv
    python -m t2d.nhanes --cdc <dir_with_xpt>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "nhanes_2013_2014_ogtt.csv"
COLUMNS = ["SEQN", "Sex", "Age", "BMI", "BloodPressure", "Glucose", "Insulin", "HbA1c", "Outcome"]
DYSGLYCAEMIA_HBA1C = 5.7
MIN_AGE = 21


def _at_resolution(s, decimals):
    """True where the value has at most `decimals` decimal places (i.e. not an imputed float)."""
    scaled = s * 10**decimals
    return np.isclose(scaled, np.round(scaled), rtol=0.0, atol=1e-6)


def _finish(df):
    df = df[(df.Age >= MIN_AGE) & df.Glucose.notna() & df.Insulin.notna() & df.HbA1c.notna()].copy()
    df["Outcome"] = (df.HbA1c >= DYSGLYCAEMIA_HBA1C).astype(int)
    return df[COLUMNS].sort_values("SEQN").reset_index(drop=True)


def from_mirror(repo):
    repo = Path(repo)
    lab = pd.read_csv(repo / "labs_new.csv")
    demo = pd.read_csv(repo / "demographic_new.csv")
    exam = pd.read_csv(repo / "examination_full.csv")
    lab = lab.assign(
        Glucose=lab.LBXGLT.where(_at_resolution(lab.LBXGLT, 0)),
        Insulin=lab.LBXIN.where(_at_resolution(lab.LBXIN, 2)),
        HbA1c=lab.LBXGH.where(_at_resolution(lab.LBXGH, 1)),
    )
    exam = exam.assign(
        BMI=exam.BMI.where(_at_resolution(exam.BMI, 1)),
        BloodPressure=exam["BP Dystolic"].where(_at_resolution(exam["BP Dystolic"], 0)),
    )
    demo = demo.assign(Sex=(demo.Gender == "Female").astype(float), Age=demo.AgeYears)
    df = lab[["SEQN", "Glucose", "Insulin", "HbA1c"]].merge(demo[["SEQN", "Sex", "Age"]], on="SEQN")
    df = df.merge(exam[["SEQN", "BMI", "BloodPressure"]], on="SEQN", how="left")
    return _finish(df)


def from_cdc(directory):
    d = Path(directory)
    rd = lambda name: pd.read_sas(d / f"{name}.XPT", format="xport")  # noqa: E731
    demo, bmx, bpx = rd("DEMO_H"), rd("BMX_H"), rd("BPX_H")
    ogtt, ins, ghb = rd("OGTT_H"), rd("INS_H"), rd("GHB_H")
    bp = bpx[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].replace(0, np.nan).mean(axis=1)
    df = (
        demo[["SEQN", "RIAGENDR", "RIDAGEYR"]]
        .merge(ogtt[["SEQN", "LBXGLT"]], on="SEQN")
        .merge(ins[["SEQN", "LBXIN"]], on="SEQN", how="left")
        .merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
        .merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
        .merge(pd.DataFrame({"SEQN": bpx.SEQN, "BloodPressure": bp}), on="SEQN", how="left")
    )
    df = df.rename(columns={"LBXGLT": "Glucose", "LBXIN": "Insulin", "LBXGH": "HbA1c", "BMXBMI": "BMI",
                            "RIDAGEYR": "Age"})
    df["Sex"] = (df.RIAGENDR == 2).astype(float)
    return _finish(df)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mirror", type=Path, help="clone of ChintanaL/diabetes-prediction-using-nhanes-dataset")
    src.add_argument("--cdc", type=Path, help="directory with DEMO_H, BMX_H, BPX_H, OGTT_H, INS_H, GHB_H .XPT files")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    df = from_mirror(args.mirror) if args.mirror else from_cdc(args.cdc)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} adults, {int(df.Outcome.sum())} with HbA1c >= {DYSGLYCAEMIA_HBA1C} % "
          f"({df.Outcome.mean():.1%}); missing: {df.isna().sum()[df.isna().sum() > 0].to_dict()} -> {args.out}")


if __name__ == "__main__":
    main()
