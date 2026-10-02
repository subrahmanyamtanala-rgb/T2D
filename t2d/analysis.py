"""Extended validation for the manuscript revision.

Stages (each writes CSV/JSON under --output-dir, default results/revision):

    cv        5 x 10-fold repeated CV on the training set, all learners with and
              without the common RFECV protocol, plus logistic regression with its
              own (logistic-ranked) RFECV. Saves fold metrics and out-of-fold
              probabilities.
    missing   Missing-data sensitivity: median, median + missingness indicators,
              iterative and KNN imputation, for LR and RFECV + stacking.
    holdout   Final models on the 80 % training set, scored on the 20 % hold-out:
              bootstrap CIs, DeLong tests, calibration, threshold analysis (thresholds
              chosen on training out-of-fold predictions), decision curves, SHAP.

    python -m t2d.analysis --stage cv
    python -m t2d.analysis --stage missing
    python -m t2d.analysis --stage holdout     # needs the cv stage's predictions
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from t2d.data import load_dataset
from t2d.evaluation import classification_metrics, preprocessing_only
from t2d.models import IMPUTERS, base_learners, build_pipeline, build_stacking, rfecv_estimator
from t2d.stats import (
    bootstrap_ci,
    calibration_intercept_slope,
    corrected_paired_ttest,
    corrected_t_interval,
    delong_ci,
    delong_test,
    holm,
    net_benefit,
    threshold_for_sensitivity,
)

warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without")
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
REFERENCE = ("lr", False)  # comparator for paired tests: logistic regression, all 16 features
SENSITIVITY_TARGETS = (0.80, 0.85, 0.90, 0.95)


def split(seed=SEED):
    X, y = load_dataset()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def classifiers(seed=SEED):
    learners = base_learners(seed)
    out = {k: learners[k] for k in ("lr", "svm", "knn", "nb", "dt", "rf", "et", "gb", "xgb", "lgbm")}
    out["stacking"] = build_stacking(random_state=seed)
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- cv
def stage_cv(out, n_splits=10, n_repeats=5, seed=SEED):
    X, _, y, _ = split(seed)
    clfs = classifiers(seed)
    template = build_pipeline(clfs["stacking"], rfecv_kwargs={"random_state": seed})
    prep_t, selector_t = preprocessing_only(template), template.named_steps["rfecv"]
    lr_selector = RFECV(
        rfecv_estimator("lr", seed),
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="roc_auc",
        min_features_to_select=3,
    )
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    metric_rows, pred_rows, sel_rows = [], [], []
    for j, (tr, te) in enumerate(cv.split(X, y)):
        repeat, fold = j // n_splits + 1, j % n_splits + 1
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        prep = clone(prep_t)
        Z_tr, Z_te = prep.fit_transform(X_tr, y_tr), prep.transform(X_te)
        kept = list(clone(selector_t).fit(Z_tr, y_tr).get_feature_names_out())
        kept_lr = list(clone(lr_selector).fit(Z_tr, y_tr).get_feature_names_out())
        sel_rows += [{"repeat": repeat, "fold": fold, "selector": "rf", "feature": f} for f in kept]
        sel_rows += [{"repeat": repeat, "fold": fold, "selector": "lr", "feature": f} for f in kept_lr]
        configs = [(n, False, list(Z_tr.columns), c) for n, c in clfs.items()]
        configs += [(n, True, kept, c) for n, c in clfs.items()]
        configs.append(("lr_own_rfecv", True, kept_lr, clfs["lr"]))
        for name, use_rfecv, cols, clf in configs:
            model = clone(clf).fit(Z_tr[cols], y_tr)
            proba = model.predict_proba(Z_te[cols])[:, 1]
            m = classification_metrics(y_te, (proba >= 0.5).astype(int), proba)
            m["brier"] = brier_score_loss(y_te, proba)
            metric_rows.append(
                {"model": name, "rfecv": use_rfecv, "repeat": repeat, "fold": fold, "n_features": len(cols), **m}
            )
            pred_rows += [
                {"model": name, "rfecv": use_rfecv, "repeat": repeat, "index": int(i), "y": int(t), "proba": float(p)}
                for i, t, p in zip(X_te.index, y_te, proba)
            ]
        log(f"cv repeat {repeat} fold {fold:>2}: RFECV(rf) kept {len(kept)}, RFECV(lr) kept {len(kept_lr)}")
    pd.DataFrame(metric_rows).to_csv(out / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "cv_oof_predictions.csv.gz", index=False)
    pd.DataFrame(sel_rows).to_csv(out / "cv_selected_features.csv", index=False)
    summarize_cv(out, n_train_frac=(n_splits - 1) / n_splits)


def summarize_cv(out, n_train_frac=0.9):
    df = pd.read_csv(out / "cv_fold_metrics.csv")
    n_total = 614
    n_tr, n_te = n_total * n_train_frac, n_total * (1 - n_train_frac)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "brier"]
    rows = []
    ref = df[(df.model == REFERENCE[0]) & (df.rfecv == REFERENCE[1])].sort_values(["repeat", "fold"])
    for (name, use_rfecv), g in df.groupby(["model", "rfecv"]):
        g = g.sort_values(["repeat", "fold"])
        row = {"model": name, "rfecv": use_rfecv, "n_features": g.n_features.mean()}
        for m in metrics:
            mean, lo, hi = corrected_t_interval(g[m], n_tr, n_te)
            row.update({f"{m}_mean": mean, f"{m}_lo": lo, f"{m}_hi": hi})
        for m in ("roc_auc", "mcc"):
            d, t, p = corrected_paired_ttest(g[m].values, ref[m].values, n_tr, n_te)
            row.update({f"{m}_diff_vs_lr": d, f"{m}_p_vs_lr": p})
        rows.append(row)
    summary = pd.DataFrame(rows)
    others = ~((summary.model == REFERENCE[0]) & (summary.rfecv == REFERENCE[1]))
    for m in ("roc_auc", "mcc"):
        summary[f"{m}_p_holm"] = np.nan
        summary.loc[others, f"{m}_p_holm"] = holm(summary.loc[others, f"{m}_p_vs_lr"].values)
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(out / "cv_summary.csv", index=False)

    sel = pd.read_csv(out / "cv_selected_features.csv")
    n_folds = sel.groupby(["repeat", "fold"]).ngroups
    freq = sel.groupby(["selector", "feature"]).size().div(n_folds).rename("frequency").reset_index()
    freq.sort_values(["selector", "frequency"], ascending=[True, False]).to_csv(
        out / "cv_selection_frequency.csv", index=False
    )
    sizes = sel.groupby(["selector", "repeat", "fold"]).size().groupby("selector").describe()
    sizes.to_csv(out / "cv_selection_sizes.csv")
    log("cv summary written")
    return summary


# ---------------------------------------------------------------------- missing
def stage_missing(out, n_splits=10, n_repeats=2, seed=SEED):
    X, _, y, _ = split(seed)
    learners = base_learners(seed)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for imp in IMPUTERS:
        template = build_pipeline(build_stacking(random_state=seed), imputer=imp, rfecv_kwargs={"random_state": seed})
        prep_t, selector_t = preprocessing_only(template), template.named_steps["rfecv"]
        for j, (tr, te) in enumerate(cv.split(X, y)):
            X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            prep = clone(prep_t)
            Z_tr, Z_te = prep.fit_transform(X_tr, y_tr), prep.transform(X_te)
            kept = list(clone(selector_t).fit(Z_tr, y_tr).get_feature_names_out())
            for name, cols, clf in (
                ("lr", list(Z_tr.columns), learners["lr"]),
                ("stacking_rfecv", kept, template.named_steps["clf"]),
            ):
                proba = clone(clf).fit(Z_tr[cols], y_tr).predict_proba(Z_te[cols])[:, 1]
                m = classification_metrics(y_te, (proba >= 0.5).astype(int), proba)
                m["brier"] = brier_score_loss(y_te, proba)
                rows.append(
                    {
                        "imputer": imp,
                        "model": name,
                        "repeat": j // n_splits + 1,
                        "fold": j % n_splits + 1,
                        "n_features": len(cols),
                        "indicators_kept": sum(c.startswith("missingindicator") for c in cols),
                        **m,
                    }
                )
            log(f"missing {imp:<16} repeat {j // n_splits + 1} fold {j % n_splits + 1:>2}: kept {len(kept)}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "missing_fold_metrics.csv", index=False)
    n_tr, n_te = 614 * (n_splits - 1) / n_splits, 614 / n_splits
    summary = []
    for (imp, name), g in df.groupby(["imputer", "model"]):
        g = g.sort_values(["repeat", "fold"])
        base = df[(df.imputer == "median") & (df.model == name)].sort_values(["repeat", "fold"])
        row = {"imputer": imp, "model": name, "n_features": g.n_features.mean(), "indicators_kept": g.indicators_kept.mean()}
        for m in ("roc_auc", "mcc", "recall", "specificity", "brier"):
            mean, lo, hi = corrected_t_interval(g[m], n_tr, n_te)
            row.update({f"{m}_mean": mean, f"{m}_lo": lo, f"{m}_hi": hi})
        d, _, p = corrected_paired_ttest(g.roc_auc.values, base.roc_auc.values, n_tr, n_te)
        row.update({"roc_auc_diff_vs_median": d, "roc_auc_p_vs_median": p})
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "missing_summary.csv", index=False)
    log("missing summary written")


# ---------------------------------------------------------------------- holdout
def _metrics_from_proba(threshold=0.5):
    def fn(y, proba):
        m = classification_metrics(y, (proba >= threshold).astype(int), proba)
        m["brier"] = brier_score_loss(y, proba)
        return m

    return fn


def stage_holdout(out, seed=SEED, n_boot=2000):
    X_tr, X_te, y_tr, y_te = split(seed)
    learners = base_learners(seed)
    configs = {
        "stacking_rfecv": build_pipeline(build_stacking(random_state=seed), rfecv_kwargs={"random_state": seed}),
        "stacking": build_pipeline(build_stacking(random_state=seed), use_rfecv=False),
        "lr": build_pipeline(learners["lr"], use_rfecv=False),
        "lr_own_rfecv": build_pipeline(learners["lr"], rfecv_kwargs={"estimator": "lr", "random_state": seed}),
        "rf_rfecv": build_pipeline(learners["rf"], rfecv_kwargs={"random_state": seed}),
        "xgb_rfecv": build_pipeline(learners["xgb"], rfecv_kwargs={"random_state": seed}),
        "svm": build_pipeline(learners["svm"], use_rfecv=False),
    }
    probas, fitted = {}, {}
    for name, pipe in configs.items():
        fitted[name] = pipe.fit(X_tr, y_tr)
        probas[name] = pipe.predict_proba(X_te)[:, 1]
        log(f"holdout fitted {name}")
    joblib.dump(fitted["stacking_rfecv"], out / "stacking_rfecv_model.joblib")
    pd.DataFrame({"y": y_te.values, **probas}, index=X_te.index).to_csv(out / "holdout_predictions.csv")

    # Bootstrap CIs and DeLong.
    rows = []
    for name, p in probas.items():
        ci = bootstrap_ci(y_te.values, p, _metrics_from_proba(), n_boot=n_boot, seed=seed)
        a, alo, ahi = delong_ci(y_te.values, p)
        ci["roc_auc"] = (a, alo, ahi)
        cal_a, cal_b = calibration_intercept_slope(y_te.values, p)
        row = {"model": name, "ap": average_precision_score(y_te, p), "cal_intercept": cal_a, "cal_slope": cal_b}
        for k, (v, lo, hi) in ci.items():
            row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(row)
    table = pd.DataFrame(rows)
    tests = [(n, *delong_test(y_te.values, probas["stacking_rfecv"], probas[n])) for n in probas if n != "stacking_rfecv"]
    tests = pd.DataFrame(tests, columns=["comparator", "auc_diff", "z", "p"])
    tests["p_holm"] = holm(tests.p.values)
    table.to_csv(out / "holdout_metrics_ci.csv", index=False)
    tests.to_csv(out / "holdout_delong.csv", index=False)

    # Threshold analysis: thresholds fixed on training out-of-fold predictions.
    oof = pd.read_csv(out / "cv_oof_predictions.csv.gz")
    oof_map = {"stacking_rfecv": ("stacking", True), "lr": ("lr", False)}
    thr_rows = []
    for name, (m, r) in oof_map.items():
        g = oof[(oof.model == m) & (oof.rfecv == r)].groupby("index").agg(y=("y", "first"), proba=("proba", "mean"))
        for target in SENSITIVITY_TARGETS:
            t = threshold_for_sensitivity(g.y.values, g.proba.values, target)
            ci = bootstrap_ci(y_te.values, probas[name], _threshold_metrics(t), n_boot=n_boot, seed=seed)
            row = {"model": name, "target_sensitivity": target, "threshold": t}
            for k, (v, lo, hi) in ci.items():
                row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
            thr_rows.append(row)
    pd.DataFrame(thr_rows).to_csv(out / "holdout_thresholds.csv", index=False)

    # Calibration on pooled training out-of-fold predictions (larger n than the hold-out).
    cal_rows = []
    for name, (m, r) in {**oof_map, "rf_rfecv": ("rf", True), "svm": ("svm", False)}.items():
        g = oof[(oof.model == m) & (oof.rfecv == r)]
        a, b = calibration_intercept_slope(g.y.values, g.proba.values)
        cal_rows.append({"model": name, "source": "training_oof", "brier": brier_score_loss(g.y, g.proba),
                         "cal_intercept": a, "cal_slope": b})
    pd.DataFrame(cal_rows).to_csv(out / "calibration_oof.csv", index=False)

    # Decision curves.
    ts = np.round(np.arange(0.05, 0.81, 0.01), 2)
    dca = {"threshold": ts}
    for name in ("stacking_rfecv", "lr", "rf_rfecv"):
        nb, all_ = net_benefit(y_te.values, probas[name], ts)
        dca[name] = nb
    dca["treat_all"] = all_
    dca["treat_none"] = np.zeros_like(ts)
    pd.DataFrame(dca).to_csv(out / "holdout_decision_curve.csv", index=False)

    # SHAP on the raw clinical inputs of the full stacking + RFECV pipeline.
    stage_shap(out, fitted["stacking_rfecv"], X_tr, X_te, seed)
    log("holdout stage written")


def _threshold_metrics(t):
    def fn(y, proba):
        pred = (proba >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn_ = np.sum((pred == 0) & (y == 1))
        return {
            "sensitivity": tp / (tp + fn_),
            "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp) if tp + fp else 0.0,
            "npv": tn / (tn + fn_) if tn + fn_ else 0.0,
            "flagged": (tp + fp) / len(y),
        }

    return fn


def stage_shap(out, model, X_tr, X_te, seed=SEED, n_background=40):
    import shap

    background = X_tr.sample(n_background, random_state=seed)
    f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X_tr.columns))[:, 1]  # noqa: E731
    explainer = shap.ExactExplainer(f, shap.maskers.Independent(background, max_samples=n_background))
    sv = explainer(X_te)
    values = pd.DataFrame(sv.values, columns=X_tr.columns, index=X_te.index)
    values.to_csv(out / "shap_values_holdout.csv")
    X_te.to_csv(out / "shap_inputs_holdout.csv")
    summary = values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    summary.to_csv(out / "shap_summary.csv")
    log("shap written")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["cv", "summarize", "missing", "holdout"], required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/revision"))
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "cv":
        stage_cv(args.output_dir, n_repeats=args.repeats)
    elif args.stage == "summarize":
        summarize_cv(args.output_dir)
    elif args.stage == "missing":
        stage_missing(args.output_dir)
    else:
        stage_holdout(args.output_dir)


if __name__ == "__main__":
    main()
