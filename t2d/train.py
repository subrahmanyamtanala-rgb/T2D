"""Train and evaluate the RFECV + stacking ensemble for T2DM prediction.

Usage:
    python -m t2d.train                      # full experiment, outputs in results/
    python -m t2d.train --cv-folds 5 --quick # faster run
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from t2d.data import DEFAULT_DATA_PATH, load_dataset
from t2d.evaluation import (
    classification_metrics,
    compare_models,
    format_summary,
    preprocessing_only,
    selection_frequency,
    stack_meta_weights,
    summarize,
)
from t2d.models import base_learners, build_pipeline, build_stacking, selected_features
from t2d.plots import plot_confusion, plot_feature_ranking, plot_rfecv_curve, plot_roc_curves

# StackingClassifier fits its meta-learner on a bare array but predicts on a
# DataFrame when pandas output is enabled; the resulting warning is harmless.
warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV in PIMA format")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=10, help="outer CV folds for model comparison")
    p.add_argument("--cv-repeats", type=int, default=1)
    p.add_argument("--rfecv-estimator", choices=["rf", "et", "lr", "gb"], default="rf")
    p.add_argument("--rfecv-folds", type=int, default=5)
    p.add_argument("--rfecv-scoring", default="roc_auc")
    p.add_argument("--min-features", type=int, default=3)
    p.add_argument("--stack-folds", type=int, default=5, help="internal CV used to build meta-features")
    p.add_argument(
        "--base-learners",
        nargs="+",
        default=None,
        help="level-0 learners (default: lr svm knn rf et gb xgb lgbm)",
    )
    p.add_argument("--no-feature-engineering", action="store_true", help="use only the 8 raw PIMA features")
    p.add_argument("--skip-cv", action="store_true", help="skip the cross-validated model comparison")
    p.add_argument("--quick", action="store_true", help="only compare stacking vs. LR/RF in CV")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.output_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Data: {len(X)} samples, {X.shape[1]} raw features, prevalence {y.mean():.3f}")
    print(f"Split: {len(X_train)} train / {len(X_test)} test (stratified)")

    rfecv_kwargs = dict(
        estimator=args.rfecv_estimator,
        cv_folds=args.rfecv_folds,
        scoring=args.rfecv_scoring,
        min_features=args.min_features,
        random_state=args.seed,
    )
    stacking = build_stacking(args.base_learners, cv_folds=args.stack_folds, random_state=args.seed)
    pipeline = build_pipeline(
        stacking, use_rfecv=True, engineer_features=not args.no_feature_engineering, rfecv_kwargs=rfecv_kwargs
    )

    learners = base_learners(args.seed)
    classifiers = {name: learners[name] for name, _ in stacking.estimators}
    if "nb" not in classifiers:
        classifiers["nb"] = learners["nb"]
    if "dt" not in classifiers:
        classifiers["dt"] = learners["dt"]
    classifiers["stacking"] = stacking

    # 1) Leakage-free cross-validated comparison on the training set.
    if not args.skip_cv:
        cv_classifiers = classifiers
        if args.quick:
            cv_classifiers = {k: v for k, v in classifiers.items() if k in ("lr", "rf", "stacking")}
        print(f"\n[1/2] {args.cv_folds}-fold x {args.cv_repeats} CV comparison (RFECV refitted in every fold)")
        results, counts, n_folds = compare_models(
            preprocessing_only(pipeline),
            pipeline.named_steps["rfecv"],
            cv_classifiers,
            X_train,
            y_train,
            n_splits=args.cv_folds,
            n_repeats=args.cv_repeats,
            random_state=args.seed,
        )
        summary = summarize(results)
        results.to_csv(out / "cv_fold_results.csv", index=False)
        summary.to_csv(out / "cv_summary.csv", index=False)
        selection_frequency(counts, n_folds).to_csv(out / "rfecv_selection_frequency.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(format_summary(summary).to_string(index=False))

    # 2) Fit the final RFECV + stacking pipeline on the whole training set; evaluate on hold-out.
    print("\n[2/2] Fitting final pipeline on the training set and scoring the hold-out test set")
    pipeline.fit(X_train, y_train)
    rfecv = pipeline.named_steps["rfecv"]
    kept = selected_features(pipeline)
    print(f"RFECV selected {len(kept)} features: {kept}")

    prep = pipeline[: pipeline.steps.index(("rfecv", rfecv))]
    Z_tr, Z_te = prep.transform(X_train), prep.transform(X_test)
    test_rows, curves = [], {}
    for name, clf in classifiers.items():
        for use_rfecv in (False, True):
            if name == "stacking" and use_rfecv:
                model, A_te = pipeline, X_test
            else:
                cols = kept if use_rfecv else list(Z_tr.columns)
                model = clone(clf).fit(Z_tr[cols], y_train)
                A_te = Z_te[cols]
            proba = model.predict_proba(A_te)[:, 1]
            metrics = classification_metrics(y_test, model.predict(A_te), proba)
            label = f"{name}{' + RFECV' if use_rfecv else ''}"
            test_rows.append({"model": name, "rfecv": use_rfecv, **metrics})
            if use_rfecv or name == "stacking":
                curves[label] = (y_test, proba, metrics["roc_auc"])
    test_table = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)
    test_table.to_csv(out / "test_metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(test_table.to_string(index=False))

    final_pred = pipeline.predict(X_test)
    final_metrics = classification_metrics(y_test, final_pred, pipeline.predict_proba(X_test)[:, 1])
    ranking = pd.Series(rfecv.ranking_, index=prep.get_feature_names_out(), name="rank")
    meta = stack_meta_weights(pipeline.named_steps["clf"])

    figs = out / "figures"
    plot_rfecv_curve(rfecv, figs / "rfecv_curve.png")
    plot_feature_ranking(ranking, figs / "rfecv_feature_ranking.png")
    plot_roc_curves(curves, figs / "roc_curves.png")
    plot_confusion(y_test, final_pred, figs / "confusion_matrix_stacking_rfecv.png", "Stacking + RFECV")

    joblib.dump(pipeline, out / "stacking_rfecv_model.joblib")
    report = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "rfecv_estimator": args.rfecv_estimator,
        "rfecv_scoring": args.rfecv_scoring,
        "candidate_features": list(ranking.index),
        "selected_features": kept,
        "feature_ranking": ranking.to_dict(),
        "base_learners": [n for n, _ in pipeline.named_steps["clf"].estimators],
        "meta_learner_coefficients": meta.to_dict(),
        "test_metrics": final_metrics,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print("\nStacking + RFECV hold-out performance:")
    for k, v in final_metrics.items():
        print(f"  {k:>18}: {v:.4f}")
    print(f"\nArtifacts written to {out.resolve()}")
    return report


if __name__ == "__main__":
    main()
