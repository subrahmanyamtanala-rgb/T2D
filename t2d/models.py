"""Base learners, the RFECV selector and the stacking ensemble pipeline."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import RFECV
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from t2d.features import ClinicalFeatureEngineer

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None


def base_learners(random_state=42):
    """Heterogeneous level-0 learners for the stack."""
    learners = {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        # Platt-scaled RBF SVM (replacement for the deprecated SVC(probability=True)).
        "svm": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=random_state),
            method="sigmoid",
            ensemble=False,
        ),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=random_state),
        "rf": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state
        ),
        "et": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state
        ),
        "gb": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state),
    }
    if XGBClassifier is not None:
        learners["xgb"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    if LGBMClassifier is not None:
        learners["lgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
            random_state=random_state,
        )
    return learners


def rfecv_estimator(name="rf", random_state=42):
    """Estimator used to rank features inside RFECV (must expose coef_ or feature_importances_)."""
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=random_state)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=random_state)
    if name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if name == "gb":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown RFECV estimator: {name!r} (choose rf, et, lr or gb)")


def build_rfecv(estimator="rf", cv_folds=5, scoring="roc_auc", min_features=3, random_state=42):
    return RFECV(
        estimator=rfecv_estimator(estimator, random_state),
        step=1,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=-1,
    )


def build_stacking(learner_names=None, cv_folds=5, random_state=42):
    """Stacking ensemble: level-0 out-of-fold probabilities -> logistic regression meta-learner."""
    learners = base_learners(random_state)
    if learner_names is None:
        learner_names = [n for n in ("lr", "svm", "knn", "rf", "et", "gb", "xgb", "lgbm") if n in learners]
    unknown = set(learner_names) - set(learners)
    if unknown:
        raise ValueError(f"Unknown base learners: {sorted(unknown)}")
    return StackingClassifier(
        estimators=[(n, learners[n]) for n in learner_names],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        stack_method="predict_proba",
        n_jobs=-1,
    )


IMPUTERS = ("median", "median_indicator", "iterative", "knn")


def build_imputer(strategy="median", random_state=42):
    """Missing-value handling compared in the missingness sensitivity analysis."""
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "median_indicator":
        # Adds a 0/1 "missingindicator_<feature>" column for every feature with missing values.
        return SimpleImputer(strategy="median", add_indicator=True)
    if strategy == "iterative":
        return IterativeImputer(max_iter=20, sample_posterior=False, random_state=random_state)
    if strategy == "knn":
        return KNNImputer(n_neighbors=10)
    raise ValueError(f"Unknown imputer: {strategy!r} (choose from {IMPUTERS})")


def build_pipeline(classifier, use_rfecv=True, engineer_features=True, rfecv_kwargs=None, imputer="median"):
    """Imputation -> feature engineering -> scaling -> [RFECV] -> classifier.

    Every step is fitted inside the pipeline, so cross-validation of the whole
    pipeline never lets test-fold information leak into imputation, scaling or
    feature selection.
    """
    steps = [
        ("impute", build_imputer(imputer)),
        ("engineer", ClinicalFeatureEngineer(enabled=engineer_features)),
        ("scale", StandardScaler()),
    ]
    if use_rfecv:
        steps.append(("rfecv", build_rfecv(**(rfecv_kwargs or {}))))
    steps.append(("clf", classifier))
    return Pipeline(steps).set_output(transform="pandas")


def selected_features(fitted_pipeline):
    """Names of the features reaching the classifier (those kept by RFECV, if present)."""
    return list(fitted_pipeline[:-1].get_feature_names_out())
