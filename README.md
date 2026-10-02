# Stacking Ensemble + RFECV for Type 2 Diabetes Mellitus Prediction

A reproducible machine-learning framework that predicts Type 2 Diabetes Mellitus (T2DM)
by combining **Recursive Feature Elimination with Cross-Validation (RFECV)** with a
**heterogeneous stacking ensemble**. It is evaluated on the PIMA Indians Diabetes dataset
(768 patients, 8 clinical measurements, 34.9 % positive).

## Framework

```
raw features (8)
   │  zeros in Glucose/BP/SkinThickness/Insulin/BMI → missing
   ▼
SimpleImputer (median)                     ┐
   ▼                                       │
ClinicalFeatureEngineer (+8 derived)       │  all steps live in one sklearn
   ▼                                       │  Pipeline, so they are re-fitted
StandardScaler                             │  on training folds only
   ▼                                       │  (no leakage into CV / test)
RFECV  (RandomForest ranker, 5-fold, AUC)  │
   ▼                                       │
StackingClassifier                         ┘
   level-0: LR · SVM (Platt) · KNN · RF · ExtraTrees · GradientBoosting · XGBoost · LightGBM
   level-1: Logistic Regression on 5-fold out-of-fold probabilities
```

* **Data cleaning** – physiologically impossible zeros are treated as missing values and
  imputed with the training-fold median.
* **Feature engineering** – 8 clinically motivated features are added (Glucose×BMI,
  Glucose×Age, HOMA-IR surrogate, insulin/glucose ratio, BMI×Age, obesity flag
  BMI ≥ 30, hyperglycaemia flag Glucose ≥ 140, pedigree×age), giving RFECV 16 candidates.
* **RFECV** – recursively drops the weakest feature (by importance) and keeps the subset
  with the best cross-validated ROC-AUC. Ranker, folds, metric and minimum subset size are
  configurable.
* **Stacking** – diverse base learners are trained on the selected features; their
  out-of-fold predicted probabilities feed a logistic-regression meta-learner.
* **Evaluation** – stratified 80/20 hold-out split. On the training part every model is
  compared *with* and *without* RFECV using stratified k-fold CV in which preprocessing
  and RFECV are re-fitted inside every fold. The final RFECV + stacking pipeline is then
  fitted on the full training set and scored once on the untouched test set.
  Metrics: accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1,
  MCC, ROC-AUC.

## Project layout

| Path | Purpose |
|---|---|
| `data/diabetes.csv` | PIMA Indians Diabetes dataset |
| `t2d/data.py` | loading and zero-as-missing cleaning |
| `t2d/features.py` | clinical feature engineering transformer |
| `t2d/models.py` | base learners, RFECV builder, stacking ensemble, full pipeline |
| `t2d/evaluation.py` | metrics, leakage-free CV comparison, summaries |
| `t2d/plots.py` | RFECV curve, feature ranking, ROC curves, confusion matrix |
| `t2d/train.py` | experiment CLI |
| `t2d/predict.py` | score new patients with a saved model |
| `results/` | metrics tables, figures and `report.json` from the reference run |
| `tests/` | pytest suite |

## Usage

```bash
pip install -r requirements.txt

# full experiment (10-fold CV comparison + hold-out test); writes to results/
python -m t2d.train

# faster: 5-fold CV, compare stacking only against LR and RF
python -m t2d.train --cv-folds 5 --quick

# variations
python -m t2d.train --rfecv-estimator lr --rfecv-scoring f1
python -m t2d.train --no-feature-engineering
python -m t2d.train --base-learners lr svm rf xgb

# predict for new patients (CSV with the 8 PIMA columns)
python -m t2d.predict --model results/stacking_rfecv_model.joblib --input patients.csv

# tests
pytest
```

Outputs in `results/`:

* `cv_summary.csv`, `cv_fold_results.csv` – CV metrics per model, with and without RFECV
* `rfecv_selection_frequency.csv` – how often each feature survived RFECV across CV folds
* `test_metrics.csv` – hold-out metrics for every model
* `report.json` – selected features, RFECV ranking, meta-learner weights, final metrics
* `figures/` – RFECV curve, feature ranking, ROC curves, confusion matrix
* `stacking_rfecv_model.joblib` – trained pipeline (not committed; regenerate with `train`)

RESULTS_PLACEHOLDER
