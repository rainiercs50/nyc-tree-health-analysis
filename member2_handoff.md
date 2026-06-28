# Member 2 Handoff: Modeling and Evaluation

**Project:** NYC Street Tree Health Predictor
**Course:** Intro to Data Science Final Project
**Member 2 role:** Modeling and Evaluation Lead
**Date:** 2026-06-28

## 1. What Member 2 completed

Member 2 built the modeling layer on top of Member 1's model-ready dataset: a preprocessing
pipeline, a baseline classifier, a tuned improved classifier, an honest evaluation for the
imbalanced target, feature-importance explainability, saved model artifacts for the app, and
three Streamlit pages (Model Prediction, Feature Importance, Hyperparameter Tuning).

## 2. Data and feature contract

- **Input:** `data/nyc_tree_member1_model_ready_sample.csv` (49,994 rows, no missing values).
- **Target:** `health` — Good / Fair / Poor (imbalanced: ~75.6% / 18.8% / 5.6%).
- **Split:** stratified 80/20 → 39,995 train / 9,999 test, `random_state=42`.
- **Numeric features:** `tree_dbh`, `problem_count` (StandardScaler).
- **Categorical features:** `species_top15_or_other`, `steward`, `guards`, `sidewalk`,
  `has_problem`, `boroname`, `dbh_group`, `curb_loc` (OneHotEncoder, `handle_unknown="ignore"`).

## 3. Models

| Stage | Model | Imbalance handling |
|---|---|---|
| Baseline | Logistic Regression (`max_iter=1000`) | `class_weight="balanced"` |
| Improved | Random Forest, tuned with GridSearchCV on macro-F1 | `class_weight="balanced_subsample"` |

Tuning searched Random Forest depth and leaf size with 3-fold stratified CV scored on
**macro-F1**. Best configuration: `max_depth=16, min_samples_leaf=2, max_features="sqrt",
n_estimators=150` (refit on the full training set as the deployed model).

## 4. Results (held-out test set)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Logistic Regression (baseline) | 0.494 | 0.357 | 0.554 |
| **Random Forest (tuned)** | **0.582** | **0.410** | **0.622** |

Per-class for the Random Forest (precision / recall / F1):

- Good: 0.83 / 0.66 / 0.74
- Fair: 0.28 / 0.30 / 0.29
- Poor: 0.13 / 0.45 / 0.20

The tuned Random Forest beats the baseline on both accuracy and macro-F1 and is the selected
model. Top feature drivers: `species_top15_or_other`, `tree_dbh`, `boroname`, then `dbh_group`
and `steward`.

> **Why not accuracy alone:** the classes are imbalanced, so macro-F1 and the confusion matrix
> are the honest headline metrics. The model is strong on the majority Good class and weak on
> the rare Poor class — say this openly in the presentation.

## 5. Files delivered

```
models/
  logistic_regression.joblib     # baseline pipeline (preprocess + model)
  random_forest.joblib           # tuned, deployed pipeline (~45 MB)
  model_metadata.json            # feature lists, dropdown options, metrics, importance
data/
  member2_model_metrics.json     # full metrics: per-class, confusion matrices, tuning results
  member2_model_comparison.csv   # baseline vs improved summary
visuals/
  08_model_comparison.png
  09_confusion_matrix_logreg.png
  10_confusion_matrix_rf.png
  11_feature_importance.png
  12_tuning_results.png
streamlit_pages/
  3_Model_Prediction.py          # multiple selectable models + probabilities
  4_Feature_Importance.py        # explainability
  5_Hyperparameter_Tuning.py     # experiment tracking
member2_modeling_notebook.ipynb  # executed notebook (data -> model -> evaluation)
scripts/
  create_member2_package.py      # reproducible: trains, evaluates, saves artifacts
  make_member2_visuals.py        # regenerates the 5 visuals from saved metrics
  build_member2_notebook.py      # builds + executes the notebook
requirements.txt                 # pinned dependencies (scikit-learn 1.7.2)
```

## 6. Handoff to Member 3 (app + deployment)

1. Copy `models/`, `data/member2_*`, and `streamlit_pages/3_*`, `4_*`, `5_*` into the app repo.
   The pages expect `models/` and `data/` at the **app root** (same convention as Member 1's pages).
2. Pages load artifacts by relative path and are cached (`st.cache_resource`), so they are fast.
3. Page 3 lets the grader switch between the baseline and Random Forest and shows class probabilities.
4. Use `requirements.txt` for the Hugging Face Space. **Pin `scikit-learn==1.7.2`** — the saved
   models were trained on it, and a different version will warn on load.
5. The only remaining required pages are **1 Introduction** and **2 Data Visualization** (Member 1,
   done) plus a **6 Conclusion** page (Member 3) and the app entry file (`app.py` / `Home.py`).

## 7. Reproducing the package

```bash
pip install -r requirements.txt
python scripts/create_member2_package.py    # trains + saves models, metrics, metadata
python scripts/make_member2_visuals.py      # regenerates the 5 PNGs
python scripts/build_member2_notebook.py    # rebuilds + executes the notebook
```

Sandbox note: the scripts use single-threaded forests and bounded tree depth so they run
within a small-memory (≈4 GB) environment; raise `n_estimators` / `max_depth` on a larger machine.

## 8. Limitations to include in the presentation

- Class imbalance: report macro-F1 and the confusion matrix, not just accuracy.
- 2015 census snapshot: not live tree health.
- Associations, not causes: feature importance shows what the model uses, not what causes poor health.
- Possible reporting/inspection bias in the recorded problem fields.
- Frame the app as an educational triage tool, not an official maintenance decision system.

## 9. Suggested presentation talking points (Member 2)

- Show the model-comparison chart: Random Forest improves accuracy (0.49 → 0.58) and macro-F1 (0.36 → 0.41).
- Show the confusion matrix and admit the Poor-class weakness honestly.
- Show feature importance: species, diameter, and borough lead.
- Explain the tuning: GridSearchCV on macro-F1 with stratified CV and balanced class weights.
