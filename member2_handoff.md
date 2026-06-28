# Member 2 Handoff: Regression Modeling, SHAP, and W&B

**Project:** NYC Street Tree Health Predictor
**Course:** Intro to Data Science Final Project
**Member 2 role:** Modeling and Evaluation Lead
**Date:** 2026-06-28

## 1. What Member 2 completed

A full modeling layer aligned to the assignment rubric (**linear regression**, **two switchable
models**, **SHAP** explainability, **Weights & Biases** tuning): a preprocessing pipeline, a Linear
Regression baseline, a tuned Random Forest Regressor, honest regression evaluation, SHAP
explainability, W&B-tracked hyperparameter search, saved artifacts, three Streamlit pages, and an
executed modeling notebook.

## 2. Problem framing (why regression)

We predict a continuous **health score**: `Poor=0, Fair=1, Good=2`. A single number lets the city
**rank trees by inspection priority** (lower score = higher priority), which is more actionable
than three fixed buckets and lets us use **linear regression** as required.

## 3. Data and feature contract

- **Input:** `data/nyc_tree_member1_model_ready_sample.csv` (49,994 rows; from Member 1).
- **Target:** `health_score` (0–2), derived from `health`. Split stratified by the original
  category, 80/20 → 39,995 train / 9,999 test, `random_state=42`.
- **Numeric:** `tree_dbh`, `problem_count` (StandardScaler).
- **Categorical:** `species_top15_or_other`, `steward`, `guards`, `sidewalk`, `has_problem`,
  `boroname`, `dbh_group`, `curb_loc` (OneHotEncoder, `handle_unknown="ignore"`).

## 4. Models (two switchable)

| Stage | Model | Notes |
|---|---|---|
| Baseline | **Linear Regression** | required linear model |
| Improved | **Random Forest Regressor** | tuned with GridSearchCV on R² + tracked in W&B |

Best RF config: `max_depth=16, min_samples_leaf=5, max_features="sqrt", n_estimators=150`
(refit on the full training set).

## 5. Results (held-out test set)

| Model | R² | RMSE | MAE | Mapped accuracy* |
|---|---|---|---|---|
| Linear Regression (baseline) | 0.059 | 0.551 | 0.425 | 0.739 |
| **Random Forest Regressor (tuned)** | **0.087** | **0.543** | **0.416** | 0.738 |

\*Mapped accuracy = round the predicted score to the nearest band (Poor/Fair/Good) and compare to
the true category.

> **Honest finding:** R² is low — tree health is only weakly predictable from these observable
> features, and the majority class is *Good*. The Random Forest still beats the linear baseline on
> R²/RMSE/MAE. Present this transparently and frame the app as a **triage/ranking** tool, not a
> precise health meter. This honesty is exactly what the rubric's "limitations" criterion rewards.

Top drivers (consistent across RF importance, SHAP, and linear coefficients): **species**,
**trunk diameter / diameter group**, **borough**, and **visible problems**.

## 6. Explainability — SHAP (rubric requirement)

`scripts/create_member2_package.py` runs `shap.TreeExplainer` on the Random Forest Regressor and
saves the SHAP values + a beeswarm and an aggregated importance plot. Page 4 displays them.
Direction of effect: more **visible problems** and being a **Norway maple** push the predicted
score **down**; larger **diameter** and **honeylocust / London planetree** push it **up**.

## 7. Hyperparameter tuning — Weights & Biases (rubric requirement)

`scripts/tune_with_wandb.py` logs each RF configuration as a W&B run (CV R² + RMSE) and selects
the best. It works two ways:

```bash
wandb login                                   # your API key, for the live dashboard
python scripts/tune_with_wandb.py
# or, no account needed (what generated the saved results):
WANDB_MODE=offline python scripts/tune_with_wandb.py
wandb sync wandb/offline-run-*                # push offline runs to your dashboard later
```

Results are saved to `data/member2_wandb_runs.csv` / `member2_wandb_summary.json`, which page 5
reads. **For the live demo, run `wandb login` once so the W&B dashboard shows the runs.**

## 8. Files delivered

```
models/
  linear_regression.joblib            # baseline pipeline
  random_forest_regressor.joblib      # tuned, deployed pipeline (~18 MB)
  model_metadata.json                 # features, options, metrics, SHAP + importance
  shap_sample.npz, eval_arrays.npz    # arrays for the SHAP/eval plots
data/
  member2_model_metrics.json          # R2/RMSE/MAE, mapped confusion, tuning results
  member2_model_comparison.csv
  member2_wandb_runs.csv, member2_wandb_summary.json
visuals/
  08_model_comparison.png  09_predicted_vs_actual_rf.png  10_residuals_rf.png
  11_confusion_mapped_rf.png  12_tuning_results.png
  13_shap_importance.png  14_shap_beeswarm.png
streamlit_pages/
  3_Model_Prediction.py  4_Feature_Importance.py  5_Hyperparameter_Tuning.py
member2_modeling_notebook.ipynb       # executed (regression -> SHAP -> comparison)
scripts/
  create_member2_package.py  make_member2_visuals.py
  tune_with_wandb.py  build_member2_notebook.py
requirements.txt
```

## 9. Handoff to Member 3 (app + deployment)

1. Copy `models/`, `data/member2_*`, and `streamlit_pages/3–5` into the app root (same convention
   as Member 1's pages: `models/` and `data/` live at the app root).
2. Pages are cached and load artifacts by relative path.
3. Use `requirements.txt` for Hugging Face. **Pin `scikit-learn==1.7.2`** (the models were trained
   on it). `shap` and `wandb` are included.
4. Remaining required pages: **1 Introduction**, **2 Data Visualization** (Member 1, done),
   **6 Conclusion** (Member 3), plus the app entry file (`Home.py`/`app.py`).
5. **Cleanup:** 5 stale classification files from an earlier iteration can be deleted —
   `models/logistic_regression.joblib`, `models/random_forest.joblib`,
   `visuals/09_confusion_matrix_logreg.png`, `visuals/10_confusion_matrix_rf.png`,
   `visuals/11_feature_importance.png`.

## 10. Reproduce

```bash
pip install -r requirements.txt
python scripts/create_member2_package.py     # train + SHAP + save artifacts/metrics
python scripts/make_member2_visuals.py       # 7 visuals
WANDB_MODE=offline python scripts/tune_with_wandb.py   # W&B-tracked sweep
python scripts/build_member2_notebook.py     # rebuild + execute the notebook
```

## 11. Presentation talking points (Member 2)

- Frame: predict a 0–2 health score to rank trees by inspection priority.
- Two models: Linear Regression baseline vs tuned Random Forest; RF wins on R²/RMSE.
- SHAP beeswarm: species, diameter, and visible problems drive the score, with direction.
- W&B: show the tracked runs and the selected best configuration.
- Be honest about the low R² and the imbalance — it is a feature of the data, not a bug in the work.
