# Member 2 Modeling + Evaluation Package

Project: **NYC Street Tree Health Predictor**
Role: **Member 2 — Modeling and Evaluation Lead**
Date: 2026-06-28

## What this package contains

- `models/random_forest.joblib` — tuned, deployed pipeline (preprocessing + Random Forest).
- `models/logistic_regression.joblib` — baseline pipeline.
- `models/model_metadata.json` — feature lists, dropdown options, headline metrics, importances.
- `data/member2_model_metrics.json` — full metrics (per-class, confusion matrices, tuning results).
- `data/member2_model_comparison.csv` — baseline vs improved summary.
- `visuals/08–12_*.png` — model comparison, two confusion matrices, feature importance, tuning.
- `streamlit_pages/3_Model_Prediction.py`, `4_Feature_Importance.py`, `5_Hyperparameter_Tuning.py`.
- `member2_modeling_notebook.ipynb` — executed notebook telling the full modeling story.
- `scripts/create_member2_package.py`, `make_member2_visuals.py`, `build_member2_notebook.py`.
- `requirements.txt` — pinned dependencies.
- `member2_handoff.md` — written handoff for Member 3.

## Member 2 responsibilities covered

1. Built a preprocessing pipeline (scaling + one-hot) on Member 1's feature contract.
2. Trained a Logistic Regression baseline with balanced class weights.
3. Tuned a Random Forest with GridSearchCV scored on macro-F1 (stratified 3-fold CV).
4. Evaluated both models: accuracy, macro-F1, confusion matrix, per-class precision/recall.
5. Produced feature-importance explainability.
6. Saved deployable artifacts and three Streamlit pages for the final app.

## Key results

- Random Forest (tuned): **accuracy 0.582, macro-F1 0.410** — beats the baseline (0.494 / 0.357).
- Best params: `max_depth=16, min_samples_leaf=2, max_features="sqrt", n_estimators=150`.
- Top drivers: species, trunk diameter, borough.
- The model is strong on Good and weak on the rare Poor class — macro-F1 is the honest metric.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/create_member2_package.py
python scripts/make_member2_visuals.py
python scripts/build_member2_notebook.py
```

## Handoff to Member 3

Copy `models/`, `data/member2_*`, and `streamlit_pages/3_*–5_*` into the app root and deploy with
the pinned `requirements.txt`. See `member2_handoff.md` for details. **Pin `scikit-learn==1.7.2`.**
