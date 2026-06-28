# Member 2 — Regression Modeling, SHAP & W&B

Project: **NYC Street Tree Health Predictor**
Role: **Member 2 — Modeling and Evaluation Lead**
Date: 2026-06-28

Predicts a continuous **tree health score** (Poor=0, Fair=1, Good=2) so trees can be ranked by
inspection priority. Built to the rubric: **linear regression**, **two switchable models**,
**SHAP** explainability, **Weights & Biases** tuning.

## Contents

- `models/linear_regression.joblib`, `models/random_forest_regressor.joblib` — the two models.
- `models/model_metadata.json` — features, dropdown options, metrics, SHAP + importances.
- `models/shap_sample.npz`, `models/eval_arrays.npz` — arrays for the plots.
- `data/member2_model_metrics.json`, `member2_model_comparison.csv` — evaluation.
- `data/member2_wandb_runs.csv`, `member2_wandb_summary.json` — W&B-tracked sweep.
- `visuals/08–14_*.png` — comparison, predicted-vs-actual, residuals, mapped confusion, tuning, SHAP.
- `streamlit_pages/3_Model_Prediction.py`, `4_Feature_Importance.py`, `5_Hyperparameter_Tuning.py`.
- `member2_modeling_notebook.ipynb` — executed notebook.
- `scripts/` — `create_member2_package.py`, `make_member2_visuals.py`, `tune_with_wandb.py`, `build_member2_notebook.py`.
- `requirements.txt`, `member2_handoff.md`.

## Key results (test set)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression (baseline) | 0.059 | 0.551 | 0.425 |
| Random Forest Regressor (tuned) | 0.087 | 0.543 | 0.416 |

Best RF: `depth=16, leaf=5, sqrt, 150 trees`. Top drivers: species, trunk diameter, borough,
visible problems. R² is low (tree health is weakly predictable) — reported honestly; the model is
useful for *ranking*, not precise measurement.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/create_member2_package.py
python scripts/make_member2_visuals.py
WANDB_MODE=offline python scripts/tune_with_wandb.py
python scripts/build_member2_notebook.py
```

## Handoff to Member 3

Copy `models/`, `data/member2_*`, and `streamlit_pages/3–5` into the app root; deploy with the
pinned `requirements.txt`. For the live W&B dashboard, run `wandb login` once. See
`member2_handoff.md` for full details (including 5 stale files to delete).
