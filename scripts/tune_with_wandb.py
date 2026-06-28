"""
Hyperparameter tuning tracked with Weights & Biases.

Each Random Forest Regressor configuration is logged as a separate W&B run so the
experiments can be compared in the W&B dashboard, and the best one is selected by
cross-validated R2. Results are also written to CSV/JSON for the Streamlit page.

Run logged in to your own account (recommended for the dashboard):
    wandb login                       # once, with your API key
    python scripts/tune_with_wandb.py

Run without an account (offline, what we used to generate the saved results):
    WANDB_MODE=offline python scripts/tune_with_wandb.py
    # later: wandb sync wandb/offline-run-*   to push to your dashboard

Outputs:
    data/member2_wandb_runs.csv
    data/member2_wandb_summary.json
"""
from __future__ import annotations
import os
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nyc_tree_member1_model_ready_sample.csv"
PROJECT = os.environ.get("WANDB_PROJECT", "nyc-tree-health-predictor")

SCORE_MAP = {"Poor": 0, "Fair": 1, "Good": 2}
NUMERIC = ["tree_dbh", "problem_count"]
CATEGORICAL = ["species_top15_or_other", "steward", "guards", "sidewalk",
               "has_problem", "boroname", "dbh_group", "curb_loc"]

df = pd.read_csv(DATA)
y = df["health"].map(SCORE_MAP).astype(float)
X = df[NUMERIC + CATEGORICAL]
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE,
                                    stratify=df["health"])
# Subsample for fast, repeatable sweeps.
X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=12000,
                                      random_state=RANDOM_STATE)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

grid = {
    "max_depth": [10, 16],
    "min_samples_leaf": [2, 5],
    "n_estimators": [150],
    "max_features": ["sqrt"],
}
combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
cv = KFold(3, shuffle=True, random_state=RANDOM_STATE)

rows = []
print(f"Tracking {len(combos)} configurations on W&B project '{PROJECT}' "
      f"(mode={os.environ.get('WANDB_MODE', 'online')})", flush=True)
for i, params in enumerate(combos, 1):
    run = wandb.init(project=PROJECT, name=f"rf_d{params['max_depth']}_l{params['min_samples_leaf']}",
                     config=params, reinit=True)
    pipe = Pipeline([("prep", preprocess),
                     ("clf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params))])
    scores = cross_validate(pipe, X_sub, y_sub, cv=cv,
                            scoring=["r2", "neg_root_mean_squared_error"], n_jobs=1)
    cv_r2 = float(scores["test_r2"].mean())
    cv_rmse = float(-scores["test_neg_root_mean_squared_error"].mean())
    wandb.log({"cv_r2": cv_r2, "cv_rmse": cv_rmse})
    wandb.summary["cv_r2"] = cv_r2
    wandb.summary["cv_rmse"] = cv_rmse
    rows.append({**params, "cv_r2": round(cv_r2, 4), "cv_rmse": round(cv_rmse, 4)})
    print(f"  [{i}/{len(combos)}] {params} -> cv_r2={cv_r2:.4f} cv_rmse={cv_rmse:.4f}", flush=True)
    run.finish()

res = pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
res.to_csv(ROOT / "data" / "member2_wandb_runs.csv", index=False)
best = res.iloc[0].to_dict()
summary = {
    "project": PROJECT,
    "mode": os.environ.get("WANDB_MODE", "online"),
    "n_runs": len(combos),
    "selection_metric": "cv_r2",
    "best": best,
}
(ROOT / "data" / "member2_wandb_summary.json").write_text(json.dumps(summary, indent=2))
print("\nBest config:", best, flush=True)
print("WANDB TUNING DONE", flush=True)
