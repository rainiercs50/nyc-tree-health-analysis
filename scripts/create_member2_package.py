"""
Member 2 package builder: Regression modeling, evaluation, and SHAP.

Business framing
----------------
We predict a continuous TREE HEALTH SCORE (Poor=0, Fair=1, Good=2) from observable
tree features. A continuous score lets the city RANK trees for inspection priority
(lower score = higher priority), which is more useful than three fixed buckets.

Two switchable models (required): Linear Regression (baseline) and a tuned
Random Forest Regressor (improved). We report regression metrics (R2, RMSE, MAE)
and also map the score back to Good/Fair/Poor for an interpretable accuracy/confusion.

Run from the project root:
    python scripts/create_member2_package.py

Inputs:
    data/nyc_tree_member1_model_ready_sample.csv
Outputs:
    models/linear_regression.joblib
    models/random_forest_regressor.joblib
    models/model_metadata.json
    models/shap_sample.npz            (shap values + transformed sample for plots)
    models/eval_arrays.npz            (y_true, predictions for plots)
    data/member2_model_metrics.json
    data/member2_model_comparison.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix,
)
import shap

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nyc_tree_member1_model_ready_sample.csv"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

TARGET_CAT = "health"
SCORE_MAP = {"Poor": 0, "Fair": 1, "Good": 2}
INV_SCORE = {v: k for k, v in SCORE_MAP.items()}
CLASS_ORDER = ["Good", "Fair", "Poor"]
NUMERIC = ["tree_dbh", "problem_count"]
CATEGORICAL = ["species_top15_or_other", "steward", "guards", "sidewalk",
               "has_problem", "boroname", "dbh_group", "curb_loc"]

print("Loading data ...", flush=True)
df = pd.read_csv(DATA)
df["health_score"] = df[TARGET_CAT].map(SCORE_MAP)
X = df[NUMERIC + CATEGORICAL].copy()
y = df["health_score"].astype(float)

# Stratify by the original category so class proportions are preserved.
X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
    X, y, df[TARGET_CAT], test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET_CAT]
)
print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}", flush=True)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])


def score_to_cat(arr):
    idx = np.clip(np.rint(arr), 0, 2).astype(int)
    return np.array([INV_SCORE[i] for i in idx])


def evaluate(name, pipe):
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    pred_cat = score_to_cat(pred)
    acc = accuracy_score(cat_test, pred_cat)
    macro_f1 = f1_score(cat_test, pred_cat, average="macro", labels=CLASS_ORDER, zero_division=0)
    cm = confusion_matrix(cat_test, pred_cat, labels=CLASS_ORDER)
    print(f"\n=== {name} ===", flush=True)
    print(f"R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}", flush=True)
    print(f"Mapped-to-category accuracy={acc:.4f}  macro-F1={macro_f1:.4f}", flush=True)
    return {"name": name, "r2": r2, "rmse": rmse, "mae": mae,
            "mapped_accuracy": acc, "mapped_macro_f1": macro_f1,
            "mapped_confusion": cm.tolist()}, pred


# ----------------------------------------------------------------- Baseline LR
print("\nTraining baseline: Linear Regression ...", flush=True)
linreg = Pipeline([("prep", preprocess), ("clf", LinearRegression())])
linreg.fit(X_train, y_train)
lr_eval, lr_pred = evaluate("Linear Regression (baseline)", linreg)

# ------------------------------------------------------ Improved RF + tuning
# Sandbox-safe: single-threaded, bounded depth, grid search on a subsample,
# final model refit on the full training set.
print("\nTuning improved model: Random Forest Regressor (GridSearchCV on R2) ...", flush=True)
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
])
param_grid = {
    "clf__n_estimators": [150],
    "clf__max_depth": [10, 16],
    "clf__min_samples_leaf": [2, 5],
    "clf__max_features": ["sqrt"],
}
X_sub, _, y_sub, _ = train_test_split(
    X_train, y_train, train_size=12000, random_state=RANDOM_STATE, stratify=cat_train
)
search = GridSearchCV(rf_pipe, param_grid, scoring="r2",
                      cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
                      n_jobs=1, refit=False, verbose=2)
search.fit(X_sub, y_sub)
print(f"\nBest CV R2 (subsample): {search.best_score_:.4f}", flush=True)
print(f"Best params: {search.best_params_}", flush=True)

final_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
final_params["n_estimators"] = 150
print(f"Refitting final RF regressor on full training set: {final_params}", flush=True)
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **final_params)),
])
rf.fit(X_train, y_train)
rf_eval, rf_pred = evaluate("Random Forest Regressor (tuned, improved)", rf)

# ---------------------------------------------------- RF feature importance
feat_names = rf.named_steps["prep"].get_feature_names_out()
importances = rf.named_steps["clf"].feature_importances_


def aggregate(values, names):
    agg = {}
    for nm, v in zip(names, values):
        raw = nm.split("__", 1)[1]
        base = next((c for c in NUMERIC + CATEGORICAL if raw == c or raw.startswith(c + "_")), raw)
        agg[base] = agg.get(base, 0.0) + float(v)
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


fi = aggregate(importances, feat_names)
print("\nRF feature importance:", {k: round(v, 3) for k, v in fi.items()}, flush=True)

# Linear Regression coefficients (signed, on standardized inputs) -> direction
lr_coef = linreg.named_steps["clf"].coef_
lr_coef_abs = aggregate(np.abs(lr_coef), feat_names)

# -------------------------------------------------------------------- SHAP
print("\nComputing SHAP values (TreeExplainer on RF regressor) ...", flush=True)
n_shap = min(250, len(X_test))
X_shap = X_test.iloc[:n_shap]
Xt = rf.named_steps["prep"].transform(X_shap)
Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
explainer = shap.TreeExplainer(rf.named_steps["clf"])
# approximate=True (Saabas) keeps SHAP fast enough for the sandbox / live app.
shap_vals = explainer.shap_values(Xt, check_additivity=False, approximate=True)
shap_abs_mean = np.abs(shap_vals).mean(axis=0)
shap_importance = aggregate(shap_abs_mean, feat_names)
print("SHAP importance:", {k: round(v, 3) for k, v in shap_importance.items()}, flush=True)

np.savez_compressed(MODELS / "shap_sample.npz",
                    shap_values=shap_vals, X=Xt,
                    feat_names=np.array(feat_names, dtype=object),
                    base_value=np.array([explainer.expected_value]).ravel())
np.savez_compressed(MODELS / "eval_arrays.npz",
                    y_true=y_test.to_numpy(), lr_pred=lr_pred, rf_pred=rf_pred,
                    cat_true=cat_test.to_numpy().astype(str))

# --------------------------------------------------------------- Artifacts
print("\nSaving model artifacts ...", flush=True)
joblib.dump(linreg, MODELS / "linear_regression.joblib")
joblib.dump(rf, MODELS / "random_forest_regressor.joblib")

cat_options = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
num_meta = {c: {"min": int(df[c].min()), "max": int(df[c].max()), "median": int(df[c].median())}
            for c in NUMERIC}
metadata = {
    "task": "regression",
    "target": "health_score",
    "target_from": TARGET_CAT,
    "score_map": SCORE_MAP,
    "class_order": CLASS_ORDER,
    "numeric_features": NUMERIC,
    "categorical_features": CATEGORICAL,
    "categorical_options": cat_options,
    "numeric_meta": num_meta,
    "models": {
        "linear_regression": {
            "file": "models/linear_regression.joblib",
            "label": "Linear Regression (baseline)",
            "r2": lr_eval["r2"], "rmse": lr_eval["rmse"], "mae": lr_eval["mae"],
            "mapped_accuracy": lr_eval["mapped_accuracy"],
        },
        "random_forest_regressor": {
            "file": "models/random_forest_regressor.joblib",
            "label": "Random Forest Regressor (tuned)",
            "r2": rf_eval["r2"], "rmse": rf_eval["rmse"], "mae": rf_eval["mae"],
            "mapped_accuracy": rf_eval["mapped_accuracy"],
            "best_params": final_params,
        },
    },
    "recommended_model": "random_forest_regressor",
    "feature_importance": {k: round(v, 4) for k, v in fi.items()},
    "shap_importance": {k: round(v, 4) for k, v in shap_importance.items()},
    "linear_abs_coef": {k: round(v, 4) for k, v in lr_coef_abs.items()},
}
(MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

metrics = {
    "task": "regression",
    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    "score_map": SCORE_MAP,
    "class_distribution_full": df[TARGET_CAT].value_counts().to_dict(),
    "cv_best_r2": float(search.best_score_),
    "tuning_results": [
        {"params": {k.replace("clf__", ""): v for k, v in p.items()},
         "mean_test_r2": float(m), "std_test_r2": float(s)}
        for p, m, s in zip(search.cv_results_["params"],
                           search.cv_results_["mean_test_score"],
                           search.cv_results_["std_test_score"])
    ],
    "linear_regression": lr_eval,
    "random_forest_regressor": rf_eval,
}
(ROOT / "data" / "member2_model_metrics.json").write_text(json.dumps(metrics, indent=2))

pd.DataFrame([
    {"model": "Linear Regression (baseline)", "r2": lr_eval["r2"], "rmse": lr_eval["rmse"],
     "mae": lr_eval["mae"], "mapped_accuracy": lr_eval["mapped_accuracy"]},
    {"model": "Random Forest Regressor (tuned)", "r2": rf_eval["r2"], "rmse": rf_eval["rmse"],
     "mae": rf_eval["mae"], "mapped_accuracy": rf_eval["mapped_accuracy"]},
]).to_csv(ROOT / "data" / "member2_model_comparison.csv", index=False)

print("\nTRAINING+SAVE DONE", flush=True)
print("Next: python scripts/make_member2_visuals.py", flush=True)
