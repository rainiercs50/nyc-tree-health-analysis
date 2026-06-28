"""
Member 2 package builder: Modeling and Evaluation.

Reproducible script that trains a baseline (Logistic Regression) and an improved
model (Random Forest), tunes the Random Forest with cross-validated search on
macro-F1, evaluates both, extracts feature importance, and saves all artifacts,
metrics, and visuals used by the Streamlit app and the presentation.

Run from the project root:
    python scripts/create_member2_package.py

Inputs:
    data/nyc_tree_member1_model_ready_sample.csv  (from Member 1)

Outputs:
    models/logistic_regression.joblib
    models/random_forest.joblib
    models/model_metadata.json
    data/member2_model_metrics.json
    data/member2_model_comparison.csv
    visuals/08_model_comparison.png
    visuals/09_confusion_matrix_logreg.png
    visuals/10_confusion_matrix_rf.png
    visuals/11_feature_importance.png
    visuals/12_tuning_results.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nyc_tree_member1_model_ready_sample.csv"
MODELS = ROOT / "models"
VISUALS = ROOT / "visuals"
MODELS.mkdir(exist_ok=True)
VISUALS.mkdir(exist_ok=True)

# Feature contract agreed with Member 1 (see member1_handoff.md section 7)
TARGET = "health"
NUMERIC = ["tree_dbh", "problem_count"]
CATEGORICAL = [
    "species_top15_or_other", "steward", "guards", "sidewalk",
    "has_problem", "boroname", "dbh_group", "curb_loc",
]
CLASS_ORDER = ["Good", "Fair", "Poor"]
HEALTH_COLORS = {"Good": "#2E7D32", "Fair": "#F9A825", "Poor": "#C62828"}

print("Loading data ...", flush=True)
df = pd.read_csv(DATA)
X = df[NUMERIC + CATEGORICAL].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}", flush=True)

# Shared preprocessing: scale numeric (helps LogReg), one-hot categorical.
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
)


def evaluate(name, pipe):
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")
    report = classification_report(
        y_test, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
    print(f"\n=== {name} ===", flush=True)
    print(f"Accuracy   : {acc:.4f}", flush=True)
    print(f"Macro-F1   : {macro_f1:.4f}", flush=True)
    print(f"Weighted-F1: {weighted_f1:.4f}", flush=True)
    print(classification_report(y_test, pred, labels=CLASS_ORDER, zero_division=0), flush=True)
    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------- Baseline
print("\nTraining baseline: Logistic Regression ...", flush=True)
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=None)),
])
logreg.fit(X_train, y_train)
logreg_eval = evaluate("Logistic Regression (baseline)", logreg)

# ---------------------------------------------------------- Improved + tuning
# Sandbox constraints (4 cores, ~3.9 GB RAM, no swap):
#  - n_jobs=1 everywhere: single-threaded avoids loky subprocess issues.
#  - Bounded max_depth keeps forest memory well under the RAM limit.
#  - Grid search runs on a stratified subsample for speed; the FINAL model is
#    refit on the full training set with the winning parameters.
print("\nTuning improved model: Random Forest (GridSearchCV on macro-F1) ...", flush=True)
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=1)),
])
param_grid = {
    "clf__n_estimators": [150],
    "clf__max_depth": [10, 16],
    "clf__min_samples_leaf": [2, 5],
    "clf__max_features": ["sqrt"],
}
X_sub, _, y_sub, _ = train_test_split(
    X_train, y_train, train_size=12000, random_state=RANDOM_STATE, stratify=y_train
)
search = GridSearchCV(
    rf_pipe, param_grid=param_grid,
    scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=1, refit=False, verbose=2,
)
search.fit(X_sub, y_sub)
print(f"\nBest CV macro-F1 (subsample): {search.best_score_:.4f}", flush=True)
print(f"Best params: {search.best_params_}", flush=True)

# Refit the final Random Forest on the FULL training set with the winning params.
final_clf_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
final_clf_params["n_estimators"] = 150
print(f"Refitting final RF on full training set: {final_clf_params}", flush=True)
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=1, **final_clf_params)),
])
rf.fit(X_train, y_train)
rf_eval = evaluate("Random Forest (tuned, improved)", rf)

# ---------------------------------------------------------- Feature importance
print("\nComputing feature importance ...", flush=True)
feat_names = rf.named_steps["prep"].get_feature_names_out()
importances = rf.named_steps["clf"].feature_importances_
# Aggregate one-hot columns back to their original feature.
agg = {}
for fname, imp in zip(feat_names, importances):
    raw = fname.split("__", 1)[1]  # drop num__ / cat__ prefix
    base = None
    for col in NUMERIC + CATEGORICAL:
        if raw == col or raw.startswith(col + "_"):
            base = col
            break
    base = base or raw
    agg[base] = agg.get(base, 0.0) + float(imp)
fi = pd.Series(agg).sort_values(ascending=False)
print(fi.to_string(), flush=True)

# ------------------------------------------------------------------- Artifacts
print("\nSaving model artifacts ...", flush=True)
joblib.dump(logreg, MODELS / "logistic_regression.joblib")
joblib.dump(rf, MODELS / "random_forest.joblib")

cat_options = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
num_meta = {
    c: {"min": int(df[c].min()), "max": int(df[c].max()), "median": int(df[c].median())}
    for c in NUMERIC
}
metadata = {
    "target": TARGET,
    "class_order": CLASS_ORDER,
    "numeric_features": NUMERIC,
    "categorical_features": CATEGORICAL,
    "categorical_options": cat_options,
    "numeric_meta": num_meta,
    "models": {
        "logistic_regression": {
            "file": "models/logistic_regression.joblib",
            "label": "Logistic Regression (baseline)",
            "accuracy": logreg_eval["accuracy"],
            "macro_f1": logreg_eval["macro_f1"],
        },
        "random_forest": {
            "file": "models/random_forest.joblib",
            "label": "Random Forest (tuned)",
            "accuracy": rf_eval["accuracy"],
            "macro_f1": rf_eval["macro_f1"],
            "best_params": final_clf_params,
        },
    },
    "recommended_model": "random_forest",
    "feature_importance": fi.round(4).to_dict(),
}
(MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

metrics = {
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "class_distribution_full": df[TARGET].value_counts().to_dict(),
    "cv_best_macro_f1": float(search.best_score_),
    "tuning_results": [
        {"params": {k.replace("clf__", ""): v for k, v in p.items()},
         "mean_test_macro_f1": float(m), "std_test_macro_f1": float(s)}
        for p, m, s in zip(search.cv_results_["params"],
                           search.cv_results_["mean_test_score"],
                           search.cv_results_["std_test_score"])
    ],
    "logistic_regression": logreg_eval,
    "random_forest": rf_eval,
}
(ROOT / "data" / "member2_model_metrics.json").write_text(json.dumps(metrics, indent=2))

comparison = pd.DataFrame([
    {"model": "Logistic Regression (baseline)", "accuracy": logreg_eval["accuracy"],
     "macro_f1": logreg_eval["macro_f1"], "weighted_f1": logreg_eval["weighted_f1"]},
    {"model": "Random Forest (tuned)", "accuracy": rf_eval["accuracy"],
     "macro_f1": rf_eval["macro_f1"], "weighted_f1": rf_eval["weighted_f1"]},
])
comparison.to_csv(ROOT / "data" / "member2_model_comparison.csv", index=False)

print("\nTRAINING+SAVE DONE", flush=True)
print("Next: python scripts/make_member2_visuals.py", flush=True)
