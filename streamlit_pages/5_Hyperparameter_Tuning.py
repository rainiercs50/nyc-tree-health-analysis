"""
Member 2 page: Hyperparameter Tuning / Experiment Tracking
Place this file in your Streamlit app's pages/ folder as 5_Hyperparameter_Tuning.py.

Expected paths from app root:
    data/member2_model_metrics.json
    data/member2_model_comparison.csv
    models/model_metadata.json
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYC Street Tree Health | Tuning", layout="wide")
st.title("🎛️ Hyperparameter Tuning & Experiment Tracking")
st.caption("Random Forest grid search on macro-F1 — modeling by Member 2")

METRICS_PATH = Path("data/member2_model_metrics.json")
COMPARISON_PATH = Path("data/member2_model_comparison.csv")
META_PATH = Path("models/model_metadata.json")

try:
    metrics = json.loads(METRICS_PATH.read_text())
    meta = json.loads(META_PATH.read_text())
except FileNotFoundError:
    st.error("Could not find Member 2 metrics. Copy the `data/` and `models/` folders into the app root.")
    st.stop()

# ------------------------------------------------------------ Approach summary
st.subheader("1. Tuning approach")
st.markdown("""
- **Search:** `GridSearchCV` over Random Forest depth and leaf size, with `sqrt` feature sampling.
- **Scoring:** **macro-F1** (not accuracy) so the rare *Poor* class counts as much as *Good*.
- **Validation:** 3-fold stratified cross-validation on the training data.
- **Imbalance handling:** `class_weight="balanced_subsample"` so the model is penalized for
  ignoring minority classes.
- The winning configuration is refit on the full training set and saved as the deployed model.
""")

# -------------------------------------------------------------- Search results
st.subheader("2. Search results (cross-validated macro-F1)")
tr = metrics.get("tuning_results", [])
if tr:
    rows = []
    for r in tr:
        p = r["params"]
        rows.append({
            "max_depth": p.get("max_depth"),
            "min_samples_leaf": p.get("min_samples_leaf"),
            "n_estimators": p.get("n_estimators"),
            "cv_macro_f1": round(r["mean_test_macro_f1"], 4),
            "cv_std": round(r["std_test_macro_f1"], 4),
        })
    res_df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    best_val = metrics.get("cv_best_macro_f1", res_df["cv_macro_f1"].max())

    res_df["label"] = ("depth=" + res_df["max_depth"].astype(str)
                       + ", leaf=" + res_df["min_samples_leaf"].astype(str))
    fig = px.bar(res_df.sort_values("cv_macro_f1"), x="cv_macro_f1", y="label",
                 orientation="h", error_x="cv_std", text="cv_macro_f1",
                 color="cv_macro_f1", color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="CV mean macro-F1",
                      yaxis_title="Candidate")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(res_df.drop(columns=["label"]), use_container_width=True)

    best = meta["models"]["random_forest"]["best_params"]
    st.success(f"**Best configuration:** {best}  •  CV macro-F1 = {best_val:.3f}")
else:
    st.caption("No tuning_results found in metrics file.")

# ----------------------------------------------------------- Baseline vs final
st.subheader("3. Baseline vs improved model (held-out test set)")
try:
    comp = pd.read_csv(COMPARISON_PATH)
    long = comp.melt(id_vars="model", value_vars=["accuracy", "macro_f1", "weighted_f1"],
                     var_name="metric", value_name="score")
    fig = px.bar(long, x="metric", y="score", color="model", barmode="group",
                 text="score")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="Metric", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp.round(4), use_container_width=True)
    st.write(
        "**Insight:** tuning the Random Forest improved both accuracy and macro-F1 over the "
        "Logistic Regression baseline. Macro-F1 is the honest headline number because the "
        "dataset is imbalanced."
    )
except FileNotFoundError:
    st.caption("Add data/member2_model_comparison.csv to show the baseline-vs-final comparison.")
