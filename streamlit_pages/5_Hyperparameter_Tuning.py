"""
Member 2 page: Hyperparameter Tuning tracked with Weights & Biases
Place in your Streamlit app's pages/ folder as 5_Hyperparameter_Tuning.py.

Expected paths from app root:
    data/member2_wandb_runs.csv
    data/member2_wandb_summary.json
    data/member2_model_comparison.csv
    models/model_metadata.json
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYC Street Tree Health | Tuning", layout="wide")
st.title("🎛️ Hyperparameter Tuning — Weights & Biases")
st.caption("Tracked Random Forest experiments and model selection — modeling by Member 2")

RUNS = Path("data/member2_wandb_runs.csv")
SUMMARY = Path("data/member2_wandb_summary.json")
COMPARISON = Path("data/member2_model_comparison.csv")
META = Path("models/model_metadata.json")

st.markdown("""
We tracked every Random Forest configuration with **Weights & Biases**. Each configuration is a
W&B run logging its cross-validated **R²** and **RMSE**, so experiments can be compared in the
W&B dashboard and the best one selected automatically.

```bash
wandb login                 # once, with your API key
python scripts/tune_with_wandb.py
# offline (no account): WANDB_MODE=offline python scripts/tune_with_wandb.py
```
""")

# ---------------------------------------------------------------- Tracked runs
st.subheader("1. Tracked experiments (cross-validated R²)")
if RUNS.exists():
    runs = pd.read_csv(RUNS)
    runs_sorted = runs.sort_values("cv_r2", ascending=False).reset_index(drop=True)
    runs_sorted["config"] = ("depth=" + runs_sorted["max_depth"].astype(str)
                             + ", leaf=" + runs_sorted["min_samples_leaf"].astype(str))
    fig = px.bar(runs_sorted.sort_values("cv_r2"), x="cv_r2", y="config", orientation="h",
                 text="cv_r2", color="cv_r2", color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="CV mean R² (3-fold)",
                      yaxis_title="Configuration")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(runs_sorted.drop(columns=["config"]), use_container_width=True)

    if SUMMARY.exists():
        s = json.loads(SUMMARY.read_text())
        best = s["best"]
        st.success(
            f"**Best configuration** (by {s['selection_metric']}): "
            f"depth={best['max_depth']}, leaf={best['min_samples_leaf']}, "
            f"trees={best['n_estimators']}  →  R²={best['cv_r2']:.4f}, RMSE={best['cv_rmse']:.4f}  "
            f"•  W&B project `{s['project']}` ({s['n_runs']} runs)"
        )
else:
    st.caption("Run scripts/tune_with_wandb.py to generate data/member2_wandb_runs.csv.")

# ----------------------------------------------------------- Baseline vs final
st.subheader("2. Selected model vs baseline (held-out test set)")
if COMPARISON.exists():
    comp = pd.read_csv(COMPARISON)
    long = comp.melt(id_vars="model", value_vars=["r2", "mapped_accuracy"],
                     var_name="metric", value_name="score")
    fig = px.bar(long, x="metric", y="score", color="model", barmode="group", text="score")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(xaxis_title="Metric (higher is better)", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp.round(4), use_container_width=True)
    st.write("**Insight:** the tuned Random Forest Regressor edges out the Linear Regression "
             "baseline on R² and RMSE. R² is low because tree health is only weakly predictable "
             "from these features — an honest finding we report rather than hide. The model is "
             "still useful for *ranking* trees by inspection priority.")
else:
    st.caption("Add data/member2_model_comparison.csv to show the comparison.")
