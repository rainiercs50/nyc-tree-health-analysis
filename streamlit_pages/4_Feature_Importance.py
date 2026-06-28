"""
Member 2 page: Feature Importance / Explainability
Place this file in your Streamlit app's pages/ folder as 4_Feature_Importance.py.

Expected paths from app root:
    models/model_metadata.json
    data/member2_model_metrics.json
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYC Street Tree Health | Feature Importance", layout="wide")
st.title("🔎 Feature Importance & Explainability")
st.caption("Which inputs drive the Random Forest — modeling by Member 2")

META_PATH = Path("models/model_metadata.json")
METRICS_PATH = Path("data/member2_model_metrics.json")

try:
    meta = json.loads(META_PATH.read_text())
except FileNotFoundError:
    st.error("Could not find models/model_metadata.json. Copy the Member 2 `models/` folder into the app root.")
    st.stop()

fi = meta.get("feature_importance", {})
fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())}) \
    .sort_values("importance", ascending=True)

st.subheader("1. Random Forest feature importance")
fig = px.bar(fi_df, x="importance", y="feature", orientation="h", text="importance",
             color="importance", color_continuous_scale="Greens")
fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.update_layout(coloraxis_showscale=False, xaxis_title="Aggregated importance",
                  yaxis_title="Feature")
st.plotly_chart(fig, use_container_width=True)

top3 = list(fi_df.sort_values("importance", ascending=False)["feature"].head(3))
st.write(
    f"**Insight:** the strongest drivers are **{', '.join(top3)}**. One-hot columns were "
    "aggregated back to their original feature so each bar reflects the full contribution "
    "of that input."
)

st.subheader("2. How to read this")
st.markdown("""
- Importance here is the Random Forest **impurity-based importance**, aggregated across all
  one-hot categories of each feature.
- Higher means the feature was more useful for splitting trees toward the correct health class.
- Importance shows **association, not causation**. A feature can look important because it
  correlates with something else (for example, species correlates with typical size and planting location).
- Tree size (`tree_dbh` / `dbh_group`) and `species` carrying high importance is consistent
  with the EDA: larger and certain species of trees show different Good/Fair/Poor mixes.
""")

# Per-class recall context, if available
try:
    metrics = json.loads(METRICS_PATH.read_text())
    rep = metrics["random_forest"]["report"]
    rows = [{"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
             "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
            for c in ["Good", "Fair", "Poor"]]
    st.subheader("3. Where the model is confident vs weak")
    st.dataframe(pd.DataFrame(rows).set_index("class").round(3), use_container_width=True)
    st.write(
        "**Insight:** the model identifies **Good** trees well but struggles on the rare "
        "**Poor** class. Feature importance tells us *what* the model uses; this table tells "
        "us *where* it still makes mistakes, which matters for an imbalanced problem."
    )
except FileNotFoundError:
    st.caption("Add data/member2_model_metrics.json to show per-class precision/recall.")
