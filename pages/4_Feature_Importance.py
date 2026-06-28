"""
Member 2 page: Explainable AI with SHAP
Place in your Streamlit app's pages/ folder as 4_Feature_Importance.py.

Expected paths from app root:
    models/model_metadata.json
    visuals/13_shap_importance.png
    visuals/14_shap_beeswarm.png
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYC Street Tree Health | Explainable AI", layout="wide")
st.title("🔎 Explainable AI — SHAP")
st.caption("What drives the predicted health score — modeling by Member 2")

META_PATH = Path("models/model_metadata.json")
BEESWARM = Path("visuals/14_shap_beeswarm.png")

try:
    meta = json.loads(META_PATH.read_text())
except FileNotFoundError:
    st.error("Could not find models/model_metadata.json. Copy the Member 2 `models/` folder into the app root.")
    st.stop()

st.markdown(
    "We explain the **Random Forest Regressor** with **SHAP** (SHapley Additive "
    "exPlanations). SHAP assigns each feature a signed contribution to a prediction: "
    "positive pushes the predicted score toward **Good (2)**, negative toward **Poor (0)**."
)

# 1. Global SHAP importance (mean |SHAP|), aggregated per original feature
si = meta.get("shap_importance", {})
si_df = pd.DataFrame({"feature": list(si.keys()), "mean_abs_shap": list(si.values())}) \
    .sort_values("mean_abs_shap", ascending=True)
st.subheader("1. Global feature importance (mean |SHAP|)")
fig = px.bar(si_df, x="mean_abs_shap", y="feature", orientation="h", text="mean_abs_shap",
             color="mean_abs_shap", color_continuous_scale="Purples")
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(coloraxis_showscale=False, xaxis_title="mean(|SHAP value|)",
                  yaxis_title="Feature")
st.plotly_chart(fig, use_container_width=True)
top3 = list(si_df.sort_values("mean_abs_shap", ascending=False)["feature"].head(3))
st.write(f"**Insight:** the biggest drivers are **{', '.join(top3)}**. One-hot categories are "
         "summed back to their original feature.")

# 2. SHAP beeswarm (direction of effect)
st.subheader("2. Direction of effect (SHAP summary)")
if BEESWARM.exists():
    st.image(str(BEESWARM), use_container_width=True)
    st.markdown("""
- Each dot is one tree; **red = high feature value**, **blue = low**.
- Dots to the **right** raise the predicted health score; to the **left** lower it.
- Readable patterns: more **visible problems** and being a **Norway maple** push the score
  **down**; larger **trunk diameter** and species like **honeylocust / London planetree**
  push it **up**.
""")
else:
    st.caption("Add visuals/14_shap_beeswarm.png to show the SHAP summary plot.")

# 3. Linear model coefficients as a complement
st.subheader("3. Cross-check: linear model coefficients")
lc = meta.get("linear_abs_coef", {})
if lc:
    lc_df = pd.DataFrame({"feature": list(lc.keys()), "abs_coefficient": list(lc.values())}) \
        .sort_values("abs_coefficient", ascending=True)
    fig = px.bar(lc_df, x="abs_coefficient", y="feature", orientation="h",
                 color="abs_coefficient", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="|coefficient| (standardized inputs)",
                      yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Insight:** the simple Linear Regression highlights similar top drivers, which "
             "gives us more confidence the signal is real and not a tree-model artifact.")

st.warning("SHAP shows **association, not causation**. A feature can look important because it "
           "correlates with something else (e.g. species correlates with typical size and "
           "planting location).")
