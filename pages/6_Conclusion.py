"""
Member 3 page: Conclusion — interactive health map, impact, limitations, next steps.
Place in pages/ as 6_Conclusion.py.

Expected paths from app root:
    data/nyc_tree_member1_clean_sample.csv
    models/model_metadata.json
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYC Street Tree Health | Conclusion", layout="wide")
st.title("✅ Conclusion & Impact")
st.caption("Wrap-up, interactive map, and honest limitations — App & Presentation by Member 3")

DATA_PATH = Path("data/nyc_tree_member1_clean_sample.csv")
META_PATH = Path("models/model_metadata.json")

HEALTH_ORDER = ["Good", "Fair", "Poor"]
HEALTH_COLORS = {"Good": "#2E7D32", "Fair": "#F9A825", "Poor": "#C62828"}


@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)


@st.cache_data
def load_meta(path=META_PATH):
    return json.loads(path.read_text())


# ------------------------------------------------------------------ Interactive map
st.subheader("1. Where the trees are — colored by health")
st.markdown(
    "A sample of street trees plotted on NYC and colored by recorded health. "
    "This is the map view the proposal asked for: a quick visual of how health "
    "is distributed across the city."
)

try:
    df = load_data()
    geo = df.dropna(subset=["latitude", "longitude", "health"]).copy()

    colA, colB = st.columns([1, 1])
    with colA:
        boroughs = st.multiselect(
            "Borough", sorted(geo["boroname"].dropna().unique()),
            default=sorted(geo["boroname"].dropna().unique()),
        )
    with colB:
        shown = st.multiselect("Health", HEALTH_ORDER, default=HEALTH_ORDER)

    geo = geo[geo["boroname"].isin(boroughs) & geo["health"].isin(shown)]
    sample_n = min(4000, len(geo))
    geo = geo.sample(sample_n, random_state=42) if len(geo) else geo

    if len(geo):
        fig = px.scatter_mapbox(
            geo, lat="latitude", lon="longitude", color="health",
            color_discrete_map=HEALTH_COLORS,
            category_orders={"health": HEALTH_ORDER},
            hover_data={"spc_common": True, "boroname": True,
                        "latitude": False, "longitude": False},
            zoom=9, height=520, opacity=0.6,
        )
        fig.update_layout(mapbox_style="carto-positron",
                          margin=dict(l=0, r=0, t=0, b=0),
                          legend_title_text="Health")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing {sample_n:,} sampled trees. Good trees dominate the map, "
                   "which matches the class imbalance reported in the data.")
    else:
        st.warning("No trees match the current filters.")
except FileNotFoundError:
    st.error("Could not find data/nyc_tree_member1_clean_sample.csv. "
             "Run the app from the repository root.")

st.divider()

# ------------------------------------------------------------------ Impact
st.subheader("2. Positive impact")
st.markdown(
    """
Street trees cool neighborhoods, clean the air, and make blocks more livable, but the
city has far more trees than inspectors. This app turns the 2015 census into a simple
**triage signal**: enter a tree's visible features and get a health score that ranks it
by how much attention it may need. It is meant to support **prioritization conversations
and public awareness**, not to replace a professional arborist.
    """
)

# ------------------------------------------------------------------ What we built
st.subheader("3. What the team built")
c1, c2, c3 = st.columns(3)
c1.markdown("**Data & EDA** *(Member 1)*\n\nCleaned ~50k-tree sample, data dictionary, "
            "six EDA visuals, documented limitations.")
c2.markdown("**Modeling** *(Member 2)*\n\nLinear-regression baseline vs. tuned Random "
            "Forest, SHAP explainability, W&B-tracked tuning.")
c3.markdown("**App & story** *(Member 3)*\n\nMultipage Streamlit app, README, slide deck, "
            "and demo script tying it together.")

st.divider()

# ------------------------------------------------------------------ Limitations
st.subheader("4. Honest limitations")
st.markdown(
    """
- **Weak signal.** The best model explains only a small share of variation in health
  (R² ≈ 0.09). Visible features alone do not determine tree health, so the score is a
  *ranking aid*, not a measurement.
- **Class imbalance.** About 76% of trees are labeled *Good*, so accuracy looks high
  even for a weak model — we report RMSE/MAE and a mapped confusion matrix instead.
- **Snapshot data.** The 2015 census is a one-time snapshot, not a live health feed.
- **Correlation, not causation.** Recorded problems are observational and may carry
  inspection or reporting bias; the model cannot prove what *causes* poor health.
    """
)

# ------------------------------------------------------------------ Next steps
st.subheader("5. Next steps")
st.markdown(
    """
- Join newer tree-health or 311 service-request data to test whether the patterns hold
  over time.
- Add neighborhood-level features (heat, traffic, soil) that may carry more signal than
  visible tree attributes alone.
- Deploy publicly (e.g., a Hugging Face Space or Streamlit Community Cloud) so residents
  and city teams can try it.
    """
)

try:
    meta = load_meta()
    rec = meta["models"][meta["recommended_model"]]
    st.success(
        f"**Final model:** {rec['label']} · R² = {rec['r2']:.3f} · "
        f"RMSE = {rec['rmse']:.3f}. Used as a triage ranking tool, not an official inspection."
    )
except (FileNotFoundError, KeyError):
    pass

st.caption("Thanks for exploring the NYC Street Tree Health Predictor.")
