"""
NYC Street Tree Health Predictor — app entry point (Home).
Run from the repository root:

    streamlit run streamlit_app.py

Streamlit automatically loads the numbered pages in pages/ into the sidebar.
App assembly, README, deck, and demo by Member 3 (App & Presentation Lead);
data/EDA by Member 1; modeling/SHAP/W&B by Member 2.
"""
import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="NYC Street Tree Health Predictor",
    page_icon="🌳",
    layout="wide",
)

DATA = Path("data")
MODELS = Path("models")

st.title("🌳 NYC Street Tree Health Predictor")
st.markdown(
    "An interactive data-science app that estimates the **health of an NYC street tree** "
    "from visible features and ranks trees by **inspection priority** — built on the "
    "2015 NYC Street Tree Census."
)

# ---- Headline numbers (loaded from the team's saved artifacts) ----------------
@st.cache_data
def load_summary():
    try:
        return json.loads((DATA / "member1_summary_stats.json").read_text())
    except FileNotFoundError:
        return {}

@st.cache_data
def load_meta():
    try:
        return json.loads((MODELS / "model_metadata.json").read_text())
    except FileNotFoundError:
        return {}

summary = load_summary()
meta = load_meta()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trees analyzed (sample)", f"{summary.get('rows', 0):,}")
c2.metric("Boroughs", len(summary.get("borough_counts", {})) or "—")
good_pct = summary.get("target_pct", {}).get("Good")
c2_label = f"{good_pct:.0f}%" if good_pct else "—"
c3.metric("Labeled 'Good'", c2_label)
rec = meta.get("models", {}).get(meta.get("recommended_model", ""), {})
c4.metric("Final model R²", f"{rec.get('r2', float('nan')):.3f}" if rec else "—")

st.divider()

# ---- How the app reads -------------------------------------------------------
left, right = st.columns([1.3, 1])
with left:
    st.subheader("What this app does")
    st.markdown(
        """
The app turns the team's data-science workflow into something a grader (or a city
team) can explore in a few minutes:

1. **Introduction** — project framing, dataset, and why tree health matters.
2. **Data Visualization** — health distribution, borough and species patterns,
   problems vs. health, and a map sample *(Member 1)*.
3. **Health-Score Prediction** — enter a tree's visible features and get a predicted
   health score with an inspection-priority band; switch between two models *(Member 2)*.
4. **Explainable AI (SHAP)** — what drives the predictions *(Member 2)*.
5. **Hyperparameter Tuning (W&B)** — how the final model was selected *(Member 2)*.
6. **Conclusion** — interactive health map, impact, limitations, and next steps.

Use the sidebar to move between pages.
        """
    )
with right:
    st.subheader("How to read the prediction")
    st.markdown(
        """
The model predicts a **continuous health score** on a 0–2 scale:

| Score | Band | Inspection priority |
|------:|:-----|:--------------------|
| ~0 | Poor | **High** |
| ~1 | Fair | Medium |
| ~2 | Good | Low |

We frame the project as **regression → triage ranking** rather than three fixed
buckets. A single score lets the city *rank* trees by how much attention they may
need, and it satisfies the course's required linear-regression model. The original
proposal framed this as classification; the team refined it to regression during
modeling, and every page reflects the score-based framing.
        """
    )

st.divider()
st.subheader("Honest framing")
st.info(
    "This is an **educational triage and learning tool**, not an official inspection "
    "system. The model learns associations from a 2015 census snapshot and cannot prove "
    "causes. The predictive signal is weak (low R²), so the score should be used to "
    "**rank** trees for a human to review — not as a precise measurement of any one tree's health."
)

st.caption(
    "Data & EDA: Member 1 · Modeling, SHAP & W&B: Member 2 · "
    "App assembly, README, deck & demo: Member 3"
)
