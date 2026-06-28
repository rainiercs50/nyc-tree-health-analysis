"""
Member 2 page: Model Prediction (two switchable regression models)
Place in your Streamlit app's pages/ folder as 3_Model_Prediction.py.

Predicts a continuous TREE HEALTH SCORE (Poor=0, Fair=1, Good=2). A lower score
means higher inspection priority.

Expected paths from app root:
    models/linear_regression.joblib
    models/random_forest_regressor.joblib
    models/model_metadata.json
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="NYC Street Tree Health | Prediction", layout="wide")
st.title("🤖 Health-Score Prediction")
st.caption("Two switchable regression models — modeling by Member 2")

MODELS_DIR = Path("models")
HEALTH_COLORS = {"Good": "#2E7D32", "Fair": "#F9A825", "Poor": "#C62828"}


@st.cache_resource
def load_metadata():
    return json.loads((MODELS_DIR / "model_metadata.json").read_text())


@st.cache_resource
def load_model(filename):
    return joblib.load(MODELS_DIR / filename)


try:
    meta = load_metadata()
except FileNotFoundError:
    st.error("Could not find models/model_metadata.json. Copy the Member 2 `models/` folder into the app root.")
    st.stop()


def score_to_band(s):
    return "Poor" if s < 0.5 else ("Fair" if s < 1.5 else "Good")


def dbh_to_group(d):
    return ("0-3 in sapling/small" if d <= 3 else "4-6 in young" if d <= 6
            else "7-12 in medium" if d <= 12 else "13-24 in mature" if d <= 24
            else "25+ in large")


# --------------------------------------------------------------- Model picker
keys = list(meta["models"].keys())
default_idx = keys.index(meta.get("recommended_model", keys[0]))
chosen_key = st.sidebar.radio("Choose a model", keys, index=default_idx,
                              format_func=lambda k: meta["models"][k]["label"])
chosen = meta["models"][chosen_key]
model = load_model(Path(chosen["file"]).name)
a, b = st.sidebar.columns(2)
a.metric("Test R²", f"{chosen['r2']:.3f}")
b.metric("Test RMSE", f"{chosen['rmse']:.3f}")
st.sidebar.caption("The score runs 0 (Poor) → 2 (Good). Switch models to compare the "
                   "linear baseline against the tuned Random Forest.")

# --------------------------------------------------------------------- Inputs
st.subheader("1. Describe the tree")
opts = meta["categorical_options"]
dbh_meta = meta["numeric_meta"]["tree_dbh"]
c1, c2, c3 = st.columns(3)
with c1:
    species = st.selectbox("Species", opts["species_top15_or_other"])
    borough = st.selectbox("Borough", opts["boroname"])
    curb = st.selectbox("Curb location", opts["curb_loc"])
with c2:
    dbh = st.slider("Trunk diameter (inches)", int(dbh_meta["min"]),
                    int(dbh_meta["max"]), int(dbh_meta["median"]))
    steward = st.selectbox("Stewardship signs", opts["steward"])
    guards = st.selectbox("Tree guard", opts["guards"])
with c3:
    sidewalk = st.selectbox("Sidewalk condition", opts["sidewalk"])
    pc_max = int(meta["numeric_meta"]["problem_count"]["max"])
    problem_count = st.slider("Visible root/trunk/branch problems", 0, pc_max, 0)
    st.metric("Derived diameter group", dbh_to_group(dbh))

row = {
    "tree_dbh": dbh, "problem_count": problem_count,
    "species_top15_or_other": species, "steward": steward, "guards": guards,
    "sidewalk": sidewalk, "has_problem": "Yes" if problem_count > 0 else "No",
    "boroname": borough, "dbh_group": dbh_to_group(dbh), "curb_loc": curb,
}
X = pd.DataFrame([row])[meta["numeric_features"] + meta["categorical_features"]]

# --------------------------------------------------------------------- Predict
st.subheader("2. Predicted health score")
score = float(model.predict(X)[0])
score = max(0.0, min(2.0, score))
band = score_to_band(score)

left, right = st.columns([1.2, 1])
with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"valueformat": ".2f"},
        title={"text": f"Predicted band: <b>{band}</b>"},
        gauge={
            "axis": {"range": [0, 2], "tickvals": [0, 1, 2],
                     "ticktext": ["Poor", "Fair", "Good"]},
            "bar": {"color": HEALTH_COLORS[band]},
            "steps": [
                {"range": [0, 0.5], "color": "#FCE4E4"},
                {"range": [0.5, 1.5], "color": "#FFF6DD"},
                {"range": [1.5, 2], "color": "#E5F3E6"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown(f"### <span style='color:{HEALTH_COLORS[band]}'>{band}</span>",
                unsafe_allow_html=True)
    st.write(f"Model: **{chosen['label']}**")
    st.write(f"Predicted score: **{score:.2f}** / 2.00")
    priority = "High" if score < 0.5 else ("Medium" if score < 1.5 else "Low")
    st.write(f"Suggested inspection priority: **{priority}** "
             "(lower score → higher priority).")

st.info("This is an educational triage/ranking tool, not an official inspection. The model "
        "learns associations from the 2015 census snapshot and cannot prove causes. The signal "
        "is weak (low R²), so use the score to *rank* trees, not as a precise health measurement.")
