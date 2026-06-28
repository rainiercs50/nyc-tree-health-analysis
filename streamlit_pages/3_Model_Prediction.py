"""
Member 2 page: Model Prediction (multiple selectable models)
Place this file in your Streamlit app's pages/ folder as 3_Model_Prediction.py.

Expected paths from app root:
    models/logistic_regression.joblib
    models/random_forest.joblib
    models/model_metadata.json
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

st.set_page_config(page_title="NYC Street Tree Health | Model Prediction", layout="wide")
st.title("🤖 Model Prediction")
st.caption("Interactive Good / Fair / Poor health prediction — modeling by Member 2")

MODELS_DIR = Path("models")
META_PATH = MODELS_DIR / "model_metadata.json"
HEALTH_ORDER = ["Good", "Fair", "Poor"]
HEALTH_COLORS = {"Good": "#2E7D32", "Fair": "#F9A825", "Poor": "#C62828"}


@st.cache_resource
def load_metadata():
    return json.loads(META_PATH.read_text())


@st.cache_resource
def load_model(filename):
    return joblib.load(MODELS_DIR / filename)


try:
    meta = load_metadata()
except FileNotFoundError:
    st.error("Could not find models/model_metadata.json. Copy the Member 2 `models/` folder into the app root.")
    st.stop()

# ----------------------------------------------------------------- Model picker
model_keys = list(meta["models"].keys())
labels = {k: meta["models"][k]["label"] for k in model_keys}
default_idx = model_keys.index(meta.get("recommended_model", model_keys[0]))
chosen_key = st.sidebar.radio(
    "Choose a model", model_keys, index=default_idx, format_func=lambda k: labels[k]
)
chosen = meta["models"][chosen_key]
model = load_model(Path(chosen["file"]).name)

m1, m2 = st.sidebar.columns(2)
m1.metric("Test accuracy", f"{chosen['accuracy']:.2f}")
m2.metric("Macro-F1", f"{chosen['macro_f1']:.2f}")
st.sidebar.caption(
    "Random Forest is the recommended model. Macro-F1 is reported alongside accuracy "
    "because the classes are imbalanced (most trees are Good)."
)


def dbh_to_group(dbh: int) -> str:
    if dbh <= 3:
        return "0-3 in sapling/small"
    if dbh <= 6:
        return "4-6 in young"
    if dbh <= 12:
        return "7-12 in medium"
    if dbh <= 24:
        return "13-24 in mature"
    return "25+ in large"


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
    problem_count = st.slider("Number of visible root/trunk/branch problems", 0, pc_max, 0)
    st.metric("Derived diameter group", dbh_to_group(dbh))

row = {
    "tree_dbh": dbh,
    "problem_count": problem_count,
    "species_top15_or_other": species,
    "steward": steward,
    "guards": guards,
    "sidewalk": sidewalk,
    "has_problem": "Yes" if problem_count > 0 else "No",
    "boroname": borough,
    "dbh_group": dbh_to_group(dbh),
    "curb_loc": curb,
}
X = pd.DataFrame([row])[meta["numeric_features"] + meta["categorical_features"]]

# --------------------------------------------------------------------- Predict
st.subheader("2. Prediction")
pred = model.predict(X)[0]
proba = dict(zip(list(model.classes_), model.predict_proba(X)[0]))
prob_df = pd.DataFrame({"health": HEALTH_ORDER,
                        "probability": [proba.get(c, 0.0) for c in HEALTH_ORDER]})

left, right = st.columns([1, 2])
with left:
    st.markdown(f"### Predicted health")
    st.markdown(
        f"<h1 style='color:{HEALTH_COLORS.get(pred, '#333')};margin-top:-10px'>{pred}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model: {chosen['label']}")
with right:
    fig = px.bar(prob_df, x="health", y="probability", color="health",
                 color_discrete_map=HEALTH_COLORS, text="probability",
                 category_orders={"health": HEALTH_ORDER})
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Predicted probability",
                      yaxis_range=[0, 1], xaxis_title="Health class")
    st.plotly_chart(fig, use_container_width=True)

st.info(
    "This is an educational triage tool, not an official inspection. The model learns "
    "associations from the 2015 census snapshot and cannot prove what causes tree health "
    "problems. Switch models in the sidebar to compare the baseline and the tuned Random Forest."
)
