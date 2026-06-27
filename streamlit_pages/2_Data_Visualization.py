"""
Member 1 page: Data Visualization
Place this file in your Streamlit app's pages/ folder as 2_Data_Visualization.py.
Expected path from app root: data/nyc_tree_member1_clean_sample.csv
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYC Street Tree Health | Data Visualization", layout="wide")
st.title("📊 Data Visualization")
st.caption("Exploratory analysis prepared by Member 1")

DATA_PATH = Path("data/nyc_tree_member1_clean_sample.csv")

@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find data/nyc_tree_member1_clean_sample.csv. Copy the Member 1 data folder into the app root.")
    st.stop()

health_order = ["Good", "Fair", "Poor"]
health_colors = {"Good":"#2E7D32", "Fair":"#F9A825", "Poor":"#C62828"}

st.sidebar.header("Filter data")
boroughs = st.sidebar.multiselect("Borough", sorted(df["boroname"].dropna().unique()), default=sorted(df["boroname"].dropna().unique()))
health = st.sidebar.multiselect("Health", health_order, default=health_order)
filtered = df[df["boroname"].isin(boroughs) & df["health"].isin(health)].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows shown", f"{len(filtered):,}")
c2.metric("Boroughs", filtered["boroname"].nunique())
c3.metric("Species", filtered["spc_common"].nunique())
c4.metric("Trees with problems", f"{(filtered['has_problem'].eq('Yes').mean() if len(filtered) else 0):.1%}")

st.subheader("1. Overall health distribution")
health_counts = filtered["health"].value_counts().reindex(health_order).dropna().reset_index()
health_counts.columns = ["health", "count"]
fig = px.bar(health_counts, x="health", y="count", color="health", color_discrete_map=health_colors, text="count")
fig.update_layout(showlegend=False, xaxis_title="Health", yaxis_title="Number of trees")
st.plotly_chart(fig, use_container_width=True)
st.write("**Insight:** Most sampled trees are labeled Good, so model evaluation should include macro-F1 and a confusion matrix instead of accuracy alone.")

st.subheader("2. Health mix by borough")
borough_health = filtered.groupby(["boroname", "health"]).size().reset_index(name="count")
if len(borough_health):
    borough_health["pct"] = borough_health.groupby("boroname")["count"].transform(lambda s: s / s.sum() * 100)
fig = px.bar(borough_health, x="boroname", y="pct", color="health", color_discrete_map=health_colors, category_orders={"health": health_order}, barmode="stack")
fig.update_layout(xaxis_title="Borough", yaxis_title="Percent of borough trees")
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Species patterns")
top_n = st.slider("Number of species to show", 5, 20, 12)
top_species = filtered["spc_common"].value_counts().head(top_n).reset_index()
top_species.columns = ["species", "count"]
fig = px.bar(top_species.sort_values("count"), x="count", y="species", orientation="h", title=f"Top {top_n} species in filtered data")
st.plotly_chart(fig, use_container_width=True)

if len(filtered):
    species_list = filtered["spc_common"].value_counts().head(top_n).index
    species_rate = filtered[filtered["spc_common"].isin(species_list)].assign(
        fair_or_poor=lambda x: x["health"].isin(["Fair", "Poor"])
    ).groupby("spc_common")["fair_or_poor"].mean().mul(100).reset_index()
    species_rate.columns = ["species", "fair_poor_pct"]
    fig = px.bar(species_rate.sort_values("fair_poor_pct"), x="fair_poor_pct", y="species", orientation="h", title="Fair/Poor rate among common species")
    fig.update_layout(xaxis_title="Fair or Poor (%)", yaxis_title="Species")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Visible problem status")
problem_health = filtered.groupby(["has_problem", "health"]).size().reset_index(name="count")
if len(problem_health):
    problem_health["pct"] = problem_health.groupby("has_problem")["count"].transform(lambda s: s / s.sum() * 100)
fig = px.bar(problem_health, x="has_problem", y="pct", color="health", color_discrete_map=health_colors, category_orders={"health": health_order}, barmode="stack")
fig.update_layout(xaxis_title="Any recorded root/trunk/branch problem?", yaxis_title="Percent")
st.plotly_chart(fig, use_container_width=True)
st.write("**Insight:** The problem fields are useful candidate predictors because they describe visible conditions a user can understand.")

st.subheader("5. Diameter group and map sample")
left, right = st.columns([1,1])
with left:
    dbh_health = filtered.groupby(["dbh_group", "health"]).size().reset_index(name="count")
    if len(dbh_health):
        dbh_health["pct"] = dbh_health.groupby("dbh_group")["count"].transform(lambda s: s / s.sum() * 100)
    fig = px.bar(dbh_health, x="dbh_group", y="pct", color="health", color_discrete_map=health_colors, category_orders={"health": health_order}, barmode="stack")
    fig.update_layout(xaxis_title="Diameter group", yaxis_title="Percent")
    st.plotly_chart(fig, use_container_width=True)
with right:
    map_df = filtered.dropna(subset=["latitude", "longitude"]).sample(min(2500, len(filtered)), random_state=42) if len(filtered) else filtered
    if len(map_df):
        st.map(map_df.rename(columns={"latitude":"lat", "longitude":"lon"})[["lat", "lon"]])
    else:
        st.warning("No rows available for the selected filters.")

st.markdown("""
### Data limitations for the presentation
- The target classes are imbalanced, with Good usually the largest class.
- The data is from the 2015 census, so it is not a live current-health system.
- Recorded problems are observational and may contain reporting or inspection bias.
- Latitude/longitude help visualization, but location alone should be used carefully to avoid overclaiming.
""")
