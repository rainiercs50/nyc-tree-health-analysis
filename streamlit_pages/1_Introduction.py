"""
Member 1 page: Introduction
Place this file in your Streamlit app's pages/ folder as 1_Introduction.py.
"""
import streamlit as st

st.set_page_config(page_title="NYC Street Tree Health | Introduction", layout="wide")
st.title("🌳 NYC Street Tree Health Predictor")
st.subheader("Final Project Introduction")

st.markdown("""
### Project goal
This app predicts whether an NYC street tree is likely to be in **Good**, **Fair**, or **Poor** health using visible tree and location features from the 2015 NYC Street Tree Census.

### Why this matters
Street trees improve air quality, provide shade, lower heat risk, and make neighborhoods more livable. A prediction app cannot replace professional tree inspection, but it can help users explore which visible conditions are associated with tree health and support better prioritization conversations.

### Dataset
- **Dataset:** NYC Open Data — 2015 Street Tree Census: Tree Data
- **Unit of analysis:** one street tree record
- **Target variable:** `health` with classes Good, Fair, and Poor
- **Example features:** tree diameter, species, borough, stewardship, guards, sidewalk damage, and recorded root/trunk/branch problems

### Final app structure
Use this page as page 1 of the final app. The complete app should include:
1. Introduction
2. Data visualization
3. Multiple selectable models
4. Feature importance / explainability
5. Hyperparameter tuning / experiment tracking
6. Conclusion
""")

col1, col2, col3 = st.columns(3)
col1.metric("Target", "Tree health")
col2.metric("Classes", "Good / Fair / Poor")
col3.metric("Deployment", "Hugging Face Space")

st.info("Member 1 owns the project framing, cleaned dataset, data dictionary, EDA visuals, and data limitations. Member 2 can use the model-ready CSV for modeling, and Member 3 can plug these pages into the final Streamlit app.")

st.markdown("""
### Main research question
Can we predict a street tree's health category from observable tree characteristics and location-related features?

### Important limitation
This is an observational dataset from a single census snapshot. The model can identify associations, but it should **not** be interpreted as proving what causes tree health problems.
""")
