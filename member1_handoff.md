# Member 1 Handoff: Data Preparation and EDA

**Project:** NYC Street Tree Health Predictor  
**Course:** Intro to Data Science Final Project  
**Member 1 role:** Data and EDA Lead  
**Date:** 2026-06-27

## 1. What Member 1 completed

Member 1 prepared the data foundation for the final Streamlit app. This includes a cleaned sample dataset, a model-ready dataset for Member 2, a data dictionary, reusable exploratory charts, and two Streamlit pages: Introduction and Data Visualization.

## 2. Dataset used

- **Source:** NYC Open Data, 2015 Street Tree Census — Tree Data
- **Target variable:** `health` with classes Good, Fair, and Poor
- **Rows after cleaning:** 49,994
- **Main feature types:** species, borough, tree diameter, stewardship, guards, sidewalk condition, and root/trunk/branch problem fields

This dataset is different from the midterm dataset and fits the final-project requirement for a Streamlit app with models, visualization, explainability, tuning/experiment tracking, and deployment on Hugging Face.

## 3. Cleaning steps

- Kept the core project columns needed for EDA, modeling, and app display.
- Dropped rows missing the target `health` or core fields needed for a usable tree record.
- Standardized blank categorical values to `Unknown` so the Streamlit app does not break on missing values.
- Filtered unrealistic/extreme `tree_dbh` values outside 0–100 inches.
- Converted root/trunk/branch problem fields into binary flags.

## 4. Feature engineering completed

| Feature | Purpose |
|---|---|
| `problem_count` | Counts how many root/trunk/branch problem flags are marked yes. |
| `has_problem` | Simple Yes/No field for whether any problem exists. |
| `dbh_group` | Groups tree diameter into easier-to-read size bands for charts and app filters. |
| `species_top15_or_other` | Keeps common species manageable for modeling and dropdowns. |
| `*_flag` problem fields | Binary predictors for visible tree problems. |

## 5. EDA visuals created

The `visuals/` folder contains seven ready-to-use charts:

1. Health distribution
2. Health mix by borough
3. Top species
4. Fair/Poor health rate among top species
5. Health by visible problem status
6. Health by tree diameter group
7. Map sample by health

## 6. Main EDA findings to say during the presentation

- The cleaned sample contains 49,994 usable tree records after dropping missing target/core fields.
- Health distribution: Good 75.57%, Fair 18.81%, Poor 5.62%.
- The largest species groups are: London planetree, honeylocust, Callery pear, pin oak, Norway maple.
- Engineered features added for app/modeling: problem_count, has_problem, dbh_group, species_top15_or_other, and binary root/trunk/branch problem flags.
- Recommended modeling note: because the target is imbalanced, compare models with macro-F1 plus a confusion matrix, not accuracy alone.

## 7. Handoff to Member 2

Use `data/nyc_tree_member1_model_ready_sample.csv` for modeling. Start with these columns:

- Target: `health`
- Numeric features: `tree_dbh`, `problem_count`
- Categorical features: `species_top15_or_other`, `steward`, `guards`, `sidewalk`, `has_problem`, `boroname`, `dbh_group`, `curb_loc`

Recommended model comparison:

- Baseline: Logistic Regression or Decision Tree
- Improved model: Random Forest, Gradient Boosting, or XGBoost/LightGBM if available
- Metrics: accuracy, macro-F1, confusion matrix, class-level precision/recall

## 8. Handoff to Member 3

Copy these files into the final Streamlit repository:

- `data/nyc_tree_member1_clean_sample.csv`
- `data/member1_data_dictionary.csv`
- `streamlit_pages/1_Introduction.py`
- `streamlit_pages/2_Data_Visualization.py`

The Data Visualization page expects the cleaned CSV at `data/nyc_tree_member1_clean_sample.csv`.

## 9. Limitations to include

- The target classes are imbalanced, so accuracy alone can be misleading.
- The data is from 2015 and should not be presented as live/current tree health.
- The model will show associations, not proven causes.
- Some visible problem fields may reflect reporting/inspection bias.
- The app should be framed as an educational/triage tool, not an official maintenance decision system.

## 10. Suggested 8-minute presentation timing for Member 1

- 0:00–0:45 — Problem and why tree health matters
- 0:45–1:30 — Dataset and target variable
- 1:30–2:45 — Cleaning and feature engineering
- 2:45–4:00 — EDA charts and key insights
- 4:00 onward — Hand off to modeling, explainability, tuning, and app demo sections
