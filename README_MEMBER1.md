# Member 1 Data + EDA Package

Project: **NYC Street Tree Health Predictor**
Role: **Member 1 — Data and EDA Lead**
Date: 2026-06-27

## What this package contains

- `data/nyc_tree_member1_clean_sample.csv` — cleaned/feature-engineered EDA dataset.
- `data/nyc_tree_member1_model_ready_sample.csv` — smaller model handoff dataset for Member 2.
- `data/member1_data_dictionary.csv` — column meanings and how each field should be used.
- `data/member1_summary_stats.json` — quick counts and summary statistics for slides/app metrics.
- `visuals/*.png` — ready-to-use EDA charts for the app, report, or presentation.
- `streamlit_pages/1_Introduction.py` — Streamlit page 1.
- `streamlit_pages/2_Data_Visualization.py` — Streamlit page 2.
- `scripts/create_member1_package.py` — reproducible script that generated the files.
- `member1_handoff.md` and `member1_handoff.pdf` — written handoff for teammates.

## Member 1 responsibilities covered

1. Loaded the prior NYC Street Tree sample.
2. Cleaned missing target/core feature values.
3. Standardized missing categorical values to `Unknown`.
4. Engineered practical features for modeling and app filters.
5. Created EDA visuals aligned with the final app's Introduction and Data Visualization pages.
6. Wrote limitation notes and handoff instructions for the modeling/app teammates.

## Key findings

- The cleaned sample contains 49,994 usable tree records after dropping missing target/core fields.
- Health distribution: Good 75.57%, Fair 18.81%, Poor 5.62%.
- The largest species groups are: London planetree, honeylocust, Callery pear, pin oak, Norway maple.
- Engineered features added for app/modeling: problem_count, has_problem, dbh_group, species_top15_or_other, and binary root/trunk/branch problem flags.
- Recommended modeling note: because the target is imbalanced, compare models with macro-F1 plus a confusion matrix, not accuracy alone.

## How teammates should use this

### Member 2 — Modeling
Use `data/nyc_tree_member1_model_ready_sample.csv`. Recommended first feature set:

- Numeric: `tree_dbh`, `problem_count`
- Categorical: `species_top15_or_other`, `steward`, `guards`, `sidewalk`, `has_problem`, `boroname`, `dbh_group`, `curb_loc`
- Target: `health`

Compare at least two models and include macro-F1, accuracy, confusion matrix, and feature importance/explainability.

### Member 3 — App + Deployment
Copy the `data/` folder into the Streamlit app root and copy `streamlit_pages/*.py` into the app's `pages/` folder. The final app should still include the other required pages:

1. Introduction
2. Data visualization
3. Multiple selectable models
4. Feature importance / explainability
5. Hyperparameter tuning / experiment tracking
6. Conclusion

Deploy the final Streamlit app to Hugging Face Spaces.
