# 🌳 NYC Street Tree Health Predictor

An interactive data-science app that estimates the health of an NYC street tree from
visible features and **ranks trees by inspection priority**, built on the
[2015 NYC Street Tree Census](https://data.cityofnewyork.us/d/uvpi-gqnh).

Final capstone for *Principles of Data Science I*. The project combines data cleaning,
exploratory analysis, regression modeling, explainability, experiment tracking, and a
multipage Streamlit app into one deliverable with a real-world, positive-impact framing.

---

## What it does

A user enters a tree's visible characteristics — species, borough, trunk diameter,
stewardship, guards, sidewalk condition, and recorded problems — and the app returns a
**predicted health score** on a 0–2 scale together with an inspection-priority band:

| Score | Band | Inspection priority |
|------:|:-----|:--------------------|
| ~0 | Poor | **High** |
| ~1 | Fair | Medium |
| ~2 | Good | Low |

### A note on framing: classification → regression
The original proposal framed this as a **classification** problem (predict Good / Fair /
Poor). During modeling the team refined it to **regression**: predict a single continuous
*health score* (`Poor=0, Fair=1, Good=2`). A score is more actionable for triage — it
lets the city *rank* trees by how much attention they may need rather than sorting them
into three fixed buckets — and it satisfies the course's required linear-regression model.
Every page in the app reflects this score-based framing.

---

## Quick start

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app from the repository root
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Streamlit automatically loads the numbered
pages in `pages/` into the sidebar.

> **scikit-learn version matters.** The saved `.joblib` models were trained on
> **scikit-learn 1.7.2** (pinned in `requirements.txt`). Loading them under a different
> version may warn or fail.

---

## App pages

| # | Page | Owner | What it shows |
|---|------|-------|---------------|
| — | **Home** (`streamlit_app.py`) | Member 3 | Project overview, how to read the score, honest framing |
| 1 | Introduction | Member 1 | Project goal, dataset, why tree health matters |
| 2 | Data Visualization | Member 1 | Health distribution, borough/species patterns, problems vs. health, map sample |
| 3 | Health-Score Prediction | Member 2 | Enter a tree → predicted score + priority; switch between two models |
| 4 | Explainable AI (SHAP) | Member 2 | What drives the predictions |
| 5 | Hyperparameter Tuning (W&B) | Member 2 | How the final model was selected |
| 6 | Conclusion | Member 3 | Interactive health map, impact, limitations, next steps |

---

## Repository structure

```
nyc-tree-health-analysis/
├── streamlit_app.py            # App entry point (Home) — run this
├── pages/                      # Multipage app (1..6, auto-loaded by Streamlit)
├── streamlit_pages/            # Original per-member page sources (reference copies)
├── data/                       # Cleaned samples, data dictionary, metrics, W&B logs
├── models/                     # Trained .joblib models + model_metadata.json
├── visuals/                    # EDA, model, and SHAP figures (.png)
├── scripts/                    # Member 2 modeling / tuning / packaging scripts
├── member1_eda_notebook.ipynb  # Member 1: data + EDA
├── member2_modeling_notebook.ipynb  # Member 2: modeling, SHAP, W&B
├── requirements.txt
├── DEMO_SCRIPT.md              # Presentation walkthrough
├── presentation/               # Final slide deck (.pptx)
└── README.md
```

---

## Dataset

- **Source:** NYC Open Data — 2015 Street Tree Census: Tree Data
  (<https://data.cityofnewyork.us/d/uvpi-gqnh>), ~683,000 records.
- **Sample used:** ~50,000 cleaned rows (`data/nyc_tree_member1_clean_sample.csv`) for a
  fast, responsive app; the model-ready sample is `nyc_tree_member1_model_ready_sample.csv`.
- **Target:** `health` (Good / Fair / Poor), mapped to a `health_score` (0–2) for regression.
- **Features:** `tree_dbh`, `spc_common`, `steward`, `guards`, `sidewalk`, `problems`,
  `boroname`, `latitude`, `longitude`, plus engineered `problem_count`, `has_problem`,
  `dbh_group`, and `species_top15_or_other`.

---

## Models & results (held-out test set)

| Model | R² | RMSE | MAE | Mapped accuracy\* |
|-------|----:|-----:|----:|-----------------:|
| Linear Regression (baseline) | 0.059 | 0.551 | 0.425 | 0.739 |
| **Random Forest Regressor (tuned)** | **0.087** | **0.543** | **0.416** | 0.738 |

\* Mapped accuracy rounds the predicted score to the nearest band and compares it to the
true category. Because ~76% of trees are labeled *Good*, accuracy looks high even for a
weak model, so we rely on RMSE/MAE and a mapped confusion matrix.

**Top drivers** (Random Forest feature importance): species, trunk diameter, borough,
and diameter group. The Random Forest is explained with **SHAP**, and tuning was tracked
with **Weights & Biases**.

---

## Limitations

- **Weak signal:** the best model explains only a small share of variation (R² ≈ 0.09).
  Use the score to *rank* trees, not to measure any single tree's health.
- **Class imbalance:** *Good* dominates; report RMSE/MAE and confusion, not accuracy alone.
- **Snapshot data:** 2015 census, not a live health feed.
- **Correlation, not causation:** recorded problems are observational and may carry bias.

**This is an educational triage and learning tool, not an official inspection system.**

---

## Team

| Member | Role | Deliverables |
|--------|------|--------------|
| Member 1 | Data & EDA Lead | Cleaned dataset, data dictionary, EDA visuals, limitations |
| Member 2 | Modeling & Evaluation Lead | Baseline + tuned models, SHAP, W&B tuning, metrics |
| Member 3 | App, Repository & Presentation Lead | Multipage app assembly, README, slide deck, demo script |

## License & sources

For educational use. Tree data © NYC Open Data (public domain). Built with
[Streamlit](https://streamlit.io/), scikit-learn, Plotly, SHAP, and Weights & Biases.
