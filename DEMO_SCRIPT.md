# Demo Script — NYC Street Tree Health Predictor

Target length: **6–8 minutes** live, three presenters. Each member narrates their own
work. Run `streamlit run streamlit_app.py` before you start and leave it on the **Home**
page. Have a backup screen recording or screenshots in case Wi-Fi/deploy fails.

**Roles in the demo**
- **M1** — Data & EDA
- **M2** — Modeling
- **M3** — App, story, driver (controls the screen)

---

## 0. Setup checklist (before the room sees anything)
- [ ] App running at `http://localhost:8501`, on the **Home** page.
- [ ] Browser zoom ~100%, sidebar expanded.
- [ ] `Health-Score Prediction` page pre-loaded once so the model is cached (fast first click).
- [ ] Backup recording open in a second tab.

---

## 1. Hook & framing — M3 *(~45s, Home page)*
> "NYC has hundreds of thousands of street trees and far fewer inspectors. Our app uses
> the 2015 Street Tree Census to estimate a tree's health from features anyone can see and
> **rank trees by inspection priority**.
>
> One thing up front: the proposal framed this as classifying trees as Good, Fair, or Poor.
> When we modeled it, we refined it to **regression** — we predict a single health *score*
> from 0 to 2. A score is more useful for triage because it lets the city rank trees instead
> of sorting them into three bins, and it satisfies the required linear-regression model."

*Point to the score table on the Home page, then hand to M1.*

---

## 2. The data & what we found — M1 *(~90s, page 2: Data Visualization)*
> "We started from the NYC Open Data census — about 683,000 trees — and cleaned a
> ~50,000-row sample. Here's the story the data tells."

- **Health distribution:** "About 76% of trees are *Good* — that imbalance matters, and it's
  why we don't trust accuracy alone."
- **By borough / species:** *(use the species slider)* "Health varies by borough and species —
  some common species have noticeably higher Fair/Poor rates."
- **Problems vs. health:** "Trees with recorded root/trunk/branch problems skew less healthy —
  these are visible features a user can actually report."
- **Map sample:** "And we can place trees geographically." *Hand to M2.*

---

## 3. The model — M2 *(~2 min, page 3 → 4 → 5)*

**Page 3 — Prediction (the centerpiece):**
> "Let's predict a real tree." *Fill the inputs live — e.g. London planetree, Brooklyn,
> diameter 6 inches, a couple of problems.*
> "The gauge shows the predicted score and band, and a suggested inspection priority. I can
> switch from the **linear baseline** to the **tuned Random Forest** in the sidebar and the
> prediction updates."

> "Honest numbers: the Random Forest reaches an R² of about 0.09. That's a **weak signal** —
> visible features alone don't determine health — so we present this as a *ranking* tool, not
> a precise measurement."

**Page 4 — SHAP:** *(~20s)* "SHAP tells us *why*: species, trunk diameter, and borough drive
the predictions the most."

**Page 5 — W&B tuning:** *(~20s)* "We tracked every Random Forest configuration in Weights &
Biases and picked the best by cross-validated R²." *Hand to M3.*

---

## 4. Conclusion, impact & honesty — M3 *(~60s, page 6: Conclusion)*
> "Putting it together: here's an interactive map colored by health" *(toggle a borough)* —
> "and our honest framing. The impact is **triage and awareness**: help prioritize tree care
> and start public conversations. The limitations are real — weak signal, class imbalance, a
> 2015 snapshot, and correlation not causation. So this is a **learning and triage tool, not
> an official inspection system**."

---

## 5. Wrap & next steps — M3 *(~30s)*
> "Next we'd join newer 311 and health data, add neighborhood features like heat and traffic,
> and deploy it publicly so residents and city teams can try it. Everything — notebooks,
> models, and this app — is in our GitHub repo. Happy to take questions."

---

## Anticipated questions (quick answers)
- **"Why is R² so low?"** — Visible census features carry limited signal for health; we're
  honest about it and use the model to rank, not to diagnose. The value is the workflow and
  the triage framing.
- **"Why regression instead of classification?"** — A continuous score ranks trees by priority
  (more actionable than three buckets) and lets us use the required linear-regression model.
- **"Isn't accuracy ~74%?"** — That's mostly the *Good* majority class. We report RMSE/MAE and
  a mapped confusion matrix so we don't overstate performance.
- **"Could this make maintenance decisions?"** — No. It's a triage aid for humans, built on a
  2015 snapshot; it shows associations, not causes.

## Timing summary
| Segment | Who | Time |
|---|---|---|
| Hook & framing | M3 | 0:45 |
| Data & EDA | M1 | 1:30 |
| Model + SHAP + W&B | M2 | 2:00 |
| Conclusion & impact | M3 | 1:00 |
| Wrap & next steps | M3 | 0:30 |
| **Total** | | **~5:45** (+ Q&A) |
