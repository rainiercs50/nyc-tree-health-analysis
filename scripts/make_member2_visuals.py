"""
Generate Member 2 regression + SHAP visuals from saved artifacts.

Run after create_member2_package.py:
    python scripts/make_member2_visuals.py

Reads:  data/member2_model_metrics.json, models/model_metadata.json,
        models/eval_arrays.npz, models/shap_sample.npz
Writes: visuals/08_model_comparison.png .. visuals/14_shap_beeswarm.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "visuals"; VIS.mkdir(exist_ok=True)
CLASS_ORDER = ["Good", "Fair", "Poor"]

metrics = json.loads((ROOT / "data" / "member2_model_metrics.json").read_text())
meta = json.loads((ROOT / "models" / "model_metadata.json").read_text())
lr = metrics["linear_regression"]
rf = metrics["random_forest_regressor"]
ev = np.load(ROOT / "models" / "eval_arrays.npz", allow_pickle=True)

# 08 model comparison: R2 (higher better) + RMSE (lower better)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
labels = ["Linear Reg.\n(baseline)", "Random Forest\n(tuned)"]
axes[0].bar(labels, [lr["r2"], rf["r2"]], color=["#90A4AE", "#2E7D32"])
axes[0].set_title("R²  (higher is better)"); axes[0].set_ylabel("R²")
for i, v in enumerate([lr["r2"], rf["r2"]]):
    axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
axes[1].bar(labels, [lr["rmse"], rf["rmse"]], color=["#90A4AE", "#1565C0"])
axes[1].set_title("RMSE  (lower is better)"); axes[1].set_ylabel("RMSE")
for i, v in enumerate([lr["rmse"], rf["rmse"]]):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
fig.suptitle("Model comparison — predicting tree health score (0=Poor … 2=Good)")
plt.tight_layout(); plt.savefig(VIS / "08_model_comparison.png", dpi=120); plt.close()
print("08 ok", flush=True)

# 09 predicted vs actual (RF) — true is 0/1/2 so jitter the x axis
rng = np.random.default_rng(0)
yt = ev["y_true"].astype(float); yp = ev["rf_pred"].astype(float)
jit = yt + rng.uniform(-0.15, 0.15, size=len(yt))
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(jit, yp, s=6, alpha=0.15, color="#2E7D32")
ax.plot([0, 2], [0, 2], "r--", lw=1, label="perfect prediction")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Poor (0)", "Fair (1)", "Good (2)"])
ax.set_xlabel("Actual health score"); ax.set_ylabel("Predicted score")
ax.set_title("Random Forest — predicted vs actual"); ax.legend()
plt.tight_layout(); plt.savefig(VIS / "09_predicted_vs_actual_rf.png", dpi=120); plt.close()
print("09 ok", flush=True)

# 10 residuals (RF)
resid = yp - yt
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(resid, bins=40, color="#1565C0", alpha=0.85)
ax.axvline(0, color="red", ls="--", lw=1)
ax.set_xlabel("Residual (predicted − actual)"); ax.set_ylabel("Count")
ax.set_title(f"Random Forest residuals (mean={resid.mean():.3f})")
plt.tight_layout(); plt.savefig(VIS / "10_residuals_rf.png", dpi=120); plt.close()
print("10 ok", flush=True)

# 11 mapped confusion matrix (RF)
cm = np.array(rf["mapped_confusion"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_ORDER)
ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_ORDER)
ax.set_xlabel("Predicted band"); ax.set_ylabel("Actual")
ax.set_title("RF score mapped to Good/Fair/Poor")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(VIS / "11_confusion_mapped_rf.png", dpi=120); plt.close()
print("11 ok", flush=True)

# 12 tuning results (CV R2)
tr = sorted(metrics["tuning_results"], key=lambda r: r["mean_test_r2"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(range(len(tr)), [r["mean_test_r2"] for r in tr],
        xerr=[r["std_test_r2"] for r in tr], color="#1565C0", alpha=0.85)
ax.set_yticks(range(len(tr)))
ax.set_yticklabels([f"depth={r['params']['max_depth']}, leaf={r['params']['min_samples_leaf']}"
                    for r in tr], fontsize=9)
ax.axvline(metrics["cv_best_r2"], color="#C62828", ls="--", lw=1,
           label=f"best={metrics['cv_best_r2']:.3f}")
ax.set_xlabel("CV mean R² (3-fold)"); ax.set_title("Random Forest hyperparameter search")
ax.legend(); plt.tight_layout(); plt.savefig(VIS / "12_tuning_results.png", dpi=120); plt.close()
print("12 ok", flush=True)

# 13 SHAP aggregated bar (mean |SHAP| per original feature)
si = meta["shap_importance"]
names = list(si.keys())[::-1]; vals = [si[k] for k in names]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(names, vals, color="#6A1B9A")
ax.set_xlabel("mean(|SHAP value|)  — impact on predicted health score")
ax.set_title("SHAP feature importance (Random Forest)")
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(VIS / "13_shap_importance.png", dpi=120); plt.close()
print("13 ok", flush=True)

# 14 SHAP beeswarm on transformed features (clean names, top 15)
sp = np.load(ROOT / "models" / "shap_sample.npz", allow_pickle=True)
sv = sp["shap_values"]; Xs = sp["X"]
fnames = [n.split("__", 1)[1] for n in sp["feat_names"]]
plt.figure()
shap.summary_plot(sv, Xs, feature_names=fnames, max_display=15, show=False)
plt.title("SHAP summary — Random Forest Regressor", fontsize=11)
plt.tight_layout(); plt.savefig(VIS / "14_shap_beeswarm.png", dpi=120, bbox_inches="tight"); plt.close()
print("14 ok", flush=True)
print("VISUALS DONE", flush=True)
