"""
Generate Member 2 evaluation visuals from the saved metrics/metadata JSON.

Run after create_member2_package.py:
    python scripts/make_member2_visuals.py

Reads:
    data/member2_model_metrics.json
    models/model_metadata.json
Writes:
    visuals/08_model_comparison.png
    visuals/09_confusion_matrix_logreg.png
    visuals/10_confusion_matrix_rf.png
    visuals/11_feature_importance.png
    visuals/12_tuning_results.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
VISUALS = ROOT / "visuals"; VISUALS.mkdir(exist_ok=True)
CLASS_ORDER = ["Good", "Fair", "Poor"]

metrics = json.loads((ROOT / "data" / "member2_model_metrics.json").read_text())
metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
lr = metrics["logistic_regression"]
rf = metrics["random_forest"]

# 08 model comparison
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Logistic Regression\n(baseline)", "Random Forest\n(tuned)"]
acc_vals = [lr["accuracy"], rf["accuracy"]]
f1_vals = [lr["macro_f1"], rf["macro_f1"]]
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w/2, acc_vals, w, label="Accuracy", color="#1565C0")
ax.bar(x + w/2, f1_vals, w, label="Macro-F1", color="#2E7D32")
for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
    ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
ax.set_ylabel("Score"); ax.set_title("Model comparison: accuracy vs macro-F1")
ax.legend(); plt.tight_layout(); plt.savefig(VISUALS / "08_model_comparison.png", dpi=120); plt.close()
print("saved 08_model_comparison.png", flush=True)


def plot_cm(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASS_ORDER))); ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER))); ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


plot_cm(lr["confusion_matrix"], "Confusion matrix - Logistic Regression",
        VISUALS / "09_confusion_matrix_logreg.png")
print("saved 09_confusion_matrix_logreg.png", flush=True)
plot_cm(rf["confusion_matrix"], "Confusion matrix - Random Forest (tuned)",
        VISUALS / "10_confusion_matrix_rf.png")
print("saved 10_confusion_matrix_rf.png", flush=True)

# 11 feature importance
fi = metadata["feature_importance"]
names = list(fi.keys())[::-1]
vals = [fi[k] for k in names]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(names, vals, color="#2E7D32")
ax.set_xlabel("Aggregated importance"); ax.set_title("Random Forest feature importance")
for i, v in enumerate(vals):
    ax.text(v + 0.002, i, f"{v:.2f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(VISUALS / "11_feature_importance.png", dpi=120); plt.close()
print("saved 11_feature_importance.png", flush=True)

# 12 tuning results
tr = sorted(metrics["tuning_results"], key=lambda r: r["mean_test_macro_f1"])
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(tr)), [r["mean_test_macro_f1"] for r in tr],
        xerr=[r["std_test_macro_f1"] for r in tr], color="#1565C0", alpha=0.85)
ax.set_yticks(range(len(tr)))
ax.set_yticklabels([
    f"depth={r['params']['max_depth']}, leaf={r['params']['min_samples_leaf']}, "
    f"trees={r['params']['n_estimators']}" for r in tr
], fontsize=8)
ax.set_xlabel("CV mean macro-F1 (3-fold)")
ax.set_title("Hyperparameter search results (Random Forest)")
ax.axvline(metrics["cv_best_macro_f1"], color="#C62828", ls="--", lw=1,
           label=f"best={metrics['cv_best_macro_f1']:.3f}")
ax.legend(); plt.tight_layout(); plt.savefig(VISUALS / "12_tuning_results.png", dpi=120); plt.close()
print("saved 12_tuning_results.png", flush=True)
print("\nVISUALS DONE", flush=True)
