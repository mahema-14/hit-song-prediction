"""Error analysis for the saved best model: confusion matrix + permutation feature importance.

Usage:
    python src/analyze.py --data data/dataset-of-10s.csv
Outputs (in models/): confusion_matrix.png, feature_importance.png, feature_importance.csv
"""
import argparse
import joblib, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dataset-of-10s.csv")
ap.add_argument("--model", default="models/best_model.joblib")
a = ap.parse_args()

bundle = joblib.load(a.model)
model, feats, name = bundle["model"], bundle["features"], bundle["name"]

df = pd.read_csv(a.data)
X, y = df[feats], df["target"].astype(int)
# same split as train_models.py (test_size=0.25, stratify, random_state=42)
_, Xte, _, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
pred = model.predict(Xte)

print(f"Model: {name}\n")
print(classification_report(yte, pred, target_names=["non-hit", "hit"], digits=4))

# 1) Confusion matrix
cm = confusion_matrix(yte, pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["non-hit", "hit"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"Confusion matrix - {name}")
fig.tight_layout(); fig.savefig("models/confusion_matrix.png", dpi=150); plt.close(fig)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# 2) Permutation feature importance (drop in F1 when a feature is shuffled)
r = permutation_importance(model, Xte, yte, scoring="f1", n_repeats=10,
                           random_state=42, n_jobs=-1)
imp = (pd.DataFrame({"feature": feats, "importance_mean": r.importances_mean,
                     "importance_std": r.importances_std})
       .sort_values("importance_mean", ascending=False))
imp.to_csv("models/feature_importance.csv", index=False)
print("\nTop 10 features:")
print(imp.head(10).round(4).to_string(index=False))

top = imp.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C72B0")
ax.set_xlabel("Drop in F1 when feature is shuffled")
ax.set_title(f"Permutation feature importance - {name}")
fig.tight_layout(); fig.savefig("models/feature_importance.png", dpi=150); plt.close(fig)
print("\nSaved: models/confusion_matrix.png, models/feature_importance.png, models/feature_importance.csv")