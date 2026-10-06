"""Predict 1 = hit / 0 = non-hit with the saved best model.

Usage:
    python src/predict.py --input data/new_songs.csv --output predictions.csv
The CSV needs the same feature columns used in training (extra columns are ignored).
"""
import argparse, joblib, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="predictions.csv")
ap.add_argument("--model", default="models/best_model.joblib")
a = ap.parse_args()

bundle = joblib.load(a.model)
df = pd.read_csv(a.input)
pred = bundle["model"].predict(df[bundle["features"]])
out = df.copy()
out["prediction"] = pred
out["label"] = out["prediction"].map({1: "hit", 0: "non-hit"})
out.to_csv(a.output, index=False)
print(f"Model: {bundle['name']} | {len(out)} songs | hits predicted: {int(pred.sum())}")
print(f"Saved -> {a.output}")
