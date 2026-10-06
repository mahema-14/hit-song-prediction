"""Train and compare 8 classifiers for hit (1) / non-hit (0) prediction.

Usage:
    python src/train_models.py --data data/dataset-of-10s.csv
"""
import argparse, os, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
ID_COLS = ["track", "artist", "uri"]          # non-numeric identifiers, dropped
TARGET = "target"


def get_models():
    """(name, estimator, param_grid) - grids kept small so it runs in minutes."""
    return [
        ("Logistic Regression", LogisticRegression(max_iter=2000),
         {"clf__C": [0.1, 1, 10]}),
        ("LDA", LinearDiscriminantAnalysis(solver="lsqr"),
         {"clf__shrinkage": [None, "auto", 0.5]}),
        ("Linear SVM", SVC(kernel="linear"),
         {"clf__C": [0.1, 1]}),
        ("RBF SVM", SVC(kernel="rbf"),
         {"clf__C": [1, 10], "clf__gamma": ["scale", 0.05]}),
        ("Polynomial SVM", SVC(kernel="poly"),
         {"clf__C": [1, 10], "clf__degree": [2, 3]}),
        ("Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1),
         {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 15],
          "clf__min_samples_leaf": [1, 3]}),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42),
         {"clf__n_estimators": [150, 300], "clf__learning_rate": [0.05, 0.1],
          "clf__max_depth": [3, 4]}),
        ("Neural Network / MLP", MLPClassifier(max_iter=500, early_stopping=True,
                                               random_state=42),
         {"clf__hidden_layer_sizes": [(64,), (64, 32)], "clf__alpha": [1e-3, 1e-2]}),
    ]


def main(a):
    df = pd.read_csv(a.data)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    print(f"Data: {X.shape[0]} songs, {X.shape[1]} features, "
          f"hits={y.sum()}, non-hits={(y == 0).sum()}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    rows, best = [], None
    for name, est, grid in get_models():
        t0 = time.time()
        pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
        p = gs.predict(Xte)
        r = dict(Model=name, CV_F1=gs.best_score_,
                 Test_Accuracy=accuracy_score(yte, p),
                 Test_Precision=precision_score(yte, p),
                 Test_Recall=recall_score(yte, p), Test_F1=f1_score(yte, p),
                 Best_Params=str(gs.best_params_))
        rows.append(r)
        print(f"{name:22s} CV F1={r['CV_F1']:.4f}  Test Acc={r['Test_Accuracy']:.4f}  "
              f"Test F1={r['Test_F1']:.4f}  ({time.time() - t0:.0f}s)")
        if best is None or r["CV_F1"] > best[0]:
            best = (r["CV_F1"], name, gs.best_estimator_)

    res = pd.DataFrame(rows).sort_values("CV_F1", ascending=False)
    os.makedirs("models", exist_ok=True)
    res.to_csv("models/results.csv", index=False)
    joblib.dump({"model": best[2], "features": list(X.columns), "name": best[1]},
                "models/best_model.joblib")
    print(f"\nBest model (by CV F1): {best[1]}  -> saved to models/best_model.joblib")
    print(res.drop(columns="Best_Params").round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset-of-10s.csv")
    main(ap.parse_args())
