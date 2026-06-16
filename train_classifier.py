"""
train_classifier.py
-------------------
Trains 6 gesture classifiers on the FULL gesture_data.csv and saves each model.

Classifiers:
  LR    – Logistic Regression
  KNN   – K-Nearest Neighbours
  SVM   – Support Vector Machine (RBF kernel)
  RF    – Random Forest
  XGB   – XGBoost
  ANN   – Multi-Layer Perceptron (6 hidden layers)

Each classifier is wrapped in a Pipeline: StandardScaler → Classifier.
Evaluation: 5-fold stratified cross-validation on ALL data.
After CV, each model is re-fitted on ALL data and saved.

Outputs:
  models/gesture_clf_LR.pkl
  models/gesture_clf_KNN.pkl
  models/gesture_clf_SVM.pkl
  models/gesture_clf_RF.pkl
  models/gesture_clf_XGB.pkl
  models/gesture_clf_ANN.pkl
  models/label_encoder.pkl     ← shared label encoder for all models

Usage:
    python train_classifier.py
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

CSV_FILE   = "gesture_data.csv"
MODEL_DIR  = "models"
VECTOR_DIM = 63

# ─── Classifier definitions ──────────────────────────────────────────────────
#
# ANN hidden layers: (512, 256, 256, 128, 64, 32)  → 6 hidden layers
#
CLASSIFIERS: dict[str, object] = {
    "LR":  LogisticRegression(
               max_iter=2000, C=1.0, solver="lbfgs"
           ),
    "KNN": KNeighborsClassifier(
               n_neighbors=7, metric="minkowski", weights="distance"
           ),
    "SVM": SVC(
               kernel="rbf", C=10, gamma="scale", probability=True
           ),
    "RF":  RandomForestClassifier(
               n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
           ),
    "XGB": XGBClassifier(
               n_estimators=300, learning_rate=0.1, max_depth=6,
               subsample=0.8, colsample_bytree=0.8,
               use_label_encoder=False, eval_metric="mlogloss",
               random_state=42, verbosity=0,
           ),
    "ANN": MLPClassifier(
               hidden_layer_sizes=(512, 256, 256, 128, 64, 32),
               activation="relu", solver="adam",
               learning_rate_init=0.001, max_iter=1000,
               early_stopping=True, validation_fraction=0.1,
               random_state=42,
           ),
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if not os.path.exists(CSV_FILE):
        sys.exit(f"[ERROR] {CSV_FILE} not found. Run collect_data.py first.")

    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} samples  |  {df['label'].nunique()} classes\n")
    print("Samples per class:")
    print(df["label"].value_counts().to_string())
    print()

    X = df[[f"v{i}" for i in range(VECTOR_DIM)]].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    return X, y, le


def build_pipeline(clf) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def evaluate_and_fit(name: str, pipeline: Pipeline, X: np.ndarray, y: np.ndarray, le: LabelEncoder):
    print(f"─── {name} {'─' * (52 - len(name))}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    t0 = time.time()
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_time = time.time() - t0

    print(f"  5-fold CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  (took {cv_time:.1f}s)")

    # Fit on ALL data
    t0 = time.time()
    pipeline.fit(X, y)
    fit_time = time.time() - t0
    print(f"  Trained on all {len(X)} samples in {fit_time:.1f}s")

    # In-sample report (fitted on full data, informational only)
    y_pred = pipeline.predict(X)
    acc = (y_pred == y).mean()
    print(f"  In-sample accuracy  : {acc:.4f}")
    print(f"  Classes             : {list(le.classes_)}\n")


def save_model(name: str, pipeline: Pipeline, le: LabelEncoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"gesture_clf_{name}.pkl")
    joblib.dump({"pipeline": pipeline, "label_encoder": le}, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved → {path}  ({size_kb:.0f} KB)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    X, y, le = load_data()

    # Save shared label encoder
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"Label encoder saved → {MODEL_DIR}/label_encoder.pkl\n")

    results: list[tuple[str, float, float]] = []

    for name, clf in CLASSIFIERS.items():
        pipeline = build_pipeline(clf)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        print(f"─── {name} {'─' * (52 - len(name))}")

        t0 = time.time()
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_time = time.time() - t0
        print(f"  5-fold CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  ({cv_time:.1f}s)")

        # Re-fit on ALL data
        t0 = time.time()
        pipeline.fit(X, y)
        fit_time = time.time() - t0
        print(f"  Full-data fit time : {fit_time:.1f}s")

        save_model(name, pipeline, le)
        results.append((name, cv_scores.mean(), cv_scores.std()))
        print()

    # ── Summary table ──
    print("=" * 52)
    print(f"  {'Model':<8}  {'CV Accuracy':>12}  {'± Std':>8}")
    print("=" * 52)
    for name, mean, std in sorted(results, key=lambda r: -r[1]):
        print(f"  {name:<8}  {mean:>12.4f}  {std:>8.4f}")
    print("=" * 52)

    best = max(results, key=lambda r: r[1])
    print(f"\n  Best model: {best[0]}  ({best[1]:.4f})")
    print(f"\nAll models saved to  {MODEL_DIR}/")


if __name__ == "__main__":
    main()
