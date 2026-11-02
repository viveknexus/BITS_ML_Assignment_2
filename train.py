"""Train five classifiers on the UCI Dry Bean dataset and persist everything the app needs.

Run from the repo root:  python train.py
Outputs: model/*.joblib, model/metrics.csv, model/meta.json, model/train_log.txt, test_data.csv
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "dry_bean.csv"
ARTIFACTS = ROOT / "model"
LABEL_COL = "Class"
RNG = 2026
HOLDOUT_FRACTION = 0.20

ARTIFACTS.mkdir(exist_ok=True)

beans = pd.read_csv(RAW_CSV)
rows_raw = len(beans)
# Exact duplicate grains would sit in both train and test and inflate the scores.
beans = beans.drop_duplicates().reset_index(drop=True)
print(f"rows: {rows_raw} -> {len(beans)} after dropping exact duplicates")

feature_cols = [c for c in beans.columns if c != LABEL_COL]
X_all, y_all = beans[feature_cols], beans[LABEL_COL]
X_fit, X_hold, y_fit, y_hold = train_test_split(
    X_all, y_all, test_size=HOLDOUT_FRACTION, stratify=y_all, random_state=RNG)
print(f"train={len(X_fit)}  hold-out={len(X_hold)}")

# Every pipeline starts with a scaler that is fitted on the training part only.
def make_pipe(estimator):
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])

# (estimator, hyper-parameter grid). Grids are searched with CV on the training split only.
candidates = {
    "bean_logistic_regression": (
        LogisticRegression(max_iter=3000, random_state=RNG),
        {"clf__C": [0.1, 1, 10, 100]}),
    "bean_decision_tree": (
        DecisionTreeClassifier(random_state=RNG),
        {"clf__max_depth": [4, 6, 8, 12, None], "clf__min_samples_leaf": [1, 5, 20]}),
    "bean_knn": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
    "bean_naive_bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
    "bean_svm": (
        SVC(kernel="rbf", probability=True, random_state=RNG),
        {"clf__C": [1, 10, 50], "clf__gamma": ["scale", 0.05]}),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
score_rows, log_lines = [], []

for file_stem, (estimator, grid) in candidates.items():
    t0 = time.time()
    search = GridSearchCV(make_pipe(estimator), grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(X_fit, y_fit)
    best = search.best_estimator_

    labels = best.classes_
    y_hat = best.predict(X_hold)
    y_prob = best.predict_proba(X_hold)
    score_rows.append({
        "Model": file_stem,
        "Accuracy": accuracy_score(y_hold, y_hat),
        "AUC": roc_auc_score(y_hold, y_prob, multi_class="ovr", average="macro", labels=labels),
        "Precision": precision_score(y_hold, y_hat, average="macro", zero_division=0),
        "Recall": recall_score(y_hold, y_hat, average="macro", zero_division=0),
        "F1": f1_score(y_hold, y_hat, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_hold, y_hat),
    })
    joblib.dump(best, ARTIFACTS / f"{file_stem}.joblib", compress=3)
    line = (f"{file_stem}: best_params={search.best_params_} "
            f"cv_f1_macro={search.best_score_:.4f} ({time.time() - t0:.1f}s)")
    print(line)
    log_lines.append(line)

metrics = pd.DataFrame(score_rows).round(4)
metrics.to_csv(ARTIFACTS / "metrics.csv", index=False)
print(metrics.to_string(index=False))

# test_data.csv: hold-out rows with every feature column plus the target column
pd.concat([X_hold, y_hold], axis=1).to_csv(ROOT / "test_data.csv", index=False)

with open(ARTIFACTS / "meta.json", "w") as fh:
    json.dump({
        "target": LABEL_COL,
        "features": feature_cols,
        "classes": [str(c) for c in sorted(y_all.unique())],
        "random_state": RNG,
        "holdout_fraction": HOLDOUT_FRACTION,
        "rows_raw": rows_raw,
        "rows_after_dedup": len(beans),
        "train_rows": len(X_fit),
        "holdout_rows": len(X_hold),
        "averaging": "macro (precision, recall, F1); AUC = one-vs-rest macro",
    }, fh, indent=2)

(ARTIFACTS / "train_log.txt").write_text("\n".join(log_lines) + "\n")
