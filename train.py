"""Train five classifiers on the UCI Dry Bean dataset and persist everything the app needs.

Run from the repo root:  python train.py
Outputs: model/*.joblib, model/metrics.csv, model/meta.json, model/train_log.txt, test_data.csv
"""
import json
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import (HOLDOUT_FRACTION, LABEL_COL, RNG, ROOT, load_clean_beans,
                        make_pipe, split_beans)

ARTIFACTS = ROOT / "model"
ARTIFACTS.mkdir(exist_ok=True)

beans, rows_raw = load_clean_beans()
print(f"rows: {rows_raw} -> {len(beans)} after dropping exact duplicates")

X_fit, X_hold, y_fit, y_hold, feature_cols = split_beans(beans)
print(f"train={len(X_fit)}  hold-out={len(X_hold)}")

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
        "classes": [str(c) for c in sorted(beans[LABEL_COL].unique())],
        "random_state": RNG,
        "holdout_fraction": HOLDOUT_FRACTION,
        "rows_raw": rows_raw,
        "rows_after_dedup": len(beans),
        "train_rows": len(X_fit),
        "holdout_rows": len(X_hold),
        "averaging": "macro (precision, recall, F1); AUC = one-vs-rest macro",
    }, fh, indent=2)

(ARTIFACTS / "train_log.txt").write_text("\n".join(log_lines) + "\n")
