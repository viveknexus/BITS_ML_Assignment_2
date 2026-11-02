"""Dry Bean classifier bench: evaluate pre-trained models on an uploaded hold-out CSV.

The app only loads saved pipelines from model/ and calls predict / predict_proba.
Nothing is ever fitted here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

HERE = Path(__file__).parent
ARTIFACT_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "bean_logistic_regression.joblib",
    "Decision Tree": "bean_decision_tree.joblib",
    "K-Nearest Neighbors": "bean_knn.joblib",
    "Naive Bayes (Gaussian)": "bean_naive_bayes.joblib",
    "Support Vector Machine (RBF)": "bean_svm.joblib",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Dry Bean Classifier Bench", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: #f4efe6; border-left: 5px solid #8c5a2b;
        padding: 0.6rem 0.9rem; border-radius: 6px;
    }
    div[data-testid="stMetricValue"] {color: #4a2f14;}
    div[data-testid="stMetricLabel"] p {color: #4a2f14;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_meta():
    with open(ARTIFACT_DIR / "meta.json") as fh:
        return json.load(fh)


@st.cache_resource
def load_pipeline(display_name):
    return joblib.load(ARTIFACT_DIR / MODEL_FILES[display_name])


def score_model(pipe, X, y_true, class_names):
    """Return (metrics dict, predictions, probabilities)."""
    y_pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": np.nan,
    }
    # One-vs-rest AUC is only defined when every class occurs in the uploaded labels.
    if set(y_true.unique()) == set(class_names):
        out["AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                   labels=pipe.classes_)
    return out, y_pred, proba


def fmt(value):
    return "n/a" if pd.isna(value) else f"{value:.4f}"


meta = read_meta()
features, target, classes = meta["features"], meta["target"], meta["classes"]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + target)", type="csv")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=False,
                                 disabled=not SAMPLE_CSV.exists())
chosen = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
st.sidebar.caption(
    f"Expected columns: {len(features)} features + `{target}`. "
    "Models are pre-trained; the file is used for evaluation only."
)

st.title("Dry Bean Classifier Bench")
st.write(
    "Seven dry-bean varieties classified from 16 shape and size measurements taken from "
    "grain images (UCI Dry Bean dataset). Pick a model, upload held-out data, "
    "and inspect how each classifier behaves."
)

if uploaded is None and not use_sample:
    st.info("Upload a CSV in the sidebar to begin. The file must contain the 16 feature "
            f"columns and the `{target}` column.")
    with st.expander("Required columns"):
        st.code(", ".join(features + [target]))
    st.stop()

# ---------------------------------------------------------------- load & validate
try:
    raw = pd.read_csv(SAMPLE_CSV if uploaded is None else uploaded)
except pd.errors.EmptyDataError:
    st.error("The uploaded file is empty.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the file as CSV: {exc}")
    st.stop()

if raw.empty:
    st.error("The uploaded file has a header but no rows.")
    st.stop()

missing_cols = [c for c in features if c not in raw.columns]
if missing_cols:
    st.error(f"Missing required feature columns: {missing_cols}")
    st.stop()

X = raw[features].apply(pd.to_numeric, errors="coerce")
has_target = target in raw.columns
bad_rows = X.isna().any(axis=1)
if has_target:
    y = raw[target].astype(str).str.strip()
    unseen_mask = ~y.isin(classes)
    if unseen_mask.any():
        st.warning(f"{int(unseen_mask.sum())} row(s) have a label the models were not trained "
                   f"on and were excluded. Known labels: {classes}")
    bad_rows = bad_rows | unseen_mask
if bad_rows.any():
    st.warning(f"Dropped {int(bad_rows.sum())} row(s) with missing/non-numeric values "
               "or unknown labels.")
keep = ~bad_rows
X = X[keep]
raw_kept = raw[keep].reset_index(drop=True)
X = X.reset_index(drop=True)
if X.empty:
    st.error("No usable rows remain after cleaning.")
    st.stop()

st.subheader("Uploaded data")
c1, c2, c3 = st.columns(3)
c1.metric("Rows used", f"{len(X):,}")
c2.metric("Feature columns", len(features))
c3.metric("Target present", "yes" if has_target else "no")
st.dataframe(raw_kept.head(10), width="stretch")

pipe = load_pipeline(chosen)

if not has_target:
    st.warning(f"No `{target}` column found: showing predictions only, no metrics.")
    preds = pipe.predict(X)
    out = raw_kept.assign(Predicted=preds)
    st.dataframe(out.head(50), width="stretch")
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
    st.stop()

y = y[keep].reset_index(drop=True)
if y.nunique() < 2:
    st.warning("Only one class is present in the uploaded labels; AUC and MCC are not meaningful.")

tab_eval, tab_compare, tab_errors = st.tabs(
    ["Evaluation", "All models side by side", "Misclassified rows"])

# ---------------------------------------------------------------- evaluation tab
with tab_eval:
    scores, y_pred, proba = score_model(pipe, X, y, classes)
    st.subheader(chosen)
    cols = st.columns(6)
    for col, name in zip(cols, METRIC_ORDER):
        col.metric(name, fmt(scores[name]))
    if pd.isna(scores["AUC"]):
        st.caption("AUC is n/a because not every class appears in the uploaded labels.")
    st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=classes,
                    yticklabels=classes, cbar=False, ax=ax)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=classes, output_dict=True,
                                       zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    pred_table = raw_kept.assign(Predicted=y_pred)
    st.download_button("Download predictions as CSV", pred_table.to_csv(index=False).encode(),
                       f"predictions_{MODEL_FILES[chosen].replace('.joblib', '')}.csv", "text/csv")


