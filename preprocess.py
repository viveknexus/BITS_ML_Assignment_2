"""Data preparation for the Dry Bean models: cleaning, hold-out split and the scaling pipeline.

Imported by train.py. The fitted scaler is stored inside every saved pipeline,
so the Streamlit app does not need this module.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "dry_bean.csv"
LABEL_COL = "Class"
RNG = 2026
HOLDOUT_FRACTION = 0.20


def load_clean_beans():
    """Read the raw CSV and drop exact duplicate rows. Returns (frame, raw_row_count)."""
    beans = pd.read_csv(RAW_CSV)
    rows_raw = len(beans)
    # Exact duplicate grains would sit in both train and test and inflate the scores.
    beans = beans.drop_duplicates().reset_index(drop=True)
    return beans, rows_raw


def split_beans(beans):
    """Stratified train / hold-out split. Returns (X_fit, X_hold, y_fit, y_hold, feature_cols)."""
    feature_cols = [c for c in beans.columns if c != LABEL_COL]
    X_all, y_all = beans[feature_cols], beans[LABEL_COL]
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X_all, y_all, test_size=HOLDOUT_FRACTION, stratify=y_all, random_state=RNG)
    return X_fit, X_hold, y_fit, y_hold, feature_cols


def make_pipe(estimator):
    """Scaler + classifier; the scaler is fitted on the training rows only when .fit() is called."""
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])
