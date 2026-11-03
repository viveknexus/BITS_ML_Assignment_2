# Dry Bean Classifier Bench

Five classifiers on the UCI Dry Bean dataset, plus a Streamlit app that evaluates the saved models on an uploaded test CSV.

## a. Problem Statement

Dry beans are sold by variety, and sorting them by hand is slow and error-prone. This project predicts which of **seven bean varieties** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) a grain belongs to, using 16 shape and size measurements extracted from a photograph of the grain. It is a **multi-class classification** problem. Five models are trained on the same data and compared on the same held-out split: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and an RBF Support Vector Machine.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository, *Dry Bean Dataset*  — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset.
- **Instances:** 13,611 raw rows; **13,543** after removing 68 exact duplicate rows (done before the split so duplicates cannot appear in both train and test).
- **Features:** 16, all numeric: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1–4.
- **Target:** `Class`, 7 classes. Hold-out counts: DERMASON 709, SIRA 527, SEKER 406, HOROZ 372, CALI 326, BARBUNYA 265, BOMBAY 104. The classes are imbalanced (BOMBAY is about 4%).
- **Missing values:** none.
- **Split:** stratified 80/20 (`random_state=2026`) giving 10,834 train and 2,709 hold-out rows. The hold-out rows are saved as `test_data.csv` (all 16 features + `Class`).
- **Preprocessing:** implemented in `preprocess.py` (duplicate removal, stratified split, `StandardScaler` pipeline). The scaler is fitted on the training rows only. The fitted scaler is saved as part of each `model/bean_*.joblib` file (a scikit-learn `Pipeline` of scaler + classifier), so there is no separate scaler file and the app needs no extra preprocessing code. Many features are strongly correlated (15 of the 120 feature pairs have |r| > 0.95, e.g. Area, ConvexArea, EquivDiameter).
- **Tuning:** `GridSearchCV` (3-fold stratified CV, macro-F1) on the training split only; the hold-out set was never used for tuning.
  Chosen parameters: LR `C=1`; Tree `max_depth=8, min_samples_leaf=20`; KNN `k=9, weights=distance`; NB `var_smoothing=1e-3`; SVM `C=10, gamma=0.05`.

## c. GitHub Repository

- Repository: `https://github.com/viveknexus/BITS_ML_Assignment_2/`

## d. Models Used

Metrics are computed on the 2,709-row hold-out set (`test_data.csv`). Precision, Recall and F1 are **macro-averaged** over the 7 classes; AUC is **one-vs-rest, macro-averaged**. Source: `model/metrics.csv`, produced by `train.py`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 
| Decision Tree | 
| KNN | 
| Naive Bayes | 
| Support Vector Machine | 