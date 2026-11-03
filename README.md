# Dry Bean Classifier

Five classifiers on the UCI Dry Bean dataset, plus a Streamlit app that evaluates the saved models on an uploaded test CSV.

## a. Problem Statement

Dry beans are sold by variety, and sorting them by hand is slow and error-prone. This project predicts which of **seven bean varieties** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) a grain belongs to, using 16 shape and size measurements extracted from a photograph of the grain. It is a **multi-class classification** problem. Five models are trained on the same data and compared on the same held-out split: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and an RBF Support Vector Machine.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository, *Dry Bean Dataset* — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset.
- **Instances:** 13,611 raw rows; **13,543** after removing 68 exact duplicate rows (done before the split so duplicates cannot appear in both train and test).
- **Features:** 16, all numeric: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1–4.
- **Target:** `Class`, 7 classes. Hold-out counts: DERMASON 709, SIRA 527, SEKER 406, HOROZ 372, CALI 326, BARBUNYA 265, BOMBAY 104. The classes are imbalanced (BOMBAY is about 4%).
- **Missing values:** none.
- **Split:** stratified 80/20 (`random_state=2026`) giving 10,834 train and 2,709 hold-out rows. The hold-out rows are saved as `test_data.csv` (all 16 features + `Class`).
- **Preprocessing:** implemented in `preprocess.py` (duplicate removal, stratified split, `StandardScaler` pipeline). The scaler is fitted on the training rows only. The fitted scaler is saved as part of each `model/bean_*.joblib` file (a scikit-learn `Pipeline` of scaler + classifier), so there is no separate scaler file and the app needs no extra preprocessing code. Many features are strongly correlated (15 of the 120 feature pairs have |r| > 0.95, e.g. Area, ConvexArea, EquivDiameter).
- **Tuning:** `GridSearchCV` (3-fold stratified CV, macro-F1) on the training split only; the hold-out set was never used for tuning.
  Chosen parameters: LR `C=100`; Tree `max_depth=8, min_samples_leaf=20`; KNN `k=9, weights=distance`; NB `var_smoothing=1e-3`; SVM `C=10, gamma=0.05`.

## c. GitHub Repository Link

GitHub Repository: https://github.com/viveknexus/BITS_ML_Assignment_2/

The repository contains the complete source code, trained classification models, preprocessing scripts, requirements.txt, test dataset and README documentation.

## d. Models Used

Metrics are computed on the 2,709-row hold-out set (`test_data.csv`). Precision, Recall and F1 are **macro-averaged** over the 7 classes; AUC is **one-vs-rest, macro-averaged**. Source: `model/metrics.csv`, produced by `train.py`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9247 | 0.9946 | 0.9355 | 0.9374 | 0.9364 | 0.909 |
| Decision Tree | 0.9055 | 0.9895 | 0.9185 | 0.9179 | 0.9182 | 0.8857 |
| KNN | 0.9228 | 0.9882 | 0.9374 | 0.9364 | 0.9369 | 0.9067 |
| Naive Bayes | 0.8992 | 0.9914 | 0.9067 | 0.9123 | 0.9092 | 0.8786 |
| Support Vector Machine | 0.9295 | 0.9949 | 0.9417 | 0.9405 | 0.9411 | 0.9147 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The standardised linear model is strong: 0.9247 accuracy, 0.9364 macro-F1 and 0.9946 AUC. Its weakest class is SIRA (F1 0.8691), with the largest confusion between SIRA and DERMASON (46 and 45 rows). It is also the smallest saved model at about 2.5 KB, making it a practical lightweight alternative. |
| Decision Tree | The tree is the second-weakest model on accuracy (0.9055) and macro-F1 (0.9182). Cross-validation selected a constrained tree (`max_depth=8`, `min_samples_leaf=20`), consistent with controlling overfitting. Its largest errors are SIRA→DERMASON (57) and DERMASON→SIRA (52), and it has the lowest SIRA F1 (0.8457). Its main advantage is interpretability. |
| KNN | KNN is competitive with Logistic Regression (0.9228 accuracy, 0.9369 macro-F1), but has the lowest AUC (0.9882), indicating less reliable probability ranking. It achieves the best per-class F1 for BARBUNYA (0.9389), CALI (0.9477) and HOROZ (0.9557). Standardisation is essential, and the saved model is the largest (about 1.3 MB) because it retains the training data. |
| Naive Bayes | Naive Bayes has the lowest accuracy (0.8992), macro-F1 (0.9092) and MCC (0.8786). Its conditional-independence assumption is a poor fit for the strongly correlated size features. BARBUNYA (F1 0.8642) and CALI (0.8869) are its weakest classes; its most frequent confusion is DERMASON→SIRA (67 rows). Its AUC (0.9914) remains higher than the tree and KNN, so its probability ranking is better than its hard predictions. |
| Support Vector Machine | SVM is the best model on every reported metric: 0.9295 accuracy, 0.9411 macro-F1, 0.9147 MCC and 0.9949 AUC. The RBF kernel improves accuracy over Logistic Regression by 0.0048 (about 13 of 2,709 hold-out rows) and gives the best F1 for SEKER (0.9537) and SIRA (0.8786). The trade-off is higher training cost and the additional probability calibration from `probability=True`. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** It leads on all six metrics, but the margin over Logistic Regression is small, so LR is a defensible lower-cost alternative. Every model struggles most with DERMASON/SIRA separation, suggesting that this limitation is driven by overlap in the available 16 shape features rather than by a single algorithm. Macro-F1 and MCC are more informative than accuracy alone because the classes are imbalanced. |

## Repository Structure

```
.
├── app.py                 Streamlit app (loads saved pipelines, never retrains)
├── preprocess.py          Data cleaning, stratified split and the scaler pipeline
├── train.py               Tuning, evaluation and saving (uses preprocess.py)
├── requirements.txt       Pinned dependencies
├── test_data.csv          Hold-out split: 16 features + Class (2,709 rows)
├── data/dry_bean.csv      Raw UCI data (converted from the original .xlsx)
└── model/
    ├── bean_logistic_regression.joblib   Each .joblib = fitted StandardScaler + classifier
    ├── bean_decision_tree.joblib         (preprocessing artifact is bundled inside)
    ├── bean_knn.joblib
    ├── bean_naive_bayes.joblib
    ├── bean_svm.joblib
    ├── metrics.csv        Source of the table above
    ├── meta.json          Feature names, classes, split info
    └── train_log.txt      Best hyper-parameters and CV score per model
```

## How to Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python train.py                 # optional: regenerates model/, metrics.csv, test_data.csv
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar.

## Streamlit App Features

1. CSV upload (test data only), with a preview, row count and validation of required columns.
2. Model selection dropdown with all five models.
3. Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
4. Confusion matrix heatmap and per-class classification report.
5. All five models side by side on the uploaded file (table and bar chart), a misclassified-rows tab with the most frequent confusions, and a download button for predictions.

The app handles missing columns, a missing target column (predictions only), empty files, rows with unknown labels or NaNs (dropped with a warning), and uploads where AUC is undefined (shown as n/a).
