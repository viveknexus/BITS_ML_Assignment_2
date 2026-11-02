
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# Support both the current root-level location
# and the proposed project data/ directory.
possible_paths = [
    BASE_DIR / "data" / "dry_bean.csv",
    BASE_DIR / "dry_bean.csv",
]

DATA_PATH = next(
    (path for path in possible_paths if path.exists()),
    None
)

TARGET = "Class"
TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Load dataset
# ==========================================

if DATA_PATH is None:
    raise FileNotFoundError(
        "dry_bean.csv not found. "
        "Place it in the project root or data/ directory."
    )

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

feature_columns = [
    column for column in df.columns
    if column != TARGET
]


# ==========================================
# 2. Check conflicting duplicate features
# ==========================================

duplicate_features = df[
    df.duplicated(
        subset=feature_columns,
        keep=False
    )
]

if not duplicate_features.empty:

    conflicting_groups = (
        duplicate_features
        .groupby(feature_columns)[TARGET]
        .nunique()
    )

    conflicts = conflicting_groups[
        conflicting_groups > 1
    ]

    if not conflicts.empty:
        raise ValueError(
            "Identical features have conflicting target labels. "
            "Investigate these records before proceeding."
        )

print("\nConflicting duplicate feature combinations: 0")


# ==========================================
# 3. Remove exact duplicate records
# ==========================================

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)

print("Duplicates removed:", duplicate_count)
print("Cleaned dataset shape:", df.shape)

# Save cleaned dataset
df.to_csv(
    OUTPUT_DIR / "dry_bean_clean.csv",
    index=False
)


# ==========================================
# 4. Separate features and target
# ==========================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 5. Stratified train/test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 6. Save training and testing datasets
# ==========================================

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv(
    OUTPUT_DIR / "train_data.csv",
    index=False
)

# Assignment requires test_data.csv in repository root
test_df.to_csv(
    BASE_DIR / "test_data.csv",
    index=False
)

print("\nTraining data saved:",
      OUTPUT_DIR / "train_data.csv")

print("Testing data saved:",
      BASE_DIR / "test_data.csv")


# ==========================================
# 7. Verify stratified class distribution
# ==========================================

distribution = pd.DataFrame({
    "Full Dataset (%)": y.value_counts(normalize=True) * 100,
    "Training (%)": y_train.value_counts(normalize=True) * 100,
    "Testing (%)": y_test.value_counts(normalize=True) * 100
}).fillna(0)

print("\nClass distribution comparison:")
print(distribution.round(2))


# ==========================================
# 8. Final validation
# ==========================================

assert len(X_train) + len(X_test) == len(df)

assert X_train.columns.tolist() == X_test.columns.tolist()

assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0

assert set(y_train.unique()) == set(y_test.unique())

print("\nPreprocessing completed successfully!")
