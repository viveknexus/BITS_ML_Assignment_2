
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/dry_bean.csv")

X = df.drop(columns=["Class"])

# ==========================================
# 1. Investigate exact duplicate records
# ==========================================

print("\n--- EXACT DUPLICATE ANALYSIS ---")

print("Repeated occurrences:", df.duplicated().sum())

duplicate_rows = df[df.duplicated(keep=False)]

print("Total rows involved in duplicate groups:",
      len(duplicate_rows))

# ==========================================
# 2. Check identical features with different labels
# ==========================================

print("\n--- DUPLICATE FEATURE ANALYSIS ---")

duplicate_features = df[
    df.duplicated(subset=X.columns, keep=False)
]

if duplicate_features.empty:
    print("No repeated feature combinations found.")
else:
    conflicting_groups = (
        duplicate_features
        .groupby(list(X.columns))["Class"]
        .nunique()
    )

    conflicting_groups = conflicting_groups[
        conflicting_groups > 1
    ]

    print("Feature combinations with conflicting labels:",
          len(conflicting_groups))

# ==========================================
# 3. Analyze highly correlated features
# ==========================================

print("\n--- TOP CORRELATED FEATURE PAIRS ---")

correlation = X.corr().abs()

upper_triangle = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
)

correlation_pairs = (
    upper_triangle
    .stack()
    .sort_values(ascending=False)
)

print(correlation_pairs.head(15).to_string())

# ==========================================
# 4. Save correlation pairs
# ==========================================

correlation_pairs.to_csv(
    "eda_outputs/correlation_pairs.csv",
    header=["Absolute_Correlation"]
)

print("\nValidation completed.")
