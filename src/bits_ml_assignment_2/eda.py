
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/dry_bean.csv")

# Separate input features and target
X = df.drop(columns=["Class"])
y = df["Class"]

os.makedirs("eda_outputs", exist_ok=True)

# ------------------------------------------------
# 1. Duplicate records
# ------------------------------------------------
print("\n1. DUPLICATE RECORDS")
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------------------------
# 2. Descriptive statistics
# ------------------------------------------------
print("\n2. DESCRIPTIVE STATISTICS")
print(X.describe().T.round(3))

# ------------------------------------------------
# 3. Class distribution
# ------------------------------------------------
print("\n3. CLASS DISTRIBUTION")
print(y.value_counts())

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Class", order=y.value_counts().index)
plt.title("Distribution of Dry Bean Classes")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("eda_outputs/class_distribution.png")
plt.show()

# ------------------------------------------------
# 4. Outlier detection using IQR
# ------------------------------------------------
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_count = ((X < lower_bound) |
                 (X > upper_bound)).sum()

outlier_report = pd.DataFrame({
    "Outlier_Count": outlier_count,
    "Outlier_Percentage":
        (outlier_count / len(X) * 100).round(2)
})

print("\n4. OUTLIER REPORT")
print(outlier_report)

outlier_report.to_csv(
    "eda_outputs/outlier_report.csv"
)

# ------------------------------------------------
# 5. Feature correlation
# ------------------------------------------------
plt.figure(figsize=(14, 10))

sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.3
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.show()

# ------------------------------------------------
# 6. Feature distributions
# ------------------------------------------------
X.hist(figsize=(16, 12), bins=30, edgecolor="black")

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig("eda_outputs/feature_distributions.png")
plt.show()

print("\nEDA completed successfully.")
print("Plots and reports saved in eda_outputs/")
