import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch Dry Bean dataset from UCI
dry_bean = fetch_ucirepo(id=602)

# Separate features and target
X = dry_bean.data.features
y = dry_bean.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save dataset
df.to_csv("dry_bean.csv", index=False)

# Initial inspection
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nFeature Information:")
print(df.info())