########## Five. Missing Data ##########

import pandas as pd

# 1. Detect Missing Values
df = pd.read_csv("students.csv")
# Check for any NaN in the DataFrame
print(df.isna()) # .isna() returns True/False for each cell
# Count NaN per column
print(df.isna().sum()) # .isna().sum() summarizes how many missing values per column

# 2. Remove Missing Data

# Drop rows with any missing value
df_clean = df.dropna()
print(df_clean)

# Drop rows in a specific column
df_clean_score = df.dropna(subset=["score"])
print(df_clean_score)
""" EXPLANATION:
Only removes rows where score is missing
Keeps other NaNs
"""

# Drop columns with NaNs
df = pd.read_csv("students.csv")
df_new = df.dropna(axis=1)
print(df_new)
""" EXPLANATION:
axis=1 → column-wise
axis=0 → row-wise
how='any' (default) → drop if any NaN
how='all' → drop only if all values are NaN
"""

# 3. Fill Missing Data

# Fill with a constant
df_filled = df.fillna(0)
print(df_filled)
""" EXPLANATION:
Replaces all NaNs with 0 (numeric columns)
For text columns, NaN becomes '0' (string)
"""

# Fill with a column mean (numeric)
df["age"] = df["age"].fillna(df["age"].mean())
print(df)
""" EXPLANATION:
Replace missing ages with the average age
Very common in ML preprocessing
"""

# Fill with column median
df["score"] = df["score"].fillna(df["score"].median())
print(df)
# EXPLANATION: Median is robust to outliers

# Fill with forward/backward values (time series)
# Forward-fill
df["attendance"] = df["attendance"].ffill()
# Backward-fill
df["attendance"] = df["attendance"].bfill()
print(df)

# 4. Fill Missing Text / Categorical Data
df["city"] = df["city"].fillna("Unknown")
print(df)
""" EXPLANATION:
NaNs in strings are not numbers
Pandas allows NaN for any dtype
"""

# 5. Combining Detection + Fill
# Quick way to see and fill in one go
df = pd.read_csv("students.csv")
# Check missing values before filling
print("Missing values before fill:")
print(df.isna().sum())
# Fill missing numeric values
df["age"] = df["age"].fillna(df["age"].mean())
df["score"] = df["score"].fillna(df["score"].median())
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())
# Fill missing categorical / text values
df["city"] = df["city"].fillna("Unknown")
# Check missing values after filling
print("\nMissing values after fill:")
print(df.isna().sum())
# Now the DataFrame is ready for ML or analytics.