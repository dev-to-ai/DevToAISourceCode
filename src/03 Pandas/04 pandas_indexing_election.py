########## Four. Indexing & Selection in Pandas ##########

import pandas as pd

# 1. Column Selection
# Columns = features in your dataset.

# Select Single Column → Series
df = pd.read_csv("students.csv")
age_series = df["age"]
print(age_series)
print(type(age_series))
# PRINT RESULT: print(type(age_series))

# Select Multiple Columns → DataFrame
subset_df = df[["age", "score"]]
print(subset_df)
print(type(subset_df))  
# PRINT RESULT: <class 'pandas.DataFrame'>

# 2. Row Selection: .loc vs .iloc

# .loc → label-based: Uses the index label
print(df.loc[0])   # row with index label 0

# .iloc → position-based: Uses integer position
print(df.iloc[0])  # first row (position 0)

# Selecting single value
# label-based
print(df.loc[0, "age"])
# position-based
print(df.iloc[0, 2]) # 0th row, 2nd column

# 3. Boolean Filtering
df_over_28 = df[df["age"] > 28]
print(df_over_28)
df_filtered = df[(df["age"] > 25) & (df["score"] > 80)]
print(df_filtered)
# Using .loc to select specific columns after filtering
print(df.loc[df["age"] > 25, ["name", "age", "score"]])

# 4. Quick Checklist
"""
Columns → df["col"] or df[["col1", "col2"]]

Rows → .loc (label) or .iloc (position)

Cells → .loc[row, col] / .iloc[row_idx, col_idx]

Boolean filters → df[condition]

Boolean + selection → df.loc[condition, columns]
"""


