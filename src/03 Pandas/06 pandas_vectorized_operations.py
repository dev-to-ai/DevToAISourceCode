########## Six. Vectorized Operations ##########

import pandas as pd

# 1. Pandas operates on entire columns at once.
df = pd.read_csv("students.csv")
df["age_plus_10"] = df["age"] + 10
print(df[["age", "age_plus_10"]].head())
""" PRINT RESULT:
    age  age_plus_10
0  23.0         33.0
1  27.0         37.0
2   NaN          NaN
3  31.0         41.0
4  25.0         35.0
"""
df["normalized_score"] = df["score"] / df["score"].max()
print(df[["score", "normalized_score"]].head())
""" PRINT RESULT:
   score  normalized_score
0   88.0          0.926316
1   92.0          0.968421
2   79.0          0.831579
3    NaN               NaN
4   95.0          1.000000
"""
df["score_squared"] = df["score"] ** 2
df["attendance_pct"] = df["attendance"] * 100
df["age_diff_from_mean"] = df["age"] - df["age"].mean()

# 2. apply() Functions
df["grade"] = df["score"].apply(
    lambda x: "Missing" if pd.isna(x) else ("A" if x >= 90 else "B")
)
print(df[["score", "grade"]])
""" PRINT RESULT:
   score grade
0   88.0     B
1   92.0     A
2   79.0     B
3    NaN     Missing
4   95.0     A
5   67.0     B
6   85.0     B
7   90.0     A
8   73.0     B
9   89.0     B
"""

# 3. Prefer Vectorized Alternatives to apply()
import numpy as np
df["grade"] = np.where(
    df["score"].isna(),
    "Missing",
    np.where(df["score"] >= 90, "A", "B")
)
print(df["grade"])
""" PRINT RESULT:
0          B
1          A
2          B
3    Missing
4          A
5          B
6          B
7          A
8          B
9          B
Name: grade, dtype: str
"""

# 4. Multiple Conditions (Still Vectorized)
df["grade"] = np.select(
    [
        df["score"].isna(),   # handle NaN FIRST
        df["score"] >= 90,
        df["score"] >= 80
    ],
    ["Missing", "A", "B"],
    default="C"
)
print(df["grade"])
""" PRINT RESULT:
0          B
1          A
2          C
3    Missing
4          A
5          C
6          B
7          A
8          C
9          B
Name: grade, dtype: str
"""

# 5. Summary
"""
Column math → vectorized

Avoid loops

apply() is last resort

Prefer NumPy functions when possible
"""