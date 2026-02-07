########## Two. Pandas Data Structure ##########

import pandas as pd

# 1. The Big Picture
"""
Series: A one-dimensional labeled array (like a column in a spreadsheet).

DataFrame: A two-dimensional, table-like structure with labeled rows and columns (similar to an Excel sheet or SQL table). This is the most commonly used pandas object.
"""

# 2. Pandas Series (1D, Single Column)
"""
What a Series Is
One-dimensional

Has:
values (NumPy array)
index (labels)
dtype
"""

# In NumPy, we access data by postion
import numpy as np
a = np.array([10, 20, 30])
print(a[0])  
# PRINT RESULT: 10

# In Pandas, we access data by meaning, not positions
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s)
""" PRINT RESULT:
a    10
b    20
c    30
dtype: int64
"""
print(s["a"]) # label-based
# PRINT RESULT: 10
""" EXPLANATION:
Index = identity, not position
Index can be: integers, strings, dates, multi-level (advanced)
"""
print(s.iloc[0]) # position-based
# PRINT RESULT: 10
print(s[s > 15])
""" PRINT RESULT:
b    20
c    30
dtype: int64
"""
# Anatomy of a Series
print(s.values) # NumPy array
# PRINT RESULT: [10 20 30]
print(s.index) # Index object
# PRINT RESULT: Index(['a', 'b', 'c'], dtype='str')
print(s.dtype) # data type
# PRINT RESULT: int64

# 3. DataFrame (2D, Multiple Columns)
df = pd.DataFrame({
    "age": [25, 30, 35],
    "score": [88, 92, 79]
})
print(df)
""" PRINT RESULT:
   age  score
0   25     88
1   30     92
2   35     79
"""
print(df["age"])
""" PRINT RESULT:
0    25
1    30
2    35
"""
print(df[["age", "score"]])
""" PRINT RESULT:
   age  score
0   25     88
1   30     92
2   35     79
"""
# DataFrame Anatomy
print(df.values) # 2D NumPy array
""" PRINT RESULT:
[[25 88]
 [30 92]
 [35 79]]
"""
print(df.index) # row labels
# PRINT RESULT: RangeIndex(start=0, stop=3, step=1)
print(df.columns) # column labels
# PRINT RESULT: Index(['age', 'score'], dtype='str')
print(df.dtypes) # column data types
""" PRINT RESULT:
age      int64
score    int64
dtype: object
"""

# 4. Relationship: DataFrame = Multiple Series

# Each column is a Series
print(type(df["age"]))
# PRINT RESULT: <class 'pandas.Series'>

# Series Share the Same Index
print(df["age"].index == df["score"].index)
# PRINT RESULT: [ True  True  True]

# 5. Selecting: Series vs DataFrame
# Single column → Series
print(df["age"])
# Multiple columns → DataFrame
print(df[["age"]])
""" EXPLANATION:
Why it matters:
Series → 1D
DataFrame → 2D
ML models often expect 2D input
"""

# 6. Row Selection Differences
# Series (1D)
s.iloc[0] # 10
s.loc["a"] # 10
# DataFrame (2D)
print(df.iloc[0]) # Selecting a row gives you a Series
""" PRINT RESULT:
age      25
score    88
Name: 0, dtype: int64
"""
df.iloc[0, 1] # 88
print(df.loc[0, "age"]) # 25

# 7. Broadcasting & Alignment Differences
# Series alignment
s1 = pd.Series([10, 20, 30], index=["a", "b", "c"])
s2 = pd.Series([1, 2, 3], index=["b", "c", "d"])
print(s1 + s2)
""" PRINT RESULT:
a     NaN
b    21.0
c    32.0
d     NaN
dtype: float64
"""
""" EXPLANATION: 
Pandas aligns by label, not position. 
Match meaning first, then compute.
"""
# DataFrame alignment (column-wise)
print(df + 10) # Adds to every element
""" PRINT RESULT:
   age  score
0   35     98
1   40    102
2   45     89
"""
print(df["age"] + df["score"]) # Aligns row-wise by index
"""
0    113
1    122
2    114
dtype: int64
"""

# 8. Adding Data: Series vs DataFrame
# Add a new column (Series → DataFrame)
df["passed"] = df["score"] > 80
print(df)
""" PRINT RESULT:
   age  score  passed
0   25     88    True
1   30     92    True
2   35     79   False
"""
# Add a new row
df.loc[3] = [40, 85, True]
print(df)
""" PRINT RESULT:
   age  score  passed
0   25     88    True
1   30     92    True
2   35     79   False
3   40     85    True
"""

# 9. When Should You Use Series Directly?
"""
Use Series when:
working on one column
applying math or transformations
generating labels (y in ML)

Use DataFrame when:
selecting multiple features
filtering rows
joining datasets
feeding ML models
"""

# 10. Summary
"""
Series is a labeled vector.
DataFrame is a table of aligned Series.

One column → Series
Many columns → DataFrame
Column selection returns Series
Row selection returns Series
DataFrame = aligned Series
"""


