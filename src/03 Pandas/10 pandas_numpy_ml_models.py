########## Pandas → NumPy → ML Models ##########

""" 
This is the bridge from data analysis → machine learning
Big picture:
Pandas (clean, labeled data)
        ↓
NumPy (raw arrays)
        ↓
ML frameworks (fast math)

ML libraries do NOT care about column names — they want numbers in arrays.
"""

import pandas as pd
import numpy as np

# 1. Converting Pandas to NumPy
df = pd.read_csv("students.csv")
# Features (X)
X = df[["age", "score"]].values
print(X)
# or (preferred, modern, recommended way: to_numpy())
X = df[["age", "score"]].to_numpy()
print(X)
""" PRINT RESULT:
[[23. 88.]
 [27. 92.]
 [nan 79.]
 [31. nan]
 [25. 95.]
 [29. 67.]
 [22. 85.]
 [34. 90.]
 [28. 73.]
 [26. 89.]]
"""

# 2. Why ML needs NumPy (not Pandas)

"""
Pandas:
Column names
Mixed dtypes
Missing values
Human-readable

NumPy:
Homogeneous arrays
Fast vectorized math
Memory efficient
GPU-friendly (via bridges)

ML models operate on matrices, not tables.
scikit-learn, PyTorch, TensorFlow / Keras accept NumPy arrays.
"""