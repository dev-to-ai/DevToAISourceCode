########## Eleven. Data Cleaning with Numpy ##########

"""
Real-world data often contains:
Missing values (NaN)
Infinite values (inf, -inf)
Invalid entries after division or logs

NumPy gives you fast, vectorized tools to clean this safely.
"""

import numpy as np

# 1. np.isnan(a) — Detect Missing Values
a = np.array([1.0, 2.0, np.nan, 4.0])
mask = np.isnan(a)
print(mask)
""" PRINT RESULT:
[False False  True False]
"""
""" EXPLANATION:
NaN = Not a Number

Often comes from:
Missing data
Bad parsing
0 / 0 operations
Returns a boolean mask

AI Use:
Identify missing features
Remove or replace missing values
Prevent model failures
"""

# 2. np.isinf(a) — Detect Infinite Values 
a = np.array([1.0, np.inf, -np.inf, 5.0])
mask = np.isinf(a)
print(mask)
""" PRINT RESULT:
[False  True  True False]
"""
""" EXPLANATION:
inf often appears from:
Division by zero
Overflow
Log(0)

Models cannot train with infinite values

Inf values can silently destroy gradients.
"""

# 3. np.nan_to_num(a) — Replace Bad Values Safely
a = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
clean = np.nan_to_num(a)
print(clean)
""" PRINT RESULT:
[ 1.00000000e+000  0.00000000e+000  1.79769313e+308 -1.79769313e+308
  5.00000000e+000]
"""
clean = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
print(clean)
""" PRINT RESULT:
[ 1.  0.  1. -1.  5.]
"""
""" EXPLANATION:
Default replacements:
NaN → 0
+inf → max float
-inf → min float

Better (AI-friendly) version
clean = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
print(clean)

AI Use:
Prevent overflow
Keep features in reasonable ranges
Avoid exploding gradients
"""

# 4. np.where(condition, x, y) — Conditional Cleaning
a = np.array([1, -2, 3, -4, 5])
result = np.where(a < 0, 0, a)
print(result)
""" PRINT RESULT:
[1 0 3 0 5]
"""
""" EXPLANATION:
Vectorized if-else

For each element:
If condition is True → use x
Else → use y

AI Use:
Clip negative values
Replace outliers
Enforce constraints (e.g., no negative prices)
"""

# 5. Real AI Cleaning Pipeline Example
X = np.array([10.0, np.nan, np.inf, -5.0, 20.0])
# Step 1: Replace NaN and inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
# Step 2: Remove negatives
X = np.where(X < 0, 0, X)
print(X)
""" PRINT RESULT:
[10.  0.  0.  0. 20.]
"""

# 6. Quick Reference Cheat Sheet
"""
Function	    Purpose
np.isnan()	    Detect missing values
np.isinf()	    Detect infinite values
np.nan_to_num()	Replace NaN / inf
np.where()	    Vectorized condition
"""
