##### Four. NumPy Indexing & Slicing

import numpy as np

# 2 rows and 3 columns
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)  
# PRINT RESULT: (2, 3)

# 1. Basic Indexing (Single Elements)
print(a[0, 1])
# PRINT RESULT: 2

# 2. Row & Column Selection
# Get a full row
print(a[1, :])
# PRINT RESULT: [4 5 6]
print(a[[0]])
# PRINT RESULT: [[1 2 3]]

# Get a full column
print(a[:, 2])
# PRINT RESULT: [3 6]

# 3. Basic Slicing
# Syntax: arr[start:end]

# First row only
print(a[0:1, :])
# PRINT RESULT: [[1 2 3]]

# First two columns
print(a[:, 0:2])
""" PRINT RESULT:
[[1 2]
 [4 5]]
"""

# 4. Multi-Dimensional Slicing
print(a[0:2, 1:3])
""" PRINT RESULT:
[[2 3]
 [5 6]]
"""
""" EXPLANATION:
Explanation:
Rows 0 to 1
Columns 1 to 2
"""

# 5. Boolean Indexing (MASKING)
print(a>3)
""" PRINT RESULT:
[[False False False]
 [ True  True  True]]
"""

print(a[a > 3])
# PRINT RESULT: [4 5 6]

# AI Example: Remove bad data
data = np.array([0.1, -0.2, 0.5, 0.9, -0.1])
clean_data = data[data > 0]
print(clean_data)
# PRINT RESULT: [0.1 0.5 0.9]

# 6. Boolean Masking with Rows
scores = np.array([60, 85])
passed = scores >= 70
print(passed)
# PRINT RESULT: [False  True]
print(scores[passed])
# PRINT RESULT: [85]
print(a[passed])
# PRINT RESULT: [[4 5 6]]
""" EXPLANATION:
When you index a 2D array a with a 1D boolean array [False, True]:
NumPy interprets the boolean array as row selection
It selects rows from a where the boolean value is True
Since passed = [False, True]:
Row 0: False → skip
Row 1: True → select
"""

# 7. Fancy Indexing (Selecting Specific Indices)
print(a[[0, 1], [1, 2]])
# PRINT RESULT: [2 6]
""" EXPLANATION:
Explanation:
(0,1) → 2
(1,2) → 6
"""
