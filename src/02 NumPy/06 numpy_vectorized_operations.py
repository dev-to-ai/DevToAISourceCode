########## Six. Vectorized Operations ##########

# Vectorization means performing operations on entire arrays at once, instead of looping element by element in Python.

import numpy as np

# 1. Element-Wise Operations

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Addition (element-wise)
print(x + y)
# PRINT RESULT: [5, 7, 9]

# Multiplication (element-wise)
print(x * y)
# PRINT RESULT: [ 4, 10, 18]

# Apply a function to all elements
a = np.sqrt(x)
print(a)
# PRINT RESULT: [1.         1.41421356 1.73205081]

# 2. Compare: Loop vs Vectorized

# Slow & ugly (Python loop)
result = []
for i in range(len(x)):
    result.append(x[i] + y[i])

# Fast - Same result, 10–100× faster for large arrays.
result = x + y

# 3. More Common Vectorized Operations

x = np.array([1, 2, 3, 4])
print(x ** 2)
# PRINT RESULT: [ 1  4  9 16]

print(np.log(x))
# PRINT RESULT:[0.         0.69314718 1.09861229 1.38629436]

print(np.exp(x))
# PRINT RESULT: [ 2.71828183  7.3890561  20.08553692 54.59815003]

print(x > 2)
# PRINT RESULT: [False, False,  True,  True])

# 4. Vectorization on 2D Data

X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Scale features
print(X * 0.1)
""" PRINT RESULT:
[[0.1 0.2 0.3]
 [0.4 0.5 0.6]]
"""

# Add bias
print(X + 1)
""" PRINT RESULT:
[[2 3 4]
 [5 6 7]]
"""

# Normalize
print(X / np.max(X))
""" PRINT RESULT:
[[0.16666667 0.33333333 0.5       ]
 [0.66666667 0.83333333 1.        ]]
"""

# 5. Summary
"""
Vectorized operations = operate on entire arrays

Element-wise math is automatic

Avoid Python loops at all costs

Faster, cleaner, scalable ML code
"""