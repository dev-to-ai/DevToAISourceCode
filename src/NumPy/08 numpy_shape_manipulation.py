########## Eight. Shape Manipulation ##########

# 1. reshape() — Change the shape, keep the data

import numpy as np

a = np.arange(6)   # [0 1 2 3 4 5]
print(a.shape)
# PRINT RESULT: (6,)

b = a.reshape(2, 3)
print(b)
""" PRINT RESULT: 
[[0 1 2]
 [3 4 5]]
"""

print(b.shape)     
# PRINT RESULT: (2, 3)

print(a.reshape(3, -1))
""" PRINT RESULT:
[[0 1]
 [2 3]
 [4 5]]
"""
# Explanation: Use -1 to auto-calculate a dimension

# 2. flatten() vs ravel() — Matrix → Vector

# flatten() → always creates a copy (slow)
x = np.array([[1, 2], [3, 4]])
f = x.flatten()
f[0] = 999
print(x)  # unchanged
""" PRINT RESULT:
[[1 2]
 [3 4]]
"""

# ravel() → view if possible (faster)
r = x.ravel()
r[0] = 999
print(x)
""" PRINT RESULT:
[[999   2]
 [  3   4]]
"""

# 3. Transpose .T — Swap axes
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)
# PRINT RESULT: (2, 3)
print(A.T.shape)
# PRINT RESULT: (3, 2)

# 4. expand_dims() — Add a dimension
x = np.array([1, 2, 3])
print(x.shape)  
# PRINT RESULT: (3,)

y = np.expand_dims(x, axis=0)
print(y)
# PRINT RESULT: [[1 2 3]]
print(y.shape)  
# PRINT RESULT: (1, 3)

z = np.expand_dims(x, axis=1)
print(z)
""" PRINT RESULT:
[[1]
 [2]
 [3]]
"""
print(z.shape)  
# PRINT RESULT: (3, 1)

# 5. squeeze() — Remove size-1 dimensions
x = np.array([[[5]]])
print(x.shape)
# PRINT RESULT: (1, 1, 1)

y = np.squeeze(x)
print(y)
# PRINT RESULT: 5
print(y.shape)
# PRINT RESULT: ()

z = np.squeeze(x, axis=0)
print(z)
# PRINT RESULT: [[5]]
print(z.shape)
# PRINT RESULT: (1, 1)
