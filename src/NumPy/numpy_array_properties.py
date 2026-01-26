##### Two. NumPy Arrays Properties (ndarray)

# 1. A NumPy array (ndarray) is:
# A same data type container
# Stored in contiguous memory
# Designed for multi-dimensional math

# 2. Create a 2D NumPy array (matrix)
import numpy as np
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr)
""" PRINT RESULT:
[[1 2 3]
 [4 5 6]]
"""
print(arr *2)
""" PRINT RESULT:
[[ 2  4  6]
 [ 8 10 12]]
"""

# 3. Core array attributes (ndarray)

# .ndim — Number of Dimensions
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a.ndim)  
# PRINT RESULT: 1
print(b.ndim)  
# PRINT RESULT: 2
""" EXPLANATION:
ndim	Meaning
1	    Vector
2	    Matrix
3+	    Tensors (DL)
"""

# .shape — Size of Each Dimension
print(a.shape)  
# PRINT RESULT: (3,)
print(b.shape)  
# PRINT RESULT: (2, 2)

# .size — Total Number of Elements
print(a.size)  
# PRINT RESULT: 3
print(b.size)  
# PRINT RESULT: 4

# .dtype — Data Type
c = np.array([1, 2, 3])
d = np.array([1.0, 2.0, 3.0])

print(c.dtype)  
# PRINT RESULT: int64
print(d.dtype)  
# PRINT RESULT: float64

arr = np.array([1, 2, 3], dtype=np.float32)
print(arr)
# PRINT RESULT: [1. 2. 3.]
# EXPLANATION:
# Deep learning prefers float32 because of memory efficiency
# Common dtypes: int32, float32, float64

