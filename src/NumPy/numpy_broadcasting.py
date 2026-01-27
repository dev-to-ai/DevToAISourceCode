##### Five. Broadcasting

"""
Broadcasting lets NumPy perform element-wise operations on arrays of different shapes without copying data.

Instead of forcing you to manually reshape arrays, NumPy automatically stretches dimensions of size 1 to match the other array.

This is why NumPy is fast, clean, and perfect for AI.
"""

# 1. Broadcasting Rules:
"""
When operating on two arrays:

Rule 1: Prepend 1s
If arrays have different number of dimensions, prepend 1s to the smaller one.
(3, 3) and (3,)
(3,) → becomes (1, 3)

Rule 2: Dimensions must match OR be 1
For each dimension:
Equal → OK
One of them is 1 → OK (will broadcast)
Otherwise → ERROR

Rule 3: Size-1 dimensions stretch
Dimensions with size 1 are copied virtually to match the other array.
No real memory copy happens → this is why NumPy is fast.
"""

# 2. Example 1: Scalar + Matrix (Easiest Case)
import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])
result = A + 10
print(result)
""" PRINT RESULT:
[[11 12 13]
 [14 15 16]]
"""
""" EXPLANATION:
10 → shape () → treated as (1, 1)
Broadcast to (2, 3)
"""

# 3. Example 2: (3,3) Matrix + (3,) Vector
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
v = np.array([10, 20, 30])
print(A + v)
""" PRINT RESULT:
[[11 22 33]
 [14 25 36]
 [17 28 39]]
"""
""" EXPLANATION:
(3,3)
(1,3) → stretched to (3,3)
"""

# 4. Example 3: Column Vector Broadcasting (VERY COMMON IN AI)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
col = np.array([[10],
                [20]])
print(A + col)
""" PRINT RESULT:
[[11 12 13]
 [24 25 26]]
"""
""" EXPLANATION:
A → (2, 3)
col → (2, 1), broadcast second dimension
"""

# 5. Example 4: Why This FAILS
A = np.ones((3, 3))
b = np.array([1, 2])
# A + b
""" EXPLANATION:
A: Shape (3, 3) - a 3×3 matrix 
b: Shape (2,) - a 1D array with 2 elements
3 vs 2 → incompatible
ValueError: operands could not be broadcast together with shapes (3,3) (2,)
"""

# 6. Visual Mental Model
"""
Broadcasting is like:
Can NumPy stretch size-1 dimensions without changing meaning?
If yes → works
If no → error
"""