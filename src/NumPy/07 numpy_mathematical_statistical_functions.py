########## Seven. Mathematical & Statistical Functions ##########

"""
AI = Linear Algebra + Statistics on steroids.
These NumPy functions are the muscles your models use to think.
"""

# 1. Must-Know Aggregation Functions
import numpy as np

x = np.array([1, 2, 3, 4])

# Sum
print(np.sum(x))
# PRINT RESULT: 10

# Mean
print(np.mean(x))
# PRINT RESULT: 2.5

# Standard Deviation
print(np.std(x))
# PRINT RESULT: 1.118033988749895
"""
Mean: (1+2+3+4)/4 = 2.5
Differences from mean: [-1.5, -0.5, 0.5, 1.5]
Squared differences: [2.25, 0.25, 0.25, 2.25]
Variance: (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5 / 4 = 1.25
Std: √1.25 ≈ 1.118033988749895
"""

# Min
print(np.min(x))
# PRINT RESULT: 1

# Max
print(np.max(x))
# PRINT RESULT: 4

# Index of smallest value
print(np.argmin(x))
# PRINT RESULT: 0

# Index of largest value 
print(np.argmax(x))
# PRINT RESULT: 3

# 2. Axis-Based Operations

a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# axis = 0 → COLUMN-WISE
b = np.sum(a, axis=0)
print(b)
# PRINT RESULT: [5, 7, 9]

# axis = 1 → ROW-WISE
b = np.sum(a, axis=1)
print(b)
# PRINT RESULT: [ 6 15]

# 3. Element-Wise Operations

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(x + y)
# PRINT RESULT: [5 7 9]

print(x - y)
# PRINT RESULT: [-3 -3 -3]

print(x * y)
# PRINT RESULT: [ 4 10 18]

print(x / y)
# PRINT RESULT: [0.25 0.4  0.5 ]

# 4. Dot Product & Linear Algebra

print(x @ y )
# PRINT RESULT: 32
# Explanation: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32

x = np.array([1, 2, 3])
w = np.array([0.1, 0.2, 0.3])
print(np.dot(x, w))
# PRINT RESULT: 1.4
print(x @ w)
# PRINT RESULT: 1.4
""" Explanation:
The dot product multiplies corresponding elements and sums them:
1×0.1 + 2×0.2 + 3×0.3 
= 0.1 + 0.4 + 0.9 
= 1.4
"""

X = np.array([[1, 2],
              [3, 4]])
W = np.array([[0.1, 0.2],
              [0.3, 0.4]])
print(X @ W)
""" PRINT RESULT: 
[[0.7 1. ]
 [1.5 2.2]]
"""
""" Explanation:
@ = matrix multiplication (dot product between rows of X and columns of W)
Shape check (VERY IMPORTANT)
X → (2 × 2)
W → (2 × 2)
Inner dimensions match → multiplication is valid
Result shape → (2 × 2)
First row of result:
Row [1, 2] · columns of W
Column 1:
1*0.1 + 2*0.3 = 0.1 + 0.6 = 0.7
Column 2:
1*0.2 + 2*0.4 = 0.2 + 0.8 = 1.0
➡ [0.7, 1.0]
Second row of result:
Row [3, 4] · columns of W
Column 1:
3*0.1 + 4*0.3 = 0.3 + 1.2 = 1.5
Column 2:
3*0.2 + 4*0.4 = 0.6 + 1.6 = 2.2
➡ [1.5, 2.2]
Final Output:
array([[0.7, 1. ],
       [1.5, 2.2]])
"""

# Transpose
# Original matrix (2x3)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("Original A (2x3):")
print(A)
""" PRINT RESULT:
[[1 2 3]
 [4 5 6]]
"""

# Transpose (3x2)
A_T = A.T
print("\nTranspose A_T (3x2):")
print(A_T)
""" PRINT RESULT: 
[[1 4]
 [2 5]
 [3 6]]
"""

# 5. Broadcasting
X = np.array([[1, 2, 3],
              [4, 5, 6]])
mean = np.mean(X, axis=0) # axis = 0 → COLUMN-WISE
print(mean)
# PRINT RESULT: [2.5 3.5 4.5]
print(X - mean)  # broadcasting
""" PRINT RESULT:
[[-1.5 -1.5 -1.5]
 [ 1.5  1.5  1.5]]
"""

# 6. Statistical Operations (AI Foundations)

# Create a 3×3 matrix filled with random numbers drawn from a normal (Gaussian) distribution
""" Explanation:
np.random.normal(loc, scale, size)
loc (0): Mean of the distribution (center)
scale (1): Standard deviation (spread)
size ((3, 3)): Output shape
"""
a = np.random.normal(0, 1, size=(3, 3))
print(a)
""" PRINT RESULT: 
[[-0.9450216   1.07661874  0.3383785 ]
 [ 0.59388529 -1.30177485 -1.27317621]
 [ 2.15643586  2.42595687  0.7109561 ]]
"""

# Create a 3×3 matrix filled with random numbers drawn from a uniform distribution between -1 and 1
b = np.random.uniform(-1, 1, size=(3, 3))
print(b)
""" PRINT RESULT: 
[[-0.2944761   0.23599038 -0.89367805]
 [ 0.00543099  0.98432292  0.72760958]
 [ 0.41617069 -0.91841548 -0.85353504]]
"""

# Median & Percentile
x = np.array([[1, 2, 3],
              [4, 5, 6]])
print(np.median(x))
# PRINT RESULT: 3.5
""" Explanation: 
The median is the middle value when all values are sorted.
(3 + 4) / 2 = 3.5
"""
print(np.percentile(x, 90))
# PRINT RESULT: 5.5
""" Explanation: 
The 90th percentile means 90% of the data falls below this value.
Flatten and sort: [1, 2, 3, 4, 5, 6] (6 elements)
Position index = 90/100 * (n - 1) = 0.9 * 5 = 4.5
This means we need to interpolate between positions 4 and 5 (0-based indexing)
Value at position 4 (5th element): 5
Value at position 5 (6th element): 6
Interpolate: 5 + 0.5 × (6 - 5) = 5.5
"""