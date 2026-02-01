########## Nine. Linear Algebra ##########

import numpy as np

# 1. Dot Product & Matrix Multiplication
# Dot Product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))
# PRINT RESULT: 32
# Explanation: 1*4 + 2*5 + 3*6 = 32

# Matrix Multiplication
X = np.array([[1, 2],
              [3, 4]])
W = np.array([[0.1, 0.2],
              [0.3, 0.4]])
Z1 = np.dot(X, W)
Z2 = X @ W
Z3 = np.matmul(X, W)
print(Z1)
""" PRINT RESULT:
[[0.7 1. ]
 [1.5 2.2]]
"""
""" EXPLANATION:
All three give the same result
@ is the modern, readable standard
"""

# 2. Vector Norms (np.linalg.norm)
v = np.array([3, 4])
print(np.linalg.norm(v))
# PRINT RESULT: 5.0
""" EXPLANATION: 
The default norm (when no ord parameter is specified) is the L2 norm (Euclidean distance). 
L2 norm = √(3² + 4²) = √(9 + 16) = √25 = 5.0
"""
print(np.linalg.norm(v, ord=1))
# PRINT RESULT: 7.0
""" EXPLANATION: 
The L1 norm (also called Manhattan or Taxicab norm) is the sum of absolute values.
L1 norm = |3| + |4| = 3 + 4 = 7.0
"""

# 3. Inverse & Pseudo-Inverse

# Matrix Inverse
A = np.array([[4, 7],
              [2, 6]])
inv_A = np.linalg.inv(A)
print(inv_A)
""" PRINT RESULT:
[[ 0.6 -0.7]
 [-0.2  0.4]]
"""
""" EXPLANATION:
Step 1: Calculate the determinant
det(A) = (4 × 6) - (7 × 2)
       = 24 - 14
       = 10
Step 2: Apply the 2×2 inverse formula
For a 2×2 matrix:
[[a, b],
 [c, d]]^(-1) = (1/det) × [[d, -b],
                           [-c, a]]
A^(-1) = (1/10) × [[6, -7],
                   [-2, 4]]
Step 3: Simplify
A^(-1) = [[6/10, -7/10],
          [-2/10, 4/10]]
       = [[0.6, -0.7],
          [-0.2, 0.4]]
"""

# Pseudo-Inverse
pinv_A = np.linalg.pinv(A)
print(pinv_A)
""" PRINT RESULT:
[[ 0.6 -0.7]
 [-0.2  0.4]]
"""
""" EXPLANATION: 
For a full-rank square matrix like A, the pseudo-inverse (pinv) is the same as the regular inverse (inv).
np.linalg.inv(A) computes the exact inverse when A is square and non-singular (det ≠ 0)
np.linalg.pinv(A) computes the Moore-Penrose pseudo-inverse, which:
For square, non-singular matrices: Returns the exact inverse
For rectangular or singular matrices: Returns the "best fit" inverse using least squares
"""

# 4. Determinant (np.linalg.det)
# Tells you if a matrix is invertible.
print(np.linalg.det(A))
# PRINT RESULT: 10.000000000000002
""" EXPLANATION: 
det(A) = (4 × 6) - (7 × 2)
       = 24 - 14
       = 10
det = 0 → NOT invertible
Used in geometry & probability (Gaussian distributions)
"""

# 5. Eigenvalues & Eigenvectors (np.linalg.eig)
A = np.array([[2, 0],
              [0, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
# PRINT RESULT: [2. 1.]
print(eigenvectors)
""" PRINT RESULT:
[[1. 0.]
 [0. 1.]]
"""
""" EXPLANATION:
This finds eigenvalues and eigenvectors.
For [[2,0],[0,1]]:
Eigenvalues are 2.0 and 1.0 (scaling factors)
Eigenvectors are [1,0] and [0,1] (directions that don't change)
Eigenvectors are special directions.
When you multiply the matrix by an eigenvector, you get the same direction, just scaled.
"""

# 6. Singular Value Decomposition (SVD)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
U, S, Vt = np.linalg.svd(A)
print(U)
# PRINT RESULT: [9.52551809 0.51430058]
print(S)
# PRINT RESULT: [[-0.61962948 -0.78489445]
print(Vt)
#  PRINT RESULT: [-0.78489445  0.61962948]]
""" EXPLANATION:
SVD breaks a matrix into three simpler pieces:
U (left part), S (importance scores), Vt (right part).
"""

# 7. Aggregations (Model Decisions)
scores = np.array([0.1, 2.5, 0.3])
print(np.argmax(scores))  # index of predicted class
# PRINT RESULT: 1
""" EXPLANATION:
The array has scores for each class
argmax finds where the biggest number is
The biggest number is 2.5
Its position (index) is 1
"""