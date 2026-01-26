##### Three. NumPy Array Creation

import numpy as np

# 1. Zero Matrix
Z = np.zeros((3, 3)) # Default dtype = float64
print(Z)
""" PRINT RESULT:
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
"""

bias = np.zeros((1, 10))  # 10-class classifier
print(bias)
# PRINT RESULT: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

arr = np.zeros((2, 10))  # 10-class classifier
print(arr)
""" PRINT RESULT:
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
"""

# 2. Ones Matrix

O = np.ones((2, 4)) # Filled with 1
print(O)
""" PRINT RESULT:
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]]
"""

# 3. Constant Value Array

F = np.full((2, 2), 7) # Every element is 7
print(F)
""" PRINT RESULT:
[[7 7]
 [7 7]]
"""

# 4. Identity Matrix

I = np.eye(3) # Diagonal = 1 and others = 0
print(I)
""" PRINT RESULT:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
"""

# 5. Range with Step

A = np.arange(0, 10, 2)
print(A)
# PRINT RESULT: [0 2 4 6 8]
# EXPLANATION:
# Like Python range, but returns a NumPy array
# Stop value is exclusive

# 6. Evenly Spaced Values

L = np.linspace(0, 1, 5)
print(L)
# PRINT RESULT: [0.   0.25 0.5  0.75 1.  ]
""" EXPLANATION:
Includes start and end
Fixed number of samples
"""

# 7. Uniform Distribution [0, 1)

R = np.random.rand(3, 3)
print(R)
""" PRINT RESULT:
[[0.47391027 0.12528521 0.10127426]
 [0.05530917 0.40850112 0.75974862]
 [0.90942298 0.0812154  0.90431228]]
"""
""" EXPLANATION:
Random numbers from uniform distribution
Range: [0, 1)
"""

# 8. Standard Normal Distribution

N = np.random.randn(3, 3)
print(N)

""" PRINT RESULT:
[[ 0.44920346  1.05888032 -1.26089291]
 [-0.35706769  0.25559797  0.47056974]
 [ 0.08777172 -0.27083336 -0.26122446]]
"""
""" EXPLANATION:
Mean = 0
Std = 1
Can be negative
"""

""" SUMMARY:
Function	Distribution / Value	AI Use Case
zeros	    All 0	                Biases, init
ones	    All 1	                Masks
full	    Constant	            Testing
eye	        Identity	            Linear algebra
arange	    Step range	            Indexing
linspace	Even spacing	        Schedules
rand	    Uniform [0,1)	        Random init
randn	    Normal (0,1)	        NN weights
"""



