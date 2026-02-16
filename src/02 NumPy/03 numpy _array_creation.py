########## Three. NumPy Array Creation ##########

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
# Like Python range, but returns a NumPy array
# Stop value is exclusive

# 6. Evenly Spaced Values
L = np.linspace(0, 1, 5)
print(L)
# PRINT RESULT: [0.   0.25 0.5  0.75 1.  ]
"""
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
"""
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
"""
Mean = 0
Std = 1
Can be negative
"""

# 9. Random Integers
R_int = np.random.randint(0, 10, size=(3, 3))  # Random integers between 0 and 9
print(R_int)
""" PRINT RESULT:
[[3 7 2]
 [8 1 4]
 [5 9 6]]
"""
# Useful for creating test data with integer labels

# 10. Reproducible Random Numbers (Crucial for Research)
np.random.seed(42)  # Set seed for reproducibility
R1 = np.random.rand(2, 2)
np.random.seed(42)  # Reset seed
R2 = np.random.rand(2, 2)
print(f"Same with seed 42:\n{R1}\n{R2}")
""" PRINT RESULT:
Same with seed 42:
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
"""
# Essential for reproducible research and debugging

# 11. Empty Array (Uninitialized Memory)
E = np.empty((3, 3))  # Contains whatever was in memory (garbage values)
print(E)
""" PRINT RESULT (values will vary):
[[0.57463108 1.29282499 0.18530846]
 [1.23342803 0.52160828 0.48133836]
 [0.14395613 1.30150094 2.16119149]]
"""
# WARNING: Faster but contains arbitrary values - always overwrite before use!

# 12. Create Arrays Like Existing Ones
template = np.array([[1, 2], [3, 4]])

zeros_like = np.zeros_like(template)  # Same shape, filled with 0
ones_like = np.ones_like(template)    # Same shape, filled with 1
full_like = np.full_like(template, 9) # Same shape, filled with 9

print(f"Zeros like:\n{zeros_like}")
""" PRINT RESULT:
Zeros like:
[[0 0]
 [0 0]]
 """
print(f"Ones like:\n{ones_like}")
""" PRINT RESULT:
Ones like:
[[1 1]
 [1 1]]
"""
print(f"Full like:\n{full_like}")
""" PRINT RESULT:
Full like:
[[9 9]
 [9 9]]
"""
# Useful when you need same dimensions as existing data

# 13. Create from Function
def f(i, j):
    return i * 10 + j

arr_func = np.fromfunction(f, (3, 3), dtype=int)
print(f"From function:\n{arr_func}")
""" PRINT RESULT:
[[ 0  1  2]
 [10 11 12]
 [20 21 22]]
"""
# Useful for creating patterned arrays programmatically

# 14. Real AI/ML Use Cases

# Neural Network Weights Initialization
n_inputs, n_neurons = 784, 256

# Different initialization strategies
weights_xavier = np.random.randn(n_inputs, n_neurons) * np.sqrt(2/(n_inputs + n_neurons))
weights_he = np.random.randn(n_inputs, n_neurons) * np.sqrt(2/n_inputs)
weights_uniform = np.random.uniform(-0.1, 0.1, (n_inputs, n_neurons))

print(f"Xavier init std: {weights_xavier.std():.4f}")
# PRINT RESULT: Xavier init std: 0.0438
print(f"He init std: {weights_he.std():.4f}")
# PRINT RESULT: He init std: 0.0505

# Bias initialization
biases = np.zeros((1, n_neurons))  # Usually initialized to zero

# Training data batch
batch_size, features = 32, 784
batch_data = np.random.randn(batch_size, features)  # Random batch

# One-hot encoding (common for classification)
n_samples, n_classes = 5, 3
labels = np.array([0, 2, 1, 0, 2])
one_hot = np.eye(n_classes)[labels]
print(f"One-hot encoded:\n{one_hot}")
""" PRINT RESULT:
One-hot encoded:
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
 """
# Learning rate schedule
initial_lr, epochs = 0.1, 50
lr_schedule = np.linspace(initial_lr, initial_lr/100, epochs)
print(f"Learning rate decay: {lr_schedule[:5]}...")
# PRINT RESULT: Learning rate decay: [0.1        0.09797959 0.09595918 0.09393878 0.09191837]...

# 15. SUMMARY:
"""
Function	        Distribution / Value	    AI Use Case
zeros	            All 0	                    Biases, initialization
ones	            All 1	                    Masks, initialization
full	            Constant	                Testing, constant values
eye	                Identity	                Linear algebra, one-hot encoding
arange	            Step range	                Indexing, loops
linspace	        Even spacing	            Learning rate schedules
logspace	        Logarithmic spacing	        Hyperparameter search
rand	            Uniform [0,1)	            Random initialization
randn	            Normal (0,1)	            Neural network weights
randint	            Random integers	            Test labels, indices
empty	            Garbage values	            Fast allocation (overwrite!)
zeros_like	        Same shape as input	        Maintain dimensions
ones_like	        Same shape as input	        Maintain dimensions
diag	            Diagonal matrix	            Feature extraction
meshgrid	        2D coordinate grids	        Gradient computation, visualization
fromfunction	    Custom function	            Patterned arrays
loadtxt/genfromtxt	File loading	            Data import
"""



