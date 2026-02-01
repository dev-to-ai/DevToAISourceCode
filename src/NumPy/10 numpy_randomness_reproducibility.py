########## Ten. Numpy Randomness & Reproducibility ##########

import numpy as np

# 1. Why Randomness Exists in AI

"""
Randomness is used for:
Initializing neural network weights
Shuffling data
Train / test splits
Data augmentation
Sampling

Problem:
If randomness changes every run → your results change → impossible to compare models.

Solution: control randomness with a seed
"""

# 2. np.random.seed() — Reproducibility
np.random.seed(42)
print(np.random.rand(5))
""" PRINT RESULT:
[0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
"""
print(np.random.rand(5))
""" PRINT RESULT:
[0.15599452 0.05808361 0.86617615 0.60111501 0.70807258]
"""
""" EXPLANATION:
seed(42) locks the random number generator
Every run produces the same “random” numbers
Different seed → different numbers
Same seed → same experiment
Think of the seed as a starting point in a random number timeline.
"""

# 3. np.random.rand() — Uniform [0, 1)
np.random.seed(42)
x = np.random.rand(5)
print(x)
""" PRINT RESULT:
[0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
"""
""" EXPLANATION:
Generates numbers uniformly between 0 and 1
Shape is defined by arguments

Common for:
Weight initialization
Probabilities
Dropout masks

Distribution: flat / even
"""

# 4. np.random.randn() — Standard Normal (Gaussian)
np.random.seed(42)
x = np.random.randn(5)
print(x)
""" PRINT RESULT:
[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]
"""
""" EXPLANATION:
Mean = 0
Standard deviation = 1
Values can be negative or positive
Bell-shaped curve

In AI:
Very common for neural network weight initialization
Works well with gradient-based learning
"""

# 5. np.random.randint() — Discrete Integers
np.random.seed(42)
x = np.random.randint(0, 10, size=5)
print(x)
""" PRINT RESULT:
[6 3 7 4 6]
"""
""" EXPLANATION:
Random integers
Lower bound is inclusive
Upper bound is exclusive

Used for:
Random labels
Index sampling
Mini-batch selection
"""

# 6. Summary Table (Old API)
"""
Function	What it Generates	Typical AI Use
rand()	    Uniform [0, 1)	    Probabilities, weights
randn()	    Normal (0, 1)	    NN initialization
randint()	Integers	        Sampling, indexing
seed()	    Reproducibility	    Fair comparison
"""

# 7. Best Practice (NEW WAY): default_rng()
# NumPy now recommends Generator API instead of global state.
rng = np.random.default_rng(seed=42)
print(rng.random(5)) # like rand()
""" PRINT RESULT:
[0.77395605 0.43887844 0.85859792 0.69736803 0.09417735]
"""
print(rng.standard_normal(5)) # like randn()
""" PRINT RESULT:
[-1.30217951  0.1278404  -0.31624259 -0.01680116 -0.85304393]
"""
print(rng.integers(0, 10, size=5)) # like randint()
""" PRINT RESULT:
[5 3 1 9 7]
"""
