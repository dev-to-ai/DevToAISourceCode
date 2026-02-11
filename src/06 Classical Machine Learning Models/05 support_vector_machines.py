########## Five. Support Vector Machines ##########

"""
What it really is
Draw the safest possible boundary.
Think of SVM as finding the best street to separate two neighborhoods.

Not just separating classes — maximizing margin.

Intuition
Only a few points matter (support vectors)
Boundary depends on the hardest cases
Ignores easy points far away

Linear SVM idea
What line stays farthest away from both classes?

Kernel trick (intuition only)
Pretend data lives in higher dimensions where it's separable.

Real-World Analogy:
Imagine two groups of people in a field:
Linear SVM: Dig a straight trench between them
Polynomial SVM: Dig a curved trench
RBF SVM: Build a circular fence around one group
The support vectors = the people standing closest to the trench
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# CREATE the data
X, y = make_classification(n_samples=200, n_features=2, 
                           n_redundant=0, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Train SVM
model = SVC(kernel='linear')
model.fit(X, y)

# Linear boundary
svm_linear = SVC(kernel='linear')

# Non-linear boundaries
svm_poly = SVC(kernel='poly', degree=3)     # Polynomial
svm_rbf = SVC(kernel='rbf', gamma='scale')  # Radial Basis Function (most popular)