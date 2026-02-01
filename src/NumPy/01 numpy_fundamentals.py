########## One. NumPy Fundamentals ##########

"""
What is NumPy?
NumPy (Numerical Python) is a core Python library for numerical computing.
It is the foundation for almost all data science, machine learning, and AI work in Python.

NumPy provides:
Fast, memory-efficient numerical operations
A powerful N-dimensional array object (ndarray)
Tools for math, statistics, linear algebra, and random numbers

Why NumPy Is Important (Especially for AI)?
Much faster than Python lists (written in C)
Handles large datasets efficiently
Basis for Pandas, SciPy, scikit-learn, TensorFlow, PyTorch
Used for data preprocessing, feature engineering, and math operations in AI
If you’re learning AI or ML, NumPy is mandatory.

Installation & Setup
Make sure you have Python installed.
Istallation: pip install numpy
Check: print(np.__version__)
"""

import numpy as np

# Create a NumPy array
a = np.array([1,2,3])
print(a)
# PRINT RESULT: [1 2 3]

# Vectorization - critical for AI performance
b= np.array([4,5,6])
print(a+b)
# PRINT RESULT: [5 7 9]

