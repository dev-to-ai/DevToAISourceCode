########## Twelve. File I/O with NumPy ##########

"""
In AI, File I/O is mainly about:
Saving datasets
Reloading preprocessed data
Storing early model outputs / checkpoints
"""

# 1. np.save() / np.load() — NumPy Binary Files (.npy)
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
np.save("data.npy", a)
b = np.load("data.npy")
print(b)
""" PRINT RESULT:
[[1 2 3]
 [4 5 6]]
"""
""" EXPLANATION:
.npy is NumPy's native binary format

Preserves:
Shape
Data type
Exact values
Fast read/write
No precision loss

AI Use:
Save preprocessed tensors
Cache feature matrices
Store embeddings

Best choice for internal AI pipelines
"""

# 2. Why .npy Is Better Than CSV (for AI)

"""
Feature	        .npy	        .csv
Speed           Fast	        Slow
Precision	    Exact	        Can lose precision
Shape	        Preserved	    Lost
Data types	    Preserved	    Strings only

Use .npy for machines, .csv for humans.
"""

# 3. np.loadtxt() — Load CSV into NumPy
data = np.loadtxt("data.csv", delimiter=",")
print(data)
""" PRINT RESULT:
[[ 1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10.]
 [11. 12. 13. 14. 15.]
 [16. 17. 18. 19. 20.]]
"""
""" EXPLANATION:
Reads text-based numeric data

Assumes:
All values are numeric
Clean formatting
No headers by default

Will crash if:
Missing values
Strings
Inconsistent columns

AI Use:
Small datasets
Clean numeric features
Quick experiments
"""

# 4. np.savetxt() — Save NumPy Array to CSV
a = np.array([[1.5, 2.3, 3.1],
              [4.0, 5.2, 6.8]])
np.savetxt("out.csv", a, delimiter=",")
""" EXPLANATION:
Converts array → text
Easy to open in Excel
Slower and larger files

Use when:
Sharing data
Debugging
Inspecting values manually
"""

# 5. Real AI Workflow Example
X = np.random.randn(1000, 20)
# expensive cleaning / feature engineering
X = (X - X.mean(axis=0)) / X.std(axis=0)
np.save("X_clean.npy", X)
X = np.load("X_clean.npy")
# model training starts here

# 6. Quick Cheat Sheet
"""
Task	    Function
Save array	np.save()
Load array	np.load()
Load CSV	np.loadtxt()
Save CSV	np.savetxt()
"""

# 7. AI Rule of Thumb
"""
If the data is for training, use .npy
If the data is for humans, use .csv
"""


