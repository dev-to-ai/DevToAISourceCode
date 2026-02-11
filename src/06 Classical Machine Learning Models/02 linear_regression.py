########## Two. Linear Regression (Regression problems) ##########
"""
What it really is:
Draw the best straight line through noisy data.
Not magic. Just averaging trends.

Statistical view
Linear regression assumes:
Relationship is roughly linear
Errors are random (noise)
No extreme multicollinearity

Even if assumptions are violated:
Predictions can still be OK
Interpretations become wrong
"""

# Visual intuition
# This code demonstrates a simple linear regression concept through visualization
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.random.rand(100) # 100 random numbers between 0 and 1
y = 3 * x + np.random.normal(0, 0.2, 100) # Follows the pattern y = 3x + noise (a perfect linear relationship plus random error)

plt.scatter(x, y)
plt.show()
# The model finds the line that minimizes average squared error.

# Residuals tell the truth
y_pred = 3 * x
residuals = y - y_pred
plt.scatter(x, residuals)
plt.axhline(0)
plt.show()
"""
ML thinking:
Random scatter → model OK
Pattern → model missing something
"""

"""
This is showing what linear regression does conceptually:
The actual data (scatter points): Has a general upward trend but with some randomness (noise)
What the model will do: Find the best straight line through these points
How it finds it: By minimizing the average squared error - meaning it finds the line where the sum of squared vertical distances between each point and the line is as small as possible
The perfect line would be y = 3x (since that's what generated the data), but the model will find something close to this based on the actual noisy data.
This is the fundamental concept behind linear regression - finding the line that best fits the data according to this "least squares" criterion.
"""

"""
When to use linear regression:
✅ Simple trends
✅ Interpretable relationships
❌ Complex interactions
❌ Heavy non-linearity
"""
