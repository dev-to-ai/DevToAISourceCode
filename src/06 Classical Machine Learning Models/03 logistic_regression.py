########## Three. Logistic Regression ##########
"""
What it really is: Linear regression + probability + threshold.

It does NOT predict classes.
It predicts probabilities.

Statistical intuition
Output = probability of class 1
Uses sigmoid curve to squash values into [0, 1]
Decision boundary is linear
"""

# Visual idea
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.linspace(-10, 10, 100) # Creates 100 evenly spaced points from -10 to 10 on the x-axis
y = 1 / (1 + np.exp(-x)) # Computes the sigmoid function

plt.plot(x, y) # Plots the characteristic S-shaped curve
plt.show()
# That's logistic regression's brain.
"""
This S-shaped curve is what allows logistic regression to make binary decisions (yes/no, 0/1, true/false).

How it works:
Input: Any real number from -∞ to +∞
Output: Always squeezed between 0 and 1
Decision boundary: At x=0, y=0.5 (the 50/50 point)

The "Thinking" Process:
1. Logistic regression calculates a linear combination of inputs (just like linear regression)
2. But instead of outputting that raw number directly, it passes it through this sigmoid function
3. The sigmoid converts the raw score into a probability between 0 and 1
4. If probability > 0.5, predict class 1; else predict class 0

Visual Intuition:
Far left (x = -10): Output ≈ 0 (very confident "no")
At center (x = 0): Output = 0.5 (completely uncertain)
Far right (x = 10): Output ≈ 1 (very confident "yes")
This smooth, probabilistic decision-making is why the sigmoid is the "brain" - it takes linear calculations and turns them into reasoned, probabilistic choices rather than hard cuts.
"""

