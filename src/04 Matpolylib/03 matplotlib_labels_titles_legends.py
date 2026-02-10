########## Three. Labels, Titles, Legends ##########

import numpy as np
import matplotlib.pyplot as plt

# 1. Axis Labels
x = np.arange(0, 10)
y = x ** 2
plt.plot(x, y)
plt.xlabel("X value")
plt.ylabel("Y value")
plt.show()

# 2. Title
plt.plot(x, y)
plt.xlabel("X value")
plt.ylabel("Y value")
plt.title("Quadratic Function")
plt.show()

# 3. Legends (When You Have More Than One Line)
plt.plot(x, x, label="y = x")
plt.plot(x, x**2, label="y = x²")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Quadratic Growth")
plt.legend() # activates labels
plt.show()

# 4. Multiple Lines: Clean Pattern
plt.plot(x, x, label="Linear")
plt.plot(x, x**2, label="Quadratic")
plt.plot(x, x**3, label="Cubic")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Growth Comparison")
plt.legend()
plt.show()

# 5. Legend Placement
plt.legend(loc="upper left")
"""
Common options:
best (This is default)
upper right
upper left
lower right
"""
