########## Four. Figure & Axes ##########

import matplotlib.pyplot as plt
import numpy as np

# 1. The Big Picture
"""
Figure  → the whole image / canvas
Axes    → one chart (x-axis + y-axis)
Plot    → lines, bars, points drawn on axes
"""
"""
One Figure
One or more Axes
Axes contain the actual data visuals
"""

# 2. The Old Way
x = np.arange(0, 10)
y = x ** 2
plt.plot(x, y)
plt.title("My Plot")
plt.show()
"""
What is happening behind the scenes:
Matplotlib secretly creates a Figure
Matplotlib secretly creates Axes
plt keeps track of "current" ones
This is fine for small scripts — bad for complex plots.
"""

# 3. The Professional Way (Explicit, OO Style)
x = np.arange(0, 10)
y = x ** 2
fig, ax = plt.subplots()
"""
fig → the Figure object (the whole canvas)
ax → one Axes (one chart area)
"""
ax.plot(x, y)
ax.set_title("Quadratic Function")
ax.set_xlabel("X value")
ax.set_ylabel("Y value")
plt.show()
"""
With OO style you can:
Control multiple plots
Avoid state bugs
Customize each chart independently
Read other people's code
Rule: If your plot has more than one axes, use OO style.
"""

# 4. Multiple Axes (Subplots)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, x)
axes[0].set_title("y = x")
axes[1].plot(x, x**2)
axes[1].set_title("y = x²")
plt.show()
"""
axes is now an array
Index it like NumPy
"""

# 5. Mixed Plot Types
x = np.arange(0, 10)
data = np.random.randn(500)
fig, axes = plt.subplots(2, 1, figsize=(6, 8))
axes[0].plot(x, x**2)
axes[0].set_title("Line Plot: y = x²")
axes[1].hist(data, bins=20)
axes[1].set_title("Histogram of Random Data")
plt.show()

# 6. Comparing plt vs ax
"""
Task	        plt	                ax
Plot data	    plt.plot()	        ax.plot()
Title	        plt.title()	        ax.set_title()
X label	        plt.xlabel()	    ax.set_xlabel()
Y label	        plt.ylabel()	    ax.set_ylabel()
Legend	        plt.legend()	    ax.legend()
Same result — different control level.
"""

# 7. When You Still Use plt
"""
Even in OO style:
plt.show()
plt.subplots()
plt.close()
Everything else → ax
"""



