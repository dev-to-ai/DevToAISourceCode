########## Five. Subplots, Layout & Figure Size ##########

# 1. Figsize (Stop Guessing Plot Size)
"""
fig, ax = plt.subplots(figsize=(8, 4))
Units: inches
(width, height)
Typical sizes:
(6, 4) → report figure
(10, 4) → wide comparison
(12, 8) → dashboard
"""

# 2. Subplots Grid (General Form)
"""
fig, axes = plt.subplots(rows, cols)
Examples:
plt.subplots(2, 2)   # 2×2 grid
plt.subplots(3, 1)   # vertical stack
plt.subplots(1, 3)   # horizontal
"""

# 3. Understanding axes Shape
"""
1 subplot:
fig, ax = plt.subplots()
ax → single Axes object
"""
"""
Multiple subplots:
fig, axes = plt.subplots(2, 2)
axes → 2D array
Indexing:
axes[0, 0]
axes[0, 1]
axes[1, 0]
axes[1, 1]
"""

# 4. 2×2 Grid Example
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

axes[0, 0].plot(x, x)
axes[0, 0].set_title("y = x")

axes[0, 1].plot(x, x**2)
axes[0, 1].set_title("y = x²")

axes[1, 0].plot(x, x**3)
axes[1, 0].set_title("y = x³")

axes[1, 1].hist(np.random.randn(500), bins=20)
axes[1, 1].set_title("Histogram")

plt.tight_layout() # auto-spacing
plt.show()

# 5. Sharing Axes
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(x, x)
axes[0].set_title("Linear")
axes[1].plot(x, x**2)
axes[1].set_title("Quadratic")
plt.show()


# 6. All Core Plot Types
x = np.arange(0, 10)
y = x ** 2
data = np.random.randn(500)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Line
axes[0, 0].plot(x, y)
axes[0, 0].set_title("Line Plot")

# Scatter
axes[0, 1].scatter(x, y)
axes[0, 1].set_title("Scatter Plot")

# Bar
axes[1, 0].bar(["A", "B", "C"], [10, 20, 15])
axes[1, 0].set_title("Bar Chart")

# Histogram
axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title("Histogram")

plt.tight_layout()
plt.show()


# 7. Dashboard Mini-Project
# One large plot on top, two smaller plots below
x = np.arange(0, 10)
data = np.random.randn(500)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Top plot spans two columns
axes[0, 0].plot(x, x**2)
axes[0, 0].set_title("Main Trend")

axes[0, 1].axis("off")  # turn off unused plot

# Bottom left
axes[1, 0].hist(data, bins=20)
axes[1, 0].set_title("Distribution")

# Bottom right
axes[1, 1].scatter(x, x)
axes[1, 1].set_title("Relationship")

plt.tight_layout()
plt.show()