########## Seven. Styling & Saving ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Figure Styles
print("Available styles:", plt.style.available)
""" PRINT RESULT:
Available styles: ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
"""
"""
Matplotlib has built-in styles:
plt.style.use('seaborn-v0_8')   # subtle, colored
Put before any plot
Can mix with OO style
"""

# 2. Colors, Line Styles, and Markers
x = np.arange(0, 10)
y1 = x
y2 = x**2

fig, ax = plt.subplots()

ax.plot(x, y1, color='blue', linestyle='--', marker='o', label="Linear")
ax.plot(x, y2, color='red', linestyle='-', marker='s', label="Quadratic")

ax.set_title("Styled Lines")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()
"""
color = line color
linestyle = solid/dashed/etc
marker = shape for points
label + legend() = clarity
"""

# 3. Gridlines
"""
ax.grid(True)
Makes reading values easier
Combine with style for polished look
"""

# 4. Saving Figures
"""
fig.savefig("my_plot.png", dpi=300)
fig.savefig("my_plot.pdf")
PNG → image
PDF → vector (scalable)
dpi controls resolution
"""

# 5. Advanced Layout (Mini-Dashboard)
x = np.arange(0, 10)
y = x**2
y2 = x**3
data = np.random.randn(500)

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
plt.style.use('seaborn-v0_8')

# Line plot
axes[0, 0].plot(x, y, marker='o', label="y=x²")
axes[0, 0].set_title("Quadratic")
axes[0, 0].legend()

# Cubic
axes[0, 1].plot(x, y2, marker='s', color='red', label="y=x³")
axes[0, 1].set_title("Cubic")
axes[0, 1].legend()

# Histogram
axes[1, 0].hist(data, bins=20, color='purple')
axes[1, 0].set_title("Random Data")

# Scatter
axes[1, 1].scatter(x, y, color='green', label="y=x² points")
axes[1, 1].set_title("Scatter Plot")
axes[1, 1].legend()

plt.tight_layout()
fig.savefig("dashboard.png", dpi=300) # Saves to the projet root folder.
plt.show()

# 6. Exercise 1: Styled Line Plot
# Sample data
x = np.arange(0, 10)
y1 = x
y2 = x**2

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 5))

# Plot two lines with markers and colors
ax.plot(x, y1, color='blue', linestyle='--', marker='o', label='Linear')
ax.plot(x, y2, color='red', linestyle='-', marker='s', label='Quadratic')

# Titles, labels, legend, grid
ax.set_title("Styled Line Plot")
ax.set_xlabel("X value")
ax.set_ylabel("Y value")
ax.legend()
ax.grid(True)

# Save figure
fig.savefig("styled_line_plot.png", dpi=300)
plt.show()

# 7. Exercise 2: Rolling Mean + Styling
# Time-series data
df = pd.DataFrame({
    "day": range(1, 21),
    "steps": np.random.randint(4000, 12000, 20)
})

# Compute 5-day rolling mean
df["steps_ma"] = df["steps"].rolling(5).mean()

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8, 5))

# Original steps
ax.plot(df["day"], df["steps"], marker='o', color='blue', label='Daily Steps')

# Rolling mean
ax.plot(df["day"], df["steps_ma"], marker='s', color='red', label='5-Day MA')

ax.set_title("Daily Steps with 5-Day Moving Average")
ax.set_xlabel("Day")
ax.set_ylabel("Steps")
ax.legend()
ax.grid(True)

# Save figure
fig.savefig("rolling_mean_plot.png", dpi=300)
plt.show()