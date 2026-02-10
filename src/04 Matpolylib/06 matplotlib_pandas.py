########## Six. Pandas + Matplotlib ##########

import pandas as pd
import matplotlib.pyplot as plt

# 1. First Pandas Plot
df = pd.DataFrame({
    "day": range(1, 8),
    "steps": [4000, 6000, 8000, 7000, 9000, 11000, 10000]
})
df.plot(x="day", y="steps")
plt.show()
"""
Pandas chose a line plot
Matplotlib rendered it
"""

# 2. Choosing Plot Type with kind
"""
df.plot(kind="line")
df.plot(kind="bar")
df.plot(kind="hist")
df.plot(kind="scatter", x="day", y="steps")
kind maps directly to Matplotlib plot types.
"""

# 3. Getting the Axes
ax = df.plot(x="day", y="steps")
ax.set_title("Weekly Steps")
ax.set_xlabel("Day")
ax.set_ylabel("Steps")
plt.show()

# 4. Plotting Multiple Columns
df = pd.DataFrame({
    "day": range(1, 8),
    "steps": [4000, 6000, 8000, 7000, 9000, 11000, 10000],
    "calories": [2200, 2100, 2000, 2050, 1950, 1900, 1980]
})

ax = df.plot(x="day")
ax.set_title("Steps and Calories Over Time")
plt.show()

# 5. Histogram from DataFrame Column
ax = df["steps"].plot(kind="hist", bins=10)
ax.set_title("Steps Distribution")
plt.show()

# 6. Rolling Mean
# Create a 3-day moving average of 'steps' column
df["steps_ma"] = df["steps"].rolling(3).mean()
"""
This does 3 things:
.rolling(3) - Creates a "window" of size 3 that slides over the data
.mean() - Calculates the average within each window
df["steps_ma"] = ... - Stores the result in a new column
"""

ax = df.plot(x="day", y="steps", label="Steps")
df.plot(x="day", y="steps_ma", ax=ax, label="3-Day Average")

ax.set_title("Steps with Moving Average")
ax.legend()
plt.show()

# 7. Exercise 1: Daily Temperature Over a Week
df = pd.DataFrame({
    "day": range(1, 8),
    "temperature": [18, 20, 22, 21, 19, 23, 24]
})
ax = df.plot(x="day", y="temperature", marker="o")
ax.set_title("Daily Temperature Over a Week")
ax.set_xlabel("Day")
ax.set_ylabel("Temperature (°C)")
plt.show()

# 8. Exercise 2: Sales by Category
df = pd.DataFrame({
    "category": ["A", "B", "C", "D"],
    "sales": [100, 150, 80, 120]
})

ax = df.plot(x="category", y="sales", kind="bar", color="skyblue")
ax.set_title("Sales by Category")
ax.set_ylabel("Sales")
plt.show()

# 9. Exercise 3: Distribution of Random Values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "values": np.random.randn(200)
})

ax = df["values"].plot(kind="hist", bins=15, color="orange")
ax.set_title("Distribution of Random Values")
ax.set_xlabel("Value")
plt.show()

