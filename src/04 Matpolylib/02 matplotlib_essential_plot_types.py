########## Two. Essential Plot Types ##########

import numpy as np
import matplotlib.pyplot as plt

# 1. Line Plot (Trends over ordered data)
x = np.arange(0, 10)
y = np.sin(x)
# plt.plot(x, y)
# plt.show()

# 2. Bar Chart (Comparing categories)
categories = ["Food", "Rent", "Transport", "Entertainment"]
expenses = [500, 1200, 150, 200]
# plt.bar(categories, expenses)
# plt.show()

# 3. Histogram (Distribution of one variable)
data = np.random.randn(1000)
# plt.hist(data, bins=30) # Bins=30 means the data range is divided into 30 equally spaced intervals
# plt.show( )
""" EXPLANATION:
x-axis = value ranges (bins)
y-axis = frequency
Each bar's height shows how many of the 1000 values fall in that interval
"""

# 4. Scatter Plot (Relationship between two variables)
x = np.random.rand(100)
y = np.random.rand(100)
# plt.scatter(x, y)
# plt.show()

# 5. Comparing Plot Types (Same Data, Different Meaning)
x = np.arange(1, 6)
y = [2, 4, 1, 5, 3]
"""
plt.plot(x, y)
plt.show()
plt.scatter(x, y)
plt.show()
"""