########## One. Matplotlib Basics ##########

# 1. What is Matplotlib?
"""
Matplotlib is a comprehensive, 2D plotting library for Python that creates publication-quality figures in a variety of formats and interactive environments across platforms. 
It's the most widely used Python library for data visualization and serves as the foundation for many other visualization libraries.
"""

# 2. Installation and Check

"""
pip install matplotlib

import matplotlib
print(matplotlib.__version__)
"""

import numpy as np
import matplotlib.pyplot as plt
""" EXPLANATION:
matplotlib is the full plotting library
pyplot is a state-based interface (like a whiteboard)
plt is the convention, not required, but universal
"""

# 3. Line Plot
x = np.arange(0, 10) # [0 1 2 3 4 5 6 7 8 9]
y = x ** 2 # [ 0  1  4  9 16 25 36 49 64 81]
# plt.plot(x, y)
""" EXPLANATION:
Creates a figure (if none exists)
Creates axes (if none exists)
Draws a line connecting (x[i], y[i])
"""
# plt.show() # Rrenders the output

# 4. What If You Don’t Pass x?
# plt.plot(y)
# plt.show()
""" EXPLANATION:
Matplotlib assumes:
x = [0 1 2 3 4 5 6 7 8 9] in this situation
or: x = [0, 1, 2, ..., len(y)-1]
"""
# plt.plot([3, 7, 2])
# plt.show() # (0,3), (1,7), (2,2)

# 5. Understanding the “State Machine”
plt.plot([1, 2, 3])
plt.plot([3, 2, 1])
# plt.show()
# EXPLANATION: Both lines appear on the same plot because plt is holding state.

# 6. Clearing vs Showing
plt.clf() # Clear current figure
plt.close() # Close window
# plt.plot(x, y)
# plt.show()   # show & reset

# 7. Very First Customization
# plt.plot(x, y, color="red", linewidth=2)
# plt.show()

# 8. Multiple Lines
plt.plot(x, x)
plt.plot(x, x**2)
plt.show()

# 9. Summary
"""
plt is stateful
plot() draws but does not display
show() displays and resets
Data comes from NumPy (or Python lists)
One figure can contain many plots
"""

