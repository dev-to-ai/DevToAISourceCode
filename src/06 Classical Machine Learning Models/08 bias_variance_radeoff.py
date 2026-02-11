########## Eight. Bias–Variance Tradeoff ##########
"""
You can't have both low bias AND low variance方差. There's always a tradeoff.

The Definitions:

Bias = Wrong assumptions
High bias: Model is too simple, misses patterns
Low bias: Model captures patterns well
"Are you aiming at the right spot?"

Variance = Sensitivity to data
High variance: Model changes a lot with new data
Low variance: Model is stable across different data
"Are your shots consistent?"

Visual mental model
Model	                Bias	            Variance
Linear regression	    High	            Low
Logistic regression	    High	            Low
KNN (small k)	        Low	                High
Decision tree	        Low	                High
Random forest	        Low	                Medium
SVM (RBF)	            Low	                Medium–High

The golden rule
You don't want the most accurate model
You want the most stable model
Not too simple (underfitting), not too complex (overfitting), but just right for your data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create noisy sine wave data
np.random.seed(42)
X = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, 20)
X = X.reshape(-1, 1)

# THREE MODELS: High Bias, Balanced, High Variance
models = [
    make_pipeline(PolynomialFeatures(1), LinearRegression()),  # Underfitting (HIGH BIAS)
    make_pipeline(PolynomialFeatures(4), LinearRegression()),  # Just right (BALANCED)
    make_pipeline(PolynomialFeatures(15), LinearRegression())  # Overfitting (HIGH VARIANCE)
]

titles = ["High Bias (Underfitting)", "Balanced (Just Right)", "High Variance (Overfitting)"]

# Plot
plt.figure(figsize=(15, 4))
X_test = np.linspace(0, 1, 200).reshape(-1, 1)

for i, (model, title) in enumerate(zip(models, titles)):
    model.fit(X, y)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, alpha=0.7, label='Training data')
    plt.plot(X_test, model.predict(X_test), 'r-', label='Model')
    plt.ylim(-2, 2)
    plt.title(title)
    plt.legend()

plt.tight_layout()
plt.show()

# Simple demonstration of the tradeoff
print("BIAS: How wrong is the model on average?")
print("VARIANCE: How much does it change with new data?\n")

print("Left:   High Bias - Too simple, misses the pattern")
print("Middle: Balanced - Gets the pattern right")
print("Right:  High Variance - Too complex, fits the noise")