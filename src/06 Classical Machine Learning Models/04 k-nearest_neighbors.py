########## Four. k-Nearest Neighbors (KNN)##########
"""
What it really is
Tell me what my neighbors are doing.

No training phase.
The dataset is the model.

Statistical intuition
Distance = similarity
Local averaging
No assumptions about shape
"""

# Visual intuition
# pip install scikit-learn
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=200, n_features=2, 
                           n_redundant=0, random_state=42)
"""
X: 200 samples with 2 features each (coordinates on a 2D plane)
y: 200 labels (0 or 1) for each sample
n_redundant=0: Set redundant features to 0
random_state=42: Ensures reproducibility (same "random" data every time)
"""

plt.scatter(X[:, 0], X[:, 1], c=y)
"""
Plots each sample as a point
X-axis: Feature 1, Y-axis: Feature 2
Colors points by their class (c=y): usually blue for class 0, orange for class 1
"""
plt.show()

"""
Prediction:
Find k closest points
Majority vote (classification)
Mean (regression)
"""

"""
Bias vs Variance in KNN
Small k → low bias, high variance (overfits)
Large k → high bias, low variance (smooths too much)
📌 Distance-based → scaling is mandatory
"""

"""
Why this is useful:
For learning classification:
Perfect practice dataset - you know the "answer" (y labels)

2 features = easy to visualize

Can see exactly what a classifier is doing

For logistic regression specifically:
You can draw the decision boundary (the line the model learns to separate blue from orange)

Visualize how well the model separates the classes

See the probabilistic "S-curve brain" in action across 2D space

Think of it as:
A toy problem for teaching computers to tell things apart. Like creating a fake dataset of "apples vs oranges" with only weight and color as features, so you can watch the learning process happen in real-time.
"""
