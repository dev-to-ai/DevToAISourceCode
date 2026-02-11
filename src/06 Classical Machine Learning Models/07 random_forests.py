########## Seven. Random Forests ##########

"""
What it really is
Many bad trees → one good forest.

Think of Random Forests as a committee of experts rather than one single decision-maker.

The Core Idea:
Random Forest = Many Decision Trees + Randomness

Each tree:
Sees random subset of data
Sees random subset of features
Makes mistakes differently

Final prediction = average vote.

How It Works (The Magic Recipe):
1. Bagging (Bootstrap Aggregating)
Give each tree a different random subset of the data
Some data points are repeated, some are left out
Like showing 100 doctors 100 different random patient samples
2. Random Feature Selection
At each split, each tree can only consider a random subset of features
Prevents all trees from looking at the same "obvious" feature
Forces trees to be diverse
3. Voting
Each tree votes on the prediction
Majority wins for classification
Average for regression

Bias–Variance magic
Individual tree: low bias, high variance
Forest: low bias, low variance

Why Random Forests Are Better:
Single Decision Tree	                    Random Forest
One "expert"	                            100 "experts"
Overfits easily	                            Generalizes well
Unstable (small change = different tree)	Stable (average smooths out noise)
High variance	                            Low variance

Real-World Analogy:
Instead of one doctor's opinion:
Get 100 doctors in a room
Give each doctor different test results
Let each doctor focus on different symptoms
Take a vote
The group wisdom > any single expert.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. CREATE DATA
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)

# 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN RANDOM FOREST (100 trees)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. PREDICT & EVALUATE
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# 5. COMPARE WITH SINGLE TREE
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)
print(f"Single Tree Accuracy:  {tree_acc:.2f}")

# 6. VISUALIZE DECISION BOUNDARIES
plt.figure(figsize=(12, 5))

# Plot Random Forest boundary
plt.subplot(1, 2, 1)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='black')
plt.title(f"Random Forest (100 trees) - Acc: {accuracy:.2f}")

# Plot Single Tree boundary
plt.subplot(1, 2, 2)
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='black')
plt.title(f"Single Decision Tree - Acc: {tree_acc:.2f}")

plt.tight_layout()
plt.show()

# 7. FEATURE IMPORTANCE
print(f"\nFeature importance: {rf.feature_importances_.round(3)}")