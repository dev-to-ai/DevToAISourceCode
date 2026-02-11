########## Six. Decision Trees ##########

"""
What they really are
A sequence of if–else rules.

Think of a Decision Tree as a game of "20 Questions" : Each question narrows down the possibilities until you have an answer.

The Core Idea:
A Decision Tree asks a series of yes/no questions about the data to make a prediction.

Example:

IF days_since_last_login > 30
    IF sessions_per_week < 2
        churn = yes

Statistical intuition
Greedy splitting
Reduces uncertainty step by step
No distribution assumptions

Why trees are dangerous alone
Memorize data easily
Very high variance
Small changes → different trees

Real-World Analogy:
A doctor's diagnostic process:
"Do you have a fever?" (Yes/No)
"Is the cough dry or wet?" (Yes/No to follow-ups)
"How long have you had symptoms?" (Threshold question)
Eventually: "You have a cold" (leaf node prediction)

Think of Decision Trees as flowchart logic - they mirror how humans actually make decisions, just optimized by math to find the most informative questions to ask.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. CREATE DATA
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CREATE AND TRAIN DECISION TREE
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 4. PREDICT AND EVALUATE
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 5. VISUALIZE
plt.figure(figsize=(12, 5))

# Plot decision boundary
plt.subplot(1, 2, 1)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='black')
plt.title("Decision Tree Boundary")

# Plot tree structure
plt.subplot(1, 2, 2)
plot_tree(tree, filled=True, feature_names=['X1', 'X2'], 
          class_names=['0', '1'], rounded=True)
plt.title("Decision Tree")

plt.tight_layout()
plt.show()