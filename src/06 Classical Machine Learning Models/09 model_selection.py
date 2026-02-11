########## Nine. Model Selection ##########

"""
ML engineers don't ask: Which model is best?

They ask:
How much data?
How noisy?
How interpretable?
How fast?
How stable?

The Core Idea:
You don't just pick one model. You try many and choose the best.

The Process (3 Simple Steps):
1. Candidate Models
Try different algorithms AND different hyperparameters
2. Evaluate Fairly
NEVER use test data for selection! Use validation data or cross-validation.
3. Select & Validate
Pick the best performer, then verify once on test set.

Visual Intuition:
Your Data
    │
    ├── Training Set (60%) ──┐
    ├── Validation Set (20%)──┼── Used for model selection
    └── Test Set (20%) ──────┘    Used ONLY at the end

Real-World Analogy:
Cooking a new recipe for guests:
Step	            ML Equivalent	        What You're Doing
Practice at home	Training	            Learn from cookbook
Taste it yourself	Validation	            Adjust seasoning
Serve to guests	    Test	                FINAL evaluation
Make it again	    Deployment	            Cook for real
You NEVER ask guests to taste while cooking! That's your test set.

The Golden Rule:
Thou shalt not touch the test set until the very end.

Key Takeaways:
Test set is sacred - Only use it ONCE at the very end
Compare fairly - Use cross-validation or validation set
More candidates = better chance - Try different algorithms AND parameters
Don't over-optimize - The "best" on validation may not be best on test
Keep it simple - Prefer simpler models if performance is similar

Think of model selection as American Idol:
Training = Practice sessions
Validation Set = Judges' feedback
Cross-Validation = Multiple judges
Test Set = Live finale (only happens once!)
Selected Model = The winner
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. CREATE DATA
X, y = make_classification(n_samples=1000, random_state=42)

# 2. SPLIT ONCE (LOCK TEST SET AWAY)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CANDIDATES
models = {
    'Tree': DecisionTreeClassifier(max_depth=3),
    'Forest': RandomForestClassifier(n_estimators=50),
    'SVM': SVC(kernel='rbf')
}

# 4. COMPARE USING CROSS-VALIDATION (on training data only)
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 5. PICK WINNER
best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nWinner: {best_name}")

# 6. FINAL EVALUATION (ONLY NOW use test set)
best_model.fit(X_train, y_train)
final_score = best_model.score(X_test, y_test)
print(f"Final test score: {final_score:.3f}")