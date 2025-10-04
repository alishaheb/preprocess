from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import numpy as np

# 1) Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2) Model (a good starting point)
gbc = GradientBoostingClassifier(
    n_estimators=200,      # number of trees
    learning_rate=0.05,   # shrinkage
    max_depth=3,          # depth of each tree
    subsample=0.9,        # stochastic gradient boosting
    random_state=42
)

# 3) Train
gbc.fit(X_train, y_train)

# 4) Evaluate
pred = gbc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=4))

# 5) Feature importance (permutation importance is more reliable)
perm = permutation_importance(gbc, X_test, y_test, n_repeats=10, random_state=42)
imp_order = np.argsort(-perm.importances_mean)
print("\nTop 10 features by permutation importance:")
for idx in imp_order[:10]:
    print(f"{idx}: mean={perm.importances_mean[idx]:.4f}, std={perm.importances_std[idx]:.4f}")
