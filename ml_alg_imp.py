from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocess_without_class import Pre

# Load and preprocess the dataset



X_processed, y_processed = pre.preprocess_data(df, target_col=target_column, apply_pca_flag=True)

# Split data
X_train, X_test, y_train, y_test = pre.split_data(X_processed, y_processed)

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

print("\nModel training and evaluation complete.")
