import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score
)

# Load the data
df = pd.read_csv("azazazaz_Hp_modified.csv", encoding="ISO-8859-1")

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Debug: print the column names to check for issues
print("Columns in the dataset:", df.columns.tolist())

# Drop unnamed columns (which may appear due to extra commas in the CSV)
unnamed_columns = [col for col in df.columns if "Unnamed:" in col]
if unnamed_columns:
    print("Dropping unnamed columns:", unnamed_columns)
    df = df.drop(columns=unnamed_columns)

# Define the target column as it appears in the dataset (case sensitive)
target_column = "Survived_1_year"
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found in the dataset. Available columns: {df.columns.tolist()}")

# Optionally drop identifier columns that are not useful for modeling
if "Patient_ID" in df.columns:
    df = df.drop(columns=["Patient_ID"])

# Define features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical (non-numeric) variables into numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Check for missing values in features
print("Missing values per column in X:\n", X.isna().sum())

# Split the data (80% train, 20% test) with stratification on the target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model: Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Additional Evaluation: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot the confusion matrix
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
# Define tick marks based on unique target values
tick_marks = range(len(set(y)))
plt.xticks(tick_marks, rotation=45)
plt.yticks(tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Additional Evaluation: ROC Curve and AUC
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
