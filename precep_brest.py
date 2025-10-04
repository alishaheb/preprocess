import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('breast_preprocessed.csv')
print("Dataset preview:")
print(data.head())

# Adjust the following lines based on your dataset:
# If your target variable is named 'target' or 'diagnosis', use that;
# otherwise, we assume the last column is the target.
if 'target' in data.columns:
    X = data.drop('target', axis=1)
    y = data['target']
elif 'diagnosis' in data.columns:
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
else:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Perceptron model
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", report)
