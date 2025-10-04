import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = "Hp.csv"
df = pd.read_csv(data_path, encoding="ISO-8859-1")

# Display the first few rows to understand the structure
print(df.head())

# Encode categorical variables if necessary
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target (assuming the last column is the target)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Compute permutation feature importance
perm_importance = permutation_importance(rf_model, X_test, y_test, scoring="accuracy")
perm_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": perm_importance.importances_mean})
perm_importance_df = perm_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=perm_importance_df["Importance"], y=perm_importance_df["Feature"], palette="coolwarm")
plt.title("Permutation Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
