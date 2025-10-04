import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/seyedalishahebrahimi/Downloads/pygoal/BankChurners.csv')

# Encode the target column: 'Attrition_Flag' (e.g., "Existing Customer" as 0 and "Attrited Customer" as 1)
le = LabelEncoder()
df['Attrition_Flag_bin'] = le.fit_transform(df['Attrition_Flag'])

# Drop columns that are not useful for prediction
df_model = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)

# Identify categorical columns
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical features
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Separate features and target variable
X = df_model.drop('Attrition_Flag_bin', axis=1)
y = df_model['Attrition_Flag_bin']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Retrieve feature importances from the model
importances = clf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances:")
print(feat_importances)

# Optional: Plotting the top 10 features for a visual overview
top_n = 10
plt.figure(figsize=(10, 6))
feat_importances.head(top_n).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
