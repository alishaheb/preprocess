import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "Hp.csv"  # Change this to the actual path of your file
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21"], errors='ignore')

# Handle categorical variables
categorical_cols = ["Treated_with_drugs", "Patient_Smoker", "Patient_Rural_Urban", "Patient_mental_condition"]
df[categorical_cols] = df[categorical_cols].astype(str).apply(LabelEncoder().fit_transform)

# Handle missing values with mean imputation for numerical columns
df = df.dropna()

# Define features and target
X = df.drop(columns=["Survived_1_year", "Patient_ID", "ID_Patient_Care_Situation"], errors='ignore')
y = df["Survived_1_year"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 6))
sns.countplot(x="Survived_1_year", data=df, palette="viridis")
plt.title("Survival Distribution")
plt.xlabel("Survived 1 Year")
plt.ylabel("Count")
plt.show()

# Feature Importance
feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], palette="coolwarm")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
