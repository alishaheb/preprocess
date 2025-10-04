import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

########################################
# Step 1: Read the CSV with header
########################################
df = pd.read_csv("Hp.csv", encoding="ISO-8859-1", header=0)

print("Columns in the DataFrame:\n", df.columns.tolist(), "\n")
print("Sample data:\n", df.head(), "\n")

########################################
# Step 2: Identify the target column
########################################
target_col = "Survived_1_year"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in the CSV.")

# Example: If your target is "Yes"/"No", convert to numeric
df[target_col] = df[target_col].replace({"Yes": 1, "No": 0, "Y": 1, "N": 0})
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")


########################################
# Step 3: Create dx1..dx6 from "Treated_with_drugs" (optional)
########################################
def create_drug_indicator(df, col="Treated_with_drugs", codes=None):
    if codes is None:
        codes = ["dx1", "dx2", "dx3", "dx4", "dx5", "dx6"]

    def split_codes(x):
        if pd.isna(x):
            return []
        return [item.strip().lower() for item in re.split(r'[\s,]+', str(x)) if item.strip()]

    for code in codes:
        df[code] = df[col].apply(lambda val: 1 if code in split_codes(val) else 0)
    return df


if "Treated_with_drugs" in df.columns:
    df = create_drug_indicator(df, col="Treated_with_drugs")
else:
    print("Column 'Treated_with_drugs' not found. Skipping dx creation.")

########################################
# Step 4: Select numeric columns for features (exclude the target)
########################################
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != target_col]

# Fill missing feature values with 0 (or another strategy)
df[feature_cols] = df[feature_cols].fillna(0)
# Drop rows where the target is missing
df.dropna(subset=[target_col], inplace=True)

print("\nNumeric feature columns (excluding target):", feature_cols)
print("Data shape after cleaning:", df.shape)
if df.shape[0] == 0:
    raise ValueError("No rows left after cleaning. Check your target or data.")

########################################
# Step 5: Train RandomForest and compute importances
########################################
X = df[feature_cols]
y = df[target_col]

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

importances = clf.feature_importances_
fi_df = pd.DataFrame({
    "feature": feature_cols,  # <-- We use the actual column names here
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nFeature Importances:\n", fi_df)

########################################
# Step 6: Plot feature importances with names (no numbers!)
########################################
# Sort feature importances by ascending order so we can invert the y-axis
fi_df_sorted = fi_df.sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(fi_df_sorted["feature"], fi_df_sorted["importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances")

# Put the largest (most important) feature at the top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
