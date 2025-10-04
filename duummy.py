import sns
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from all_machine_learning_task import X_test

# Load data
df = pd.read_csv("cleanedairline.csv")
# Standardizing the data (important for PCA)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
print(df.head(10))
#check Permutation Importance
perm_importance = permutation_importance(rf_model, X_test, y_test, scoring="accuracy")
perm_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": perm_importance.importances_mean})
perm_importance_df = perm_importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=perm_importance_df["Importance"], y=perm_importance_df["Feature"], palette="coolwarm")
plt.title("Permutation Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()





# Apply PCA
# pca = PCA()
# pca.fit(df_scaled)
#
# # Plot explained variance ratio
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.title("Explained Variance vs. Number of Components")
# plt.show()
