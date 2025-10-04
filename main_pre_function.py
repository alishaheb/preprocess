import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dataa import read_csv_with_encoding,df

#1 laod the data
#2 drop the columns treated_with_drugs




def plot_correlation_matrix(df):
    print("Plotting correlation matrix...")
    plt.figure(figsize=(10, 10))
    # Filter to only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("No numeric columns available for correlation matrix.")
        return
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()




def handle_missing_values(df):
    """
    Handles missing values by:
    - Dropping columns with more than 50% missing values.
    - Dropping rows with any missing values.

    Parameters:
    - df: Pandas DataFrame

    Returns:
    - df_cleaned: DataFrame with missing values handled.
    """
    print("🔍 Handling missing values...")

    # Count missing values before processing
    total_missing_before = df.isnull().sum().sum()
    if total_missing_before == 0:
        print("✅ No missing values detected.")
        return df
    else:
        print(f"🛑 Missing values detected: {total_missing_before}")

    # Drop columns with more than 50% missing values
    col_threshold = df.shape[0] * 0.5  # If more than 50% missing, drop the column
    df_cleaned = df.dropna(thresh=col_threshold, axis=1)

    # Drop rows with any missing values
    df_cleaned = df_cleaned.dropna()

    # Count missing values after processing
    total_missing_after = df_cleaned.isnull().sum().sum()
    print(f"✅ Missing values after processing: {total_missing_after}")
    print(f"📊 Final dataset shape: {df_cleaned.shape}")

    return df_cleaned


def remove_duplicates(df):
    print("Removing duplicates...")
    return df.drop_duplicates()




def candidate_for_one_hot(df, threshold=0.1):
    """
    Identifies columns that are good candidates for one-hot encoding.

    Parameters:
      df (pd.DataFrame): The input dataframe.
      threshold (float): The maximum ratio of unique values to total rows for a column to be considered categorical.

    Returns:
      List of column names that are candidates for one-hot encoding.
    """
    candidates = []
    for col in df.columns:
        # Check if the column is of object type or categorical type
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            ratio = df[col].nunique() / len(df[col])
            if ratio < threshold:
                candidates.append(col)
    return candidates


# Example usage:
# df = pd.read_csv("your_data.csv")
# columns_to_encode = candidate_for_one_hot(df, threshold=0.1)
# print("Columns to one-hot encode:", columns_to_encode)


def remove_low_variance(df, threshold=0.01):
    print("Removing low-variance features...")
    low_variance_cols = df.var(numeric_only=True)[df.var(numeric_only=True) < threshold].index.tolist()
    return df.drop(columns=low_variance_cols)


def remove_outliers(df):
    print("Removing outliers...")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_cleaned


def remove_multicollinearity(df, threshold=10.0):
    print("Removing multicollinearity...")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    while True:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i) for i in range(len(numeric_cols))]
        max_vif = vif_data["VIF"].max()
        if max_vif < threshold:
            break
        drop_col = vif_data.sort_values("VIF", ascending=False).iloc[0]["Feature"]
        numeric_cols.remove(drop_col)
        df = df.drop(columns=[drop_col])
    return df


def handle_imbalance(df, target_col):
    print("Handling class imbalance...")
    if df[target_col].dtype in ['object', 'category'] or len(df[target_col].unique()) < 10:
        X, y = df.drop(columns=[target_col]), df[target_col]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_col])], axis=1)
    return df


def select_features(df, target_col):
    print("Selecting important features...")
    X, y = df.drop(columns=[target_col]), df[target_col]
    scores = mutual_info_classif(X, y) if y.dtype in ['object', 'category'] or len(
        y.unique()) < 10 else mutual_info_regression(X, y)
    selected_features = X.columns[scores > 0.01]
    return df[selected_features.tolist() + [target_col]]


def apply_pca(X, n_components=0.95):
    print("Applying PCA...")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def preprocess_data(df, target_col=None, normalize=False, apply_pca_flag=False):
    print("Starting data preprocessing...")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_low_variance(df)
    df = remove_outliers(df)
    df = remove_multicollinearity(df)

    if target_col and target_col in df.columns:
        df = handle_imbalance(df, target_col)
        df = select_features(df, target_col)

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numerical_features:
        numerical_features.remove(target_col)

    transformer = ColumnTransformer([
        ('num', MinMaxScaler() if normalize else StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    pipeline = Pipeline([
        ('transform', transformer)
    ])

    X = df.drop(columns=[target_col]) if target_col else df
    y = df[target_col] if target_col else None
    X_transformed = pipeline.fit_transform(X)

    if apply_pca_flag:
        X_transformed = apply_pca(X_transformed)

    return X_transformed, y


def split_data(X, y, test_size=0.2, random_state=42):
    print("Splitting data into training and testing sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main():
    #see the data

    # drop the columns treated_with_drugs
    candidate_for_one_hot(df, threshold=0.1)

    # Plot Correlation Matrix
    plot_correlation_matrix(df)

    # # Handle Missing Values
    handle_missing_values(df)
    print(df.head())
    # Remove Duplicates
    remove_duplicates(df)

    # Remove Low-Variance Features
    remove_low_variance(df)

    # Remove Outliers
    remove_outliers(df)

# Remove Multicollinearity
# remove_multicollinearity(df)

# Handle Class Imbalance
# handle_imbalance(df, target_column)

# Select Important Features
# select_features(df, target_column)

# Preprocess Data
# X_processed, y_processed = preprocess_data(df, target_col=target_column, normalize=True, apply_pca_flag=True)

# Split Data
# X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

# Print Final Shapes
# print("Preprocessing complete.")
# print(f"Training set shape: {_train.shape}, Testing set shape: {X_test.shape}")


if __name__ == "__main__":
    main()
