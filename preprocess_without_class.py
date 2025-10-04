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

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def handle_missing_values(df):
    if df.isnull().sum().sum() == 0:
        print("No missing values.")
        return df
    df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_cleaned[df_cleaned.select_dtypes(include=["number"]).columns] = num_imputer.fit_transform(
        df_cleaned.select_dtypes(include=["number"]))
    df_cleaned[df_cleaned.select_dtypes(include=["object"]).columns] = cat_imputer.fit_transform(
        df_cleaned.select_dtypes(include=["object"]))
    return df_cleaned

def remove_duplicates(df):
    return df.drop_duplicates()

def remove_low_variance(df, threshold=0.01):
    low_variance_cols = df.var(numeric_only=True)[df.var(numeric_only=True) < threshold].index.tolist()
    return df.drop(columns=low_variance_cols)

def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_cleaned

def remove_multicollinearity(df, threshold=10.0):
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
    if df[target_col].dtype in ['object', 'category'] or len(df[target_col].unique()) < 10:
        X, y = df.drop(columns=[target_col]), df[target_col]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_col])], axis=1)
    return df

def select_features(df, target_col):
    X, y = df.drop(columns=[target_col]), df[target_col]
    scores = mutual_info_classif(X, y) if y.dtype in ['object', 'category'] or len(y.unique()) < 10 else mutual_info_regression(X, y)
    selected_features = X.columns[scores > 0.01]
    return df[selected_features.tolist() + [target_col]]

def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def preprocess_data(df, target_col=None, normalize=False, apply_pca_flag=False):
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
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    file_path = "cleanedairline.csv"
    target_column = "recommended"
    df = load_data(file_path)
    X_processed, y_processed = preprocess_data(df, target_col=target_column, normalize=True, apply_pca_flag=True)
    X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)
    print("Preprocessing complete.")
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    plot_correlation_matrix(df)
    handle_missing_values(df)
