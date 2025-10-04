import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from dataa import df

class BasicDataloader:
    def __init__(self, df, target_column=None, config=None):
        self.df = df.copy()
        self.target_column = target_column
        self.config = config or {
            "handle_missing": True,
            "remove_duplicates": True,
            "remove_low_variance": True,
            "remove_outliers": True,
            "remove_multicollinearity": True,
            "handle_imbalance": True,
            "feature_selection": True,
            "normalize": True,
            "apply_pca": False,
            "scaling_method": "standard"  # Options: "standard", "minmax", "robust"
        }

        self.X_processed = None
        self.y_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def detect_data_types(self):
        """ Ensure correct data types before processing. """
        #if all_strings = df['col'].apply(lambda x: isinstance(x, str)).all() #todo:check if all the columns are strings
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                # If only unique values are 0/1, convert to integer
                unique_values = self.df[col].dropna().unique()
                if set(unique_values) <= {0, 1}:
                    self.df[col] = self.df[col].astype(int)

    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 8))

        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=["number"])

        # Remove low-variance columns
        numeric_df = numeric_df.loc[:, numeric_df.var() > 0]

        # Compute correlation
        corr_matrix = numeric_df.corr()

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Updated Correlation Matrix Heatmap")
        plt.show()


    def handle_missing_values(self):
        if not self.config["handle_missing"]:
            return

        print("\n🔍 Checking for missing values before dropping columns...\n")

        # Convert common missing value representations into NaN
        #self.df.replace(["", " ", "NA", "N/A", "null", "NaN", "nan"], np.nan, inplace=True)

        # Calculate missing percentages
        missing_percentage = self.df.isnull().sum() / len(self.df) * 100
        print("\n📊 Missing Values Percentage Before Dropping Columns:\n", missing_percentage)

        # Manually select columns to drop (where missing % > 50%)
        cols_to_drop = missing_percentage[missing_percentage > 50].index.tolist()

        if cols_to_drop:
            print(f"\n🛠 Dropping Columns (More than 50% Missing Data): {cols_to_drop}")
            self.df.drop(columns=cols_to_drop, inplace=True)
        else:
            print("\n✅ No columns needed to be dropped (all have enough data).")

        # Ensure correct data type handling
        self.df = self.df.apply(pd.to_numeric, errors='ignore')

        # Handle missing values in numeric columns
        # num_imputer = SimpleImputer(strategy="median")
        # numeric_cols = self.df.select_dtypes(include=["number"]).columns
        # self.df[numeric_cols] = num_imputer.fit_transform(self.df[numeric_cols])
        #
        # # Handle missing values in categorical columns
        # cat_imputer = SimpleImputer(strategy="most_frequent")
        # categorical_cols = self.df.select_dtypes(include=["object"]).columns
        # self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])

        print("\n✅ Missing values handled successfully.")

    def remove_duplicates(self):
        if self.config["remove_duplicates"]:
            self.df.drop_duplicates(inplace=True)

    def remove_low_variance(self, threshold=0.01):
        if self.config["remove_low_variance"]:
            low_variance_cols = self.df.var(numeric_only=True)[
                self.df.var(numeric_only=True) < threshold].index.tolist()
            self.df.drop(columns=low_variance_cols, inplace=True)

    def remove_outliers(self):
        if not self.config["remove_outliers"]:
            return
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        Q1, Q3 = self.df[numeric_cols].quantile(0.25), self.df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[
            ~((self.df[numeric_cols] < (Q1 - 1.5 * IQR)) | (self.df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        ]

    def remove_multicollinearity(self, threshold=10.0):
        if not self.config["remove_multicollinearity"]:
            return
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        while True:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = numeric_cols
            vif_data["VIF"] = [variance_inflation_factor(self.df[numeric_cols].values, i) for i in
                               range(len(numeric_cols))]
            max_vif = vif_data["VIF"].max()
            if max_vif < threshold:
                break
            drop_col = vif_data.sort_values("VIF", ascending=False).iloc[0]["Feature"]
            numeric_cols.remove(drop_col)
            self.df.drop(columns=[drop_col], inplace=True)

    def handle_imbalance(self):
        if not self.config["handle_imbalance"] or self.target_column not in self.df.columns:
            return
        X, y = self.df.drop(columns=[self.target_column]), self.df[self.target_column]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.df = pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[self.target_column])],
            axis=1)

    def preprocess_data(self):
        self.detect_data_types()  # Identify numerical, categorical, text, etc.
        self.handle_missing_values()  # Impute or remove missing data
        self.remove_duplicates()  # Avoid duplicate data skewing analysis
        self.remove_outliers()  # Address extreme values (e.g., using IQR or z-score)
        self.remove_multicollinearity()  # Drop highly correlated features (e.g., via VIF)
        self.remove_low_variance()  # Remove features with almost no variation
        self.handle_imbalance()          # (Optional) Only for classification tasks

    # def split_data(self, test_size=0.2):
    #     X, y = self.df.drop(columns=[self.target_column]), self.df[self.target_column]
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=y,
    #                                                                             random_state=42)


if __name__ == "__main__":
    # print(df.head(10))
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21","ID_Patient_Care_Situation", "Patient_ID"], errors='ignore')

    # Apply Label Encoding to categorical columns
    categorical_cols = ["Treated_with_drugs", "Patient_Smoker", "Patient_Rural_Urban", "Patient_mental_condition"]

    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # Convert NaNs to string before encoding
    # Define target column
    target=df['Survived_1_year']
    print(f"the target columns is this {df.Survived_1_year.head(10)}")
    processor = BasicDataloader(df, target_column="Survived_1_year")
    print(df.head(10))
    processor.plot_correlation_matrix()
    processor.preprocess_data()
    #processor.split_data()

