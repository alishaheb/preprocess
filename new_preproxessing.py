import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    def __init__(self, impute_strategy="mean", scaling_method="standard", outlier_method="IQR"):
        """
        Initializes the preprocessing pipeline with specified options.

        Parameters:
        - impute_strategy: Strategy for imputing missing values ('mean', 'median', etc.)
        - scaling_method: Method for scaling ('standard' for StandardScaler, 'minmax' for MinMaxScaler, or None)
        - outlier_method: Method for outlier treatment (currently supports 'IQR')
        """
        self.impute_strategy = impute_strategy
        self.scaling_method = scaling_method
        self.outlier_method = outlier_method

        # Set up the imputer
        self.imputer = SimpleImputer(strategy=self.impute_strategy)

        # Set up the scaler based on the selected method
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the preprocessing pipeline on the input DataFrame.

        Steps:
        1. Impute missing values.
        2. Handle outliers (currently using IQR-based capping).
        3. Scale the data.

        Returns:
            A preprocessed pandas DataFrame.
        """
        print("Starting preprocessing pipeline...")
        print("Original data shape:", data.shape)

        # Step 1: Impute missing values
        data_imputed = pd.DataFrame(
            self.imputer.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        print(f"Missing values imputed using strategy: {self.impute_strategy}")

        # Step 2: Outlier detection and treatment using the IQR method
        if self.outlier_method == "IQR":
            Q1 = data_imputed.quantile(0.25)
            Q3 = data_imputed.quantile(0.75)
            IQR = Q3 - Q1
            # Cap the data at Q1 - 1.5*IQR and Q3 + 1.5*IQR
            data_capped = data_imputed.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
            print("Outliers capped using IQR method.")
        else:
            data_capped = data_imputed

        # Step 3: Scaling the data if scaler is provided
        if self.scaler is not None:
            data_scaled = pd.DataFrame(
                self.scaler.fit_transform(data_capped),
                columns=data.columns,
                index=data.index
            )
            print(f"Data scaled using: {self.scaling_method} scaling.")
        else:
            data_scaled = data_capped
            print("No scaling applied.")

        print("Preprocessing completed.\n")
        return data_scaled

    def __str__(self):
        """
        Returns a string representation of the preprocessing configuration.
        """
        return (f"DataPreprocessor("
                f"impute_strategy='{self.impute_strategy}', "
                f"scaling_method='{self.scaling_method}', "
                f"outlier_method='{self.outlier_method}')")


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = pd.DataFrame({
        'Feature_A': np.random.randn(100),
        'Feature_B': np.random.randn(100)
    })

    # Introduce missing values in 'Feature_A'
    missing_indices = np.random.choice(data.index, size=10, replace=False)
    data.loc[missing_indices, 'Feature_A'] = np.nan

    # Introduce outliers in 'Feature_B'
    outlier_indices = np.random.choice(data.index, size=5, replace=False)
    data.loc[outlier_indices, 'Feature_B'] *= 10

    print("Original Data Summary:")
    print(data.describe(), "\n")

    # Create a preprocessing pipeline object
    preprocessor = DataPreprocessor(impute_strategy="mean", scaling_method="standard", outlier_method="IQR")

    # Print the object to see its configuration
    print("Preprocessing Pipeline Configuration:")
    print(preprocessor, "\n")

    # Apply the preprocessing pipeline
    processed_data = preprocessor.fit_transform(data)

    print("Processed Data Summary:")
    print(processed_data.describe())
