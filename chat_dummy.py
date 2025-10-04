import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# List of fallback encodings to try if needed
fallback_encodings = ["utf-8", "windows-1252", "latin1", "ascii"]


def read_csv_with_encoding(file_path):
    """
    Reads a CSV file using ISO-8859-1 encoding by default, with fallback options if needed.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: The read DataFrame if successful; otherwise, None.
    """
    try:
        # Try with ISO-8859-1 first
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        return df
    except Exception as e:
        print(f"Failed with ISO-8859-1: {e}\nTrying fallback encodings...")
        for encoding in fallback_encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df  # Return the DataFrame if reading succeeds
            except Exception:
                continue  # Try the next encoding if reading fails
        print("Failed to read the file with all provided encodings.")
        return None


class DataPreprocessor:
    def __init__(self, impute_strategy="mean", scaling_method="standard", outlier_method="IQR"):
        """
        Initializes the preprocessing pipeline with specified options.

        Parameters:
            impute_strategy (str): Strategy for imputing missing numeric values ('mean', 'median', etc.)
            scaling_method (str): Method for scaling ('standard' for StandardScaler,
                                  'minmax' for MinMaxScaler, or None)
            outlier_method (str): Method for outlier treatment (currently supports 'IQR')
        """
        self.impute_strategy = impute_strategy
        self.scaling_method = scaling_method
        self.outlier_method = outlier_method

        # Set up the imputer for numeric columns
        self.numeric_imputer = SimpleImputer(strategy=self.impute_strategy)

        # Set up the scaler based on the selected method for numeric data
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the preprocessing pipeline on the input DataFrame.
        Separates numeric and non-numeric columns to apply appropriate transformations.

        Steps:
            1. For numeric columns:
               - Drop columns with all missing values.
               - Impute missing numeric values using the specified numeric strategy.
               - Apply outlier treatment (IQR method) and scaling.
            2. For non-numeric columns:
               - Impute missing values using the 'most_frequent' strategy.

        Parameters:
            data (pd.DataFrame): The input DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print("Starting preprocessing pipeline...")
        print("Original data shape:", data.shape)

        # Identify numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

        # Process numeric columns if they exist
        if len(numeric_cols) > 0:
            numeric_data = data[numeric_cols]
            # Identify numeric columns with at least one observed (non-null) value
            valid_numeric_cols = numeric_data.columns[numeric_data.notnull().any()]
            dropped_numeric_cols = list(set(numeric_cols) - set(valid_numeric_cols))

            if dropped_numeric_cols:
                print("Dropping numeric columns with all missing values:", dropped_numeric_cols)

            if len(valid_numeric_cols) > 0:
                numeric_data_valid = numeric_data[valid_numeric_cols]
                # Impute missing numeric values
                numeric_imputed = pd.DataFrame(
                    self.numeric_imputer.fit_transform(numeric_data_valid),
                    columns=valid_numeric_cols,
                    index=data.index
                )
                print(f"Numeric missing values imputed using strategy: {self.impute_strategy}")

                # Outlier detection and treatment using the IQR method
                if self.outlier_method == "IQR":
                    Q1 = numeric_imputed.quantile(0.25)
                    Q3 = numeric_imputed.quantile(0.75)
                    IQR = Q3 - Q1
                    numeric_imputed = numeric_imputed.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
                    print("Numeric outliers capped using IQR method.")

                # Scaling numeric data if a scaler is provided
                if self.scaler is not None:
                    numeric_imputed = pd.DataFrame(
                        self.scaler.fit_transform(numeric_imputed),
                        columns=valid_numeric_cols,
                        index=data.index
                    )
                    print(f"Numeric data scaled using: {self.scaling_method} scaling.")
            else:
                numeric_imputed = pd.DataFrame(index=data.index)
                print("No valid numeric columns detected for imputation and scaling.")

            # Optionally, re-add dropped columns as all-NaN (if you wish to preserve them)
            if dropped_numeric_cols:
                dropped_df = pd.DataFrame(np.nan, index=data.index, columns=dropped_numeric_cols)
                numeric_imputed = pd.concat([numeric_imputed, dropped_df], axis=1)
        else:
            numeric_imputed = pd.DataFrame(index=data.index)
            print("No numeric columns detected for processing.")

        # Process non-numeric columns if they exist
        if len(non_numeric_cols) > 0:
            non_numeric_data = data[non_numeric_cols]
            # Impute missing non-numeric values using most frequent strategy
            non_numeric_imputer = SimpleImputer(strategy="most_frequent")
            non_numeric_imputed = pd.DataFrame(
                non_numeric_imputer.fit_transform(non_numeric_data),
                columns=non_numeric_cols,
                index=data.index
            )
            print("Non-numeric missing values imputed using 'most_frequent' strategy.")
        else:
            non_numeric_imputed = pd.DataFrame(index=data.index)
            print("No non-numeric columns detected.")

        # Combine the numeric and non-numeric parts back together
        data_processed = pd.concat([numeric_imputed, non_numeric_imputed], axis=1)

        print("Preprocessing completed.\n")
        return data_processed

    def __str__(self):
        """
        Returns a string representation of the preprocessing configuration.
        """
        return (f"DataPreprocessor("
                f"impute_strategy='{self.impute_strategy}', "
                f"scaling_method='{self.scaling_method}', "
                f"outlier_method='{self.outlier_method}')")


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # File path to the CSV file
    file_path = "Hp.csv"

    # Read the data using the custom function (with ISO-8859-1 as default)
    df = read_csv_with_encoding(file_path)

    if df is not None:
        print("Data loaded successfully!\n")

        # Print basic information about the loaded data
        print("Data Summary:")
        print(df.describe(include="all"), "\n")

        # Create and display the preprocessing pipeline configuration
        preprocessor = DataPreprocessor(impute_strategy="mean", scaling_method="standard", outlier_method="IQR")
        print("Preprocessing Pipeline Configuration:")
        print(preprocessor, "\n")

        # Apply the preprocessing pipeline to the loaded data
        processed_data = preprocessor.fit_transform(df)
        print("Processed Data Summary:")
        print(processed_data.describe(include="all"))

        # Save the processed DataFrame to a new CSV file
        output_file = "chat_dummy_processed_Hp.csv"
        processed_data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    else:
        print("Failed to load data.")
