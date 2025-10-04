import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

encodings = ["utf-8", "ISO-8859-1", "windows-1252", "latin1", "ascii"]


def read_csv_with_encoding(file_path):
    """Reads a CSV file while automatically detecting its encoding."""
    try:
        df = pd.read_csv(file_path)
        return df
    except UnicodeDecodeError:
        # If detected encoding fails, try other encodings from the list
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, dtype=str)
                return df  # If reading with encoding succeeds, return the dataframe
            except UnicodeDecodeError:
                continue  # Try the next encoding if reading fails
        print("Failed to read the file with all provided encodings.")
        return None


# Call the function and store the DataFrame in a global variable
file_path = "Hp.csv"
df = read_csv_with_encoding(file_path)

# Now df is accessible globally
if df is not None:
    print("Data loaded successfully!")
else:
    print("Failed to load data.")


def handle_missing_values(df):
    print("\n🔍 Checking for missing values before dropping columns...\n")

    # Convert common missing value representations into NaN
    df.replace(["", " ", "NA", "N/A", "null", "NaN", "nan"], np.nan, inplace=True)

    # Calculate missing percentages
    missing_percentage = df.isnull().sum() / len(df) * 100
    print("\n📊 Missing Values Percentage Before Dropping Columns:\n", missing_percentage)

    # Manually select columns to drop (where missing % > 50%)
    cols_to_drop = missing_percentage[missing_percentage > 50].index.tolist()

    if cols_to_drop:
        print(f"\n🛠 Dropping Columns (More than 50% Missing Data): {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        print("\n✅ No columns needed to be dropped (all have enough data).")

    # Ensure correct data type handling
    df = df.apply(pd.to_numeric, errors='ignore')

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



def identify_multivalue_columns(df):
    """
    Identifies columns that contain multiple values separated by a delimiter.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: List of column names that contain multiple values.
    """
    multivalue_columns = []

    for column in df.columns:
        if df[column].dtype == object:  # Check only string-based columns
            sample_values = df[column].dropna().astype(str).sample(min(10, len(df)))  # Sample a few rows
            for value in sample_values:
                if any(char in value for char in [" ", ",", ";", "|"]):  # Check for common delimiters
                    multivalue_columns.append(column)
                    break

    return multivalue_columns


import pandas as pd


def identify_multivalue_columns(df, delimiters=[" ", ",", ";", "|"]):
    """
    Identifies columns that contain multiple values separated by common delimiters.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    delimiters (list): List of delimiters to check for multi-value columns.

    Returns:
    list: List of column names that contain multiple values.
    """
    multivalue_columns = []

    for column in df.columns:
        if df[column].dtype == object:  # Check only string-based columns
            sample_values = df[column].dropna().astype(str).sample(min(10, len(df)))  # Sample a few rows
            for value in sample_values:
                if any(delim in value for delim in delimiters):  # Check for multiple delimiters
                    multivalue_columns.append(column)
                    break  # No need to check further if one value contains a delimiter

    return multivalue_columns


def process_multivalue_columns(df, columns_to_split=None, delimiters=[" ", ",", ";", "|"]):
    """
    Processes identified columns with multiple values separated by a delimiter.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_split (list): List of columns to process. If None, it will auto-detect them.
    delimiters (list): List of delimiters used in the column.

    Returns:
    pd.DataFrame: Processed DataFrame with separate binary columns.
    """
    df = df.copy()

    # Auto-detect multi-value columns if none are specified
    if columns_to_split is None:
        columns_to_split = identify_multivalue_columns(df, delimiters)

    for column in columns_to_split:
        # Ensure column exists
        if column not in df.columns:
            continue

        # Standardize text format (strip spaces, convert to uppercase)
        df[column] = df[column].astype(str).str.strip().str.upper()

        # Replace multiple delimiters with a single space for consistency
        for delim in delimiters:
            df[column] = df[column].str.replace(delim, " ")

        # Perform one-hot encoding (splitting by space)
        encoded_columns = df[column].str.get_dummies(sep=" ")

        # Merge the new columns with the original dataset
        df = pd.concat([df, encoded_columns], axis=1)

        # Drop the original column after processing
        df.drop(columns=[column], inplace=True)

    return df
# Example usage
# file_path = "your_dataset.csv"
# df = pd.read_csv(file_path, encoding="ISO-8859-1")
# identified_columns = identify_multivalue_columns(df)
# print("Columns to split:", identified_columns)
# df = process_multivalue_columns(df)  # Process identified columns
# df.to_csv("processed_dataset.csv", index=False)  # Save the processed data


# Example usage
# file_path = "your_dataset.csv"
# df = pd.read_csv(file_path, encoding="ISO-8859-1")
# df = process_multivalue_columns(df, columns_to_split=["Treated_with_drugs"])  # Specify relevant column(s)
# df.to_csv("processed_dataset.csv", index=False)  # Save the processed data

if __name__ == "__main__":
    handle_missing_values(df)
    process_multivalue_columns(df)
    print(df.describe())
    #df.to_csv("xx.csv", index=False)  # Save the processed data
    #print(df.head(10))

# Define features and target

# Define features (X) and target (y)
# X = df.drop(columns=["recommended"])  # Features
# y = df["recommended"]  # Target
# target_column = y
# # Split the data (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(f"X_train={X_train.shape}, X_test={X_test.shape}, Y_train={y_train.shape}, Y_test={y_test.shape}")
# print(target_column.head(5))
