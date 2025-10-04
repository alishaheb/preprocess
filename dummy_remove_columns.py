import pandas as pd
import numpy as np

def clean_dataframe(df, missing_threshold=0.5):
    """
    Identifies and removes unnecessary columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    missing_threshold (float): The threshold for missing values (default is 50%).

    Returns:
    pd.DataFrame: A cleaned DataFrame with the identified columns removed.
    list: A list of removed column names.
    """

    # Identify unnamed columns
    unnamed_columns = [col for col in df.columns if "Unnamed" in col]

    # Identify columns with high missing values
    missing_values = df.isnull().sum() / len(df)
    high_missing_columns = missing_values[missing_values > missing_threshold].index.tolist()

    # Identify ID-like columns
    id_columns = [col for col in df.columns if "ID" in col]

    # Combine all identified columns and remove duplicates
    columns_to_remove = list(set(unnamed_columns + high_missing_columns + id_columns))

    # Drop the identified columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

    return df_cleaned, columns_to_remove

# Example usage:
# df = pd.read_csv("your_file.csv", encoding="latin1")
# df_cleaned, removed_columns = clean_dataframe(df)
# print("Removed columns:", removed_columns)
# print("New shape:", df_cleaned.shape)
if __name__ == "__main__":
    df = pd.read_csv("Hp.csv", encoding="latin1")
    df_cleaned, removed_columns = clean_dataframe(df)
    print("Removed columns:", removed_columns)
    print("New shape:", df_cleaned.shape)
    print(df_cleaned.head(10))
    print(df_cleaned.shape[1])
