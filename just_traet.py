import pandas as pd
import re


def create_drug_indicator(df, col='Treated_with_drugs', codes=None, delimiter_pattern=r'[\s,]+'):
    """
    Creates binary indicator columns for specified drug codes (case-insensitive) based on the content of a column.

    The function splits the string in the column using a regex that considers both commas and whitespace as delimiters.
    This ensures that combinations like "dx1 dx3 dx4" are properly split into individual tokens.

    For each code in the provided list, a new column is added to the DataFrame:
      - 1 if the code (ignoring case) is present in the split tokens.
      - 0 otherwise.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column name containing the drug treatment data (default 'Treated_with_drugs').
        codes (list): List of codes to check. Defaults to ['dx1','dx2','dx3','dx4','dx5','dx6'].
        delimiter_pattern (str): Regular expression pattern used to split the string.
                                 Defaults to r'[\s,]+' which splits on commas and any whitespace.

    Returns:
        pd.DataFrame: The modified DataFrame with additional binary indicator columns for each code.
    """
    if codes is None:
        codes = ['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6']

    def split_codes(x):
        if pd.isna(x):
            return []
        # Use regex to split by commas or whitespace, filtering out any empty strings.
        return [item.strip().lower() for item in re.split(delimiter_pattern, str(x)) if item.strip()]

    # Create a binary indicator column for each specified code.
    for code in codes:
        df[code] = df[col].apply(lambda x: 1 if code.lower() in split_codes(x) else 0)

    return df


def clean_dataframe_columns(df, missing_threshold=0.9):
    """
    Removes columns that have 'Unnamed' in the name and those columns where the proportion of missing values
    is greater than or equal to the specified threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missing_threshold (float): The threshold for missing values to drop a column (default is 0.9, i.e., 90%).

    Returns:
        pd.DataFrame: DataFrame with the specified columns removed.
    """
    # Identify columns with 'Unnamed' in the name.
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col]

    # Identify columns with missing value proportion greater than or equal to the threshold.
    cols_to_drop += [col for col in df.columns if df[col].isnull().mean() >= missing_threshold]

    # Remove duplicates if any and drop the identified columns.
    cols_to_drop = list(set(cols_to_drop))
    return df.drop(columns=cols_to_drop)


# Example usage:
if __name__ == '__main__':
    # Load your dataset (update the path as needed)
    df = pd.read_csv('Hp.csv', encoding='ISO-8859-1')

    # Clean the DataFrame by removing unwanted columns
    df_clean = clean_dataframe_columns(df, missing_threshold=0.9)

    # Create binary indicator columns from 'Treated_with_drugs'
    df_modified = create_drug_indicator(df_clean, col='Treated_with_drugs')

    # Save the modified DataFrame to a new CSV file
    df_modified.to_csv('remove_empty_test_Hp_modified.csv', index=False)
    print("Modified DataFrame saved as 'azazazaz_second_test_Hp_modified.csv'.")
