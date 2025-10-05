import pandas as pd
import os
import argparse


def preprocess_breast_cancer_data(df):
    """
    Preprocessing for breast_cancer_data.csv.
    For example:
      - If a 'diagnosis' column exists, map it to numerical values (e.g., M=1, B=0).
      - Fill missing numeric values with the mean.
      - Fill missing categorical values with the mode.
    """
    # Example: Encode diagnosis if present
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Process numeric columns: fill missing values with mean
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Process categorical columns: fill missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def preprocess_hp_data(df):
    """
    Preprocessing for Hp.csv.
    For example:
      - Fill missing numeric values with the mean.
      - Fill missing categorical values with the mode.
    """
    # Process numeric columns: fill missing values with mean
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Process categorical columns: fill missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def main(dataset):
    # Check if the dataset file exists
    if not os.path.exists(dataset):
        print(f"File {dataset} does not exist.")
        return

    # Read the dataset
    df = pd.read_csv(dataset)

    # Determine which preprocessing to apply based on the filename
    dataset_lower = dataset.lower()
    if 'breast_cancer' in dataset_lower:
        df_processed = preprocess_breast_cancer_data(df)
    elif 'hp' in dataset_lower:
        df_processed = preprocess_hp_data(df)
    else:
        print("Dataset not recognized for separate preprocessing. Exiting.")
        return

    # Create an output directory if it does not exist
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the preprocessed data
    output_path = os.path.join(output_dir, os.path.basename(dataset))
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset (breast_cancer_data.csv or Hp.csv).")
    parser.add_argument("Hp.csv", type=str, required=True, help="Path to the dataset CSV file.")
    args = parser.parse_args()
    main(args.dataset)
