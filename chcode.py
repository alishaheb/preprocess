import pandas as pd
import re

df = pd.read_csv('Hp.csv', encoding='ISO-8859-1')


def split_mixed_columns_vectorized():
    df = pd.read_csv('Hp.csv', encoding='ISO-8859-1')

    for col in df.select_dtypes(include=['object']):  # Only process text-based columns
        if df[col].str.contains(r'[a-zA-Z].*\d|\d.*[a-zA-Z]', regex=True, na=False).any():
            # Extract letters and numbers in separate columns using vectorized operations
            df[col + '_letters'] = df[col].str.replace(r'[^a-zA-Z]', '', regex=True)
            df[col + '_numbers'] = df[col].str.replace(r'[^0-9]', '', regex=True)

    df.to_csv("p.csv", index=False)
    print("Processed CSV saved as 'processed_output.csv'")

x=random
split_mixed_columns_vectorized()
