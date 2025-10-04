import pandas as pd

data = {

    "Column2": [100, 200, 300],  # Pure numbers (int)
    "Column3": ["hello", "world", "python"],  # Pure text
    "Column1": ["A123", "B456", "C789"],  # Mixed letters + numbers
    "Column4": ["123A", "456B", "789C"],  # Mixed numbers + letters
}

df = pd.DataFrame(data)
print(df.dtypes)
for col in df.select_dtypes(include=['object']):  # Only process text-based columns
    # if df[col].str.contains(r'[a-zA-Z].*\d|\d.*[a-zA-Z]', regex=True, na=False).any():
    # Extract letters and numbers in separate columns using vectorized operations
    print(df[col].str.extract(r'([a-zA-Z]+)(\d+)'))
    df[col + '_letters'] = df[col].str.replace(r'[^a-zA-Z]', '', regex=True)
    df[col + '_numbers'] = df[col].str.replace(r'[^0-9]', '', regex=True)
    # remove the original column
    df.drop(columns=[col], inplace=True)

    print(df)
