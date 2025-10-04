import pandas as pd
import chardet
def read_csv_with_encoding(file_path):


    """Reads a CSV file while automatically detecting its encoding."""
    try:
        # Detect encoding using chardet
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read(100000))  # Read a portion of the file
            encoding = result["encoding"] if result["encoding"] else "utf-8"

        # Read CSV with detected encoding
        df = pd.read_csv(file_path, encoding=encoding, dtype=str)
        return df
    except UnicodeDecodeError:
        # If detected encoding fails, try ISO-8859-1
        try:
            df = pd.read_csv(file_path, encoding="iso-8859-1", dtype=str)
            return df
        except Exception as e:
            print(f"Failed to read the file: {e}")
            return None


if __name__ == "__main__":
    file_path = "/Users/seyedalishahebrahimi/Downloads/csv/zoocust1 - Copy.csv"
df = read_csv_with_encoding(file_path)

if df is not None:
    print("Data loaded successfully!")
else:
    print("Failed to load data.")
