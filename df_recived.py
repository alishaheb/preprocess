from dataa import read_csv_with_encoding



if __name__ == "__main__":
    file_path = "/Users/seyedalishahebrahimi/Downloads/csv/zoocust1 - Copy.csv"
    df = read_csv_with_encoding(file_path)
    df.head(5)
    if df is not None:
        print("Data loaded successfully!")
    else:
        print("Failed to load data.")
