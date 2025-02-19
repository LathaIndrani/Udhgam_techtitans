import pandas as pd

file_path = "C:/Users/kavya/Downloads/merge-csv.com__67a223cf49d15.csv"
df = pd.read_csv(file_path, low_memory=False)

print(df.head(10))

if not any(col.strip().lower() in ["ddate", "amccode", "amcname", "yardcode"] for col in df.columns):
    # If first row is metadata, read again with header in the correct row
    df = pd.read_csv(file_path, low_memory=False, header=1)

df.columns = df.columns.str.strip()

print("Updated Columns in dataset:", df.columns)

df.rename(columns={'Minimum': 'MinPrice', 'Maximum': 'MaxPrice'}, inplace=True)

df['MinPrice'] = pd.to_numeric(df['MinPrice'], errors='coerce')
df['MaxPrice'] = pd.to_numeric(df['MaxPrice'], errors='coerce')

df = df.dropna(subset=['MinPrice', 'MaxPrice'])

print(df.head())  # Display first few rows

cleaned_path = "C:/Users/kavya/Downloads/cleaned_data_final.csv"
df.to_csv(cleaned_path, index=False)

print(f"Data processing complete. Cleaned file saved as '{cleaned_path}'.")
