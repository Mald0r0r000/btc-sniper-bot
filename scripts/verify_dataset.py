import pandas as pd

df = pd.read_csv('dataset.csv')
print("--- Dataset Verification ---")
print(f"Total Rows: {len(df)}")
print("Columns:", df.columns.tolist())

if 'future_4h_return_pct' in df.columns:
    valid = df['future_4h_return_pct'].notna().sum()
    print(f"Non-null future_4h_return_pct: {valid}")
    print("Head values:", df['future_4h_return_pct'].head(5).tolist())
else:
    print("Missig future_4h_return_pct column")
