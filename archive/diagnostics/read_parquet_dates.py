import pandas as pd

df = pd.read_parquet("crsp/crsp_daily.parquet", columns=["date"])
print(f"Rows      : {len(df):,}")
print(f"Date min  : {df['date'].min().date()}")
print(f"Date max  : {df['date'].max().date()}")