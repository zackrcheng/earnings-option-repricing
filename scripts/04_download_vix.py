import pandas as pd, os

os.makedirs("vix", exist_ok=True)

vix = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS")
vix.columns = ["date", "vix"]   # rename first
vix["date"] = pd.to_datetime(vix["date"])  # then parse
vix["vix"]  = pd.to_numeric(vix["vix"], errors="coerce")
vix = vix.dropna()

vix.to_csv("vix/vix_daily.csv", index=False)
print(f"Saved: {len(vix):,} rows  ({vix['date'].min().date()} → {vix['date'].max().date()})")