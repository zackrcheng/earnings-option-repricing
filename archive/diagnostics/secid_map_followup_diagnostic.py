import pandas as pd

secid_map = pd.read_csv("optionm/secid_map.csv", dtype=str)

print(f"Total rows in secid_map : {len(secid_map):,}")
print(f"Unique (cusip8, secid) pairs : {secid_map[['cusip8','secid']].drop_duplicates().shape[0]:,}")

rows_per_cusip = secid_map.groupby("cusip8").size()
print("\nRows per cusip8 distribution:")
print(rows_per_cusip.value_counts().sort_index())
print(f"\nMean rows per cusip8: {rows_per_cusip.mean():.2f}")

# show an example of a cusip8 with multiple rows
multi = rows_per_cusip[rows_per_cusip > 1].index
if len(multi) > 0:
    print(f"\nExample multi-row cusip8 ({multi[0]}):")
    print(secid_map[secid_map["cusip8"] == multi[0]].to_string())