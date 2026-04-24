import pandas as pd

secid_map = pd.read_csv("optionm/secid_map.csv", dtype=str)

# cardinality check
mapping_counts = secid_map.groupby("cusip8")["secid"].nunique()

print("\n[DIAGNOSTIC] cusip8 → secid cardinality:")
print(mapping_counts.value_counts().sort_index())
print(f"\n  Total unique cusip8s : {mapping_counts.shape[0]:,}")
print(f"  Total unique secids  : {secid_map['secid'].nunique():,}")
print(f"  Mean secids per cusip: {mapping_counts.mean():.2f}")

# inspect high-multiplicity cases
suspicious = mapping_counts[mapping_counts >= 4].index
if len(suspicious) > 0:
    print(f"\n  cusip8s with 4+ secids: {len(suspicious):,}")
    print(secid_map[secid_map["cusip8"].isin(suspicious[:5])].sort_values("cusip8").to_string())
else:
    print("\n  No cusip8s with 4+ secids — mapping looks clean")