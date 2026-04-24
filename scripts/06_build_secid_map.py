import pandas as pd
import wrds
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVENTS_PATH = "ibes/ibes_stage1_events.csv"
OUTPUT_PATH = "optionm/secid_map.csv"

# ── LOAD EVENTS ───────────────────────────────────────────────────────────────
events = pd.read_csv(EVENTS_PATH)

print("Columns found:", events.columns.tolist())
print(f"Rows: {len(events):,}")

for col in ["cusip_x", "cusip_y"]:
    n_null   = events[col].isna().sum()
    n_unique = events[col].nunique()
    sample   = events[col].dropna().iloc[0] if not events[col].isna().all() else "N/A"
    print(f"  {col}: {n_unique:,} unique, {n_null:,} nulls, sample='{sample}'")

# ── CHOOSE CUSIP COLUMN ───────────────────────────────────────────────────────
CUSIP_COL = "cusip_x"
DATE_COL  = "anndats"

events["cusip8"] = events[CUSIP_COL].astype(str).str.strip().str[:8]
events[DATE_COL] = pd.to_datetime(events[DATE_COL])

cusip8_list = events["cusip8"].dropna().unique().tolist()
cusip8_list = [
    c for c in cusip8_list
    if len(c) == 8 and c not in ("00000000", "nannannan")
]
print(f"\nUnique valid CUSIP8s to match: {len(cusip8_list):,}")

# ── CONNECT TO WRDS ───────────────────────────────────────────────────────────
db = wrds.Connection(wrds_username='rh3245')

# ── QUERY OPTIONMETRICS SECNMD ────────────────────────────────────────────────
# NOTE: optionm.secnmd does NOT have an expire_date column.
# It stores history as multiple rows per secid/cusip, each with its own
# effect_date.  We derive expire_date in pandas after fetching.
cusip_tuple = tuple(cusip8_list)

query = f"""
    SELECT secid,
           cusip,
           effect_date
    FROM   optionm.secnmd
    WHERE  cusip IN {cusip_tuple}
    ORDER BY cusip, effect_date
"""

print("Querying optionm.secnmd ...")
secnmd = db.raw_sql(query)
db.close()

print(f"Raw SECNMD rows returned: {len(secnmd):,}")
print(secnmd.head())

# ── DERIVE expire_date ────────────────────────────────────────────────────────
# For each (cusip, secid) group, the record is valid from effect_date until
# the day before the next effect_date.  The last record is valid indefinitely.
secnmd["cusip8"]      = secnmd["cusip"].astype(str).str.strip().str[:8]
secnmd["effect_date"] = pd.to_datetime(secnmd["effect_date"])

secnmd = secnmd.sort_values(["cusip8", "secid", "effect_date"]).reset_index(drop=True)

secnmd["expire_date"] = (
    secnmd
    .groupby(["cusip8", "secid"])["effect_date"]
    .shift(-1)
    - pd.Timedelta(days=1)
)
secnmd["expire_date"] = secnmd["expire_date"].fillna(pd.Timestamp("2099-12-31"))

# ── MATCH RATE ────────────────────────────────────────────────────────────────
matched_cusips = set(secnmd["cusip8"].unique())
unmatched      = [c for c in cusip8_list if c not in matched_cusips]

print(f"\nCUSIP8s matched to SECID : {len(matched_cusips):,}")
print(f"CUSIPs with no SECID     : {len(unmatched):,}")
if unmatched:
    print(f"  First 10 unmatched     : {unmatched[:10]}")

# ── BUILD MAP ─────────────────────────────────────────────────────────────────
secid_map = (
    secnmd[["secid", "cusip8", "effect_date", "expire_date"]]
    .drop_duplicates()
    .sort_values(["cusip8", "effect_date"])
)

# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs("optionm", exist_ok=True)
secid_map.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅  Saved {len(secid_map):,} rows → {OUTPUT_PATH}")
print(secid_map.head(10))

# ── DIAGNOSTIC: DATE-CONSTRAINED MATCH RATE ───────────────────────────────────
merged = events.merge(secid_map, on="cusip8", how="left")
merged_valid = merged[
    (merged[DATE_COL] >= merged["effect_date"]) &
    (merged[DATE_COL] <= merged["expire_date"])
]

print(f"\nEvents with valid SECID match (date-constrained): "
      f"{len(merged_valid):,} out of {len(events):,}")
print(f"Match rate: {len(merged_valid) / len(events) * 100:.1f}%")

# ── FLAG IF cusip_y MIGHT BE BETTER ──────────────────────────────────────────
events["cusip8_y"] = events["cusip_y"].astype(str).str.strip().str[:8]
extra = set(events["cusip8_y"].dropna().unique()) - matched_cusips
print(f"\nAdditional CUSIP8s in cusip_y not already matched: {len(extra):,}")
if extra:
    print("  Consider switching CUSIP_COL to 'cusip_y' or merging both")