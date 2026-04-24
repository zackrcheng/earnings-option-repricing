"""
download_crsp_daily_parquet_V4.py
Run this ONCE to pull all raw data files.

IBES event universe : 2015-01-01 → 2025-12-31
Event window        : [−10, +10] TRADING DAYS around each event date
TOTALVAR lookback   : 91 TRADING DAYS ending at t−1 for each event

Calendar bounds passed to WRDS SQL are intentionally generous:

  DATE_START = 2014-08-01
    91 trading days ≈ 128 calendar days before 2015-01-01 → 2014-08-26.
    Using 2014-08-01 adds a ~25-day calendar buffer on the left.

  DATE_END = 2026-01-20
    10 trading days ≈ 14 calendar days after 2025-12-31 → 2026-01-14.
    Using 2026-01-20 adds a ~6-day calendar buffer on the right.

Changes from V3:
  - crsp.dsf        → crsp_a_stock.dsf_v2  (legacy table frozen at 2024-12-31)
  - crsp.dsenames   → crsp_names derived from crsp_a_stock.dsf_v2
                      (crsp_a_stock.stocknames also frozen at 2024-12-31)
  - Column renames  → aliased back to original names so all downstream
                      Phase 3/4/5 code requires zero changes
  - dlyret dtype    → stored as string in CRSP v2; coerced via pd.to_numeric()
                      after pull before saving to parquet
  - comnam          → not present in dsf_v2; written as NULL placeholder so
                      the column exists in the CSV for any downstream reference
"""

import wrds
import pandas as pd
import os

db = wrds.Connection(wrds_username="rh3245")

os.makedirs("crsp",      exist_ok=True)
os.makedirs("compustat", exist_ok=True)

DATE_START = "2014-08-01"   # 91 trading-day TOTALVAR lookback for earliest Jan 2015 events
DATE_END   = "2026-01-20"   # +10 trading-day window for latest Dec 2025 events

# ── 1. CRSP Names — derived from dsf_v2 ──────────────────────────────────────
# crsp_a_stock.stocknames and crsp.dsenames are both frozen at 2024-12-31.
# We therefore derive the names mapping directly from dsf_v2, which runs
# through 2025-12-31.  comnam is absent from dsf_v2 so a NULL placeholder
# is written; it is not used in any merge key.
# nameendt logic:
#   - dlydelflg = 'N' means the security is still active on that row.
#   - Any other dlydelflg value indicates a deletion event.
#   - MAX of deletion-flagged dates gives the last day of the CUSIP window.
#   - COALESCE to '2099-12-31' for securities with no deletion in the window
#     (i.e. still active), so the downstream date-bracket merge never drops them.
print("Downloading crsp_names (derived from dsf_v2)...")
crsp_names = db.raw_sql(f"""
    SELECT
        permno,
        hdrcusip                                              AS ncusip,
        CAST(NULL AS TEXT)                                    AS comnam,
        MIN(ticker)                                           AS ticker,
        MIN(dlycaldt)                                         AS namedt,
        COALESCE(
            MAX(CASE WHEN dlydelflg != 'N' THEN dlycaldt END),
            '2099-12-31'
        )                                                     AS nameendt
    FROM crsp_a_stock.dsf_v2
    WHERE dlycaldt BETWEEN '{DATE_START}' AND '{DATE_END}'
      AND hdrcusip IS NOT NULL
    GROUP BY permno, hdrcusip
""", date_cols=["namedt", "nameendt"])
crsp_names.to_csv("crsp/crsp_names.csv", index=False)
print(f"  → {len(crsp_names):,} rows saved")
print(f"     namedt   min : {crsp_names['namedt'].min().date()}")
print(f"     nameendt max : {crsp_names['nameendt'].max().date()}")

# ── 2. CCM Link Table (small) ─────────────────────────────────────────────────
# crsp.ccmxpf_linktable is not affected by the CRSP v2 migration and
# remains available and current under the legacy schema path.
print("Downloading ccm_link...")
ccm_link = db.raw_sql("""
    SELECT gvkey, lpermno, lpermco, linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LC','LU','LS')
      AND linkprim IN ('P','C')
""", date_cols=["linkdt", "linkenddt"])
ccm_link.to_csv("compustat/ccm_link.csv", index=False)
print(f"  → {len(ccm_link):,} rows saved")

# ── 3. CRSP Daily (LARGE) ─────────────────────────────────────────────────────
# Source : crsp_a_stock.dsf_v2  (covers through 2025-12-31)
# All v2 column names are aliased back to the original v1 names so that
# every downstream script (Phase 3 TOTALVAR, Phase 4 merge, Phase 5 cleaning)
# continues to work without modification.
#
# Column mapping (v2 → alias):
#   dlycaldt     → date
#   dlyprc       → prc
#   dlyret       → ret      ← NOTE: stored as string dtype in v2; coerced below
#   dlyvol       → vol
#   shrout       → shrout   (unchanged)
#   dlycumfacshr → cfacshr
#
# Covers:
#   • [−10, +10] trading-day event window for every IBES event
#   • 91-trading-day TOTALVAR lookback (ending t−1) for every IBES event
print("Downloading crsp_daily from dsf_v2 (this will take a while)...")
crsp_daily = db.raw_sql(f"""
    SELECT
        permno,
        dlycaldt     AS date,
        dlyprc       AS prc,
        dlyret       AS ret,
        dlyvol       AS vol,
        shrout,
        dlycumfacshr AS cfacshr
    FROM crsp_a_stock.dsf_v2
    WHERE dlycaldt BETWEEN '{DATE_START}' AND '{DATE_END}'
""", date_cols=["date"])

# dlyret is stored as string dtype in CRSP v2 — coerce to float64 before
# saving so that TOTALVAR variance calculations work correctly downstream.
crsp_daily["ret"] = pd.to_numeric(crsp_daily["ret"], errors="coerce")

crsp_daily.to_parquet("crsp/crsp_daily.parquet", index=False)
print(f"  → {len(crsp_daily):,} rows saved")
print(f"     date min : {crsp_daily['date'].min().date()}")
print(f"     date max : {crsp_daily['date'].max().date()}")
del crsp_daily

db.close()
print("\n✅ All downloads complete.")