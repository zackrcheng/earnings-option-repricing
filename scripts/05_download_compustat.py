"""
compustat_qtr_download_V7.py
Run this ONCE to pull Compustat quarterly fundamentals.

IBES event universe : 2015-01-01 → 2025-12-31

FIX vs V6 (57.6% sich non-null → expected ~95%+):

  Root cause diagnosed:
    The LATERAL was successfully finding annual rows but those rows
    themselves had NULL in the sich column — so NULL was returned even
    though the join matched.  Without 'AND fa.sich IS NOT NULL' inside
    the LATERAL, we were fetching the most recent annual row regardless
    of whether it contained a usable SIC code.

  Two-stage fix:
    Stage 1 — LATERAL with 'AND fa.sich IS NOT NULL':
      Now skips null-sich annual rows and finds the most recent one
      that actually carries a value.  Recovers the bulk of missing codes.

    Stage 2 — COALESCE fallback to comp.company.sic:
      For firms that genuinely never appear in funda under INDL/STD/D/C
      (e.g. shell companies, very recent IPOs), comp.company.sic provides
      the current static SIC code.  This is the last-resort fallback.

  Priority: sich from funda (time-varying) > sic from company (static).
"""

import wrds
import os

db = wrds.Connection(wrds_username="rh3245")
os.makedirs("compustat", exist_ok=True)

DATE_START = "2013-10-01"
DATE_END   = "2026-06-30"

comp = db.raw_sql(f"""
    SELECT
        f.gvkey,
        f.datadate,
        f.rdq,
        f.fyearq,
        f.fqtr,
        f.saleq,
        f.atq,
        COALESCE(sic_lookup.sich, co.sic::smallint) AS sich
    FROM comp.fundq AS f

    -- Stage 1: most recent annual row that actually has a sich value
    LEFT JOIN LATERAL (
        SELECT fa.sich
        FROM comp.funda AS fa
        WHERE fa.gvkey    = f.gvkey
          AND fa.indfmt   = 'INDL'
          AND fa.datafmt  = 'STD'
          AND fa.popsrc   = 'D'
          AND fa.consol   = 'C'
          AND fa.fyear   <= f.fyearq
          AND fa.sich    IS NOT NULL     -- KEY FIX: skip null-sich rows
        ORDER BY fa.fyear DESC
        LIMIT 1
    ) AS sic_lookup ON TRUE

    -- Stage 2: static company-level SIC as last-resort fallback
    LEFT JOIN comp.company AS co
        ON f.gvkey = co.gvkey

    WHERE f.datadate BETWEEN '{DATE_START}' AND '{DATE_END}'
      AND f.indfmt   = 'INDL'
      AND f.datafmt  = 'STD'
      AND f.popsrc   = 'D'
      AND f.consol   = 'C'
""", date_cols=["datadate", "rdq"])

# ── diagnostics ────────────────────────────────────────────────────────────
print(f"  Rows          : {len(comp):,}")
print(f"  datadate min  : {comp['datadate'].min().date()}")
print(f"  datadate max  : {comp['datadate'].max().date()}")
print(f"  rdq min       : {comp['rdq'].dropna().min().date()}")
print(f"  rdq max       : {comp['rdq'].dropna().max().date()}")
print(f"  sich non-null : {comp['sich'].notna().sum():,}  "
      f"({comp['sich'].notna().mean():.1%} of rows)")

comp.to_parquet("compustat/compustat_qtr.parquet", index=False)
print(f"\n✅ Saved → 'compustat/compustat_qtr.parquet'  ({len(comp):,} rows)")
db.close()