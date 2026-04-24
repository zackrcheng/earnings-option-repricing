"""
phase5_impkurt_V5.py  ── CORRECTED
─────────────────────────────────────────────────────────────────
Constructs IMPKURT from Stage 2 skew panel and merges into the
final panel (panel_stage4_final.csv) → saves panel_stage5_final.csv

Fixes applied vs. V4:
  [C1] exdate added to merge keys → prevents many-to-many inflation
  [C2] LN_PC_VLM explicitly merged from Stage 2 if missing
  [M3] LN_VIX removed from winsorization (market-level variable)
  [M4] LN_MRKCAP unit verified via median check before conversion
  [m5] IMPKURT uses direct log formula — no intermediate IV_ATM col
  [m6] Numeric IDs cast to int before merge — avoids float/str mismatch
  [m7] SIC2: pd.to_numeric() first → eliminates "0n"/"No" artefacts
       and correctly zero-pads SIC codes < 1000 (e.g. agriculture)
  [m8] LN_IMPSKEW corr check wrapped defensively — not in required cols
─────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────
STAGE2_PATH = "optionm2025/optionm_stage2_skew.csv"
FINAL_PATH  = "final2025/panel_stage4_final.csv"
OUT_PATH    = "final2025/panel_stage5_final.csv"

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load Stage 2
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 — Load Stage 2 skew panel")
print("=" * 65)

stage2 = pd.read_csv(STAGE2_PATH, low_memory=False)

print(f"  Stage 2 rows loaded : {len(stage2):,}")
print(f"  Columns             : {stage2.columns.tolist()}")

# Confirm the columns needed for IMPKURT are present
required_stage2_cols = ["secid", "date", "event_date", "exdate",   # [C1] exdate required
                        "iv_put", "iv_call", "LN_IMPVOL",
                        "LN_PC_OI", "LN_PC_VLM"]                   # [C2] LN_PC_VLM required

missing_s2 = [c for c in required_stage2_cols if c not in stage2.columns]
if missing_s2:
    raise ValueError(f"Stage 2 is missing required columns: {missing_s2}")
else:
    print(f"  Required Stage 2 columns confirmed ✅")

# ══════════════════════════════════════════════════════════════
# STEP 2 — Construct IMPKURT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — Construct IMPKURT")
print("=" * 65)

# [m5] FIX: Direct log formula — mathematically identical to
#     ln((iv_put + iv_call) / (2 * exp(LN_IMPVOL)))
#   = ln(iv_put + iv_call) - ln(2) - LN_IMPVOL
# Avoids exp() precision loss and removes the need for an
# intermediate IV_ATM column that would pollute the output file.

n_bad_put  = (stage2["iv_put"]  <= 0).sum()
n_bad_call = (stage2["iv_call"] <= 0).sum()
n_bad_atm  = (stage2["LN_IMPVOL"].isna()).sum()

print(f"  iv_put  <= 0     : {n_bad_put:,}  rows (will be NaN)")
print(f"  iv_call <= 0     : {n_bad_call:,}  rows (will be NaN)")
print(f"  LN_IMPVOL null   : {n_bad_atm:,}  rows (will be NaN)")

numerator = stage2["iv_put"] + stage2["iv_call"]

stage2["IMPKURT"] = np.where(
    numerator > 0,
    np.log(numerator) - np.log(2) - stage2["LN_IMPVOL"],
    np.nan
)

# ── Distribution check ─────────────────────────────────────────
print(f"\n  IMPKURT computed    : {stage2['IMPKURT'].notna().sum():,} / {len(stage2):,}")
print(f"  Null IMPKURT        : {stage2['IMPKURT'].isna().sum():,}")
print(f"\n  IMPKURT distribution:")
print(stage2["IMPKURT"].describe().to_string())
print(f"\n  % IMPKURT > 0       : {(stage2['IMPKURT'] > 0).mean()*100:.1f}%")
print(f"  (>0 = wings expensive relative to ATM → fat tails implied)")

# [m8] FIX: LN_IMPSKEW is not in required_stage2_cols — wrap defensively
# to avoid an unguarded KeyError if it is ever absent from Stage 2.
if "LN_IMPSKEW" in stage2.columns:
    corr = stage2["IMPKURT"].corr(stage2["LN_IMPSKEW"].abs())
    print(f"\n  Corr(IMPKURT, |LN_IMPSKEW|) : {corr:.4f}")
    print(f"  (expected positive — more skew = more wing demand)")
else:
    print("\n  NOTE: LN_IMPSKEW not in Stage 2 — skipping corr diagnostic")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Load Final Panel
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — Load final panel")
print("=" * 65)

final = pd.read_csv(FINAL_PATH, low_memory=False)

print(f"  Final panel rows   : {len(final):,}")
print(f"  Final panel cols   : {final.columns.tolist()}")

# Confirm merge keys exist in final panel
# [C1] FIX: exdate is required as a merge key
for col in ["secid", "date", "event_date", "exdate"]:
    assert col in final.columns,  f"MISSING merge key in final panel : {col}"
    assert col in stage2.columns, f"MISSING merge key in stage2      : {col}"

print(f"  Merge keys confirmed: secid, date, event_date, exdate ✅")

# ══════════════════════════════════════════════════════════════
# STEP 4 — Prepare Stage 2 Merge Slice
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — Prepare Stage 2 merge slice")
print("=" * 65)

# [C2] FIX: Bring LN_PC_VLM in from Stage 2 if it is absent from
# the final panel. It was never forwarded through stages 3–4 but
# is a required Phase 5 regression variable.

MERGE_KEYS = ["secid", "date", "event_date", "exdate"]

cols_to_merge = MERGE_KEYS + ["IMPKURT"]

if "LN_PC_VLM" not in final.columns:
    print("  LN_PC_VLM absent from panel — will merge from Stage 2 ✅")
    cols_to_merge.append("LN_PC_VLM")
else:
    print("  LN_PC_VLM already present in panel — skipping from Stage 2")

# Also bring LN_PC_OI if missing
if "LN_PC_OI" not in final.columns:
    print("  LN_PC_OI absent from panel — will merge from Stage 2 ✅")
    cols_to_merge.append("LN_PC_OI")
else:
    print("  LN_PC_OI already present in panel — skipping from Stage 2")

stage2_merge = stage2[cols_to_merge].copy()

# [m6] FIX: Cast numeric IDs to int before merging to avoid silent
# mismatches from float formatting (e.g., "10001.0" vs "10001") or
# leading zero loss that .astype(str) alone does not prevent.
stage2_merge["secid"] = stage2_merge["secid"].astype(float).astype(int)
final["secid"]        = final["secid"].astype(float).astype(int)

# Dates to stripped strings — consistent format both sides
for col in ["date", "event_date", "exdate"]:
    stage2_merge[col] = stage2_merge[col].astype(str).str.strip()
    final[col]        = final[col].astype(str).str.strip()

# [C1] FIX: Check for duplicate keys in stage2_merge BEFORE merging.
# Duplicates here would cause silent row inflation that the post-merge
# assertion would catch — but we want to surface the root cause early.
dupes = stage2_merge.duplicated(subset=MERGE_KEYS).sum()
print(f"\n  Duplicate keys in stage2_merge : {dupes}")

if dupes > 0:
    print("\n  ⚠️  Duplicate key sample:")
    dup_mask = stage2_merge.duplicated(subset=MERGE_KEYS, keep=False)
    print(stage2_merge[dup_mask].sort_values(MERGE_KEYS).head(10).to_string(index=False))
    raise ValueError(
        f"{dupes:,} duplicate (secid, date, event_date, exdate) keys found in "
        "stage2_merge. Resolve in Stage 2 construction before merging."
    )

print(f"  No duplicate keys ✅")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Merge into Final Panel
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 — Merge IMPKURT (and any missing vars) into final panel")
print("=" * 65)

n_before = len(final)

final = final.merge(
    stage2_merge,
    on=MERGE_KEYS,
    how="left"
)

n_after = len(final)

# [C1] Guard: a left-merge on unique keys can never inflate rows.
# If it does, the dedup check above should have caught it, but we
# keep this assertion as a belt-and-suspenders safety net.
assert n_before == n_after, (
    f"Row count changed after merge: {n_before:,} → {n_after:,}. "
    "This should have been caught by the duplicate-key check above."
)

print(f"  Rows before merge  : {n_before:,}")
print(f"  Rows after merge   : {n_after:,}  (no row inflation ✅)")
print(f"  IMPKURT coverage   : {final['IMPKURT'].notna().mean()*100:.1f}%")
print(f"  IMPKURT null rows  : {final['IMPKURT'].isna().sum():,}")

if "LN_PC_VLM" in final.columns:
    print(f"  LN_PC_VLM coverage : {final['LN_PC_VLM'].notna().mean()*100:.1f}%")

# Inspect unmatched rows if any
if final["IMPKURT"].isna().any():
    print("\n  Sample rows with null IMPKURT (unmatched after merge):")
    print(final[final["IMPKURT"].isna()][
        ["secid", "date", "event_date", "exdate", "ticker"]
    ].head(5).to_string(index=False))

# ══════════════════════════════════════════════════════════════
# STEP 6 — Derive Phase 5 Variables
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 — Derive Phase 5 variables")
print("=" * 65)

# ── LN_EXPTIME ─────────────────────────────────────────────────
if "dte" in final.columns:
    final["LN_EXPTIME"] = np.where(
        final["dte"] > 0,
        np.log(final["dte"]),
        np.nan
    )
    n_bad_dte = (final["dte"] <= 0).sum()
    print(f"  LN_EXPTIME coverage : {final['LN_EXPTIME'].notna().mean()*100:.1f}%")
    print(f"  dte <= 0 (zeroed)   : {n_bad_dte:,}")
else:
    print("  WARNING: 'dte' not found — LN_EXPTIME not created")

# ── LN_MRKCAP ──────────────────────────────────────────────────
# [M4] FIX: Verify unit of LN_MKTCAP via median before converting.
# CRSP shrout is in thousands of shares → prc × shrout = $thousands.
# We want $millions. But if a prior stage already converted to millions
# (i.e. divided by 1,000 before logging), subtracting ln(1000) again
# would silently understate market cap by 3 orders of magnitude.
if "LN_MKTCAP" in final.columns:
    median_mktcap_raw = np.exp(final["LN_MKTCAP"].median())
    print(f"\n  Median exp(LN_MKTCAP) = {median_mktcap_raw:,.0f}")

    # Decision thresholds:
    #   > 1,000,000 → raw value is in $thousands (typical CRSP)
    #                  e.g. $20B firm = 20,000,000 in thousands
    #   1,000–1,000,000 → likely already in $millions
    #                  e.g. $20B firm ≈ 20,000 in millions
    #   < 1,000 → unexpected — investigate before proceeding

    if median_mktcap_raw > 1_000_000:
        print("  Units: $thousands confirmed → subtracting ln(1,000)")
        final["LN_MRKCAP"] = final["LN_MKTCAP"] - np.log(1_000)
        unit_label = "$thousands → $millions"

    elif median_mktcap_raw > 1_000:
        print("  Units: already in $millions — LN_MRKCAP = LN_MKTCAP")
        final["LN_MRKCAP"] = final["LN_MKTCAP"].copy()
        unit_label = "$millions (no conversion needed)"

    else:
        raise ValueError(
            f"Unexpected LN_MKTCAP units: median raw value = {median_mktcap_raw:.0f}. "
            "Inspect LN_MKTCAP construction in Stage 3 before proceeding."
        )

    print(f"  LN_MRKCAP unit     : {unit_label}")
    print(f"  LN_MRKCAP coverage : {final['LN_MRKCAP'].notna().mean()*100:.1f}%")
    print(f"  Implied median mktcap: ${np.exp(final['LN_MRKCAP'].median()):,.0f}M")

else:
    print("  WARNING: 'LN_MKTCAP' not found — LN_MRKCAP not created")

# ── QUARTER_FE ─────────────────────────────────────────────────
final["event_date_dt"] = pd.to_datetime(final["event_date"])
final["QUARTER_FE"] = (
    final["event_date_dt"].dt.year.astype(str) + "Q" +
    final["event_date_dt"].dt.quarter.astype(str)
)
print(f"\n  QUARTER_FE unique   : {final['QUARTER_FE'].nunique()}")
print(f"  QUARTER_FE range    : "
      f"{final['QUARTER_FE'].min()} → {final['QUARTER_FE'].max()}")

# Drop helper column
final.drop(columns=["event_date_dt"], inplace=True)

# ── SIC2 ───────────────────────────────────────────────────────
# [m7] FIX: Convert sic to numeric first, then to zero-padded integer
# string. This correctly handles:
#   • NaN / null  → pd.to_numeric coerces to NaN, propagated via .where()
#   • Float repr  → "111.0" becomes int 111 before zfill, not "11"
#   • Leading zeros → SIC 111 (agriculture) correctly becomes "01"
# The string-pipeline approach in V4 produced "0n" for nulls and
# wrong prefixes for SIC codes < 1000 after zfill on float strings.
if "sic" in final.columns:
    sic_num = pd.to_numeric(final["sic"], errors="coerce")
    valid   = sic_num.notna()
    sic_str = (
        sic_num.where(valid).fillna(0)
               .astype(int).astype(str)
               .str.zfill(4).str[:2]
    )
    final["SIC2"] = sic_str.where(valid, other=np.nan)

    n_sic2_null = final["SIC2"].isna().sum()
    print(f"\n  SIC2 unique (excl. null): {final['SIC2'].nunique()}")
    print(f"  SIC2 null count         : {n_sic2_null:,}")

    if n_sic2_null > 0:
        print("  Sample null SIC2 rows:")
        print(final[final["SIC2"].isna()][["secid", "ticker", "sic"]]
              .head(5).to_string(index=False))
else:
    print("  WARNING: 'sic' not found — SIC2 not created")

# ══════════════════════════════════════════════════════════════
# STEP 7 — Winsorization (1st / 99th percentile — firm-level vars only)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 — Winsorization (1st / 99th — firm-level vars only)")
print("=" * 65)

# [M3] FIX: LN_VIX is a market-level variable — identical value for
# every firm on a given date. Cross-sectional winsorization would
# clip legitimate crisis observations (e.g. COVID spike, 2022 rate
# shock) and distort the time-series variation we want to capture.
# It is intentionally excluded from this list.

FIRM_VARS_TO_WINSORIZE = [
    "LN_IMPSKEW",
    "LN_IMPVOL",
    "LN_PC_OI",
    "LN_PC_VLM",
    "IMPKURT",
    "LN_EXPTIME",
    "LN_MRKCAP",
    "LN_TOTALVAR",
    "Dispersion",
    # LN_VIX intentionally excluded — market-level, same value per date
]

print("  NOTE: LN_VIX excluded from winsorization (market-level variable)")

vars_present = [v for v in FIRM_VARS_TO_WINSORIZE if v in final.columns]
vars_missing = [v for v in FIRM_VARS_TO_WINSORIZE if v not in final.columns]

if vars_missing:
    print(f"  WARNING — Not found, skipped: {vars_missing}")

for var in vars_present:
    p01 = final[var].quantile(0.01)
    p99 = final[var].quantile(0.99)
    n_clipped_lo = (final[var] < p01).sum()
    n_clipped_hi = (final[var] > p99).sum()
    final[var]   = final[var].clip(lower=p01, upper=p99)
    print(f"  {var:<18}  lo={n_clipped_lo:>4,}  hi={n_clipped_hi:>4,}  "
          f"clipped  [{p01:.4f}, {p99:.4f}]")

# ══════════════════════════════════════════════════════════════
# STEP 8 — Final Coverage Summary and Save
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 8 — Final coverage summary and save")
print("=" * 65)

FINAL_VARS = [
    "permno",
    "LN_IMPSKEW",
    "IMPKURT",
    "LN_IMPVOL",
    "LN_PC_OI",
    "LN_PC_VLM",
    "LN_EXPTIME",
    "LN_MRKCAP",
    "LN_TOTALVAR",
    "LN_VIX",
    "Dispersion",
    "SIC2",
    "QUARTER_FE",
]

print(f"\n  Coverage summary (post-winsorization):")
for v in FINAL_VARS:
    if v in final.columns:
        pct  = final[v].notna().mean() * 100
        flag = "" if pct >= 95 else "  ⚠️  LOW COVERAGE" if pct < 80 else ""
        print(f"    {v:<18} : {pct:>6.1f}% non-null{flag}")
    else:
        print(f"    {v:<18} : ❌ NOT FOUND IN PANEL")

print(f"\n  Total rows          : {len(final):,}")
print(f"  Unique events       : {final.groupby(['secid','event_date']).ngroups:,}")
print(f"  Unique firms        : {final['permno'].nunique():,}")
print(f"  Date range          : {final['date'].min()} → {final['date'].max()}")
print(f"  Quarter range       : "
      f"{final['QUARTER_FE'].min()} → {final['QUARTER_FE'].max()}")

final.to_csv(OUT_PATH, index=False)

print(f"\n✅  Saved {len(final):,} rows → {OUT_PATH}")

# ── Sample rows ────────────────────────────────────────────────
print(f"\n  Sample (first 3 rows):")
display_cols = [c for c in [
    "secid", "ticker", "event_date", "exdate", "date",
    "LN_IMPSKEW", "IMPKURT", "LN_MRKCAP",
    "LN_TOTALVAR", "Dispersion", "QUARTER_FE", "SIC2"
] if c in final.columns]

print(final[display_cols].head(3).to_string(index=False))