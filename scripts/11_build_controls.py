"""
phase3_phase4_controls_v11.py

Two root-cause fixes for 55 % PERMNO coverage:

Root cause A (Step 2) — ncusip miss
  For firms like AFL, IBES cusip8 ≠ CRSP ncusip.
  Fix: after the ncusip pass, run a second pass matching on
  the CRSP stocknames 'ticker' column for events still unmatched.

Root cause B (Step 3) — ±1 day event-date shift
  Stage 2 likely shifted after-close announcements to t+1
  (first market-open day), while IBES 'anndats' is the announcement
  night. This breaks the exact event_date == anndats join.
  Fix: cascade join
    Tier A : ticker + event_date (±2 cal-day window, exact first)
    Tier B : ticker + pends (fiscal-quarter key, no date needed)
  Each tier only fills rows still missing permno.

Root cause C (Step 3) — Dispersion ±1 day date-shift
  The same t+1 shift that affects PERMNO matching also breaks the
  exact-match Dispersion join.  ~100% of missing Dispersion rows are
  recoverable by allowing a ±1 day tolerance against permno_map anndats.
  Fix: exact match first, then ±1d fill, both sourced from permno_map.
"""

import pandas as pd
import numpy as np
import os
import gc

EVENTS_PATH = "ibes2025/ibes_stage1_events.csv"
SKEW_PATH   = "optionm2025/optionm_stage2_skew.csv"
CRSP_DAILY  = "crsp/crsp_daily.parquet"
CRSP_NAMES  = "crsp/crsp_names.csv"
CCM_LINK    = "compustat/ccm_link.csv"
COMPUSTAT   = "compustat/compustat_qtr.parquet"
VIX_PATH    = "vix/vix_daily.csv"
OUTPUT_PATH = "final2025/panel_stage4_final.csv"

os.makedirs("final2025", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD IBES EVENTS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Load IBES events")
print("=" * 65)

events = pd.read_csv(EVENTS_PATH, parse_dates=["anndats", "pends"])
cusip_col = "cusip_x" if "cusip_x" in events.columns else "cusip"
events["cusip8"]      = events[cusip_col].astype(str).str.strip().str[:8]
events["_ticker_up"]  = events["ticker"].astype(str).str.strip().str.upper()
print(f"  Events loaded : {len(events):,}")
print(f"  CUSIP col used: {cusip_col!r}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — IBES → CRSP: ncusip (primary) + ticker (fallback)
#
# Root-cause A fix:
#   The original code only used cusip8 ↔ ncusip.  Firms whose IBES/OptionMetrics
#   CUSIP differs from CRSP ncusip (e.g. post-split suffix change, reissued CUSIP)
#   were silently dropped.  We now run a second pass on unmatched events using the
#   'ticker' column that exists in crsp.dsenames / crsp_names.csv.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — IBES → CRSP: ncusip primary + ticker fallback")
print("=" * 65)

crsp_names = pd.read_csv(CRSP_NAMES)
crsp_names.columns = crsp_names.columns.str.lower()
crsp_names["namedt"]    = pd.to_datetime(crsp_names["namedt"],   errors="coerce")
crsp_names["nameendt"]  = pd.to_datetime(crsp_names["nameendt"], errors="coerce")
crsp_names["nameenddt"] = crsp_names["nameendt"].fillna(pd.Timestamp("2099-12-31"))
crsp_names["ncusip"]    = crsp_names["ncusip"].astype(str).str.strip().str[:8]
crsp_names["permno"]    = pd.to_numeric(crsp_names["permno"], errors="coerce").astype("Int64")

has_crsp_ticker = "ticker" in crsp_names.columns
if has_crsp_ticker:
    crsp_names["_crsp_ticker_up"] = crsp_names["ticker"].astype(str).str.strip().str.upper()
    print(f"  CRSP names rows: {len(crsp_names):,}  |  ticker column present ✓")
else:
    print(f"  CRSP names rows: {len(crsp_names):,}  |  WARNING: no 'ticker' col → fallback disabled")

_BRACKET_COLS = ["permno", "namedt", "nameenddt"]


def _date_bracket(df, date_col="anndats"):
    """Keep rows where date_col is within [namedt, nameenddt]."""
    return df.loc[
        (df[date_col] >= df["namedt"]) &
        (df[date_col] <= df["nameenddt"])
    ].copy()


def _dedup_events(df):
    return (df.sort_values("namedt", ascending=False)
              .drop_duplicates(subset=["ticker", "anndats", "pends"])
              .reset_index(drop=True))


# ── Primary: ncusip match ─────────────────────────────────────────────────────
ev_ncusip = (events
             .merge(crsp_names[_BRACKET_COLS + ["ncusip"]],
                    left_on="cusip8", right_on="ncusip",
                    how="inner")
             .pipe(_date_bracket)
             .pipe(_dedup_events))
ev_ncusip["_link"] = "ncusip"

ncusip_event_keys = set(
    zip(ev_ncusip["_ticker_up"], ev_ncusip["anndats"], ev_ncusip["pends"])
)
print(f"\n  [ncusip]  {len(ev_ncusip):>6,} events  |  {ev_ncusip['ticker'].nunique():,} unique tickers")

# ── Fallback: ticker match ────────────────────────────────────────────────────
ev_ticker_fb = pd.DataFrame()

if has_crsp_ticker:
    events["_ekey"] = list(zip(events["_ticker_up"],
                                events["anndats"],
                                events["pends"]))
    events_um = events[~events["_ekey"].isin(ncusip_event_keys)].copy()
    print(f"  Unmatched after ncusip pass: {len(events_um):,} events "
          f"({events_um['ticker'].nunique():,} tickers)")

    if len(events_um) > 0:
        ev_ticker_fb = (events_um
                        .merge(crsp_names[_BRACKET_COLS + ["_crsp_ticker_up"]],
                               left_on="_ticker_up", right_on="_crsp_ticker_up",
                               how="inner")
                        .pipe(_date_bracket)
                        .pipe(_dedup_events))
        ev_ticker_fb["_link"] = "ticker"

        new_tickers = (set(ev_ticker_fb["_ticker_up"])
                       - set(ev_ncusip["_ticker_up"]))
        print(f"  [ticker]  {len(ev_ticker_fb):>6,} events  |  "
              f"{ev_ticker_fb['ticker'].nunique():,} tickers "
              f"({len(new_tickers):,} net-new vs ncusip)")
        if new_tickers:
            print(f"  Sample net-new tickers: {sorted(new_tickers)[:20]}")
else:
    events_um = events[~events["_ekey"].isin(ncusip_event_keys)].copy()
    print(f"  Skipping ticker fallback — {len(events_um):,} events remain unlinked")

# ── Combine: ncusip preferred where both methods fire ────────────────────────
ev_crsp = (pd.concat([ev_ncusip, ev_ticker_fb], ignore_index=True)
             .sort_values("_link")          # "ncusip" < "ticker" → ncusip wins
             .drop_duplicates(subset=["ticker", "anndats", "pends"])
             .reset_index(drop=True))

print(f"\n  Combined : {len(ev_crsp):>6,} events  |  {ev_crsp['ticker'].nunique():,} unique tickers")
print(f"  By link method:")
print(ev_crsp["_link"].value_counts().to_string())

_disp_cols = ["Dispersion"] if "Dispersion" in ev_crsp.columns else []
permno_map = ev_crsp[["ticker", "anndats", "pends", "permno"] + _disp_cols].copy()

# Clean up temporary columns
events = events.drop(columns=["_ekey", "_ticker_up"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LOAD STAGE 2 SKEW PANEL; ATTACH PERMNO (cascade join)
#
# Root-cause B fix:
#   Stage 2 shifts after-close events to t+1; IBES stores the announcement
#   night date.  An exact event_date==anndats join misses these events.
#   Cascade strategy:
#     Tier A: unified (ticker_upper, event_date) lookup built from permno_map
#             with delta ∈ {0, +1, -1, +2, -2} calendar days, preferring delta=0.
#     Tier B: (ticker_upper, pends) lookup — fiscal quarter is unambiguous
#             regardless of date shift; fills anything Tier A missed.
#
# Root-cause C fix:
#   The same ±1d shift breaks the exact-match Dispersion join.
#   Fix: attach Dispersion with exact match first, then ±1d tolerance fill,
#   both sourced directly from permno_map (the authoritative IBES source).
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Load Stage 2 skew panel; attach PERMNO (cascade)")
print("=" * 65)

skew = pd.read_csv(SKEW_PATH,
                   parse_dates=["date", "event_date", "pends", "exdate"])
skew["_tu"] = skew["ticker"].astype(str).str.strip().str.upper()
print(f"  Skew rows: {len(skew):,}")

# ── Pre-merge diagnostics ─────────────────────────────────────────────────────
pmap_diag = (permno_map
             .assign(_tu=lambda d: d["ticker"].str.strip().str.upper())
             .rename(columns={"anndats": "event_date"})
             [["_tu", "event_date", "pends", "permno"]])

skew_ev = skew[["_tu", "event_date", "pends"]].drop_duplicates()
print(f"\n  Unique skew events : {len(skew_ev):,}")
print(f"  permno_map events  : {len(pmap_diag):,}")

def _pct_overlap(left, right, keys, label):
    n = len(left[keys].drop_duplicates().merge(
               right[keys].drop_duplicates(), on=keys))
    d = len(left[keys].drop_duplicates())
    print(f"  {label:50s}: {n:>6,}/{d:,}  ({100*n/d:.1f}%)")

_pct_overlap(skew_ev, pmap_diag, ["_tu","event_date","pends"], "exact (ticker+event_date+pends)")
_pct_overlap(skew_ev, pmap_diag, ["_tu","event_date"],          "relax (ticker+event_date)")
_pct_overlap(skew_ev, pmap_diag, ["_tu","pends"],               "pends (ticker+pends)")

# ±1 day shift overlap
pmap_shifted = (pd.concat([
    pmap_diag.assign(event_date=pmap_diag["event_date"] + pd.Timedelta(days=d))
    for d in [-1, 0, 1]
]).drop_duplicates(subset=["_tu","event_date"]))
_pct_overlap(skew_ev, pmap_shifted, ["_tu","event_date"], "±1 day (ticker+event_date±1)")

# AFL-specific diagnostic
print("\n  ── AFL diagnostic ──")
print("  AFL rows in permno_map (event_date, pends, permno):")
afl_pm = pmap_diag[pmap_diag["_tu"] == "AFL"].head(6)
print(afl_pm[["_tu","event_date","pends","permno"]].to_string(index=False)
      if len(afl_pm) else "    *** AFL NOT FOUND in permno_map ***")
print("  AFL unique events in skew:")
afl_sk = skew_ev[skew_ev["_tu"] == "AFL"].head(6)
print(afl_sk.to_string(index=False)
      if len(afl_sk) else "    *** AFL NOT FOUND in skew ***")

# ── Build Tier A lookup: (ticker_upper, event_date) with ±2 day tolerance ─────
#    Sort ascending by |delta| so that drop_duplicates keeps the exact match.
tier_a_frames = []
for delta in [0, 1, -1, 2, -2]:
    tmp = (permno_map[["ticker","anndats","permno"]].copy()
           .assign(_tu=lambda d: d["ticker"].str.strip().str.upper(),
                   event_date=lambda d: d["anndats"] + pd.Timedelta(days=delta),
                   _absd=abs(delta)))
    tier_a_frames.append(tmp[["_tu","event_date","permno","_absd"]])

lookup_a = (pd.concat(tier_a_frames, ignore_index=True)
              .sort_values("_absd")
              .drop_duplicates(subset=["_tu","event_date"])
              [["_tu","event_date","permno"]])

# ── Build Tier B lookup: (ticker_upper, pends) ───────────────────────────────
lookup_b = (permno_map[["ticker","pends","permno"]]
            .assign(_tu=lambda d: d["ticker"].str.strip().str.upper())
            [["_tu","pends","permno"]]
            .drop_duplicates(subset=["_tu","pends"]))

# ── Tier A merge ─────────────────────────────────────────────────────────────
skew = skew.merge(lookup_a, on=["_tu","event_date"], how="left")
n_a = skew["permno"].notna().sum()
print(f"\n  Tier A (±2d event_date) : {n_a:,}/{len(skew):,} rows "
      f"({100*n_a/len(skew):.1f}%)")

# ── Tier B merge (fills rows still missing permno) ────────────────────────────
mask_b = skew["permno"].isna()
if mask_b.sum() > 0:
    fill_b = (skew.loc[mask_b]
                  .reset_index()
                  [["index","_tu","pends"]]
                  .merge(lookup_b, on=["_tu","pends"], how="left")
                  .drop_duplicates(subset=["index"])
                  .set_index("index")["permno"])
    skew.loc[mask_b, "permno"] = fill_b
    n_b = skew.loc[mask_b, "permno"].notna().sum()
    print(f"  Tier B (pends fallback)  : +{n_b:,} rows filled  "
          f"({100*skew['permno'].notna().mean():.1f}% total)")

skew["permno"] = skew["permno"].astype("Int64")
n_total = skew["permno"].notna().sum()
print(f"\n  Final PERMNO coverage: {n_total:,}/{len(skew):,} ({100*n_total/len(skew):.1f}%)")

# ── Attach Dispersion: exact match first, then ±1d tolerance fill ────────────
#
#   Root-cause C fix: the same t+1 date shift that affects PERMNO matching
#   breaks the exact-match Dispersion join.  We first do an exact match on
#   (ticker, event_date, pends) against permno_map, then fill anything still
#   missing by shifting permno_map anndats by ±1 calendar day.  Both passes
#   use permno_map as the authoritative source so the fill is never overwritten.
#
if "Dispersion" in permno_map.columns:
    # Drop any Dispersion column already in skew (e.g. carried over from Stage 2 CSV)
    if "Dispersion" in skew.columns:
        skew = skew.drop(columns=["Dispersion"])

    # Pass 1 — exact match
    disp_exact = (permno_map[["ticker", "anndats", "pends", "Dispersion"]]
                  .rename(columns={"anndats": "event_date"}))
    skew = skew.merge(disp_exact, on=["ticker", "event_date", "pends"], how="left")
    n_exact = skew["Dispersion"].notna().sum()
    print(f"\n  Dispersion (exact match)      : {n_exact:,}/{len(skew):,} "
          f"({100*n_exact/len(skew):.1f}%)")

    # Pass 2 — ±1d tolerance fill sourced from permno_map
    disp_source = (permno_map[["ticker", "anndats", "pends", "Dispersion"]]
                   .dropna(subset=["Dispersion"])
                   .drop_duplicates(subset=["ticker", "anndats", "pends"]))

    disp_filled = 0
    for delta in [1, -1]:
        still_missing = skew["Dispersion"].isna()
        if still_missing.sum() == 0:
            break
        shifted = disp_source.copy()
        shifted["event_date"] = shifted["anndats"] + pd.Timedelta(days=delta)
        shifted = shifted.drop(columns=["anndats"])
        skew = skew.merge(
            shifted.rename(columns={"Dispersion": "_Disp_fill"}),
            on=["ticker", "event_date", "pends"],
            how="left"
        )
        mask = still_missing & skew["_Disp_fill"].notna()
        skew.loc[mask, "Dispersion"] = skew.loc[mask, "_Disp_fill"]
        disp_filled += mask.sum()
        skew = skew.drop(columns=["_Disp_fill"])

    print(f"  Dispersion filled via ±1d     : +{disp_filled:,} rows")
    print(f"  Dispersion coverage after fix : "
          f"{skew['Dispersion'].notna().mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CRSP DAILY: LOAD AND COMPUTE MARKET CAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — CRSP daily: load and compute market cap")
print("=" * 65)

crsp = (pd.read_parquet(CRSP_DAILY)
        if CRSP_DAILY.endswith(".parquet")
        else pd.read_csv(CRSP_DAILY, parse_dates=["date"]))
crsp.columns = crsp.columns.str.lower()
crsp["date"]    = pd.to_datetime(crsp["date"])
crsp["permno"]  = pd.to_numeric(crsp["permno"],  errors="coerce").astype("Int64")
crsp["prc"]     = pd.to_numeric(crsp["prc"],     errors="coerce")
crsp["ret"]     = pd.to_numeric(crsp["ret"],     errors="coerce")
crsp["shrout"]  = pd.to_numeric(crsp["shrout"],  errors="coerce")
crsp["vol"]     = pd.to_numeric(crsp["vol"],     errors="coerce")
# ── FIX: pull cfacshr; mktcap in $millions (shrout is in thousands) ───────────
crsp["cfacshr"] = pd.to_numeric(crsp["cfacshr"], errors="coerce")
crsp["mktcap"]  = crsp["prc"].abs() * crsp["shrout"] / 1000   # $millions
print(f"  CRSP daily rows (all): {len(crsp):,}")

# ── Capture complete NYSE trading calendar before filtering ───────────────────
#    Must be done here (before subset) to cover all possible event dates,
#    including firms that may have sparse CRSP coverage in the sample window.
trading_calendar = pd.to_datetime(sorted(crsp["date"].dropna().unique()))
print(f"  Trading calendar captured: {len(trading_calendar):,} days "
      f"({trading_calendar[0].date()} → {trading_calendar[-1].date()})")

relevant_permnos = set(skew["permno"].dropna().astype(int).unique())
crsp = crsp[crsp["permno"].isin(relevant_permnos)].copy()
print(f"  CRSP filtered to {len(relevant_permnos):,} permnos → {len(crsp):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TOTALVAR: 91-DAY HISTORICAL RETURN VARIANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — TOTALVAR: 91-day historical return variance")
print("=" * 65)

# ev_crsp already has the full (permno, anndats) set from Step 2
event_ids = (ev_crsp[["permno","anndats"]]
             .drop_duplicates()
             .dropna(subset=["permno"])
             .copy())
event_ids["permno"] = event_ids["permno"].astype(int)

crsp_by_perm = {
    int(p): g[["date","ret"]].reset_index(drop=True)
    for p, g in crsp.groupby("permno")
}

totalvar_rows = []
for _, row in event_ids.iterrows():
    pno = int(row["permno"])
    ann = row["anndats"]
    if pno not in crsp_by_perm:
        continue
    dfp   = crsp_by_perm[pno]
    start = ann - pd.Timedelta(days=91)
    end   = ann - pd.Timedelta(days=1)
    rets  = dfp.loc[(dfp["date"] >= start) & (dfp["date"] <= end), "ret"].dropna()
    if len(rets) >= 10:
        totalvar_rows.append({"permno": pno, "anndats": ann,
                               "TOTALVAR": float(rets.var(ddof=1))})

totalvar = pd.DataFrame(totalvar_rows)
print(f"  TOTALVAR computed: {len(totalvar):,} / {len(event_ids):,} events "
      f"({100*len(totalvar)/max(len(event_ids),1):.1f}%)")
del crsp_by_perm
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MERGE CRSP DAILY VARS ONTO PANEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — Merge CRSP daily vars onto panel")
print("=" * 65)

# ── FIX: include cfacshr in slim frame ───────────────────────────────────────
crsp_slim = crsp[["permno","date","prc","ret","vol","mktcap","cfacshr"]].copy()
del crsp
gc.collect()

skew["permno"]      = skew["permno"].astype("Int64")
crsp_slim["permno"] = crsp_slim["permno"].astype("Int64")

skew = skew.merge(crsp_slim, on=["permno","date"], how="left")
print(f"  mktcap   coverage: {skew['mktcap'].notna().mean()*100:.1f}%")
print(f"  cfacshr  coverage: {skew['cfacshr'].notna().mean()*100:.1f}%")
# mktcap is already in $millions; log is therefore ln(mktcap_millions)
skew["LN_MKTCAP"] = np.log(skew["mktcap"].abs().replace(0, np.nan))

# TOTALVAR: match on (permno, event_date)
# Tier A: exact event_date == anndats
# Tier B: event_date ± 1 day (mirrors the date-shift logic from Step 3)
totalvar["permno"] = totalvar["permno"].astype("Int64")

tv_frames = []
for delta in [0, 1, -1]:
    tmp = totalvar.copy()
    tmp["event_date"] = tmp["anndats"] + pd.Timedelta(days=delta)
    tmp["_absd"] = abs(delta)
    tv_frames.append(tmp[["permno","event_date","TOTALVAR","_absd"]])

tv_lookup = (pd.concat(tv_frames)
               .sort_values("_absd")
               .drop_duplicates(subset=["permno","event_date"])
               [["permno","event_date","TOTALVAR"]])

skew = skew.merge(tv_lookup, on=["permno","event_date"], how="left")
print(f"  TOTALVAR coverage: {skew['TOTALVAR'].notna().mean()*100:.1f}%")
skew["LN_TOTALVAR"] = np.log(skew["TOTALVAR"].replace(0, np.nan))

del crsp_slim, totalvar, tv_lookup
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — COMPUSTAT QUARTERLY FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — Compustat quarterly fundamentals")
print("=" * 65)

ccm = pd.read_csv(CCM_LINK)
ccm.columns = ccm.columns.str.lower()
lperm_col = "lpermno" if "lpermno" in ccm.columns else "permno"
ccm["permno"]    = pd.to_numeric(ccm[lperm_col],    errors="coerce").astype("Int64")
ccm["gvkey"]     = ccm["gvkey"].astype(str).str.strip().str.zfill(6)
ccm["linkdt"]    = pd.to_datetime(ccm["linkdt"],     errors="coerce")
ccm["linkenddt"] = (pd.to_datetime(ccm["linkenddt"], errors="coerce")
                      .fillna(pd.Timestamp("2099-12-31")))
if "linktype" in ccm.columns:
    ccm = ccm[ccm["linktype"].isin(["LC","LU","LS"])]
if "linkprim" in ccm.columns:
    ccm = ccm[ccm["linkprim"].isin(["P","C"])]
print(f"  CCM link records (LC/LU/LS, P/C): {len(ccm):,}")

comp = (pd.read_parquet(COMPUSTAT)
        if COMPUSTAT.endswith(".parquet")
        else pd.read_csv(COMPUSTAT, parse_dates=["datadate"]))
comp.columns = comp.columns.str.lower()

if "sich" in comp.columns and "sic" not in comp.columns:
    comp = comp.rename(columns={"sich": "sic"})
    print("  Renamed 'sich' → 'sic'")

comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
comp["gvkey"]    = comp["gvkey"].astype(str).str.strip().str.zfill(6)
comp["saleq"]    = pd.to_numeric(comp["saleq"], errors="coerce")
comp["atq"]      = pd.to_numeric(comp["atq"],   errors="coerce")
comp["sic"] = (pd.to_numeric(comp["sic"], errors="coerce")
               .apply(lambda x: str(int(x)).zfill(4) if pd.notna(x) else np.nan))
print(f"  Compustat quarterly rows : {len(comp):,}")
print(f"  sic non-null in parquet  : {comp['sic'].notna().mean()*100:.1f}%")

# permno → gvkey with date bracket; include ev_crsp permno/anndats pairs
event_ids2 = (ev_crsp[["permno","anndats"]]
              .drop_duplicates()
              .dropna(subset=["permno"])
              .copy())
event_ids2["permno"] = event_ids2["permno"].astype("Int64")

ev_perm = event_ids2.merge(
    ccm[["permno","gvkey","linktype","linkdt","linkenddt"]],
    on="permno", how="inner"
)
ev_perm = ev_perm[
    (ev_perm["anndats"] >= ev_perm["linkdt"]) &
    (ev_perm["anndats"] <= ev_perm["linkenddt"])
].copy()
link_priority = {"LC": 0, "LU": 1, "LS": 2}
ev_perm["_lp"] = ev_perm["linktype"].map(link_priority)
ev_perm = (ev_perm.sort_values("_lp")
                  .drop_duplicates(subset=["permno","anndats"])
                  .drop(columns=["_lp"]))
print(f"  PERMNO→GVKEY valid links : {len(ev_perm):,}")

ev_comp = (ev_perm
           .merge(comp[["gvkey","datadate","saleq","atq","sic"]],
                  on="gvkey", how="left")
           .query("datadate <= anndats")
           .sort_values("datadate", ascending=False)
           .drop_duplicates(subset=["permno","anndats"])
           [["permno","anndats","saleq","atq","sic"]]
           .copy())
print(f"  Compustat matches        : {len(ev_comp):,}")

# Merge with ±1 day tolerance (same logic as TOTALVAR)
ev_comp["permno"] = ev_comp["permno"].astype("Int64")
comp_frames = []
for delta in [0, 1, -1]:
    tmp = ev_comp.copy()
    tmp["event_date"] = tmp["anndats"] + pd.Timedelta(days=delta)
    tmp["_absd"] = abs(delta)
    comp_frames.append(tmp[["permno","event_date","saleq","atq","sic","_absd"]])

comp_lookup = (pd.concat(comp_frames)
                 .sort_values("_absd")
                 .drop_duplicates(subset=["permno","event_date"])
                 [["permno","event_date","saleq","atq","sic"]])

skew = skew.merge(comp_lookup, on=["permno","event_date"], how="left")
print(f"  saleq coverage: {skew['saleq'].notna().mean()*100:.1f}%")
print(f"  atq   coverage: {skew['atq'].notna().mean()*100:.1f}%")
print(f"  sic   coverage: {skew['sic'].notna().mean()*100:.1f}%")

del comp, ccm, ev_perm, ev_comp, comp_lookup
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — VIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — VIX")
print("=" * 65)

vix = pd.read_csv(VIX_PATH)
vix.columns = vix.columns.str.upper()
date_col = next(c for c in vix.columns if "DATE" in c)
vix_col  = next(c for c in vix.columns if "VIX"  in c)
vix = vix.rename(columns={date_col: "date", vix_col: "vix"})
vix["date"] = pd.to_datetime(vix["date"])
vix["vix"]  = pd.to_numeric(vix["vix"], errors="coerce")
vix = vix.dropna(subset=["vix"])
vix["LN_VIX"] = np.log(vix["vix"])
print(f"  VIX rows: {len(vix):,}  "
      f"({vix['date'].min().date()} → {vix['date'].max().date()})")
skew = skew.merge(vix[["date","LN_VIX"]], on="date", how="left")
print(f"  LN_VIX coverage: {skew['LN_VIX'].notna().mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — FINAL CLEANUP AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9 — Final cleanup and save")
print("=" * 65)

# ── FIX: recompute rel_day as true trading-day offset ─────────────────────────
#    The Stage 2 rel_day column is calendar days ((date - event_date).dt.days).
#    We overwrite it here using the CRSP trading calendar captured in Step 4.
#    td_to_int maps each calendar date to its integer position in the sorted
#    sequence of NYSE trading days; the difference of two positions is therefore
#    the signed number of trading days between them.
td_to_int      = {d: i for i, d in enumerate(trading_calendar)}
date_int       = skew["date"].map(td_to_int)
event_date_int = skew["event_date"].map(td_to_int)
skew["rel_day"] = (date_int - event_date_int).astype("Int64")
print(f"  rel_day recomputed as trading-day offset (not calendar days)")
print(f"  rel_day range: {skew['rel_day'].min()} → {skew['rel_day'].max()}")

n_before = len(skew)
skew = skew[skew["rel_day"].between(-10, 10)].copy()
dropped = n_before - len(skew)
print(f"  Rows dropped outside [-10,+10]: {dropped:,}")
if dropped == 0:
    print("  Window bounds confirmed ✅")

skew = skew.drop_duplicates(subset=["secid","date","event_date","pends"])

# Drop internal helper columns
skew = skew.drop(columns=[c for c in ["_tu","_ticker_up"] if c in skew.columns])

output_cols = [
    "secid","permno","ticker","pends","event_date","date","rel_day",
    "Dispersion","exdate","dte",
    "LN_IMPSKEW","LN_IMPVOL","LN_PC_OI","LN_PC_VLM",
    "iv_put","delta_put","oi_put","strike_put",
    "iv_call","delta_call","oi_call","strike_call",
    "LN_MKTCAP","mktcap","prc","ret","vol","cfacshr",
    "TOTALVAR","LN_TOTALVAR",
    "saleq","atq","sic",
    "LN_VIX",
]
output_cols = [c for c in output_cols if c in skew.columns]
skew = skew[output_cols].copy()

print("\n  Coverage summary:")
check_cols = ["permno","Dispersion","LN_MKTCAP","LN_TOTALVAR",
              "saleq","atq","sic","LN_VIX","LN_IMPSKEW"]
for col in check_cols:
    if col in skew.columns:
        print(f"    {col:20s}: {skew[col].notna().mean()*100:5.1f}% non-null")

print(f"\n  Total rows     : {len(skew):,}")
print(f"  Unique events  : {skew.groupby(['secid','event_date']).ngroups:,}")
print(f"  Unique firms   : {skew['secid'].nunique():,}")
print(f"  Date range     : {skew['date'].min().date()} → {skew['date'].max().date()}")

skew.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved {len(skew):,} rows → {OUTPUT_PATH}")
print(skew[output_cols[:12]].head(5).to_string())