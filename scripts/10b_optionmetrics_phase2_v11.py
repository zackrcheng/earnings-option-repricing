import pandas as pd
import numpy as np
import os
import gc
import warnings

# ── NYSE trading-day calendar ─────────────────────────────────────────────────
try:
    import pandas_market_calendars as mcal
    _MCal_AVAILABLE = True
except ImportError:
    _MCal_AVAILABLE = False
    warnings.warn(
        "pandas_market_calendars not found — falling back to Mon–Fri business "
        "days (no NYSE holiday removal).\n"
        "Install with:  pip install pandas-market-calendars",
        stacklevel=2
    )

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_DIR = "optionm2025"
EVENTS_PATH = "ibes2025/ibes_stage1_events.csv"
OUTPUT_PATH = "optionm2025/optionm_stage2_skew.csv"

# FIX 1: range corrected to match downloaded data (2015-01-01 to 2025-12-31).
# 2013–2014 parquet files do not exist locally; 2025 is now included.
YEARS = list(range(2015, 2026))

# FIX 2: declare column lists for pd.read_parquet(..., columns=...).
# Loading only these columns roughly halves peak RAM per year vs reading
# every column in the file.
OPT_COLS = [
    "secid", "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "impl_volatility", "delta",
    "open_interest", "volume",
]
VOL_COLS = ["secid", "date", "cp_flag", "volume", "delta"]


def to_secid_str(s):
    """Coerce any secid representation to a clean digit string."""
    return (pd.to_numeric(s, errors="coerce")
              .astype("Int64")
              .astype(str)
              .str.strip())


# ── TRADING-DAY CALENDAR HELPERS ──────────────────────────────────────────────

def build_trading_day_index(start_date, end_date):
    s = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    e = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    if _MCal_AVAILABLE:
        nyse     = mcal.get_calendar('NYSE')
        sched    = nyse.schedule(start_date=s, end_date=e)
        td_array = pd.DatetimeIndex(sched.index).normalize()
    else:
        td_array = pd.bdate_range(start=s, end=e).normalize()
    td_index = {ts: i for i, ts in enumerate(td_array)}
    return td_array, td_index


def _get_pos(d, td_index, td_array):
    d = pd.Timestamp(d).normalize()
    if d in td_index:
        return td_index[d]
    return int(td_array.searchsorted(d, side='left'))


def td_diff(date_series, ref_series, td_array, td_index):
    """Signed trading-day distance: date − ref_date."""
    dates = pd.to_datetime(date_series).dt.normalize()
    refs  = pd.to_datetime(ref_series).dt.normalize()
    pos_d = np.array([_get_pos(d, td_index, td_array) for d in dates])
    pos_r = np.array([_get_pos(r, td_index, td_array) for r in refs])
    return pd.Series(pos_d - pos_r, index=date_series.index, dtype='int64')


# ── LOAD EVENTS ───────────────────────────────────────────────────────────────
events = pd.read_csv(EVENTS_PATH, parse_dates=["anndats", "pends"])
events["cusip8"] = events["cusip_x"].str[:8]
print(f"Events loaded: {len(events):,}")

# ── LOAD SECID MAP ────────────────────────────────────────────────────────────
secid_map = pd.read_csv("optionm/secid_map.csv")
secid_map["secid"] = to_secid_str(secid_map["secid"])
secid_map = secid_map[["cusip8", "secid"]].drop_duplicates()
events = events.merge(secid_map, on="cusip8", how="inner")
events["secid"] = to_secid_str(events["secid"])
print(f"Events after SECID match: {len(events):,}")

# ── BUILD TRADING-DAY CALENDAR ────────────────────────────────────────────────
# FIX 3: replace the dynamic calendar-day-offset approximation
#   (_td_cal_start = events["anndats"].min() - pd.Timedelta(days=65))
# with hardcoded bounds matched to the known data range.
#
#   Left  : earliest events ~Jan 2015 → 30-TD lookback reaches ~Nov 2014
#            → start from 2014-10-01 (safe buffer)
#   Right : latest events ~Dec 2025 → +10-TD post-event reaches ~Jan 2026
#            → end at 2026-03-31 (safe buffer)
#
# No calendar-day approximation anywhere in the calendar construction.
td_array, td_index = build_trading_day_index('2014-10-01', '2026-03-31')

print(f"\nTrading-day calendar : {td_array[0].date()} → {td_array[-1].date()}")
print(f"  Total trading days : {len(td_array):,}")
print(f"  Calendar source    : "
      f"{'NYSE (pandas_market_calendars)' if _MCal_AVAILABLE else 'Mon–Fri (fallback)'}")

# ── BUILD EVENT PANEL ─────────────────────────────────────────────────────────
# One row per (event, trading-day offset) from −10 to +10 NYSE trading days.
# td_array and _get_pos are module-level globals used throughout the script.
window_rows = []
for _, row in events.iterrows():
    ev_ts  = pd.Timestamp(row["anndats"]).normalize()
    ev_pos = _get_pos(ev_ts, td_index, td_array)
    for offset in range(-10, 11):
        pos     = int(np.clip(ev_pos + offset, 0, len(td_array) - 1))
        td_date = td_array[pos]
        window_rows.append({
            "secid"      : row["secid"],
            "event_date" : ev_ts,
            "date"       : td_date,
            "ticker"     : row["ticker"],
            "pends"      : row["pends"],
        })

event_panel = pd.DataFrame(window_rows)
event_panel["date"]       = pd.to_datetime(event_panel["date"])
event_panel["event_date"] = pd.to_datetime(event_panel["event_date"])
event_panel["pends"]      = pd.to_datetime(event_panel["pends"])
event_panel["secid"]      = event_panel["secid"].astype(str)
print(f"\nEvent panel rows: {len(event_panel):,}")
print(f"Event panel secid sample : {event_panel['secid'].drop_duplicates().head(5).tolist()}")
print(f"Event panel date range   : {event_panel['date'].min().date()} → "
      f"{event_panel['date'].max().date()}")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def attach_nearest_expiry(sub):
    """
    Keep only rows whose DTE equals the minimum DTE for that
    (secid, date, event_date, pends) group.

    FIX 4: the old version grouped only by ["secid", "date"], which is
    incorrect when two events for the same firm have overlapping ±10-TD
    windows (the same calendar date then belongs to two different events).
    Adding event_date and pends to the groupby ensures each event gets its
    own nearest expiry selection.  The nearest_dte helper column is also
    dropped after use to avoid polluting downstream DataFrames.
    """
    grp_cols = ["secid", "date", "event_date", "pends"]
    nearest  = (sub.groupby(grp_cols)["dte"]
                   .min()
                   .reset_index()
                   .rename(columns={"dte": "nearest_dte"}))
    sub = sub.merge(nearest, on=grp_cols)
    return sub[sub["dte"] == sub["nearest_dte"]].drop(columns="nearest_dte").copy()


def select_otm_pair(group):
    puts  = group[group["cp_flag"] == "P"].copy()
    calls = group[group["cp_flag"] == "C"].copy()
    if puts.empty or calls.empty:
        return None
    puts["delta_dist"]  = (puts["delta"].abs()  - 0.25).abs()
    calls["delta_dist"] = (calls["delta"].abs() - 0.25).abs()
    puts  = puts.sort_values(["delta_dist", "open_interest"], ascending=[True, False])
    calls = calls.sort_values(["delta_dist", "open_interest"], ascending=[True, False])
    return pd.Series({
        "iv_put"      : puts.iloc[0]["impl_volatility"],
        "iv_call"     : calls.iloc[0]["impl_volatility"],
        "oi_put"      : puts.iloc[0]["open_interest"],
        "oi_call"     : calls.iloc[0]["open_interest"],
        "delta_put"   : puts.iloc[0]["delta"],
        "delta_call"  : calls.iloc[0]["delta"],
        "strike_put"  : puts.iloc[0]["strike_price"],
        "strike_call" : calls.iloc[0]["strike_price"],
        "exdate"      : puts.iloc[0]["exdate"],
        "dte"         : puts.iloc[0]["dte"],
    })


def select_atm_pair(group):
    puts  = group[group["cp_flag"] == "P"].copy()
    calls = group[group["cp_flag"] == "C"].copy()
    if puts.empty or calls.empty:
        return None
    return pd.Series({
        "iv_put_atm"  : puts["impl_volatility"].mean(),
        "iv_call_atm" : calls["impl_volatility"].mean(),
    })


def safe_apply(df, groupby_cols, func):
    result = (df.groupby(groupby_cols)
                .apply(func, include_groups=False)
                .dropna()
                .reset_index())
    return result


# ── PROCESS ONE YEAR ──────────────────────────────────────────────────────────
def process_year(yr, event_panel):
    """
    Returns
    -------
    result   : DataFrame of per-(secid, date, event) skew variables, or None.
    vol_evts : DataFrame [secid, event_date, LN_PC_VLM] — ONE ROW PER EVENT.
               Returning event-level aggregates instead of daily rows keeps the
               yearly_vol_evts accumulator list tiny across 11 years.
    """
    # ── Slice event panel to dates falling in this calendar year ──────────────
    ep_yr = (
        event_panel[event_panel["date"].dt.year == yr]
        [["secid", "date", "event_date", "ticker", "pends"]]
        .sort_values("event_date")
        # Keep only one event_date per (secid, date) to prevent a many-to-many
        # join when overlapping event windows share a trading date.
        .drop_duplicates(subset=["secid", "date"])
        .copy()
    )

    if ep_yr.empty:
        print(f"\n[{yr}] No event-panel rows for this year — skipping")
        return None, None

    path = os.path.join(PARQUET_DIR, f"opts_{yr}.parquet")
    if not os.path.exists(path):
        print(f"\n[{yr}] Parquet file not found — skipping")
        return None, None

    # ── FIX 5: load parquet ONCE with column selection ────────────────────────
    # OPT_COLS excludes optionid, bid/ask sizes, etc. — roughly halves RAM.
    # The volume slice is then extracted from this already-loaded DataFrame,
    # eliminating the second pd.read_parquet call that existed in V10 and
    # was the primary cause of the out-of-memory crash.
    print(f"\n[{yr}] Loading parquet...", end=" ")
    df = pd.read_parquet(path, columns=OPT_COLS)
    print(f"{len(df):,} rows")

    # ── Normalise dtypes ──────────────────────────────────────────────────────
    df["secid"]           = to_secid_str(df["secid"])
    df["date"]            = pd.to_datetime(df["date"])
    df["exdate"]          = pd.to_datetime(df["exdate"])
    df["impl_volatility"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
    df["delta"]           = pd.to_numeric(df["delta"],           errors="coerce")
    df["best_bid"]        = pd.to_numeric(df["best_bid"],        errors="coerce")
    df["best_offer"]      = pd.to_numeric(df["best_offer"],      errors="coerce")
    df["open_interest"]   = pd.to_numeric(df["open_interest"],   errors="coerce")
    df["volume"]          = pd.to_numeric(df["volume"],          errors="coerce").fillna(0)
    df["strike_price"]    = pd.to_numeric(df["strike_price"],    errors="coerce")
    df["abs_delta"]       = df["delta"].abs()

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"  [{yr}] Parquet secid sample : {df['secid'].dropna().unique()[:4].tolist()}")
    print(f"  [{yr}] Panel  secid sample  : {ep_yr['secid'].unique()[:4].tolist()}")
    print(f"  [{yr}] Panel rows for year  : {len(ep_yr):,}")

    # ── 30-TRADING-DAY lookback volume ────────────────────────────────────────
    # FIX 6 (memory): extract the OTM volume slice from the already-loaded df
    # rather than calling pd.read_parquet a second time.  This alone eliminates
    # the peak-RAM doubling that caused the OOM crash in V10.
    df_vol = df.loc[
        (df["abs_delta"] >= 0.10) & (df["abs_delta"] <= 0.40),
        ["secid", "date", "cp_flag", "volume"]
    ].copy()

    # Prior-year data is still loaded separately (only VOL_COLS, minimal RAM)
    # for events in Jan–Feb whose 30-TD window crosses the year boundary.
    prior_path = os.path.join(PARQUET_DIR, f"opts_{yr - 1}.parquet")
    if os.path.exists(prior_path):
        prior = pd.read_parquet(prior_path, columns=VOL_COLS)
        prior["secid"]     = to_secid_str(prior["secid"])
        prior["date"]      = pd.to_datetime(prior["date"])
        prior["volume"]    = pd.to_numeric(prior["volume"], errors="coerce").fillna(0)
        prior["abs_delta"] = prior["delta"].abs()
        prior = prior.loc[
            (prior["abs_delta"] >= 0.10) & (prior["abs_delta"] <= 0.40),
            ["secid", "date", "cp_flag", "volume"]
        ].copy()
        df_vol = pd.concat([prior, df_vol], ignore_index=True)
        del prior
        gc.collect()
        print(f"  [{yr}] Prior-year volume appended from opts_{yr - 1}.parquet")

    # Pre-aggregate to (secid, date, cp_flag) daily sums — makes the per-event
    # loop below much faster since it operates on a much smaller table.
    daily_vol = (df_vol.groupby(["secid", "date", "cp_flag"])["volume"]
                       .sum()
                       .reset_index())
    del df_vol
    gc.collect()

    # For each event in this year, sum OTM volume over the 30 NYSE trading days
    # immediately before the event.  Returns one compact row per event, so
    # yearly_vol_evts accumulates very little data across all 11 years.
    ep_lookback = (
        event_panel.loc[
            event_panel["event_date"].dt.year == yr,
            ["secid", "event_date"]
        ]
        .drop_duplicates()
        .copy()
    )

    vol_ev_rows = []
    for _, row in ep_lookback.iterrows():
        secid    = row["secid"]
        ev_ts    = pd.Timestamp(row["event_date"]).normalize()
        ev_pos   = _get_pos(ev_ts, td_index, td_array)
        start_p  = max(0, ev_pos - 30)
        td_start = td_array[start_p]

        mask = (
            (daily_vol["secid"] == secid) &
            (daily_vol["date"]  >= td_start) &
            (daily_vol["date"]  <  ev_ts)       # event day excluded
        )
        sub = daily_vol[mask]
        if sub.empty:
            continue

        put_v  = float(sub.loc[sub["cp_flag"] == "P", "volume"].sum())
        call_v = float(sub.loc[sub["cp_flag"] == "C", "volume"].sum())
        if put_v > 0 and call_v > 0:
            vol_ev_rows.append({
                "secid"      : secid,
                "event_date" : ev_ts,
                "LN_PC_VLM"  : np.log(put_v / call_v),
            })

    vol_evts = pd.DataFrame(vol_ev_rows) if vol_ev_rows else None
    del daily_vol, ep_lookback
    gc.collect()

    # ── Event-window filter ───────────────────────────────────────────────────
    df = df.merge(ep_yr, on=["secid", "date"], how="inner")
    print(f"  [{yr}] After event-panel filter : {len(df):,}")

    if df.empty:
        print(f"  [{yr}] ⚠ Zero rows — verify secid/date overlap above")
        return None, vol_evts

    # ── Hard filters ──────────────────────────────────────────────────────────
    df["midpoint"] = (df["best_bid"] + df["best_offer"]) / 2
    df["spread"]   = ((df["best_offer"] - df["best_bid"])
                      / df["midpoint"].replace(0, np.nan))
    df["dte"]      = (df["exdate"] - df["date"]).dt.days

    df = df[
        (df["best_bid"] > 0) &
        (df["spread"]   < 0.50) &
        (df["dte"]      >= 10) &
        (df["dte"]      <= 60)
    ].dropna(subset=["impl_volatility", "delta"]).copy()
    print(f"  [{yr}] After hard filters        : {len(df):,}")

    if df.empty:
        return None, vol_evts

    # ── FIX 7: exdate ≥ event_date ────────────────────────────────────────────
    # An option that expires BEFORE the announcement cannot embed any
    # information about the event outcome.  This filter ensures every contract
    # selected genuinely straddles the announcement date.
    df = df[df["exdate"] >= df["event_date"]].copy()
    print(f"  [{yr}] After exdate≥event_date   : {len(df):,}")

    if df.empty:
        return None, vol_evts

    # ── FIX 8 (memory): drop columns not needed for groupby-apply ─────────────
    # Freeing these columns before the two heavy safe_apply() calls reduces
    # peak RAM during the most expensive part of process_year by ~20–30%.
    df.drop(
        columns=["best_bid", "best_offer", "midpoint", "spread", "volume"],
        inplace=True, errors="ignore"
    )
    gc.collect()

    # ── Split OTM / ATM ───────────────────────────────────────────────────────
    otm_df = df[(df["abs_delta"] >= 0.10) & (df["abs_delta"] <= 0.40)].copy()
    atm_df = df[(df["abs_delta"] >= 0.40) & (df["abs_delta"] <= 0.60)].copy()
    del df
    gc.collect()

    if otm_df.empty or atm_df.empty:
        print(f"  [{yr}] OTM or ATM slice empty after delta filter")
        return None, vol_evts

    # ── Nearest expiry ────────────────────────────────────────────────────────
    otm_df = attach_nearest_expiry(otm_df)
    atm_df = attach_nearest_expiry(atm_df)

    # ── OTM pair selection ────────────────────────────────────────────────────
    otm_pairs = safe_apply(
        otm_df,
        ["secid", "date", "event_date", "ticker", "pends"],
        select_otm_pair
    )
    del otm_df

    if otm_pairs.empty or "iv_put" not in otm_pairs.columns:
        print(f"  [{yr}] OTM apply produced no valid pairs")
        return None, vol_evts

    otm_pairs["exdate"] = pd.to_datetime(otm_pairs["exdate"])

    # ── ATM pair selection ────────────────────────────────────────────────────
    atm_pairs = safe_apply(
        atm_df,
        ["secid", "date", "event_date", "ticker", "pends"],
        select_atm_pair
    )
    del atm_df
    gc.collect()

    if atm_pairs.empty or "iv_put_atm" not in atm_pairs.columns:
        print(f"  [{yr}] ATM apply produced no valid pairs")
        return None, vol_evts

    # ── Merge OTM + ATM ───────────────────────────────────────────────────────
    merge_keys = ["secid", "date", "event_date", "ticker", "pends"]
    result = otm_pairs.merge(atm_pairs, on=merge_keys, how="inner")
    del otm_pairs, atm_pairs
    gc.collect()

    # ── Guard: zero/negative IV or OI before logs ────────────────────────────
    result = result[
        (result["iv_put"]      > 0) &
        (result["iv_call"]     > 0) &
        (result["iv_put_atm"]  > 0) &
        (result["iv_call_atm"] > 0) &
        (result["oi_put"]      > 0) &
        (result["oi_call"]     > 0)
    ].copy()

    if result.empty:
        print(f"  [{yr}] All rows filtered by zero-vol/OI guard")
        return None, vol_evts

    # ── Construct log variables ───────────────────────────────────────────────
    result["LN_IMPSKEW"] = np.log(result["iv_put"]  / result["iv_call"])
    result["LN_IMPVOL"]  = np.log((result["iv_put_atm"] + result["iv_call_atm"]) / 2)
    result["LN_PC_OI"]   = np.log(result["oi_put"]  / result["oi_call"])

    log_cols = ["LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=log_cols)
    result = result.drop(columns=["iv_put_atm", "iv_call_atm"])

    # ── Signed trading-day distance from the announcement ────────────────────
    result["rel_day"] = td_diff(
        result["date"], result["event_date"], td_array, td_index
    ).values

    print(f"  [{yr}] Final output rows          : {len(result):,}")
    print(f"  [{yr}] rel_day range              : "
          f"{result['rel_day'].min()} → {result['rel_day'].max()}")
    return result, vol_evts


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
yearly_results  = []
yearly_vol_evts = []

for yr in YEARS:
    result_yr, vol_evts_yr = process_year(yr, event_panel)
    if result_yr  is not None and not result_yr.empty:
        yearly_results.append(result_yr)
    if vol_evts_yr is not None and not vol_evts_yr.empty:
        yearly_vol_evts.append(vol_evts_yr)
    gc.collect()

if not yearly_results:
    print("\n⚠  No yearly results produced. Inspect the secid/date diagnostics above.")
    raise SystemExit(1)

# ── COMBINE SKEW RESULTS ──────────────────────────────────────────────────────
print("\nCombining yearly skew results...")
combined = pd.concat(yearly_results, ignore_index=True)
del yearly_results
print(f"Combined skew rows: {len(combined):,}")

# ── ATTACH VOLUME RATIO ───────────────────────────────────────────────────────
# FIX 9: replace the old vol_pivot + event re-iteration section (which held
# all daily volume rows in RAM simultaneously) with a simple concat + merge,
# since process_year now returns compact event-level aggregates.
print("Attaching LN_PC_VLM...")

if yearly_vol_evts:
    vol_result = pd.concat(yearly_vol_evts, ignore_index=True)
    del yearly_vol_evts
    gc.collect()
    # A (secid, event_date) pair can appear in two consecutive years' outputs
    # when the event falls in Jan–Feb and the prior-year lookback pass also
    # processes it.  Keep only the first occurrence.
    vol_result = vol_result.drop_duplicates(subset=["secid", "event_date"], keep="first")
    print(f"Volume ratio rows: {len(vol_result):,}")
else:
    vol_result = pd.DataFrame(columns=["secid", "event_date", "LN_PC_VLM"])
    print("No volume data collected")

final = combined.merge(vol_result, on=["secid", "event_date"], how="left")
del combined, vol_result
gc.collect()

# ── WINSORIZE AT 1%–99% ───────────────────────────────────────────────────────
# FIX 10: guard against a missing column (e.g. LN_PC_VLM when no volume data
# was collected) rather than crashing with a KeyError.
print("\nWinsorizing at 1%–99%...")
for var in ["LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI", "LN_PC_VLM"]:
    if var not in final.columns:
        print(f"  {var}: column missing — skipping winsorize")
        continue
    lo = final[var].quantile(0.01)
    hi = final[var].quantile(0.99)
    final[var] = final[var].clip(lower=lo, upper=hi)
    print(f"  {var}: [{lo:.4f}, {hi:.4f}]")

# ── SAVE ──────────────────────────────────────────────────────────────────────
final.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Saved {len(final):,} rows → {OUTPUT_PATH}")
print(final[[
    "secid", "ticker", "pends", "date", "event_date", "rel_day",
    "exdate", "dte",
    "iv_put",  "delta_put",  "oi_put",  "strike_put",
    "iv_call", "delta_call", "oi_call", "strike_call",
    "LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI", "LN_PC_VLM",
]].head(10))