import pandas as pd
import numpy as np
import wrds
import gc
import os
import warnings
warnings.filterwarnings('ignore')

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

os.makedirs('optionm',     exist_ok=True)
os.makedirs('optionm2025', exist_ok=True)

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
    if d in td_index:
        return td_index[d]
    return int(td_array.searchsorted(d, side='left'))


def td_offset(date_series, n, td_array, td_index):
    dates   = pd.to_datetime(date_series).dt.normalize()
    pos     = np.array([_get_pos(d, td_index, td_array) for d in dates])
    new_pos = np.clip(pos + n, 0, len(td_array) - 1).astype(int)
    return pd.Series(
        td_array[new_pos].values,
        index=date_series.index,
        dtype='datetime64[ns]'
    )


def td_diff(date_series, ref_series, td_array, td_index):
    dates = pd.to_datetime(date_series).dt.normalize()
    refs  = pd.to_datetime(ref_series).dt.normalize()
    pos_d = np.array([_get_pos(d, td_index, td_array) for d in dates])
    pos_r = np.array([_get_pos(r, td_index, td_array) for r in refs])
    return pd.Series(pos_d - pos_r, index=date_series.index, dtype='int64')


# ── TRADING-DAY CALENDAR ──────────────────────────────────────────────────────
td_array, td_index = build_trading_day_index('2014-10-01', '2026-03-31')

print(f"Trading-day calendar : {td_array[0].date()} → {td_array[-1].date()}")
print(f"  Total trading days : {len(td_array):,}")
print(f"  Calendar source    : "
      f"{'NYSE (pandas_market_calendars)' if _MCal_AVAILABLE else 'Mon–Fri (fallback)'}")

# ── COLUMN SELECTION — limits RAM per parquet read ────────────────────────────
OPT_COLS = [
    "secid", "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "impl_volatility", "delta",
    "open_interest", "volume",
]
VOL_COLS = ["secid", "date", "cp_flag", "volume", "delta"]

# ============================================================
# PHASE 2: OptionMetrics — Extract Daily Skew
# ============================================================

events = pd.read_csv('ibes2025/ibes_stage1_events.csv')
events.columns = events.columns.str.lower().str.strip()
events['event_date'] = pd.to_datetime(events['event_date'])
events['pends']      = pd.to_datetime(events['pends'])

events['cusip']  = events['cusip_x'].fillna(events.get('cusip_y', events['cusip_x']))
events['cusip8'] = events['cusip'].astype(str).str[:8]

print(f"\nStage 1 events loaded: {len(events):,}")
print(f"Null cusip8: {events['cusip8'].isna().sum()}")

db = wrds.Connection(wrds_username='rh3245')

# ============================================================
# STEP 1: CUSIP → SECID mapping
# ============================================================
cusip_list = events['cusip8'].unique().tolist()

secid_map = db.get_table(
    library='optionm_all',
    table='securd',
    columns=['secid', 'cusip']
)
secid_map.columns = secid_map.columns.str.lower()
secid_map = secid_map[secid_map['cusip'].isin(cusip_list)]
secid_map = secid_map.drop_duplicates(subset='cusip', keep='last')

print(f"\nCUSIPs in events         : {len(cusip_list):,}")
print(f"CUSIPs matched to SECID  : {len(secid_map):,}")

events = events.merge(secid_map, left_on='cusip8', right_on='cusip',
                      how='left', suffixes=('', '_om'))
events['secid'] = pd.to_numeric(events['secid'], errors='coerce')
events = events[events['secid'].notna()].copy()
events['secid'] = events['secid'].astype(int)
print(f"Events matched to SECID  : {len(events):,}")

# ============================================================
# STEP 2: Build per-firm date windows
# ============================================================
events['pull_start'] = td_offset(events['event_date'], -35, td_array, td_index)
events['pull_end']   = td_offset(events['event_date'], +12, td_array, td_index)

secid_windows = (
    events.groupby('secid')
          .agg(firm_start=('pull_start', 'min'),
               firm_end=('pull_end',   'max'))
          .reset_index()
)

overall_start = secid_windows['firm_start'].min()
overall_end   = secid_windows['firm_end'].max()
years_needed  = list(range(overall_start.year, overall_end.year + 1))

available_tables = db.list_tables(library='optionm_all')
year_tables = [f'opprcd{y}' for y in years_needed
               if f'opprcd{y}' in available_tables]

print(f"\nDate range       : {overall_start.date()} → {overall_end.date()}")
print(f"Year tables found: {year_tables}")
print(f"Unique SECIDs    : {len(secid_windows):,}")

# ============================================================
# STEP 3: Pull from each year table, chunked by SECID
# ============================================================
chunk_size = 50
year_files = []

for table in year_tables:
    year = int(table.replace('opprcd', ''))
    year_start_ts = pd.Timestamp(f'{year}-01-01')
    year_end_ts   = pd.Timestamp(f'{year}-12-31')

    year_windows = secid_windows[
        (secid_windows['firm_end']   >= year_start_ts) &
        (secid_windows['firm_start'] <= year_end_ts)
    ].copy()

    if year_windows.empty:
        print(f"\nSkipping {table} — no firm windows overlap {year}")
        continue

    print(f"\nPulling {table} | {len(year_windows):,} firms with windows in {year}")
    year_chunks = []
    table_rows  = 0

    for i in range(0, len(year_windows), chunk_size):
        chunk = year_windows.iloc[i:i + chunk_size]

        firm_conditions = " OR ".join([
            f"(secid = {int(row.secid)} "
            f"AND date BETWEEN "
            f"'{max(row.firm_start, year_start_ts).date()}' "
            f"AND '{min(row.firm_end, year_end_ts).date()}')"
            for row in chunk.itertuples()
        ])

        sql = f"""
            SELECT secid, date, exdate, cp_flag, strike_price,
                   best_bid, best_offer, impl_volatility, delta,
                   open_interest, volume
            FROM optionm_all.{table}
            WHERE ({firm_conditions})
              AND best_bid        > 0
              AND best_offer      > 0
              AND impl_volatility IS NOT NULL
              AND delta           IS NOT NULL
              AND ABS(delta) BETWEEN 0.10 AND 0.60
              AND (exdate - date) BETWEEN 10 AND 60
              AND (best_offer - best_bid)
                  / NULLIF((best_bid + best_offer) / 2.0, 0) < 0.50
        """
        # FIX A: optionid removed from SELECT — was loaded but never used,
        #        wasting RAM in every chunk and in the saved parquet files.
        try:
            df = db.raw_sql(sql)
            df.columns = df.columns.str.lower()
            year_chunks.append(df)
            table_rows += len(df)
        except Exception as e:
            print(f"  Chunk {i // chunk_size + 1} failed: {e}")

    if year_chunks:
        year_df   = pd.concat(year_chunks, ignore_index=True)
        year_file = f'optionm/opts_{year}.parquet'
        year_df.to_parquet(year_file, index=False)
        year_files.append(year_file)
        print(f"  → {table_rows:,} rows  |  saved to {year_file}")
        del year_chunks, year_df
        gc.collect()
    else:
        print(f"  → No data for {year}")

# ============================================================
# STEP 4: Build event panel
#
# FIX B: replaces the old pattern of loading all year parquet
# files into a single DataFrame (pd.concat over 12 files).
# Instead, one compact row is generated per (event, offset)
# combination, and yearly processing is done one file at a time.
# This eliminates the primary OOM source in the original STEP 4.
# ============================================================
print("\nBuilding event panel...")
window_rows = []
for _, row in events.iterrows():
    ev_ts  = pd.Timestamp(row['event_date']).normalize()
    ev_pos = _get_pos(ev_ts, td_index, td_array)
    for offset in range(-10, 11):
        pos     = int(np.clip(ev_pos + offset, 0, len(td_array) - 1))
        td_date = td_array[pos]
        window_rows.append({
            'secid'      : int(row['secid']),
            'event_date' : ev_ts,
            'date'       : td_date,
            'ticker'     : row['ticker'],
            'pends'      : row['pends'],
        })

event_panel = pd.DataFrame(window_rows)
event_panel['date']       = pd.to_datetime(event_panel['date'])
event_panel['event_date'] = pd.to_datetime(event_panel['event_date'])
event_panel['pends']      = pd.to_datetime(event_panel['pends'])
event_panel['secid']      = event_panel['secid'].astype(int)
print(f"Event panel rows      : {len(event_panel):,}")
print(f"Event panel date range: {event_panel['date'].min().date()} → "
      f"{event_panel['date'].max().date()}")


# ============================================================
# STEP 5: Per-year processing helpers
# ============================================================

def attach_nearest_expiry(sub):
    """
    FIX C: groupby now includes event_date and pends so that
    two events for the same firm with overlapping ±10-TD windows
    each get their own independent nearest-expiry selection.
    The helper column nearest_dte is dropped after use.
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


def process_year(yr, event_panel, year_files_set):
    """
    Process one year's parquet file, returning:
      result   — per-(secid, date, event) skew DataFrame, or None
      vol_evts — per-(secid, event_date) LN_PC_VLM DataFrame, or None

    FIX D: LN_PC_VLM is now computed HERE and returned as a compact
    event-level aggregate rather than accumulating daily volume rows
    across all years and re-iterating in the post-loop section.
    This eliminates the second full parquet reload that existed in
    the original STEP 10, which was the second major OOM source.
    """
    path = f'optionm/opts_{yr}.parquet'
    if path not in year_files_set or not os.path.exists(path):
        print(f"\n[{yr}] Parquet not available — skipping")
        return None, None

    ep_yr = (
        event_panel[event_panel["date"].dt.year == yr]
        [["secid", "date", "event_date", "ticker", "pends"]]
        .sort_values("event_date")
        # FIX E: drop_duplicates on (secid, date) prevents a many-to-many
        # join when two events for the same firm share a window trading date.
        .drop_duplicates(subset=["secid", "date"])
        .copy()
    )

    if ep_yr.empty:
        print(f"\n[{yr}] No event-panel rows — skipping")
        return None, None

    # ── Load parquet ONCE with column selection ───────────────────────────────
    # FIX F: OPT_COLS excludes optionid and any other unused columns.
    # The volume slice is derived from this single load, eliminating the
    # second pd.read_parquet call (and the associated RAM doubling) that
    # existed in original STEP 10.
    print(f"\n[{yr}] Loading parquet...", end=" ")
    df = pd.read_parquet(path, columns=OPT_COLS)
    print(f"{len(df):,} rows")

    df["secid"]           = pd.to_numeric(df["secid"],           errors="coerce").astype(int)
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

    # ── 30-TD volume slice extracted from the already-loaded df ──────────────
    df_vol = df.loc[
        (df["abs_delta"] >= 0.10) & (df["abs_delta"] <= 0.40),
        ["secid", "date", "cp_flag", "volume"]
    ].copy()

    # Prior year: loaded separately but only VOL_COLS (5 columns) for events
    # whose 30-TD lookback crosses the year boundary (Jan–Feb events).
    prior_path = f'optionm/opts_{yr - 1}.parquet'
    if prior_path in year_files_set and os.path.exists(prior_path):
        prior = pd.read_parquet(prior_path, columns=VOL_COLS)
        prior["secid"]     = pd.to_numeric(prior["secid"], errors="coerce").astype(int)
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

    # Pre-aggregate to daily (secid, date, cp_flag) sums — the per-event loop
    # below then operates on a much smaller table.
    daily_vol = (df_vol.groupby(["secid", "date", "cp_flag"])["volume"]
                       .sum().reset_index())
    del df_vol
    gc.collect()

    # Compute LN_PC_VLM per event — one compact row per event.
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
        secid    = int(row["secid"])
        ev_ts    = pd.Timestamp(row["event_date"]).normalize()
        ev_pos   = _get_pos(ev_ts, td_index, td_array)
        start_p  = max(0, ev_pos - 30)
        td_start = td_array[start_p]

        mask = (
            (daily_vol["secid"] == secid) &
            (daily_vol["date"]  >= td_start) &
            (daily_vol["date"]  <  ev_ts)
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

    # ── Event-window merge ────────────────────────────────────────────────────
    # FIX G: merging on ["secid", "date"] rather than on secid alone avoids
    # the many-to-many explosion that occurred in original STEP 5 when one
    # firm had many events and every option row was replicated per event before
    # the window filter cut it back down.
    df = df.merge(ep_yr, on=["secid", "date"], how="inner")
    print(f"  [{yr}] After event-panel filter : {len(df):,}")
    if df.empty:
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

    # FIX H: keep only contracts that expire ON OR AFTER the event date.
    # A contract expiring before the announcement cannot embed information
    # about the event outcome.
    df = df[df["exdate"] >= df["event_date"]].copy()
    print(f"  [{yr}] After exdate≥event_date   : {len(df):,}")
    if df.empty:
        return None, vol_evts

    # FIX I: drop heavy columns before the two groupby-apply calls to reduce
    # their peak RAM by ~20–30 %.
    df.drop(
        columns=["best_bid", "best_offer", "midpoint", "spread", "volume"],
        inplace=True, errors="ignore"
    )
    gc.collect()

    # ── OTM / ATM split ───────────────────────────────────────────────────────
    otm_df = df[(df["abs_delta"] >= 0.10) & (df["abs_delta"] <= 0.40)].copy()
    atm_df = df[(df["abs_delta"] >= 0.40) & (df["abs_delta"] <= 0.60)].copy()
    del df
    gc.collect()

    if otm_df.empty or atm_df.empty:
        print(f"  [{yr}] OTM or ATM slice empty after delta filter")
        return None, vol_evts

    otm_df = attach_nearest_expiry(otm_df)
    atm_df = attach_nearest_expiry(atm_df)

    merge_keys = ["secid", "date", "event_date", "ticker", "pends"]

    otm_pairs = safe_apply(otm_df, merge_keys, select_otm_pair)
    del otm_df
    if otm_pairs.empty or "iv_put" not in otm_pairs.columns:
        print(f"  [{yr}] OTM apply produced no valid pairs")
        return None, vol_evts
    otm_pairs["exdate"] = pd.to_datetime(otm_pairs["exdate"])

    atm_pairs = safe_apply(atm_df, merge_keys, select_atm_pair)
    del atm_df
    gc.collect()
    if atm_pairs.empty or "iv_put_atm" not in atm_pairs.columns:
        print(f"  [{yr}] ATM apply produced no valid pairs")
        return None, vol_evts

    result = otm_pairs.merge(atm_pairs, on=merge_keys, how="inner")
    del otm_pairs, atm_pairs
    gc.collect()

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

    result["LN_IMPSKEW"] = np.log(result["iv_put"]  / result["iv_call"])
    result["LN_IMPVOL"]  = np.log((result["iv_put_atm"] + result["iv_call_atm"]) / 2)
    result["LN_PC_OI"]   = np.log(result["oi_put"]  / result["oi_call"])

    log_cols = ["LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=log_cols)
    result = result.drop(columns=["iv_put_atm", "iv_call_atm"])

    result["rel_day"] = td_diff(
        result["date"], result["event_date"], td_array, td_index
    ).values

    print(f"  [{yr}] Final output rows          : {len(result):,}")
    print(f"  [{yr}] rel_day range              : "
          f"{result['rel_day'].min()} → {result['rel_day'].max()}")
    return result, vol_evts


# ============================================================
# STEP 6: Main processing loop (year by year)
# ============================================================
year_files_set = set(year_files)
all_years = sorted(
    int(f.split('_')[1].replace('.parquet', ''))
    for f in year_files
)

yearly_results  = []
yearly_vol_evts = []

for yr in all_years:
    result_yr, vol_evts_yr = process_year(yr, event_panel, year_files_set)
    if result_yr  is not None and not result_yr.empty:
        yearly_results.append(result_yr)
    if vol_evts_yr is not None and not vol_evts_yr.empty:
        yearly_vol_evts.append(vol_evts_yr)
    gc.collect()

if not yearly_results:
    print("\n⚠  No yearly results produced. Inspect the diagnostics above.")
    db.close()
    raise SystemExit(1)

# ============================================================
# STEP 7: Combine results and attach LN_PC_VLM
#
# FIX J: replaces the original vol_pivot construction (which
# held all daily volume rows in RAM at once) with a simple
# concat + merge of compact event-level aggregates already
# computed inside process_year.
# ============================================================
print("\nCombining yearly results...")
combined = pd.concat(yearly_results, ignore_index=True)
del yearly_results
print(f"Combined skew rows: {len(combined):,}")

print("Attaching LN_PC_VLM...")
if yearly_vol_evts:
    vol_result = pd.concat(yearly_vol_evts, ignore_index=True)
    del yearly_vol_evts
    gc.collect()
    # A Jan–Feb event can appear in consecutive years' outputs because the
    # prior-year lookback pass also processes it — keep first occurrence.
    vol_result = vol_result.drop_duplicates(
        subset=["secid", "event_date"], keep="first"
    )
    print(f"Volume ratio rows: {len(vol_result):,}")
else:
    vol_result = pd.DataFrame(columns=["secid", "event_date", "LN_PC_VLM"])
    print("No volume data collected")

final = combined.merge(vol_result, on=["secid", "event_date"], how="left")
del combined, vol_result
gc.collect()

# ============================================================
# STEP 8: Winsorise at 1%–99%
# FIX K: guard against missing column rather than KeyError.
# ============================================================
print("\nWinsorizing at 1%–99%...")
for var in ["LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI", "LN_PC_VLM"]:
    if var not in final.columns:
        print(f"  {var}: column missing — skipping")
        continue
    lo = final[var].quantile(0.01)
    hi = final[var].quantile(0.99)
    final[var] = final[var].clip(lower=lo, upper=hi)
    print(f"  {var}: [{lo:.4f}, {hi:.4f}]")

# ============================================================
# STEP 9: Diagnostics + Save
# ============================================================
print("\n=== VARIABLE SUMMARIES ===")
for var in ["LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI", "LN_PC_VLM"]:
    if var in final.columns:
        print(f"\n{var}:\n{final[var].describe()}")

print(f"\nTotal rows (firm × date) : {len(final):,}")
print(f"Unique firms             : {final['ticker'].nunique():,}")
print(f"Unique events            : {final.groupby(['ticker','pends']).ngroups:,}")

print(f"\nrel_day range            : {final['rel_day'].min()} → {final['rel_day'].max()}")
print(f"rel_day distribution (near-uniform = trading-day window ✅):")
print(final['rel_day'].value_counts().sort_index().to_string())

for var in ["LN_IMPSKEW", "LN_IMPVOL", "LN_PC_OI", "LN_PC_VLM"]:
    if var in final.columns:
        print(f"Null {var:<15}: {final[var].isna().sum():,}")

final.to_csv('optionm2025/optionm_stage2_skew.csv', index=False)
print("\nSaved: optionm2025/optionm_stage2_skew.csv")

db.close()