import wrds
import pandas as pd
import os


def filter_ibes_to_sp500(
    username='rh3245',
    summary_path='ibes2025/ibessummary.csv',
    actuals_path='ibes2025/ibesactuals.csv',
    output_dir='ibes2025/ibes_filtered'
):
    """
    Filters pre-downloaded IBES summary and actuals CSV files to S&P 500
    constituents only, using time-matched S&P 500 membership.

    Requires a WRDS connection only to download two small reference tables:
      - wrdsapps_link_crsp_ibes.ibcrsphist  (IBES ticker → PERMNO link)
      - crsp.msp500list                      (S&P 500 membership windows)

    S&P 500 membership is TIME-MATCHED:
      - Summary  : firm must have been in S&P 500 on the STATPERS date.
      - Actuals  : firm must have been in S&P 500 on the ANNDATS date.
    """

    if output_dir and not output_dir.endswith('/'):
        output_dir = output_dir + '/'
    os.makedirs(output_dir, exist_ok=True)

    SENTINEL = pd.Timestamp('2099-12-31')
    BATCH_CUTOFF_THRESHOLD = 0.05   # if >5% of rows share one date → treat as cutoff

    def fix_batch_cutoff(series, label):
        """
        Null-fills NaT values, then detects whether WRDS has stamped a hard
        batch-cutoff date on currently-active rows instead of leaving them NULL.
        If more than BATCH_CUTOFF_THRESHOLD of all rows share the same date,
        that date is extended to SENTINEL (2099-12-31).
        Returns the corrected series.
        """
        # Step 1: standard null-fill
        n_null = series.isna().sum()
        series = series.fillna(SENTINEL)

        # Step 2: batch-cutoff detection on remaining explicit dates
        non_sentinel = series[series < pd.Timestamp('2099-01-01')]
        if len(non_sentinel) == 0:
            print(f"  [{label}] No explicit dates remaining after null-fill.")
            return series

        top_date       = non_sentinel.value_counts().idxmax()
        top_count      = non_sentinel.value_counts().iloc[0]
        top_share      = top_count / len(series)

        print(f"  [{label}] null count (before fill)   : {n_null:,}")
        print(f"  [{label}] most common explicit date  : {top_date.date()}"
              f"  ({top_count:,} rows, {top_share:.1%} of table)")

        if top_share > BATCH_CUTOFF_THRESHOLD:
            print(f"  [{label}] *** Batch-cutoff detected — extending {top_date.date()}"
                  f" → 2099-12-31 ***")
            series = series.replace(top_date, SENTINEL)
        else:
            print(f"  [{label}] No batch-cutoff pattern (share < {BATCH_CUTOFF_THRESHOLD:.0%}).")

        print(f"  [{label}] max after fix               : {series.max()}")
        return series

    # ── STEP 1: Download reference tables from WRDS ───────────────────────────
    print("=" * 65)
    print("STEP 1 — Download reference tables from WRDS")
    print("=" * 65)

    db = wrds.Connection(wrds_username=username)
    print("Connection successful.\n")

    checks = {
        'wrdsapps_link_crsp_ibes.ibcrsphist': 'SELECT COUNT(*) FROM wrdsapps_link_crsp_ibes.ibcrsphist',
        'crsp.msp500list'                   : 'SELECT COUNT(*) FROM crsp.msp500list',
    }
    all_ok = True
    for label, chk_query in checks.items():
        try:
            db.raw_sql(chk_query)
            print(f"  {label:50s} : OK")
        except Exception as e:
            print(f"  {label:50s} : FAILED ({e})")
            all_ok = False

    if not all_ok:
        print("\nOne or more required tables could not be accessed. Aborting.")
        db.close()
        return None, None

    print()

    # IBES ticker → PERMNO link table
    print("  Downloading IBES-CRSP link table...")
    link = db.raw_sql("""
        SELECT ticker,
               permno,
               sdate,
               edate
        FROM wrdsapps_link_crsp_ibes.ibcrsphist
    """, date_cols=['sdate', 'edate'])

    print(f"\n  --- ibcrsphist edate diagnostics ---")
    link['edate'] = fix_batch_cutoff(link['edate'], 'ibcrsphist.edate')
    print(f"\n  Link table rows : {len(link):,}  |  "
          f"Unique tickers  : {link['ticker'].nunique():,}")

    # S&P 500 constituent membership windows
    print("\n  Downloading S&P 500 membership list...")
    sp500 = db.raw_sql("""
        SELECT permno,
               start,
               ending
        FROM crsp.msp500list
    """, date_cols=['start', 'ending'])

    print(f"\n  --- msp500list ending diagnostics ---")
    sp500['ending'] = fix_batch_cutoff(sp500['ending'], 'msp500list.ending')
    print(f"\n  S&P 500 rows    : {len(sp500):,}  |  "
          f"Unique permnos  : {sp500['permno'].nunique():,}")

    db.close()
    print("\n  WRDS connection closed.")

    # ── STEP 2: Build ticker-level S&P 500 membership windows ─────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Build time-matched ticker-level S&P 500 windows")
    print("=" * 65)

    membership = link.merge(sp500, on='permno', how='inner')
    membership['valid_start'] = membership[['sdate', 'start']].max(axis=1)
    membership['valid_end']   = membership[['edate', 'ending']].min(axis=1)
    membership = membership[
        membership['valid_start'] <= membership['valid_end']
    ][['ticker', 'valid_start', 'valid_end']].drop_duplicates().reset_index(drop=True)

    print(f"  Valid (ticker, window) pairs : {len(membership):,}")
    print(f"  Unique tickers in S&P 500    : {membership['ticker'].nunique():,}")
    print(f"  valid_end max                : {membership['valid_end'].max()}")

    # ── STEP 3: Filter IBES Summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Filter IBES Summary  (statpers time-matched)")
    print("=" * 65)

    print(f"  Loading '{summary_path}'...")
    summary = pd.read_csv(summary_path, parse_dates=['statpers', 'fpedats'])
    print(f"  Rows before filter : {len(summary):,}")
    print(f"  Unique tickers     : {summary['ticker'].nunique():,}")
    print(f"  statpers range     : {summary['statpers'].min().date()} → "
          f"{summary['statpers'].max().date()}")

    sum_f = summary.merge(membership, on='ticker', how='inner')
    sum_f = sum_f[
        (sum_f['statpers'] >= sum_f['valid_start']) &
        (sum_f['statpers'] <= sum_f['valid_end'])
    ].drop(columns=['valid_start', 'valid_end']).drop_duplicates().reset_index(drop=True)

    print(f"  Rows after filter  : {len(sum_f):,}")
    print(f"  Unique tickers     : {sum_f['ticker'].nunique():,}")
    print(f"  Date range         : {sum_f['statpers'].min().date()} → "
          f"{sum_f['statpers'].max().date()}")

    summary_out = f"{output_dir}ibes_summary_sp500_filtered.csv"
    sum_f.to_csv(summary_out, index=False)
    print(f"  Saved → '{summary_out}'")

    # ── STEP 4: Filter IBES Actuals ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Filter IBES Actuals  (anndats time-matched)")
    print("=" * 65)

    print(f"  Loading '{actuals_path}'...")
    actuals = pd.read_csv(actuals_path, parse_dates=['pends', 'anndats', 'actdats'])
    print(f"  Rows before filter : {len(actuals):,}")
    print(f"  Unique tickers     : {actuals['ticker'].nunique():,}")
    print(f"  anndats range      : {actuals['anndats'].min().date()} → "
          f"{actuals['anndats'].max().date()}")

    act_f = actuals.merge(membership, on='ticker', how='inner')
    act_f = act_f[
        (act_f['anndats'] >= act_f['valid_start']) &
        (act_f['anndats'] <= act_f['valid_end'])
    ].drop(columns=['valid_start', 'valid_end']).drop_duplicates().reset_index(drop=True)

    act_f = act_f.sort_values(['ticker', 'anndats', 'pdicity']).reset_index(drop=True)

    print(f"  Rows after filter  : {len(act_f):,}")
    print(f"  Unique tickers     : {act_f['ticker'].nunique():,}")
    print(f"  Date range         : {act_f['anndats'].min().date()} → "
          f"{act_f['anndats'].max().date()}")
    if 'pdicity' in act_f.columns:
        print(f"  Periodicity split  : {act_f['pdicity'].value_counts().to_dict()}")

    # Diagnostic: 2025 survival check
    raw_2025  = actuals[actuals['anndats'] > '2024-12-31']
    kept_2025 = act_f[act_f['anndats']    > '2024-12-31']
    print(f"\n  Raw actuals with anndats > 2024  : {len(raw_2025):,}")
    print(f"  Kept after SP500 filter          : {len(kept_2025):,}")

    actuals_out = f"{output_dir}ibes_actuals_sp500_filtered.csv"
    act_f.to_csv(actuals_out, index=False)
    print(f"  Saved → '{actuals_out}'")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
    print(f"  Summary  : {len(sum_f):,} rows  → '{summary_out}'")
    print(f"  Actuals  : {len(act_f):,} rows  → '{actuals_out}'")

    return sum_f, act_f


if __name__ == "__main__":
    filter_ibes_to_sp500(
        username='rh3245',
        summary_path='ibes2025/ibessummary.csv',
        actuals_path='ibes2025/ibesactuals.csv',
        output_dir='ibes2025/ibes_filtered'
    )