import wrds
import pandas as pd
import os


def download_ibes_actuals_sp500(username='rh3245',
                                start_date='2014-01-01',
                                end_date='2024-12-31',
                                output_dir='ibes'):
    """
    Downloads IBES Actual EPS Announcements from WRDS for S&P 500 firms only,
    saving the full date range as a single CSV file.

    The IBES actuals table (ibes.actu_epsus) contains the actual EPS values
    reported by firms along with the announcement date.

    S&P 500 membership is TIME-MATCHED to the announcement date (anndats):
    a firm is included only if it was an S&P 500 constituent on the date
    it announced earnings.

    Parameters
    ----------
    username   : str - your WRDS username
    start_date : str - start date in 'YYYY-MM-DD' format (filters on anndats)
    end_date   : str - end date in 'YYYY-MM-DD' format   (filters on anndats)
    output_dir : str - folder to save CSV file
    """

    # Normalise output directory path
    if output_dir and not output_dir.endswith('/'):
        output_dir = output_dir + '/'

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Connecting to WRDS as '{username}'...")
        db = wrds.Connection(wrds_username=username)
        print("Connection successful.\n")

        # Verify access to required tables before querying
        print("Verifying access to required tables...")
        checks = {
            'ibes.actu_epsus'                    : 'SELECT COUNT(*) FROM ibes.actu_epsus',
            'wrdsapps_link_crsp_ibes.ibcrsphist' : 'SELECT COUNT(*) FROM wrdsapps_link_crsp_ibes.ibcrsphist',
            'crsp.msp500list'                    : 'SELECT COUNT(*) FROM crsp.msp500list',
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
            return None

        # Quick schema inspection so we can confirm column names at runtime
        print("\nInspecting ibes.actu_epsus columns...")
        cols_df = db.raw_sql("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'ibes'
              AND table_name   = 'actu_epsus'
            ORDER BY ordinal_position
        """)
        print(f"  Available columns: {list(cols_df['column_name'])}\n")

        print(f"Downloading IBES actuals from {start_date} to {end_date}...")

        query = """
            SELECT a.ticker,
                   a.cusip,
                   a.cname,
                   a.oftic,
                   a.measure,
                   a.pdicity,
                   a.pends,
                   a.value,
                   a.anndats,
                   a.anntims,
                   a.actdats,
                   a.acttims

            FROM ibes.actu_epsus a

            -- Step 1: Link IBES ticker to CRSP PERMNO via WRDS link table.
            --         Match on announcement date so the link is contemporaneous.
            INNER JOIN wrdsapps_link_crsp_ibes.ibcrsphist lnk
                ON  a.ticker   = lnk.ticker
                AND a.anndats  BETWEEN lnk.sdate AND lnk.edate

            -- Step 2: Filter to S&P 500 constituents, time-matched to anndats.
            --         COALESCE handles currently active members where ending IS NULL.
            INNER JOIN crsp.msp500list sp
                ON  lnk.permno = sp.permno
                AND a.anndats  BETWEEN sp.start
                    AND COALESCE(sp.ending, CURRENT_DATE)

            WHERE a.anndats  BETWEEN '{start}' AND '{end}'
              AND a.measure  = 'EPS'
              AND a.pdicity  IN ('QTR', 'ANN')
        """.format(start=start_date, end=end_date)

        df = db.raw_sql(query, date_cols=['pends', 'anndats', 'actdats'])
        db.close()
        print("Connection closed.")

        if df is None or len(df) == 0:
            print("No data returned. Check your date range and filters.")
            return None

        # Remove any duplicates that can arise from the join
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"Removed {before - after:,} duplicate rows after join.")

        # Sort for readability
        df = df.sort_values(['ticker', 'anndats', 'pdicity']).reset_index(drop=True)

        # Save to single CSV
        start_str = start_date.replace('-', '')
        end_str   = end_date.replace('-', '')
        output_file = f"{output_dir}ibes_actuals_sp500_{start_str}_{end_str}.csv"
        df.to_csv(output_file, index=False)

        print(f"\n--- Download Summary ---")
        print(f"  Total rows        : {after:,}")
        print(f"  Date range        : {df['anndats'].min()} to {df['anndats'].max()}")
        print(f"  Unique tickers    : {df['ticker'].nunique():,}")
        print(f"  Periodicity split : {df['pdicity'].value_counts().to_dict()}")
        print(f"  Saved to          : '{output_file}'")

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    download_ibes_actuals_sp500(
        username='rh3245',
        start_date='2014-01-01',
        end_date='2024-12-31',
        output_dir='ibes'
    )