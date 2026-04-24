import wrds
import pandas as pd
import os


def download_ibes_summary_sp500(username='rh3245',
                                start_date='2014-01-01',
                                end_date='2024-12-31',
                                output_dir='ibes'):
    """
    Downloads IBES Summary Statistics from WRDS for S&P 500 firms only,
    saving the full date range as a single CSV file.

    S&P 500 membership is TIME-MATCHED: a firm is included only if it was
    an S&P 500 constituent at the time of the STATPERS date.

    Parameters
    ----------
    username   : str - your WRDS username
    start_date : str - start date in 'YYYY-MM-DD' format
    end_date   : str - end date in 'YYYY-MM-DD' format
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
            'ibes.statsum_epsus'                 : 'SELECT COUNT(*) FROM ibes.statsum_epsus',
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

        print(f"\nDownloading IBES data from {start_date} to {end_date}...")

        query = """
            SELECT s.ticker,
                   s.cusip,
                   s.statpers,
                   s.fpedats,
                   s.meanest,
                   s.medest,
                   s.stdev,
                   s.numest,
                   s.measure,
                   s.fpi
            FROM ibes.statsum_epsus s

            -- Step 1: Link IBES ticker to CRSP PERMNO via WRDS link table.
            --         lnk.sdate / lnk.edate define when the link is valid.
            INNER JOIN wrdsapps_link_crsp_ibes.ibcrsphist lnk
                ON  s.ticker   = lnk.ticker
                AND s.statpers BETWEEN lnk.sdate AND lnk.edate

            -- Step 2: Filter to S&P 500 constituents.
            --         sp.start / sp.ending define the membership window.
            --         COALESCE handles currently active members where ending IS NULL.
            INNER JOIN crsp.msp500list sp
                ON  lnk.permno = sp.permno
                AND s.statpers BETWEEN sp.start
                    AND COALESCE(sp.ending, CURRENT_DATE)

            WHERE s.statpers BETWEEN '{start}' AND '{end}'
              AND s.measure = 'EPS'
              AND s.fpi    = '6'
        """.format(start=start_date, end=end_date)

        df = db.raw_sql(query, date_cols=['statpers', 'fpedats'])
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

        # Save to single CSV
        start_str = start_date.replace('-', '')
        end_str   = end_date.replace('-', '')
        output_file = f"{output_dir}ibes_summary_sp500_{start_str}_{end_str}.csv"
        df.to_csv(output_file, index=False)

        print(f"\n--- Download Summary ---")
        print(f"  Total rows : {after:,}")
        print(f"  Date range : {df['statpers'].min()} to {df['statpers'].max()}")
        print(f"  Unique tickers : {df['ticker'].nunique():,}")
        print(f"  Saved to   : '{output_file}'")

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    download_ibes_summary_sp500(
        username='rh3245',
        start_date='2014-01-01',
        end_date='2024-12-31',
        output_dir='ibes'
    )