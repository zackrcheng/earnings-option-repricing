import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

# --- Load ---
summary = pd.read_csv('ibes/ibes_summary_sp500_20140101_20241231.csv')
actuals = pd.read_csv('ibes/ibes_actuals_sp500_20140101_20241231.csv')

# --- Parse dates ---
summary['fpedats']  = pd.to_datetime(summary['fpedats'])
summary['statpers'] = pd.to_datetime(summary['statpers'])
actuals['pends']    = pd.to_datetime(actuals['pends'])
actuals['anndats']  = pd.to_datetime(actuals['anndats'])

# --- Filter: quarterly EPS only ---
summary = summary[summary['measure'] == 'EPS']
actuals = actuals[
    (actuals['measure']  == 'EPS') &
    (actuals['pdicity']  == 'QTR')
]

# --- Step 1: Merge on firm-quarter ---
merged = summary.merge(
    actuals,
    left_on  = ['ticker', 'fpedats'],
    right_on = ['ticker', 'pends'],
    how      = 'inner'
)

# --- Step 2: Keep only pre-announcement consensus, most recent per firm-quarter ---
merged = merged[merged['statpers'] <= merged['anndats']]
merged = (merged
          .sort_values('statpers')
          .groupby(['ticker', 'pends'], as_index=False)
          .last())

# --- Step 3: Build trading calendar ---
nyse         = mcal.get_calendar('NYSE')
schedule     = nyse.schedule(start_date='2013-01-01', end_date='2025-12-31')
trading_days = schedule.index.normalize()

def get_next_trading_day(date, cal):
    date = pd.Timestamp(date).normalize()
    future = cal[cal > date]
    return future[0] if len(future) > 0 else date

# --- Step 4: Apply announcement time correction ---
# anntims is already HH:MM:SS format e.g. '16:06:00'
# Extract hour directly from string
merged['hour'] = pd.to_numeric(
    merged['anntims'].astype(str).str[:2],
    errors='coerce'          # NaN or unparseable values become NaN
)

merged['event_date'] = merged['anndats']

# Only shift when time is known and after market close (hour >= 16)
time_known  = merged['hour'].notna()
after_close = time_known & (merged['hour'] >= 16)

merged.loc[after_close, 'event_date'] = (
    merged.loc[after_close, 'anndats']
    .apply(lambda d: get_next_trading_day(d, trading_days))
)

merged['date_adjusted'] = after_close.astype(int)

# --- Step 5: Construct key variables ---
# Scale by |meanest|; rows where |meanest| < 0.01 set to NaN
# to avoid near-zero division in loss/breakeven quarters.
# Replace with price-scaled version after CRSP merge in Stage 2.

denominator = merged['meanest'].abs()
valid_denom = denominator >= 0.01

merged['SUE'] = np.where(
    valid_denom,
    (merged['value'] - merged['meanest']) / denominator,
    np.nan
)

merged['Dispersion'] = np.where(
    valid_denom & merged['stdev'].notna(),
    merged['stdev'] / denominator,
    np.nan
)

# Require at least 2 analysts for dispersion to be meaningful
merged.loc[merged['numest'] < 2, 'Dispersion'] = np.nan

# Winsorise both variables at 1%-99%
for var in ['SUE', 'Dispersion']:
    low  = merged[var].quantile(0.01)
    high = merged[var].quantile(0.99)
    merged[var] = merged[var].clip(lower=low, upper=high)

# --- Step 6: Diagnostics ---
print(merged[['ticker', 'pends', 'anndats', 'anntims', 'hour',
              'event_date', 'date_adjusted',
              'value', 'meanest', 'stdev',
              'SUE', 'Dispersion']].head(20).to_string())

print(f"\nFinal rows:                      {len(merged):,}")
print(f"After-close shifts applied:      {after_close.sum():,}")
print(f"Rows with unknown/missing time:  {(~time_known).sum():,}")
print(f"SUE null (denom too small):      {merged['SUE'].isna().sum():,}")
print(f"Dispersion null:                 {merged['Dispersion'].isna().sum():,}")
print(f"\nSUE summary:\n{merged['SUE'].describe()}")
print(f"\nDispersion summary:\n{merged['Dispersion'].describe()}")

# --- Step 7: Save ---
merged.to_csv('ibes/ibes_stage1_events.csv', index=False)
print("\nSaved: ibes/ibes_stage1_events.csv")