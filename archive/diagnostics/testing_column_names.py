import pandas as pd

summary = pd.read_csv('ibes/ibes_summary_sp500_20140101_20241231.csv')
actuals = pd.read_csv('ibes/ibes_actuals_sp500_20140101_20241231.csv')

print("=== SUMMARY COLUMNS ===")
print(summary.columns.tolist())
print(f"Shape: {summary.shape}")
print()
print(summary.head(3).to_string())

print("\n=== ACTUALS COLUMNS ===")
print(actuals.columns.tolist())
print(f"Shape: {actuals.shape}")
print()
print(actuals.head(3).to_string())