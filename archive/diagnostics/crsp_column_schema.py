import wrds
db = wrds.Connection(wrds_username="rh3245")

for tbl in ["crsp_a_stock.dsf_v2", "crsp.dsf_v2"]:
    try:
        result = db.raw_sql(f"SELECT * FROM {tbl} LIMIT 1")
        print(f"\n{tbl} columns:")
        for col in result.columns:
            print(f"  {col}: {result[col].dtype}  sample={result[col].iloc[0]}")
    except Exception as e:
        print(f"{tbl}: ERROR — {e}")

db.close()