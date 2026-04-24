import wrds
db = wrds.Connection(wrds_username="rh3245")

candidates = [
    "crsp_a_stock.dsf",
    "crsp_a_stock.dsf_v2",
    "crsp.dsf_v2",
    "crsp.dsf2",
]

for tbl in candidates:
    try:
        test = db.raw_sql(f"""
            SELECT MAX(date) AS max_date, COUNT(*) AS row_count
            FROM {tbl}
            WHERE date >= '2025-01-01'
        """, date_cols=["max_date"])
        print(f"{tbl}: {test.to_dict('records')}")
    except Exception as e:
        print(f"{tbl}: ERROR — {e}")

db.close()