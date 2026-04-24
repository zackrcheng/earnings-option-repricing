import wrds
db = wrds.Connection(wrds_username="rh3245")

test = db.raw_sql("""
    SELECT MAX(date) AS max_date, COUNT(*) AS row_count
    FROM crsp.dsf
    WHERE date >= '2025-01-01'
""", date_cols=["max_date"])

print(test)
db.close()