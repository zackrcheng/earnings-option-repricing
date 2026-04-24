import wrds
db = wrds.Connection(wrds_username="rh3245")

test = db.raw_sql("""
    SELECT MAX(dlycaldt) AS max_date, COUNT(*) AS row_count
    FROM crsp_a_stock.dsf_v2
    WHERE dlycaldt >= '2025-01-01'
""")
print(test)
db.close()