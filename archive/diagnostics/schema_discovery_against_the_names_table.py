import wrds
db = wrds.Connection(wrds_username="rh3245")

result = db.raw_sql("""
    SELECT MAX(nameenddt) AS max_date, COUNT(*) AS row_count
    FROM crsp_a_stock.stocknames
    WHERE nameenddt >= '2025-01-01'
""")
print(result)
db.close()