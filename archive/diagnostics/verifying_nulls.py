import wrds
db = wrds.Connection(wrds_username="rh3245")

result = db.raw_sql("""
    SELECT 
        COUNT(*)                                    AS total_rows,
        COUNT(nameenddt)                            AS non_null_nameenddt,
        SUM(CASE WHEN nameenddt IS NULL THEN 1 END) AS null_nameenddt,
        MAX(nameenddt)                              AS max_nameenddt
    FROM crsp_a_stock.stocknames
""")
print(result)
db.close()