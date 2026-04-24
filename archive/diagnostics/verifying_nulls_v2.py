import wrds
db = wrds.Connection(wrds_username="rh3245")

result = db.raw_sql("""
    SELECT 
        COUNT(*)                                         AS total,
        COUNT(ticker)                                    AS non_null_ticker,
        COUNT(hdrcusip)                                  AS non_null_hdrcusip
    FROM crsp_a_stock.dsf_v2
    WHERE dlycaldt >= '2025-01-01'
""")
print(result)
db.close()