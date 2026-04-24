import wrds

def find_ibes_link_table(username='rh3245'):
    db = wrds.Connection(wrds_username=username)

    print("=== Searching for IBES-CRSP link table ===\n")

    # %% escapes the % so SQLAlchemy doesn't treat it as a parameter placeholder
    query = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_name ILIKE '%%ibcrsp%%'
        ORDER BY table_schema, table_name
    """
    results = db.raw_sql(query)
    print("Tables with 'ibcrsp' in the name:")
    print(results)
    print()

    # Also search more broadly for any ibes-related link tables
    query2 = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE (table_name ILIKE '%%ibes%%' OR table_name ILIKE '%%iblink%%')
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """
    results2 = db.raw_sql(query2)
    print("Tables with 'ibes' or 'iblink' in the name:")
    print(results2)

    db.close()

if __name__ == "__main__":
    find_ibes_link_table(username='rh3245')