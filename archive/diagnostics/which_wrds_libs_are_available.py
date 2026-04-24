import wrds
db = wrds.Connection(wrds_username='rh3245')

# Step 1: find all schemas with 'option' in the name
all_libs = db.list_libraries()
option_libs = [l for l in all_libs if 'option' in l.lower()]
print("=== OPTION-RELATED SCHEMAS ===")
print(option_libs)

# Step 2: list tables inside each one
for lib in option_libs:
    try:
        tables = db.list_tables(library=lib)
        print(f"\n=== TABLES IN {lib} ===")
        print(tables)
    except Exception as e:
        print(f"Could not list {lib}: {e}")