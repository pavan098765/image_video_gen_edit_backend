import sqlite3
import pandas as pd

# ---- CONFIG ----
DB_PATH = "controller.db"     # Path to your SQLite database
OUTPUT_PATH = "controller_export.xlsx"  # Excel file to save

# Connect to SQLite DB
conn = sqlite3.connect(DB_PATH)

# Get list of all tables
tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table';", conn
)['name'].tolist()

print(f"Found {len(tables)} tables: {tables}")

# Create a Pandas Excel writer using openpyxl
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    for table in tables:
        print(f"Exporting table: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        # Excel sheet names cant exceed 31 chars
        sheet_name = table[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

conn.close()
print(f"? Export complete! Saved as {OUTPUT_PATH}")
