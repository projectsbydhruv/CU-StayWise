import pandas as pd
from sqlalchemy import create_engine, text
from db_config import get_connection_string


def main():
    print("Reading CSV file...")
    df = pd.read_csv("final_integrated.csv")
    print(f"Loaded {len(df):,} rows from final_integrated.csv")

    # If the CSV still has 'Incident ZIP', normalize it (harmless if not present)
    if "Incident ZIP" in df.columns and "incident_zip" not in df.columns:
        df = df.rename(columns={"Incident ZIP": "incident_zip"})
        print("Renamed column 'Incident ZIP' -> 'incident_zip'")

    conn_str = get_connection_string()
    print(f"Connecting to Postgres using: {conn_str}")

    engine = create_engine(conn_str)

    table_name = "final_integrated_raw"
    view_name = "final_integrated"

    # 1) Write the CSV to the raw table
    print(f"\nWriting data to table '{table_name}' (may take a minute)...")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Done writing table '{table_name}'.")

    # 2) Create / refresh materialized view
    with engine.begin() as conn:  # begin() auto-commits on success
        print(f"\nDropping materialized view '{view_name}' if it exists...")
        conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {view_name};"))

        print(f"Creating materialized view '{view_name}' from '{table_name}'...")
        conn.execute(text(f"""
            CREATE MATERIALIZED VIEW {view_name} AS
            SELECT * FROM {table_name};
        """))

        print("Creating indexes on materialized view (where columns exist)...")

        # Always try index on bbl
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{view_name}_bbl ON {view_name} (bbl);"))
        except Exception as e:
            print(f"  ⚠️ Could not create index on bbl: {e}")

        # Try index on incident_zip if present
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{view_name}_incident_zip ON {view_name} (incident_zip);"))
        except Exception as e:
            print(f"  ⚠️ Could not create index on incident_zip: {e}")

    print("\n✅ Database setup complete.")
    print(f"- Raw table        : {table_name}")
    print(f"- Materialized view: {view_name}")
    print("You can now run: .venv\\Scripts\\python.exe app.py")


if __name__ == "__main__":
    main()
