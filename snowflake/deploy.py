"""Deploy transforms as Snowflake UDFs and stored procedures.

Usage:
    uv run python snowflake/deploy.py --action upload-reference
    uv run python snowflake/deploy.py --action create-sproc
    uv run python snowflake/deploy.py --action all

Requires SNOWFLAKE_* environment variables to be set (see .env.example).
"""

import argparse
from pathlib import Path

from backends.snowflake_backend import SnowflakeBackend
from transforms.reference_data import ReferenceDataManifest


def upload_reference_data(conn, ref_dir: str = "reference_data"):
    """Upload local Parquet reference files to Snowflake stage."""
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    if not manifest.tables:
        print(f"No Parquet files found in {ref_dir}/")
        return
    for table_name, path in manifest.tables.items():
        stage_path = f"@reference_data_stage/{path.name}"
        print(f"Uploading {path} -> {stage_path}")
        conn.raw_sql(
            f"PUT 'file://{path.resolve()}' {stage_path} "
            f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        )
        print(f"  Loading into table {table_name}...")
        conn.raw_sql(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM @reference_data_stage/{path.name}
            (FILE_FORMAT => (TYPE = PARQUET))
        """)
        print(f"  Done: {table_name}")


def create_transform_sproc(conn):
    """Create a stored procedure that runs the transform pipeline."""
    sproc_sql = """
    CREATE OR REPLACE PROCEDURE run_feature_pipeline(
        source_table VARCHAR,
        output_table VARCHAR
    )
    RETURNS VARCHAR
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.11'
    PACKAGES = ('ibis-framework', 'pyarrow')
    HANDLER = 'main'
    AS
    $$
def main(session, source_table: str, output_table: str) -> str:
    import ibis

    conn = ibis.snowflake.connect(session=session)
    table = conn.table(source_table)

    # Example: clip + ratio
    clipped = table.mutate(
        value_clipped=ibis.greatest(ibis.least(table["value"], 100.0), 0.0)
    )

    conn.create_table(output_table, clipped, overwrite=True)
    return f"Pipeline complete: {output_table}"
    $$;
    """
    conn.raw_sql(sproc_sql)
    print("Created stored procedure: run_feature_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Deploy to Snowflake")
    parser.add_argument(
        "--action",
        choices=["upload-reference", "create-sproc", "all"],
        required=True,
    )
    args = parser.parse_args()

    backend = SnowflakeBackend()
    conn = backend.connect()

    try:
        if args.action in ("upload-reference", "all"):
            upload_reference_data(conn)
        if args.action in ("create-sproc", "all"):
            create_transform_sproc(conn)
    finally:
        backend.disconnect(conn)


if __name__ == "__main__":
    main()
