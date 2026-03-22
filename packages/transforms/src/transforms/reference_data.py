"""Reference data management.

Reference tables (lookup maps, normalization parameters, etc.) are
computed at training time in Snowflake, snapshotted to Parquet, and
loaded into DuckDB at inference API startup.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import ibis
import ibis.expr.types as ir

from schemas.transforms import ReferenceDataManifest

__all__ = ["ReferenceDataManifest", "load_reference_tables", "snapshot_table_to_parquet"]


def load_reference_tables(
    conn: ibis.BaseBackend,
    manifest: ReferenceDataManifest,
) -> dict[str, ir.Table]:
    """Load all reference tables from manifest into the given backend."""
    loaded = {}
    for table_name, path in manifest.tables.items():
        conn.read_parquet(str(path), table_name=table_name)
        loaded[table_name] = conn.table(table_name)
    return loaded


def snapshot_table_to_parquet(
    conn: ibis.BaseBackend,
    table_name: str,
    output_path: str | Path,
    source_query_hash: str | None = None,
) -> Path:
    """Snapshot a table from the backend to a local Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = conn.table(table_name)
    result = conn.execute(table)
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_table = pa.Table.from_pandas(result)
    pq.write_table(arrow_table, output_path)

    # Read-merge-write manifest.json to preserve entries from prior snapshots
    manifest_path = output_path.parent / "manifest.json"
    manifest_meta: dict[str, str] = {}
    if manifest_path.exists():
        manifest_meta = json.loads(manifest_path.read_text())
    manifest_meta["snapshot_timestamp"] = datetime.now(timezone.utc).isoformat()
    if source_query_hash is not None:
        manifest_meta["source_query_hash"] = source_query_hash
    manifest_path.write_text(json.dumps(manifest_meta, indent=2))

    return output_path
