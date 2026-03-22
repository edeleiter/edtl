import json

import ibis
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from transforms.reference_data import (
    ReferenceDataManifest,
    load_reference_tables,
    snapshot_table_to_parquet,
)


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def ref_dir(tmp_path):
    """Create a directory with two reference parquet files."""
    pq.write_table(
        pa.table({"category": ["A", "B", "C"], "code": [0, 1, 2]}),
        tmp_path / "category_map.parquet",
    )
    pq.write_table(
        pa.table({"feature": ["x"], "mean": [5.0], "std": [2.0]}),
        tmp_path / "normalization_params.parquet",
    )
    return tmp_path


def test_manifest_from_directory(ref_dir):
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    assert "category_map" in manifest.tables
    assert "normalization_params" in manifest.tables


def test_manifest_from_directory_with_manifest_json(ref_dir):
    meta = {"snapshot_timestamp": "2025-01-15T12:00:00+00:00", "source_query_hash": "abc123"}
    (ref_dir / "manifest.json").write_text(json.dumps(meta))
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    assert manifest.snapshot_timestamp is not None
    assert manifest.source_query_hash == "abc123"


def test_manifest_staleness_days():
    from datetime import datetime, timezone, timedelta
    ts = datetime.now(timezone.utc) - timedelta(days=45)
    manifest = ReferenceDataManifest(snapshot_timestamp=ts)
    days = manifest.staleness_days()
    assert days is not None
    assert 44 < days < 46


def test_manifest_staleness_none_when_no_timestamp():
    manifest = ReferenceDataManifest()
    assert manifest.staleness_days() is None


def test_manifest_warn_if_stale():
    from datetime import datetime, timezone, timedelta
    ts = datetime.now(timezone.utc) - timedelta(days=45)
    manifest = ReferenceDataManifest(snapshot_timestamp=ts)
    with pytest.warns(UserWarning, match="45"):
        manifest.warn_if_stale(max_days=30)


def test_load_reference_tables_into_duckdb(con, ref_dir):
    manifest = ReferenceDataManifest.from_directory(ref_dir)
    loaded = load_reference_tables(con, manifest)
    assert "category_map" in loaded
    result = con.execute(loaded["category_map"])
    assert len(result) == 3
    assert list(result["category"]) == ["A", "B", "C"]


def test_snapshot_table_to_parquet(con, tmp_path):
    # Create a table in DuckDB
    con.raw_sql("CREATE TABLE test_snap AS SELECT 1 AS id, 'hello' AS msg")
    out_path = tmp_path / "output" / "test_snap.parquet"
    result_path = snapshot_table_to_parquet(con, "test_snap", out_path, source_query_hash="xyz")
    assert result_path.exists()
    # Verify manifest.json was created
    manifest_path = tmp_path / "output" / "manifest.json"
    assert manifest_path.exists()
    meta = json.loads(manifest_path.read_text())
    assert "snapshot_timestamp" in meta
    assert meta["source_query_hash"] == "xyz"
    # Verify the parquet content
    loaded = pq.read_table(result_path).to_pydict()
    assert loaded["id"] == [1]
    assert loaded["msg"] == ["hello"]
