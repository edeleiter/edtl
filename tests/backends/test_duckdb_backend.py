import ibis
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backends.base import Backend
from backends.duckdb_backend import DuckDBBackend


def test_duckdb_backend_is_backend():
    backend = DuckDBBackend()
    assert isinstance(backend, Backend)


def test_duckdb_backend_returns_ibis_connection():
    backend = DuckDBBackend()
    conn = backend.connect()
    assert hasattr(conn, "table")
    assert hasattr(conn, "execute")


def test_duckdb_backend_execute_expression():
    backend = DuckDBBackend()
    conn = backend.connect()
    t = ibis.memtable({"x": [1, 2, 3], "y": [10, 20, 30]})
    result = conn.execute(t.mutate(z=t.x + t.y))
    assert list(result["z"]) == [11, 22, 33]


def test_duckdb_backend_load_parquet(tmp_path):
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    path = tmp_path / "ref.parquet"
    pq.write_table(table, path)

    backend = DuckDBBackend()
    conn = backend.connect()
    ref_table = backend.load_reference_table(conn, "ref_test", str(path))
    result = conn.execute(ref_table)
    assert len(result) == 2
    assert list(result["name"]) == ["a", "b"]


def test_duckdb_backend_disconnect():
    backend = DuckDBBackend()
    conn = backend.connect()
    backend.disconnect(conn)
