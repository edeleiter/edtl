import ibis
import pytest

from transforms.features.categorical import one_hot_flag, label_encode_from_map


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def sample_table():
    return ibis.memtable(
        {
            "color": ["red", "blue", "green", "red", "blue"],
            "size": ["S", "M", "L", "XL", "M"],
        }
    )


def test_one_hot_flag(con, sample_table):
    expr = one_hot_flag(sample_table, "color", "red")
    result = con.execute(expr)
    assert list(result["color_is_red"]) == [1, 0, 0, 1, 0]


def test_label_encode_from_map(con, sample_table):
    mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}
    expr = label_encode_from_map(sample_table, "size", mapping)
    result = con.execute(expr)
    assert list(result["size_encoded"]) == [0, 1, 2, 3, 1]
