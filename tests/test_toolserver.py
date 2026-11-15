import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraud_arena.toolserver import Tables


@pytest.fixture
def tables(tmp_path):
    d = tmp_path / "brindlemark"
    d.mkdir()
    pq.write_table(pa.table({"ID": [1, 2, 3], "AMT": [5.0, 6.0, 900.0]}), d / "CARD_AUTHS.parquet")
    (tmp_path / "truth").mkdir()
    pq.write_table(pa.table({"label": ["ato"]}), tmp_path / "truth" / "labels.parquet")
    return Tables(d), tmp_path


def test_reads_own_tables(tables):
    t, _ = tables
    assert t.meta["CARD_AUTHS"]["rows"] == 3
    assert t.query("select max(AMT) m from CARD_AUTHS").to_pylist() == [{"m": 900.0}]


@pytest.mark.parametrize("bad", ["copy (select 1) to 'x.csv'", "select 1; select 2", "attach 'x.db'",
                                 "set enable_external_access = true", "install httpfs"])
def test_refuses_writes_and_settings(tables, bad):
    with pytest.raises(Exception):
        tables[0].query(bad)


def test_no_file_access_after_load(tables):
    t, root = tables
    with pytest.raises(Exception):
        t.query(f"select * from read_parquet('{root}/truth/labels.parquet')")
