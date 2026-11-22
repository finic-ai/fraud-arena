"""The bank's tables as of a case time: an in-memory DuckDB holding only rows whose LOAD_TS is at or before the case
time. After loading, file access is switched off and the configuration locked, so a player's SQL can't read any
file (truth included) or turn access back on."""

import glob
from pathlib import Path

import duckdb


def open_asof(bank_dir, case_time, *, lock=True):
    db = duckdb.connect(":memory:")
    db.execute("SET TimeZone = 'UTC'")
    for f in sorted(glob.glob(str(Path(bank_dir) / "*.parquet"))):
        name = Path(f).stem
        db.execute(f"""CREATE TABLE "{name}" AS SELECT * FROM read_parquet('{f}')
                       WHERE LOAD_TS IS NULL OR LOAD_TS <= CAST(? AS TIMESTAMPTZ)""", [str(case_time)])
    if lock:
        db.execute("SET enable_external_access = false")
        db.execute("SET lock_configuration = true")
    return db
