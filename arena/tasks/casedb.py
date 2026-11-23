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
    _profile_as_of(db, bank_dir, case_time)
    if lock:
        db.execute("SET enable_external_access = false")
        db.execute("SET lock_configuration = true")
    return db


def _profile_as_of(db, bank_dir, case_time):
    """CUSTOMERS holds each customer's latest record, so an email or phone changed after the case time would show its
    later value. Roll those fields back: to the last change before the case time if there is one, else blank (the
    bank keeps only new values, not old ones)."""
    f = Path(bank_dir) / "PROFILE_CHANGES.parquet"
    if not f.exists():
        return
    for field, col in (("EMAIL", "NEW_EMAIL"), ("PHONE", "NEW_PHONE")):
        db.execute(f"""
            UPDATE CUSTOMERS c SET {field} = (
                SELECT p.{col} FROM read_parquet('{f}') p
                WHERE p.CUST_ID = c.CUST_ID AND p.FIELD_CHANGED = '{field}' AND p.CHANGE_TS <= CAST(? AS TIMESTAMPTZ)
                ORDER BY p.CHANGE_TS DESC LIMIT 1)
            WHERE c.CUST_ID IN (SELECT CUST_ID FROM read_parquet('{f}')
                                WHERE FIELD_CHANGED = '{field}' AND CHANGE_TS > CAST(? AS TIMESTAMPTZ))""",
                   [str(case_time), str(case_time)])
    db.execute(f"""UPDATE CUSTOMERS SET ADDRESS = NULL, CITY = NULL, STATE = NULL, ZIP = NULL
                   WHERE CUST_ID IN (SELECT CUST_ID FROM read_parquet('{f}')
                                     WHERE FIELD_CHANGED = 'ADDRESS' AND CHANGE_TS > CAST(? AS TIMESTAMPTZ))""",
               [str(case_time)])


def later_profile_changes(bank_dir):
    """customer -> times of email/phone/address changes, to keep alert cases free of profile values from the future."""
    f = Path(bank_dir) / "PROFILE_CHANGES.parquet"
    out = {}
    if f.exists():
        for cust, ts in duckdb.sql(f"""SELECT CUST_ID, CHANGE_TS FROM read_parquet('{f}')
                                       WHERE FIELD_CHANGED IN ('EMAIL', 'PHONE', 'ADDRESS')""").fetchall():
            out.setdefault(cust, []).append(ts)
    return out
