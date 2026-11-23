"""Simulate the tasks world once and lay out what the tasks need.

    cases/NAME/truth/world/   the simulator's truth (write_truth): never shown to players
    cases/NAME/truth/obs/     Brindlemark's raw observer tables, as written
    cases/NAME/truth/rows.parquet   every Brindlemark row with a customer, linked to its truth event and cause
    cases/NAME/bank/          Brindlemark's tables as the bank sees them, each with LOAD_TS (when the row reached the
                              bank) and with columns that would reveal later outcomes masked (see MASKS)

Players only ever see bank/ rows with LOAD_TS at or before their case time (cli.py loads them that way)."""

import glob
import json
import os
import time
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
# The tasks tier of Runmere's Fraud World (worlds/fraud-world/world.yaml at its default fraud_density); set
# FA_TASKS_WORLD to that file. The local copy is the fallback.
WORLD = Path(__import__("os").environ.get("FA_TASKS_WORLD") or REPO / "worlds" / "fraud-world-tasks.yaml")
OBSERVER = "brindlemark"

# Columns that hold an end-of-run state (a later return, closure, block or dispute decision) rather than the state at
# the time of the row. They are dropped or reset in bank/ so that an as-of view cannot see the future through them.
# Returns, closures and dispute outcomes stay visible through their own rows (ACH_RETURNS, CHECKS returns, FRAUD_TAGS,
# CALLS) once those rows have loaded.
MASKS = {
    "ACCOUNTS": {"drop": ["STATUS", "RESTRICTED_FLG", "BALANCE"]},
    "CARDS": {"drop": ["STATUS"]},
    "DISPUTES": {"drop": ["MERCHANT_WON"]},
    "ACH": {"replace": {"STATUS": "CASE WHEN o.STATUS = 'RETURNED' THEN 'POSTED' ELSE o.STATUS END"}},
    "CHECKS": {"replace": {"RETURN_REASON": "CASE WHEN o.STATUS = 'POSTED' THEN NULL ELSE o.RETURN_REASON END"}},
    "REVIEW_QUEUE": {"drop": ["DECIDED_TS", "STATUS"]},
    "ALERTS": {"drop": ["RESOLUTION"]},
}


def simulate(out, *, seed, roots, days, world=WORLD, log=print):
    import runmere
    out = Path(out)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    w = runmere.World.load(str(world))
    run = w.start(seed, 1, roots)
    for d in range(10, days + 1, 10):
        run.run_until_day(d)
        log(f"  day {d}/{days} ({time.time() - t0:.0f}s)")
    if days % 10:
        run.run_until_day(days)
    run.write_observer(OBSERVER, str(out / "truth" / "obs"))
    run.write_truth(str(out / "truth" / "world"))
    meta = {"seed": seed, "roots": roots, "days": days, "world": (str(Path(world).relative_to(REPO)) if Path(world).is_relative_to(REPO) else str(world)),
            "world_version": runmere.World.load(str(world)).version(),
            "seconds": round(time.time() - t0)}
    (out / "world.json").write_text(json.dumps(meta, indent=1))
    log(f"  simulated in {meta['seconds']}s")
    return meta


def connect_raw(out):
    """DuckDB with the raw observer tables (as views named like the tables) and truth tables (t_<name>)."""
    out = Path(out)
    db = duckdb.connect()
    for f in sorted(glob.glob(str(out / "truth" / "obs" / "*.parquet"))):
        db.execute(f'CREATE VIEW "{Path(f).stem}" AS SELECT * FROM read_parquet(\'{f}\')')
    for f in sorted(glob.glob(str(out / "truth" / "world" / "*.parquet"))):
        db.execute(f'CREATE VIEW "t_{Path(f).stem}" AS SELECT * FROM read_parquet(\'{f}\')')
    return db


def observer_tables(out):
    return sorted(Path(f).stem for f in glob.glob(str(Path(out) / "truth" / "obs" / "*.parquet")))


def materialize(out, log=print):
    """Write bank/ (as-of-ready tables) and truth/rows.parquet."""
    out = Path(out)
    db = connect_raw(out)
    (out / "bank").mkdir(exist_ok=True)
    # A row about an entity (a customer, account, card, device...) is written once, in its latest version, with that
    # version's load time; the bank has known the entity since it was created. Its LOAD_TS is therefore the entity's
    # creation (the earliest valid_from of its attributes), so that as-of views keep it from then on.
    db.execute(f"""CREATE TABLE created AS SELECT entity, min(valid_from) created FROM t_hidden
                   WHERE entity IN (SELECT entity FROM t_row_links WHERE observer = '{OBSERVER}' AND entity IS NOT NULL)
                   GROUP BY 1""")
    db.execute(f"""CREATE TABLE links AS SELECT "table" AS tbl, "row" AS rn, event, l.entity,
                          CASE WHEN l.entity IS NOT NULL THEN least(coalesce(c.created, l.load_time), l.load_time)
                               ELSE l.load_time END AS load_time
                   FROM t_row_links l LEFT JOIN created c ON c.entity = l.entity WHERE observer = '{OBSERVER}'""")
    parts = []
    for name in observer_tables(out):
        cols = [c[0] for c in db.execute(f'DESCRIBE "{name}"').fetchall()]
        m = MASKS.get(name, {})
        sel = []
        for c in cols:
            if c in m.get("drop", []) or c == "LOAD_TS":
                continue
            sel.append(f'{m["replace"][c]} AS "{c}"' if c in m.get("replace", {}) else f'o."{c}"')
        src = f'(SELECT *, row_number() OVER () - 1 AS _rn FROM "{name}")'
        q = f"""SELECT {', '.join(sel)},
                       l.load_time AS LOAD_TS
                FROM {src} o LEFT JOIN links l ON l.tbl = '{name}' AND l.rn = o._rn ORDER BY o._rn"""
        db.execute(f"COPY ({q}) TO '{out / 'bank' / (name + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
        if "CUST_ID" in cols:
            parts.append(f"SELECT '{name}' AS tbl, _rn AS rn, CUST_ID FROM {src}")
    db.execute("CREATE TABLE cr AS " + " UNION ALL ".join(parts))
    db.execute(f"""COPY (SELECT cr.tbl, cr.rn, cr.CUST_ID, l.event, l.load_time, e.time, e.event AS ev, e.actor,
                               e.actor_type, e.cause_scenario AS sc, e.cause_instance AS inst, e.cause_label AS lab
                        FROM cr JOIN links l ON l.tbl = cr.tbl AND l.rn = cr.rn
                        LEFT JOIN t_events e ON e.key = l.event) TO '{out / 'truth' / 'rows.parquet'}' (FORMAT parquet)""")
    # entity maps: customer -> identity entity, account -> account entity, device -> device entity
    for name, key in (("CUSTOMERS", "CUST_ID"), ("ACCOUNTS", "ACCT_ID"), ("DEVICES", "DEVICE_ID"), ("CARDS", "CARD_ID")):
        extra = ", o.CUST_ID" if name in ("ACCOUNTS", "CARDS") else ""
        db.execute(f"""COPY (SELECT o.{key}{extra}, l.entity FROM (SELECT *, row_number() OVER () - 1 AS _rn FROM "{name}") o
                            JOIN links l ON l.tbl = '{name}' AND l.rn = o._rn)
                       TO '{out / 'truth' / ('map_' + name.lower() + '.parquet')}' (FORMAT parquet)""")
    log(f"  bank/ has {len(observer_tables(out))} tables")


def bank_tables(out):
    return sorted(Path(f).stem for f in glob.glob(str(Path(out) / "bank" / "*.parquet")))
