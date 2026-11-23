"""Check that the ring plan menu changes what rings do: run Fraud World, set plans at day 30, count each scheme's
scenario starts from day 30 to 90. Usage: test_ring_knobs.py WORLD OUT_DIR"""
import json
import sys
from pathlib import Path

import duckdb
import runmere

WORLD, OUT = sys.argv[1], Path(sys.argv[2])
SEED, ROOTS, SWITCH, END = 7, 1500, 30, 90
world = runmere.World.load(WORLD)

CASES = {
    "unset": {},
    "all_quiet": {"*": {"plan_quiet": True}},
    "all_volume_0": {"*": {"plan_volume": 0.0}},
    "r10_promo_farm_x3": {10: {"plan_tactic_promo_farm": 3.0}},
    "r6_card_testing_x3_volume_x2": {6: {"plan_volume": 2.0, "plan_tactic_card_testing": 3.0}},
    "r1_all_brindlemark": {1: {"plan_where": "planned", "plan_share_brindlemark": 1.0, "plan_share_arbelton": 0.0}},
}


def run_case(name, plan):
    run = world.start(SEED, 1, ROOTS)
    run.run_until_day(SWITCH)
    for row in run.entities("ring", ["number"]).to_pylist():
        ring_id, number = row["entity"], row["number"]
        for key in ("*", number):
            for attr, value in plan.get(key, {}).items():
                run.intervene("ring", ring_id, attr, value)
    switch_time = run.time if not callable(run.time) else run.time()
    run.run_until_day(END)
    inst = run.truth_table("instances")
    if name == "unset":
        print("instances columns:", inst.schema.names, flush=True)
    start_col = next(c for c in inst.schema.names if c in ("started", "start", "created", "time", "started_at"))
    rows = duckdb.sql(f"SELECT scenario, count(*) AS n FROM inst WHERE {start_col} >= to_timestamp({switch_time} / 1000.0) GROUP BY 1").fetchall()
    return {r[0]: r[1] for r in rows}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, plan in CASES.items():
        results[name] = run_case(name, plan)
        print("done", name, sum(results[name].values()), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    base = results["unset"]
    keys = sorted({k for r in results.values() for k in r})
    print("scenario".ljust(32) + "".join(n[:14].rjust(15) for n in results))
    for k in keys:
        print(k.ljust(32) + "".join(str(r.get(k, 0)).rjust(15) for r in results.values()))
