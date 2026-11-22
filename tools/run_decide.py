"""Run the alert-decisions format: python tools/run_decide.py CASES_DIR PLAYER OUT_JSON [--n N] [--parallel P]"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.tasks.decide import play, score

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+"); ap.add_argument("--player", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--parallel", type=int, default=6)
a = ap.parse_args()
jobs = []
for cases in a.cases:
    cases = Path(cases)
    ids = sorted(p.name for p in (cases / "alert").iterdir() if (p / "case.json").exists())
    for cid in ids[: a.n or None]:
        jobs.append((f"s{json.loads((cases / 'index.json').read_text())['seed']}-{cid}", cases, cid))
def one(job):
    name, cases, cid = job
    try:
        return name, play(a.player, cases / "alert" / cid)
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}
with ThreadPoolExecutor(a.parallel) as pool:
    results = dict(pool.map(one, jobs))
errors = {k: v["error"] for k, v in results.items() if "error" in v}
ok = {k: v for k, v in results.items() if "error" not in v}
keys = {}
for name, cases, cid in jobs:
    keys[name] = json.loads((cases / "truth" / "keys" / f"{cid}.json").read_text())
sc = score(keys, ok)
Path(a.out).write_text(json.dumps({"player": a.player, "score": sc, "errors": errors, "results": results}, indent=1))
summary = {k: v for k, v in sc.items() if k != "rows"}
print(a.player, json.dumps(summary), f"errors={len(errors)}", f"cost=${sum(v.get('cost', 0) for v in ok.values()):.3f}",
      f"avg packet {sum(v['packet_chars'] for v in ok.values()) // max(len(ok), 1)} chars")
for k, v in list(errors.items())[:3]: print("  error", k, v[:200])
