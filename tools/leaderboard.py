"""Alert-decisions leaderboard over several seeds: the five columns per player with paired bootstrap 95% CIs
(resampling cases within each seed), the difference between two players, and PR-AUC per seed.

    python tools/leaderboard.py CASES... --results grayson=a.json,b.json jev=c.json,d.json [--rate 0.10]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.tasks.decide import leaderboard_metrics  # noqa: E402

COLS = ["roc_auc", "pr_auc", "log_loss", "brier", "precision_at_80_recall", "fraud_type", "customer_role"]

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--results", nargs="+", required=True, help="name=file1.json,file2.json")
ap.add_argument("--rate", type=float, default=None)
ap.add_argument("--boots", type=int, default=1000)
ap.add_argument("--json", help="write the board (point estimates and 95%% CIs) here")
a = ap.parse_args()

keys, seed_of = {}, {}
for c in a.cases:
    c = Path(c)
    seed = json.loads((c / "index.json").read_text())["seed"]
    for f in (c / "truth" / "keys").glob("*.json"):
        cid = f"s{seed}-{f.stem}"
        keys[cid] = json.loads(f.read_text())
        seed_of[cid] = seed
players = {}
for spec in a.results:
    name, files = spec.split("=", 1)
    res = {}
    for f in files.split(","):
        res.update(json.loads(Path(f).read_text())["results"])
    players[name] = {c: r for c, r in res.items() if c in keys and r.get("answer")}
common = sorted(set(keys).intersection(*[set(r) for r in players.values()]))
seeds = sorted({seed_of[c] for c in common})
by_seed = {s: [c for c in common if seed_of[c] == s] for s in seeds}


def metrics(name, ids):
    return leaderboard_metrics({c: keys[c] for c in ids}, {c: players[name][c] for c in ids}, rate=a.rate)


point = {n: metrics(n, common) for n in players}
boots = {n: {k: [] for k in COLS} for n in players}
rng = random.Random(0)
for _ in range(a.boots):
    ids = [c for s in seeds for c in rng.choices(by_seed[s], k=len(by_seed[s]))]
    # duplicates need distinct ids for the dict-based metric
    ids2 = [f"{c}#{i}" for i, c in enumerate(ids)]
    for n in players:
        m = leaderboard_metrics({x: keys[c] for x, c in zip(ids2, ids)},
                                {x: players[n][c] for x, c in zip(ids2, ids)}, rate=a.rate)
        for k in COLS:
            boots[n][k].append(m[k])


def ci(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


fraud = sum(keys[c]["disposition"] == "fraud" for c in common)
print(f"{len(common)} alerts ({fraud} fraud) over seeds {seeds}; rate {a.rate or 'default'}")
print(f"{'':10}" + "".join(f"{k:>26}" for k in COLS))
for n in players:
    print(f"{n:10}" + "".join(f"{point[n][k]:>8.3f} [{ci(boots[n][k])[0]:.3f},{ci(boots[n][k])[1]:.3f}]"
                              for k in COLS))
names = list(players)
if len(names) == 2:
    x, y = names
    row = ""
    for k in COLS:
        d = [p - q for p, q in zip(boots[x][k], boots[y][k]) if p is not None and q is not None]
        lo, hi = ci(d)
        row += f"{point[x][k] - point[y][k]:>+8.3f} [{lo:+.3f},{hi:+.3f}]"
    print(f"{x}-{y:<{9 - len(x)}}" + row)
print("\nPR-AUC per seed:")
for s in seeds:
    print(f"  s{s} ({len(by_seed[s])} alerts): " + "  ".join(f"{n} {metrics(n, by_seed[s])['pr_auc']:.3f}" for n in players))

if a.json:
    board = {"rate": a.rate, "seeds": seeds, "n": len(common), "fraud": fraud, "columns": COLS, "players": {}}
    for n in players:
        board["players"][n] = {k: {"value": point[n][k], "ci95": list(ci(boots[n][k]))} for k in COLS}
        board["players"][n]["pr_auc_by_seed"] = {f"s{s}": metrics(n, by_seed[s])["pr_auc"] for s in seeds}
    Path(a.json).write_text(json.dumps(board, indent=1))
    print("wrote", a.json)

