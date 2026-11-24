"""Summarize Defender board runs into the website's board data.

    python tools/summarize_defender.py --runs runs/defender --out defender.json

Reads RUNS/<seed>/manifest.json and RUNS/<seed>/<player>/result.json (from run_defender_roster.py). Per player and
seed: the composite score (sum of the weekly defender scores: fraud losses prevented + 10% of the mule and laundering
flow kept out of the bank - added friction, all against the same weeks replayed with the bank's built-in rules), its
parts, and the model cost. Per player: the mean composite over seeds with a 95% bootstrap interval (seeds resampled
together for all players, so every player is compared on the same seeds), ordered by the mean. Only players
with a result on every seed are written (the others are listed as not ranked). cost_usd is the total over all seeds.
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path

RESAMPLES = 2000


def seed_row(result, player_dir):
    weeks = result["detail"]
    d = [w["defender"] for w in weeks]
    cost, calls = 0.0, 0
    for f in sorted((player_dir / "logs").glob("defender-week*.traj.json")):
        stats = (json.loads(f.read_text()).get("info") or {}).get("model_stats") or {}
        cost += stats.get("instance_cost") or 0.0
        calls += stats.get("api_calls") or 0
    return {
        "composite": round(sum(x["score"] for x in d), 2),
        "losses_prevented": round(sum(x["prevented"] for x in d), 2),
        "added_friction": round(sum(x["added_friction"] for x in d), 2),
        "mule_flow_cut": round(sum(x["no_defense"]["mule_flow"] - x["candidate"]["mule_flow"] for x in d), 2),
        "losses_exposed": round(result["headline"]["losses_exposed"], 2),
        "deploy_errors": sum(1 for w in weeks if w.get("deploy_error")),
        "weeks": len(weeks),
        "cost_usd": round(cost, 4),
        "api_calls": calls,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    seeds, meta_players, settings = [], {}, None
    per = {}  # player -> seed -> row
    for sdir in sorted(p for p in runs.iterdir() if p.is_dir() and p.name.isdigit()):
        man = json.loads((sdir / "manifest.json").read_text())
        seeds.append(int(sdir.name))
        settings = settings or {k: man.get(k) for k in ("households", "warmup_days", "weeks", "intensity", "max_turns", "hub")}
        meta_players.update(man["players"])
        for pid in man["players"]:
            res = sdir / pid / "result.json"
            if res.exists():
                per.setdefault(pid, {})[int(sdir.name)] = seed_row(json.loads(res.read_text()), sdir / pid)
    complete = [p for p, rows in per.items() if set(rows) == set(seeds)]
    rng = random.Random(0)
    draws = [[rng.choice(seeds) for _ in seeds] for _ in range(RESAMPLES)]
    players = {}
    for pid in sorted(complete, key=lambda p: -mean([per[p][s]["composite"] for s in seeds])):
        rows, m = per[pid], meta_players[pid]
        ci = None
        if len(seeds) > 1:
            boots = sorted(mean([rows[s]["composite"] for s in d]) for d in draws)
            ci = [round(boots[int(0.025 * RESAMPLES)], 2), round(boots[int(0.975 * RESAMPLES) - 1], 2)]
        players[pid] = {
            "meta": {"display_name": m["display_name"], "provider": m["provider"], "kind": "LLM",
                     "model_id": m["model_id"]},
            "score": {"mean": round(mean([rows[s]["composite"] for s in seeds]), 2), "ci95": ci},
            "prevented": round(mean([rows[s]["losses_prevented"] for s in seeds]), 2),
            "mule_cut": round(mean([rows[s]["mule_flow_cut"] for s in seeds]), 2),
            "friction": round(mean([rows[s]["added_friction"] for s in seeds]), 2),
            "cost_usd": round(sum(rows[s]["cost_usd"] for s in seeds), 4),  # total over all seeds
            "by_seed": {str(s): {"score": rows[s]["composite"], "prevented": rows[s]["losses_prevented"],
                                 "mule_cut": rows[s]["mule_flow_cut"], "friction": rows[s]["added_friction"],
                                 "exposed": rows[s]["losses_exposed"]} for s in seeds},
        }
    out = {"updated": date.today().isoformat(), "seeds": seeds, "weeks": (settings or {}).get("weeks") or 1,
           "households": (settings or {}).get("households"),
           "world_sha": ((settings or {}).get("hub") or {}).get("sha256"), "players": players}
    out = {k: v for k, v in out.items() if v is not None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{'#':>2} {'player':10s} {'score':>10s} {'95% CI':>22s} {'prevented':>10s} {'mule cut':>10s} {'friction':>9s} {'cost':>8s}")
    for i, (pid, e) in enumerate(players.items(), 1):
        ci = e["score"]["ci95"]
        print(f"{i:>2} {pid:10s} {e['score']['mean']:>10,.0f} {(f'[{ci[0]:,.0f}, {ci[1]:,.0f}]' if ci else '-'):>22s} "
              f"{e['prevented']:>10,.0f} {e['mule_cut']:>10,.0f} {e['friction']:>9,.0f} {e['cost_usd']:>8.2f}")
    for pid in sorted(set(per) - set(complete)):
        print(f"   {pid:10s} not ranked: results on seeds {sorted(per[pid])} of {seeds}")


if __name__ == "__main__":
    main()
