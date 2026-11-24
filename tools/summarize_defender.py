"""Summarize Fraud Strategy runs into the website's board data (src/data/fraud-strategy.json).

    python tools/summarize_defender.py --runs runs/fraud-strategy --out fraud-strategy.json

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
        "weekly": [x["score"] for x in d],
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
    # Headline: net savings as a share of the seed's exposed losses (the losses under the bank's built-in rules), so
    # every seed counts equally whatever its fraud volume: (prevented + mule weight x mule cut - friction) / exposed.
    for rows in per.values():
        for r in rows.values():
            r["pct"] = 100 * r["composite"] / r["losses_exposed"] if r["losses_exposed"] else 0.0

    def boot_ci(vals):
        if len(seeds) < 2:
            return None
        boots = sorted(mean([vals[s] for s in d]) for d in draws)
        return [round(boots[int(0.025 * RESAMPLES)], 2), round(boots[int(0.975 * RESAMPLES) - 1], 2)]

    for pid in sorted(complete, key=lambda p: -mean([per[p][s]["pct"] for s in seeds])):
        rows, m = per[pid], meta_players[pid]
        ci = boot_ci({s: rows[s]["composite"] for s in seeds})
        players[pid] = {
            "meta": {"display_name": m["display_name"], "provider": m["provider"], "kind": "LLM",
                     "model_id": m["model_id"]},
            "net_pct": {"mean": round(mean([rows[s]["pct"] for s in seeds]), 2),
                        "ci95": boot_ci({s: rows[s]["pct"] for s in seeds})},
            "score": {"mean": round(mean([rows[s]["composite"] for s in seeds]), 2), "ci95": ci},
            "prevented": round(mean([rows[s]["losses_prevented"] for s in seeds]), 2),
            "mule_cut": round(mean([rows[s]["mule_flow_cut"] for s in seeds]), 2),
            "friction": round(mean([rows[s]["added_friction"] for s in seeds]), 2),
            "cost_usd": round(sum(rows[s]["cost_usd"] for s in seeds), 4),  # total over all seeds
            "by_seed": {str(s): {"net_pct": round(rows[s]["pct"], 2), "score": rows[s]["composite"], "prevented": rows[s]["losses_prevented"],
                                 "mule_cut": rows[s]["mule_flow_cut"], "friction": rows[s]["added_friction"],
                                 "exposed": rows[s]["losses_exposed"]} for s in seeds},
        }
    # Paired comparison (LMArena-style upper-bound rank): A is clearly better than B when the 95% interval of their
    # mean per seed-week score difference (same seed-week = identical fraud) is above 0. Intervals by cluster
    # bootstrap over seeds; with few seeds (< 5) the seed-level bootstrap is degenerate, so seed-weeks are resampled
    # within each seed instead (stratified), and the meta says which.
    ids = list(players)
    method = "seed_cluster" if len(seeds) >= 5 else "weeks_within_seed"
    rng2 = random.Random(1)
    def diffs(pa, pb):  # seed -> list of per-week differences
        # weekly score difference as a share of the seed's exposed losses (the headline's scale)
        return {s: [100 * (x - y) / (per[pa][s]["losses_exposed"] or 1)
                    for x, y in zip(per[pa][s]["weekly"], per[pb][s]["weekly"])] for s in seeds}
    def ci(dd):
        means = []
        for _ in range(RESAMPLES):
            if method == "seed_cluster":
                pick = [rng2.choice(seeds) for _ in seeds]
                vals = [v for s in pick for v in dd[s]]
            else:
                vals = [rng2.choice(dd[s]) for s in seeds for _ in dd[s]]
            means.append(sum(vals) / len(vals))
        means.sort()
        return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]
    better = {p: [] for p in ids}  # p -> players clearly better than p
    for i, pa in enumerate(ids):
        for pb in ids[i + 1:]:
            lo, hi = ci(diffs(pa, pb))
            if lo > 0:
                better[pb].append(pa)
            elif hi < 0:
                better[pa].append(pb)
    for p in ids:
        players[p]["rank_ub"] = 1 + len(better[p])
        players[p]["clearly_behind"] = sorted(better[p])
    # Robustness: each model's rank on each seed (by net %), the mean rank over seeds (as in comparisons of methods over
    # several datasets), and Kendall's W, the agreement of the per-seed rankings (1 = every seed ranks models alike).
    for s in seeds:
        for i, p in enumerate(sorted(ids, key=lambda q: -per[q][s]["pct"]), 1):
            players[p]["by_seed"][str(s)]["rank"] = i
    for p in ids:
        players[p]["mean_rank"] = round(mean([players[p]["by_seed"][str(s)]["rank"] for s in seeds]), 2)
    n, m = len(ids), len(seeds)
    kendall_w = None
    if n > 1 and m > 1:
        sums = [sum(players[p]["by_seed"][str(s)]["rank"] for s in seeds) for p in ids]
        avg = sum(sums) / n
        kendall_w = round(12 * sum((x - avg) ** 2 for x in sums) / (m * m * (n ** 3 - n)), 3)
    out = {"updated": date.today().isoformat(), "seeds": seeds, "weeks": (settings or {}).get("weeks") or 1,
           "households": (settings or {}).get("households"),
           "world_sha": ((settings or {}).get("hub") or {}).get("sha256"),
           "rank_method": {"order": "net_pct: (prevented + mule weight x mule cut - friction) / exposed losses, mean over seeds",
                           "rank_ub": "1 + players clearly better (paired 95% CI on the mean seed-week difference in net_pct > 0)",
                           "bootstrap": method, "resamples": RESAMPLES},
           "kendall_w": kendall_w, "players": players}
    out = {k: v for k, v in out.items() if v is not None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{'ub':>3} {'#':>2} {'player':10s} {'net %':>6s} {'score':>10s} {'95% CI':>22s} {'prevented':>10s} {'mule cut':>10s} {'friction':>9s} {'cost':>8s}")
    for i, (pid, e) in enumerate(players.items(), 1):
        ci = e["score"]["ci95"]
        print(f"{e['rank_ub']:>3} {i:>2} {pid:10s} {e['net_pct']['mean']:>6.1f} {e['score']['mean']:>10,.0f} {(f'[{ci[0]:,.0f}, {ci[1]:,.0f}]' if ci else '-'):>22s} "
              f"{e['prevented']:>10,.0f} {e['mule_cut']:>10,.0f} {e['friction']:>9,.0f} {e['cost_usd']:>8.2f}")
    print(f"mean rank over seeds: " + ", ".join(f"{p} {e['mean_rank']}" for p, e in players.items()))
    print(f"Kendall's W (agreement of per-seed rankings): {kendall_w}")
    for pid in sorted(set(per) - set(complete)):
        print(f"   {pid:10s} not ranked: results on seeds {sorted(per[pid])} of {seeds}")


if __name__ == "__main__":
    main()
