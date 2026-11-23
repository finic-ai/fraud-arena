"""Play one Fraud Arena 1v1 match.

    python run_match.py --defender openrouter-model-id|builtin|approve-all \
                        --attacker openrouter-model-id|builtin|drain --rounds 4 --episodes 20 --out runs/NAME

Model players need OPENROUTER_API_KEY in the environment (it stays in this process, never in a container).
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arena.engine import Duel
from arena.policy import default_policy, validate
from arena.views import write_status, write_view

BASELINES = {
    "defender": {"builtin": None, "approve-all": default_policy("defender")},
    "attacker": {"builtin": None, "drain": {"ato_cash_out": {"default": "drain", "rules": []}}},
}
REFERENCE_CACHE = Path(__file__).resolve().parent / "runs" / "_reference"


def reference(duel, seed, n):
    """Both roles on the world's built-in policies: the score's zero point."""
    path = REFERENCE_CACHE / f"seed{seed}-n{n}.json"
    if path.exists():
        return json.loads(path.read_text())
    batch = duel.run_batch(seed, n, {})
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"defender": batch["defender"], "attacker": batch["attacker"]}
    path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defender", required=True)
    ap.add_argument("--attacker", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="round r uses seed + r")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--turn-cost", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from arena.harness import Player  # imported here so baseline-only matches don't need Docker

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    duel = Duel()
    entrants = {"defender": args.defender, "attacker": args.attacker}
    players, fixed = {}, {}
    for role, who in entrants.items():
        if who in BASELINES[role]:
            fixed[role] = BASELINES[role][who]
        else:
            players[role] = Player(role, who, out / role, step_limit=args.steps, cost_limit=args.turn_cost)
            (out / role / "policy.json").write_text(json.dumps(default_policy(role), indent=1))
    policies = {role: (fixed[role] if role in fixed else default_policy(role)) for role in entrants}

    history = {role: [] for role in entrants}  # (round, batch, score) per role, for views
    rounds, started = [], time.time()
    try:
        for r in range(args.rounds):
            for role, player in players.items():
                write_view(player.workdir / "data.db", role, history[role])
                write_status(player.workdir / "status.md", role, r, args.rounds,
                             [(rn, s) for rn, _, s in history[role]], policies[role])
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {role: pool.submit(p.turn, r, args.rounds, out / "logs" / f"{role}-turn{r}.traj.json")
                           for role, p in players.items()}
                turns = {role: f.result() for role, f in futures.items()}
            for role, player in players.items():
                try:
                    candidate = player.policy()
                    problems = validate(candidate, role)
                except Exception as error:
                    problems = [f"unreadable policy.json: {error}"]
                if problems:
                    turns[role]["invalid_policy"] = problems
                    (player.workdir / "policy.json").write_text(json.dumps(policies[role], indent=1))
                else:
                    policies[role] = candidate
            seed = args.seed + r
            batch = duel.run_batch(seed, args.episodes, policies)
            ref = reference(duel, seed, args.episodes)
            scores = {role: {"reward": round(batch[role], 2), "reference": round(ref[role], 2),
                             "score": round(batch[role] - ref[role], 2)} for role in entrants}
            for role in entrants:
                history[role].append((r + 1, batch, scores[role]))
            rounds.append({"round": r + 1, "seed": seed, "scores": scores, "turns": turns,
                           "policies": json.loads(json.dumps(policies))})
            print(f"round {r + 1}: defender {scores['defender']['score']:+10.1f}  attacker "
                  f"{scores['attacker']['score']:+10.1f}  " +
                  "  ".join(f"{k}:{v['status']}/{v['steps']}st/${v['cost']:.3f}" for k, v in turns.items()),
                  flush=True)
    finally:
        for p in players.values():
            p.close()

    total = {role: round(sum(x["scores"][role]["score"] for x in rounds), 2) for role in entrants}
    cost = round(sum(t["cost"] for x in rounds for t in x["turns"].values()), 4)
    result = {"entrants": entrants, "rounds_played": len(rounds), "episodes_per_round": args.episodes,
              "total_score": total, "model_cost_usd": cost, "seconds": round(time.time() - started),
              "rounds": rounds}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"total_score": total, "model_cost_usd": cost, "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
