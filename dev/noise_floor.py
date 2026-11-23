"""1v1 noise floor: how much does a defender's score move from simulation randomness alone?
Plays N weeks where the defender's policy is the bank's built-in rules (identical to the no-defense counterfactual)
and the attacker's plan is a fixed change; the defender score should be ~0 and the attacker score should repeat.
Any spread is noise from the forks diverging. Writes JSON to argv[1]."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.fw.match import Match
out = Path(sys.argv[1]); roots = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
m = Match(out.parent / "noise-work", roots=roots, warmup_days=20, intensity=3.0)
m.set_plan({"all": {"plan_tactic_ach_return": 2, "plan_tactic_card_testing": 2}})
# the defender redeploys the built-in rules as its own policy (so the candidate is a different fork with the same rules)
builtin_policy = {"format": 1, "rules": [dict(r) for r in m.builtin]}
res = []
for wk in range(3):
    print("deploy:", m.deploy(builtin_policy), flush=True)
    r, *_ = m.play_week()
    d = r["defender"]
    res.append({"week": r["week"], "defender": d["score"], "prevented": d["prevented"], "friction": d["added_friction"],
                "losses_cand": d["candidate"]["losses"], "losses_nodef": d["no_defense"]["losses"], "attacker": r["attacker"]["score"]})
    print(res[-1], flush=True)
    out.write_text(json.dumps(res, indent=1))
