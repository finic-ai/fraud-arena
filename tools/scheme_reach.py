"""Which ring schemes can reach the defended bank: from one match checkpoint, boost one scheme (x3, every crew) for a
week in a fork and measure the ring's take at Brindlemark against an unboosted fork. Writes JSON to argv[1]."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.fw.match import Match, INSTITUTION, ROLE, WEIGHTS, iso
from arena.fw import scoring
from arena.fw.config import INSTITUTIONS

out = Path(sys.argv[1]); roots = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
m = Match(out.parent / "reach-work", world=Path("worlds/fraud-world-arena.yaml"), roots=roots, warmup_days=30, intensity=3.0)
start, end_day = iso(m.cand.time), m.warmup_days + 7

def take(run, tag):
    truth, owners = m._export(run, out.parent / "reach-work" / tag)
    total, other = 0.0, 0.0
    for number, ring in m.rings.items():
        r = scoring.ring_score(truth, owners, ring=ring, number=number, institutions=list(INSTITUTIONS),
                               start=start, end=iso(run.time), weights=WEIGHTS)
        for inst, v in r["by_institution"].items():
            x = v["losses"] + WEIGHTS.mule_flow * v["mule_flow"]
            if inst == INSTITUTION: total += x
            else: other += x
    return total, other

results = {}
base = m.cand.fork(); base.undeploy(ROLE); base.set_outside(None); base.run_until_day(end_day)
results["(none)"] = take(base, "none")
print("base", results["(none)"], flush=True)
for attr in [a for a in m.menu if a.startswith("plan_tactic_")]:
    run = m.cand.fork(); run.undeploy(ROLE); run.set_outside(None)
    for ring in m.rings.values():
        run.intervene("ring", ring, attr, 3.0)
    run.run_until_day(end_day)
    results[attr] = take(run, attr)
    print(attr, [round(x) for x in results[attr]], flush=True)
    out.write_text(json.dumps(results, indent=1))
