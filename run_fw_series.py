"""Run a model's Fraud Arena 1v1 series: one match per seed, the same seeds and the same opponent for every model, and
report mean, worst and best over seeds with a bootstrap interval (FrontierSWE-style mean@k).

    python run_fw_series.py --defender claude:sonnet --attacker claude:opus --seeds 7,8,9,10,11 --out runs/series-NAME
    python run_fw_series.py --summarize runs/series-NAME
Matches run one after another here (each takes 1.5-2 h); to go faster, run one seed per machine and summarize the folder.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def summarize(folder):
    rows = []
    for f in sorted(Path(folder).glob("seed-*/result.json")):
        r = json.loads(f.read_text())
        h = r.get("headline") or {}
        rows.append({"seed": r.get("seed"), "share": h.get("defender_share_prevented"),
                     "net_saved": h.get("net_saved"), "attacker": h.get("attacker_extra_take")})
    out = {"matches": rows}
    for key in ("share", "net_saved", "attacker"):
        xs = [x[key] for x in rows if x[key] is not None]
        if not xs:
            continue
        boots = sorted(sum(random.Random(i).choices(xs, k=len(xs))) / len(xs) for i in range(2000))
        out[key] = {"mean": sum(xs) / len(xs), "worst": min(xs), "best": max(xs),
                    "ci95": [boots[50], boots[1949]], "n": len(xs)}
    (Path(folder) / "series.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defender", default="claude:sonnet")
    ap.add_argument("--attacker", default="claude:opus")
    ap.add_argument("--seeds", default="7,8,9,10,11")
    ap.add_argument("--out")
    ap.add_argument("--summarize")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="passed to run_fw_match.py after --")
    a = ap.parse_args()
    if a.summarize:
        print(json.dumps(summarize(a.summarize), indent=1))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = [x for x in a.extra if x != "--"]
    for seed in a.seeds.split(","):
        dest = out / f"seed-{seed}"
        if (dest / "result.json").exists():
            continue
        cmd = [sys.executable, str(REPO / "run_fw_match.py"), "--defender", a.defender, "--attacker", a.attacker,
               "--seed", seed, "--out", str(dest), *extra]
        with open(out / f"seed-{seed}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    print(json.dumps(summarize(out), indent=1))


if __name__ == "__main__":
    main()
