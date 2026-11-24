"""Run the Fraud Strategy roster on one seed: warm the seed up once, then play every defender from that warm-up.

    python run_defender_roster.py --seed 201 --out runs/fraud-strategy
    python run_defender_roster.py --seed 7 --players deepseek,luna --roots 1200 --warmup 30 --weeks 2 --jobs 2

Each defender is a model from roster.json playing through the benchmark harness (mini-SWE-agent over OpenRouter;
OPENROUTER_API_KEY must be set) against the world's own fraud: no attacking agent, rules-only policies. Results land in
OUT/<seed>/<player>/result.json with logs; OUT/<seed>/manifest.json records the settings. A player whose result
already exists is skipped, so a stopped run can be resumed. Summarize with tools/summarize_defender.py.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="runs/fraud-strategy", help="results root (OUT/<seed>/<player>/)")
    ap.add_argument("--roster", default=str(REPO / "roster.json"))
    ap.add_argument("--players", help="comma-separated roster ids (default: the whole roster)")
    ap.add_argument("--roots", type=int, default=5000, help="households")
    ap.add_argument("--warmup", type=int, default=60, help="days of history before the first week")
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=0, help="defenders played at once (default: all)")
    ap.add_argument("--cache", help="warm-up cache root (default: OUT/cache)")
    ap.add_argument("--world", help="a world file (default: the tasks tier of the pinned hub world)")
    a = ap.parse_args()

    roster = json.loads(Path(a.roster).read_text())["players"]
    want = a.players.split(",") if a.players else [p["id"] for p in roster]
    unknown = set(want) - {p["id"] for p in roster}
    if unknown:
        sys.exit(f"not in the roster: {sorted(unknown)}")
    players = [p for p in roster if p["id"] in want]
    if a.world:
        world = Path(a.world).resolve()
    else:
        sys.path.insert(0, str(REPO))
        from arena.world_source import PIN, tier
        world = tier("tasks")
    out = Path(a.out).resolve() / str(a.seed)
    cache = Path(a.cache).resolve() if a.cache else out.parent / "cache"
    out.mkdir(parents=True, exist_ok=True)
    common = ["--attacker", "none", "--defender-only", "--seed", str(a.seed), "--roots", str(a.roots),
              "--warmup", str(a.warmup), "--weeks", str(a.weeks), "--intensity", str(a.intensity),
              "--max-turns", str(a.max_turns), "--world", str(world), "--history-cache", str(cache)]
    manifest = {"seed": a.seed, "households": a.roots, "warmup_days": a.warmup, "weeks": a.weeks,
                "intensity": a.intensity, "max_turns": a.max_turns, "world": world.name,
                "players": {p["id"]: p for p in players}}
    if not a.world:
        manifest["hub"] = {"world_id": PIN["world_id"], "version": PIN["version"], "sha256": PIN["sha256"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # 1. The seed's warm-up, once.
    t = time.time()
    warm_out = out / f"_warm-{int(t)}"
    r = subprocess.run([sys.executable, str(REPO / "run_fw_match.py"), "--defender", "none", "--warm-only",
                        "--out", str(warm_out)] + common, cwd=REPO, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-2000:], flush=True)
    if r.returncode:
        sys.exit(f"warm-up failed (exit {r.returncode})")
    print(f"warm-up ready in {time.time() - t:.0f}s", flush=True)

    # 2. Every defender from that warm-up.
    def play(p):
        dest = out / p["id"]
        if (dest / "result.json").exists():
            return p["id"], "skipped (result exists)"
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        started = time.time()
        with open(out / f"{p['id']}.log", "w") as log:
            code = subprocess.run([sys.executable, str(REPO / "run_fw_match.py"), "--defender",
                                   f"openrouter:{p['model_id']}", "--out", str(dest)] + common,
                                  cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        return p["id"], f"exit {code} in {time.time() - started:.0f}s"

    with ThreadPoolExecutor(max_workers=a.jobs or len(players)) as pool:
        for pid, status in pool.map(play, players):
            print(f"{pid}: {status}", flush=True)


if __name__ == "__main__":
    main()
