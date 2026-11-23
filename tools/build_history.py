"""Build a seed's 1v1 warm-up once and save it to a history cache (run_fw_match.py --history-cache), so every
defender on that seed starts from the identical history without simulating it again."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.fw.match import Match
ap = argparse.ArgumentParser()
ap.add_argument("--world", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--roots", type=int, default=5000); ap.add_argument("--warmup", type=int, default=60)
ap.add_argument("--intensity", type=float, default=3.0); ap.add_argument("--cache", required=True)
a = ap.parse_args()
Match(Path(a.cache) / "work", world=Path(a.world), seed=a.seed, roots=a.roots, warmup_days=a.warmup,
      intensity=a.intensity, history_cache=a.cache)
print("history cached in", a.cache)
