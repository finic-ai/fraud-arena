"""fraud-arena pilot: the static pilot, one institution and seed, each defender against the same world.

    fraud-arena pilot --world worlds/fraud-world/world.yaml --out /home/ubuntu/pilot --sandbox /srv/arena \
        --defender claude-code:sonnet --defender codex:gpt-6.1-sol --roots 2280 --start-day 180 \
        --period-days 30 --horizon-days 120 --seed 1000 --agent-user agent --proxy http://127.0.0.1:3128
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import EpisodeConfig
from .episode import Episode
from .queue import Job, Queue
from .runners import RUNNERS


def _runner(spec: str, a):
    kind, model = spec.split(":", 1)
    env = {}
    if kind == "claude-code" and os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    env |= {"FA_GRAYSON_STATE": ".arena/grayson", "FA_GRAYSON_BUDGET": str(a.grayson_budget)}
    if os.environ.get("GRAYSON_API_KEY"):
        env["GRAYSON_API_KEY"] = os.environ["GRAYSON_API_KEY"]
    return RUNNERS[kind](model, extra_env=env, as_user=a.agent_user, proxy=a.proxy, timeout_s=a.session_timeout)


def pilot(a) -> int:
    from .engine import RunmereEngine

    jobs = []
    for spec in a.defender:
        cfg = EpisodeConfig(world=Path(a.world), institution=a.institution, seed=a.seed, roots=a.roots,
                            partitions=a.partitions, start_day=a.start_day, period_days=a.period_days,
                            horizon_days=a.horizon_days, grayson_calls_per_period=a.grayson_budget)
        name = spec.replace(":", "-")
        sandbox = Path(a.sandbox) / name
        eng = RunmereEngine(sandbox, agent_user=a.agent_user)
        ws = sandbox / f"{cfg.institution}-s{cfg.seed}" / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        ep = Episode(cfg, eng, None if spec == "reference" else _runner(spec, a), Path(a.out) / name, workspace=ws,
                     on_period=lambda r, n=name: print(json.dumps({"episode": n, **r.__dict__}, default=str),
                                                       flush=True))
        jobs.append(Job(name, spec.split(":")[0], ep.run))
    results = Queue(Path(a.out) / "queue.jsonl", parallel=a.parallel).run(jobs)
    summary = {k: (v["score"] | {"tokens": v["tokens"]} if isinstance(v, dict) else f"{type(v).__name__}: {v}")
               for k, v in results.items()}
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    # The agent user shares the workspace through the `arena` group; what the harness writes there must stay
    # writable for it.
    os.umask(0o002)
    ap = argparse.ArgumentParser(prog="fraud-arena")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--world", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sandbox", required=True)
    p.add_argument("--defender", action="append", required=True, help="runner:model, or 'reference'")
    p.add_argument("--institution", default="brindlemark")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--roots", type=int)
    p.add_argument("--partitions", type=int, default=4)
    p.add_argument("--start-day", type=int, default=180)
    p.add_argument("--period-days", type=int, default=30)
    p.add_argument("--horizon-days", type=int, default=120)
    p.add_argument("--grayson-budget", type=int, default=200)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--agent-user")
    p.add_argument("--proxy")
    p.add_argument("--session-timeout", type=int, default=2 * 3600)
    a = ap.parse_args(argv)
    return pilot(a)


if __name__ == "__main__":
    raise SystemExit(main())
