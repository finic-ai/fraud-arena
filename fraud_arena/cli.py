"""fraud-arena pilot: the static pilot, one institution and seed, each defender against the same world.

    fraud-arena pilot --world worlds/fraud-world/world.yaml --out /home/ubuntu/pilot --sandbox /srv/arena \
        --defender claude-code:sonnet --defender codex:gpt-6.1-sol --roots 2280 --start-day 180 \
        --period-days 30 --horizon-days 120 --seed 1000

Each defender runs as its own player user (arena-p1, arena-p2, ...; sandbox/setup-box.sh). `--credentials
subscription` (the default) runs on CLAUDE_CODE_OAUTH_TOKEN and the Codex login; `--credentials api-key
--model-budget-tokens N` sends every model call through the model gateway, which holds ANTHROPIC_API_KEY and
OPENAI_API_KEY, so no key, token or login reaches a player.
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


def _users(a) -> list[str | None]:
    return [f"arena-p{n}" for n in range(1, 13)] if a.player_users else [None] * 12


CODEX_USER, CODEX_HOME = "arena-codex", "/home/arena-codex/.codex"


def _gateway(a, specs: list[str | None]):
    """With `--credentials api-key`, the run's model gateway, for the models these runner:model specs name; None
    with subscriptions or when no player calls a model."""
    from .gateway import RUNNER_PROVIDER, Gateway

    models = {s.split(":", 1)[1] for s in specs if s and s.split(":", 1)[0] in RUNNER_PROVIDER}
    if a.credentials != "api-key" or not models:
        return None
    if not a.model_budget_tokens:
        raise SystemExit("a run with model players needs --model-budget-tokens: the token budget the owner approved")
    return Gateway(Path(a.out) / "gateway" / "usage.jsonl", models=models, budget_tokens=a.model_budget_tokens)


def _credentials(a, kind: str, name: str, user: str | None, gateway) -> tuple[dict[str, str], str | None, str | None]:
    """A session's model access: (environment, Unix user, proxy).

    - `api-key`: the gateway's address and the caller's own gateway token; the keys stay with the harness.
    - `subscription`: Claude Code gets CLAUDE_CODE_OAUTH_TOKEN, through a file in its own folder, never on a command
      line, and reaches the providers through the proxy. Every Codex session runs as `arena-codex` with its one
      ChatGPT login (copies of a login invalidate each other), so Codex players share that user and its folders.
    """
    if a.credentials == "api-key":
        return (gateway.env(name, kind) if gateway is not None else {}), user, None
    env = {}
    if kind == "claude-code":
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise SystemExit("--credentials subscription needs CLAUDE_CODE_OAUTH_TOKEN for Claude Code players")
        env["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    if kind == "codex" and user is not None:
        user, env["CODEX_HOME"] = CODEX_USER, CODEX_HOME
    return env, user, a.proxy


def _runner(spec: str, a, name: str, user: str | None, gateway, broker):
    kind, model = spec.split(":", 1)
    if kind == "baseline":
        return RUNNERS[kind](model)
    env, user, proxy = _credentials(a, kind, name, user, gateway)
    return RUNNERS[kind](model, extra_env=broker.env(name) | env, as_user=user, proxy=proxy,
                         timeout_s=a.session_timeout)


def pilot(a) -> int:
    from .engine import RunmereEngine
    from .sandbox import private_dir

    from .grayson import Broker

    jobs = []
    users = iter(_users(a))
    gateway = _gateway(a, [s for s in a.defender if s != "reference"])
    brokers = []
    for spec in a.defender:
        cfg = EpisodeConfig(world=Path(a.world), institution=a.institution, seed=a.seed, roots=a.roots,
                            partitions=a.partitions, start_day=a.start_day, period_days=a.period_days,
                            horizon_days=a.horizon_days, grayson_calls_per_period=a.grayson_budget)
        name = spec.replace(":", "-")
        # Grayson's key stays with the grader: each defender has its own broker, its budget per period.
        broker = Broker(Path(a.out) / name / "grayson", budget=a.grayson_budget,
                        enabled=bool(os.environ.get("GRAYSON_API_KEY")))
        brokers.append(broker)
        runner = None if spec == "reference" else _runner(spec, a, name, next(users), gateway, broker)
        sandbox = private_dir(Path(a.sandbox) / name, getattr(runner, "as_user", None))
        eng = RunmereEngine(sandbox)
        ws = sandbox / f"{cfg.institution}-s{cfg.seed}" / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        def on_period(r, n=name, b=broker):
            b.reset_period()
            print(json.dumps({"episode": n, **r.__dict__}, default=str), flush=True)

        ep = Episode(cfg, eng, runner, Path(a.out) / name, workspace=ws, on_period=on_period)
        jobs.append(Job(name, spec.split(":")[0], ep.run))
    try:
        results = Queue(Path(a.out) / "queue.jsonl", parallel=a.parallel).run(jobs)
    finally:
        for b in brokers:
            b.close()
        if gateway is not None:
            gateway.close()
    summary = {k: (v["score"] | {"tokens": v["tokens"]} if isinstance(v, dict) else f"{type(v).__name__}: {v}")
               for k, v in results.items()}
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _player_runner(spec: str, a, user: str | None, name: str, gateway):
    """Grayson reaches a player only through the game's broker (FA_GRAYSON_URL/TOKEN, set by the game): its API key
    stays with the grader."""
    kind, model = spec.split(":", 1)
    if kind == "baseline":
        return RUNNERS[kind](model)
    env, user, proxy = _credentials(a, kind, name, user, gateway)
    return RUNNERS[kind](model, extra_env=env, as_user=user, proxy=proxy, timeout_s=a.session_timeout)


def _text_generator_spec(a) -> str | None:
    """With attacker-authored text the generator is the attackers' model, for their rows and every other row alike."""
    if a.attacker_text == "on":
        if not a.attackers or a.attackers.split(":", 1)[0] == "baseline":
            raise SystemExit("--attacker-text on needs --attackers runner:model (the text generator is their model)")
        if a.text_generator not in ("none", a.attackers):
            raise SystemExit(f"--attacker-text on generates with the attackers' model ({a.attackers}); "
                             f"drop --text-generator {a.text_generator}")
        return a.attackers
    return None if a.text_generator == "none" else a.text_generator


def _textgen(cfg, a, log, gateway):
    """The text generator runs grader side, as the harness user (Codex on a subscription as its shared login)."""
    from .textgen import TextGen, make_generator

    if not cfg.text_generator:
        return None
    kind = cfg.text_generator.split(":", 1)[0]
    env, user, proxy = _credentials(a, kind, "textgen", None, gateway)
    if a.credentials == "subscription" and kind == "codex" and a.player_users:
        user, env["CODEX_HOME"] = CODEX_USER, CODEX_HOME
    gen = make_generator(cfg.text_generator, as_user=user, proxy=proxy, extra_env=env,
                         workdir=Path(a.out) / "textgen-work")
    return TextGen(gen, cfg.text_cache, seed=cfg.seed, world=cfg.world, workers=a.text_workers, log=log,
                   match=str(Path(a.out).resolve()))


def game(a) -> int:
    from .attacker.game_api import Attackers
    from .config import INSTITUTIONS
    from .engine import RunmereEngine
    from .game import Game, GameConfig, Player

    cfg = GameConfig(world=Path(a.world), seed=a.seed, roots=a.roots, partitions=a.partitions,
                     start_day=a.start_day, period_days=a.period_days, horizon_days=a.horizon_days,
                     grayson_calls_per_period=a.grayson_budget, grayson=a.grayson == "on",
                     settle_days=a.settle_days, workflow_timeout_s=a.workflow_timeout,
                     workflows=a.workflows == "on", outside_mode=a.outside_mode,
                     text_generator=_text_generator_spec(a), attacker_text=a.attacker_text == "on",
                     text_cache=Path(a.text_cache).expanduser(), replay=a.replay == "on" or a.watch)
    engine = RunmereEngine(Path(a.sandbox))
    if (a.credentials == "subscription" and a.player_users and a.defenders.startswith("codex:")
            and (a.attackers or "").startswith("codex:")):
        raise SystemExit("with --credentials subscription, Codex players share one user and its folders, so Codex "
                         "can't play both sides: an attacker's view is truth. Use --credentials api-key.")
    gateway = _gateway(a, [a.defenders, a.attackers, cfg.text_generator])
    users = iter(_users(a))
    insts = a.institution or list(INSTITUTIONS)
    players = [Player(f"defender-{i}", "defender", i, _player_runner(a.defenders, a, next(users), f"defender-{i}",
                                                                      gateway)) for i in insts]
    rest = list(users)
    attackers = Attackers(Path(a.world), engine.day_iso(cfg.for_institution(insts[0]), 0))
    log = lambda rec: print(json.dumps(rec, default=str), flush=True)
    g = Game(cfg, engine, players, Path(a.out), Path(a.sandbox), attacker_api=attackers,
             attacker_runners=(lambda i, ring: _player_runner(a.attackers, a, rest[i], f"attacker-R{ring}", gateway))
             if a.attackers else None,
             max_attackers=min(a.max_attackers, len(rest)), log=log, textgen=_textgen(cfg, a, log, gateway))
    watch = _watch(Path(a.out)) if a.watch else None
    try:
        result = g.run()
    finally:
        if watch is not None:
            watch.close()
        if gateway is not None:
            gateway.close()
    print(json.dumps({"defenders": result["defenders"], "attackers": result["attackers"],
                      "tokens": result["tokens"]}, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    # A player shares its own folder with the harness through its own group, and nobody else may read it.
    os.umask(0o007)
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
    p.add_argument("--player-users", action=argparse.BooleanOptionalAction, default=True,
                   help="run each defender as its own Unix user (arena-pN); --no-player-users for a local run")
    p.add_argument("--credentials", choices=["subscription", "api-key"], default="subscription",
                   help="subscription: CLAUDE_CODE_OAUTH_TOKEN and the Codex login, through --proxy; api-key: the "
                        "harness's model gateway holds ANTHROPIC_API_KEY / OPENAI_API_KEY and no player sees a key")
    p.add_argument("--proxy", default="http://127.0.0.1:3128", help="the allowlisting proxy (subscriptions)")
    p.add_argument("--model-budget-tokens", type=int, help="api-key: the run's approved model budget, all players")
    p.add_argument("--session-timeout", type=int, default=2 * 3600)
    q = sub.add_parser("game", help="defenders at the institutions vs attackers running rings, one world")
    q.add_argument("--world", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--sandbox", required=True)
    q.add_argument("--defenders", required=True, help="runner:model for every defender")
    q.add_argument("--attackers", help="runner:model for every attacker (none: rings stay scripted)")
    q.add_argument("--max-attackers", type=int, default=4)
    q.add_argument("--institution", action="append", help="default: every institution with defender decisions")
    q.add_argument("--seed", type=int, default=1000)
    q.add_argument("--roots", type=int)
    q.add_argument("--partitions", type=int, default=4)
    q.add_argument("--start-day", type=int, default=180)
    q.add_argument("--period-days", type=int, default=30)
    q.add_argument("--horizon-days", type=int, default=120)
    q.add_argument("--grayson-budget", type=int, default=50000, help="Grayson calls per defender per period")
    q.add_argument("--grayson", choices=["on", "off"], default="on", help="off: the no-Grayson game")
    q.add_argument("--workflows", choices=["on", "off"], default="on",
                   help="off: no workflow.py (an engine without external decisions at event gates)")
    q.add_argument("--outside-mode", choices=["batch", "inline"], default="batch",
                   help="inline: the engine calls the harness at the gate (opt in; needs Run.set_outside)")
    q.add_argument("--workflow-timeout", type=float, default=600.0, help="seconds per batch of a workflow.py")
    q.add_argument("--text-generator", default="none",
                   help="claude-code:sonnet|claude-code:opus|codex:<model>|none: rewrites the defenders' rich text")
    q.add_argument("--text-cache", default="~/.cache/fraud-arena/textgen", help="kept across boundaries and games")
    q.add_argument("--text-workers", type=int, default=4)
    q.add_argument("--attacker-text", choices=["on", "off"], default="off",
                   help="attackers brief their schemes' texts; every text is generated with the attackers' model")
    q.add_argument("--settle-days", type=int, default=30, help="frozen run-on after the last period, before scoring")
    q.add_argument("--player-users", action=argparse.BooleanOptionalAction, default=True,
                   help="run each player as its own Unix user (arena-pN); --no-player-users for a local run")
    q.add_argument("--credentials", choices=["subscription", "api-key"], default="subscription",
                   help="subscription: CLAUDE_CODE_OAUTH_TOKEN and the Codex login, through --proxy; api-key: the "
                        "harness's model gateway holds ANTHROPIC_API_KEY / OPENAI_API_KEY and no player sees a key")
    q.add_argument("--proxy", default="http://127.0.0.1:3128", help="the allowlisting proxy (subscriptions)")
    q.add_argument("--model-budget-tokens", type=int, help="api-key: the run's approved model budget, all players")
    q.add_argument("--session-timeout", type=int, default=2 * 3600)
    q.add_argument("--replay", choices=["on", "off"], default="on", help="write <out>/replay/ as the match runs")
    q.add_argument("--watch", action="store_true",
                   help="serve the viewer on 127.0.0.1 and follow the match live (stops with the game)")
    r = sub.add_parser("replay", help="(re)build <out>/replay/ from a match's outputs")
    r.add_argument("out")
    r.add_argument("--game-truth", help="a truth folder for the game run (default: <out>/score/game/truth)")
    r.add_argument("--reference-truth", help="default: <out>/score/reference/truth")
    r.add_argument("--no-scores", action="store_true", help="skip the exact scores at each boundary")
    w = sub.add_parser("watch", help="serve the viewer for a running or finished match on 127.0.0.1")
    w.add_argument("out")
    w.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)
    return {"pilot": pilot, "game": game, "replay": replay, "watch": watch}[a.cmd](a)


def _watch(out: Path, port: int = 0):
    import sys

    from .watch import Watch

    w = Watch(out, port=port)
    print(f"watch the match at {w.url}", file=sys.stderr, flush=True)
    return w


def replay(a) -> int:
    from .replay import rebuild

    out = Path(a.out)
    stats = rebuild(out, game_truth=Path(a.game_truth) if a.game_truth else None,
                    ref_truth=Path(a.reference_truth) if a.reference_truth else None, scores=not a.no_scores)
    print(json.dumps({"replay": str(out / "replay" / "replay.jsonl"), **stats}, indent=2))
    return 0


def watch(a) -> int:
    import time

    w = _watch(Path(a.out), a.port)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
