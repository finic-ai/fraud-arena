"""The Fraud Arena game: defender agents at the institutions against attacker agents running rings, in one world.

One run of Fraud World is warmed up, then forked in memory into the game run and a reference run (every institution
on its built-in rules, every ring scripted). At each period boundary every player works at once, each in its own
sandbox (its own Unix user, workspace and login state):

- a defender sees its institution's own tables as of the boundary (its MCP tool server) and leaves `bundle/`, which
  is deployed into its institution's defender role in the game run;
- an attacker sees its ring's own outcomes (`view.json`) and leaves `plan.json`, chosen from the world's menu
  (`menu.json`), which sets its ring's plan in the game run.

Both runs then advance to the next boundary. A session that hits a usage limit waits for the reset and runs again at
the same boundary; the runs stay in memory, so nothing is checkpointed. At the horizon:

- each defender scores losses prevented at its institution minus the friction it caused, against the reference run;
- each attacker scores the money its ring took from victims in the game run, raw and against the scripted ring.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import duckdb

from . import bundle as bundles
from .config import INSTITUTIONS, EpisodeConfig, ScoreWeights
from .episode import brief_text, EpisodeState
from .runners import AgentRunner, UsageLimited
from .scoring import paired, window_score


@dataclass
class Player:
    name: str
    side: str  # "defender" | "attacker"
    target: str  # institution, or ring id
    runner: AgentRunner


@dataclass
class Turn:
    player: str
    boundary: int
    ok: bool
    seconds: float
    usage: dict
    applied: str | None
    error: str | None
    waits: int = 0
    review: bool | None = None


@dataclass
class GameConfig:
    world: Path
    seed: int
    roots: int | None = None
    partitions: int = 4
    start_day: int = 180
    period_days: int = 30
    horizon_days: int = 120
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    grayson_calls_per_period: int = 200
    max_waits: int = 20
    # After the last period the world runs on with every bundle and plan frozen and no sessions, so fraud of the
    # last weeks can settle (chargebacks, returns, disputes arrive late) before scoring.
    settle_days: int = 30

    def boundaries(self):
        end = self.start_day + self.horizon_days
        return list(range(self.start_day, end, self.period_days)) + [end]

    def for_institution(self, inst: str) -> EpisodeConfig:
        return EpisodeConfig(world=self.world, institution=inst, seed=self.seed, roots=self.roots,
                             partitions=self.partitions, start_day=self.start_day, period_days=self.period_days,
                             horizon_days=self.horizon_days, weights=self.weights,
                             grayson_calls_per_period=self.grayson_calls_per_period)


class Game:
    def __init__(self, cfg: GameConfig, engine, players: list[Player], out: Path, sandbox: Path, *, attacker_api,
                 attacker_runners=None, max_attackers: int = 4, log=print):
        """`attacker_runners(i, ring)` makes the runner of the i-th attacker; attackers are chosen after the warm-up,
        one per ring among the `max_attackers` most active ones."""
        self.cfg, self.engine, self.players = cfg, engine, list(players)
        self.attacker_runners, self.max_attackers = attacker_runners, max_attackers
        self.ring_ids: dict[int, str] = {}
        self.deployed: dict[str, dict] = {}  # role -> deployment in force in the game run
        self.good: dict[str, dict] = {}  # role -> last deployment that ran through a period
        self.role_of_decision: dict[str, str] = {}
        self.owner_of_role: dict[str, str] = {}
        self.last_tables: dict[str, Path] = {}
        self.out, self.sandbox = Path(out), Path(sandbox)
        self.attacker = attacker_api
        self.log = log
        self.turns: list[Turn] = []
        self.state = {p.name: EpisodeState() for p in players}

    def workspace(self, p: Player) -> Path:
        ws = self.sandbox / p.name / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        any_cfg = cfg.for_institution(next(iter(INSTITUTIONS)))
        base = self.engine.start(any_cfg)
        t0 = time.monotonic()
        base.run_until_day(cfg.start_day)
        self.log({"event": "warmed_up", "day": cfg.start_day, "seconds": round(time.monotonic() - t0, 1)})
        game, ref = base.fork(), base.fork()
        del base
        if self.attacker_runners:
            for i, ring in enumerate(self.attacker.active_rings(game, self.max_attackers)):
                p = Player(f"attacker-r{ring}", "attacker", ring, self.attacker_runners(i, ring))
                self.players.append(p)
                self.state[p.name] = EpisodeState()
        self.ring_loss_at_start = {}
        for p in self.players:
            if p.side == "attacker":
                self.ring_ids[p.target] = self.attacker.entity(game, p.target)
                self.ring_loss_at_start[p.target] = self.attacker.ring_loss(game, p.target)
        self.log({"event": "players", "players": [(p.name, p.runner.name, p.runner.model) for p in self.players]})
        bounds = cfg.boundaries()
        for i in range(len(bounds) - 1):
            b, nxt = bounds[i], bounds[i + 1]
            turns = self._period(game, b, first=(i == 0))
            self.turns += turns
            for t in turns:
                self.log({"event": "turn", **asdict(t)})
            t0 = time.monotonic()
            game = self._advance(game, nxt)
            ref.run_until_day(nxt)
            self.log({"event": "advanced", "day": nxt, "seconds": round(time.monotonic() - t0, 1)})
            self._write_progress()
        if cfg.settle_days:
            end = bounds[-1] + cfg.settle_days
            t0 = time.monotonic()
            game = self._advance(game, end)
            ref.run_until_day(end)
            self.log({"event": "settled", "day": end, "seconds": round(time.monotonic() - t0, 1)})
        return self._score(game, ref)

    def _advance(self, game, day: int):
        """Advances a fork of the game run, so that a deployed rule that fails at run time (say, a missing feature
        compared with a number) can't end the game: the run goes back to the boundary, that role returns to its last
        bundle that ran (or its built-in policy), the defender hears why next period, and the period runs again."""
        for _ in range(len(self.deployed) + 2):
            trial = game.fork()
            try:
                trial.run_until_day(day)
            except ValueError as e:
                decision = str(e).split(":", 1)[0].strip()
                role = self.role_of_decision.get(decision)
                if role is None or role not in self.deployed:
                    raise
                good = self.good.get(role)
                if good is not None:
                    game.deploy(role, good)
                    self.deployed[role] = good
                else:
                    game.undeploy(role)
                    del self.deployed[role]
                owner = self.owner_of_role[role]
                msg = f"your bundle failed at run time and was taken back ({'previous bundle' if good is not None else 'built-in rules'} in force): {e}"
                self.state[owner].last_error = msg
                self.log({"event": "bundle_failed", "player": owner, "role": role, "error": str(e)[:300]})
                continue
            self.good.update(self.deployed)
            return trial
        raise RuntimeError(f"the game run can't advance to day {day}")

    def _period(self, game, b: int, first: bool) -> list[Turn]:
        jobs = [(p, self._prepare(game, p, b, first)) for p in self.players]
        with ThreadPoolExecutor(len(jobs)) as pool:
            results = list(pool.map(lambda j: self._session(j[0], b, *j[1]), jobs))
        turns = []
        for (p, _), (res, waits) in zip(jobs, results):
            turns.append(self._apply(game, p, b, res, waits))
        return turns

    def _prepare(self, game, p: Player, b: int, first: bool):
        ws = self.workspace(p)
        cfg = self.cfg
        if p.side == "defender":
            ecfg = cfg.for_institution(p.target)
            st = self.state[p.name]
            data = self.engine.write_tables(game, ecfg, tag=p.name)
            since = max(cfg.start_day, b - cfg.period_days)
            outcomes = self.engine.own_outcomes(game, ecfg, since, b)
            outcomes["fraud_signals"] = self.engine.fraud_signals(data, self.last_tables.get(p.name), ecfg, b)
            outcomes["fraud_signals_note"] = ("rows new since your last turn" if p.name in self.last_tables
                                              else "rows of the last 30 days")
            self.last_tables[p.name] = data
            prompt = brief_text(ecfg, b, outcomes, st, first=first)
            (ws / ".arena" / "grayson").mkdir(parents=True, exist_ok=True)
            (ws / ".arena" / "grayson" / "calls.json").write_text(json.dumps({"used": 0}))
            self.engine.write_tables(game, ecfg, tag=p.name)
            return prompt, (lambda: self.engine.tool_server(game, ecfg, workspace=ws, tag=p.name))
        view = self.attacker.ring_view(game, p.target, max(cfg.start_day, b - cfg.period_days), b)
        (ws / "view.json").write_text(json.dumps(view, indent=2, default=str))
        (ws / "menu.json").write_text(json.dumps(self.attacker.menu(p.target), indent=2))
        st = self.state[p.name]
        bounds = cfg.boundaries()
        prompt = self.attacker.brief(ring=p.target, day=b, end_day=cfg.start_day + cfg.horizon_days,
                                     period_days=cfg.period_days, first=first, last_error=st.last_error,
                                     boundary=bounds.index(b) + 1, boundaries=len(bounds) - 1)
        return prompt, None

    def _session(self, p: Player, b: int, prompt: str, server_factory):
        waits = 0
        while True:
            try:
                if server_factory is None:
                    return p.runner.run(self.workspace(p), prompt, [], self.out / "logs" / f"{p.name}-{b}.log"), waits
                with server_factory() as server:
                    return p.runner.run(self.workspace(p), prompt, [server],
                                        self.out / "logs" / f"{p.name}-{b}.log"), waits
            except UsageLimited as e:
                waits += 1
                if waits > self.cfg.max_waits:
                    raise
                until = e.retry_at or (time.time() + 30 * 60)
                self.log({"event": "usage_limited", "player": p.name, "until": until})
                time.sleep(max(60.0, until - time.time()))

    def _apply(self, game, p: Player, b: int, res, waits) -> Turn:
        st = self.state[p.name]
        applied, err = None, None
        try:
            if p.side == "defender":
                ecfg = self.cfg.for_institution(p.target)
                bun = bundles.load(self.workspace(p) / "bundle", role=ecfg.role,
                                   decisions=self.engine.decisions_of(ecfg))
                if bun.digest != st.last_deploy:
                    role_of = self.engine.role_of(ecfg)
                    parts = bundles.split_by_role(bun.policy, role_of)
                    self.role_of_decision.update(role_of)
                    for role in [r for r in ecfg.roles if r in set(role_of.values())]:
                        self.owner_of_role[role] = p.name
                        if role in parts:
                            game.deploy(role, parts[role])
                            self.deployed[role] = parts[role]
                        elif role in self.deployed:
                            game.undeploy(role)
                            del self.deployed[role]
                    keep = self.out / "bundles" / f"{p.name}-{b}-{bun.digest[:12]}.json"
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    keep.write_text(json.dumps({"policy": bun.policy, "notes": bun.notes}, default=str))
                applied = st.last_deploy = bun.digest
            else:
                plan_path = self.workspace(p) / "plan.json"
                if not plan_path.exists():
                    raise ValueError("plan.json is missing; the ring keeps its last plan")
                plan = json.loads(plan_path.read_text())
                self.attacker.apply_plan(game, p.target, plan)
                keep = self.out / "plans" / f"{p.name}-{b}.json"
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_text(json.dumps(plan, indent=2))
                applied = f"plan@{b}"
        except Exception as e:  # refused bundles and plans go back to the player next period
            err = f"{type(e).__name__}: {e}"
        review = None
        if p.side == "defender":
            rv = self.workspace(p) / "review" / f"day-{b}.md"
            review = rv.exists() and rv.stat().st_size > 200
            if not review:
                err = (err + "; " if err else "") + f"the weekly review review/day-{b}.md is missing"
            else:
                keep = self.out / "reviews" / f"{p.name}-{b}.md"
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_text(rv.read_text()[:50000])
        st.last_error = err
        return Turn(p.name, b, res.ok, round(res.seconds, 1), res.usage, applied, err, waits, review)

    def _write_progress(self):
        tmp = self.out / "progress.json.tmp"
        tmp.write_text(json.dumps([asdict(t) for t in self.turns], indent=2, default=str))
        tmp.rename(self.out / "progress.json")

    def _score(self, game, ref) -> dict:
        cfg = self.cfg
        any_cfg = cfg.for_institution(next(iter(INSTITUTIONS)))
        start = self.engine.day_iso(any_cfg, cfg.start_day)
        end = self.engine.day_iso(any_cfg, cfg.start_day + cfg.horizon_days + cfg.settle_days)
        exports = {name: self.engine.export_scoring(run, any_cfg, self.out / "score" / name)
                   for name, run in (("game", game), ("reference", ref))}
        result = {"defenders": {}, "attackers": {}}
        for p in self.players:
            if p.side == "defender":
                ecfg = cfg.for_institution(p.target)
                g = window_score(*exports["game"], institution=p.target, role=ecfg.role, start=start, end=end,
                                 weights=cfg.weights)
                r = window_score(*exports["reference"], institution=p.target, role=ecfg.role, start=start, end=end,
                                 weights=cfg.weights)
                result["defenders"][p.name] = paired(g, r, cfg.weights).as_dict() | {"institution": p.target}
            else:
                base = self.ring_loss_at_start.get(p.target, 0.0)
                g = self.attacker.ring_loss(game, p.target) - base
                r = self.attacker.ring_loss(ref, p.target) - base
                result["attackers"][p.name] = {"ring": p.target, "stolen": g, "stolen_by_scripted_ring": r,
                                               "vs_scripted": g - r}
        tokens = {}
        for t in self.turns:
            tokens.setdefault(t.player, {})
            for k, v in t.usage.items():
                tokens[t.player][k] = tokens[t.player].get(k, 0) + v
        result |= {"config": {k: str(v) for k, v in asdict(cfg).items()},
                   "players": [{"name": p.name, "side": p.side, "target": p.target, "runner": p.runner.name,
                                "model": p.runner.model} for p in self.players],
                   "turns": [asdict(t) for t in self.turns], "tokens": tokens, "finished": time.time()}
        tmp = self.out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, default=str))
        tmp.rename(self.out / "result.json")
        return result


def ring_take(truth: Path, ring: str, start: str, end: str) -> float:
    """Money a ring's operations took from victims in the window: tainted postings caused by an instance anchored at
    the ring, excluding moves between the ring's own and its mules' accounts."""
    c = duckdb.connect()
    T = str(truth)
    row = c.execute(f"""
        select coalesce(sum(p.dirty), 0)::double
        from '{T}/postings.parquet' p
        join '{T}/instances.parquet' i on i.instance = p.cause_instance
        where i.anchor = ? and p.dirty > 0
          and p.time >= ?::timestamptz and p.time < ?::timestamptz
          and p.payer not in (select subject from '{T}/labels.parquet' where label in ('ring_account', 'money_mule'))
        """, [ring, start, end]).fetchone()
    return float(row[0])
