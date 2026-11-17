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
import shutil
import threading
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
from .scoring import paired, ring_score, window_score


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
    grayson_calls_per_period: int = 50000
    # False is the no-Grayson game: the broker refuses every call and the brief says Grayson isn't available.
    grayson: bool = True
    workflow_timeout_s: float = 600.0
    # False when the engine can't hand event-gate cases out yet: workflow.py isn't offered or deployed.
    workflows: bool = True
    max_waits: int = 20
    # After the last period the world runs on with every bundle and plan frozen and no sessions, so fraud of the
    # last weeks can settle (chargebacks, returns, disputes arrive late) before scoring.
    settle_days: int = 30
    # The defender works its institution's review queue and acts on accounts (Fraud World's manual review).
    manual_review: bool = True
    # Rewrites the defenders' rich text cells with a generator (fraud_arena.textgen): claude-code:<model>,
    # codex:<model>, or None for the world's own template text.
    text_generator: str | None = None
    text_cache: Path | None = None
    # Attackers write a brief per scheme (`briefs/<scheme>.md`) and the rows their rings caused are written from it,
    # with the generator above (the attackers' model); every other row from the world's brief.
    attacker_text: bool = False
    # Writes `<out>/replay/` as the match runs (fraud_arena.replay); the runs then step one simulated day at a time.
    replay: bool = True

    def boundaries(self):
        end = self.start_day + self.horizon_days
        return list(range(self.start_day, end, self.period_days)) + [end]

    def for_institution(self, inst: str) -> EpisodeConfig:
        return EpisodeConfig(world=self.world, institution=inst, seed=self.seed, roots=self.roots,
                             partitions=self.partitions, start_day=self.start_day, period_days=self.period_days,
                             horizon_days=self.horizon_days, weights=self.weights,
                             grayson_calls_per_period=self.grayson_calls_per_period if self.grayson else 0,
                             workflows=self.workflows)


class Game:
    def __init__(self, cfg: GameConfig, engine, players: list[Player], out: Path, sandbox: Path, *, attacker_api,
                 attacker_runners=None, max_attackers: int = 4, log=print, textgen=None):
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
        self.grayson_gates: dict[str, tuple[str, dict]] = {}  # decision -> (defender, question)
        self.broker = None
        self.workflows: dict[str, object] = {}  # defender -> its running Workflow
        self.workflow_dec: dict[str, str] = {}  # decision -> defender whose workflow decides it
        self.wf_snap: dict[str, Path | None] = {}  # defender -> frozen bundle its workflow runs from
        self.good_wf_snap: dict[str, Path | None] = {}
        self.workflow_stats: dict[str, dict] = {}
        self.queue_stats: dict[str, dict] = {}
        self.integrations: dict[str, object] = {}  # defender -> its integrations Resolver
        self._kinds: dict[str, dict[str, str]] | None = None
        self.good_integrations: dict[str, object] = {}
        self._stats_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._cur_day = 0
        self._run_now = None
        self._text_lock = threading.Lock()
        self.queue_grayson: dict[str, tuple[str, dict]] = {}  # defender -> (institution, grayson config)
        self.textgen = textgen if textgen is not None else _textgen(cfg, Path(out))
        self.briefs: dict[int, dict[str, str]] = {}  # ring -> scheme -> its attacker's latest brief
        self.brief_notes: dict[str, list[str]] = {}
        self.text_causes = None
        self.text_truth: Path | None = None
        self.text_folders: dict[str, Path] = {}
        self.text_stats: dict = {}
        from .review import Reviewer
        self.reviewer = Reviewer()
        self.review_on: set[str] = set()
        self.typology: dict[str, list] = {}
        self.review_feedback: dict[str, dict] = {}
        self.out, self.sandbox = Path(out), Path(sandbox)
        self.attacker = attacker_api
        self.log = log
        self.turns: list[Turn] = []
        self.state = {p.name: EpisodeState() for p in players}
        self.replay = None
        self.grayson_asked: dict[str, int] = {}

    def workspace(self, p: Player) -> Path:
        ws = self.sandbox / p.name / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        from .grayson import Broker
        self.broker = Broker(self.out / "grayson", budget=cfg.grayson_calls_per_period, enabled=cfg.grayson)
        try:
            return self._run()
        finally:
            for wf in self.workflows.values():
                wf.close()
            self.broker.close()

    def _run(self) -> dict:
        cfg = self.cfg
        if self.textgen is not None and Path(cfg.world).exists():
            from .textgen import uncovered_text_columns
            missing = uncovered_text_columns(Path(cfg.world))
            if missing:
                raise ValueError(f"free-text columns with no text generation spec (they would show the world's "
                                 f"template text): {missing}")
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
        if cfg.manual_review:
            for p in self.players:
                if p.side == "defender":
                    try:
                        self.reviewer.hand_desk_to_defender(game, p.target)
                        self.review_on.add(p.name)
                    except Exception as e:  # a world without review desks: detection only
                        self.log({"event": "no_manual_review", "player": p.name, "why": str(e)[:200]})
        for p in self.players:
            if p.side == "defender" and hasattr(p.runner, "extra_env"):
                p.runner.extra_env = dict(p.runner.extra_env) | self.broker.env(p.name)
        self.log({"event": "players", "players": [(p.name, p.runner.name, p.runner.model) for p in self.players]})
        bounds = cfg.boundaries()
        if cfg.replay:
            from .replay import Recorder
            self.replay = Recorder(self.out, log=self.log)
            self.replay.start(config=asdict(cfg), players=self._player_records(), boundaries=bounds,
                              settle_days=cfg.settle_days, epoch_iso=self.engine.day_iso(any_cfg, 0), game=game,
                              ref=ref)
        for i in range(len(bounds) - 1):
            b, nxt = bounds[i], bounds[i + 1]
            self.broker.reset_period()
            turns = self._period(game, b, first=(i == 0))
            self.turns += turns
            for t in turns:
                self.log({"event": "turn", **asdict(t)})
            if self.replay is not None:
                self.replay.boundary(b, i, [asdict(t) for t in turns])
            t0 = time.monotonic()
            game = self._advance(game, nxt)
            self._run_reference(ref, nxt)
            self.log({"event": "advanced", "day": nxt, "seconds": round(time.monotonic() - t0, 1)})
            self._write_progress()
        if cfg.settle_days:
            end = bounds[-1] + cfg.settle_days
            t0 = time.monotonic()
            game = self._advance(game, end)
            self._run_reference(ref, end)
            self.log({"event": "settled", "day": end, "seconds": round(time.monotonic() - t0, 1)})
        result = self._score(game, ref)
        if self.replay is not None:
            self.replay.finish(weights=cfg.weights)
        return result

    def _player_records(self) -> list[dict]:
        return [{"name": p.name, "side": p.side, "target": p.target, "runner": p.runner.name, "model": p.runner.model}
                for p in self.players]

    def _harness_counts(self) -> dict:
        """Each defender's harness-side work so far (cumulative; the replay records it per day)."""
        out = {}
        for p in self.players:
            if p.side != "defender":
                continue
            h = {}
            wf = self.workflow_stats.get(p.name)
            if wf:
                h["workflow"] = {"cases": wf["cases"], "actions": dict(wf["actions"])}
            q = self.queue_stats.get(p.name)
            if q:
                h["queue"] = dict(q)
            if self.grayson_asked.get(p.name):
                h["grayson"] = self.grayson_asked[p.name]
            if h:
                out[p.name] = h
        return out

    def _replay_day(self, run, d: int, which: str = "game") -> None:
        if self.replay is not None:
            self.replay.day(run, d, which, self._harness_counts() if which == "game" else None)

    def _run_reference(self, ref, day: int) -> None:
        if self.replay is None:
            ref.run_until_day(day)
            return
        d = int(ref.day)
        while d < day:
            d += 1
            ref.run_until_day(d)
            self._replay_day(ref, d - 1, "reference")

    def _advance(self, game, day: int):
        """Advances a fork of the game run, so that a deployed rule that fails at run time (say, a missing feature
        compared with a number) can't end the game: the run goes back to the boundary, that role returns to its last
        bundle that ran (or its built-in policy), the defender hears why next period, and the period runs again."""
        for _ in range(len(self.deployed) + 2):
            trial = game.fork()
            if hasattr(trial, "set_outside"):
                trial.set_outside(self._decide_case)
            try:
                self._run_with_grayson(trial, day)
            except ValueError as e:
                if self.replay is not None:
                    self.replay.rollback(int(game.day), self._harness_counts())
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
                self._set_workflow(owner, self.good_wf_snap.get(owner))
                if owner in self.good_integrations:
                    self.integrations[owner] = self.good_integrations[owner]
                else:
                    self.integrations.pop(owner, None)
                msg = f"your bundle failed at run time and was taken back ({'previous bundle' if good is not None else 'built-in rules'} in force): {e}"
                self.state[owner].last_error = msg
                self.log({"event": "bundle_failed", "player": owner, "role": role, "error": str(e)[:300]})
                continue
            self.good.update(self.deployed)
            self.good_wf_snap.update(self.wf_snap)
            self.good_integrations = dict(self.integrations)
            return trial
        raise RuntimeError(f"the game run can't advance to day {day}")

    def _run_with_grayson(self, run, day: int) -> None:
        """Runs to `day`. When cases wait for outside decisions (a defender's workflow, or Grayson at a gate), the run
        goes one simulated day at a time and each time it stops, the waiting cases go out in one batch per decision and
        their answers go back before it continues."""
        outside = bool(self.grayson_gates or self.workflow_dec or self.integrations) and hasattr(run, "pending_external")
        if not outside and not self.queue_grayson and not self.workflows:
            if self.replay is None:
                run.run_until_day(day)
                return
            d = int(run.day)
            while d < day:
                d += 1
                run.run_until_day(d)
                self._replay_day(run, d - 1)
            return
        d = int(run.day)
        while d < day:
            d += 1
            self._day_stats = {"stops": 0, "workflow": 0, "grayson_asked": 0, "answered": 0, "defaulted": 0}
            # Review queues are worked every hour (held payments are often due within hours); otherwise whole days.
            hourly = bool(self.queue_grayson) or any(w.reviews for w in self.workflows.values())
            marks = [d - 1 + h / 24 for h in range(1, 24)] + [d] if hourly else [d]
            for t in marks:
                # The engine stops before `t` whenever instant decisions (event gates) wait; each must be answered
                # before it goes on, and nothing else may touch the run until it reaches `t`.
                while run.run_until_day(min(t, day)):
                    self._day_stats["stops"] += 1
                    self._answer_outside(run, d)
                if outside:
                    self._answer_outside(run, d)  # cases waiting at state gates and after events
                if hourly:
                    self._work_queues(run, t)
            if outside and self._day_stats["stops"] + self._day_stats["answered"]:
                self.log({"event": "outside_decisions", "day": d, **self._day_stats})
            self._replay_day(run, d - 1)
            if (self.workflows or self.integrations) and d < day:
                self._write_live(run)

    def _write_live(self, run) -> None:
        """Defenders with a workflow or integrations get their tables (and the public data) as of each day's end,
        with the rich text rewritten as at a boundary (rows written before come from the cache): integrations read
        the grader's copy, a workflow reads `live/` in its workspace."""
        owners = sorted(set(self.workflows) | set(self.integrations))
        if not owners:
            return
        if self.textgen is not None and self.cfg.attacker_text:
            self._row_causes(run, None)

        def one(owner):
            p = next(p for p in self.players if p.name == owner)
            ecfg = self.cfg.for_institution(p.target)
            rewrite = None
            if self.textgen is not None:
                def rewrite(folder, inst=p.target):
                    stats = self.textgen.rewrite(folder, inst, causes=self.text_causes, briefs=self.briefs)
                    with self._text_lock:
                        self._count_text(stats)
            try:
                grader = self.out / "live" / owner
                self.engine.write_live(run, ecfg, grader, rewrite=rewrite)
                if owner in self.integrations:
                    self.integrations[owner].data.load(grader)
                if owner in self.workflows:
                    dest = self.workspace(p) / "live"
                    tmp = dest.with_name("live.tmp")
                    shutil.rmtree(tmp, ignore_errors=True)
                    shutil.copytree(grader, tmp)
                    shutil.rmtree(dest, ignore_errors=True)
                    tmp.rename(dest)
            except Exception as e:
                self.log({"event": "live_tables_failed", "player": owner, "error": str(e)[:200]})
        with ThreadPoolExecutor(len(owners)) as pool:
            list(pool.map(one, owners))

    def _set_integrations(self, p, ints: dict) -> None:
        """Puts a defender's integrations in force, reading its tables as of this boundary."""
        from . import integrations as integ
        if not ints:
            self.integrations.pop(p.name, None)
            return
        data = integ.Data()
        tables = self.last_tables.get(p.name)
        if tables is not None and Path(tables).exists():
            data.load(Path(tables))
        self.integrations[p.name] = integ.Resolver(
            ints, data, self.broker.client(p.name), case_view=self._case_view,
            observer=self.cfg.for_institution(p.target).observer,
            log=lambda rows, o=p.name: self._log_live(o, float(self._cur_day), rows))

    def _answer_outside(self, run, d: int) -> int:
        waiting = run.pending_external()
        if not waiting:
            return 0
        from .bundle import grayson_action
        from .workflow import WorkflowError
        self._cur_day, self._run_now = d, run
        feature_answers = self._answer_features([w for w in waiting if w.get("needs")])
        waiting = [w for w in waiting if not w.get("needs")]
        answers = []
        by_wf: dict[tuple[str, str], list] = {}
        by_owner: dict[str, list] = {}
        for w in waiting:
            avail = list(w.get("available") or w["actions"])
            owner = self.workflow_dec.get(w["decision"])
            if owner is not None:
                by_wf.setdefault((owner, w["decision"]), []).append((w, avail))
                continue
            if w["decision"] not in self.grayson_gates:
                continue
            owner, q = self.grayson_gates[w["decision"]]
            if "questions" in q:
                questions = {n: {"type": "choice", "instructions": v["instructions"], "criteria": v["criteria"]}
                             for n, v in q["questions"].items()}
            else:
                crit = {a: m for a, m in q["criteria"].items() if a in avail}
                if not crit:
                    continue
                questions = {"decision": {"type": "choice", "instructions": q["instructions"], "criteria": crit}}
            req = {"data": self._case_view(w) | {"decision": w["decision"]}, "questions": questions}
            by_owner.setdefault(owner, []).append((w, q, avail, req))
        def run_owner(owner):
            # One defender's batches in order (its workflow is one process); defenders run side by side, so one
            # waiting on Grayson doesn't hold the others.
            res = []
            for (o, decision), items in sorted(by_wf.items()):
                if o != owner:
                    continue
                t0 = time.monotonic()
                try:
                    acts = self.workflows[owner].decide(decision, [w["features"] | {"_actions": avail}
                                                                   for w, avail in items])
                except WorkflowError as e:
                    return ValueError(f"{decision}: {e}")
                res.append((decision, items, acts, time.monotonic() - t0))
            return res
        owners = sorted({o for o, _ in by_wf})
        if len(owners) > 1:
            with ThreadPoolExecutor(len(owners)) as pool:
                results = list(pool.map(run_owner, owners))
        else:
            results = [run_owner(o) for o in owners]
        for owner, res in zip(owners, results):
            if isinstance(res, ValueError):
                raise res
            for decision, items, acts, secs in res:
                answers += [(w["key"], a) for (w, _), a in zip(items, acts)]
                st = self.workflow_stats.setdefault(owner, {"cases": 0, "batches": 0, "seconds": 0.0, "actions": {}})
                st["cases"] += len(items)
                st["batches"] += 1
                st["seconds"] += secs
                for a in acts:
                    st["actions"][f"{decision}:{a}"] = st["actions"].get(f"{decision}:{a}", 0) + 1
        asked = 0
        for owner, items in by_owner.items():
            client = self.broker.client(owner)
            if client is None:
                continue
            outs = client.ask_many([r for *_, r in items])
            asked += len(items)
            self.grayson_asked[owner] = self.grayson_asked.get(owner, 0) + len(items)
            logged = []
            for (w, q, avail, req), o in zip(items, outs):
                act = grayson_action(q, w["features"], o.get("answers") or {}, avail)
                logged.append({"where": w["decision"], "data": req["data"], "answers": o.get("answers"),
                               "error": o.get("error"), "action": act})
                if act:
                    answers.append((w["key"], act))
            self._log_live(owner, float(run.day), logged)
        # An instant decision has no timeout: one nobody could answer (Grayson unreachable or undecided) takes the
        # gate's default, as a real gate fails open or closed to its default when its decision service is down.
        done = {k for k, _ in answers}
        defaulted = [(w["key"], w["default_action"]) for w in waiting
                     if w.get("gate") == "event" and w["key"] not in done and w.get("default_action")]
        answers += defaulted
        if answers or feature_answers:
            run.answer_external({k: a for k, a in answers} | feature_answers)
        st = getattr(self, "_day_stats", None)
        if st is not None:
            st["workflow"] += sum(len(v) for v in by_wf.values())
            st["grayson_asked"] += asked
            st["answered"] += len(answers) + len(feature_answers)
            st["defaulted"] += len(defaulted)
        return len(answers)

    def _decide_case(self, case: dict, view=None):
        """The engine's synchronous outside decider: called at the gate, mid-step, possibly from several threads at
        once. Returns the needed features (integrations) or an action (a workflow, a Grayson gate config, or the
        gate's default)."""
        from .bundle import grayson_action
        from .workflow import WorkflowError
        if case.get("needs"):
            return self._answer_features([case], run=view)[case["key"]]
        avail = list(case.get("available") or case["actions"])
        owner = self.workflow_dec.get(case["decision"])
        if owner is not None:
            wf = self.workflows[owner]
            t0 = time.monotonic()
            try:
                with wf.lock:
                    act = wf.decide(case["decision"], [case["features"] | {"_actions": avail}])[0]
            except WorkflowError as e:
                raise ValueError(f"{case['decision']}: {e}") from None
            with self._stats_lock:
                st = self.workflow_stats.setdefault(owner, {"cases": 0, "batches": 0, "seconds": 0.0, "actions": {}})
                st["cases"] += 1
                st["batches"] += 1
                st["seconds"] += time.monotonic() - t0
                k = f"{case['decision']}:{act}"
                st["actions"][k] = st["actions"].get(k, 0) + 1
            return act
        if case["decision"] in self.grayson_gates:
            owner, q = self.grayson_gates[case["decision"]]
            client = self.broker.client(owner)
            if client is not None:
                if "questions" in q:
                    questions = {n: {"type": "choice", "instructions": v["instructions"], "criteria": v["criteria"]}
                                 for n, v in q["questions"].items()}
                else:
                    questions = {"decision": {"type": "choice", "instructions": q["instructions"],
                                              "criteria": {a: m for a, m in q["criteria"].items() if a in avail}}}
                req = {"data": self._case_view(case) | {"decision": case["decision"]}, "questions": questions}
                o = client.ask_many([req])[0]
                act = grayson_action(q, case["features"], o.get("answers") or {}, avail)
                self._log_live(owner, float(self._cur_day), [{"where": case["decision"], "data": req["data"],
                                                             "answers": o.get("answers"), "error": o.get("error"),
                                                             "action": act}])
                if act:
                    return act
        return case.get("default_action") or avail[0]

    # Case features holding text the harness rewrites with a model before players see it (textgen RICH columns:
    # applications' free text, claim statements, return comments, beneficiary-change notes). In a waiting case they
    # are still the world's template text, which gives fraud away, so outside code doesn't get them; short routine
    # text (P2P and wire notes, names, labels) is shown as the tables show it.
    RICH_CASE_TEXT = frozenset({"occupation", "employer", "source_of_funds", "account_purpose", "statement", "comment",
                                "beneficiary_change_note"})

    def _feature_kinds(self) -> dict[str, dict[str, str]]:
        if self._kinds is None:
            self._kinds = {}
            for inst in INSTITUTIONS:
                try:
                    for dp in self.engine.decision_points(self.cfg.for_institution(inst)):
                        self._kinds[dp["decision"]] = {f["name"]: str(f.get("kind", "")) for f in dp["features"]}
                except Exception:
                    pass
        return self._kinds

    def _case_view(self, w: dict) -> dict:
        """What outside code shows a decision model about a case: its features, without free text (the world's text
        reaches players only in its model-written form), plus its display identifiers."""
        feats = {k: v for k, v in (w.get("features") or {}).items() if k not in self.RICH_CASE_TEXT}
        return feats | {k: v for k, v in (w.get("identifiers") or {}).items() if v is not None}

    def _answer_features(self, cases: list[dict], run=None) -> dict[str, dict]:
        """Values for the outside features cases wait for, from their defenders' integrations; a feature no
        integration gives (or one that failed) gets no information: uniform options, or 0."""
        if not cases:
            return {}
        by_owner: dict[str, list] = {}
        for c in cases:
            owner = self.owner_of_role.get(self.role_of_decision.get(c["decision"], c.get("role")), None)
            by_owner.setdefault(owner, []).append(c)
        out: dict[str, dict] = {}
        for owner, cs in by_owner.items():
            res = self.integrations.get(owner)
            got = res.answer(cs, run=run if run is not None else self._run_now) if res is not None else {}
            for c in cs:
                ans = got.get(c["key"], {})
                for n in c["needs"]:
                    if n not in ans:
                        i = res.integrations.get(n) if res is not None else None
                        ans[n] = ({o: 1.0 / len(i.options) for o in i.options} if i is not None and
                                  i.source == "grayson" else 0.0)
                out[c["key"]] = ans
        return out

    def _set_workflow(self, owner: str, snap: Path | None) -> None:
        """Puts `owner`'s workflow from the frozen bundle `snap` in force (None: no workflow)."""
        from .workflow import Workflow
        if self.wf_snap.get(owner) == snap and (snap is None or owner in self.workflows):
            return
        old = self.workflows.pop(owner, None)
        if old is not None:
            old.close()
        for dcs in [d for d, o in self.workflow_dec.items() if o == owner]:
            del self.workflow_dec[dcs]
        self.wf_snap[owner] = snap
        if snap is None:
            return
        p = next(p for p in self.players if p.name == owner)
        wf = Workflow(owner, snap, snap, self.workspace(p), as_user=getattr(p.runner, "as_user", None),
                      env=self.broker.env(owner), timeout_s=self.cfg.workflow_timeout_s)
        self.workflows[owner] = wf
        for dcs in wf.decisions:
            self.workflow_dec[dcs] = owner

    def _queue_cases(self, run, inst: str, kinds) -> list[dict]:
        """The institution's pending review-queue cases, each as its `REVIEW_QUEUE` row shows it (every column the
        world declares, except the observer's own display ids), plus `_id` for the harness."""
        ecfg = self.cfg.for_institution(inst)
        cols = self.engine.queue_columns(ecfg)
        direct = sorted({p[0] for _, p, _ in cols} | {"kind"})
        inst_id = {r["name"]: r["id"] for r in run.rows("institution", ["name"])}.get(inst)
        rows = [r for r in run.rows("review_case", direct + ["institution", "manual_verdict"],
                                    where_attr="manual_verdict", equals="pending")
                if r.get("institution") == inst_id and (not kinds or r.get("kind") in kinds)]
        if not rows:
            return []
        looked: dict[tuple[str, str], dict] = {}
        world = self.engine.spec(ecfg)
        rc = ((world.get("world") or world).get("entities") or {}).get("review_case", {}).get("attributes", {})
        for _, path, _ in cols:
            if len(path) == 2 and (path[0], path[1]) not in looked:
                etype = rc[path[0]]["entity"]
                looked[(path[0], path[1])] = {e["id"]: e.get(path[1]) for e in run.rows(etype, [path[1]])}
        cases = []
        for r in rows:
            c = {"_id": r["id"]}
            for col, path, upper in cols:
                v = r.get(path[0]) if len(path) == 1 else looked[(path[0], path[1])].get(r.get(path[0]))
                if v is None:
                    continue
                c[col] = str(v).upper() if upper and isinstance(v, str) else v
            cases.append(json.loads(json.dumps(c, default=str)))
        return cases

    def _log_live(self, owner: str, day: float, rows: list[dict]) -> None:
        """What live Grayson was sent and answered (grader side; each defender gets its own at the next boundary)."""
        if not rows:
            return
        f = self.out / "grayson-live" / f"{owner}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock, open(f, "a") as fh:
            for r in rows:
                fh.write(json.dumps({"day": round(day, 4)} | r, default=str) + "\n")

    def _work_queues(self, run, day: float) -> None:
        """Each defender's review queue, worked by Grayson (a `review_queue.grayson` config) or by its workflow's
        `review(cases)`: verdicts are set as the cases' `manual_verdict`, as a human review's would be (they don't
        use the human team's capacity)."""
        from .bundle import grayson_action
        from .workflow import WorkflowError
        for owner, (inst, q) in list(self.queue_grayson.items()):
            cases = self._queue_cases(run, inst, q["kinds"])
            if not cases:
                continue
            client = self.broker.client(owner)
            if client is None:
                continue
            if "questions" in q:
                questions = {n: {"type": "choice", "instructions": v["instructions"], "criteria": v["criteria"]}
                             for n, v in q["questions"].items()}
            else:
                questions = {"decision": {"type": "choice", "instructions": q["instructions"],
                                          "criteria": {a: m for a, m in q["criteria"].items()
                                                       if a in ("release", "reject")}}}
            reqs = [{"data": {k: v for k, v in c.items() if k != "_id"}, "questions": questions} for c in cases]
            outs = client.ask_many(reqs)
            self.grayson_asked[owner] = self.grayson_asked.get(owner, 0) + len(reqs)
            done, logged = 0, []
            for c, req, o in zip(cases, reqs, outs):
                act = grayson_action(q, req["data"], o.get("answers") or {}, ["release", "reject"])
                logged.append({"where": "review_queue", "case": c.get("CASE_NO"), "data": req["data"],
                               "answers": o.get("answers"), "error": o.get("error"), "verdict": act})
                if act:
                    run.intervene("review_case", c["_id"], "manual_verdict", act)
                    done += 1
            self._log_live(owner, day, logged)
            self._queue_stats(owner, len(cases), done)
        for owner, wf in list(self.workflows.items()):
            if not wf.reviews:
                continue
            p = next(p for p in self.players if p.name == owner)
            cases = self._queue_cases(run, p.target, wf.review_kinds)
            if not cases:
                continue
            try:
                acts = wf.review([{k: v for k, v in c.items() if k != "_id"} for c in cases])
            except WorkflowError as e:
                raise ValueError(f"review_queue: {e}") from None
            done = 0
            for c, act in zip(cases, acts):
                if act:
                    run.intervene("review_case", c["_id"], "manual_verdict", act)
                    done += 1
            self._queue_stats(owner, len(cases), done)

    def _queue_stats(self, owner: str, seen: int, decided: int) -> None:
        st = self.queue_stats.setdefault(owner, {"looks": 0, "case_looks": 0, "decided": 0})
        st["looks"] += 1
        st["case_looks"] += seen
        st["decided"] += decided

    def _period(self, game, b: int, first: bool) -> list[Turn]:
        if self.textgen is not None:
            self._row_causes(game, b)
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
            if self.textgen is not None and data is not None:
                stats = self.textgen.rewrite(data, p.target, causes=self.text_causes, briefs=self.briefs)
                self.log({"event": "textgen", "player": p.name, "boundary": b, **stats})
                self._count_text(stats)
                self.text_folders[p.target] = Path(data)
            since = max(cfg.start_day, b - cfg.period_days)
            outcomes = self.engine.own_outcomes(game, ecfg, since, b)
            outcomes["fraud_signals"] = self.engine.fraud_signals(data, self.last_tables.get(p.name), ecfg, b)
            outcomes["fraud_signals_note"] = ("rows new since your last turn" if p.name in self.last_tables
                                              else "rows of the last 30 days")
            if p.name in self.review_on:
                from .review import REVIEWS_PER_DAY
                outcomes["manual_review"] = {
                    "capacity_this_period": REVIEWS_PER_DAY[p.target] * cfg.period_days,
                    "last_period": self.review_feedback.get(p.name, {"applied": 0, "refused": []})}
            self.last_tables[p.name] = data
            live_log = self.out / "grayson-live" / f"{p.name}.jsonl"
            if live_log.exists():
                (ws / ".arena").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(live_log, ws / ".arena" / "grayson-live.jsonl")
            prompt = brief_text(ecfg, b, outcomes, st, first=first, families=self.engine.fraud_families(ecfg))
            self.engine.write_tables(game, ecfg, tag=p.name)
            return prompt, (lambda: self.engine.tool_server(game, ecfg, workspace=ws, tag=p.name))
        view = self.attacker.ring_view(game, p.target, max(cfg.start_day, b - cfg.period_days), b)
        (ws / "view.json").write_text(json.dumps(view, indent=2, default=str))
        (ws / "menu.json").write_text(json.dumps(self.attacker.menu(p.target), indent=2))
        st = self.state[p.name]
        bounds = cfg.boundaries()
        prompt = self.attacker.brief(ring=p.target, day=b, end_day=cfg.start_day + cfg.horizon_days,
                                     period_days=cfg.period_days, first=first, last_error=st.last_error,
                                     boundary=bounds.index(b) + 1, boundaries=len(bounds) - 1,
                                     attacker_text=cfg.attacker_text, text_notes=self.brief_notes.get(p.name))
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
                bdir = self.workspace(p) / "bundle"
                decisions = self.engine.decisions_of(ecfg)
                bun = bundles.load(bdir, role=ecfg.role, decisions=decisions)
                digest = bundles.digest_files(bdir, bun.digest)
                if digest != st.last_deploy:
                    role_of = self.engine.role_of(ecfg)
                    policy = json.loads(json.dumps(bun.policy))
                    policy.pop("review_queue", None)
                    gray = bundles.take_grayson(policy)
                    from . import integrations as integ
                    if "outside" in policy:
                        raise bundles.BundleError("declare outside features through `integrations`")
                    try:
                        ints = integ.parse(policy)
                    except integ.IntegrationError as e:
                        raise bundles.BundleError(str(e)) from None
                    if any(i.source == "grayson" for i in ints.values()) and not self.cfg.grayson:
                        raise bundles.BundleError("Grayson isn't available in this game")
                    integ.declare(policy, ints)
                    if (gray or (bun.policy.get("review_queue") or {}).get("grayson")) and not self.cfg.grayson:
                        raise bundles.BundleError("Grayson isn't available in this game")
                    snap, claimed = None, []
                    if (bdir / "workflow.py").exists() and not self.cfg.workflows:
                        raise bundles.BundleError("decision workflows (workflow.py) aren't available in this game")
                    if (bdir / "workflow.py").exists():
                        from .workflow import Workflow, WorkflowError
                        snap = self.sandbox / p.name / "deployed" / f"{b}-{digest[:12]}"
                        try:
                            probe = Workflow(p.name, bdir, snap, self.workspace(p),
                                             as_user=getattr(p.runner, "as_user", None),
                                             env=self.broker.env(p.name), timeout_s=self.cfg.workflow_timeout_s)
                        except WorkflowError as e:
                            raise bundles.BundleError(str(e)) from None
                        claimed, reviews = probe.decisions, probe.reviews
                        probe.close()
                        bad = [d for d in claimed if d not in decisions]
                        if reviews and (bun.policy.get("review_queue") or {}).get("grayson"):
                            raise bundles.BundleError("the review queue is worked by workflow.py's review() and by a "
                                                      "review_queue grayson config; pick one")
                        if reviews and p.name not in self.review_on:
                            raise bundles.BundleError("review() needs manual review, which isn't on in this game")
                        if bad or not (claimed or reviews):
                            raise bundles.BundleError(
                                f"workflow.py DECISIONS {bad or claimed} aren't this role's decision points: "
                                f"{sorted(decisions)}")
                        both = sorted(set(claimed) & set(gray))
                        if both:
                            raise bundles.BundleError(f"{both}: decided by workflow.py and by a grayson gate; pick one")
                        ruled = {r["decision"] for r in policy.get("rules", [])}
                        policy["rules"] = list(policy.get("rules", [])) + [
                            bundles.external_rule(d) for d in claimed if d not in ruled]
                    sent_out = {r["decision"] for r in policy.get("rules", []) if r.get("action") == "external"}
                    orphans = sorted(sent_out - set(claimed) - set(gray))
                    if orphans:
                        raise bundles.BundleError(
                            f"rules send {orphans} to `external`, but neither workflow.py (DECISIONS) nor a grayson "
                            f"config decides them")
                    for d in sent_out:
                        policy.setdefault("external", {}).setdefault(d, {})
                    for decision in [d for d, o in self.grayson_gates.items() if o[0] == p.name]:
                        del self.grayson_gates[decision]
                    for decision, q in gray.items():
                        self.grayson_gates[decision] = (p.name, q)
                    rq = (bun.policy.get("review_queue") or {}).get("grayson")
                    if rq and p.name in self.review_on:
                        q = bundles.take_grayson({"external": {"queue": {"grayson": json.loads(json.dumps(rq))}}})["queue"]
                        self.queue_grayson[p.name] = (p.target, q | {"kinds": list(
                            (bun.policy.get("review_queue") or {}).get("kinds") or [])})
                    else:
                        self.queue_grayson.pop(p.name, None)
                    parts = bundles.split_by_role(policy, role_of)
                    self.role_of_decision.update(role_of)
                    for role in [r for r in ecfg.roles if r in set(role_of.values())]:
                        self.owner_of_role[role] = p.name
                        if role in parts:
                            self.deployed[role] = bundles.deploy_external(game, role, parts[role])
                        elif role in self.deployed:
                            game.undeploy(role)
                            del self.deployed[role]
                    self._set_workflow(p.name, snap)
                    self._set_integrations(p, ints)
                    if snap is not None:
                        live = self.last_tables.get(p.name)
                        dest = self.workspace(p) / "live"
                        if live is not None and Path(live).exists():
                            # At the boundary, live/ is the boundary's own (already rewritten) tables.
                            tmp = dest.with_name("live.tmp")
                            shutil.rmtree(tmp, ignore_errors=True)
                            shutil.copytree(live, tmp)
                            shutil.rmtree(dest, ignore_errors=True)
                            tmp.rename(dest)
                        else:
                            self.engine.write_live(game, ecfg, dest)
                    keep = self.out / "bundles" / f"{p.name}-{b}-{digest[:12]}.json"
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    keep.write_text(json.dumps({"policy": bun.policy, "notes": bun.notes,
                                                "workflow": (bdir / "workflow.py").read_text()[:200000]
                                                if snap else None}, default=str))
                applied = st.last_deploy = digest
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
        if p.side == "attacker" and self.cfg.attacker_text:
            self._take_briefs(p, b)
        if p.side == "defender" and p.name in self.review_on:
            from .review import REVIEWS_PER_DAY
            rpath = self.workspace(p) / "bundle" / "reviews.json"
            rres = self.reviewer.apply(game, p.target, rpath, REVIEWS_PER_DAY[p.target] * self.cfg.period_days)
            self.typology.setdefault(p.name, []).extend(rres.typology)
            self.review_feedback[p.name] = rres.summary()
            if rpath.exists():
                keep = self.out / "reviews-applied" / f"{p.name}-{b}.json"
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_text(rpath.read_text())
                rpath.rename(self.workspace(p) / "review" / f"applied-day-{b}.json") if (self.workspace(p) / "review").exists() else rpath.unlink()
            self.log({"event": "manual_review", "player": p.name, "boundary": b, **rres.summary()})
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
                if p.name in self.typology:
                    from .review import grade_typology
                    result["defenders"][p.name]["typology"] = grade_typology(
                        exports["game"][0], self.typology[p.name], self.engine.fraud_families(any_cfg))
            else:
                base = self.ring_loss_at_start.get(p.target, {"observed": 0.0, "background": 0.0})
                gl, rl = self.attacker.ring_loss(game, p.target), self.attacker.ring_loss(ref, p.target)
                # Scored like the defenders are charged, on the observed institutions only: the losses and the mule
                # flow (at its weight) of the postings the ring's operations caused. Background is displacement.
                rid = self.ring_ids[p.target]
                gs = ring_score(*exports["game"], ring=rid, institutions=list(INSTITUTIONS), start=start, end=end,
                                weights=cfg.weights, number=p.target)
                rs = ring_score(*exports["reference"], ring=rid, institutions=list(INSTITUTIONS), start=start,
                                end=end, weights=cfg.weights, number=p.target)
                result["attackers"][p.name] = {
                    "ring": p.target, "score": gs["score"], "score_of_scripted_ring": rs["score"],
                    "vs_scripted": gs["score"] - rs["score"], "game": gs, "reference": rs,
                    "incident_losses_observed": gl["observed"] - base["observed"],
                    "incident_losses_background": gl["background"] - base["background"],
                    "scripted_incident_losses_background": rl["background"] - base["background"]}
        tokens = {}
        for t in self.turns:
            tokens.setdefault(t.player, {})
            for k, v in t.usage.items():
                tokens[t.player][k] = tokens[t.player].get(k, 0) + v
        result |= {"config": {k: str(v) for k, v in asdict(cfg).items()},
                   "players": self._player_records(),
                   "turns": [asdict(t) for t in self.turns], "tokens": tokens, "text": self._text_summary(),
                   "decisioning": self._decisioning_summary(),
                   "finished": time.time()}
        tmp = self.out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, default=str))
        tmp.rename(self.out / "result.json")
        return result


    def _decisioning_summary(self) -> dict:
        """How each defender decided: its workflow's cases and actions, and its Grayson calls (paid and cached)."""
        out = {}
        for p in self.players:
            if p.side != "defender":
                continue
            paid = cached = 0
            log = self.out / "grayson" / p.name / "log.jsonl"
            if log.exists():
                for line in log.read_text().splitlines():
                    if line.strip():
                        cached_ = json.loads(line).get("cached")
                        paid, cached = paid + (not cached_), cached + bool(cached_)
            wf = self.workflow_stats.get(p.name)
            out[p.name] = {"workflow": wf and wf | {"seconds": round(wf["seconds"], 1)},
                           "review_queue": self.queue_stats.get(p.name),
                           "integrations": (self.integrations[p.name].stats if p.name in self.integrations else None),
                           "grayson_calls": paid,
                           "grayson_cached": cached}
        return out

    def _row_causes(self, game, b: int):
        """The run's truth as of the boundary, grader side: which rows the attacker rings caused (for their briefs),
        and at the last boundary the labels for the style check."""
        bounds = self.cfg.boundaries()
        write = getattr(self.engine, "write_truth", None)
        self.text_causes = None
        if write is None or not (self.cfg.attacker_text or (b is not None and b == bounds[-2])):
            return
        from .textgen import RowCauses

        truth = self.out / "textgen-truth"
        if truth.exists():
            shutil.rmtree(truth)
        write(game, truth)
        self.text_truth = truth
        rings = sorted(p.target for p in self.players if p.side == "attacker")
        self.text_causes = RowCauses(truth, rings if self.cfg.attacker_text else [])

    def _take_briefs(self, p: Player, b: int):
        from .textgen import read_briefs, scheme_texts

        allowed = set(scheme_texts(self.cfg.world, p.target))
        briefs, notes = read_briefs(self.workspace(p) / "briefs", allowed)
        self.briefs[p.target] = briefs
        self.brief_notes[p.name] = notes
        if briefs:
            keep = self.out / "briefs" / f"{p.name}-{b}"
            keep.mkdir(parents=True, exist_ok=True)
            for scheme, text in briefs.items():
                (keep / f"{scheme}.md").write_text(text)
        self.log({"event": "attacker_briefs", "player": p.name, "boundary": b, "schemes": sorted(briefs),
                  "refused": notes})

    def _count_text(self, stats: dict):
        agg = self.text_stats.setdefault(stats.get("generator", "?"), {"rows": 0, "generated": 0, "cache_hits": 0,
                                                                     "fallbacks": 0, "from_attacker_briefs": 0,
                                                                     "refusals": {}, "tokens": {}, "seconds": 0.0})
        for t in (stats.get("tables") or {}).values():
            for k in ("generated", "cache_hits", "fallbacks", "from_attacker_briefs"):
                agg[k] += t.get(k, 0)
            agg["rows"] += t.get("rows", 0)
            for why, n in (t.get("refusals") or {}).items():
                agg["refusals"][why] = agg["refusals"].get(why, 0) + n
        for k, v in (stats.get("tokens") or {}).items():
            agg["tokens"][k] = agg["tokens"].get(k, 0) + v
        agg["seconds"] += stats.get("seconds", 0.0)

    def _text_summary(self) -> dict | None:
        if self.textgen is None:
            return None
        out = {"attacker_text": self.cfg.attacker_text, "by_model": self.text_stats}
        if self.text_truth is not None and self.text_folders:
            from . import textcheck

            folders = dict(self.text_folders)
            pub = next(iter(self.text_folders.values())) / "public"
            if pub.is_dir():
                folders["public"] = pub
            ring_rows = self.text_causes.ring_rows() if self.text_causes is not None and self.cfg.attacker_text \
                else None
            try:
                out["textcheck"] = textcheck.check(self.text_truth, folders, ring_rows)
            except Exception as e:  # the check must never cost a match its result
                out["textcheck"] = {"error": f"{type(e).__name__}: {e}"}
        return out


def _textgen(cfg: GameConfig, out: Path):
    from .textgen import TextGen, make_generator

    gen = make_generator(cfg.text_generator, workdir=out / "textgen-work")
    if gen is None:
        return None
    return TextGen(gen, cfg.text_cache or out / "textgen-cache", seed=cfg.seed, world=cfg.world,
                   match=str(out.resolve()))


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
