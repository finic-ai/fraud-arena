"""One defender episode: warm up, checkpoint, fork candidate and reference, then a period loop.

At each boundary the agent gets a brief (the task, its institution's own outcomes last period, and the result of its
last deploy), works in its persistent workspace through the task's tool server, and leaves `bundle/`; the harness
deploys it into the candidate run only. Both runs then advance to the next boundary. At the horizon the candidate is
scored against the reference over the episode window.

The episode saves its state at every boundary (both runs' checkpoints and the period log), so a session stopped by a
subscription usage limit resumes from the same boundary with the same world.
"""
from __future__ import annotations

import json
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable, Protocol

from . import bundle as bundles
from .config import EpisodeConfig
from .runners import AgentRunner, McpServer, UsageLimited
from .scoring import paired, window_score


class RunLike(Protocol):
    day: int

    def run_until_day(self, day: int) -> None: ...
    def deploy(self, role: str, policy: dict) -> None: ...
    def checkpoint(self) -> bytes: ...


class Engine(Protocol):
    """What the episode needs from Runmere; `fraud_arena.engine.RunmereEngine` binds it to `runmere.Run`."""

    def start(self, cfg: EpisodeConfig) -> RunLike: ...
    def restore(self, cfg: EpisodeConfig, checkpoint: bytes) -> RunLike: ...
    def decisions_of(self, cfg: EpisodeConfig) -> set[str]: ...
    def tool_server(self, run: RunLike, cfg: EpisodeConfig) -> AbstractContextManager[McpServer]: ...
    def own_outcomes(self, run: RunLike, cfg: EpisodeConfig, start_day: int, end_day: int) -> dict: ...
    def export_scoring(self, run: RunLike, cfg: EpisodeConfig, out: Path) -> tuple[Path, Path]: ...
    def day_iso(self, cfg: EpisodeConfig, day: int) -> str: ...


@dataclass
class PeriodRecord:
    boundary: int
    runner: str
    model: str
    ok: bool
    seconds: float
    usage: dict
    deployed: str | None
    deploy_error: str | None
    session_error: str | None = None


@dataclass
class EpisodeState:
    next_period: int = 0
    periods: list[PeriodRecord] = field(default_factory=list)
    last_deploy: str | None = None
    last_error: str | None = None


def brief_text(cfg: EpisodeConfig, boundary: int, outcomes: dict, state: EpisodeState, *, first: bool) -> str:
    tpl = resources.files("fraud_arena").joinpath("briefs/defender.md").read_text()
    last = ("No bundle is deployed yet; the institution's built-in rules decide." if state.last_deploy is None
            else f"Your bundle {state.last_deploy[:12]} is deployed.")
    if state.last_error:
        last += f"\nYour last bundle was NOT deployed: {state.last_error}\nThe previous one stays in place."
    return tpl.format(
        institution=cfg.institution, role=cfg.role, day=boundary, period_days=cfg.period_days,
        end_day=cfg.start_day + cfg.horizon_days, phase=cfg.phase,
        first="This is the first boundary: build your stack from the history." if first
        else "Continue from your workspace and notes.",
        deploy_state=last, outcomes=json.dumps(outcomes, indent=2, sort_keys=True),
        grayson_budget=cfg.grayson_calls_per_period)


class Episode:
    def __init__(self, cfg: EpisodeConfig, engine: Engine, runner: AgentRunner | None, out: Path, *,
                 workspace: Path | None = None, on_period: Callable[[PeriodRecord], None] | None = None):
        self.cfg, self.engine, self.runner, self.out = cfg, engine, runner, Path(out)
        # The agent's workspace lives outside `out`, which holds the checkpoints and scoring exports it must not read.
        self.workspace = Path(workspace) if workspace else self.out / "workspace"
        self.on_period = on_period

    def _save(self, state: EpisodeState, cand: RunLike, ref: RunLike) -> None:
        for name, run in (("candidate", cand), ("reference", ref)):
            tmp = self.out / f"{name}.ckpt.tmp"
            tmp.write_bytes(run.checkpoint())
            tmp.rename(self.out / f"{name}.ckpt")
        tmp = self.out / "state.json.tmp"
        tmp.write_text(json.dumps(asdict(state), indent=2))
        tmp.rename(self.out / "state.json")

    def _load(self) -> tuple[EpisodeState, RunLike, RunLike] | None:
        if not (self.out / "state.json").exists():
            return None
        raw = json.loads((self.out / "state.json").read_text())
        state = EpisodeState(raw["next_period"], [PeriodRecord(**p) for p in raw["periods"]],
                             raw["last_deploy"], raw["last_error"])
        cand = self.engine.restore(self.cfg, (self.out / "candidate.ckpt").read_bytes())
        ref = self.engine.restore(self.cfg, (self.out / "reference.ckpt").read_bytes())
        return state, cand, ref

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg
        loaded = self._load()
        if loaded is None:
            base = self.engine.start(cfg)
            base.run_until_day(cfg.start_day)
            ck = base.checkpoint()
            cand, ref = self.engine.restore(cfg, ck), self.engine.restore(cfg, ck)
            state = EpisodeState()
            self._save(state, cand, ref)
        else:
            state, cand, ref = loaded
        decisions = self.engine.decisions_of(cfg)
        bounds = cfg.boundaries()
        for i in range(state.next_period, len(bounds) - 1):
            b, nxt = bounds[i], bounds[i + 1]
            rec = self._period(cfg, cand, state, decisions, b, i == 0)
            state.periods.append(rec)
            if self.on_period:
                self.on_period(rec)
            cand.run_until_day(nxt)
            ref.run_until_day(nxt)
            state.next_period = i + 1
            self._save(state, cand, ref)
        return self._score(cand, ref, state)

    def _period(self, cfg, cand, state, decisions, b, first) -> PeriodRecord:
        if self.runner is None:
            return PeriodRecord(b, "reference", "-", True, 0.0, {}, None, None)
        outcomes = self.engine.own_outcomes(cand, cfg, max(cfg.start_day, b - cfg.period_days), b)
        prompt = brief_text(cfg, b, outcomes, state, first=first)
        (self.workspace / ".arena").mkdir(parents=True, exist_ok=True)
        (self.workspace / ".arena" / "grayson").mkdir(exist_ok=True)
        (self.workspace / ".arena" / "grayson" / "calls.json").write_text(json.dumps({"used": 0}))
        with self.engine.tool_server(cand, cfg) as server:
            try:
                res = self.runner.run(self.workspace, prompt, [server], self.out / "logs" / f"period-{b}.log")
            except UsageLimited:
                raise
        deployed, err = None, None
        try:
            bun = bundles.load(self.workspace / "bundle", role=cfg.role, decisions=decisions)
            if bun.digest != state.last_deploy:
                cand.deploy(cfg.role, bun.policy)
                keep = self.out / "bundles" / f"{b}-{bun.digest[:12]}.json"
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_text(json.dumps({"policy": bun.policy, "notes": bun.notes}, default=str))
            deployed = bun.digest
            state.last_deploy = bun.digest
        except bundles.BundleError as e:
            err = str(e)
        except Exception as e:  # the engine refusing a deploy: its message goes back to the agent
            err = f"{type(e).__name__}: {e}"
        state.last_error = err
        return PeriodRecord(b, res.runner, res.model, res.ok, round(res.seconds, 1), res.usage, deployed, err,
                            res.error)

    def _score(self, cand, ref, state) -> dict:
        cfg = self.cfg
        start, end = self.engine.day_iso(cfg, cfg.start_day), self.engine.day_iso(cfg, cfg.start_day + cfg.horizon_days)
        scores = {}
        for name, run in (("candidate", cand), ("reference", ref)):
            truth, owners = self.engine.export_scoring(run, cfg, self.out / "score" / name)
            scores[name] = window_score(truth, owners, institution=cfg.institution, role=cfg.role, start=start,
                                        end=end, weights=cfg.weights)
        p = paired(scores["candidate"], scores["reference"], cfg.weights)
        tokens = {}
        for rec in state.periods:
            for k, v in rec.usage.items():
                tokens[k] = tokens.get(k, 0) + v
        result = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "score": p.as_dict(),
                  "periods": [asdict(r) for r in state.periods], "tokens": tokens, "finished": time.time()}
        tmp = self.out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, default=str))
        tmp.rename(self.out / "result.json")
        return result
