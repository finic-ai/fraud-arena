"""The attacker side of an adversarial episode: at each period boundary, each ring with an agent operator gets one
session, and its plan is set on the ring in the run before the run goes on. Rings without an operator keep their
scripted behaviour in the same run.

The panel keeps, per ring, the last boundary's snapshot (for the period's view) and every plan, in `state.json` under
its folder, so an episode stopped by a usage limit resumes at the same boundary. A plan is set in the run that the
operator attacks (the candidate's); a paired reference run replays the same plans with `replay`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from . import menu, view
from .operator import BoundaryResult, OperatorConfig, make_runner, run_boundary


class AttackerPanel:
    def __init__(self, operators: list[OperatorConfig], *, world: Path, root: Path, epoch_ms: int):
        rings = [o.ring for o in operators]
        if len(set(rings)) != len(rings):
            raise ValueError("one operator per ring")
        self.operators = sorted(operators, key=lambda o: o.ring)
        self.world = Path(world)
        self.root = Path(root)
        self.epoch_ms = epoch_ms
        self.runners = {o.ring: make_runner(o) for o in self.operators}
        self.state = self._load()

    def _path(self) -> Path:
        return self.root / "state.json"

    def _load(self) -> dict:
        p = self._path()
        return json.loads(p.read_text()) if p.exists() else {"snapshots": {}, "plans": []}

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path().with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, sort_keys=True))
        os.replace(tmp, self._path())

    def boundary(self, run, *, day: int, period_days: int, boundary: int, boundaries: int) -> list[BoundaryResult]:
        """Runs each operator's session at the boundary the run has reached and sets the plans. A ring whose plan for
        this day is already recorded (a resumed episode) gets it again without a new session."""
        out = []
        for cfg in self.operators:
            key = str(cfg.ring)
            done = [p for p in self.state["plans"] if p["ring"] == cfg.ring and p["day"] == day]
            if done:
                menu.apply_plan(run, cfg.ring, done[-1]["plan"])
                continue
            now = view.snapshot(run, cfg.ring)
            before = self.state["snapshots"].get(key)
            start = before["day"] if before else day - period_days
            v = view.ring_view(now, before and before["snap"], ring=cfg.ring, day=day, period_start=start,
                               epoch_ms=self.epoch_ms)
            result = run_boundary(cfg, self.runners[cfg.ring], workspace=self.root / f"R{cfg.ring}" / "workspace",
                                  logs=self.root / "logs", world=self.world, view=v, day=day,
                                  period_days=period_days, boundary=boundary, boundaries=boundaries)
            menu.apply_plan(run, cfg.ring, result.plan)
            self.state["snapshots"][key] = {"day": day, "snap": now}
            self.state["plans"].append({"ring": cfg.ring, "day": day, "plan": result.plan,
                                        "result": asdict(result)})
            self._save()
            out.append(result)
        return out

    def replay(self, run, day: int) -> None:
        """Sets the plans recorded for `day` in another run of the same episode (the paired reference run)."""
        for p in self.state["plans"]:
            if p["day"] == day:
                menu.apply_plan(run, p["ring"], p["plan"])
