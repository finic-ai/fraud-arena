"""The attacker side as `Game` uses it: one object per game, over the menu, view and operator modules."""
from __future__ import annotations

import json

from datetime import datetime
from pathlib import Path

from . import menu as menus
from . import operator
from . import view as views


class Attackers:
    def __init__(self, world: Path, epoch_iso: str):
        self.world = Path(world)
        self.epoch_ms = int(datetime.fromisoformat(epoch_iso.replace("Z", "+00:00")).timestamp() * 1000)
        self.before: dict[int, dict] = {}
        self.schemas: dict[int, dict] = {}

    def rings(self) -> list[int]:
        return sorted(menus.RINGS)

    def schema(self, ring: int) -> dict:
        if ring not in self.schemas:
            self.schemas[ring] = menus.plan_schema(self.world, ring)
        return self.schemas[ring]

    def entity(self, run, ring: int) -> str:
        rows = run.rows("ring", ["number"], where_attr="number", equals=ring)
        if len(rows) != 1:
            raise ValueError(f"no ring R{ring}")
        return rows[0]["id"]

    def active_rings(self, run, limit: int) -> list[int]:
        """Rings with the most incidents and accounts so far: the ones a plan changes most."""
        scored = []
        for r in self.rings():
            try:
                snap = views.snapshot(run, r)
            except ValueError:
                continue
            activity = len(snap["incidents"]) + len(snap["accounts"])
            if activity:
                scored.append((activity, r))
        return [r for _, r in sorted(scored, reverse=True)[:limit]]

    def ring_view(self, run, ring: int, start_day: int, end_day: int) -> dict:
        now = views.snapshot(run, ring)
        v = views.ring_view(now, self.before.get(ring), ring=ring, day=end_day, period_start=start_day,
                            epoch_ms=self.epoch_ms)
        views.check_view(v)
        self.before[ring] = now
        return v

    def menu(self, ring: int) -> dict:
        return self.schema(ring)

    def brief(self, *, ring: int, day: int, end_day: int, period_days: int, first: bool, last_error: str | None,
              boundary: int, boundaries: int) -> str:
        text = operator.brief(ring=ring, schema=self.schema(ring), day=day, period_days=period_days,
                              boundary=boundary, boundaries=boundaries)
        text += (f"\n\nThe ring's capacity is limited: a scheme runs at most {self.MAX_SCHEME_MULTIPLIER:g}x its "
                 "usual rate (volume times the scheme's tactic weight); weights above that are scaled down.")
        if last_error:
            text += f"\n\nYour last plan was NOT applied: {last_error}\nThe ring kept its previous plan."
        return text

    # A ring can't grow its operation much in a period: its members, identities, mules and devices are what they
    # are. Until the world ties a plan to the ring's resources, the combined multiplier of a scheme (volume times its
    # tactic weight) is capped; the tactic weights of a plan above the cap are scaled down (owner, 2026-11-16).
    MAX_SCHEME_MULTIPLIER = 3.0

    def cap(self, plan: dict) -> dict:
        plan = json.loads(json.dumps(plan))
        vol = float(plan.get("volume", 1.0))
        if vol > self.MAX_SCHEME_MULTIPLIER:
            plan["volume"] = vol = self.MAX_SCHEME_MULTIPLIER
        tactics = plan.get("tactics") or {}
        for t, w in list(tactics.items()):
            if vol * float(w) > self.MAX_SCHEME_MULTIPLIER:
                tactics[t] = round(self.MAX_SCHEME_MULTIPLIER / vol, 4) if vol > 0 else w
        return plan

    def apply_plan(self, run, ring: int, plan: dict) -> list:
        errors = menus.validate_plan(plan, self.schema(ring))
        if errors:
            raise ValueError("; ".join(errors[:10]))
        return menus.apply_plan(run, ring, self.cap(plan), world=self.world)

    def ring_loss(self, run, ring: int) -> dict[str, float]:
        """What the ring has taken so far, by the world's own account (its incidents' losses, grader side), split into
        the observed institutions where defenders play (what the attacker is scored on) and the background ones
        (reported as displacement, never scored)."""
        from ..config import INSTITUTIONS
        names = {r["id"]: r["name"] for r in run.rows("institution", ["name"])}
        rows = run.rows("incident", ["loss", "institution"], where_attr="ring", equals=self.entity(run, ring))
        out = {"observed": 0.0, "background": 0.0}
        for r in rows:
            name = names.get(r.get("institution"), r.get("institution"))
            out["observed" if name in INSTITUTIONS else "background"] += float(r.get("loss") or 0)
        return out
