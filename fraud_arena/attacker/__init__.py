"""Agent attackers (M6): ring operators that set a fraud ring's strategy once per period, from Fraud World's menu.

- `menu`: the plan menu from the world's declared interventions, the plan's JSON Schema, validation, and the
  interventions a plan sets.
- `view`: what a ring knows at a boundary, filtered from truth on the grader's side.
- `operator`: one headless agent session per ring per boundary, through `fraud_arena.runners`.
- `panel`: the operators of an episode, boundary by boundary, with their state for resuming.
"""
from .menu import RINGS, apply_plan, interventions, plan_menu, plan_schema, validate_plan
from .operator import BoundaryResult, OperatorConfig, make_runner, run_boundary
from .panel import AttackerPanel
from .view import active_rings, snapshot, view_of


def ring_view(run, ring: int, start_day: int, end_day: int, **kw) -> dict:
    """What ring R`ring` knows at the boundary on `end_day` about the period since `start_day` (`view.view_of`)."""
    return view_of(run, ring, start_day, end_day, **kw)


__all__ = ["AttackerPanel", "RINGS", "BoundaryResult", "OperatorConfig", "active_rings", "apply_plan",
           "interventions", "make_runner", "plan_menu", "plan_schema", "ring_view", "run_boundary", "snapshot",
           "validate_plan"]
