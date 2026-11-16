"""Agent attackers (M6): ring operators that set a fraud ring's strategy once per period, from Fraud World's menu.

- `menu`: the plan menu from the world's declared interventions, the plan's JSON Schema, validation, and the
  interventions a plan sets.
- `view`: what a ring knows at a boundary, filtered from truth on the grader's side.
- `operator`: one headless agent session per ring per boundary, through `fraud_arena.runners`.
"""
from .menu import RINGS, apply_plan, interventions, plan_schema, validate_plan
from .operator import BoundaryResult, OperatorConfig, make_runner, run_boundary
from .view import ring_view, snapshot

__all__ = ["RINGS", "BoundaryResult", "OperatorConfig", "apply_plan", "interventions", "make_runner", "plan_schema",
           "ring_view", "run_boundary", "snapshot", "validate_plan"]
