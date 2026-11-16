"""Agent ring operators: one headless agent session per active ring per period boundary.

At each boundary the harness, on the grader's side, builds the ring's view (`view.ring_view`) and writes it to the
ring's workspace with the plan's schema and the brief; the runner (any `fraud_arena.runners.AgentRunner`: Claude
Code, Codex, ...) works in the workspace, which persists between boundaries, and leaves `plan.json`. The harness
validates the plan and sets it on the ring (`menu.apply_plan`). A missing or invalid plan runs the ring as scripted
for the period. Rings without an operator keep their scripted behaviour in the same episode.

The attacker gets no MCP server and no data beyond `view.json`. Every plan, session result, token count and the
model that answered are logged to `attacker.jsonl` in the log folder.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

from ..runners import RUNNERS, AgentRunner, SessionResult, UsageLimited
from . import menu

__all__ = ["OperatorConfig", "BoundaryResult", "make_runner", "brief", "run_boundary", "UsageLimited"]


@dataclass(frozen=True)
class OperatorConfig:
    """One agent attacker: the ring it operates, the runner and the model."""

    ring: int
    runner: str = "claude-code"
    model: str = "opus"
    timeout_s: int = 3600
    extra_env: dict[str, str] = field(default_factory=dict)
    # As the defender's sessions: run as this user, with network only through this allowlisting proxy.
    as_user: str | None = None
    proxy: str | None = None

    def __post_init__(self):
        if self.ring not in menu.RINGS:
            raise ValueError(f"no ring R{self.ring}")
        if self.runner not in RUNNERS:
            raise ValueError(f"runner is one of {sorted(RUNNERS)}")


@dataclass
class BoundaryResult:
    ring: int
    day: int
    plan: dict
    used_scripted: bool
    plan_errors: list[str]
    runner: str
    model: str
    model_version: str | None
    ok: bool
    seconds: float
    usage: dict
    session_id: str | None
    session_error: str | None


def make_runner(cfg: OperatorConfig) -> AgentRunner:
    return RUNNERS[cfg.runner](cfg.model, timeout_s=cfg.timeout_s, extra_env=cfg.extra_env, as_user=cfg.as_user,
                               proxy=cfg.proxy)


def _menu_text(schema: dict) -> str:
    lines = []
    for k, p in schema["properties"].items():
        if k == "notes":
            continue
        if p.get("type") == "object":
            keys = ", ".join(f"{n} [{s['minimum']}-{s['maximum']}]" for n, s in p["properties"].items())
            lines.append(f"- `{k}`: {p['description']} Keys: {keys}.")
        elif "enum" in p:
            lines.append(f"- `{k}`: one of {', '.join(p['enum'])}. {p['description']}")
        elif p.get("type") == "boolean":
            lines.append(f"- `{k}`: true or false. {p['description']}")
        else:
            lines.append(f"- `{k}`: {p['minimum']} to {p['maximum']}. {p['description']}")
    return "\n".join(lines)


AUTHORING = ("You don't write any messages, scripts or techniques. You only set the ring's strategy for the next "
             "period, by choosing\nvalues from a fixed menu, and the world carries it out with its own scripted "
             "machinery.")
AUTHORING_TEXT = ("You don't write scripts or techniques, or the messages the ring sends its victims. You set the "
                  "ring's strategy for the\nnext period by choosing values from a fixed menu, and the world carries "
                  "it out with its own scripted machinery.\nYou may also brief how the texts that the ring's schemes "
                  "leave at the institutions and in public read (below).")


def brief(*, ring: int, schema: dict, day: int, period_days: int, boundary: int, boundaries: int,
          text_schemes: dict[str, list[str]] | None = None, text_notes: list[str] | None = None) -> str:
    """The attacker's brief. With `text_schemes` (attacker-authored text is on: scheme -> the texts it produces),
    it also explains the scheme briefs."""
    text = resources.files(__package__).joinpath("brief.md").read_text()
    return text.format(ring=f"R{ring}", day=day, period_days=period_days, next_day=day + period_days,
                       boundary=boundary, boundaries=boundaries, menu=_menu_text(schema),
                       authoring=AUTHORING if text_schemes is None else AUTHORING_TEXT,
                       texts="" if text_schemes is None else _texts_text(text_schemes, text_notes or []))


def _texts_text(schemes: dict[str, list[str]], notes: list[str]) -> str:
    text = resources.files(__package__).joinpath("text_brief.md").read_text()
    lines = [f"- `{s}`: " + "; ".join(kinds) for s, kinds in sorted(schemes.items())] or \
        ["- (none of the ring's schemes writes such texts)"]
    out = text.format(schemes="\n".join(lines))
    if notes:
        out += "\nLast period's briefs: " + "; ".join(notes)
    return "\n" + out


def _model_version(log: Path | None) -> str | None:
    """The model that answered, from the session log: Claude Code's `modelUsage`, or Codex's `model`."""
    if log is None or not log.exists():
        return None
    for line in reversed(log.read_text().splitlines()):
        try:
            out = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            if isinstance(out.get("modelUsage"), dict) and out["modelUsage"]:
                return ",".join(sorted(out["modelUsage"]))
            if isinstance(out.get("model"), str):
                return out["model"]
    return None


def run_boundary(cfg: OperatorConfig, runner: AgentRunner, *, workspace: Path, logs: Path, world: Path, view: dict,
                 day: int, period_days: int, boundary: int, boundaries: int) -> BoundaryResult:
    """One session of ring R`cfg.ring` at the boundary on `day`. Returns the plan to set (the scripted plan, `{}`,
    when the agent leaves none or an invalid one). Raises `UsageLimited` when the subscription's limit is reached,
    so the queue can resume the same boundary later."""
    schema = menu.plan_schema(world, cfg.ring)
    workspace.mkdir(parents=True, exist_ok=True)
    plan_path = workspace / "plan.json"
    plan_path.unlink(missing_ok=True)
    (workspace / "view.json").write_text(json.dumps(view, indent=2, sort_keys=True))
    (workspace / "plan.schema.json").write_text(json.dumps(schema, indent=2))
    prompt = brief(ring=cfg.ring, schema=schema, day=day, period_days=period_days, boundary=boundary,
                   boundaries=boundaries)
    log = logs / f"R{cfg.ring}" / f"day{day:04d}.log"
    result: SessionResult = runner.run(workspace, prompt, [], log)
    errors: list[str] = []
    plan: dict = {}
    if not plan_path.exists():
        errors.append("no plan.json")
    else:
        try:
            plan = json.loads(plan_path.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"plan.json isn't JSON: {e}")
        else:
            errors = menu.validate_plan(plan, schema)
    used_scripted = bool(errors)
    if used_scripted:
        plan = {}
    out = BoundaryResult(cfg.ring, day, plan, used_scripted, errors, result.runner, result.model,
                         _model_version(result.raw_log), result.ok, result.seconds, result.usage, result.session_id,
                         result.error)
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "attacker.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), **asdict(out)}, sort_keys=True) + "\n")
    return out
