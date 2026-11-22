"""Turn the defender's policy into the deployment the engine runs, with the harness's case logging.

Every decision of the defender's role gets, first, a logging rule that asks the harness for a numeric outside feature
(`arena_log`) and never fires: the harness records the case's observed features and identifiers, which become the
defender's training data for `arena fit`. After it come the defender's rules
for the decision, or, for a decision the defender wrote no rules for, the bank's built-in rules (copied from the
world's `brindlemark_rules`), so leaving a decision alone keeps the bank's behaviour exactly.
"""

from pathlib import Path

import yaml

LOG = "arena_log"


def world_bits(world_path, role, builtin_policy):
    spec = yaml.safe_load(Path(world_path).read_text())
    env = spec.get("env") or {}
    builtin = [dict(r) for r in env["policies"][builtin_policy]["rules"]]
    gates = {}
    for name, d in (env.get("decisions") or {}).items():
        if d.get("role") != role:
            continue
        at = d.get("at") or {}
        gates[name] = "event" if ("event" in at and not at.get("after")) else "wait"
    return builtin, gates


def compose(policy, builtin, gates, *, log=True):
    """The deployment for `policy` (the defender's JSON: rules and optional models)."""
    policy = policy or {}
    own = list(policy.get("rules") or [])
    covered = {r["decision"] for r in own}
    rules = []
    for decision in gates:
        if log and gates[decision] == "event":  # after-event decisions (e.g. txn monitoring) can't be labeled: skip
            rules.append({"decision": decision, "when": f"outside.{LOG} != none and outside.{LOG} > 1",
                          "action": _any_action(decision, builtin, own)})
        rules += [r for r in own if r["decision"] == decision] if decision in covered else \
                 [r for r in builtin if r["decision"] == decision]
    dep = {"format": 5, "rules": rules}
    if policy.get("models"):
        dep["models"] = policy["models"]
    outside = {LOG: {"type": "number"}} if log else {}
    if outside:
        dep["outside"] = outside
        reading = {d for d, g in gates.items() if g == "event"} if log else set()
        dep["external"] = {d: ({} if gates.get(d) == "event" else {"timeout": "1h"}) for d in reading}
    else:
        dep["format"] = 1
    return dep


def _any_action(decision, builtin, own):
    for r in own + builtin:
        if r["decision"] == decision:
            return r["action"]
    raise ValueError(f"no action known for {decision}")
