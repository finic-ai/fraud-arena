"""Turn the defender's policy into the deployment the engine runs, with the harness's case logging.

Every decision of the defender's role gets, first, a logging rule that asks the harness for a numeric outside feature
(`arena_log`) and never fires: the harness records the case's observed features and identifiers, which become the
defender's training data for `arena fit`. After it come the defender's rules
for the decision, or, for a decision the defender wrote no rules for, the bank's built-in rules (copied from the
world's `brindlemark_rules`), so leaving a decision alone keeps the bank's behaviour exactly.
"""

import ast
from pathlib import Path

import yaml

LOG = "arena_log"
DEFAULTS = {}  # decision -> its default action (for the logging rule of a decision with no rules anywhere)


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
        DEFAULTS[name] = d.get("default_action")
    return builtin, gates


def _obs_names(node):
    return sorted({f"obs.{n.attr}" for n in ast.walk(node)
                   if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "obs"})


def null_safe(expr):
    """`expr` with every comparison that orders values (<, <=, >, >=) or does arithmetic guarded against missing
    values: `obs.x > 3` becomes `(obs.x != none and obs.x > 3)`, so a comparison involving a missing value is false
    instead of an error that stops the bank's systems. Equality and membership tests already handle missing values and
    are left alone. Only the guarded comparisons change; the rest of the text is kept as written. An expression Python
    can't parse is returned unchanged, for the engine to judge."""
    if not isinstance(expr, str) or not expr.strip():
        return expr
    if "\n" in expr:  # offsets below are per line: one line keeps them simple (newlines are only spacing here)
        expr = " ".join(expr.split())
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        ordering = any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops)
        arithmetic = any(isinstance(n, ast.BinOp) for n in ast.walk(node))
        names = _obs_names(node)
        if (ordering or arithmetic) and names:
            spans.append((node.col_offset, node.end_col_offset, names))
    return _guard(expr, spans) if spans else expr


def _guard(expr, spans):
    # Outermost spans only (a comparison inside another guarded one is covered by the outer guard), right to left.
    spans = [s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)]
    out = expr
    for start, end, names in sorted(spans, reverse=True):
        guard = " and ".join(f"{n} != none" for n in names)
        out = f"{out[:start]}({guard} and {out[start:end]}){out[end:]}"
    return out


def compose(policy, builtin, gates, *, log=True, lists=None, native=False, origin=None):
    """The deployment for `policy` (the defender's JSON: rules and optional models). With `native` (an engine with
    `Run.take_cases`), the run records the cases itself (`record_cases`) instead of a logging rule that reads an
    outside feature, so only a defender's lists need outside code."""
    record = log and native
    log = log and not native
    policy = policy or {}
    src = list(policy.get("rules") or [])
    # The defender's rules as the engine runs them: missing values make a comparison false (null_safe). `origin`, when
    # given, is filled with each composed rule's index in the defender's own list (None for built-in and default rules).
    own = [{**r, "when": null_safe(r["when"])} if isinstance(r, dict) and "when" in r else r for r in src]
    index = {id(r): i for i, r in enumerate(own)}
    covered = {r["decision"] for r in own}
    rules = []
    for decision in gates:
        if log and gates[decision] == "event":  # after-event decisions (e.g. txn monitoring) can't be labeled: skip
            rules.append({"decision": decision, "when": f"outside.{LOG} != none and outside.{LOG} > 1",
                          "action": _any_action(decision, builtin, own)})
        mine = [r for r in own if r["decision"] == decision] if decision in covered else \
               [r for r in builtin if r["decision"] == decision]
        if record and not mine and gates[decision] == "event" and DEFAULTS.get(decision):
            # The policy must decide (and so record) every event gate, as the logging rule made it do: with no
            # rule of its own, a gate takes its default action, as it did then.
            mine = [{"decision": decision, "action": DEFAULTS[decision]}]
        rules += mine
    if origin is not None:
        origin[:] = [index.get(id(r)) for r in rules]
    dep = {"format": 7, "record_cases": True, "rules": rules} if record else {"format": 5, "rules": rules}
    if policy.get("models"):
        dep["models"] = policy["models"]
    outside = {LOG: {"type": "number"}} if log else {}
    for name in lists or {}:  # the defender's lists, answered by the harness at decision time
        outside[f"list_{name}"] = {"type": "number"}
    if outside:
        dep["outside"] = outside
        reading = {d for d, g in gates.items() if g == "event"} if log else set()
        if lists:
            reading |= set(gates)
        dep["external"] = {d: ({} if gates.get(d) == "event" else {"timeout": "1h"}) for d in reading}
    elif not record:
        dep["format"] = 1
    return dep


def _any_action(decision, builtin, own):
    for r in own + builtin:
        if r["decision"] == decision:
            return r["action"]
    if DEFAULTS.get(decision):
        return DEFAULTS[decision]
    raise ValueError(f"no action known for {decision}")
