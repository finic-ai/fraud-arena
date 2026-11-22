"""The `arena` command for one player. The runner writes a wrapper script that runs this module with the player's
workspace baked in, so a player can only ever reach its own folder.

Workspace layout (written by the runner): role, status.md, brief.md, decisions.json (defender) or menu.json
(attacker), policy.json, notes.md, and data/ (defender: bank.duckdb and Parquet; attacker: results.json)."""

import json
import re
import sys
from pathlib import Path

USAGE = """usage:
  arena status                 the round, your scores so far and your current policy
  arena brief                  your role, goal and how scoring works
  arena policy help            the policy format: your decisions/features (defender) or the plan menu (attacker)
  arena policy show            print your current policy
  arena policy set FILE        check the JSON policy in FILE (write it with the Write tool, e.g. drafts/policy.json)
                               and save it for the next week
  arena tables                 (defender) the bank's tables and their columns
  arena query "SQL"            (defender) a read-only DuckDB query on the bank's tables, up to 200 rows
  arena results                (attacker) your syndicate's results so far
  arena note "TEXT"            append a line to your notes (kept between turns)
  arena notes                  print your notes"""


def main(work, args):
    work = Path(work)
    role = (work / "role").read_text().strip()
    if not args or args[0] in ("help", "-h", "--help"):
        return print(USAGE)
    cmd = args[0]
    if cmd == "status":
        print((work / "status.md").read_text())
    elif cmd == "brief":
        print((work / "brief.md").read_text())
    elif args[:2] == ["policy", "help"]:
        print(policy_help(work, role))
    elif args[:2] == ["policy", "show"]:
        print((work / "policy.json").read_text())
    elif args[:2] == ["policy", "set"]:
        if len(args) != 3:
            sys.exit("usage: arena policy set FILE")
        target = (work / args[2]).resolve() if not Path(args[2]).is_absolute() else Path(args[2]).resolve()
        if work.resolve() not in target.parents:
            sys.exit("the policy file must be inside your workspace (e.g. drafts/policy.json)")
        text = target.read_text()
        try:
            policy = json.loads(text)
        except Exception as error:
            sys.exit(f"policy NOT saved: not valid JSON ({error})")
        problems = check(work, role, policy)
        if problems:
            sys.exit("policy NOT saved:\n" + "\n".join(f"  - {p}" for p in problems[:30]))
        (work / "policy.json").write_text(json.dumps(policy, indent=1))
        print("policy saved; it takes effect at the start of next week")
    elif cmd == "tables" and role == "defender":
        import duckdb
        db = duckdb.connect(str(work / "data" / "bank.duckdb"), read_only=True)
        for (name,) in db.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1").fetchall():
            cols = [c[0] for c in db.execute(f"DESCRIBE {name}").fetchall()]
            n = db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"{name} ({n} rows): {', '.join(cols)}")
    elif cmd == "query" and role == "defender" and len(args) == 2:
        import duckdb
        db = duckdb.connect(str(work / "data" / "bank.duckdb"), read_only=True)
        try:
            cur = db.execute(args[1])
        except Exception as error:
            sys.exit(f"SQL error: {error}")
        print("\t".join(d[0] for d in cur.description or []))
        rows = cur.fetchmany(201)
        for row in rows[:200]:
            print("\t".join("" if v is None else str(v) for v in row))
        if len(rows) > 200:
            print("... (more than 200 rows: aggregate or add LIMIT)")
    elif cmd == "results" and role == "attacker":
        print((work / "data" / "results.json").read_text())
    elif cmd == "note" and len(args) == 2:
        with open(work / "notes.md", "a") as f:
            f.write(args[1].rstrip() + "\n")
        print("noted")
    elif cmd == "notes":
        p = work / "notes.md"
        print(p.read_text() if p.exists() else "(no notes yet)")
    else:
        sys.exit(USAGE)


def policy_help(work, role):
    if role == "defender":
        decisions = json.loads((work / "decisions.json").read_text())
        lines = ["Your policy is a Runmere deployment (format 1): JSON with \"format\": 1, \"rules\" and optional \"models\".",
                 "Each rule: {\"decision\": <name>, \"when\": \"<expression over obs.<feature>, score.<model>>\", \"action\": <action>}.",
                 "Per decision, the first rule whose `when` holds decides; a decision with rules but none holding takes its default;",
                 "a decision with no rules keeps the bank's built-in policy. Expressions: and/or/not, comparisons, arithmetic,",
                 "e.g. \"obs.amount > 2000 and obs.new_device\". Features can be none (missing): ordering or arithmetic on none",
                 "is an error that rolls back your whole policy, so guard them: \"obs.x != none and obs.x > 24\". Models (optional): {\"name\": {\"kind\": \"logistic\", \"features\": [...],",
                 "\"weights\": [...], \"intercept\": 0}} read as score.<name> (also kinds linear, trees).", "", "Your decisions:"]
        for d in decisions:
            feats = ", ".join(f"{f[0]}:{f[1]}" for f in d["features"])
            lines.append(f"- {d['name']}: actions {d['actions']} (default {d['default_action']})\n    features: {feats}")
            for f, vals in (d.get("code_values") or {}).items():
                lines.append(f"    {f} values (lower case, quoted): {', '.join(repr(v) for v in vals)}")
        return "\n".join(lines)
    menu = json.loads((work / "menu.json").read_text())
    lines = ["Your plan is JSON: {\"all\": {<attribute>: value, ...}, \"rings\": {\"<ring number 1-10>\": {<attribute>: value}}}.",
             "\"all\" applies to every ring; \"rings\" overrides per ring. Unmentioned attributes keep their last value;",
             "null resets one to the ring's own scripted habit. The attributes:"]
    for attr, d in menu.items():
        rng = f"{d['min']} to {d['max']}" if "min" in d else (f"one of {json.dumps(d['values'])}" if "values" in d else "see the world")
        lines.append(f"- {attr}: {rng}")
    lines.append("plan_volume multiplies every scheme of a ring; plan_tactic_<scheme> multiplies one scheme (1 = as scripted);"
                 " plan_quiet stops the ring; plan_where 'planned' + plan_share_<institution> picks where it attacks.")
    return "\n".join(lines)


def check(work, role, policy):
    """Structural checks; the engine checks expressions when it deploys."""
    problems = []
    if role == "defender":
        decisions = {d["name"]: d for d in json.loads((work / "decisions.json").read_text())}
        if not isinstance(policy, dict) or policy.get("format") not in (1,):
            return ['needs "format": 1']
        rules = policy.get("rules")
        if not isinstance(rules, list):
            return ['needs "rules": [...]']
        models = policy.get("models") or {}
        for i, r in enumerate(rules):
            d = decisions.get(r.get("decision")) if isinstance(r, dict) else None
            if d is None:
                problems.append(f"rules[{i}]: unknown decision {r.get('decision') if isinstance(r, dict) else r!r}")
                continue
            if r.get("action") not in d["actions"]:
                problems.append(f"rules[{i}]: action must be one of {d['actions']}")
            feats = {f[0] for f in d["features"]}
            for f in re.findall(r"obs\.([a-zA-Z_0-9]+)", str(r.get("when", ""))):
                if f not in feats:
                    problems.append(f"rules[{i}]: {d['name']} has no feature {f!r}")
            for m in re.findall(r"score\.([a-zA-Z_0-9]+)", str(r.get("when", ""))):
                if m not in models:
                    problems.append(f"rules[{i}]: no model named {m!r} in \"models\"")
        return problems
    menu = json.loads((work / "menu.json").read_text())
    if not isinstance(policy, dict):
        return ["the plan must be a JSON object"]
    blocks = [("all", policy.get("all") or {})] + [(f"rings.{k}", v or {}) for k, v in (policy.get("rings") or {}).items()]
    for k in (policy.get("rings") or {}):
        if k not in {str(n) for n in range(1, 11)}:
            problems.append(f"rings: {k!r} is not a ring number 1-10")
    for where, block in blocks:
        for attr, value in block.items():
            d = menu.get(attr)
            if d is None:
                problems.append(f"{where}: unknown attribute {attr!r}")
            elif value is None:
                continue
            elif "min" in d and not (isinstance(value, (int, float)) and d["min"] <= value <= d["max"]):
                problems.append(f"{where}.{attr}: must be a number from {d['min']} to {d['max']}")
            elif "values" in d and value not in d["values"]:
                problems.append(f"{where}.{attr}: must be one of {d['values']}")
    return problems


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
