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
  arena policy set FILE        check the policy in FILE (JSON or YAML; write it with the Write tool, e.g.
                               drafts/week2.yaml) and save it for the next week
  arena rule list              (defender) your rules, numbered
  arena rule add --decision D --when "EXPR" --action A [--at N]
                               (defender) add one rule (at position N, default the end) and save
  arena rule remove N          (defender) remove rule N and save
  arena fit logistic|trees --decision D --features a,b,c [--where "SQL"] [--balanced] [--depth 3] [--trees 60]
                               [--save NAME]
                               (defender) train a scoring model on the case table of decision D (label: fraud_known),
                               report test metrics, and with --save add it to your policy as models.NAME (read in
                               rules as score.NAME)
  arena tables                 (defender) the bank's tables and their columns
  arena query "SQL"            (defender) a read-only DuckDB query on the bank's tables, up to 200 rows. The
                               cases_<decision> tables hold every decision the bank made, with its observed features,
                               the action taken and fraud_known (fraud discovered so far)
  arena results                (attacker) your syndicate's results so far
  arena note "TEXT"            append a line to your notes (kept between turns)
  arena notes                  print your notes
  arena grayson --table T --question "Q" --options yes,no --as NAME [--where "SQL"] [--limit 50]
                               (defender, when Grayson is enabled) ask Grayson Q about rows of T (e.g. cases with memos
                               or notes); saves the answers as table NAME (key + a probability per option)"""


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
        except Exception:
            try:
                import yaml
                policy = yaml.safe_load(text)
            except Exception as error:
                sys.exit(f"policy NOT saved: not valid JSON or YAML ({error})")
        problems = check(work, role, policy)
        if problems:
            sys.exit("policy NOT saved:\n" + "\n".join(f"  - {p}" for p in problems[:30]))
        (work / "policy.json").write_text(json.dumps(policy, indent=1))
        print("policy saved; it takes effect at the start of next week")
    elif cmd == "rule" and role == "defender":
        rule_command(work, role, args[1:])
    elif cmd == "fit" and role == "defender":
        fit_command(work, role, args[1:])
    elif cmd == "grayson" and role == "defender":
        grayson_command(work, args[1:])
    elif cmd == "tables" and role == "defender":
        from arena.fw.tools import connect
        db = connect(work)
        for (name,) in db.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1").fetchall():
            cols = [c[0] for c in db.execute(f"DESCRIBE {name}").fetchall()]
            n = db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"{name} ({n} rows): {', '.join(cols)}")
    elif cmd == "query" and role == "defender" and len(args) == 2:
        from arena.fw.tools import connect
        db = connect(work)
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


def _opts(args, flags, bools=()):
    out, i = {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and a[2:] in bools:
            out[a[2:]] = True
            i += 1
        elif a.startswith("--") and a[2:] in flags and i + 1 < len(args):
            out[a[2:]] = args[i + 1]
            i += 2
        else:
            sys.exit(f"unexpected argument {a!r}")
    return out


def _save(work, role, policy, what):
    problems = check(work, role, policy)
    if problems:
        sys.exit(f"{what} NOT saved:\n" + "\n".join(f"  - {p}" for p in problems[:30]))
    (work / "policy.json").write_text(json.dumps(policy, indent=1))
    print(f"{what} saved; the policy takes effect at the start of next week")


def rule_command(work, role, args):
    policy = json.loads((work / "policy.json").read_text())
    rules = policy.setdefault("rules", [])
    if args[:1] == ["list"]:
        for i, r in enumerate(rules, 1):
            print(f"{i}. {r['decision']}: when {r.get('when', 'always')} -> {r['action']}")
        return print("" if rules else "(no rules: every decision keeps the bank's built-in rules)")
    if args[:1] == ["remove"] and len(args) == 2 and args[1].isdigit() and 1 <= int(args[1]) <= len(rules):
        rules.pop(int(args[1]) - 1)
        return _save(work, role, policy, "rule removal")
    if args[:1] == ["add"]:
        o = _opts(args[1:], ("decision", "when", "action", "at"))
        if "decision" not in o or "action" not in o:
            sys.exit("usage: arena rule add --decision D --when \"EXPR\" --action A [--at N]")
        r = {"decision": o["decision"], "action": o["action"]}
        if o.get("when"):
            r["when"] = o["when"]
        at = int(o["at"]) - 1 if o.get("at", "").isdigit() else len(rules)
        rules.insert(max(0, min(at, len(rules))), r)
        return _save(work, role, policy, "rule")
    sys.exit("usage: arena rule list | add --decision D --when \"EXPR\" --action A [--at N] | remove N")


def fit_command(work, role, args):
    if not args or args[0] not in ("logistic", "trees"):
        sys.exit("usage: arena fit logistic|trees --decision D --features a,b,c [--where SQL] [--balanced] [--save NAME]")
    o = _opts(args[1:], ("decision", "features", "where", "depth", "trees", "save"), bools=("balanced",))
    if "decision" not in o or "features" not in o:
        sys.exit("--decision and --features are required")
    from arena.fw.tools import fit
    try:
        report, model = fit(work, args[0], o["decision"], [f.strip() for f in o["features"].split(",") if f.strip()],
                            where=o.get("where"), depth=int(o.get("depth", 3)), trees=int(o.get("trees", 60)),
                            balanced=bool(o.get("balanced")))
    except Exception as error:
        sys.exit(f"fit failed: {error}")
    print(report)
    if o.get("save"):
        name = o["save"]
        if not name.replace("_", "").isalnum() or not name.islower():
            sys.exit("--save NAME: lower case letters, digits and _")
        policy = json.loads((work / "policy.json").read_text())
        policy.setdefault("models", {})[name] = model
        _save(work, role, policy, f"model {name!r}")
        print(f"use it in rules as score.{name} (a probability 0-1), e.g. "
              f"arena rule add --decision {o['decision']} --when \"score.{name} > 0.8\" --action ...")


def grayson_command(work, args):
    if not (work / "grayson_enabled").exists():
        sys.exit("Grayson isn't available in this match")
    o = _opts(args, ("table", "question", "options", "as", "where", "limit"))
    if not all(k in o for k in ("table", "question", "options", "as")):
        sys.exit("usage: arena grayson --table T --question \"Q\" --options yes,no --as NAME [--where SQL] [--limit 50]")
    opts = [x.strip() for x in o["options"].split(",") if x.strip()]
    from arena.fw.tools import grayson_ask
    try:
        print(grayson_ask(work, o["table"], o.get("where"), o["question"], {x: x for x in opts}, o["as"],
                          limit=int(o.get("limit", 50)), calls_file=work / "grayson_calls_this_turn"))
    except Exception as error:
        sys.exit(f"grayson failed: {error}")


def policy_help(work, role):
    if role == "defender":
        decisions = json.loads((work / "decisions.json").read_text())
        lines = ["Your policy is a Runmere deployment (format 1), JSON or YAML: \"format\": 1, \"rules\", and optional \"models\"",
                 "(and \"grayson\" when enabled). For small changes use `arena rule add/remove`; to train a model on the",
                 "bank's labeled cases use `arena fit` (it adds the model block for you with --save).",
                 "Each rule: {\"decision\": <name>, \"when\": \"<expression over obs.<feature>, score.<model>>\", \"action\": <action>}.",
                 "Per decision, the first rule whose `when` holds decides; a decision with rules but none holding takes its default;",
                 "a decision with no rules keeps the bank's built-in policy. Expressions: and/or/not, comparisons, arithmetic,",
                 "e.g. \"obs.amount > 2000 and obs.new_device\". Features can be none (missing): ordering or arithmetic on none",
                 "is an error that rolls back your whole policy, so guard them: \"obs.x != none and obs.x > 24\". Models (optional): {\"name\": {\"kind\": \"logistic\", \"features\": [...],",
                 "\"weights\": [...], \"intercept\": 0}} read as score.<name> (also kinds linear, trees).", ""]
        if (work / "grayson_enabled").exists():
            lines += ["GRAYSON (optional): a decision model that answers questions about a case with calibrated probabilities.",
                      "Add a top-level \"grayson\": {\"<question_name>\": {\"question\": \"<what to judge>\", \"options\": {\"yes\": \"<meaning>\",",
                      "\"no\": \"<meaning>\"}, \"decisions\": [\"<decision>\", ...]}} and read it in rules as outside.<question_name>.<option>",
                      "(a probability 0-1). Grayson sees the case's observed features. It is asked only when a rule's evaluation reaches",
                      "the read, so put cheap conditions first: \"obs.amount > 200 and outside.scam_payment.yes > 0.8\". Each week has a",
                      (work / "grayson_enabled").read_text().strip() + "; past it, every option reads 1/n.", ""]
        lines.append("Your decisions:")
        for d in decisions:
            feats = ", ".join(f"{f[0]}:{f[1]}" for f in d["features"])
            lines.append(f"- {d['name']}: actions {d['actions']} (default {d['default_action']})\n    features: {feats}")
            for f, vals in (d.get("code_values") or {}).items():
                lines.append(f"    {f} values (lower case, quoted): {', '.join(repr(v) for v in vals)}")
        return "\n".join(lines)
    menu = json.loads((work / "menu.json").read_text())
    lines = ["Your plan is JSON or YAML: {\"all\": {<attribute>: value, ...}, \"rings\": {\"<crew number 1-10>\": {<attribute>: value}}}.",
             "\"all\" applies to every crew; \"rings\" overrides per crew. plan_volume is relative to the match's usual pace (1 = usual). Unmentioned attributes keep their last value;",
             "null resets one to the ring's own scripted habit. The attributes:"]
    for attr, d in menu.items():
        rng = f"{d['min']} to {d['max']}" if "min" in d else (f"one of {json.dumps(d['values'])}" if "values" in d else "see the world")
        lines.append(f"- {attr}: {rng}")
    lines.append("plan_volume multiplies every scheme of a ring; plan_tactic_<scheme> multiplies one scheme (1 = as scripted);"
                 " plan_quiet stops the ring. Targeting is fixed: every crew aims at Brindlemark wherever its schemes allow.")
    lines.append(f"BUDGET: the syndicate has limited people, money and mules. plan_volume summed over the 10 rings must be at "
                 f"most {VOLUME_BUDGET:g} (each unset ring counts 1, so scripted is 10), and the 22 scheme multipliers must "
                 f"sum to at most {TACTIC_BUDGET:g} (each scheme counts its highest multiplier across rings; scripted is 22). "
                 f"To push some rings or schemes harder, turn others down.")
    return "\n".join(lines)


VOLUME_BUDGET = 12.0  # sum of plan_volume over the 10 rings (scripted: 10 x 1)
TACTIC_BUDGET = 26.0  # sum over the 22 schemes of each scheme's multiplier (scripted: 22 x 1)


def plan_effort(plan, menu):
    """The plan's effort: (total volume across rings, total scheme multipliers). Unset values count as 1."""
    plan = plan or {}
    tactics = [a for a in menu if a.startswith("plan_tactic_")]
    volume, tactic_max = 0.0, {t: 0.0 for t in tactics}
    for ring in range(1, 11):
        merged = dict(plan.get("all") or {})
        merged.update((plan.get("rings") or {}).get(str(ring)) or {})
        v = merged.get("plan_volume")
        volume += 1.0 if v is None else float(v)
        for t in tactics:
            x = merged.get(t)
            tactic_max[t] = max(tactic_max[t], 1.0 if x is None else float(x))
    return volume, sum(tactic_max.values())


def budget_problems(plan, menu):
    volume, tactics = plan_effort(plan, menu)
    out = []
    if volume > VOLUME_BUDGET + 1e-9:
        out.append(f"over the volume budget: plan_volume summed over the 10 rings is {volume:.2f}, at most {VOLUME_BUDGET:g} "
                   f"(unset rings count 1)")
    if tactics > TACTIC_BUDGET + 1e-9:
        out.append(f"over the scheme budget: the 22 scheme multipliers sum to {tactics:.2f}, at most {TACTIC_BUDGET:g} "
                   f"(each scheme counts its highest multiplier across rings; unset counts 1)")
    return out


def check(work, role, policy):
    """Structural checks; the engine checks expressions when it deploys."""
    problems = []
    if role == "defender":
        decisions = {d["name"]: d for d in json.loads((work / "decisions.json").read_text())}
        if not isinstance(policy, dict) or policy.get("format") not in (1,):
            return ['needs "format": 1']
        grayson = policy.get("grayson")
        if grayson is not None:
            if not (work / "grayson_enabled").exists():
                return ['"grayson" is not available in this match']
            from arena.fw.grayson import check_section
            problems += check_section(grayson, list(decisions.values()))
        unknown = set(policy) - {"format", "name", "rules", "models", "grayson"}
        if unknown:
            problems.append(f"unknown top-level keys: {sorted(unknown)}")
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
            for q, opt in re.findall(r"outside\.([a-zA-Z_0-9]+)\.([a-zA-Z_0-9]+)", str(r.get("when", ""))):
                spec = (grayson or {}).get(q)
                if not spec:
                    problems.append(f"rules[{i}]: outside.{q} isn't a question in \"grayson\"")
                elif opt not in (spec.get("options") or {}):
                    problems.append(f"rules[{i}]: grayson.{q} has no option {opt!r}")
                elif d["name"] not in (spec.get("decisions") or []):
                    problems.append(f"rules[{i}]: add {d['name']!r} to grayson.{q}.decisions")
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
    return problems + budget_problems(policy, menu)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
