"""The policy format both players submit, and its evaluator.

A policy maps each of the player's decisions to a default action and an ordered rule list; the first rule whose
conditions all hold decides. Conditions compare one of the decision's observed features with a constant.
This module has no dependencies, so the player's `arena` CLI can import it inside the container.
"""

DECISIONS = {
    "card_auth": {
        "role": "defender",
        "actions": ["approve", "decline", "step_up"],
        "default": "approve",
        "features": {
            "amount": "purchase amount, USD",
            "ecommerce": "true if an online purchase",
            "new_device": "true if the device was never seen in this customer's logins in 90 days",
            "txn_count_7d": "the customer's card purchases in the last 7 days",
            "contact_changed_7d": "true if the customer's email/phone changed in the last 7 days",
            "failed_logins_1d": "failed logins to the customer's account in the last day",
            "ip_risk": "IP risk score from 0 (clean) to 1 (risky)",
        },
    },
    "payment_hold": {
        "role": "defender",
        "actions": ["release", "hold", "block"],
        "default": "release",
        "features": {
            "amount": "P2P payment amount, USD",
            "new_payee": "true if the customer never paid this payee in 90 days",
            "balance_share": "amount as a share of the account balance (0 to 1+)",
            "age": "the customer's age",
            "sends_7d": "the customer's P2P payments in the last 7 days",
        },
        "help": "Every outgoing P2P payment from a customer's checking account: release it, hold it for review "
                "(the payment doesn't go through), or block it. Holding or blocking a legitimate payment costs $4.",
    },
    "ato_route": {
        "role": "attacker",
        "actions": ["cheap", "stealth"],
        "default": "cheap",
        "features": {"victim_age": "the victim's age"},
        "help": "How to log in: cheap (emulator + data-center proxy) or stealth (phone farm + residential proxy, "
                "phished credentials; logs in more reliably but costs money per login and purchase).",
    },
    "ato_takeover": {
        "role": "attacker",
        "actions": ["change_email", "keep_contact"],
        "default": "change_email",
        "features": {"failed_logins_7d": "your failed logins to this victim in the last 7 days"},
        "help": "After getting in: change the account email (locks the victim out of alerts) or keep the "
                "victim's contact details (quieter).",
    },
    "ato_cash_out": {
        "role": "attacker",
        "actions": ["small", "medium", "large", "blend", "drain", "stop"],
        "default": "medium",
        "features": {
            "hijacked": "true if you changed this victim's contact details",
            "approved_1d": "your purchases on this card approved in the last day",
            "refused_1d": "your purchases on this card declined or stepped up in the last day",
            "days_in": "days since the takeover",
        },
        "help": "Each cash-out step: purchase size small/medium/large, blend (look like the victim's normal "
                "spending), drain (max out), or stop working this victim.",
    },
    "ato_after_decline": {
        "role": "attacker",
        "actions": ["retry_lower", "wait"],
        "default": "retry_lower",
        "features": {"amount": "the declined purchase amount, USD"},
        "help": "After a decline: retry at half the amount (only if the amount is at least 20) or wait.",
    },
}

OPS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
MAX_RULES = 20
MAX_CONDITIONS = 6


def decisions_of(role):
    return {name: spec for name, spec in DECISIONS.items() if spec["role"] == role}


def default_policy(role):
    return {name: {"default": spec["default"], "rules": []} for name, spec in decisions_of(role).items()}


def validate(policy, role):
    """Return a list of human-readable problems; empty means valid."""
    errors = []
    if not isinstance(policy, dict):
        return ["the policy must be a JSON object mapping decision names to {default, rules}"]
    mine = decisions_of(role)
    for name in policy:
        if name not in mine:
            errors.append(f"unknown decision {name!r}; yours are {sorted(mine)}")
    for name, spec in mine.items():
        entry = policy.get(name)
        if entry is None:
            continue  # missing decisions keep their default
        if not isinstance(entry, dict):
            errors.append(f"{name}: must be an object with 'default' and 'rules'")
            continue
        if entry.get("default", spec["default"]) not in spec["actions"]:
            errors.append(f"{name}.default: must be one of {spec['actions']}")
        rules = entry.get("rules", [])
        if not isinstance(rules, list) or len(rules) > MAX_RULES:
            errors.append(f"{name}.rules: must be a list of at most {MAX_RULES} rules")
            continue
        for i, rule in enumerate(rules):
            where = f"{name}.rules[{i}]"
            if not isinstance(rule, dict) or rule.get("action") not in spec["actions"]:
                errors.append(f"{where}: needs 'action', one of {spec['actions']}")
                continue
            conds = rule.get("when", [])
            if not isinstance(conds, list) or not 0 < len(conds) <= MAX_CONDITIONS:
                errors.append(f"{where}.when: must be a list of 1 to {MAX_CONDITIONS} conditions")
                continue
            for j, cond in enumerate(conds):
                if (not isinstance(cond, list) or len(cond) != 3 or cond[0] not in spec["features"]
                        or cond[1] not in OPS or not isinstance(cond[2], (int, float, bool))):
                    errors.append(f"{where}.when[{j}]: must be [feature, op, number|true|false] with feature in "
                                  f"{sorted(spec['features'])} and op in {sorted(OPS)}")
    return errors


def decide(policy, name, features):
    """The action for decision `name` given its features (a dict). Returns (action, rule index or None)."""
    entry = (policy or {}).get(name) or {}
    for i, rule in enumerate(entry.get("rules", [])):
        if all(OPS[op](features[feat], value) for feat, op, value in rule["when"]):
            return rule["action"], i
    return entry.get("default", DECISIONS[name]["default"]), None


def format_help(role):
    lines = [f"Your policy is a JSON object with one entry per decision you make ({role}):", ""]
    for name, spec in decisions_of(role).items():
        lines.append(f"{name}: actions {spec['actions']} (default {spec['default']!r})")
        if spec.get("help"):
            lines.append(f"  {spec['help']}")
        for feat, desc in spec["features"].items():
            lines.append(f"  feature {feat}: {desc}")
        lines.append("")
    lines += [
        "Each entry: {\"default\": <action>, \"rules\": [{\"when\": [[feature, op, value], ...], \"action\": <action>}, ...]}",
        f"Rules are checked in order and the first whose conditions ALL hold decides; otherwise the default applies.",
        f"op is one of {sorted(OPS)}; value is a number or true/false. At most {MAX_RULES} rules per decision, "
        f"{MAX_CONDITIONS} conditions per rule.",
    ]
    return "\n".join(lines)
