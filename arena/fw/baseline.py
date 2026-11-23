"""A scripted defender that only trains models: the reference for "what do models alone get you" in the 1v1.

Each week, for each high-volume decision, it fits a logistic model on all of the decision's numeric and bool
features against the confirmed labels (fraud_known), reads the cost-aware report's best cutoff, and deploys
`score.<model> >= cutoff -> action` where that cutoff made money on the held-out cases. Decisions it can't model
profitably keep the bank's built-in rules. No lists, no hand rules, no LLM.
"""

import json
from pathlib import Path

from .tools import DEFAULT_ACTION, NUMERIC, fit

DECISIONS = ("brind_card_auth", "brind_login_check", "brind_account_open_review", "brind_payment_hold",
             "brind_ach_debit_review")


def model_policy(work, log=print):
    work = Path(work)
    decisions = {d["name"]: d for d in json.loads((work / "decisions.json").read_text())}
    policy = {"format": 1, "rules": [], "models": {}}
    for name in DECISIONS:
        d = decisions.get(name)
        if d is None:
            continue
        feats = [f[0] for f in d["features"] if str(f[1]).startswith(NUMERIC)]
        try:
            _, model, best = fit(work, "logistic", name, feats, return_best=True)
        except Exception as error:  # too few labels, no test fraud, ...
            log(f"  {name}: no model ({error})")
            continue
        if not best or best[1] <= 0:
            log(f"  {name}: no profitable cutoff on the test cases")
            continue
        key = name.replace("brind_", "m_")
        policy["models"][key] = model
        policy["rules"].append({"decision": name, "when": f"score.{key} >= {best[0]:.6g}",
                                "action": DEFAULT_ACTION[name]})
        # A decision with own rules no longer falls back to the built-in ones: keep them after the model's rule.
        log(f"  {name}: deploy score >= {best[0]:.4g} -> {DEFAULT_ACTION[name]} (test net ${best[1]:,.0f})")
    return policy
