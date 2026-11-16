"""A defender's manual review: account actions and held-case verdicts, applied as interventions (Fraud World DESIGN.md
§10.9), capped at its review team's capacity; and its typology labels, graded against truth at the end.

`bundle/reviews.json`:

    {"actions":  [{"kind": "account", "id": "<ACCT_NO>", "action": "restrict"}, ...],
     "verdicts": [{"case_no": "<REVIEW_QUEUE.CASE_NO>", "verdict": "reject"}, ...],
     "typology": [{"kind": "account", "id": "<ACCT_NO>", "type": "account_takeover"}, ...]}

Kinds by institution: `account` (Brindlemark `ACCOUNTS.ACCT_NO`, Dovrex `WALLETS.ACCT_NO`), `shop_account`
(Hopcrate and Kellmont `CUSTOMER_ACCOUNTS.CUSTOMER_NO`), `courier` (Hopcrate `COURIERS.COURIER_NO`). Typology labels
don't count against the capacity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REVIEWS_PER_DAY = {"brindlemark": 10, "dovrex": 8, "hopcrate": 12, "kellmont": 12}
DESK = {"brindlemark": "BRINDLEMARK", "dovrex": "DOVREX", "hopcrate": "HOPCRATE", "kellmont": "KELLMONT"}
KINDS = {
    "brindlemark": {"account": ("account", "number")},
    "dovrex": {"account": ("account", "number")},
    "hopcrate": {"shop_account": ("shop_account", "customer_no"), "courier": ("courier", "courier_no")},
    "kellmont": {"shop_account": ("shop_account", "customer_no")},
}
ACTIONS = {"account": {"none", "watch", "restrict", "close"}, "shop_account": {"none", "watch", "lock", "close"},
           "courier": {"none", "watch", "hold_payouts", "deactivate"}}
VERDICTS = {"release", "reject"}


@dataclass
class ReviewResult:
    applied: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    typology: list = field(default_factory=list)  # (entity id, kind, type)

    def summary(self) -> dict:
        return {"applied": len(self.applied), "refused": self.refused[:20], "typology_labels": len(self.typology)}


class Reviewer:
    """Grader side: finds entities by the identifiers the institution's tables show and applies the actions."""

    def __init__(self):
        self._inst_ids: dict[str, str] | None = None
        self._merchant_ids: dict[str, str] | None = None

    def _ids(self, run):
        if self._inst_ids is None:
            self._inst_ids = {r["name"]: r["id"] for r in run.rows("institution", ["name"])}
            self._merchant_ids = {r["brand_key"]: r["id"] for r in run.rows("business", ["brand_key"])
                                  if r.get("brand_key") in KINDS}
        return self._inst_ids, self._merchant_ids

    def hand_desk_to_defender(self, run, institution: str) -> None:
        rows = run.rows("review_desk", ["name"], where_attr="name", equals=DESK[institution])
        if len(rows) != 1:
            raise ValueError(f"no single review desk {DESK[institution]}")
        run.intervene("review_desk", rows[0]["id"], "worked_by", "defender")

    def find(self, run, institution: str, kind: str, ident: str) -> str:
        if kind not in KINDS[institution]:
            raise ValueError(f"{institution} has no `{kind}`; its kinds are {sorted(KINDS[institution])}")
        etype, attr = KINDS[institution][kind]
        inst_ids, merchant_ids = self._ids(run)
        scope_attr, scope = (("institution", inst_ids.get(institution)) if etype == "account"
                             else ("merchant", merchant_ids.get(institution)) if etype == "shop_account"
                             else (None, None))
        rows = run.rows(etype, [attr] + ([scope_attr] if scope_attr else []), where_attr=attr, equals=ident)
        if scope_attr:
            rows = [r for r in rows if r.get(scope_attr) == scope]
        if len(rows) != 1:
            raise ValueError(f"`{ident}` names {len(rows)} {kind}s at {institution}")
        return rows[0]["id"]

    def apply(self, run, institution: str, path: Path, capacity: int) -> ReviewResult:
        res = ReviewResult()
        if not path.exists():
            return res
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            res.refused.append(f"reviews.json isn't JSON: {e}")
            return res
        work = [("action", a) for a in doc.get("actions", [])] + [("verdict", v) for v in doc.get("verdicts", [])]
        if len(work) > capacity:
            res.refused.append(f"{len(work)} reviews asked; the team's capacity this period is {capacity}: "
                               f"only the first {capacity} were done")
            work = work[:capacity]
        for what, item in work:
            try:
                if what == "action":
                    kind, action = item.get("kind"), item.get("action")
                    if action not in ACTIONS.get(kind, ()):
                        raise ValueError(f"action `{action}` isn't one of {sorted(ACTIONS.get(kind, ()))}")
                    eid = self.find(run, institution, kind, str(item.get("id")))
                    run.intervene(KINDS[institution][kind][0], eid, "manual_action", action)
                else:
                    verdict = item.get("verdict")
                    if verdict not in VERDICTS:
                        raise ValueError(f"verdict `{verdict}` isn't release or reject")
                    rows = run.rows("review_case", ["case_no"], where_attr="case_no", equals=str(item.get("case_no")))
                    if len(rows) != 1:
                        raise ValueError(f"case `{item.get('case_no')}` names {len(rows)} cases")
                    run.intervene("review_case", rows[0]["id"], "manual_verdict", verdict)
                res.applied.append(item)
            except Exception as e:  # each refused review goes back to the defender
                res.refused.append(f"{item}: {e}")
        for item in doc.get("typology", []):
            try:
                eid = self.find(run, institution, item.get("kind"), str(item.get("id")))
                res.typology.append((eid, item.get("kind"), str(item.get("type"))))
            except Exception as e:
                res.refused.append(f"typology {item}: {e}")
        return res


def grade_typology(truth: Path, labelled: list[tuple[str, str, str]], families: set[str]) -> dict:
    """Precision and recall of the defender's typology labels against the world's fraud labels on the same entities
    (grader side). A label counts as right when the entity carries a label of that family."""
    import duckdb
    if not labelled:
        return {"labels": 0}
    c = duckdb.connect()
    rows = c.execute(f"""select subject, label from '{truth}/labels.parquet'
                         where subject in (select unnest(?::varchar[]))""",
                     [sorted({e for e, _, _ in labelled})]).fetchall()
    have: dict[str, set[str]] = {}
    for s, l in rows:
        if l in families:
            have.setdefault(s, set()).add(l)
    right = sum(1 for e, _, t in labelled if t in have.get(e, set()))
    fraud_named = sum(1 for e, _, t in labelled if t != "genuine")
    genuine_right = sum(1 for e, _, t in labelled if t == "genuine" and not have.get(e))
    return {"labels": len(labelled), "right_family": right, "named_fraud": fraud_named,
            "precision": round(right / fraud_named, 3) if fraud_named else None,
            "genuine_right": genuine_right}
