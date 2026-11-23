"""Question-Bottleneck Scoring (QBS) on the alert cases, as in Finic's QBS whitepaper: a decision model answers a
pool of plain-language yes/no questions about the case packet (all in one call), each probability becomes a
standardized log-odds feature, and an L2-regularized logistic regression over a forward-selected handful of them,
plus a categorical prior (which monitor raised the alert), gives the score. Fitted on one set of seeds and frozen,
then scored on held-out seeds.
"""

import json
import math
import os
from pathlib import Path
from urllib.request import Request

from .decide import packet, _normalize, _post_with_backoff

QUESTIONS = {
    # holistic analyst judgments
    "real_customer_acting": "Would a careful analyst believe this activity is the real account holder acting for "
                            "themselves, as they normally do?",
    "identity_coherent": "Would a careful person looking at the name, email, phone and address believe they belong to "
                         "one real individual?",
    "identity_synthetic": "Does the identity look synthetic: real-looking pieces (name, address, phone, email) that "
                          "don't belong together?",
    "story_consistent": "Do the stated reasons, memos, notes and call notes tell one consistent story that fits the "
                        "payments and activity?",
    "routine_for_customer": "Is the alerted activity routine for this customer, given their history?",
    "analyst_would_escalate": "Would an experienced fraud analyst escalate this alert rather than close it?",
    # account takeover / access
    "someone_else_in_control": "Are there signs someone other than the account holder is in control of the account "
                               "(new device or network plus changed contact details, unrecognized activity)?",
    "new_device_risky": "Did the activity come from a device or network the customer hadn't used before, in a risky "
                        "way (SIM change, remote session, proxy or hosting network)?",
    "contact_changed_recently": "Did the email, phone, address or two-factor setting change shortly before the "
                                "alerted activity?",
    # scams
    "customer_coached": "Does it look like the customer was coached, rushed or deceived into making the payment?",
    "scam_language": "Do memos, notes, call notes or claim statements suggest a scam (romance, investment, "
                     "impersonation, tech support, a stranger's instructions)?",
    "payee_risky": "Is the payee or beneficiary new, shared by several other customers, or otherwise unusual for this "
                   "customer?",
    # first-party / disputes / claims
    "claim_contradicted": "Does the customer's own activity contradict their dispute or claim (their usual device, "
                          "the item delivered or used, a pattern of similar claims)?",
    "customer_benefits": "Does the account holder themselves gain from the suspicious activity (keeps the money, "
                         "makes false claims, withdraws quickly)?",
    "honest_mistake_likely": "Is an honest mistake or misunderstanding the most likely explanation?",
    # mules / money movement
    "pass_through": "Does money arrive from others and leave again quickly, with little ordinary activity in between?",
    "set_up_to_receive": "Does the account look set up to receive payments from scam victims or to pass on other "
                         "people's money?",
    "funds_from_unknown": "Does money arrive from sources that have no clear relationship to the customer?",
    # cards
    "card_used_by_other": "Are there signs the customer's card details were used by someone else (test "
                          "authorizations, an unfamiliar device, merchants the customer never used)?",
    # onboarding / promo / synthetic
    "new_account_quick_value": "Was the account opened recently and used quickly to extract value (bonuses, "
                               "credit, transfers out)?",
    "bonus_farming": "Does the account look farmed for sign-up or referral bonuses?",
    "shares_rare_identifiers": "Does the customer share a device, phone, address or payee with only a few other "
                               "customers in a way that suggests coordination?",
    # ACH / checks / cash
    "ach_or_check_abuse": "Are there ACH returns, reversed deposits or check problems that suggest abuse rather than "
                          "a cash-flow mistake?",
    "structuring": "Are cash deposits or withdrawals kept just under reporting thresholds or split across days?",
    # benign explanations
    "benign_explanation": "Is there an obvious harmless explanation (travel, a new phone, a household member, a known "
                          "payee, a routine bill or purchase)?",
    "long_stable_history": "Does the customer have a long, stable history with the bank up to the alert?",
}
YES_NO = {"yes": "the statement holds", "no": "it doesn't hold"}


def ask(player, case_dir):
    """p(yes) for every question, in one call, and the cost."""
    text = packet(case_dir)
    if player == "grayson":
        url, key, field, model = "https://api.finic.ai/v1/ask", os.environ["GRAYSON_API_KEY"], "data", "grayson-1"
        qs = {q: {"type": "choice", "instructions": t, "criteria": YES_NO} for q, t in QUESTIONS.items()}
    elif player.startswith("jev"):
        url, key, field = "https://openrouter.ai/api/v1/systemone", os.environ["OPENROUTER_API_KEY"], "state"
        model = "typesafe/" + (player.split(":", 1)[1] if ":" in player else "jev-1.14")
        qs = {q: {"type": "noul", "instructions": t} for q, t in QUESTIONS.items()}
    else:
        raise ValueError(player)
    body = {"model": model, field: {"case_packet": text}, "questions": qs}
    req = Request(url, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key,
                  "Content-Type": "application/json", "User-Agent": "fraud-arena/0.1"})
    d = _post_with_backoff(req)
    out = {}
    for q in QUESTIONS:
        a = (d.get("answers") or {}).get(q) or {}
        if "noul" in a:
            out[q] = float(a["noul"])
        else:
            out[q] = _normalize(a.get("probabilities") or {}, YES_NO)["yes"]
    return out, float((d.get("usage") or {}).get("cost") or 0.0)


def _logit(p):
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def design(rows, questions, monitors):
    """Standardized log-odds of the chosen questions (standardization from the rows given) plus monitor one-hots."""
    import numpy as np
    X = np.array([[_logit(r["answers"][q]) for q in questions] for r in rows]) if questions else np.zeros((len(rows), 0))
    M = np.array([[1.0 if r["monitor"] == m else 0.0 for m in monitors] for r in rows])
    return X, M


class QBS:
    """Forward selection on cross-validated AUC, then an L2 logistic fit, frozen."""

    def __init__(self, C=1.0, max_questions=8, min_gain=0.002):
        self.C, self.max_questions, self.min_gain = C, max_questions, min_gain

    def _fit(self, rows, questions):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X, M = design(rows, questions, self.monitors)
        mu, sd = (X.mean(0), X.std(0) + 1e-9) if questions else (None, None)
        Z = np.hstack([(X - mu) / sd if questions else X, M])
        y = np.array([r["y"] for r in rows])
        clf = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y)
        return clf, mu, sd

    def _cv_auc(self, rows, questions, groups):
        import numpy as np
        from sklearn.metrics import roc_auc_score
        ys, ps = [], []
        for g in sorted(set(groups)):
            tr = [r for r, gg in zip(rows, groups) if gg != g]
            te = [r for r, gg in zip(rows, groups) if gg == g]
            clf, mu, sd = self._fit(tr, questions)
            ps += list(self._predict(clf, mu, sd, te, questions))
            ys += [r["y"] for r in te]
        return roc_auc_score(ys, ps)

    def _predict(self, clf, mu, sd, rows, questions):
        import numpy as np
        X, M = design(rows, questions, self.monitors)
        Z = np.hstack([(X - mu) / sd if questions else X, M])
        return clf.predict_proba(Z)[:, 1]

    def fit(self, rows, groups):
        self.monitors = sorted({r["monitor"] for r in rows})
        chosen, best, self.path = [], self._cv_auc(rows, [], groups), []
        self.path.append(("(monitor prior only)", best))
        while len(chosen) < self.max_questions:
            cand = [(self._cv_auc(rows, chosen + [q], groups), q) for q in QUESTIONS if q not in chosen]
            auc, q = max(cand)
            if auc - best < self.min_gain:
                break
            chosen.append(q); best = auc; self.path.append((q, auc))
        self.questions = chosen
        self.clf, self.mu, self.sd = self._fit(rows, chosen)
        return self

    def predict(self, rows):
        return self._predict(self.clf, self.mu, self.sd, rows, self.questions)
