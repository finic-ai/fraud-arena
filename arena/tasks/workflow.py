"""Alert decisions, workflow mode: the decision decomposed into small yes/no evidence questions per alert family,
answered by the model about the same case packet as the direct mode, and combined by fixed code.

Two combiners, both fixed before any official case is scored:
  hand   the same hand-written weights for every model (TypeSafe-style workflow logic): the score starts at the
         real-world base rate and each answer moves its log-odds by its weight times (2 p_yes - 1)
  tuned  each model's own logistic weights, fit once on the calibration seeds and frozen (fit_tuned / save)
"""

import json
import math
import os
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from .decide import packet, _normalize, _post_with_backoff
from .schema import REAL_WORLD_ALERT_FRAUD_RATE

FAMILY_OF = {
    "card_fraud_dispute": "card", "card_nonreceipt_dispute": "card", "card_step_up": "card",
    "risky_login": "access", "new_device_contact_change": "access",
    "p2p_unauthorized_claim": "payments", "p2p_hold": "payments", "wire_callback": "payments",
    "ach_push_callback": "payments", "fraud_report_call": "payments",
    "ach_return": "incoming", "p2p_pass_through": "incoming", "cash_near_threshold": "incoming",
    "application_referred": "onboarding", "referral_velocity": "onboarding",
}

# family -> question -> (text, hand weight). Positive weights point to fraud, negative to a benign explanation.
QUESTIONS = {
    "card": {
        "compromise_signs": ("Are there signs the card details were stolen and used by someone else (small test "
                             "authorizations, a new device or network, a merchant the customer never used, many "
                             "purchases in a short time)?", 1.0),
        "claim_contradicted": ("Does the customer's own activity contradict their dispute or claim (made from their "
                               "own usual device, the item delivered or used, a pattern of similar disputes)?", 1.0),
        "details_phished": ("Do call notes or messages suggest someone tricked the customer into sharing card "
                            "details or a one-time code?", 0.8),
        "honest_mistake": ("Does it look like an honest mistake (a merchant name they don't recognize but did use, "
                           "a recurring charge, a family member's purchase)?", -1.0),
        "routine_history": ("Does the customer have a long, consistent card history at similar merchants and "
                            "amounts?", -0.5),
    },
    "access": {
        "new_device_network": ("Did the login come from a device or network the customer had not used before?", 0.8),
        "sim_or_remote": ("Is there a SIM change, a remote-access session, or a hosting, VPN or proxy network "
                          "around the login?", 0.8),
        "contact_change": ("Did the email, phone, address or two-factor setting change around the login?", 1.0),
        "money_out_after": ("Did money leave the account soon after (transfers, new payees, a new card or wallet "
                            "token)?", 1.0),
        "customer_unrecognized": ("Did the customer report activity they don't recognize?", 1.0),
        "benign_explanation": ("Is there a harmless explanation (travel, a new phone, the customer's usual "
                               "pattern)?", -1.0),
    },
    "payments": {
        "new_payee": ("Is this one of the first payments to this payee?", 0.5),
        "payee_shared": ("Do several other customers also pay this payee?", 0.8),
        "scam_language": ("Do memos, notes, call notes or claim statements suggest a scam (romance, investment, "
                          "impersonation, tech support, a stranger's instructions)?", 1.0),
        "coached_or_rushed": ("Does it look like the customer was coached or rushed (escalating amounts, urgency, "
                              "a call just before the payment)?", 0.8),
        "unfamiliar_device": ("Was the payment made from a device or network the customer doesn't usually use?", 0.8),
        "known_relationship": ("Is the payee someone the customer has an ongoing relationship with (repeat "
                               "payments, household, a known business)?", -1.0),
    },
    "incoming": {
        "in_then_out": ("Does money arrive from a new source and leave again within a few days?", 1.0),
        "returns_reversals": ("Are there ACH returns, or deposits that were pulled back or reversed?", 0.8),
        "new_account": ("Was the account opened within 30 days before the alert?", 0.6),
        "structuring": ("Are cash amounts kept just below $10,000 or split across days?", 0.8),
        "bonus_then_drain": ("Did the account receive a sign-up or referral bonus and then send most of its money "
                             "out?", 0.8),
        "regular_income": ("Is the incoming money regular pay or other routine income the customer has had for "
                           "months?", -1.0),
    },
    "onboarding": {
        "weak_identity": ("Do the identity checks look weak (low document or ID match, a very new email or phone, "
                          "a name that doesn't match the email)?", 1.0),
        "rare_shared_identifiers": ("Does the customer share an address, phone or device with only a few other "
                                    "customers?", 1.0),
        "referral_chain": ("Was the customer referred within a chain of new accounts referring each other?", 0.8),
        "lookalike_contacts": ("Do the email or phone follow a pattern shared with other new accounts?", 0.6),
        "unknown_employer_pay": ("Does pay arrive from an employer that pays few other customers, soon after "
                                 "opening?", 0.6),
        "ordinary_applicant": ("Does it look like an ordinary applicant (consistent identity, normal first "
                               "activity)?", -1.0),
    },
}
YES_NO = {"yes": "the statement holds", "no": "it doesn't hold"}
SYSTEM = """You are a fraud analyst. You get one alert case packet from a bank and answer each evidence question
with the probability that the answer is yes. Reply with JSON only: {"<question_name>": p_yes, ...} for every
question, each p_yes between 0 and 1."""


def family(case_dir):
    return FAMILY_OF[json.loads((Path(case_dir) / "case.json").read_text())["alert"]["type"]]


def ask(player, case_dir):
    """The player's p(yes) for every question of the case's family, and the cost."""
    text = packet(case_dir)
    qs = QUESTIONS[family(case_dir)]
    if player == "grayson" or player.startswith("jev"):
        if player == "grayson":
            url, key, field, model = ("https://api.finic.ai/v1/ask", os.environ["GRAYSON_API_KEY"], "data", "grayson-1")
        else:
            url, key, field = "https://openrouter.ai/api/v1/systemone", os.environ["OPENROUTER_API_KEY"], "state"
            model = "typesafe/" + (player.split(":", 1)[1] if ":" in player else "jev-1.14")
        body = {"model": model, field: {"case_packet": text},
                "questions": {q: {"type": "choice", "instructions": t, "criteria": YES_NO} for q, (t, _) in qs.items()}}
        req = Request(url, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key,
                      "Content-Type": "application/json", "User-Agent": "fraud-arena/0.1"})
        d = _post_with_backoff(req)
        out = {q: _normalize(((d.get("answers") or {}).get(q) or {}).get("probabilities") or {}, YES_NO)["yes"]
               for q in qs}
        return out, float((d.get("usage") or {}).get("cost") or 0.0)
    prompt = SYSTEM + "\n\n" + text + "\n\n## Evidence questions\n" + "\n".join(f"{q}: {t}" for q, (t, _) in qs.items())
    if player.startswith("claude:"):
        p = subprocess.run(["claude", "-p", "--model", player.split(":", 1)[1], "--tools", "", "--strict-mcp-config",
                            "--mcp-config", '{"mcpServers": {}}', "--setting-sources", "project",
                            "--no-session-persistence", "--max-turns", "1", "--output-format", "json"],
                           input=prompt, capture_output=True, text=True, timeout=600)
        d = json.loads(p.stdout)
        content, cost = d.get("result") or "", float(d.get("total_cost_usd") or 0)
    elif player.startswith("openrouter:"):
        body = {"model": player.split(":", 1)[1], "temperature": 0, "response_format": {"type": "json_object"},
                "usage": {"include": True}, "messages": [{"role": "user", "content": prompt}]}
        req = Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                               "Content-Type": "application/json"})
        with urlopen(req, timeout=180) as r:
            d = json.load(r)
        content, cost = d["choices"][0]["message"]["content"] or "", float((d.get("usage") or {}).get("cost") or 0)
    else:
        raise ValueError(player)
    m = re.search(r"\{.*\}", content, re.S)
    ans = json.loads(m.group(0)) if m else {}
    out = {}
    for q in qs:
        try:
            out[q] = min(max(float(ans.get(q)), 0.0), 1.0)
        except (TypeError, ValueError):
            out[q] = 0.5
    return out, cost


def hand_score(fam, answers):
    """The hand-written workflow: start at the base rate; each answer moves the log-odds by weight x (2p - 1)."""
    z = math.log(REAL_WORLD_ALERT_FRAUD_RATE / (1 - REAL_WORLD_ALERT_FRAUD_RATE))
    for q, (_, w) in QUESTIONS[fam].items():
        z += w * (2 * answers.get(q, 0.5) - 1)
    return 1 / (1 + math.exp(-z))


def _features(fam, answers):
    """Tuned-combiner features: family indicators, and each family's answers as log-odds (0 for other families)."""
    row = [1.0 if f == fam else 0.0 for f in QUESTIONS]
    for f, qs in QUESTIONS.items():
        for q in qs:
            p = min(max(answers.get(q, 0.5), 0.01), 0.99) if f == fam else 0.5
            row.append(math.log(p / (1 - p)))
    return row


def fit_tuned(rows, prior_sample, C=0.5):
    """Fit a model's tuned combiner on calibration cases: rows = [(family, answers, y)]. Benign cases are weighted so
    the fit sees the real-world base rate, making the output calibrated to it. Returns a JSON-able combiner."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    X = np.array([_features(f, a) for f, a, _ in rows])
    y = np.array([yy for _, _, yy in rows])
    P, N = y.sum(), len(y) - y.sum()
    w_neg = (P / REAL_WORLD_ALERT_FRAUD_RATE - P) / N
    clf = LogisticRegression(C=C, max_iter=5000, fit_intercept=False).fit(X, y, sample_weight=np.where(y == 1, 1.0, w_neg))
    return {"coef": clf.coef_[0].round(6).tolist(), "families": list(QUESTIONS),
            "questions": [f"{f}.{q}" for f, qs in QUESTIONS.items() for q in qs], "n": len(rows), "fraud": int(P)}


def tuned_score(combiner, fam, answers):
    z = sum(c * x for c, x in zip(combiner["coef"], _features(fam, answers)))
    return 1 / (1 + math.exp(-z))
