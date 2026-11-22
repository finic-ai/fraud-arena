"""Alert decisions: the alert-review cases in a decision-model format.

The agentic alert review gives a model tools and lets it investigate. Here every player gets the same complete case
packet (the alert, the customer's profile and timeline as of the case time, and who shares each of the customer's
keys) and answers three fixed multiple-choice questions with probabilities. No tools, one answer per case. It
measures judgment on given evidence, and suits decision models (calibrated probabilities over records and text) as
well as LLMs.

Players:
  baseline                 a heuristic over the packet's alert type (a floor)
  openrouter:<model id>    any LLM, one call, JSON probabilities
  grayson                  Finic's Grayson decision API (data + choice questions); key from GRAYSON_API_KEY
"""

import contextlib
import io
import json
import math
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .cli import Case, cmd_customer, LINK_KINDS

TYPOLOGIES = {
    "ato": "account takeover: someone else logged in to or took over the customer's account",
    "scam_victim": "the customer was talked into sending money themselves (romance, investment, impersonation, "
                   "tech-support, grandparent scams; diverted payroll)",
    "mule": "the customer's account receives and passes on other people's stolen money",
    "synthetic_identity": "the account was opened in a stolen or made-up identity (bust-outs, ACH/ACATS pulls)",
    "first_party": "the real customer commits the fraud (false claims or disputes, refund abuse, return fraud)",
    "card_compromise": "the customer's card details were stolen and used (card testing, CNP fraud, card-detail "
                       "phishing, wallet provisioning)",
    "promo_abuse": "accounts farmed for sign-up or referral bonuses",
    "check_fraud": "check kiting or counterfeit or altered checks",
    "other": "any other fraud (e.g. structuring)",
    "none": "no fraud",
}
QUESTIONS = {
    "disposition": {
        "instructions": "Does this alert involve fraud? Fraud means the customer is a perpetrator, a victim or a mule "
                        "of a fraud that had already started by the case time.",
        "criteria": {"fraud": "fraud is involved", "not_fraud": "no fraud: a benign look-alike"}},
    "typology": {
        "instructions": "Which kind of fraud is this (none if no fraud)?",
        "criteria": TYPOLOGIES},
    "customer_role": {
        "instructions": "What is this customer's role?",
        "criteria": {"perpetrator": "the customer commits or controls the fraud",
                     "victim": "the customer loses money or their account or card is abused",
                     "mule": "the customer's account passes on other people's stolen money",
                     "legitimate": "no fraud involving this customer"}},
}


_PACKET_LOCK = __import__("threading").Lock()


def packet(case_dir, max_chars=40000):
    """The case packet as text: alert, profile and timeline, and every link kind with key sizes. Built once per case
    and cached as packet.txt, so every player sees the identical packet (built under a lock: it captures stdout)."""
    cached = Path(case_dir) / "packet.txt"
    with _PACKET_LOCK:
        if cached.exists():
            return cached.read_text()
        text = _build_packet(case_dir, max_chars)
        cached.write_text(text)
        return text


def _build_packet(case_dir, max_chars):
    c = Case(case_dir)
    c._needs_keys = True  # build the link index on first use
    case = c.case
    a = case["alert"]
    out = io.StringIO()
    out.write(f"# Alert at Brindlemark Bank (case {case['case_id']})\n"
              f"Customer: {case['customer']}\nAlert type: {a['type']} ({a['monitor']})\nReason: {a['reason']}\n"
              f"Alert time: {a['time']}\nData as of: {case['case_time']}\n"
              f"Base rate: about {round(100 * case.get('fraud_share', 0.45))}% of alerts in this set involve fraud.\n\n")
    with contextlib.redirect_stdout(out):
        try:
            cmd_customer(c, case["customer"])
        except SystemExit as e:  # the customer isn't visible as of the case time (a case-builder issue)
            print(f"## Profile\n  ({e.code})")
        print("\n## Links: other customers sharing this customer's keys (with how many customers share each key)")
        from .links import links_of
        for kind in LINK_KINDS:
            for k, size, others in links_of(c.db, case["customer"], kind, limit=8):
                print(f"{kind} {k}: shared by {size} customer(s)" + ("" if others else " (only this customer)")
                      + (": " + ", ".join(o for o, _, _ in others) if others else "")
                      + (f" and {size - 1 - len(others)} more" if size - 1 > len(others) else ""))
    text = out.getvalue()
    return text if len(text) <= max_chars else text[:max_chars] + "\n... (packet truncated)"


# --- players ---------------------------------------------------------------------------------------------------

def _normalize(probs, options):
    p = {o: max(0.0, float(probs.get(o) or 0.0)) for o in options}
    s = sum(p.values())
    return {o: (v / s if s > 0 else 1 / len(options)) for o, v in p.items()}


def answer_baseline(case_dir, text):
    a = json.loads((Path(case_dir) / "case.json").read_text())["alert"]["type"]
    p = 0.6 if a in ("referral_velocity", "p2p_pass_through", "card_step_up", "new_device_contact_change") else 0.4
    return {"disposition": {"fraud": p, "not_fraud": 1 - p},
            "typology": _normalize({"none": 1 - p, "ato": p / 3, "card_compromise": p / 3, "first_party": p / 3},
                                   TYPOLOGIES),
            "customer_role": {"perpetrator": p / 2, "victim": p / 2, "mule": 0.0, "legitimate": 1 - p}}


LLM_SYSTEM = """You are a fraud analyst. You get one alert case packet from a bank and answer three questions with
probabilities. Reply with JSON only, exactly this shape (each question's probabilities sum to 1):
{"disposition": {"fraud": p, "not_fraud": p}, "typology": {<option>: p, ...}, "customer_role": {<option>: p, ...}}"""


def _question_text():
    lines = []
    for q, spec in QUESTIONS.items():
        lines.append(f"{q}: {spec['instructions']}")
        for o, meaning in spec["criteria"].items():
            lines.append(f"  {o}: {meaning}")
    return "\n".join(lines)


def answer_llm(model, text, *, retries=2):
    key = os.environ["OPENROUTER_API_KEY"]
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "usage": {"include": True},
            "messages": [{"role": "system", "content": LLM_SYSTEM},
                         {"role": "user", "content": text + "\n\n## Questions\n" + _question_text()}]}
    last = None
    for _ in range(retries + 1):
        try:
            req = Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with urlopen(req, timeout=180) as r:
                d = json.load(r)
            content = d["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", content, re.S)
            ans = json.loads(m.group(0))
            return ({q: _normalize(ans.get(q) or {}, QUESTIONS[q]["criteria"]) for q in QUESTIONS},
                    float((d.get("usage") or {}).get("cost") or 0.0))
        except Exception as e:  # noqa: BLE001  (a failed parse counts as uniform after retries)
            last = e
    return ({q: _normalize({}, QUESTIONS[q]["criteria"]) for q in QUESTIONS}, 0.0) if last else None


def answer_claude(model, text):
    """Claude through Claude Code with no tools (for smoke tests on the user's login): one answer."""
    import subprocess
    prompt = LLM_SYSTEM + "\n\n" + text + "\n\n## Questions\n" + _question_text()
    p = subprocess.run(["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config", "--mcp-config",
                        '{"mcpServers": {}}', "--setting-sources", "project", "--no-session-persistence",
                        "--max-turns", "1", "--output-format", "json"], input=prompt, capture_output=True, text=True,
                       timeout=600)
    d = json.loads(p.stdout)
    m = re.search(r"\{.*\}", d.get("result") or "", re.S)
    ans = json.loads(m.group(0)) if m else {}
    return {q: _normalize(ans.get(q) or {}, QUESTIONS[q]["criteria"]) for q in QUESTIONS}, float(d.get("total_cost_usd") or 0)


def answer_grayson(text):
    key = os.environ["GRAYSON_API_KEY"]
    body = {"model": "grayson-1", "data": {"case_packet": text},
            "questions": {q: {"type": "choice", "instructions": s["instructions"], "criteria": s["criteria"]}
                          for q, s in QUESTIONS.items()}}
    req = Request("https://api.finic.ai/v1/ask", data=json.dumps(body).encode(),
                  headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                           "User-Agent": "fraud-arena/0.1"})
    with urlopen(req, timeout=120) as r:
        d = json.load(r)
    out = {}
    for q, spec in QUESTIONS.items():
        a = (d.get("answers") or {}).get(q) or {}
        probs = a.get("probabilities") or {}
        if isinstance(probs, list):  # [{"option": .., "probability": ..}]
            probs = {x.get("option") or x.get("label"): x.get("probability") for x in probs}
        out[q] = _normalize(probs, spec["criteria"])
    return out, 0.0


def play(player, case_dir):
    text = packet(case_dir)
    started = time.time()
    if player == "baseline":
        ans, cost = answer_baseline(case_dir, text), 0.0
    elif player == "grayson":
        ans, cost = answer_grayson(text)
    elif player.startswith("openrouter:"):
        ans, cost = answer_llm(player.split(":", 1)[1], text)
    elif player.startswith("claude:"):
        ans, cost = answer_claude(player.split(":", 1)[1], text)
    else:
        raise ValueError(f"unknown player {player!r}")
    return {"answer": ans, "cost": cost, "seconds": round(time.time() - started, 1), "packet_chars": len(text)}


# --- scoring -----------------------------------------------------------------------------------------------------

def score(keys, results):
    """keys: case id -> answer key; results: case id -> play() result."""
    rows = []
    for cid, r in results.items():
        k = keys[cid]
        y = 1 if k["disposition"] == "fraud" else 0
        p = min(max(r["answer"]["disposition"]["fraud"], 1e-4), 1 - 1e-4)
        typ = max(r["answer"]["typology"], key=r["answer"]["typology"].get)
        role = max(r["answer"]["customer_role"], key=r["answer"]["customer_role"].get)
        rows.append({"case_id": cid, "y": y, "p": p, "typology_ok": typ == k["typology"] if y else None,
                     "role_ok": role == k["customer_role"],
                     "p_true_typology": r["answer"]["typology"].get(k["typology"], 0.0)})
    n = len(rows)
    if not n:
        return {}
    pos = [x["p"] for x in rows if x["y"]]
    neg = [x["p"] for x in rows if not x["y"]]
    auc = (sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))) if pos and neg else None
    bins = [[] for _ in range(5)]
    for x in rows:
        bins[min(int(x["p"] * 5), 4)].append(x)
    ece = sum(len(b) / n * abs(sum(x["p"] for x in b) / len(b) - sum(x["y"] for x in b) / len(b)) for b in bins if b)
    fraud = [x for x in rows if x["y"]]
    return {"n": n, "fraud_share": round(len(fraud) / n, 3),
            "accuracy": round(sum((x["p"] >= 0.5) == bool(x["y"]) for x in rows) / n, 3),
            "auc": round(auc, 3) if auc is not None else None,
            "log_loss": round(-sum(math.log(x["p"] if x["y"] else 1 - x["p"]) for x in rows) / n, 3),
            "brier": round(sum((x["p"] - x["y"]) ** 2 for x in rows) / n, 3),
            "ece": round(ece, 3),
            "typology_acc_on_fraud": round(sum(bool(x["typology_ok"]) for x in fraud) / len(fraud), 3) if fraud else None,
            "role_acc": round(sum(x["role_ok"] for x in rows) / n, 3),
            "rows": rows}
