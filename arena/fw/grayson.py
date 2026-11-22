"""Grayson as an optional defender tool, through Runmere's outside features (SPEC §8.8, deployment format 4).

The defender's policy may carry a `grayson` section: questions Grayson answers about a case, each with its options.
Rules read the answers as `outside.<question>.<option>` (a probability). The engine asks only when a rule's evaluation
reaches such a read, so cheap conditions first keep the calls down. The harness answers each case by calling Grayson
with the case's observed features; a weekly call budget caps the cost, and cases past it get each option at 1/n.

    "grayson": {
      "scam_payment": {"question": "Is this outgoing payment a scam?", "options": {"yes": "...", "no": "..."},
                       "decisions": ["brind_payment_hold"]}
    }
"""

import json
import os
import threading
import time
from urllib.request import Request, urlopen

ENDPOINT = "https://api.finic.ai/v1/ask"
NAME_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def check_section(section, decisions):
    problems = []
    if not isinstance(section, dict) or not section:
        return ['"grayson" must map question names to {question, options, decisions}']
    names = {d["name"] for d in decisions}
    for q, spec in section.items():
        if not q or set(q) - NAME_OK:
            problems.append(f"grayson.{q}: names are lower case letters, digits and _")
        if not isinstance(spec, dict) or not isinstance(spec.get("question"), str) or not spec["question"].strip():
            problems.append(f"grayson.{q}: needs a \"question\" (text)")
            continue
        opts = spec.get("options")
        if not isinstance(opts, dict) or not 2 <= len(opts) <= 6 or any(set(o) - NAME_OK for o in opts):
            problems.append(f"grayson.{q}.options: 2 to 6 options, each name -> what it means (names lower case)")
        ds = spec.get("decisions")
        if not isinstance(ds, list) or not ds or any(d not in names for d in ds):
            problems.append(f"grayson.{q}.decisions: the decisions whose rules read it, from {sorted(names)}")
    return problems


def to_deployment(policy):
    """The defender's policy as a Runmere deployment: format 1 without Grayson, format 4 with it."""
    section = policy.get("grayson")
    dep = {k: v for k, v in policy.items() if k != "grayson"}
    if not section:
        return dep
    dep["format"] = 4
    dep["outside"] = {q: {"options": list(spec["options"])} for q, spec in section.items()}
    dep["external"] = {d: {} for spec in section.values() for d in spec["decisions"]}
    return dep


class GraysonRunner:
    """Answers the engine's outside-feature requests by calling Grayson, within a weekly budget."""

    def __init__(self, section, *, key=None, budget_per_week=200, log=None):
        self.section = section or {}
        self.key = key or os.environ.get("GRAYSON_API_KEY", "")
        self.budget, self.log = budget_per_week, log
        self.lock = threading.Lock()
        self.cache, self.calls_this_week, self.stats = {}, 0, {"calls": 0, "cached": 0, "over_budget": 0, "errors": 0}

    def new_week(self):
        self.calls_this_week = 0

    def __call__(self, case, view):
        answers = {}
        for q in case.get("needs", []):
            spec = self.section[q]
            answers[q] = self._ask(q, spec, case)
        return answers

    def _ask(self, q, spec, case):
        feats = case.get("features") or {}
        key = (q, json.dumps(feats, sort_keys=True, default=str))
        with self.lock:
            if key in self.cache:
                self.stats["cached"] += 1
                return self.cache[key]
            if self.calls_this_week >= self.budget or not self.key:
                self.stats["over_budget"] += 1
                return {o: 1.0 / len(spec["options"]) for o in spec["options"]}
            self.calls_this_week += 1
            self.stats["calls"] += 1
        data = {"decision": case.get("decision"), "case_features": feats}
        body = {"model": "grayson-1", "data": data,
                "questions": {q: {"type": "choice", "instructions": spec["question"], "criteria": spec["options"]}}}
        started = time.time()
        try:
            req = Request(ENDPOINT, data=json.dumps(body, default=str).encode(),
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                                   "User-Agent": "fraud-arena/0.1"})
            with urlopen(req, timeout=30) as resp:
                answer = json.load(resp)
            probs = parse_probabilities(answer, q, list(spec["options"]))
        except Exception as error:
            with self.lock:
                self.stats["errors"] += 1
            probs = {o: 1.0 / len(spec["options"]) for o in spec["options"]}
            answer = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
        if self.log:
            with self.lock, open(self.log, "a") as f:
                f.write(json.dumps({"q": q, "features": feats, "probabilities": probs, "seconds": round(time.time() - started, 2),
                                    "raw": answer}, default=str) + "\n")
        with self.lock:
            self.cache[key] = probs
        return probs


def parse_probabilities(answer, q, options):
    """Grayson's reply to probabilities per option. Accepts {"answers": {q: {"probabilities": {...}}}}, or the option
    map directly under the question; anything else raises, so the error is visible in the log."""
    a = (answer.get("answers") or answer).get(q, answer.get(q))
    if isinstance(a, dict):
        for k in ("probabilities", "distribution", "options", "probs"):
            if isinstance(a.get(k), dict):
                a = a[k]
                break
    if not isinstance(a, dict) or not all(o in a for o in options):
        raise ValueError(f"unexpected Grayson answer for {q}: {str(answer)[:300]}")
    total = sum(float(a[o]) for o in options) or 1.0
    return {o: max(0.0, min(1.0, float(a[o]) / total)) for o in options}
