"""The bundle a defender deploys at a period boundary, and its checks before the engine sees it.

Layout, in the agent's workspace:

    bundle/policy.yaml     a Runmere deployed policy: rules over the decision's observation, optionally reading
                           model scores (the engine checks names, types and hidden data at deploy)
    bundle/models/*.json   models the policy declares with `file:`: a model's fields (`kind`, `features`, `weights`
                           or `trees`, ...) or just the trees list; `fa-model` writes them from scikit-learn or
                           XGBoost. Inlined here so the engine gets one document
    bundle/NOTES.md        optional: what changed and why (kept with the run, never scored)

The harness only checks shape and limits; the engine's deploy is the authority, and its error goes back to the agent
at the next boundary while the previous bundle stays deployed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_BYTES = 20 * 1024 * 1024
MAX_RULES = 2000


class BundleError(Exception):
    pass


@dataclass(frozen=True)
class Bundle:
    policy: dict
    digest: str
    notes: str


def load(bundle_dir: Path, *, role: str, decisions: set[str]) -> Bundle:
    path = bundle_dir / "policy.yaml"
    if not path.exists():
        raise BundleError("bundle/policy.yaml is missing")
    size = sum(p.stat().st_size for p in bundle_dir.rglob("*") if p.is_file())
    if size > MAX_BYTES:
        raise BundleError(f"bundle is {size} bytes; the limit is {MAX_BYTES}")
    try:
        policy = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise BundleError(f"policy.yaml isn't valid YAML: {e}") from None
    if not isinstance(policy, dict):
        raise BundleError("policy.yaml must be a mapping")
    if policy.get("role", role) != role:
        raise BundleError(f"this task deploys into role {role!r}, not {policy.get('role')!r}")
    policy.pop("role", None)
    policy.setdefault("format", 1)
    rules = policy.get("rules", [])
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        raise BundleError(f"rules must be a list of at most {MAX_RULES}")
    for i, r in enumerate(rules):
        d = (r or {}).get("decision")
        if d not in decisions:
            raise BundleError(f"rule {i}: decision {d!r} isn't one of this role's: {sorted(decisions)}")
    models = policy.get("models") or {}
    if not isinstance(models, dict):
        raise BundleError("models must be a mapping of name to model")
    for name, m in models.items():
        if isinstance(m, dict) and "file" in m:
            f = (bundle_dir / m["file"]).resolve()
            if bundle_dir.resolve() not in f.parents or not f.exists():
                raise BundleError(f"model {name}: file {m['file']!r} isn't in the bundle")
            try:
                body = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                raise BundleError(f"model {name}: {m['file']} isn't JSON: {e}") from None
            rest = {k: v for k, v in m.items() if k != "file"}
            models[name] = rest | ({"trees": body} if isinstance(body, list) else body)
    canon = json.dumps(policy, sort_keys=True, default=str).encode()
    notes = (bundle_dir / "NOTES.md").read_text()[:20000] if (bundle_dir / "NOTES.md").exists() else ""
    return Bundle(policy, hashlib.sha256(canon).hexdigest(), notes)


def digest_files(bundle_dir: Path, policy_digest: str) -> str:
    """The deployed bundle's identity: the policy, and every file a workflow can load (reviews.json is per turn)."""
    h = hashlib.sha256(policy_digest.encode())
    for f in sorted(p for p in bundle_dir.rglob("*") if p.is_file() and p.name != "reviews.json"
                    and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(bundle_dir)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def external_rule(decision: str) -> dict:
    """The rule that sends every case at `decision` to the defender's workflow."""
    return {"decision": decision, "action": "external"}


def take_grayson(policy: dict) -> dict[str, dict]:
    """Grayson at the gates: a deployment's `external: {<decision>: {timeout, grayson: {...}}}` sends that decision's
    waiting cases to Grayson, answered by the harness at the end of each simulated day. The engine sees only the
    timeout; the harness keeps the Grayson part, in one of two shapes:
    - one question whose choice is the action: `{instructions, criteria: {<action>: <meaning>}}`;
    - several questions combined with the case's features in a logistic layer, with score bands mapped to actions:
      `{questions: {<q>: {instructions, criteria: {<option>: <meaning>}}}, combine: {intercept, weights:
      {"<q>.<option>" | "<feature>": w}, bands: [{min: p, action}], otherwise: <action>}}`."""
    out = {}
    for decision, ext in (policy.get("external") or {}).items():
        if not (isinstance(ext, dict) and "grayson" in ext):
            continue
        g = ext.pop("grayson") or {}
        if "questions" in g:
            qs = {str(k): {"instructions": str(v.get("instructions", "")), "criteria": dict(v.get("criteria") or {})}
                  for k, v in (g.get("questions") or {}).items()}
            comb = g.get("combine") or {}
            bands = sorted(({"min": float(b["min"]), "action": str(b["action"])} for b in comb.get("bands", [])),
                           key=lambda b: -b["min"])
            if not qs or not bands or "otherwise" not in comb:
                raise BundleError(f"grayson at {decision}: questions need combine.bands and combine.otherwise")
            out[decision] = {"questions": qs, "combine": {
                "intercept": float(comb.get("intercept", 0.0)),
                "weights": {str(k): float(v) for k, v in (comb.get("weights") or {}).items()},
                "bands": bands, "otherwise": str(comb["otherwise"])}}
        else:
            out[decision] = {"instructions": str(g.get("instructions", "")), "criteria": dict(g.get("criteria") or {})}
    return out


def grayson_action(q: dict, features: dict, answers: dict, available) -> str | None:
    """The action for one waiting case from Grayson's answers (and, when combining, the case's features)."""
    import math
    if "questions" not in q:
        choice = ((answers or {}).get("decision") or {}).get("choice")
        return choice if choice in available else None
    c = q["combine"]
    z = c["intercept"]
    for key, w in c["weights"].items():
        if "." in key and key.split(".", 1)[0] in q["questions"]:
            qn, opt = key.split(".", 1)
            probs = ((answers or {}).get(qn) or {}).get("probabilities") or {}
            x = probs.get(opt)
            if x is None:
                x = 1.0 if ((answers or {}).get(qn) or {}).get("choice") == opt else 0.0
        else:
            v = features.get(key)
            x = float(v) if isinstance(v, (int, float, bool)) else 0.0
        z += w * float(x)
    p = 1.0 / (1.0 + math.exp(-z))
    for band in c["bands"]:
        if p >= band["min"] and band["action"] in available:
            return band["action"]
    return c["otherwise"] if c["otherwise"] in available else None


def split_by_role(policy: dict, role_of: dict[str, str]) -> dict[str, dict]:
    """One deployment per role (the engine deploys into one role at a time): each role's rules, and the models its
    rules read (`score.<model>`)."""
    out: dict[str, dict] = {}
    models = policy.get("models") or {}
    for r in policy.get("rules", []):
        role = role_of[r["decision"]]
        out.setdefault(role, {k: v for k, v in policy.items() if k not in ("rules", "models", "external")} | {"rules": []})
        out[role]["rules"].append(r)
    for decision, ext in (policy.get("external") or {}).items():
        role = role_of.get(decision)
        if role in out:
            out[role].setdefault("external", {})[decision] = ext
    for role, dep in out.items():
        text = json.dumps(dep["rules"])
        used = {m: models[m] for m in models if f"score.{m}" in text}
        if used:
            dep["models"] = used
    return out
