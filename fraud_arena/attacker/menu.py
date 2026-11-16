"""The ring plan menu: what a ring operator may choose for a period, as Fraud World declares it.

Fraud World declares each plan attribute of `ring` in `env.interventions` (SPEC §8.8) with its type and range. This
module reads those declarations, builds the JSON Schema of one ring's `plan.json` from them, validates a plan, and
turns it into the interventions that the harness sets at the boundary. A plan only chooses among things the ring's
scripted machinery already does; the agent writes no fraud content.

A plan key that is left out means "as scripted": every boundary sets every plan attribute, so a key dropped from one
period's plan goes back to the ring's scripted behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Each ring's schemes (DESIGN.md §5.8) by the short name of their tactic weight (`plan_tactic_<name>`), the
# institutions its schemes choose among by the ring's memory, and the "with what" switches that its schemes read.
# `check_menu` verifies these against the world's declarations.
RINGS: dict[int, dict] = {
    1: {"tactics": ["bank_hopping", "mule_flow"], "where": ["arbelton", "brindlemark"],
        "with": ["devices", "exits", "identities", "mule_target"]},
    2: {"tactics": ["phishing_campaign", "vishing_card_details", "courier_ato", "delivery_customer_ato", "ato_phishing",
                    "mule_flow"],
        "where": ["arbelton", "brindlemark", "cedarmoor", "dovrex", "everhollow"],
        "with": ["devices", "mule_target"]},
    3: {"tactics": ["corporate_ato", "bec_vendor", "payroll_diversion", "mule_flow"], "where": [],
        "with": ["devices", "mule_target"]},
    4: {"tactics": ["pig_butchering", "romance"], "where": [], "with": []},
    5: {"tactics": ["synthetic_bustout"], "where": ["arbelton", "brindlemark", "cedarmoor"], "with": []},
    6: {"tactics": ["card_testing", "cnp_fraud", "triangulation"], "where": [], "with": ["devices", "exits"]},
    7: {"tactics": ["ach_return"], "where": [], "with": ["devices", "exits", "identities"]},
    8: {"tactics": ["mule_flow"], "where": [], "with": ["devices", "mule_target"]},
    9: {"tactics": ["grandparent", "tech_support"], "where": [], "with": []},
    10: {"tactics": ["refund_service", "promo_farm", "courier_rental"], "where": [], "with": ["devices"]},
}

DESCRIPTIONS = {
    "volume": "Attempts this period as a multiple of the ring's scripted level, for all its schemes. 1 is as scripted.",
    "quiet": "Stop every scheme for the period (true), or run (false).",
    "timing": "`scripted`: the ring's usual timing with its campaign waves. `burst`: the same weekly volume, all of it "
              "on Fridays and Saturdays.",
    "where": "Shares of the ring's institution choices, for the schemes that choose an institution. Leave it out to "
             "choose by what the ring has seen succeed (scripted). An institution left out gets none.",
    "tactics": "Weight of each scheme relative to its scripted level (1). 0 stops a scheme, 2 doubles it.",
    "devices": "`own`: members stick to the devices they operate. `spread`: use the ring's whole device pool evenly. "
               "`scripted`: as usual.",
    "exits": "`local`: proxy exits in the victim's region. `any`: any of the ring's exits. `scripted`: as usual.",
    "identities": "For new accounts: `stolen` copies of real people only, `synthetic` identities only, or `scripted`.",
    "mule_target": "How many mules the ring recruits in all; above the number it has, it recruits more (to replace "
                   "lost ones). Leave it out for the scripted target.",
}


@dataclass(frozen=True)
class Declared:
    attribute: str
    kind: str  # bool, float, int, code
    min: float | None = None
    max: float | None = None
    values: tuple[str, ...] = ()
    clear: bool = False


def declarations(world: Path) -> dict[str, Declared]:
    """The world's declared interventions on `ring`, by attribute name, typed from the attributes."""
    p = Path(world).resolve()
    return dict(_declarations(str(p), p.stat().st_mtime_ns))


@lru_cache(maxsize=8)
def _declarations(path: str, _mtime: int) -> dict[str, Declared]:
    spec = yaml.load(Path(path).read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    attrs = {}
    for k, a in spec["world"]["entities"]["ring"]["attributes"].items():
        optional = k.endswith("?") or bool(a.get("optional"))
        attrs[k.rstrip("?")] = (a, optional)
    out: dict[str, Declared] = {}
    for d in (spec.get("env") or {}).get("interventions", {}).values():
        if d["entity"] != "ring":
            continue
        a, optional = attrs[d["attribute"]]
        kind = str(a["type"]).split("|")[0].strip()
        values = tuple(d.get("values") or a.get("values") or ())
        out[d["attribute"]] = Declared(d["attribute"], kind, d.get("min"), d.get("max"), values,
                                       optional or "null" in str(a["type"]))
    return out


def check_menu(decl: dict[str, Declared]) -> list[str]:
    """What `RINGS` names that the world doesn't declare."""
    need = {"plan_volume", "plan_quiet", "plan_timing", "plan_where", "plan_devices", "plan_exits", "plan_identities",
            "plan_mule_target"}
    for r in RINGS.values():
        need |= {f"plan_tactic_{t}" for t in r["tactics"]} | {f"plan_share_{i}" for i in r["where"]}
    return sorted(n for n in need if n not in decl)


def _number(d: Declared, description: str) -> dict:
    return {"type": "integer" if d.kind == "int" else "number", "minimum": d.min, "maximum": d.max,
            "description": description}


def _code(d: Declared, description: str) -> dict:
    return {"enum": list(d.values), "description": description}


def plan_schema(world: Path, ring: int) -> dict:
    """The JSON Schema of `plan.json` for one ring, from the world's declarations."""
    decl = declarations(world)
    missing = check_menu(decl)
    if missing:
        raise ValueError(f"the world doesn't declare: {', '.join(missing)}")
    menu = RINGS[ring]
    props: dict = {
        "volume": _number(decl["plan_volume"], DESCRIPTIONS["volume"]),
        "quiet": {"type": "boolean", "description": DESCRIPTIONS["quiet"]},
        "timing": _code(decl["plan_timing"], DESCRIPTIONS["timing"]),
        "tactics": {
            "type": "object", "description": DESCRIPTIONS["tactics"], "additionalProperties": False,
            "properties": {t: _number(decl[f"plan_tactic_{t}"], f"Weight of {t}.") for t in menu["tactics"]},
        },
        "notes": {"type": "string", "maxLength": 4000,
                  "description": "Your reasoning, for the record. The world doesn't read it."},
    }
    if menu["where"]:
        props["where"] = {
            "type": "object", "description": DESCRIPTIONS["where"], "additionalProperties": False, "minProperties": 1,
            "properties": {i: _number(decl[f"plan_share_{i}"], f"Share for {i}.") for i in menu["where"]},
        }
    for w in menu["with"]:
        d = decl[f"plan_{w}"]
        props[w] = _number(d, DESCRIPTIONS[w]) if d.kind in ("int", "float") else _code(d, DESCRIPTIONS[w])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Plan of ring R{ring} for one period",
        "type": "object",
        "additionalProperties": False,
        "properties": props,
    }


def validate_plan(plan: object, schema: dict) -> list[str]:
    """Errors of a plan against its schema (the subset of JSON Schema that `plan_schema` writes)."""
    errors: list[str] = []

    def check(v, s, path):
        if "enum" in s:
            if v not in s["enum"]:
                errors.append(f"{path}: {v!r} isn't one of {s['enum']}")
            return
        t = s.get("type")
        if t == "object":
            if not isinstance(v, dict):
                errors.append(f"{path}: an object")
                return
            for k in v:
                if k not in s["properties"]:
                    errors.append(f"{path}: no key {k!r} in the menu")
            if len(v) < s.get("minProperties", 0):
                errors.append(f"{path}: at least {s['minProperties']} key")
            for k, sub in s["properties"].items():
                if k in v:
                    check(v[k], sub, f"{path}.{k}")
        elif t == "boolean":
            if not isinstance(v, bool):
                errors.append(f"{path}: true or false")
        elif t in ("number", "integer"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (t == "integer" and not float(v).is_integer()):
                errors.append(f"{path}: a{'n integer' if t == 'integer' else ' number'}")
            elif not (s["minimum"] <= v <= s["maximum"]):
                errors.append(f"{path}: {v} is outside [{s['minimum']}, {s['maximum']}]")
        elif t == "string":
            if not isinstance(v, str) or len(v) > s.get("maxLength", 10**9):
                errors.append(f"{path}: a string of at most {s.get('maxLength')} characters")

    check(plan, schema, "plan")
    if not errors and isinstance(plan.get("where"), dict) and sum(plan["where"].values()) <= 0:
        errors.append("plan.where: the shares sum to more than 0")
    return errors


def interventions(plan: dict, ring: int) -> list[tuple[str, object]]:
    """The ring's plan attributes and their values for the period, every one of them, in a fixed order. A key the plan
    leaves out sets the attribute's unset form (`None` or `scripted`)."""
    menu = RINGS[ring]
    out: list[tuple[str, object]] = [
        ("plan_volume", plan.get("volume")),
        ("plan_quiet", bool(plan.get("quiet", False))),
        ("plan_timing", plan.get("timing", "scripted")),
    ]
    where = plan.get("where")
    out.append(("plan_where", "planned" if where else "scripted"))
    for i in ["arbelton", "brindlemark", "cedarmoor", "dovrex", "everhollow"]:
        out.append((f"plan_share_{i}", (where or {}).get(i, 0.0) if where else None))
    tactics = plan.get("tactics") or {}
    for t in sorted({t for r in RINGS.values() for t in r["tactics"]}):
        out.append((f"plan_tactic_{t}", tactics.get(t) if t in menu["tactics"] else None))
    out += [("plan_devices", plan.get("devices", "scripted")), ("plan_exits", plan.get("exits", "scripted")),
            ("plan_identities", plan.get("identities", "scripted")), ("plan_mule_target", plan.get("mule_target"))]
    return out


def apply_plan(run, ring: int, plan: dict) -> list[tuple[str, object]]:
    """Sets the plan on the ring in a `runmere.Run` at its boundary. Returns what it set."""
    rows = run.rows("ring", ["number"], where_attr="number", equals=ring)
    if len(rows) != 1:
        raise ValueError(f"no ring R{ring}")
    sets = interventions(plan, ring)
    for attr, value in sets:
        run.intervene("ring", rows[0]["id"], attr, value)
    return sets
