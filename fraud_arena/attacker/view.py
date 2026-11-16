"""The ring operator's view at a period boundary: what one ring knows of its own operation, and nothing else.

Why a harness function and not a world observer: everything a ring knows is truth in Fraud World (its members, its
accounts, `ring_feedback`, its incidents are all `hidden` or truth-only), and an observer can't read hidden data by
design (SPEC §13 rules 10 and 11). Making them `private` to a ring role would change the visibility of dozens of
attributes that the defenders' checks rely on. So the grader side reads truth through `runmere.Run.rows` and this
module filters it to the ring's own knowledge, by an allow-list:

- the ring's own memory, as the world keeps it: decayed counts of successes and failures that reached it from each
  institution (`ring_feedback`, days after each attempt) and when the last one came;
- its own attempts (its incidents): how many, by scheme and institution, and how each went at the time of the attempt
  (`succeeded`, `blocked`, still `attempted`), and the money it took out;
- its own accounts at each institution: open, restricted or closed (it notices when it loses access);
- how many mules it has recruited, its mule target, and the plan it is running.

It never reads labels, alerts, decisions, reviews, cases, institutions' rules or models, other rings, or anything
about an institution beyond the ring's own accounts and attempts there. `ring_view` builds its output from fixed keys
only, and `check_view` refuses any other key, so a change here can't add a field by accident.
"""
from __future__ import annotations

from datetime import datetime

INSTITUTIONS = ["arbelton", "brindlemark", "cedarmoor", "dovrex", "everhollow"]
PLAN_ATTRS = ["plan_volume", "plan_quiet", "plan_timing", "plan_where", "plan_devices", "plan_exits",
              "plan_identities", "plan_mule_target"]
DAY_MS = 86_400_000

VIEW_KEYS = {"ring", "day", "period", "memory", "attempts", "attempts_total", "accounts", "mules", "plan_in_force"}


def _institution_names(run) -> dict[str, str]:
    return {r["id"]: r["name"] for r in run.rows("institution", ["name"])}


def snapshot(run, ring: int) -> dict:
    """The ring's own state at the boundary, from truth. Kept by the harness, never shown as it is."""
    memory_attrs = [f"{k}_{i}" for i in INSTITUTIONS for k in ("ok", "fail", "seen")]
    rows = run.rows("ring", ["number", "mules", "mule_target", "launched_at", *memory_attrs, *PLAN_ATTRS],
                    where_attr="number", equals=ring)
    if len(rows) != 1:
        raise ValueError(f"no ring R{ring}")
    me = rows[0]
    names = _institution_names(run)
    incidents = run.rows("incident", ["family", "institution", "result", "loss"], where_attr="ring", equals=me["id"])
    accounts = run.rows("account", ["institution", "status", "restricted"], where_attr="ring", equals=me["id"])
    return {
        "ring": me,
        "incidents": {i["id"]: {"family": i["family"], "institution": names.get(i["institution"]),
                                "result": i["result"], "loss": i["loss"]} for i in incidents},
        "accounts": [{"institution": names.get(a["institution"]), "status": a["status"],
                      "restricted": bool(a["restricted"])} for a in accounts],
    }


def _ms(t) -> int | None:
    """A time as `Run.rows` writes it (RFC 3339), in milliseconds since the Unix epoch."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return int(t)
    return int(datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp() * 1000)


def _day(epoch_ms: int, t) -> int | None:
    ms = _ms(t)
    return None if ms is None else int((ms - epoch_ms) // DAY_MS)


def ring_view(now: dict, before: dict | None, *, ring: int, day: int, period_start: int, epoch_ms: int) -> dict:
    """What ring R`ring` knows at the boundary on `day`: its state now, and what changed since `before` (the
    snapshot at the last boundary, or None at the first)."""
    me = now["ring"]
    memory = {}
    for i in INSTITUTIONS:
        seen = _ms(me.get(f"seen_{i}"))
        launched = _ms(me.get("launched_at"))
        heard = seen is not None and launched is not None and seen > launched
        memory[i] = {
            "recent_successes": round(float(me.get(f"ok_{i}") or 0.0), 3),
            "recent_failures": round(float(me.get(f"fail_{i}") or 0.0), 3),
            "last_heard_day": _day(epoch_ms, seen) if heard else None,
        }
    old = (before or {}).get("incidents", {})
    new_ids = sorted(k for k in now["incidents"] if k not in old)

    def tally(ids):
        out: dict[str, dict] = {}
        for k in ids:
            inc = now["incidents"][k]
            key = f"{inc['family']}@{inc['institution'] or 'none'}"
            t = out.setdefault(key, {"scheme": inc["family"], "institution": inc["institution"], "attempts": 0,
                                     "succeeded": 0, "blocked": 0, "pending": 0, "taken": 0.0})
            t["attempts"] += 1
            t[{"succeeded": "succeeded", "blocked": "blocked"}.get(inc["result"], "pending")] += 1
            t["taken"] = round(t["taken"] + float(inc["loss"] or 0.0), 2)
        return sorted(out.values(), key=lambda t: (t["scheme"], t["institution"] or ""))

    accounts: dict[str, dict] = {}
    for a in now["accounts"]:
        inst = a["institution"] or "none"
        c = accounts.setdefault(inst, {"open": 0, "restricted": 0, "closed": 0})
        if a["status"] == "closed":
            c["closed"] += 1
        elif a["restricted"]:
            c["restricted"] += 1
        else:
            c["open"] += 1
    view = {
        "ring": f"R{ring}",
        "day": day,
        "period": {"start_day": period_start, "end_day": day},
        "memory": memory,
        "attempts": tally(new_ids),
        "attempts_total": tally(sorted(now["incidents"])),
        "accounts": dict(sorted(accounts.items())),
        "mules": {"recruited": int(me.get("mules") or 0),
                  "target": me.get("plan_mule_target") if me.get("plan_mule_target") is not None
                  else int(me.get("mule_target") or 0)},
        "plan_in_force": {k.removeprefix("plan_"): me.get(k) for k in PLAN_ATTRS},
    }
    check_view(view)
    return view


def check_view(view: dict) -> None:
    """Refuses a view with a key outside the allow-list."""
    extra = set(view) - VIEW_KEYS
    if extra:
        raise ValueError(f"the ring view has keys it may not have: {sorted(extra)}")
    for row in view["attempts"] + view["attempts_total"]:
        if set(row) != {"scheme", "institution", "attempts", "succeeded", "blocked", "pending", "taken"}:
            raise ValueError(f"an attempts row has other keys: {sorted(row)}")
    for m in view["memory"].values():
        if set(m) != {"recent_successes", "recent_failures", "last_heard_day"}:
            raise ValueError("a memory row has other keys")
