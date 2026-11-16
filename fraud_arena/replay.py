"""Match replays: what happened in a game, day by day, for the viewer (`viewer/`) and the website.

Grader side, and public after the match (the world is public): `<out>/replay/replay.jsonl` holds one JSON frame per
line, the first being the match's description (also in `replay/meta.json`). While the match runs, the game appends
frames from what is cheap to read each day (the incidents' state, the harness's own counters); when it ends, the file
is rebuilt from the truth exports scoring already wrote, so the finished replay is exact. Nothing here is written into
a player's workspace, and the folder is the harness user's alone while the match runs: its frames are truth.

Frames (`t`):
- `meta`: format, world, config, players, nodes, boundaries, days.
- `day`: one simulated day [day, day + 1) of one run (`game` or `reference`): `flows` [source, target, kind, count,
  amount] with kind g (genuine), f (fraud that went through), c (fraud an adverse decision stopped), x (genuine
  activity an adverse decision stopped); `rings` per ring (and `fraudsters`, the lone ones): attempts, successes,
  blocked, losses (observed, background, by node); `decisions` per institution: decision -> action -> [good, fraud];
  `harness` per defender: workflow cases and actions, review-queue work, Grayson asks (game run only).
- `boundary`: a period boundary, with each player's turn and what it changed.
- `score`: the score so far (`exact`: as scoring computes it, over the truth up to that day).
- `rollback`: live only; a bundle failed at run time and the game re-ran from `day`: frames after it are void.
- `end`: the match is over.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import INSTITUTIONS, ScoreWeights

FORMAT = "fraud-arena-replay/1"
DAY_S = 86400
MERCHANT_TYPES = ("business", "seller", "platform_store", "courier")
LABELS = {"brindlemark": "Brindlemark", "dovrex": "Dovrex", "hopcrate": "Hopcrate", "kellmont": "Kellmont"}


def nodes(rings: list[str], players: list[dict]) -> list[dict]:
    by_target = {(p["side"], str(p["target"])): p["name"] for p in players}
    out = [{"id": i, "kind": "institution", "label": LABELS.get(i, i.title()),
            "player": by_target.get(("defender", i))} for i in INSTITUTIONS]
    out += [{"id": "background", "kind": "background", "label": "Other banks"},
            {"id": "merchants", "kind": "merchants", "label": "Merchants"},
            {"id": "web", "kind": "web", "label": "Public web & outside"},
            {"id": "fraudsters", "kind": "fraudsters", "label": "Lone fraudsters"}]
    for r in sorted(rings, key=lambda r: int(r[1:])):
        out.append({"id": r, "kind": "ring", "label": r, "player": by_target.get(("attacker", r[1:]))})
    return out


def world_version(world: Path | str | None) -> dict:
    out: dict = {"path": str(world) if world else None}
    p = Path(str(world)) if world else None
    if p is None or not p.is_file():
        return out
    text = p.read_bytes()
    out["sha256"] = hashlib.sha256(text).hexdigest()[:16]
    m = re.search(rb"^id:\s*(\S+)", text, re.M)
    if m:
        out["id"] = m.group(1).decode()
    lock = p.with_name("world.lock")
    if lock.exists():
        out["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    try:
        out["commit"] = subprocess.run(["git", "-C", str(p.parent), "rev-parse", "--short=12", "HEAD"],
                                       capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    return out


def match_meta(*, config: dict, players: list[dict], boundaries: list[int], settle_days: int, epoch_iso: str,
               rings: list[str]) -> dict:
    epoch = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    return {"t": "meta", "format": FORMAT, "status": "running", "source": "live", "revision": 0,
            "world": world_version(config.get("world")), "epoch": epoch.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": {k: str(v) for k, v in config.items()},
            "players": players, "boundaries": boundaries, "start_day": boundaries[0],
            "end_day": boundaries[-1] + settle_days, "settle_days": settle_days,
            "nodes": nodes(rings, players), "synthetic": False}


def _diff(now, prev):
    if isinstance(now, dict):
        out = {}
        for k, v in now.items():
            d = _diff(v, (prev or {}).get(k) if isinstance(prev, dict) else None)
            if d:
                out[k] = d
        return out
    if isinstance(now, (int, float)):
        d = now - (prev or 0)
        return round(d, 2) if isinstance(d, float) else d
    return None


def _plan_changes(plan, prev) -> list[str]:
    if prev is None:
        return sorted(plan) if isinstance(plan, dict) else []
    if not isinstance(plan, dict):
        return []
    return sorted(k for k in set(plan) | set(prev) if plan.get(k) != prev.get(k))


def _lines(text: str | None, n: int = 3) -> list[str]:
    return [ln.strip()[:200] for ln in (text or "").splitlines() if ln.strip()][:n]


def turn_record(out: Path, t: dict, side: str, prev_plan) -> tuple[dict, object]:
    """A player's turn at a boundary as the replay shows it: the turn, and what it changed (from the bundles and
    plans the game keeps)."""
    rec = {k: t.get(k) for k in ("player", "ok", "seconds", "waits", "review")}
    rec["side"] = side
    rec["error"] = (t.get("error") or None) and str(t["error"])[:400]
    applied, b, name = t.get("applied"), t.get("boundary"), t.get("player")
    plan = prev_plan
    if side == "defender":
        ch: dict = {"digest": applied[:12] if applied else None, "changed": False}
        f = out / "bundles" / f"{name}-{b}-{applied[:12]}.json" if applied else None
        if f is not None and f.exists():
            bun = json.loads(f.read_text())
            pol = bun.get("policy") or {}
            models = pol.get("models") or {}
            ext = pol.get("external") or {}
            ch.update(changed=True, rules=len(pol.get("rules") or []), models=len(models),
                      model_kinds=sorted({str((m or {}).get("kind", "?")) for m in models.values()
                                          if isinstance(m, dict)}),
                      workflow=bool(bun.get("workflow")),
                      grayson=sorted(d for d, v in ext.items() if isinstance(v, dict) and "grayson" in v)
                      + (["review_queue"] if (pol.get("review_queue") or {}).get("grayson") else []),
                      notes=_lines(bun.get("notes")))
        rv = out / "reviews" / f"{name}-{b}.md"
        if rv.exists():
            ch["review"] = _lines(rv.read_text(), 2)
        rec["changes"] = ch
    else:
        f = out / "plans" / f"{name}-{b}.json"
        ch = {"changed": False}
        if applied and f.exists():
            plan = json.loads(f.read_text())
            keys = _plan_changes(plan, prev_plan)
            ch = {"changed": bool(keys), "keys": keys, "plan": plan}
        rec["changes"] = ch
    return rec, plan


class Recorder:
    """Appends live frames as the game runs. Every method is fail-soft: a replay problem never stops a match."""

    def __init__(self, out: Path, log=print):
        self.out = Path(out)
        self.dir = self.out / "replay"
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.path = self.dir / "replay.jsonl"
        self.log = log
        self.seconds = 0.0
        self.days = 0
        self.meta: dict = {}
        self.ring_node: dict[str, str] = {}
        self.inst_name: dict[str, str] = {}
        self.snap: dict[str, dict | None] = {}
        self.kept: dict[str, dict | None] = {}
        self.harness_days: dict[int, dict] = {}
        self.harness_prev: dict = {}
        self.plans: dict[str, object] = {}
        self.sides: dict[str, str] = {}
        self.broken = False
        self._fh = None

    def _guard(self, what, fn, *a):
        if self.broken:
            return None
        t0 = time.perf_counter()
        try:
            return fn(*a)
        except Exception as e:
            self.broken = True
            self.log({"event": "replay_failed", "where": what, "error": f"{type(e).__name__}: {e}"[:300]})
            return None
        finally:
            self.seconds += time.perf_counter() - t0

    def _write(self, frame: dict):
        self._fh.write(json.dumps(frame, separators=(",", ":"), default=str) + "\n")
        self._fh.flush()

    def start(self, *, config: dict, players: list[dict], boundaries: list[int], settle_days: int, epoch_iso: str,
              game, ref):
        self._guard("start", self._start, config, players, boundaries, settle_days, epoch_iso, game, ref)

    def _start(self, config, players, boundaries, settle_days, epoch_iso, game, ref):
        self.sides = {p["name"]: p["side"] for p in players}
        try:
            self.ring_node = {r["id"]: f"R{int(r['number'])}" for r in game.rows("ring", ["number"])}
            self.inst_name = {r["id"]: r["name"] for r in game.rows("institution", ["name"])}
        except Exception:  # an engine without entity reads: no ring data live
            pass
        self.meta = match_meta(config=config, players=players, boundaries=boundaries, settle_days=settle_days,
                               epoch_iso=epoch_iso, rings=sorted(set(self.ring_node.values())))
        self._fh = open(self.path, "w")
        self._write(self.meta)
        self._write_meta()
        self.snap = {"game": self._incidents(game), "reference": self._incidents(ref)}
        self.kept = dict(self.snap)

    def _write_meta(self):
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1, default=str))
        tmp.rename(self.dir / "meta.json")

    def _incidents(self, run):
        try:
            rows = run.rows("incident", ["ring", "institution", "result", "loss"])
        except Exception:
            return None
        return {r["id"]: (r.get("ring"), r.get("institution"), r.get("result"), float(r.get("loss") or 0))
                for r in rows}

    def _node(self, inst_id) -> str:
        name = self.inst_name.get(inst_id, inst_id)
        if name is None:
            return "web"
        return name if name in INSTITUTIONS else "background"

    def day(self, run, d: int, which: str, harness: dict | None = None):
        self._guard("day", self._day, run, d, which, harness)

    def _day(self, run, d, which, harness):
        now = self._incidents(run)
        prev = self.snap.get(which)
        rings, flows = ring_changes(now, prev, self.ring_node, self._node) if now is not None and prev is not None \
            else ({}, [])
        self.snap[which] = now
        frame = {"t": "day", "day": d, "run": which, "src": "live", "flows": flows, "rings": rings}
        if harness is not None:
            delta = _diff(harness, self.harness_prev)
            self.harness_prev = json.loads(json.dumps(harness))
            if delta:
                self.harness_days[d] = delta
                frame["harness"] = delta
        self.days += 1
        self._write(frame)

    def boundary(self, b: int, index: int, turns: list[dict]):
        self._guard("boundary", self._boundary, b, index, turns)

    def _boundary(self, b, index, turns):
        recs = []
        for t in turns:
            rec, self.plans[t["player"]] = turn_record(self.out, t, self.sides.get(t["player"], "defender"),
                                                       self.plans.get(t["player"]))
            recs.append(rec)
        self._write({"t": "boundary", "day": b, "index": index, "turns": recs})
        self.kept = dict(self.snap)

    def rollback(self, b: int, harness: dict | None):
        self._guard("rollback", self._rollback, b, harness)

    def _rollback(self, b, harness):
        self._write({"t": "rollback", "day": b})
        self.snap["game"] = self.kept.get("game")
        for d in [d for d in self.harness_days if d >= b]:
            del self.harness_days[d]
        if harness is not None:
            self.harness_prev = json.loads(json.dumps(harness))

    def finish(self, *, weights: ScoreWeights | None = None):
        if self._fh is not None:
            self._fh.close()
        if self.broken or not self.meta:
            return None
        t0 = time.perf_counter()
        try:
            stats = rebuild(self.out, harness=self.harness_days, weights=weights, log=self.log,
                            live_seconds=round(self.seconds, 3), live_days=self.days)
        except Exception as e:
            self.log({"event": "replay_failed", "where": "finish", "error": f"{type(e).__name__}: {e}"[:300]})
            return None
        stats["finish_seconds"] = round(time.perf_counter() - t0, 2)
        self.log({"event": "replay", **stats})
        return stats


def ring_changes(now: dict, prev: dict, ring_node: dict, node_of) -> tuple[dict, list]:
    """What changed between two snapshots of the incidents ({id: (ring, institution, result, loss)}): per ring, the
    new attempts, successes, blocks and losses, and the flows they make (ring -> institution)."""
    rings: dict[str, dict] = {}
    flows: dict[tuple, list] = {}
    for iid, (ring, inst, res, loss) in now.items():
        p = prev.get(iid)
        before_res, before_loss = (p[2], p[3]) if p else (None, 0.0)
        new = p is None
        dl = round(loss - before_loss, 2)
        if not new and res == before_res and not dl:
            continue
        rk = ring_node.get(ring, "fraudsters") if ring else "fraudsters"
        node = node_of(inst)
        st = rings.setdefault(rk, {"attempts": 0, "successes": 0, "blocked": 0, "loss_observed": 0.0,
                                   "loss_background": 0.0, "by_node": {}})
        by = st["by_node"].setdefault(node, [0, 0, 0, 0.0])
        if new:
            st["attempts"] += 1
            by[0] += 1
        if res != before_res and res == "succeeded":
            st["successes"] += 1
            by[1] += 1
            flows.setdefault((rk, node, "f"), [0, 0.0])[0] += 1
        if res != before_res and res == "blocked":
            st["blocked"] += 1
            by[2] += 1
            flows.setdefault((rk, node, "c"), [0, 0.0])[0] += 1
        if dl:
            st["loss_observed" if node in INSTITUTIONS else "loss_background"] += dl
            by[3] = round(by[3] + dl, 2)
            flows.setdefault((rk, node, "f"), [0, 0.0])[1] += dl
    for st in rings.values():
        st["loss_observed"] = round(st["loss_observed"], 2)
        st["loss_background"] = round(st["loss_background"], 2)
    return rings, [[s, t, k, n, round(a, 2)] for (s, t, k), (n, a) in sorted(flows.items())]


def _iso(epoch_iso: str, day: float) -> str:
    e = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00"))
    return (e + timedelta(days=day)).astimezone(timezone.utc).isoformat()


def truth_days(truth: Path, owners: Path | None, *, epoch_iso: str, start_day: int, end_day: int) -> tuple[dict, list]:
    """Day frames of one run from its truth export (`score/<run>/truth`, `owners.parquet`): exact flows between the
    nodes, caught and stopped activity, decisions, and the rings' incidents. Returns ({day: frame}, ring node ids)."""
    import duckdb
    import pyarrow as pa

    from .attribution import ring_instances
    from .scoring import ADVERSE

    c = duckdb.connect()
    T = str(truth)
    e0 = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00")).timestamp()
    lo, hi = _iso(epoch_iso, start_day), _iso(epoch_iso, end_day)
    dexpr = f"floor((epoch(time) - {e0}) / {DAY_S})::int"
    hidden = f"'{T}/hidden.parquet'"
    latest = lambda typ, attr: c.execute(
        f"select entity, arg_max(value, valid_from) from {hidden} where type = ? and attribute = ? group by 1",
        [typ, attr]).fetchall()
    ring_node = {e: f"R{int(v)}" for e, v in latest("ring", "number") if v is not None}
    fr_ring = {e: v for e, v in latest("fraudster", "ring")}
    node: dict[str, str] = {}
    inst_name: dict[str, str] = {}
    if (Path(T) / "balances.parquet").exists():
        for e, typ in c.execute(f"select entity, type from '{T}/balances.parquet'").fetchall():
            node[e] = "merchants" if typ in MERCHANT_TYPES else "web" if typ == "external" else "background"
    if owners is not None and Path(owners).exists():
        for e, inst in c.execute(f"select entity, institution from '{owners}'").fetchall():
            node[e] = inst if inst in INSTITUTIONS else "background"
            inst_name[e] = inst
    for f, r in fr_ring.items():
        node[f] = ring_node.get(r, "fraudsters")
    # Only the ring's own accounts are the ring's: a taken-over account (`controlled_by`) stays its victim's.
    for e, r in latest("account", "ring"):
        if r is not None:
            node[e] = ring_node.get(r, "fraudsters")
    c.register("nodes", pa.table({"entity": list(node), "node": list(node.values())}))
    roles = [(i["role"], n) for n, i in INSTITUTIONS.items()] + [(i["analyst"], n) for n, i in INSTITUTIONS.items()
                                                                 if i.get("analyst")]
    c.register("roles", pa.table({"role": [r for r, _ in roles], "inst": [n for _, n in roles]}))
    inst_ring: dict[str, str] = {}
    for e, rn in sorted(ring_node.items(), key=lambda x: int(x[1][1:])):
        for i in ring_instances(c, truth, ring=e, number=int(rn[1:])):
            inst_ring.setdefault(i, rn)
    c.register("inst_ring", pa.table({"instance": list(inst_ring), "ring": list(inst_ring.values())},
                                      schema=pa.schema([("instance", pa.string()), ("ring", pa.string())])))

    days: dict[int, dict] = {}

    def frame(d):
        return days.setdefault(d, {"flows": [], "rings": {}, "decisions": {}})

    for d, s, t, k, n, amt in c.execute(f"""
            select {dexpr.replace('time', 'p.time')} d, coalesce(ns.node, 'web'), coalesce(nt.node, 'web'),
                   case when p.cause_kind = 'scenario' then 'f' else 'g' end, count(*), sum(p.amount)::double
            from '{T}/postings.parquet' p left join nodes ns on ns.entity = p.payer
            left join nodes nt on nt.entity = p.payee
            where p.amount > 0 and p.time >= ?::timestamptz and p.time < ?::timestamptz
            group by all order by all""", [lo, hi]).fetchall():
        frame(d)["flows"].append([s, t, k, n, round(amt, 2)])
    adverse = "(" + ", ".join(f"'{a}'" for a in ADVERSE) + ")"
    has_events = (Path(T) / "events.parquet").exists()
    ev = (f"left join (select key, cause_kind, cause_instance from '{T}/events.parquet') e on e.key = x.key"
          if has_events else "left join (select null::varchar key, null::varchar cause_kind, "
                             "null::varchar cause_instance) e on false")
    rows = c.execute(f"""
        with x as (select * from '{T}/decisions.parquet' where time >= ?::timestamptz and time < ?::timestamptz),
             amt as (select key, max(amount)::double a from '{T}/postings.parquet'
                     where key in (select key from x where action in {adverse}) group by key)
        select {dexpr.replace('time', 'x.time')} d, r.inst, x.decision, x.action,
               coalesce(e.cause_kind = 'scenario', false) fraud, coalesce(ir.ring, 'fraudsters') ring,
               count(*), coalesce(sum(amt.a), 0)::double
        from x join roles r on r.role = x.role {ev}
        left join inst_ring ir on ir.instance = e.cause_instance
        left join amt on amt.key = x.key
        group by all order by all""", [lo, hi]).fetchall()
    caught: dict[tuple, list] = {}
    for d, inst, dec, act, fraud, ring, n, amt in rows:
        f = frame(d)
        cell = f["decisions"].setdefault(inst, {}).setdefault(dec, {}).setdefault(act, [0, 0])
        cell[1 if fraud else 0] += n
        if act in ADVERSE:
            key = (d, ring if fraud else inst, inst, "c" if fraud else "x")
            v = caught.setdefault(key, [0, 0.0])
            v[0] += n
            v[1] += amt
    for (d, s, t, k), (n, amt) in sorted(caught.items()):
        frame(d)["flows"].append([s, t, k, n, round(amt, 2)])

    hist = c.execute(f"""select entity, attribute, epoch(valid_from), value from {hidden}
                         where type = 'incident' and attribute in ('ring', 'institution', 'result', 'loss')
                         order by entity, valid_from, attribute""").fetchall()
    by_inc: dict[str, list] = {}
    for e, a, t, v in hist:
        by_inc.setdefault(e, []).append((a, t, v))
    lo_d, hi_d = start_day, end_day

    def node_of(inst):
        if inst is None:
            return "web"
        name = inst_name.get(inst)
        if name is None:
            return "background"
        return name if name in INSTITUTIONS else "background"

    for e, evs in by_inc.items():
        ring = next((v for a, _, v in evs if a == "ring" and v), None)
        rk = ring_node.get(ring, "fraudsters") if ring else "fraudsters"
        insts = [v for a, _, v in evs if a == "institution"]
        nd = node_of(insts[-1] if insts else None)
        created = min(t for _, t, _ in evs)
        res, loss = None, 0.0
        for a, t, v in evs + [("_created", created, None)]:
            d = int((t - e0) // DAY_S)
            inside = lo_d <= d < hi_d
            if a == "_created":
                if inside:
                    st = _ring_cell(frame(d)["rings"], rk)
                    st["attempts"] += 1
                    st["by_node"].setdefault(nd, [0, 0, 0, 0.0])[0] += 1
                continue
            if a == "result":
                if v != res and v in ("succeeded", "blocked") and inside:
                    st = _ring_cell(frame(d)["rings"], rk)
                    st["successes" if v == "succeeded" else "blocked"] += 1
                    st["by_node"].setdefault(nd, [0, 0, 0, 0.0])[1 if v == "succeeded" else 2] += 1
                res = v
            elif a == "loss":
                x = float(v or 0)
                dl = round(x - loss, 2)
                loss = x
                if dl and inside:
                    st = _ring_cell(frame(d)["rings"], rk)
                    st["loss_observed" if nd in INSTITUTIONS else "loss_background"] = round(
                        st["loss_observed" if nd in INSTITUTIONS else "loss_background"] + dl, 2)
                    by = st["by_node"].setdefault(nd, [0, 0, 0, 0.0])
                    by[3] = round(by[3] + dl, 2)
    return {d: f for d, f in days.items() if lo_d <= d < hi_d}, sorted(set(ring_node.values()))


def _ring_cell(rings: dict, rk: str) -> dict:
    return rings.setdefault(rk, {"attempts": 0, "successes": 0, "blocked": 0, "loss_observed": 0.0,
                                 "loss_background": 0.0, "by_node": {}})


def boundary_scores(game_x, ref_x, *, players: list[dict], weights: ScoreWeights, epoch_iso: str, start_day: int,
                    days: list[int], log=print) -> dict[int, dict]:
    """The score each player would have if the match ended at each of `days`: scoring's own functions over the truth
    up to that day (no settling: late chargebacks and returns of the last weeks aren't in yet)."""
    import duckdb

    from .scoring import paired, ring_score, window_score

    start = _iso(epoch_iso, start_day)
    rid = {}
    c = duckdb.connect()
    for p in players:
        if p["side"] == "attacker":
            row = c.execute(f"""select entity from '{game_x[0]}/hidden.parquet' where type = 'ring'
                                and attribute = 'number' and try_cast(value as bigint) = ? limit 1""",
                            [int(p["target"])]).fetchone()
            if row:
                rid[p["name"]] = row[0]

    def one(job):
        day, p = job
        end = _iso(epoch_iso, day)
        if p["side"] == "defender":
            role = INSTITUTIONS[p["target"]]["role"]
            g = window_score(*game_x, institution=p["target"], role=role, start=start, end=end, weights=weights)
            r = window_score(*ref_x, institution=p["target"], role=role, start=start, end=end, weights=weights)
            s = paired(g, r, weights)
            return day, p, {"composite": round(s.composite, 2), "losses_prevented": round(s.losses_prevented, 2),
                            "added_friction": round(s.added_friction, 2),
                            "mule_flow_prevented": round(s.mule_flow_prevented, 2),
                            "displacement": round(s.displacement, 2)}
        if p["name"] not in rid:
            return day, p, None
        kw = dict(ring=rid[p["name"]], institutions=list(INSTITUTIONS), start=start, end=end, weights=weights,
                  number=int(p["target"]))
        gs, rs = ring_score(*game_x, **kw), ring_score(*ref_x, **kw)
        return day, p, {"score": round(gs["score"], 2), "score_of_scripted_ring": round(rs["score"], 2),
                        "vs_scripted": round(gs["score"] - rs["score"], 2)}

    jobs = [(d, p) for d in days for p in players]
    out: dict[int, dict] = {d: {"defenders": {}, "attackers": {}} for d in days}
    with ThreadPoolExecutor(4) as pool:
        for day, p, s in pool.map(one, jobs):
            if s is not None:
                out[day]["defenders" if p["side"] == "defender" else "attackers"][p["name"]] = s
    return out


def final_scores(result: dict) -> dict:
    d = {n: {k: round(float(v.get(k) or 0), 2) for k in ("composite", "losses_prevented", "added_friction",
                                                        "mule_flow_prevented", "displacement")}
         for n, v in (result.get("defenders") or {}).items()}
    a = {n: {k: round(float(v.get(k) or 0), 2) for k in ("score", "score_of_scripted_ring", "vs_scripted")}
         for n, v in (result.get("attackers") or {}).items()}
    return {"defenders": d, "attackers": a}


def _weights(config: dict) -> ScoreWeights:
    w = config.get("weights")
    if isinstance(w, ScoreWeights):
        return w
    try:
        return ScoreWeights(**ast.literal_eval(w)) if isinstance(w, str) else ScoreWeights(**(w or {}))
    except Exception:
        return ScoreWeights()


def _epoch(config: dict) -> str:
    world = Path(str(config.get("world", "")))
    if world.is_file():
        m = re.search(r"^clock:\s*\n(?:\s+.*\n)*?\s+epoch:\s*\"?([^\"\n]+)\"?", world.read_text(), re.M)
        if m:
            return m.group(1).strip()
    return "2025-01-01T00:00:00Z"


def set_meta(out: Path, **fields) -> None:
    """Sets fields of a finished replay's description (its first line and `meta.json`)."""
    rdir = Path(out) / "replay"
    lines = (rdir / "replay.jsonl").read_text().splitlines()
    meta = json.loads(lines[0]) | fields
    lines[0] = json.dumps(meta, separators=(",", ":"), default=str)
    tmp = rdir / "replay.jsonl.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(rdir / "replay.jsonl")
    (rdir / "meta.json.tmp").write_text(json.dumps(meta, indent=1, default=str))
    (rdir / "meta.json.tmp").rename(rdir / "meta.json")


def read_frames(path: Path) -> list[dict]:
    """The frames of a replay file, with rolled-back days dropped."""
    frames: list[dict] = []
    if not path.exists():
        return frames
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line the game was still writing
        if f.get("t") == "rollback":
            b = f["day"]
            frames = [x for x in frames if not (x.get("t") == "day" and x.get("run") == "game" and x["day"] >= b)]
            continue
        frames.append(f)
    return frames


def _grayson_live(out: Path) -> dict[int, dict]:
    days: dict[int, dict] = {}
    for f in sorted((out / "grayson-live").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Logged at the run's time: an answer at time 181.0 or 180.5 is day 180's.
            h = days.setdefault(math.ceil(float(r["day"])) - 1, {}).setdefault(f.stem, {})
            h["grayson"] = h.get("grayson", 0) + 1
            if r.get("where") == "review_queue" and r.get("verdict"):
                q = h.setdefault("queue", {})
                q["decided"] = q.get("decided", 0) + 1
    return days


def rebuild(out: Path, *, game_truth: Path | None = None, ref_truth: Path | None = None,
            game_owners: Path | None = None, ref_owners: Path | None = None, harness: dict | None = None,
            weights: ScoreWeights | None = None, scores: bool = True, log=print, live_seconds=None,
            live_days=None) -> dict:
    """(Re)builds `<out>/replay/` from what a match left: `result.json` (or `progress.json` while it runs), the
    truth exports of scoring (`score/game`, `score/reference`), the bundles, plans and reviews it kept, and the live
    frames of an earlier replay for what only the live game saw (the harness's per-day counters)."""
    out = Path(out)
    rdir = out / "replay"
    rdir.mkdir(parents=True, exist_ok=True)
    old = read_frames(rdir / "replay.jsonl")
    old_meta = next((f for f in old if f.get("t") == "meta"), None)
    if old_meta is None and (rdir / "meta.json").exists():
        old_meta = json.loads((rdir / "meta.json").read_text())
    result = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else None
    if result is None and old_meta is None:
        raise FileNotFoundError(f"{out}: no result.json and no replay to rebuild from")
    config = (result or {}).get("config") or (old_meta or {}).get("config") or {}
    players = (result or {}).get("players") or (old_meta or {}).get("players") or []
    if result is not None:
        turns = result.get("turns") or []
    elif (out / "progress.json").exists():
        turns = json.loads((out / "progress.json").read_text())
    else:
        turns = []
    epoch = (old_meta or {}).get("epoch") or _epoch(config)
    start, period, horizon = (int(float(config.get(k, d))) for k, d in
                              (("start_day", 180), ("period_days", 30), ("horizon_days", 120)))
    settle = int(float(config.get("settle_days", 30)))
    bounds = list(range(start, start + horizon, period)) + [start + horizon]
    end_day = bounds[-1] + settle
    weights = weights or _weights(config)

    def export(name, truth, owners):
        t = truth or out / "score" / name / "truth"
        o = owners or out / "score" / name / "owners.parquet"
        return (t, o if Path(o).exists() else None) if (Path(t) / "postings.parquet").exists() else None

    gx, rx = export("game", game_truth, game_owners), export("reference", ref_truth, ref_owners)
    t0 = time.perf_counter()
    days: dict[str, dict[int, dict]] = {"game": {}, "reference": {}}
    rings: set[str] = set()
    notes = []
    for name, x in (("game", gx), ("reference", rx)):
        d = None
        if x is not None:
            try:
                d, rn = truth_days(x[0], x[1], epoch_iso=epoch, start_day=start, end_day=end_day)
            except Exception as e:
                notes.append(f"{name}: truth export unreadable ({type(e).__name__}: {str(e)[:200]})")
                if name == "game":
                    gx = None
                else:
                    rx = None
        if d is None:
            for f in old:
                if f.get("t") == "day" and f.get("run") == name:
                    days[name][f["day"]] = {k: v for k, v in f.items() if k in ("flows", "rings", "decisions")} \
                                           | {"src": f.get("src", "live")}
            notes.append(f"{name}: no truth export; live frames kept" if days[name] else f"{name}: no data")
            continue
        if x[1] is None:
            notes.append(f"{name}: no owners.parquet; accounts aren't mapped to institutions")
        days[name] = {k: v | {"src": "truth"} for k, v in d.items()}
        rings |= set(rn)
    truth_s = time.perf_counter() - t0
    for f in old:
        for r in (f.get("rings") or {}) if f.get("t") == "day" else ():
            if r.startswith("R"):
                rings.add(r)
    rings |= {f"R{p['target']}" for p in players if p.get("side") == "attacker"}

    if harness is None:
        harness = {f["day"]: f["harness"] for f in old if f.get("t") == "day" and f.get("harness")}
        if not harness:
            harness = _grayson_live(out)

    side = {p["name"]: p["side"] for p in players}
    by_b: dict[int, list] = {}
    for t in turns:
        by_b.setdefault(int(t["boundary"]), []).append(t)
    plans: dict[str, object] = {}
    bframes = {}
    for i, b in enumerate(sorted(by_b)):
        recs = []
        for t in by_b[b]:
            rec, plans[t["player"]] = turn_record(out, t, side.get(t["player"], "defender"), plans.get(t["player"]))
            recs.append(rec)
        bframes[b] = {"t": "boundary", "day": b, "index": bounds.index(b) if b in bounds else i, "turns": recs}

    t1 = time.perf_counter()
    sframes: dict[int, dict] = {}
    if scores and gx and rx and gx[1] and rx[1]:
        mids = [b for b in bounds[1:] if b < end_day and b <= max([start, *by_b]) + period]
        for day, s in boundary_scores(gx, rx, players=players, weights=weights, epoch_iso=epoch, start_day=start,
                                      days=mids, log=log).items():
            sframes[day] = {"t": "score", "day": day, "exact": True, **s}
    if result is not None and (result.get("defenders") or result.get("attackers")):
        sframes[end_day] = {"t": "score", "day": end_day, "exact": True, "final": True, **final_scores(result)}
    score_s = time.perf_counter() - t1

    meta = match_meta(config=config, players=players, boundaries=bounds, settle_days=settle, epoch_iso=epoch,
                      rings=sorted(rings))
    meta.update(status="finished" if result is not None else "running",
                source="truth" if gx else "live", revision=int((old_meta or {}).get("revision", 0)) + 1,
                notes=notes, synthetic=bool((old_meta or {}).get("synthetic")))
    if (old_meta or {}).get("synthetic_note"):
        meta["synthetic_note"] = old_meta["synthetic_note"]
    if live_seconds is not None:
        meta["overhead"] = {"live_seconds": live_seconds, "live_days": live_days}
    lines = [meta]
    last = max([end_day - 1, *days["game"], *days["reference"]] or [end_day - 1])
    for d in range(start, max(last + 1, end_day + 1)):
        if d in bframes:
            lines.append(bframes[d])
        if d in sframes:
            lines.append(sframes[d])
        if d >= end_day:
            continue
        for name in ("game", "reference"):
            f = days[name].get(d)
            if f is None and not (name == "game" and d in harness):
                continue
            fr = {"t": "day", "day": d, "run": name, "src": (f or {}).get("src", "truth"),
                  "flows": (f or {}).get("flows", []), "rings": (f or {}).get("rings", {}),
                  "decisions": (f or {}).get("decisions", {})}
            if name == "game" and harness.get(d):
                fr["harness"] = harness[d]
            lines.append(fr)
    if result is not None:
        lines.append({"t": "end", "day": end_day, "finished": result.get("finished")})
    tmp = rdir / "replay.jsonl.tmp"
    with open(tmp, "w") as fh:
        for f in lines:
            fh.write(json.dumps(f, separators=(",", ":"), default=str) + "\n")
    tmp.rename(rdir / "replay.jsonl")
    mtmp = rdir / "meta.json.tmp"
    mtmp.write_text(json.dumps(meta, indent=1, default=str))
    mtmp.rename(rdir / "meta.json")
    size = (rdir / "replay.jsonl").stat().st_size
    return {"frames": len(lines), "bytes": size, "truth_seconds": round(truth_s, 2),
            "score_seconds": round(score_s, 2), "source": meta["source"], "notes": notes}
