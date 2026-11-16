"""A toy economy behind the engine interface `Game` uses, for the replay tests and the viewer's sample replay.

It is NOT Fraud World: a few banks, merchants and rings, card purchases, payroll and withdrawals, ring attacks whose
amounts a defender's `obs.amount > N` rules decline, chargebacks and laundering. Its truth export has the columns of
the real one, so the game, scoring and the replay run on it unchanged. A replay made from it says `synthetic: true`.

    python -m fraud_arena.demo <out>      # plays a scripted match into <out>
"""
from __future__ import annotations

import json
import random
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import INSTITUTIONS
from .runners import McpServer, SessionResult

EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)
BANKS = ["brindlemark", "dovrex", "arbelton", "cedarmoor"]
SHOPS = ["hopcrate", "kellmont"]
DECISION = {"brindlemark": "brind_card_auth", "dovrex": "dovrex_withdrawal_review", "hopcrate": "hop_checkout",
            "kellmont": "kell_checkout"}
BUILTIN = {"brindlemark": 2500.0, "dovrex": 3000.0, "hopcrate": 1800.0, "kellmont": 1800.0}
RINGS = {1: {"home": "arbelton", "plan": {"volume": 1.0, "targets": {"arbelton": 2, "cedarmoor": 1, "brindlemark": 1},
                                          "max_amount": 1500}},
         2: {"home": "dovrex", "plan": {"volume": 1.0, "targets": {"brindlemark": 3, "dovrex": 1}, "max_amount": 4000}},
         5: {"home": "cedarmoor", "plan": {"volume": 0.6, "targets": {"cedarmoor": 1, "dovrex": 1},
                                           "max_amount": 2000}},
         7: {"home": "dovrex", "plan": {"volume": 1.2, "targets": {"hopcrate": 2, "kellmont": 2},
                                        "max_amount": 2600}}}
ACCOUNTS_PER_BANK = 30
MERCHANTS = 6


def _t(day: float) -> datetime:
    return EPOCH + timedelta(days=day)


class SimRun:
    def __init__(self, seed: int):
        self.seed = seed
        self.day = 0.0
        self.deployed: dict[str, dict] = {}
        self.plans: dict[int, dict] = {}
        self.postings: list[tuple] = []
        self.decisions: list[tuple] = []
        self.events: list[tuple] = []
        self.instances: list[tuple] = []
        self.incidents: dict[str, dict] = {}
        self.hidden: list[tuple] = []
        self.pending: list[tuple] = []  # (day, posting) settling later: laundering, chargebacks
        self.n = 0
        for n in RINGS:
            self.hidden.append((f"ring-{n}", "ring", "number", _t(0), str(n)))
            self.hidden.append((f"fr-{n}", "fraudster", "ring", _t(0), f"ring-{n}"))
            for k in range(3):
                self.hidden.append((f"acc-r{n}-{k}", "account", "ring", _t(0), f"ring-{n}"))
        self.hidden.append(("fr-lone", "fraudster", "ring", _t(0), None))

    def fork(self):
        r = SimRun.__new__(SimRun)
        r.__dict__ = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                      for k, v in self.__dict__.items()}
        r.incidents = {k: dict(v) for k, v in self.incidents.items()}
        return r

    def deploy(self, role, dep):
        self.deployed[role] = dep

    def undeploy(self, role):
        self.deployed.pop(role, None)

    def threshold(self, inst: str) -> float:
        dep = self.deployed.get(INSTITUTIONS[inst]["role"])
        if dep is None:
            return BUILTIN[inst]
        th = [float(m.group(1)) for r in dep.get("rules", []) if r.get("action") == "decline"
              for m in [re.search(r"obs\.amount\s*>\s*([0-9.]+)", r.get("when", "") or "")] if m]
        return min(th) if th else BUILTIN[inst]

    def _id(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def _post(self, day, payer, payee, amount, cause=None, key=None):
        scen, inst = cause or (None, None)
        self.postings.append((key or self._id("k"), _t(day), payer, payee, round(amount, 2),
                              round(amount, 2) if cause else 0.0, "scenario" if cause else "process", scen, inst))

    def _decide(self, day, inst, amount, actor, cause=None) -> tuple[bool, str]:
        """The institution's gate on an event (none when `inst` has no gate): whether it lets it through, and the
        event's key."""
        key = self._id("e")
        scen, iid = cause or (None, None)
        self.events.append((key, _t(day), "payment", actor, "scenario" if cause else "process", scen, iid))
        if inst not in DECISION:
            return True, key
        ok = amount <= self.threshold(inst)
        self.decisions.append((key, _t(day), DECISION[inst], INSTITUTIONS[inst]["role"],
                               "approve" if ok else "decline"))
        return ok, key

    def run_until_day(self, d: float) -> int:
        while self.day + 1e-9 < d:
            day = int(self.day)
            self._simulate(day)
            self.day = float(day + 1)
        return 0

    def _simulate(self, day: int):
        g = random.Random(f"{self.seed}-{day}-genuine")
        accts = [(b, f"acc-{b}-{i}") for b in BANKS for i in range(ACCOUNTS_PER_BANK)]
        shops = [f"biz-m{i}" for i in range(MERCHANTS)] + [f"biz-{s}" for s in SHOPS]
        for bank, a in accts:
            for _ in range(2):
                if g.random() < 0.55:
                    shop = g.choice(shops)
                    amt = min(g.lognormvariate(4.0, 0.9), 2400)
                    t = day + g.random()
                    ok, key = self._decide(t, bank if bank == "brindlemark" else None, amt, a)
                    shop_inst = shop[4:] if shop[4:] in SHOPS else None
                    if ok and shop_inst:
                        ok, key = self._decide(t, shop_inst, amt, a)
                    if ok:
                        self._post(t, a, shop, amt, key=key)
            if day % 14 == 0:
                self._post(day + 0.3, "ext-payroll", a, g.uniform(1200, 3200))
            if g.random() < 0.08:
                amt = g.uniform(100, 3500)
                ok, key = self._decide(day + g.random(), bank, amt, a) if bank == "dovrex" else (True, None)
                if ok:
                    self._post(day + g.random(), a, "ext-web", amt, key=key)
        for _ in range(12):
            (_, a), (_, b) = g.sample(accts, 2)
            self._post(day + g.random(), a, b, g.uniform(10, 300))
        for n, spec in RINGS.items():
            self._ring_day(day, n, spec)
        lone = random.Random(f"{self.seed}-{day}-lone")
        if lone.random() < 0.6:
            self._attack(day + lone.random(), lone, None, lone.choice(BANKS + SHOPS), lone.uniform(80, 1200))
        due = [p for p in self.pending if p[0] < day + 1]
        self.pending = [p for p in self.pending if p[0] >= day + 1]
        for t, payer, payee, amt, cause in due:
            self._post(t, payer, payee, amt, cause)
        for (inst_id, rec) in self.incidents.items():
            if rec["result"] == "attempted" and rec["settle"] < day + 1:
                rec["result"] = "succeeded"
                self.hidden.append((inst_id, "incident", "result", _t(rec["settle"]), "succeeded"))

    def _ring_day(self, day, n, spec):
        r = random.Random(f"{self.seed}-{day}-ring{n}")
        plan = self.plans.get(n) or spec["plan"]
        vol = float(plan.get("volume", 1.0))
        targets = plan.get("targets") or spec["plan"]["targets"]
        cap = float(plan.get("max_amount", spec["plan"]["max_amount"]))
        k = sum(1 for _ in range(8) if r.random() < 0.3 * vol)
        names, weights = list(targets), [float(w) for w in targets.values()]
        for _ in range(k):
            self._attack(day + r.random(), r, n, r.choices(names, weights)[0], r.uniform(150, cap))

    def _attack(self, t, r, ring, target, amount):
        u = r.random()
        iid = self._id("inc")
        scen = f"r{ring}_attack" if ring else "lone_fraud"
        anchor = f"ring-{ring}" if ring else "fr-lone"
        self.instances.append((scen, iid, anchor, _t(t)))
        for a, v in (("ring", f"ring-{ring}" if ring else None), ("institution", f"inst-{target}"),
                     ("result", "attempted"), ("loss", "0.00")):
            self.hidden.append((iid, "incident", a, _t(t), v))
        victim = f"acc-{r.choice(BANKS) if target in SHOPS else target}-{r.randrange(ACCOUNTS_PER_BANK)}"
        mule = f"acc-r{ring}-{r.randrange(3)}" if ring else "acc-cedarmoor-0"
        rec = {"ring": ring, "institution": target, "result": "attempted", "loss": 0.0, "settle": t + 0.5}
        self.incidents[iid] = rec
        ok, key = self._decide(t, target, amount, anchor, (scen, iid))
        if not ok or (target not in DECISION and u < 0.1):
            rec["result"] = "blocked"
            self.hidden.append((iid, "incident", "result", _t(t + 0.01), "blocked"))
            return
        cause = (scen, iid)
        if target in SHOPS:
            self._post(t, victim, f"biz-{target}", amount, cause, key)
            self.pending.append((t + 5, f"biz-{target}", victim, amount, cause))
        else:
            self._post(t, victim, mule, amount, cause, key)
            self.pending.append((t + 1.5, mule, "ext-web", round(amount * 0.9, 2), cause))
        rec["loss"] = round(amount, 2)
        self.hidden.append((iid, "incident", "loss", _t(t + 0.5), f"{amount:.2f}"))

    def rows(self, etype, attrs, where_attr=None, equals=None):
        if etype == "ring":
            out = [{"id": f"ring-{n}", "number": n} for n in RINGS]
        elif etype == "institution":
            out = [{"id": f"inst-{i}", "name": i} for i in BANKS + SHOPS]
        elif etype == "incident":
            out = [{"id": k, "ring": f"ring-{v['ring']}" if v["ring"] else None, "institution": f"inst-{v['institution']}",
                    "result": v["result"], "loss": v["loss"]} for k, v in sorted(self.incidents.items())]
        else:
            raise ValueError(f"no type `{etype}`")
        return [{"id": o["id"], **{a: o.get(a) for a in attrs}} for o in out
                if where_attr is None or o.get(where_attr) == equals]


class SimEngine:
    def __init__(self, seed: int = 7):
        self.seed = seed

    def start(self, cfg): return SimRun(self.seed)
    def decisions_of(self, cfg): return {DECISION[cfg.institution]}
    def role_of(self, cfg): return {DECISION[cfg.institution]: cfg.role}
    def write_tables(self, run, cfg, tag=None): return None
    def fraud_signals(self, data, prev, cfg, b): return {}
    def fraud_families(self, cfg): return {"ato", "card_fraud"}
    def own_outcomes(self, run, cfg, a, b): return {}
    def day_iso(self, cfg, day): return _t(day).isoformat()

    @contextmanager
    def tool_server(self, run, cfg, workspace=None, tag=None):
        yield McpServer("arena", url="http://127.0.0.1:9/mcp")

    def export_scoring(self, run, cfg, out: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        t = out / "truth"
        t.mkdir(parents=True, exist_ok=True)
        ts = pa.timestamp("ms", tz="UTC")

        def write(name, cols, rows, types=None):
            data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
            schema = pa.schema([(c, (types or {}).get(c, pa.string())) for c in cols])
            pq.write_table(pa.table(data, schema=schema), t / f"{name}.parquet")

        write("postings", ["key", "time", "payer", "payee", "amount", "dirty", "cause_kind", "cause_scenario",
                           "cause_instance"], run.postings,
              {"time": ts, "amount": pa.float64(), "dirty": pa.float64()})
        write("decisions", ["key", "time", "decision", "role", "action"], run.decisions, {"time": ts})
        write("events", ["key", "time", "event", "actor", "cause_kind", "cause_scenario", "cause_instance"],
              run.events, {"time": ts})
        write("instances", ["scenario", "instance", "anchor", "time"], run.instances, {"time": ts})
        write("hidden", ["entity", "type", "attribute", "valid_from", "value"], run.hidden, {"valid_from": ts})
        write("labels", ["label", "subject"], [("ring_account", f"acc-r{n}-{k}") for n in RINGS for k in range(3)])
        ents = ([(f"acc-{b}-{i}", "account") for b in BANKS for i in range(ACCOUNTS_PER_BANK)]
                + [(f"acc-r{n}-{k}", "account") for n in RINGS for k in range(3)]
                + [(f"biz-m{i}", "business") for i in range(MERCHANTS)] + [(f"biz-{s}", "business") for s in SHOPS]
                + [("ext-payroll", "external"), ("ext-web", "external")]
                + [(f"inst-{i}", "institution") for i in BANKS + SHOPS])
        write("balances", ["entity", "type", "closing"], [(e, ty, 0.0) for e, ty in ents], {"closing": pa.float64()})
        owners = ([(f"acc-{b}-{i}", b) for b in BANKS for i in range(ACCOUNTS_PER_BANK)]
                  + [(f"acc-r{n}-{k}", RINGS[n]["home"]) for n in RINGS for k in range(3)]
                  + [(f"biz-{s}", s) for s in SHOPS] + [(f"inst-{i}", i) for i in BANKS + SHOPS])
        pq.write_table(pa.table({"entity": [e for e, _ in owners], "institution": [i for _, i in owners]}),
                       out / "owners.parquet")
        return t, out / "owners.parquet"


class SimAttackers:
    def active_rings(self, run, limit): return [2, 7][:limit]
    def entity(self, run, ring): return f"ring-{ring}"
    def ring_view(self, run, ring, a, b): return {"ring": ring, "day": b}
    def menu(self, ring): return {"type": "object"}
    def brief(self, **k): return "plan"

    def apply_plan(self, run, ring, plan):
        run.plans[ring] = plan

    def ring_loss(self, run, ring):
        out = {"observed": 0.0, "background": 0.0}
        for v in run.incidents.values():
            if v["ring"] == ring:
                out["observed" if v["institution"] in INSTITUTIONS else "background"] += v["loss"]
        return out


class Scripted:
    """A player that writes its turn without a model: `write(workspace, boundary_index)`."""

    def __init__(self, name, model, write):
        self.name, self.model, self.write, self.i = name, model, write, 0

    def run(self, ws, prompt, servers, log):
        self.write(Path(ws), self.i)
        self.i += 1
        return SessionResult(self.name, self.model, True, "ok", 1.0 + self.i, {"output_tokens": 1000})


def _defender(inst: str, thresholds: list[float | None], notes: list[str]):
    def write(ws: Path, i: int):
        b = ws / "bundle"
        b.mkdir(parents=True, exist_ok=True)
        th = thresholds[min(i, len(thresholds) - 1)]
        rules = [] if th is None else [{"decision": DECISION[inst], "when": f"obs.amount > {th:g}",
                                        "action": "decline"}]
        rules += [{"decision": DECISION[inst], "when": f"obs.velocity_24h > {5 + j}", "action": "decline"}
                  for j in range(i if th is not None else 0)]
        (b / "policy.yaml").write_text(json.dumps({"rules": rules}))
        (b / "NOTES.md").write_text(notes[min(i, len(notes) - 1)] + "\n")
    return write


def play(out: Path, sandbox: Path, *, seed: int = 7, log=lambda r: None, replay: bool = True):
    """Plays the scripted demo match (8 weekly periods and 2 settling weeks)."""
    from .game import Game, GameConfig, Player

    cfg = GameConfig(world=Path("demo-economy"), seed=seed, start_day=28, period_days=7, horizon_days=56,
                     settle_days=14, manual_review=False, replay=replay)
    bounds = cfg.boundaries()[:-1]

    def defender(inst, thresholds, notes):
        inner = _defender(inst, thresholds, notes)

        def write(ws, i):
            inner(ws, i)
            (ws / "review").mkdir(exist_ok=True)
            (ws / "review" / f"day-{bounds[min(i, len(bounds) - 1)]}.md").write_text(
                f"# Week {i + 1} review\n\n" + "Losses by scheme and what the rules caught. " * 8)
        return write

    players = [
        Player("defender-brindlemark", "defender", "brindlemark", Scripted("demo", "tightens", defender(
            "brindlemark", [2000, 1200, 800, 600, 500], ["Lower the amount limit on card auths",
                                                         "Ring R2 moved under 2k; follow it down",
                                                         "Add velocity rules; keep the limit at 800",
                                                         "Hold the line at 600", "Fine-tune velocity"]))),
        Player("defender-dovrex", "defender", "dovrex", Scripted("demo", "steady", defender(
            "dovrex", [1500], ["Review withdrawals over 1.5k"]))),
        Player("defender-hopcrate", "defender", "hopcrate", Scripted("demo", "idle", defender(
            "hopcrate", [None], ["Built-in rules are fine"]))),
        Player("defender-kellmont", "defender", "kellmont", Scripted("demo", "strict", defender(
            "kellmont", [400, 250], ["Decline orders over 400", "Decline orders over 250"]))),
    ]

    def attacker(ring):
        def write(ws: Path, i: int):
            base = dict(RINGS[ring]["plan"])
            if ring == 2:
                caps = [3800, 1900, 1100, 750, 550, 450, 450, 450]
                base |= {"max_amount": caps[min(i, 7)], "volume": 1.0 + 0.25 * min(i, 4),
                         "targets": {"brindlemark": max(1, 3 - i // 2), "dovrex": 1 + i // 2}}
            else:
                base |= {"volume": [1.2, 1.6, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0][min(i, 7)],
                         "targets": {"hopcrate": 2 + i // 3, "kellmont": max(0, 2 - i // 2)}}
            (ws / "plan.json").write_text(json.dumps(base))
        return write

    g = Game(cfg, SimEngine(seed), players, out, sandbox, attacker_api=SimAttackers(),
             attacker_runners=lambda i, ring: Scripted("demo", f"ring-{ring}-plan", attacker(ring)), log=log)
    return g.run()


SYNTHETIC_NOTE = ("Synthetic: a scripted demo match in a toy economy (fraud_arena.demo), not Fraud World and not "
                  "model players. Shapes and units are real; the numbers mean nothing.")


def main(argv=None) -> int:
    import tempfile

    from .replay import set_meta

    out = Path((argv or sys.argv[1:])[0])
    with tempfile.TemporaryDirectory() as sb:
        play(out, Path(sb))
    set_meta(out, synthetic=True, synthetic_note=SYNTHETIC_NOTE, world={"id": "toy economy (synthetic)", "path": None})
    print(out / "replay" / "replay.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
