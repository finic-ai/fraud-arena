"""Who did what to whom at Brindlemark, from the simulator's truth (never shown to players).

involvements(): one row per (customer, fraud scenario instance) that touched the customer's Brindlemark rows, with
    the typology family, how the customer took part (victim / self / mule / opened / incidental), the crew (ring) if a
    ring ran it, and when it started and became visible to the bank.
customer_truth(as_of): each customer's role and typology from the involvements that started by `as_of`.

Roles: a stolen or synthetic identity that a fraudster opened is a perpetrator (its typology is the scheme it was
opened for); a real customer whose account moves others' stolen money (a ring's recruited mule, or a mule scenario)
is a mule; a real customer who commits first-party fraud is a perpetrator; a real customer whose account or card a
fraudster used, or who was scammed into paying, is a victim. A real customer who only *received* money from a fraud
(e.g. sold something to a ring account) and nothing else is "incidental": neither a clean negative nor a fraud
subject, so the case builders leave them out of sampling."""

from collections import defaultdict
from pathlib import Path

import duckdb

from .schema import PAYING_EVENTS, SCENARIOS

RING_PREFIX = {f"r{i}_": f"R{i}" for i in range(1, 11)}
ROLE_PRIORITY = {"perpetrator": 3, "mule": 2, "victim": 1}


def _ring_of_scenario(sc):
    for p, r in sorted(RING_PREFIX.items(), key=lambda x: -len(x[0])):
        if sc and sc.startswith(p):
            return r
    return None


class Truth:
    def __init__(self, out):
        out = Path(out)
        self.out = out
        db = duckdb.connect()
        w = out / "truth" / "world"
        db.execute(f"CREATE VIEW hidden AS SELECT * FROM read_parquet('{w / 'hidden.parquet'}')")
        db.execute(f"CREATE VIEW rows AS SELECT * FROM read_parquet('{out / 'truth' / 'rows.parquet'}')")
        for m in ("customers", "accounts", "devices", "cards"):
            db.execute(f"CREATE VIEW map_{m} AS SELECT * FROM read_parquet('{out / 'truth' / f'map_{m}.parquet'}')")
        self.db = db
        rings = db.execute("""SELECT entity, value FROM hidden WHERE type = 'ring' AND attribute = 'name'""").fetchall()
        self.ring_name = dict(rings)
        # identity kind and operator per customer
        self.identity = {}
        for cust, kind, op in db.execute("""
                SELECT m.CUST_ID, max(CASE WHEN h.attribute = 'kind' THEN h.value END),
                       max(CASE WHEN h.attribute = 'operator' THEN h.value END)
                FROM map_customers m JOIN hidden h ON h.entity = m.entity AND h.type = 'identity'
                GROUP BY 1""").fetchall():
            self.identity[cust] = {"kind": kind or "real", "operator": op}
        # account ring / controlled_by changes per customer: (time, attribute, value)
        self.acct_hist = defaultdict(list)
        for cust, attr, val, ts in db.execute("""
                SELECT a.CUST_ID, h.attribute, h.value, h.valid_from FROM map_accounts a
                JOIN hidden h ON h.entity = a.entity AND h.type = 'account'
                WHERE h.attribute IN ('ring', 'controlled_by') AND h.value IS NOT NULL
                ORDER BY h.valid_from, a.CUST_ID, h.attribute, h.value""").fetchall():
            self.acct_hist[cust].append((ts, attr, self.ring_name.get(val, val) if attr == "ring" else val))
        # the fraudster's ring, for operators of opened identities
        self.fraudster_ring = {e: self.ring_name.get(v) for e, v in db.execute(
            "SELECT entity, value FROM hidden WHERE type = 'fraudster' AND attribute = 'ring' AND value IS NOT NULL").fetchall()}
        self.inv = self._involvements()

    def _involvements(self):
        rows = self.db.execute("""
            SELECT CUST_ID, sc, inst, tbl, ev, actor_type, time, load_time, rn, actor FROM rows
            WHERE sc IS NOT NULL AND (lab = 'fraud' OR sc IN ('ring_cover_activity', 'ring_operation'))
            ORDER BY time, tbl, rn""").fetchall()
        groups = defaultdict(list)
        for r in rows:
            groups[(r[0], r[1], r[2])].append(r)
        out = []
        for (cust, sc, inst), rs in groups.items():
            family, kind = SCENARIOS.get(sc, ("other", "victim"))
            ident = self.identity.get(cust, {"kind": "real"})
            evs = {(r[4], r[5]) for r in rs}
            real = ident["kind"] == "real"
            ring = _ring_of_scenario(sc)
            if kind == "cover":
                part = "cover"
            elif not real:
                part = "opened"          # a fraudster's own account, whatever money passed through it
            elif kind == "opened":
                part = "mule" if self._ring_at(cust, rs[-1][6]) else "incidental"
            elif kind == "victim":
                if family == "scam_victim":
                    paid = any(e in PAYING_EVENTS and a == "person" for e, a in evs)
                    part = "victim" if paid else ("mule" if self._ring_at(cust, rs[-1][6]) else "incidental")
                elif sc in ("triangulation", "r6_triangulation") and not any(a == "fraudster" for _, a in evs):
                    part = "incidental"  # bought the ring's listing with the customer's own card
                else:
                    part = "victim"
            elif kind == "self":
                part = "self"
            else:
                part = "mule"
            if ring is None and sc in ("ring_mule_flow", "ring_cover_activity", "ring_operation"):
                ring = self._ring_at(cust, rs[-1][6]) or (self.fraudster_ring.get(ident.get("operator")) if not real else None)
            if ring is None and not real:
                ring = self.fraudster_ring.get(ident.get("operator"))
            role = {"victim": "victim", "self": "perpetrator", "mule": "mule", "opened": "perpetrator"}.get(part)
            out.append({"cust": cust, "scenario": sc, "instance": inst, "family": family, "part": part, "role": role,
                        "ring": ring, "start": rs[0][6], "end": rs[-1][6],
                        "first_load": min(r[7] for r in rs if r[7] is not None),
                        "n_rows": len(rs), "tables": sorted({r[3] for r in rs}),
                        "events": sorted({e for e, _ in evs}),
                        "fraudster_acted": any(a == "fraudster" for _, a in evs),
                        "fraudster": next((r[9] for r in rs if r[5] == "fraudster"), None),
                        # per row: (event time, load time, event, actor type): what the bank could see, and when
                        "rows": [(r[6], r[7], r[4], r[5]) for r in rs]})
        return out

    def _ring_at(self, cust, ts):
        ring = None
        for t, attr, val in self.acct_hist.get(cust, []):
            if t <= ts and attr == "ring":
                ring = val
        return ring

    def ring_at(self, cust, ts):
        return self._ring_at(cust, ts)

    def customer(self, cust, as_of=None, since=None):
        """Role, typology, ring and supporting involvements of one customer, from involvements that started in
        [since, as_of]. Returns None for a customer with no fraud involvement there (a clean customer) and a dict with
        role 'incidental' when only incidental involvements exist."""
        ident = self.identity.get(cust, {"kind": "real"})
        invs = [i for i in self.inv if i["cust"] == cust and (as_of is None or i["start"] <= as_of)
                and (since is None or i["end"] >= since)]
        return resolve(ident, invs, self._ring_at(cust, as_of) if as_of else None)

    def by_customer(self):
        d = defaultdict(list)
        for i in self.inv:
            d[i["cust"]].append(i)
        return d

    def ever_touched(self):
        """Customers with any fraud involvement, ring account marks, or a non-real identity, at any time."""
        s = {i["cust"] for i in self.inv}
        s |= {c for c, h in self.acct_hist.items() if h}
        s |= {c for c, v in self.identity.items() if v["kind"] != "real"}
        s.discard(None)  # a fraud event on an account with no Brindlemark customer (e.g. an outside account)
        return s


def resolve(ident, invs, ring_now=None):
    real = ident["kind"] == "real"
    if not invs and real and not ring_now:
        return None
    if not real:
        opened = [i for i in invs if i["part"] == "opened" and i["family"]]
        # the scheme the identity was opened for: the earliest 'opened' involvement of a ring/opener scenario
        prim = next((i for i in opened if SCENARIOS.get(i["scenario"], (None, ""))[1] == "opened"), None) or \
            (opened[0] if opened else None)
        fam = prim["family"] if prim else "synthetic_identity"
        rings = [i["ring"] for i in invs if i["ring"]]
        return {"role": "perpetrator", "typology": fam, "ring": rings[0] if rings else None,
                "scenario": prim["scenario"] if prim else None, "instance": prim["instance"] if prim else None,
                "identity": ident["kind"], "involvements": invs}
    real_invs = [i for i in invs if i["role"]]
    if not real_invs:
        if ring_now:
            return {"role": "mule", "typology": "mule", "ring": ring_now, "scenario": None, "instance": None,
                    "identity": "real", "involvements": invs}
        return {"role": "incidental", "typology": None, "ring": None, "scenario": None, "instance": None,
                "identity": "real", "involvements": invs}
    best = sorted(real_invs, key=lambda i: (-ROLE_PRIORITY[i["role"]], i["start"]))[0]
    typ = best["family"] if best["role"] != "mule" else "mule"
    return {"role": best["role"], "typology": typ, "ring": best["ring"], "scenario": best["scenario"],
            "instance": best["instance"], "identity": "real", "involvements": invs}
