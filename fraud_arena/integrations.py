"""A defender's integrations: the data sources its decision points call when a case arrives, as a fraud team's
decision platform does. Declared in `bundle/policy.yaml`:

    integrations:
      claims_90d:                       # a lookup in the institution's own records
        source: own_data
        sql: "select count(*) from CLAIMS where CUST_ID = $cust and CLAIM_TS > $now - interval 90 day"
        params: {cust: P2P.CUST_ID}     # a case identifier (`<TABLE>.<COLUMN>`) or `feature.<name>`
        default: 0                      # when the lookup finds nothing or fails
      payee_site:                       # the public web (ads, domains, websites, block explorer)
        source: public
        sql: "select * from WEBSITES where DOMAIN = $d"
        params: {d: feature.payee_domain}
      scam_story:                       # Grayson, with the case and other integrations as its data
        source: grayson
        instructions: "Is this payment part of a scam the customer is being coached through?"
        criteria: {yes: "...", no: "..."}
        data: [case, claims_90d, payee_site]

A lookup whose result is one value (the first column of the first row) is a numeric feature, `outside.<name>` in
rules and models; a Grayson integration is a probability per option, `outside.<name>.<option>`. Lookups also serve
as context for Grayson (`data:`), as their rows. The engine asks for a feature only when a rule or model reaches it,
so cases decided earlier cost nothing.

The data is the institution's own tables as the harness last refreshed them for the game (see `Data`), never truth.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {"own_data", "public", "grayson"}
PARAM = re.compile(r"\$([a-z_][a-z0-9_]*)")
MAX_ROWS = 50


class IntegrationError(Exception):
    pass


@dataclass
class Integration:
    name: str
    source: str
    sql: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    default: object = None
    instructions: str = ""
    criteria: dict[str, str] = field(default_factory=dict)
    data: list[str] = field(default_factory=list)

    @property
    def options(self) -> list[str]:
        return list(self.criteria)


def parse(policy: dict) -> dict[str, Integration]:
    """Takes `integrations` out of a policy (the engine never sees it) and checks it."""
    raw = policy.pop("integrations", None) or {}
    if not isinstance(raw, dict):
        raise IntegrationError("integrations must be a mapping of name to integration")
    out: dict[str, Integration] = {}
    for name, spec in raw.items():
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", str(name)):
            raise IntegrationError(f"integration name `{name}`: lower case letters, digits and _")
        spec = spec or {}
        src = spec.get("source")
        if src not in SOURCES:
            raise IntegrationError(f"integration `{name}`: source is one of {sorted(SOURCES)}")
        if src == "grayson":
            crit = spec.get("criteria") or {}
            if not isinstance(crit, dict) or len(crit) < 2:
                raise IntegrationError(f"integration `{name}`: grayson needs criteria with at least two options")
            data = [str(d) for d in (spec.get("data") or ["case"])]
            out[name] = Integration(name, src, instructions=str(spec.get("instructions", "")),
                                    criteria={str(k): str(v) for k, v in crit.items()}, data=data)
        else:
            sql = spec.get("sql")
            if not isinstance(sql, str) or not sql.strip().lower().startswith(("select", "with")):
                raise IntegrationError(f"integration `{name}`: sql must be one SELECT query")
            params = {str(k): str(v) for k, v in (spec.get("params") or {}).items()}
            for p in PARAM.findall(sql):
                if p != "now" and p not in params:
                    raise IntegrationError(f"integration `{name}`: ${p} has no entry in params")
            out[name] = Integration(name, src, sql=sql, params=params, default=spec.get("default"))
    for i in out.values():
        for d in i.data:
            if d != "case" and d not in out:
                raise IntegrationError(f"integration `{i.name}`: data `{d}` isn't an integration")
            if d in out and out[d].source == "grayson":
                raise IntegrationError(f"integration `{i.name}`: data `{d}` is a Grayson integration; give it "
                                       f"lookups or `case`")
    return out


def declare(policy: dict, integrations: dict[str, Integration]) -> set[str]:
    """Declares the integrations the policy's rules and models read as engine outside features, and adds the
    `external` entry each decision that reads one needs. Returns the names the engine may ask for."""
    models = policy.get("models") or {}
    reads = lambda text: {n for n in integrations if re.search(rf"\boutside\.{n}\b", text)}
    model_reads = {m: reads(json.dumps(spec)) for m, spec in models.items()}
    used: set[str] = set()
    for r in policy.get("rules", []):
        when = str(r.get("when", ""))
        got = reads(when) | {n for m, ns in model_reads.items() if re.search(rf"\bscore\.{m}\b", when) for n in ns}
        if got:
            used |= got
            policy.setdefault("external", {}).setdefault(r["decision"], {})
    if used:
        policy["format"] = max(int(policy.get("format", 1)), 2)
        outside = policy.setdefault("outside", {})
        for n in used:
            i = integrations[n]
            outside[n] = {"options": i.options} if i.source == "grayson" else {"type": "number"}
    return used


def _iso(ms) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Data:
    """The institution's tables for lookups, loaded into memory from a folder of parquet files (its own tables and
    `public/`, also as `PUBLIC_<TABLE>`). The defender writes the SQL, so once loaded the connection can read no
    file and its settings are locked: a query sees these tables and nothing else on the host."""

    def __init__(self):
        self.db = None
        self.tables: set[str] = set()

    def load(self, folder: Path) -> None:
        import duckdb
        folder = Path(folder)
        db = duckdb.connect()
        names = set()
        for f in sorted(folder.glob("*.parquet")):
            db.execute(f'create table "{f.stem}" as select * from read_parquet(?)', [str(f)])
            names.add(f.stem)
        for f in sorted((folder / "public").glob("*.parquet")):
            db.execute(f'create table "PUBLIC_{f.stem}" as select * from read_parquet(?)', [str(f)])
            if f.stem not in names:
                db.execute(f'create view "{f.stem}" as select * from "PUBLIC_{f.stem}"')
            names |= {f.stem, f"PUBLIC_{f.stem}"}
        db.execute("set enable_external_access = false")
        db.execute("set lock_configuration = true")
        old, self.db, self.tables = self.db, db, names
        if old is not None:
            old.close()

    def query(self, sql: str, values: dict) -> tuple[list[str], list[tuple]]:
        if self.db is None:
            return [], []
        order = []

        def sub(m):
            order.append(m.group(1))
            return "?"
        q = PARAM.sub(sub, sql)
        cur = self.db.cursor().execute(q, [values.get(p) for p in order])
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchmany(MAX_ROWS)


def _param_value(ref: str, case: dict):
    if ref.startswith("feature."):
        return (case.get("features") or {}).get(ref.split(".", 1)[1])
    return (case.get("identifiers") or {}).get(ref)


def _scalar(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


class Resolver:
    """Computes the features waiting cases need, for one defender: lookups per case, Grayson in one batch."""

    def __init__(self, integrations: dict[str, Integration], data: Data, grayson_client, *, case_view=None, log=None):
        self.integrations = integrations
        self.data = data
        self.client = grayson_client
        self.case_view = case_view or (lambda case: case.get("features") or {})
        self.log = log or (lambda rows: None)
        self.stats = {"lookups": 0, "lookup_errors": 0, "grayson": 0, "grayson_errors": 0}

    def _lookup(self, i: Integration, case: dict) -> tuple[list[str], list[tuple]] | None:
        values = {p: _param_value(ref, case) for p, ref in i.params.items()}
        values["now"] = _iso(case.get("time"))
        self.stats["lookups"] += 1
        try:
            return self.data.query(i.sql, values)
        except Exception:
            self.stats["lookup_errors"] += 1
            return None

    def answer(self, cases: list[dict]) -> dict[str, dict]:
        """`{case key: {feature: value}}` for every feature each case needs that is one of these integrations."""
        out: dict[str, dict] = {}
        rows_cache: dict[tuple[str, str], tuple | None] = {}

        def rows(name, case):
            k = (name, case["key"])
            if k not in rows_cache:
                rows_cache[k] = self._lookup(self.integrations[name], case)
            return rows_cache[k]

        asks = []
        for c in cases:
            ans = out.setdefault(c["key"], {})
            for n in c.get("needs") or []:
                i = self.integrations.get(n)
                if i is None:
                    continue
                if i.source == "grayson":
                    data = {}
                    for d in i.data:
                        if d == "case":
                            data["case"] = self.case_view(c)
                        else:
                            r = rows(d, c)
                            data[d] = [dict(zip(r[0], row)) for row in r[1]] if r else None
                    asks.append((c, i, {"data": json.loads(json.dumps(data, default=str)),
                                        "questions": {n: {"type": "choice", "instructions": i.instructions,
                                                          "criteria": i.criteria}}}))
                else:
                    r = rows(n, c)
                    v = _scalar(r[1][0][0]) if r and r[1] and r[1][0] else None
                    if v is None:
                        v = _scalar(i.default)
                    ans[n] = 0.0 if v is None else v
        if asks:
            outs = self.client.ask_many([a[2] for a in asks]) if self.client is not None else \
                [{"error": "Grayson isn't available in this game"} for _ in asks]
            logged = []
            for (c, i, req), o in zip(asks, outs):
                self.stats["grayson"] += 1
                probs = ((o.get("answers") or {}).get(i.name) or {}).get("probabilities") or {}
                p = {opt: float(probs.get(opt, 0.0)) for opt in i.options}
                tot = sum(p.values())
                if "error" in o or tot <= 0:
                    self.stats["grayson_errors"] += 1
                    p = {opt: 1.0 / len(i.options) for opt in i.options}  # no answer: no information
                else:
                    p = {k: v / tot for k, v in p.items()}
                out[c["key"]][i.name] = p
                logged.append({"where": c.get("decision"), "integration": i.name, "data": req["data"],
                               "answers": o.get("answers"), "error": o.get("error"), "given": p})
            self.log(logged)
        return out
