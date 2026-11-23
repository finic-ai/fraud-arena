"""Export-time rewrites of the bank tables (bank/), applied after world.materialize and before any case is built.

Nothing here changes the world's logic; it only rewrites what the bank's tables show, deterministically per seed:

1. Realistic contact details. The world writes reserved names (`example.com`, `isp.example`, ...). They are mapped to
   real-looking providers that keep the world's meaning (the largest free provider, an older-skewed portal, an
   internet provider's mail, a privacy provider, a disposable service), and every personal local part gets a 4-6 digit
   suffix derived from a hash of the original address, so the addresses very likely belong to no real person. The
   map is a pure function of the address, so an address is the same in every table (CUSTOMERS, P2P, PAYEES,
   PROFILE_CHANGES, TREASURY_USERS, REVIEW_QUEUE...). Addresses a crew generates from one template (the world's
   shared style and number block) keep a shared look: their suffix is derived from the crew's template, not the
   address. Company domains (`*.example.com` ...) map to invented domains (COMPANY_DOMAINS, or a fixed rule) that
   were checked to be unregistered (tools/check_domains.py).

2. Per-seed shape variation. Every fraud scheme's money amounts are scaled by a factor drawn per (seed, scheme)
   (0.6-1.7, keeping round-number granularity), and counterparties that only fraud uses (a crew's funding
   originators) get names and entry descriptions drawn per seed from a pool. The same rule applies to every scheme, so
   two seeds of one scheme differ in amounts, fee levels and originator names. Schemes defined by a legal threshold
   (structuring under the cash-report line) keep their amounts.
"""

import glob
import hashlib
import json
import random
from pathlib import Path

import duckdb

# --- 1. contact details -----------------------------------------------------------------------------------------------

# world provider -> [(real domain, weight)]. The world's provider shares (Fraud World's email
# `provider`) encode: example.com the largest free provider (more among the young), example.net an older-skewed
# portal, example.org a mid-age webmail, inbox.example a young-skewed device-maker mail, isp.example an internet
# provider's mail (older people), privatemail.example a privacy provider, tempbox.example a disposable service.
PROVIDERS = {
    "example.com": [("gmail.com", 1.0)],
    "example.net": [("yahoo.com", 0.7), ("aol.com", 0.3)],
    "example.org": [("outlook.com", 0.55), ("hotmail.com", 0.45)],
    "inbox.example": [("icloud.com", 1.0)],
    "isp.example": [("comcast.net", 0.35), ("att.net", 0.25), ("verizon.net", 0.2), ("sbcglobal.net", 0.2)],
    "privatemail.example": [("proton.me", 0.7), ("tutanota.com", 0.3)],
    "tempbox.example": [("guerrillamail.com", 0.6), ("10minutemail.com", 0.4)],
}

# Company domains seen in the bank's tables -> invented domains. Every value was checked with
# `dig +short NS` / whois to be unregistered when it was chosen (tools/check_domains.py re-checks a build).
COMPANY_DOMAINS = {
    "hessmark-mfg.example.com": "hessmark-mfg-works.com",
    "stbrennanhealth.example.org": "stbrennan-healthsys.org",
    "quillstack.example.com": "quillstack-software-hq.com",
}
# Other company domains (generated business websites: `<slug>.example.<tld>`) follow a fixed rule, with overrides for
# any rule output found registered.
COMPANY_RULE_SUFFIX = "-bizhq"
COMPANY_OVERRIDES = {}


def _h(*parts):
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def _pick(options, h):
    x = (h % 10_000) / 10_000
    acc = 0.0
    for dom, w in options:
        acc += w
        if x < acc:
            return dom
    return options[-1][0]


def company_domain(dom):
    dom = dom.lower()
    if dom in COMPANY_DOMAINS:
        return COMPANY_DOMAINS[dom]
    if dom in COMPANY_OVERRIDES:
        return COMPANY_OVERRIDES[dom]
    slug, _, tld = dom.partition(".example")
    tld = tld.lstrip(".") or "com"
    return f"{slug}{COMPANY_RULE_SUFFIX}.{tld}"


def is_company(dom):
    return ".example." in dom or (dom.endswith(".example") and dom not in PROVIDERS)


def suffix(h):
    n = 4 + h % 3                       # 4-6 digits
    return str(h // 7 % (10 ** n)).zfill(n)


def map_email(addr, templates=None):
    """One address -> its exported form. `templates` maps template-generated addresses (lower case) to their
    template id (a crew's shared style), whose suffix and provider are shared."""
    if addr is None or "@" not in addr:
        return addr
    local, _, dom = addr.strip().lower().rpartition("@")
    if is_company(dom):
        return f"{local}@{company_domain(dom)}"
    if dom not in PROVIDERS:
        return addr.lower()
    tpl = (templates or {}).get(f"{local}@{dom}")
    if tpl is not None:
        h = _h("tpl", tpl, dom)
        # keep the template's number block visible: the shared suffix goes in front of the address's own digits
        stem = local.rstrip("0123456789")
        digits = local[len(stem):]
        return f"{stem}{str(h % 1000).zfill(3)}{digits}@{_pick(PROVIDERS[dom], h // 1000)}"
    h = _h("addr", local, dom)
    return f"{local}{suffix(h)}@{_pick(PROVIDERS[dom], h // 10**7)}"


def map_url(url):
    if url is None:
        return url
    dom = url.lower()
    return company_domain(dom) if is_company(dom) else url


def template_addresses(out):
    """Addresses the world generated from a crew's shared template: {address: template id}."""
    w = Path(out) / "truth" / "world" / "hidden.parquet"
    db = duckdb.connect()
    rows = db.execute(f"""
        WITH e AS (SELECT entity, attribute, arg_max(value, valid_from) v FROM read_parquet('{w}')
                   WHERE type = 'email_address' AND attribute IN ('ring', 'ring_text') GROUP BY ALL)
        SELECT max(v) FILTER (WHERE attribute = 'ring_text'), max(v) FILTER (WHERE attribute = 'ring')
        FROM e GROUP BY entity HAVING max(v) FILTER (WHERE attribute = 'ring') IS NOT NULL""").fetchall()
    return {t.lower(): r for t, r in rows if t}


# table -> columns holding an email address / handle, or a company website
EMAIL_COLUMNS = {
    "CUSTOMERS": ["EMAIL"], "P2P": ["CP_HANDLE"], "PAYEES": ["PAYEE_HANDLE"], "PROFILE_CHANGES": ["NEW_EMAIL"],
    "TREASURY_USERS": ["EMAIL"], "REVIEW_QUEUE": ["CP_HANDLE"], "BENEFICIARIES": [], "ACCOUNT_APPLICATIONS": [],
}
URL_COLUMNS = {"BUSINESS_APPLICATIONS": ["WEBSITE"]}


# --- 2. shape variation -----------------------------------------------------------------------------------------------

THRESHOLD_BOUND = {"structuring"}
PRIMARY_AMT = ["ACH", "P2P", "CARD_AUTHS", "CONSUMER_WIRES", "CASH", "CHECKS", "INTERNAL_TRANSFERS", "CARD_CREDITS"]
DERIVED_AMT = ["ACH_RETURNS", "DISPUTES", "CLAIMS", "FRAUD_TAGS", "REVIEW_QUEUE"]
ORIGINATOR_POOL = [
    ("BRIGHTWAVE PAYOUTS", "GIG_PAYOUT"), ("NOVAQUILL DISBURSE", "SETTLEMENT"), ("KESTRELPAY", "PAYROLL"),
    ("LUMENPATH FUNDING", "LOAN"), ("TALLOWAY STAFFING", "PAYROLL"), ("ORRINDALE CAPITAL", "LOAN"),
    ("VEXLEY PAYMENTS", "SETTLEMENT"), ("CINDERFALL SERVICES", "GIG_PAYOUT"), ("HOLLISTRA PAY CO", "PAYROLL"),
    ("MARRIN LABOR GROUP", "PAYROLL"), ("QUINTREL REWARDS", "SETTLEMENT"), ("SABLEROCK FINANCE", "LOAN"),
    ("TIDEWELL GIGS", "GIG_PAYOUT"), ("ARDENTRY PAYROLL", "PAYROLL"), ("PELLIMORE CASH ADV", "LOAN"),
    ("ZANTHRO DISBURSEMENTS", "SETTLEMENT"), ("COBBLEFIELD WORKS", "PAYROLL"), ("FENWISK PAYOUT", "GIG_PAYOUT"),
]


def _granular_round(orig, new):
    cents = round(orig * 100)
    for unit in (5000, 1000, 100):
        if cents and cents % unit == 0:
            return round(round(new * 100 / unit) * unit / 100, 2) or unit / 100
    return round(new, 2)


def scheme_factors(scenarios, seed):
    rng = random.Random(f"amounts:{seed}")
    out = {}
    for sc in sorted(scenarios):
        f = rng.uniform(0.6, 1.7)
        out[sc] = 1.0 if sc in THRESHOLD_BOUND else round(f if abs(f - 1) > 0.12 else f + 0.25, 3)
    return out


def rewrite_bank(out, seed, log=print):
    """Rewrite bank/*.parquet in place: amounts and fraud-only originators (2), then contact details (1). Writes
    truth/export.json (the factors and renames, for graders)."""
    out = Path(out)
    bank = out / "bank"
    db = duckdb.connect()
    db.execute(f"CREATE TABLE rows AS SELECT tbl, rn, CUST_ID, sc, lab FROM read_parquet('{out / 'truth' / 'rows.parquet'}')")
    fraud_sc = [r[0] for r in db.execute("SELECT DISTINCT sc FROM rows WHERE lab = 'fraud' AND sc IS NOT NULL").fetchall()]
    factors = scheme_factors(fraud_sc, seed)
    db.execute("CREATE TABLE fac AS SELECT * FROM (VALUES " +
               ", ".join(f"('{k}', {v})" for k, v in factors.items()) + ") t(sc, f)" if factors else
               "CREATE TABLE fac (sc VARCHAR, f DOUBLE)")
    tables = {Path(f).stem: f for f in glob.glob(str(bank / "*.parquet"))}
    for t, f in tables.items():
        db.execute(f'CREATE TABLE "{t}" AS SELECT *, row_number() OVER () - 1 AS _rn FROM read_parquet(\'{f}\')')
    db.create_function("gr", _granular_round, ["DOUBLE", "DOUBLE"], "DOUBLE")
    # primary rows: scale by their scheme's factor; remember (customer, old amount) -> new amount for derived rows
    db.execute("CREATE TABLE amtmap (CUST_ID VARCHAR, old DOUBLE, new DOUBLE)")
    for t in PRIMARY_AMT:
        if t not in tables:
            continue
        db.execute(f"""CREATE TEMP TABLE s AS SELECT o._rn, o.CUST_ID, CAST(o.AMT AS DOUBLE) old,
                              gr(CAST(o.AMT AS DOUBLE), CAST(o.AMT AS DOUBLE) * fac.f) new
                       FROM "{t}" o JOIN rows r ON r.tbl = '{t}' AND r.rn = o._rn AND r.lab = 'fraud'
                       JOIN fac USING (sc) WHERE o.AMT IS NOT NULL AND fac.f <> 1""")
        db.execute(f'UPDATE "{t}" SET AMT = s.new FROM s WHERE "{t}"._rn = s._rn')
        db.execute("INSERT INTO amtmap SELECT CUST_ID, old, any_value(new) FROM s GROUP BY ALL")
        db.execute("DROP TABLE s")
    db.execute("CREATE TABLE am AS SELECT CUST_ID, old, any_value(new) new FROM amtmap GROUP BY ALL")
    for t in DERIVED_AMT:
        if t in tables:
            db.execute(f"""UPDATE "{t}" SET AMT = am.new FROM am
                           WHERE am.CUST_ID = "{t}".CUST_ID AND am.old = CAST("{t}".AMT AS DOUBLE)""")
    # fraud-only originators: every credit row from them is fraud-caused
    rng = random.Random(f"originators:{seed}")
    pool = ORIGINATOR_POOL[:]
    rng.shuffle(pool)
    origs = [r[0] for r in db.execute("""
        SELECT o.CP_NAME FROM "ACH" o LEFT JOIN rows r ON r.tbl = 'ACH' AND r.rn = o._rn
        WHERE o.DIRECTION = 'CREDIT' AND o.CP_COMPANY_ID IS NULL AND o.CP_NAME IS NOT NULL AND o.ENTRY_DESC <> 'TRANSFER'
        GROUP BY 1 HAVING bool_and(coalesce(r.lab, '') = 'fraud') ORDER BY count(*) DESC, 1""").fetchall()]
    renames = {}
    for i, name in enumerate(origs):
        new, desc = pool[i % len(pool)]
        renames[name] = {"name": new if i < len(pool) else f"{new} {i // len(pool) + 1}", "desc": desc}
    for old, r in renames.items():
        db.execute("""UPDATE "ACH" SET CP_NAME = ?, ENTRY_DESC = CASE WHEN ENTRY_DESC IN ('PAYROLL', 'GIG_PAYOUT',
                      'SETTLEMENT', 'LOAN') THEN ? ELSE ENTRY_DESC END WHERE CP_NAME = ? AND CP_COMPANY_ID IS NULL""",
                   [r["name"], r["desc"], old])
        for t in ("ACH_RETURNS",):
            db.execute(f'UPDATE "{t}" SET CP_NAME = ? WHERE CP_NAME = ?', [r["name"], old])
        if "REVIEW_QUEUE" in tables:
            db.execute('UPDATE "REVIEW_QUEUE" SET COUNTERPARTY = ? WHERE COUNTERPARTY = ?', [r["name"], old])
    # contact details
    tpl = template_addresses(out)
    db.create_function("map_email", lambda a: map_email(a, tpl), ["VARCHAR"], "VARCHAR")
    db.create_function("map_url", map_url, ["VARCHAR"], "VARCHAR")
    n_mapped = 0
    for t, cols in EMAIL_COLUMNS.items():
        for c in cols:
            if t in tables and c in _cols(db, t):
                db.execute(f'UPDATE "{t}" SET "{c}" = map_email("{c}") WHERE "{c}" LIKE \'%@%\'')
                n_mapped += 1
    for t, cols in URL_COLUMNS.items():
        for c in cols:
            if t in tables and c in _cols(db, t):
                db.execute(f'UPDATE "{t}" SET "{c}" = map_url("{c}")')
    for t, f in tables.items():
        db.execute(f'COPY (SELECT * EXCLUDE (_rn) FROM "{t}" ORDER BY _rn) TO \'{f}\' (FORMAT parquet, COMPRESSION zstd)')
    leftovers = scan_reserved(bank)
    meta = {"seed": seed, "amount_factors": factors, "originator_renames": renames,
            "template_addresses": len(tpl), "leftover_reserved_names": leftovers}
    (out / "truth" / "export.json").write_text(json.dumps(meta, indent=1))
    log(f"  export: {len(factors)} scheme factors, {len(renames)} originators renamed, {len(tpl)} template "
        f"addresses, reserved names left: {leftovers or 'none'}")
    return meta


def _cols(db, t):
    return {r[0] for r in db.execute(f'DESCRIBE "{t}"').fetchall()}


def scan_reserved(bank):
    """Columns that still hold a reserved example name after the rewrite: {table.column: count}."""
    db = duckdb.connect()
    left = {}
    for f in sorted(glob.glob(str(Path(bank) / "*.parquet"))):
        t = Path(f).stem
        for c, ty, *_ in db.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").fetchall():
            if "VARCHAR" not in ty:
                continue
            n = db.execute(f"""SELECT count(*) FROM read_parquet('{f}') WHERE regexp_matches(lower("{c}"),
                               '(^|[.@])example([.]|$)')""").fetchone()[0]
            if n:
                left[f"{t}.{c}"] = n
    return left
