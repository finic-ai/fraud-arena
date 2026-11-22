"""The `case` command for one workspace. The runner writes a wrapper script that runs this module with the workspace
baked in. Everything it shows comes from the bank's tables as of the case time (casedb.open_asof); it never opens
truth. Every command is logged to the workspace's log.jsonl."""

import json
import sys
import time
from pathlib import Path

from .schema import ALERT_ACTIONS, DISPOSITIONS, INCIDENT_ACTIONS, ROLES, TYPOLOGIES

USAGE = """usage:
  casefile brief                      the case and how it is scored
  casefile tables                     the bank's tables and columns, as of the case time
  casefile query "SQL"                read-only DuckDB SQL over the bank's tables as of the case time (up to 200 rows)
  casefile customer CUST_ID [--all]   profile, accounts, cards and timeline (latest 150 events; --all for every one)
  casefile links CUST_ID --by KIND    customers sharing each KIND key with CUST_ID, and how many customers share it
                                  KIND: device ip phone email address ach_originator payee merchant referral p2p_sender
  casefile note "TEXT"                append a note to notes.md
  casefile notes                      print your notes
  casefile submit FILE                check and submit your answer (JSON file, e.g. drafts/answer.json)"""

# per table: the time column and the columns that summarise a row in a customer's timeline
TIMELINE = {
    "ACCOUNT_APPLICATIONS": ("APP_TS", ["PRODUCT", "CHANNEL", "DECISION", "ID_MATCH", "IP_TYPE", "IP_RISK", "DEVICE_RISK",
                                        "IP_DIST_KM", "REFERRAL_CODE", "STATED_INCOME", "EMPLOYER", "DEVICE_ID", "IP_ADDR"]),
    "KYC_CHECKS": ("CHECK_TS", ["RESULT", "ID_MATCH", "DOC_SCORE", "EMAIL_AGE_DAYS", "PHONE_RISK", "PHONE_LINE_TYPE",
                                "PHONE_PREPAID", "ADDRESS_TYPE", "EMAIL_NAME_MATCH", "FACE_MATCH", "WATCHLIST_HIT"]),
    "LOGINS": ("LOGIN_TS", ["RESULT", "CHANNEL", "AUTH_METHOD", "IP_TYPE", "IP_COUNTRY", "IP_REGION", "IP_RISK",
                            "IP_DIST_KM", "DEVICE_RISK", "SIM_CHANGE_72H", "REMOTE_SESSION", "DEVICE_ID", "IP_ADDR"]),
    "PROFILE_CHANGES": ("CHANGE_TS", ["FIELD", "CHANNEL", "RESET_METHOD", "NEW_EMAIL", "NEW_EMAIL_AGE_DAYS",
                                      "NEW_PHONE", "NEW_PHONE_LINE_TYPE", "IP_DIST_KM", "DEVICE_ID", "IP_ADDR"]),
    "PAYEES": ("ADDED_TS", ["PAYEE_KIND", "PAYEE_NAME", "PAYEE_HANDLE", "PAYEE_ROUTING", "PAYEE_ACCT_NO", "DEVICE_ID"]),
    "P2P": ("TXN_TS", ["DIRECTION", "AMT", "STATUS", "CP_NAME", "CP_HANDLE", "MEMO", "NOTE", "IP_RISK", "DEVICE_ID"]),
    "ACH": ("EFFECTIVE_TS", ["DIRECTION", "AMT", "STATUS", "ENTRY_DESC", "CP_NAME", "CP_COMPANY_ID", "CP_ROUTING",
                             "CP_ACCT_NO", "SEC_CODE"]),
    "ACH_RETURNS": ("RETURN_TS", ["RETURN_CODE", "ENTRY_KIND", "AMT", "CP_NAME"]),
    "CARD_AUTHS": ("AUTH_TS", ["AMT", "RESP_CD", "MERCHANT_NAME", "MERCHANT_ID", "MCC_CAT", "CHANNEL", "POS_REGION",
                               "IP_RISK", "DEVICE_ID"]),
    "CARD_TOKENS": ("TOKEN_TS", ["METHOD", "CARD_ID", "DEVICE_ID"]),
    "CARD_CREDITS": ("POST_TS", ["TYPE", "AMT", "MERCHANT_NAME"]),
    "DISPUTES": ("LOAD_TS", ["TYPE", "REASON", "AMT", "MERCHANT_NAME", "TXN_TS"]),
    "CLAIMS": ("CLAIM_TS", ["KIND", "AMT", "DECISION", "STATEMENT"]),
    "CALLS": ("CALL_TS", ["CHANNEL", "REASON", "DURATION_S", "NOTES"]),
    "CASH": ("TXN_TS", ["TYPE", "AMT", "CHANNEL", "BRANCH", "STATUS"]),
    "CHECKS": ("TXN_TS", ["TYPE", "AMT", "CHECK_KIND", "CP_NAME", "CHANNEL", "STATUS", "RETURN_REASON"]),
    "CONSUMER_WIRES": ("WIRE_TS", ["AMT", "STATUS", "BENEFICIARY_NAME", "BENEFICIARY_BANK", "INTERNATIONAL_FLG",
                                   "PURPOSE", "NOTE", "CHANNEL"]),
    "INTERNAL_TRANSFERS": ("TXN_TS", ["AMT", "FROM_ACCT", "TO_ACCT", "PURPOSE", "CHANNEL", "STATUS"]),
    "REFERRALS": ("PAID_TS", ["ROLE", "OTHER_CUST_ID", "REWARD_USD"]),
    "BONUSES": ("EVENT_TS", ["STAGE", "KIND", "BONUS_AMT", "REQUIRED_DEPOSITS", "QUALIFYING_DEPOSITS"]),
    "FRAUD_TAGS": ("TAG_TS", ["SOURCE", "REPORTED_TYPE", "RECORD_TYPE", "AMT"]),
    "REVIEW_QUEUE": ("OPENED_TS", ["CASE_TYPE", "SOURCE", "AMT", "COUNTERPARTY", "NOTE"]),
    "ALERTS": ("EVENT_TS", ["TYPE", "REASON"]),
    "ACCOUNT_PARTIES": ("ADDED_TS", ["ROLE", "RELATIONSHIP", "ADDED_CUST_ID", "CHANNEL", "NOTE"]),
}
LINK_KINDS = ["device", "ip", "phone", "email", "address", "ach_originator", "payee", "merchant", "referral",
              "p2p_sender"]


def fmt(v):
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        return str(v)[:19]
    return str(v)


def print_rows(cur, limit=200):
    print("\t".join(d[0] for d in cur.description or []))
    rows = cur.fetchmany(limit + 1)
    for row in rows[:limit]:
        print("\t".join(fmt(v) for v in row))
    if len(rows) > limit:
        print(f"... (more than {limit} rows: aggregate or add LIMIT)")


class Case:
    def __init__(self, work):
        self.work = Path(work)
        self.case = json.loads((self.work / "case.json").read_text())
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from .casedb import open_asof
            bank = (self.work / self.case["bank"]).resolve()
            self._db = open_asof(bank, self.case["case_time"], lock=False)
            from .links import build_keys
            if self._needs_keys:
                build_keys(self._db)
            self._db.execute("SET enable_external_access = false")
            self._db.execute("SET lock_configuration = true")
        return self._db

    _needs_keys = False


def cmd_tables(c):
    db = c.db
    for (name,) in db.execute("SELECT table_name FROM duckdb_tables() WHERE NOT internal AND table_name NOT IN "
                              "('custkeys', 'keysize') ORDER BY 1").fetchall():
        cols = db.execute(f'DESCRIBE "{name}"').fetchall()
        n = db.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        print(f"{name} ({n} rows): {', '.join(col[0] + ' ' + col[1].split('(')[0].lower() for col in cols)}")


def cmd_customer(c, cust, show_all=False):
    db = c.db
    prof = db.execute("SELECT * FROM CUSTOMERS WHERE CUST_ID = ?", [cust])
    rows = prof.fetchall()
    if not rows:
        sys.exit(f"no customer {cust} as of the case time")
    cols = [d[0] for d in prof.description]
    print("## Profile (current record)")
    for k, v in zip(cols, rows[0]):
        if k != "LOAD_TS":
            print(f"  {k}: {fmt(v)}")
    print(f"  FIRST_SEEN (LOAD_TS): {fmt(rows[0][cols.index('LOAD_TS')])}")
    for t, q in (("Accounts", "SELECT ACCT_ID, PRODUCT, TYPE, BUSINESS_NAME, CREDIT_LIMIT, LOAD_TS AS OPENED_OR_SEEN FROM ACCOUNTS WHERE CUST_ID = ?"),
                 ("Cards", "SELECT CARD_ID, ACCT_ID, CARD_TYPE, NETWORK, LOAD_TS AS SEEN FROM CARDS WHERE CUST_ID = ?")):
        print(f"## {t}")
        cur = db.execute(q, [cust])
        print_rows(cur, 50)
    events, counts = [], {}
    tables = {r[0] for r in db.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    for t, (tcol, cols) in TIMELINE.items():
        if t not in tables:
            continue
        have = {r[0] for r in db.execute(f'DESCRIBE "{t}"').fetchall()}
        cols = [x for x in cols if x in have]
        if "CUST_ID" not in have or tcol not in have:
            continue
        for row in db.execute(f'SELECT {tcol}, {", ".join(cols)} FROM "{t}" WHERE CUST_ID = ?', [cust]).fetchall():
            parts = [f"{k}={fmt(v)}" for k, v in zip(cols, row[1:]) if v is not None and fmt(v) != ""]
            events.append((row[0], t, " ".join(parts)))
            counts[t] = counts.get(t, 0) + 1
    events.sort(key=lambda e: (e[0] is None, e[0]))
    print(f"## Timeline: {len(events)} events (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")")
    shown = events if show_all else events[-150:]
    if len(shown) < len(events):
        print(f"  (showing the latest {len(shown)}; `casefile customer {cust} --all` for all, or `casefile query`)")
    for ts, t, s in shown:
        print(f"  {fmt(ts)}  {t:<18} {s}")


def cmd_links(c, cust, kind):
    if kind not in LINK_KINDS:
        sys.exit(f"--by must be one of {', '.join(LINK_KINDS)}")
    c._needs_keys = True
    from .links import links_of
    res = links_of(c.db, cust, kind)
    if not res:
        print(f"{cust} has no {kind} keys as of the case time")
        return
    for k, size, others in res:
        print(f"{kind} {k}: shared by {size} customer(s) in total"
              + ("" if others else " (only this customer)"))
        for o, first, last in others:
            print(f"    {o}  first {fmt(first)}  last {fmt(last)}")
        if size - 1 > len(others):
            print(f"    ... and {size - 1 - len(others)} more")


def validate(case, ans):
    errs = []
    if not isinstance(ans, dict):
        return ["the answer must be a JSON object"]
    if case["task"] == "alert":
        if ans.get("disposition") not in DISPOSITIONS:
            errs.append(f"disposition must be one of {DISPOSITIONS}")
        if ans.get("typology") not in TYPOLOGIES:
            errs.append(f"typology must be one of {TYPOLOGIES}")
        if ans.get("customer_role") not in ROLES:
            errs.append(f"customer_role must be one of {ROLES}")
        conf = ans.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            errs.append("confidence must be a number from 0 to 1")
        if ans.get("recommended_action") not in ALERT_ACTIONS:
            errs.append(f"recommended_action must be one of {ALERT_ACTIONS}")
        if not isinstance(ans.get("rationale"), str) or not ans.get("rationale").strip():
            errs.append("rationale must be a non-empty string")
        if ans.get("disposition") == "not_fraud" and ans.get("typology") not in ("none", None):
            errs.append("a not_fraud answer has typology none")
    else:
        if not isinstance(ans.get("is_incident"), bool):
            errs.append("is_incident must be true or false")
        if ans.get("typology") not in TYPOLOGIES:
            errs.append(f"typology must be one of {TYPOLOGIES}")
        mem = ans.get("members")
        if not isinstance(mem, list):
            errs.append("members must be a list of {cust_id, role}")
        else:
            for m in mem:
                if not isinstance(m, dict) or not isinstance(m.get("cust_id"), str) or m.get("role") not in ROLES[:3]:
                    errs.append(f"bad member {m!r}: needs cust_id and role in {ROLES[:3]}")
                    break
        if not isinstance(ans.get("link_keys", []), list):
            errs.append("link_keys must be a list of strings")
        if not isinstance(ans.get("summary"), str):
            errs.append("summary must be a string")
        acts = ans.get("recommended_actions", [])
        if not isinstance(acts, list) or any(a not in INCIDENT_ACTIONS for a in acts):
            errs.append(f"recommended_actions must be a list from {INCIDENT_ACTIONS}")
    return errs


def main(work, args):
    work = Path(work)
    c = Case(work)
    started = time.time()
    out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": args}
    try:
        run(c, args)
        out["ok"] = True
    except SystemExit as e:
        out["ok"] = e.code in (0, None)
        out["error"] = str(e.code)[:300] if e.code not in (0, None) else None
        raise
    finally:
        out["seconds"] = round(time.time() - started, 2)
        with open(work / "log.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")


def run(c, args):
    if not args or args[0] in ("help", "-h", "--help"):
        return print(USAGE)
    cmd = args[0]
    if cmd == "brief":
        print((c.work / "brief.md").read_text())
    elif cmd == "tables":
        cmd_tables(c)
    elif cmd == "query":
        if len(args) != 2:
            sys.exit('usage: casefile query "SQL"')
        try:
            cur = c.db.execute(args[1])
        except Exception as error:
            sys.exit(f"SQL error: {str(error)[:600]}")
        print_rows(cur)
    elif cmd == "customer":
        if len(args) < 2:
            sys.exit("usage: casefile customer CUST_ID [--all]")
        cmd_customer(c, args[1], "--all" in args[2:])
    elif cmd == "links":
        if len(args) != 4 or args[2] != "--by":
            sys.exit("usage: casefile links CUST_ID --by KIND")
        cmd_links(c, args[1], args[3])
    elif cmd == "note":
        if len(args) != 2:
            sys.exit('usage: casefile note "TEXT"')
        with open(c.work / "notes.md", "a") as f:
            f.write(args[1].rstrip() + "\n")
        print("noted")
    elif cmd == "notes":
        print((c.work / "notes.md").read_text() or "(no notes yet)")
    elif cmd == "submit":
        if len(args) != 2:
            sys.exit("usage: casefile submit FILE")
        target = (c.work / args[1]).resolve() if not Path(args[1]).is_absolute() else Path(args[1]).resolve()
        if c.work.resolve() not in target.parents:
            sys.exit("the answer file must be inside your workspace (e.g. drafts/answer.json)")
        try:
            ans = json.loads(target.read_text())
        except Exception as error:
            sys.exit(f"answer NOT submitted: not valid JSON ({error})")
        errs = validate(c.case, ans)
        if errs:
            sys.exit("answer NOT submitted:\n" + "\n".join(f"  - {e}" for e in errs))
        (c.work / "answer.json").write_text(json.dumps(ans, indent=1))
        print("answer submitted (you may resubmit; the last valid answer counts)")
    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
