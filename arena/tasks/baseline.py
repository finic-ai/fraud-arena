"""Non-LLM heuristic baselines. They see exactly what a player sees: the case file and the bank's tables as of the
case time (casedb.open_asof); never truth. Hand-set points, no fitting on the cases."""

import json
import math
from pathlib import Path

from .casedb import open_asof
from .links import build_keys

SIGNALS = """
WITH c AS (SELECT CAST(? AS VARCHAR) cust, CAST(? AS TIMESTAMPTZ) T)
SELECT
  (SELECT date_diff('day', min(LOAD_TS), (SELECT T FROM c)) FROM ACCOUNTS WHERE CUST_ID = (SELECT cust FROM c)) acct_age,
  (SELECT max(IP_RISK) FROM ACCOUNT_APPLICATIONS WHERE CUST_ID = (SELECT cust FROM c)) app_ip_risk,
  (SELECT min(ID_MATCH) FROM ACCOUNT_APPLICATIONS WHERE CUST_ID = (SELECT cust FROM c)) app_id_match,
  (SELECT count(*) FROM REFERRALS WHERE CUST_ID = (SELECT cust FROM c)) referrals,
  (SELECT count(*) FROM BONUSES WHERE CUST_ID = (SELECT cust FROM c) AND STAGE = 'PAID') bonuses,
  (SELECT count(*) FROM LOGINS WHERE CUST_ID = (SELECT cust FROM c) AND IP_TYPE IN ('HOSTING','VPN','PROXY')) risky_logins,
  (SELECT count(*) FROM LOGINS WHERE CUST_ID = (SELECT cust FROM c) AND REMOTE_SESSION = 'Y') remote,
  (SELECT count(*) FROM LOGINS WHERE CUST_ID = (SELECT cust FROM c) AND SIM_CHANGE_72H = 'Y') sim,
  (SELECT count(*) FROM PROFILE_CHANGES WHERE CUST_ID = (SELECT cust FROM c)
      AND CHANGE_TS >= (SELECT T FROM c) - INTERVAL 30 DAY) profile_changes,
  (SELECT count(*) FROM DISPUTES WHERE CUST_ID = (SELECT cust FROM c) AND TYPE = 'OPENED' AND REASON = 'FRAUD') fraud_disputes,
  (SELECT count(*) FROM DISPUTES WHERE CUST_ID = (SELECT cust FROM c) AND TYPE = 'OPENED' AND REASON <> 'FRAUD') other_disputes,
  (SELECT count(*) FROM CLAIMS WHERE CUST_ID = (SELECT cust FROM c)) claims,
  (SELECT count(*) FROM ACH_RETURNS WHERE CUST_ID = (SELECT cust FROM c) AND RETURN_CODE IN ('R10','R05','R07','R29')) unauth_returns,
  (SELECT count(*) FROM ACH_RETURNS WHERE CUST_ID = (SELECT cust FROM c)) returns,
  (SELECT count(DISTINCT CP_HANDLE) FROM P2P WHERE CUST_ID = (SELECT cust FROM c) AND DIRECTION = 'IN'
      AND TXN_TS >= (SELECT T FROM c) - INTERVAL 14 DAY) p2p_senders_14d,
  (SELECT coalesce(max(AMT), 0) FROM CONSUMER_WIRES WHERE CUST_ID = (SELECT cust FROM c)) max_wire,
  (SELECT count(*) FROM CASH WHERE CUST_ID = (SELECT cust FROM c) AND AMT BETWEEN 8000 AND 9999.99) near_ctr,
  (SELECT count(*) FROM CALLS WHERE CUST_ID = (SELECT cust FROM c) AND REASON IN ('FRAUD_REPORT','SUSPICIOUS_CONTACT')) fraud_calls,
  (SELECT date_diff('year', DOB, (SELECT T FROM c)::DATE) FROM CUSTOMERS WHERE CUST_ID = (SELECT cust FROM c)) age,
  (SELECT count(*) FROM CARD_AUTHS WHERE CUST_ID = (SELECT cust FROM c) AND AMT < 2 AND AUTH_TS >= (SELECT T FROM c) - INTERVAL 14 DAY) tiny_auths
"""


def signals(db, cust, T):
    cur = db.execute(SIGNALS, [cust, str(T)])
    return dict(zip([d[0] for d in cur.description], cur.fetchone()))


def assess(s, monitor=None):
    """(probability of fraud, typology, role) from the signals."""
    z = -1.2
    new = s["acct_age"] is not None and s["acct_age"] <= 60
    typ, role = "none", "legitimate"
    pts = {}
    if new and (s["referrals"] or s["bonuses"]):
        pts["promo_abuse"] = 1.2 + 0.4 * min(s["referrals"], 3) + (0.8 if (s["app_ip_risk"] or 0) >= 0.5 else 0)
    if new and ((s["app_ip_risk"] or 0) >= 0.7 or (s["app_id_match"] is not None and s["app_id_match"] < 0.6)):
        pts["synthetic_identity"] = 1.5
    if s["fraud_disputes"] or s["tiny_auths"] >= 2:
        pts["card_compromise"] = 1.0 + 0.5 * min(s["fraud_disputes"], 3) + 0.5 * (s["tiny_auths"] >= 2)
    if s["claims"] or (s["profile_changes"] and (s["risky_logins"] or s["sim"])) or s["unauth_returns"]:
        pts["ato"] = 1.0 + 0.7 * bool(s["sim"]) + 0.5 * bool(s["unauth_returns"]) + 0.3 * bool(s["claims"])
    if (s["remote"] or s["max_wire"] >= 5000) and (s["age"] or 0) >= 60:
        pts["scam_victim"] = 1.4 + 0.6 * bool(s["remote"])
    if s["p2p_senders_14d"] >= 4 and not new:
        pts["mule"] = 1.0 + 0.2 * min(s["p2p_senders_14d"] - 4, 5)
    if s["near_ctr"]:
        pts["other"] = 1.5 + 0.3 * min(s["near_ctr"], 3)
    if s["other_disputes"] >= 2:
        pts["first_party"] = 0.8 + 0.4 * min(s["other_disputes"], 4)
    if pts:
        typ = max(pts, key=pts.get)
        z += pts[typ]
        z += 0.3 * (len(pts) - 1)
    p = 1 / (1 + math.exp(-z))
    if p < 0.5:
        return p, "none", "legitimate"
    role = {"promo_abuse": "perpetrator", "synthetic_identity": "perpetrator", "card_compromise": "victim",
            "ato": "victim", "scam_victim": "victim", "mule": "mule", "other": "perpetrator",
            "first_party": "perpetrator"}[typ]
    return p, typ, role


def alert_answer(ws):
    case = json.loads((Path(ws) / "case.json").read_text())
    db = open_asof((Path(ws) / case["bank"]).resolve(), case["case_time"])
    s = signals(db, case["customer"], case["case_time"])
    p, typ, role = assess(s, case["alert"]["type"])
    fraud = p >= 0.5
    act = ("restrict_account" if role in ("perpetrator", "mule") else "contact_customer") if fraud else "close_no_action"
    return {"disposition": "fraud" if fraud else "not_fraud", "typology": typ if fraud else "none",
            "customer_role": role if fraud else "legitimate", "confidence": round(p, 3), "recommended_action": act,
            "rationale": "heuristic baseline: " + ", ".join(f"{k}={v}" for k, v in s.items() if v not in (None, 0))}


STRONG = ["device", "phone", "referral", "payee", "ach_originator", "address", "ip", "p2p_sender"]


def incident_answer(ws, max_key_size=8):
    """Seed risk decides the verdict; expansion is one hop over rare strong keys from the seed, kept when the new
    customer also looks risky on its own or shares two keys with the seed."""
    case = json.loads((Path(ws) / "case.json").read_text())
    db = open_asof((Path(ws) / case["bank"]).resolve(), case["case_time"], lock=False)
    build_keys(db)
    T = case["case_time"]
    seed = case["seed_customers"]
    assessed = {c: assess(signals(db, c, T)) for c in seed}
    risky = [c for c, (p, _, _) in assessed.items() if p >= 0.5]
    an = case["anomaly"]
    kind = an["kind"]
    is_inc = len(risky) >= max(1, math.ceil(len(seed) / 2)) or (kind in ("referral", "ach_originator", "device")
                                                               and an.get("customers_with_key", 99) <= 15)
    if not is_inc:
        return {"is_incident": False, "typology": "none", "members": [], "link_keys": [f"{kind}:{an['key']}"],
                "summary": f"heuristic baseline: only {len(risky)} of {len(seed)} seed customers look risky",
                "recommended_actions": ["close_no_action"]}
    rows = db.execute(f"""SELECT b.CUST_ID, count(DISTINCT a.kind || a.k) shared, string_agg(DISTINCT a.kind, ',')
                          FROM custkeys a JOIN keysize s USING (kind, k) JOIN custkeys b ON b.kind = a.kind AND b.k = a.k
                          WHERE a.CUST_ID IN (SELECT unnest(?)) AND s.customers <= {max_key_size}
                            AND a.kind IN (SELECT unnest(?)) AND b.CUST_ID NOT IN (SELECT unnest(?))
                          GROUP BY 1""", [seed, STRONG, seed]).fetchall()
    if kind == "monitor":
        rows = []
    members = {}
    for c in seed:
        p, typ, role = assessed[c]
        members[c] = role if role != "legitimate" else None
    for c, shared, kinds in rows:
        p, typ, role = assess(signals(db, c, T))
        if p >= 0.5 or shared >= 2:
            members[c] = role if role != "legitimate" else None
    typs = [assessed[c][1] for c in seed if assessed[c][1] != "none"]
    typ = max(set(typs), key=typs.count) if typs else "other"
    default_role = {"card_compromise": "victim", "ato": "victim", "scam_victim": "victim", "mule": "mule"}.get(typ, "perpetrator")
    out = [{"cust_id": c, "role": r or default_role} for c, r in members.items()]
    return {"is_incident": True, "typology": typ, "members": out, "link_keys": [f"{kind}:{an['key']}"],
            "summary": f"heuristic baseline: {len(risky)} of {len(seed)} seed customers look risky; "
                       f"{len(out) - len(seed)} added over rare shared keys",
            "recommended_actions": ["restrict_account"]}
