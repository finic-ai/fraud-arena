"""The bank's alert monitors (task A) and anomaly detectors (task B seeds), on bank-visible data only.

Every monitor reads bank/ (Brindlemark's own tables) and nothing from truth/. An alert's time is the LOAD_TS of the
row that fired it (when the bank could first have seen it). The monitors are deliberately simple and noisy, as a
bank's first-line rules are: they fire on fraud and on its look-alikes alike, and the case builder then samples them.
"""

import glob
from pathlib import Path

import duckdb


def bank_db(out):
    db = duckdb.connect()
    for f in sorted(glob.glob(str(Path(out) / "bank" / "*.parquet"))):
        db.execute(f'CREATE VIEW "{Path(f).stem}" AS SELECT * FROM read_parquet(\'{f}\')')
    return db


# Each monitor: (name, description, SQL returning CUST_ID, ts, reason, ref). The reason is the alert text the analyst
# gets; ref is the triggering row's id. Thresholds are stated in the description shown in case briefs.
MONITORS = [
    ("card_fraud_dispute", "a cardholder disputed card transactions as fraud",
     """SELECT CUST_ID, LOAD_TS ts,
               'Cardholder disputed a $' || CAST(AMT AS DECIMAL(12,2)) || ' card transaction at ' || coalesce(MERCHANT_NAME, '?')
               || ' as fraud' reason, DISPUTE_ID ref
        FROM DISPUTES WHERE TYPE = 'OPENED' AND REASON = 'FRAUD'"""),
    ("card_nonreceipt_dispute", "a cardholder disputed a card transaction as goods/services not received",
     """SELECT CUST_ID, LOAD_TS ts,
               'Cardholder disputed a $' || CAST(AMT AS DECIMAL(12,2)) || ' charge at ' || coalesce(MERCHANT_NAME, '?')
               || ' as ' || lower(replace(REASON, '_', ' ')) reason, DISPUTE_ID ref
        FROM DISPUTES WHERE TYPE = 'OPENED' AND REASON <> 'FRAUD'"""),
    ("p2p_unauthorized_claim", "the customer filed a claim for unauthorized or misdirected P2P payments",
     """SELECT CUST_ID, LOAD_TS ts,
               'Customer filed a ' || lower(replace(KIND, '_', ' ')) || ' claim for $' || CAST(AMT AS DECIMAL(12,2)) reason,
               CLAIM_ID ref
        FROM CLAIMS"""),
    ("ach_return", "an ACH entry on the customer's account came back returned",
     """SELECT CUST_ID, LOAD_TS ts,
               'ACH ' || lower(ENTRY_KIND) || ' of $' || CAST(AMT AS DECIMAL(12,2)) || ' from/to ' || coalesce(CP_NAME, '?')
               || ' returned ' || RETURN_CODE reason, RETURN_ID ref
        FROM ACH_RETURNS"""),
    ("ach_push_callback", "an outbound ACH transfer was held for a callback by the bank's payment rules",
     """SELECT CUST_ID, LOAD_TS ts,
               'Outbound ACH of $' || CAST(AMT AS DECIMAL(12,2)) || ' to ' || coalesce(CP_NAME, '?') || ' held for callback' reason,
               ACH_ID ref
        FROM ACH WHERE STATUS = 'CALLBACK'"""),
    ("p2p_hold", "an outbound P2P payment was held by the bank's payment rules",
     """SELECT CUST_ID, LOAD_TS ts,
               'P2P payment of $' || CAST(AMT AS DECIMAL(12,2)) || ' to ' || coalesce(CP_NAME, CP_HANDLE, '?') || ' held' reason,
               P2P_ID ref
        FROM P2P WHERE STATUS = 'HELD'"""),
    ("wire_callback", "a consumer wire was held for callback, or is large (>= $5,000)",
     """SELECT CUST_ID, LOAD_TS ts,
               'Outbound wire of $' || CAST(AMT AS DECIMAL(12,2)) || ' to ' || coalesce(BENEFICIARY_NAME, '?')
               || CASE WHEN STATUS = 'CALLBACK' THEN ' held for callback' ELSE ' (large wire)' END reason, WIRE_ID ref
        FROM CONSUMER_WIRES WHERE STATUS = 'CALLBACK' OR AMT >= 5000"""),
    ("card_step_up", "a card authorization was stepped up or declined by the bank's card rules",
     """SELECT CUST_ID, LOAD_TS ts,
               'Card authorization of $' || CAST(AMT AS DECIMAL(12,2)) || ' at ' || coalesce(MERCHANT_NAME, '?') || ' ('
               || coalesce(CHANNEL, '?') || ') ' || CASE WHEN RESP_CD = '1A' THEN 'stepped up' ELSE 'declined by rule' END reason,
               AUTH_ID ref
        FROM CARD_AUTHS WHERE RESP_CD IN ('1A', '05')"""),
    ("referral_velocity", "a new customer (opened <= 30 days) in a referral chain whose referrer was paid 3+ rewards "
                          "within 7 days",
     """WITH r AS (SELECT * FROM REFERRALS),
             burst AS (SELECT a.CUST_ID referrer, a.PAID_TS ts, count(*) n FROM r a JOIN r b
                       ON b.CUST_ID = a.CUST_ID AND b.ROLE = 'REFERRER' AND b.PAID_TS BETWEEN a.PAID_TS - INTERVAL 7 DAY AND a.PAID_TS
                       WHERE a.ROLE = 'REFERRER' GROUP BY ALL HAVING count(*) >= 3),
             opened AS (SELECT CUST_ID, min(APP_TS) opened FROM ACCOUNT_APPLICATIONS GROUP BY 1)
        SELECT r.CUST_ID, max(r.LOAD_TS) ts,
               'Referral reward paid; the referral chain''s referrer earned ' || max(burst.n) || ' rewards within 7 days' reason,
               min(r.REFERRAL_ID) ref
        FROM r JOIN burst ON burst.referrer = CASE WHEN r.ROLE = 'REFERRER' THEN r.CUST_ID ELSE r.OTHER_CUST_ID END
                         AND r.PAID_TS = burst.ts
        JOIN opened o ON o.CUST_ID = r.CUST_ID AND r.PAID_TS <= o.opened + INTERVAL 30 DAY
        GROUP BY r.CUST_ID, r.REFERRAL_ID"""),
    ("new_device_contact_change", "contact details or password changed from a device the customer had not used before",
     """WITH first_dev AS (SELECT CUST_ID, DEVICE_ID, min(LOGIN_TS) first_ts FROM LOGINS GROUP BY ALL)
        SELECT p.CUST_ID, p.LOAD_TS ts,
               lower(p.FIELD) || ' change via ' || coalesce(p.CHANNEL, p.RESET_METHOD, '?') || ' from a device first seen '
               || CAST(round(epoch(p.CHANGE_TS - f.first_ts) / 3600, 1) AS VARCHAR) || 'h earlier' reason, p.CHANGE_ID ref
        FROM PROFILE_CHANGES p JOIN first_dev f ON f.CUST_ID = p.CUST_ID AND f.DEVICE_ID = p.DEVICE_ID
        WHERE p.CHANGE_TS - f.first_ts <= INTERVAL 2 DAY
          AND EXISTS (SELECT 1 FROM LOGINS l WHERE l.CUST_ID = p.CUST_ID AND l.LOGIN_TS < f.first_ts - INTERVAL 7 DAY)"""),
    ("risky_login", "a successful login from a hosting/VPN/proxy IP, with a SIM change in the last 72h, or in a remote "
                    "session, for a customer who had not logged in that way before",
     """WITH l AS (SELECT *, row_number() OVER (PARTITION BY CUST_ID ORDER BY LOGIN_TS) k FROM LOGINS
                   WHERE RESULT = 'OK' AND (IP_TYPE IN ('HOSTING', 'VPN', 'PROXY') OR SIM_CHANGE_72H = 'Y' OR REMOTE_SESSION = 'Y'))
        SELECT CUST_ID, LOAD_TS ts,
               'Login from ' || IP_TYPE || ' IP (' || coalesce(IP_COUNTRY, '?') || ', risk ' || CAST(IP_RISK AS VARCHAR) || ')'
               || CASE WHEN SIM_CHANGE_72H = 'Y' THEN ', SIM changed in last 72h' ELSE '' END
               || CASE WHEN REMOTE_SESSION = 'Y' THEN ', remote-access session' ELSE '' END reason, LOGIN_ID ref
        FROM l WHERE k = 1"""),
    ("application_referred", "an account application was referred by onboarding checks or scored high IP/device risk",
     """SELECT CUST_ID, LOAD_TS ts,
               'Account application ' || lower(DECISION) || ' (ID match ' || CAST(ID_MATCH AS VARCHAR) || ', IP risk '
               || CAST(IP_RISK AS VARCHAR) || ', device risk ' || CAST(DEVICE_RISK AS VARCHAR) || ', IP ' || coalesce(IP_TYPE, '?') || ')'
               reason, APP_ID ref
        FROM ACCOUNT_APPLICATIONS WHERE DECISION <> 'APPROVED' OR IP_RISK >= 0.7 OR DEVICE_RISK >= 0.7"""),
    ("p2p_pass_through", "5+ incoming P2P payments from 3+ senders in 7 days, most of it sent on within 2 days",
     """WITH inn AS (SELECT CUST_ID, date_trunc('day', TXN_TS) d, LOAD_TS, AMT, CP_HANDLE FROM P2P WHERE DIRECTION = 'IN'),
             w AS (SELECT a.CUST_ID, a.d, max(b.LOAD_TS) ts, count(*) n, count(DISTINCT b.CP_HANDLE) senders, sum(b.AMT) amt
                   FROM (SELECT DISTINCT CUST_ID, d FROM inn) a JOIN inn b ON b.CUST_ID = a.CUST_ID
                    AND b.d BETWEEN a.d - INTERVAL 6 DAY AND a.d GROUP BY ALL HAVING count(*) >= 5 AND count(DISTINCT b.CP_HANDLE) >= 3),
             o AS (SELECT w.CUST_ID, w.d, sum(p.AMT) out_amt FROM w JOIN P2P p ON p.CUST_ID = w.CUST_ID AND p.DIRECTION = 'OUT'
                    AND p.TXN_TS BETWEEN w.d - INTERVAL 6 DAY AND w.d + INTERVAL 2 DAY GROUP BY ALL)
        SELECT w.CUST_ID, w.ts, n || ' incoming P2P payments ($' || CAST(w.amt AS DECIMAL(12,2)) || ') from ' || senders
               || ' senders in 7 days; $' || CAST(o.out_amt AS DECIMAL(12,2)) || ' sent on' reason, w.CUST_ID || '-' || CAST(w.d AS DATE) ref
        FROM w JOIN o USING (CUST_ID, d) WHERE o.out_amt >= 0.7 * w.amt"""),
    ("cash_near_threshold", "cash deposits between $8,000 and $10,000",
     """SELECT CUST_ID, LOAD_TS ts, 'Cash ' || lower(TYPE) || ' of $' || CAST(AMT AS DECIMAL(12,2)) reason, CASH_ID ref
        FROM CASH WHERE AMT BETWEEN 8000 AND 9999.99 AND TYPE ILIKE '%DEPOSIT%'"""),
    ("fraud_report_call", "the customer called to report fraud or a suspicious contact",
     """SELECT CUST_ID, LOAD_TS ts, 'Customer call: ' || lower(replace(REASON, '_', ' ')) reason, CALL_ID ref
        FROM CALLS WHERE REASON IN ('FRAUD_REPORT', 'SUSPICIOUS_CONTACT')"""),
]
MONITOR_DESCRIPTIONS = {m[0]: m[1] for m in MONITORS}


def run_monitors(out, names=None):
    """All alerts over the whole run: dicts with monitor, cust, ts, reason, ref (deduplicated per monitor, customer
    and day)."""
    db = bank_db(out)
    alerts = []
    for name, _, sql in MONITORS:
        if names and name not in names:
            continue
        try:
            rows = db.execute(f"SELECT * FROM ({sql}) WHERE CUST_ID IS NOT NULL AND ts IS NOT NULL ORDER BY ts, CUST_ID, ref, reason").fetchall()
        except duckdb.Error as error:
            raise RuntimeError(f"monitor {name}: {error}") from error
        seen = set()
        for cust, ts, reason, ref in rows:
            k = (cust, ts.date())
            if k in seen:
                continue
            seen.add(k)
            alerts.append({"monitor": name, "cust": cust, "ts": ts, "reason": reason, "ref": str(ref)})
    return alerts
