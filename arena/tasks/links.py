"""Link keys between customers, from the bank's own tables: which customers share a device, IP, phone, email,
address, ACH originator, payee, merchant or referral. Used by `case links`, by the case builder to find seeds, and by
the baseline. Works on any DuckDB connection whose views are the bank's tables (as of a case time, or the whole run).
"""

KINDS = ["device", "ip", "phone", "email", "address", "ach_originator", "payee", "merchant", "referral", "p2p_sender"]

# kind -> SQL selecting (CUST_ID, key, ts). Keys are normalised to make equal things equal (lower-case emails, the
# originator's company id or name, a payee's account or handle).
KEY_SQL = {
    "device": """SELECT CUST_ID, DEVICE_ID k, LOGIN_TS ts FROM LOGINS WHERE DEVICE_ID IS NOT NULL
                 UNION ALL SELECT CUST_ID, DEVICE_ID, APP_TS FROM ACCOUNT_APPLICATIONS WHERE DEVICE_ID IS NOT NULL
                 UNION ALL SELECT CUST_ID, DEVICE_ID, TOKEN_TS FROM CARD_TOKENS WHERE DEVICE_ID IS NOT NULL
                 UNION ALL SELECT CUST_ID, DEVICE_ID, ADDED_TS FROM PAYEES WHERE DEVICE_ID IS NOT NULL
                 UNION ALL SELECT CUST_ID, DEVICE_ID, CHANGE_TS FROM PROFILE_CHANGES WHERE DEVICE_ID IS NOT NULL""",
    "ip": """SELECT CUST_ID, IP_ADDR k, LOGIN_TS ts FROM LOGINS WHERE IP_ADDR IS NOT NULL
             UNION ALL SELECT CUST_ID, IP_ADDR, APP_TS FROM ACCOUNT_APPLICATIONS WHERE IP_ADDR IS NOT NULL
             UNION ALL SELECT CUST_ID, IP_ADDR, ADDED_TS FROM PAYEES WHERE IP_ADDR IS NOT NULL
             UNION ALL SELECT CUST_ID, IP_ADDR, CHANGE_TS FROM PROFILE_CHANGES WHERE IP_ADDR IS NOT NULL""",
    "phone": """SELECT CUST_ID, PHONE k, LOAD_TS ts FROM CUSTOMERS WHERE PHONE IS NOT NULL
                UNION ALL SELECT CUST_ID, NEW_PHONE, CHANGE_TS FROM PROFILE_CHANGES WHERE NEW_PHONE IS NOT NULL""",
    "email": """SELECT CUST_ID, lower(EMAIL) k, LOAD_TS ts FROM CUSTOMERS WHERE EMAIL IS NOT NULL
                UNION ALL SELECT CUST_ID, lower(NEW_EMAIL), CHANGE_TS FROM PROFILE_CHANGES WHERE NEW_EMAIL IS NOT NULL""",
    "address": """SELECT CUST_ID, upper(ADDRESS) || ', ' || coalesce(ZIP, '') k, LOAD_TS ts FROM CUSTOMERS
                  WHERE ADDRESS IS NOT NULL""",
    "ach_originator": """SELECT CUST_ID, coalesce(CP_COMPANY_ID, CP_NAME) k, EFFECTIVE_TS ts FROM ACH
                         WHERE DIRECTION = 'CREDIT' AND coalesce(CP_COMPANY_ID, CP_NAME) IS NOT NULL""",
    "payee": """SELECT CUST_ID, lower(CP_HANDLE) k, TXN_TS ts FROM P2P WHERE DIRECTION = 'OUT' AND CP_HANDLE IS NOT NULL
                UNION ALL SELECT CUST_ID, coalesce(PAYEE_ROUTING || ':' || PAYEE_ACCT_NO, lower(PAYEE_HANDLE)), ADDED_TS
                          FROM PAYEES WHERE coalesce(PAYEE_ACCT_NO, PAYEE_HANDLE) IS NOT NULL
                UNION ALL SELECT CUST_ID, BENEFICIARY_ROUTING || ':' || BENEFICIARY_ACCT_NO, WIRE_TS FROM CONSUMER_WIRES
                          WHERE BENEFICIARY_ACCT_NO IS NOT NULL
                UNION ALL SELECT CUST_ID, CP_ROUTING || ':' || CP_ACCT_NO, EFFECTIVE_TS FROM ACH
                          WHERE DIRECTION = 'DEBIT' AND CP_ACCT_NO IS NOT NULL AND ENTRY_DESC = 'TRANSFER'""",
    "merchant": """SELECT CUST_ID, MERCHANT_ID k, AUTH_TS ts FROM CARD_AUTHS WHERE MERCHANT_ID IS NOT NULL""",
    "referral": """SELECT CUST_ID, OTHER_CUST_ID k, PAID_TS ts FROM REFERRALS WHERE OTHER_CUST_ID IS NOT NULL
                   UNION ALL SELECT OTHER_CUST_ID, CUST_ID, PAID_TS FROM REFERRALS WHERE OTHER_CUST_ID IS NOT NULL""",
    "p2p_sender": """SELECT CUST_ID, lower(CP_HANDLE) k, TXN_TS ts FROM P2P WHERE DIRECTION = 'IN' AND CP_HANDLE IS NOT NULL""",
}


def build_keys(db, kinds=KINDS):
    """Create table custkeys(kind, k, CUST_ID, first_ts, last_ts, n) on db."""
    parts = [f"SELECT '{kind}' kind, k, CUST_ID, min(ts) first_ts, max(ts) last_ts, count(*) n FROM ({KEY_SQL[kind]}) "
             f"WHERE k IS NOT NULL AND CUST_ID IS NOT NULL GROUP BY ALL" for kind in kinds]
    db.execute("CREATE OR REPLACE TABLE custkeys AS " + " UNION ALL ".join(parts))
    db.execute("""CREATE OR REPLACE TABLE keysize AS SELECT kind, k, count(DISTINCT CUST_ID) customers
                  FROM custkeys GROUP BY ALL""")
    # a referral key is the other customer; make the referrer itself a holder of its own key, so 'referral' groups
    # the referrer with everyone it referred
    return db


def links_of(db, cust, kind, limit=60):
    """[(key, customers sharing it overall, [(other cust, first_ts, last_ts)])] for one customer and key kind."""
    out = []
    keys = db.execute("""SELECT c.k, s.customers FROM custkeys c JOIN keysize s USING (kind, k)
                         WHERE c.kind = ? AND c.CUST_ID = ? ORDER BY s.customers, c.k""", [kind, cust]).fetchall()
    for k, size in keys:
        others = db.execute("""SELECT CUST_ID, first_ts, last_ts FROM custkeys WHERE kind = ? AND k = ? AND CUST_ID <> ?
                               ORDER BY first_ts LIMIT ?""", [kind, k, cust, limit]).fetchall()
        out.append((k, size, others))
    return out
