"""The briefs players read (`case brief`)."""

from .schema import ALERT_ACTIONS, INCIDENT_ACTIONS, ROLES, TYPOLOGIES

TYPOLOGY_HELP = """Typologies (pick one):
  ato                 account takeover: someone else logged in / took over the customer's account (incl. pulls
                      from the customer's account via its stolen online-banking login)
  scam_victim         the customer was talked into sending money themselves (romance, investment, impersonation,
                      tech-support, grandparent scams; diverted payroll)
  mule                the customer's account receives and passes on other people's stolen money
  synthetic_identity  the account was opened in a stolen or made-up identity (bust-outs, ACH/ACATS pulls)
  first_party         the real customer commits the fraud (false claims or disputes, refund abuse, return fraud)
  card_compromise     the customer's card details were stolen and used (card testing, CNP fraud, vishing for card
                      details, wallet provisioning)
  promo_abuse         accounts farmed for sign-up / referral bonuses (often in stolen or synthetic identities)
  check_fraud         check kiting or counterfeit / altered checks
  other               any other fraud (e.g. structuring)
  none                no fraud"""

ROLE_HELP = """Customer roles:
  perpetrator   the account holder (or whoever opened the account in that identity) commits the fraud
  victim        the customer lost money or had their account / card misused by a third party, or was scammed
  mule          the account moves others' stolen money (witting or not)
  legitimate    nothing fraudulent involves this customer"""

TOOLS = """## Tools (only `case` commands work)
  case brief                 this brief
  case tables                the bank's tables and columns (as of the case time)
  case query "SQL"           read-only DuckDB SQL over the bank's tables, as of the case time (up to 200 rows)
  case customer CUST_ID      profile, accounts, cards and a timeline of the customer's activity
  case links CUST_ID --by device|ip|phone|email|address|ach_originator|payee|merchant|referral|p2p_sender
                             other customers sharing each key with this customer, and how many customers share it
  case note "TEXT"           keep a note
  case submit FILE           check and submit your answer (JSON); you may resubmit, the last valid one counts

You only see what the bank had loaded by the case time (every table has LOAD_TS). Account and card *status* columns
are not available (they would show later outcomes). Amounts are USD. Everything is simulated; no real people."""


def alert_brief(case):
    a = case["alert"]
    share = case.get("fraud_share", 0.45)
    return f"""# Alert review at Brindlemark Bank (case {case['case_id']})

You are a fraud analyst at Brindlemark, a digital bank in a simulated economy. One of the bank's monitors raised an
alert on customer **{case['customer']}**. Review the customer's data as of the case time and decide whether fraud is
involved, what kind, and what this customer's role is.

- Alert type: `{a['type']}` ({a['monitor']})
- Alert reason: {a['reason']}
- Alert time: {a['time']}  (reference {a['ref']})
- Case time (your data is as of): {case['case_time']}

Base rate: in this review set about {round(share * 100)}% of the alerts involve fraud (a sampling choice; real alert
queues are far more benign). Fraud here means the customer is a perpetrator, a victim or a mule of a fraud that had
already started by the case time; benign look-alikes come from the same monitors.

{TYPOLOGY_HELP}

{ROLE_HELP}

{TOOLS}

## Your answer
Write JSON with the Write tool to drafts/answer.json, then run `case submit drafts/answer.json`:
{{"disposition": "fraud" | "not_fraud",
  "typology": one of {TYPOLOGIES} ("none" when not_fraud),
  "customer_role": one of {ROLES} ("legitimate" when not_fraud),
  "confidence": probability 0-1 that this is fraud (calibrated: it is scored with the Brier score),
  "recommended_action": one of {ALERT_ACTIONS},
  "rationale": "2-4 sentences: the evidence that decided it"}}

Scoring: disposition accuracy and fraud F1, typology and role accuracy on fraud cases, Brier score of `confidence`,
and a cost score where a missed fraud costs 5 and a false alarm on a legitimate customer costs 1.
"""


def incident_brief(case):
    an = case["anomaly"]
    seed = ", ".join(case["seed_customers"])
    return f"""# Incident investigation at Brindlemark Bank (case {case['case_id']})

You are a fraud investigator at Brindlemark, a digital bank in a simulated economy. A detector flagged an anomaly:

- **{an['text']}**
- Seed customers: {seed}
- Window: {case['window_start']} to the case time {case['case_time']} (your data is as of the case time)

Decide whether this is a real fraud incident (customers touched by one coordinated actor, e.g. one fraud ring crew
running one kind of scheme, or one fraudster's run of victims) or a benign cluster (shared carrier or office IPs,
households, an employer's payroll, a merchant's service problem, a popular payee...). If it is real, find **all**
Brindlemark customers in it within the window, seed and beyond, and each one's role. Some links are decoys: a shared
key alone does not make a customer a member. Coarse keys (shared by many customers) link little.

{TYPOLOGY_HELP}

{ROLE_HELP}

{TOOLS}

## Your answer
Write JSON with the Write tool to drafts/answer.json, then run `case submit drafts/answer.json`:
{{"is_incident": true | false,
  "typology": one of {TYPOLOGIES} (the incident's main family; "none" if not an incident),
  "members": [{{"cust_id": "...", "role": one of {ROLES[:3]}}}, ...]   (seed members you confirm plus every
             customer you add; leave out seed customers you think are not part of it; [] if not an incident),
  "link_keys": ["kind:value", ...]   (the keys that tie the members together),
  "summary": "3-6 sentences: what happened, how the members are linked, what you excluded and why",
  "recommended_actions": a list from {INCIDENT_ACTIONS}}}

Scoring: the verdict, the typology, precision / recall / F1 of the members you add beyond the seed (members you
could not have reached from the seed through the bank's data are not counted against you), role accuracy on the
members you got right, and on benign clusters how few customers you wrongly name.
"""
