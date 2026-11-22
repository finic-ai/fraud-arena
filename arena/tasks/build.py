"""Build the alert-review (A) and incident (B) cases from one simulated world.

    cases/NAME/world.json         seed, roots, days
    cases/NAME/bank/              the bank's tables (world.materialize)
    cases/NAME/truth/             simulator truth, rows, keys/<case>.json answer keys     (never shown to players)
    cases/NAME/alert/<ID>/        one workspace per alert case: case.json, brief.md, drafts/
    cases/NAME/incident/<ID>/     one workspace per incident case
    cases/NAME/index.json         the cases (public part) and the build summary

Alert cases (A): monitor alerts on customers with fraud started and visible by the case time, and on customers the
world never touched at all (benign look-alikes from the same monitors), mixed to a stated fraud share.

Incident cases (B): a truth group is every Brindlemark customer one ring crew touched in one typology family (or, for
independent fraudsters, one scenario run by one fraudster) within a window. The seed is the members holding the
bank-visible key that best singles the group out (an IP, device, merchant, payee, originator, referrer...). False-
positive incidents are clusters of the same key kinds whose holders the world never touched (carrier NAT IPs, office
IPs, employers' payroll, a merchant's non-receipt disputes, households)."""

import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from .casedb import open_asof
from .links import build_keys
from .monitors import MONITOR_DESCRIPTIONS, run_monitors
from .truth import Truth

ALERT_DELAY = timedelta(hours=24)      # the analyst works the alert a day after it fired
MAX_KEY_SIZE = 40                      # keys shared by more customers than this are too coarse to seed or expand on
REACH_KEY_SIZE = 25                    # answerability: members must be reachable from the seed over keys this rare
REACH_KINDS = ["device", "ip", "phone", "email", "address", "ach_originator", "payee", "merchant", "referral",
               "p2p_sender"]
SEED_KINDS = ["device", "ip", "merchant", "payee", "ach_originator", "referral", "address", "p2p_sender"]

KIND_TEXT = {
    "device": "logged in or transacted from the same device {k}",
    "ip": "used the same IP address {k}",
    "merchant": "had card authorizations at merchant {k}",
    "payee": "paid the same payee {k}",
    "ach_originator": "received ACH credits from the same originator {k}",
    "referral": "were linked by referral rewards to customer {k}",
    "address": "share the address {k}",
    "p2p_sender": "received P2P payments from the same sender {k}",
    "phone": "share the phone number {k}",
    "email": "share the email {k}",
}


def iso(ts):
    return ts.isoformat() if ts is not None else None


# --- alert cases ------------------------------------------------------------------------------------------------------

def alert_candidates(out, truth):
    touched = truth.ever_touched()
    fraud, benign = [], []
    for a in run_monitors(out):
        T = a["ts"] + ALERT_DELAY
        r = truth.customer(a["cust"], as_of=T)
        if r is None:
            if a["cust"] not in touched:
                benign.append({**a, "case_time": T, "truth": None})
            continue
        if r["role"] in ("incidental", None):
            continue
        # answerability: some row of the customer's fraud must have reached the bank by the case time
        if not any(i["first_load"] <= T for i in r["involvements"] if i["role"] or i["part"] == "cover"):
            continue
        fraud.append({**a, "case_time": T, "truth": r})
    return fraud, benign


def sample_alerts(fraud, benign, n, fraud_share, rng):
    """Stratified: fraud alerts spread over typologies (then monitors), one case per customer; benign alerts drawn
    from the same monitors as the fraud ones (look-alikes), then from the rest."""
    n_fraud = round(n * fraud_share)
    by_typ = defaultdict(list)
    for a in fraud:
        by_typ[a["truth"]["typology"]].append(a)
    for v in by_typ.values():
        rng.shuffle(v)
        # one alert per customer within a typology, preferring monitors not yet used
    chosen, custs, monitors_used = [], set(), Counter()
    while len(chosen) < n_fraud and any(by_typ.values()):
        for typ in sorted(by_typ, key=lambda t: (len([c for c in chosen if c["truth"]["typology"] == t]), t)):
            pool = [a for a in by_typ[typ] if a["cust"] not in custs]
            if not pool:
                by_typ[typ] = []
                continue
            pool.sort(key=lambda a: monitors_used[a["monitor"]])
            a = pool[0]
            by_typ[typ].remove(a)
            chosen.append(a)
            custs.add(a["cust"])
            monitors_used[a["monitor"]] += 1
            break
    n_benign = n - len(chosen)
    by_mon = defaultdict(list)
    for a in benign:
        by_mon[a["monitor"]].append(a)
    for v in by_mon.values():
        rng.shuffle(v)
    picks = []
    target = [a["monitor"] for a in chosen]
    rng.shuffle(target)
    for m in target + sorted(by_mon) * 10:
        if len(picks) >= n_benign:
            break
        pool = [a for a in by_mon.get(m, []) if a["cust"] not in custs]
        if pool:
            a = pool[0]
            by_mon[m].remove(a)
            picks.append(a)
            custs.add(a["cust"])
    cases = chosen + picks
    rng.shuffle(cases)
    return cases


def alert_key(a):
    r = a["truth"]
    if r is None:
        return {"disposition": "not_fraud", "typology": "none", "customer_role": "legitimate", "ring": None,
                "scenario": None}
    return {"disposition": "fraud", "typology": r["typology"], "customer_role": r["role"], "ring": r["ring"],
            "scenario": r["scenario"], "identity": r["identity"],
            "involvements": [{k: (iso(v) if hasattr(v, "isoformat") else v) for k, v in i.items()}
                             for i in r["involvements"]]}


# --- incident cases ---------------------------------------------------------------------------------------------------

def truth_groups(truth):
    """{group key: [(cust, resolved truth, start, first_load)]} for customers with a ring crew (key: ring, typology)
    or an independent fraudster's scenario run (key: scenario, instance anchor)."""
    import duckdb
    anchors = dict(duckdb.sql(f"SELECT instance, anchor FROM read_parquet('{truth.out / 'truth' / 'world' / 'instances.parquet'}')").fetchall())
    groups = defaultdict(list)
    for cust in truth.ever_touched():
        r = truth.customer(cust)
        if not r or r["role"] in ("incidental", None):
            continue
        invs = [i for i in r["involvements"] if i["role"] or i["part"] == "cover"]
        if not invs:
            continue
        start = min(i["start"] for i in invs)
        first_load = min(i["first_load"] for i in invs)
        if r["ring"]:
            key = ("ring", r["ring"], r["typology"])
        else:
            key = ("independent", r["scenario"], anchors.get(r["instance"]))
        groups[key].append({"cust": cust, "truth": r, "start": start, "first_load": first_load})
    return groups


def waves(members, max_span=timedelta(days=35), min_size=4, single_if_at_most=12):
    members = sorted(members, key=lambda m: m["start"])
    if len(members) <= single_if_at_most:
        return [members]
    out, i = [], 0
    while i < len(members):
        s = members[i]["start"]
        w = [m for m in members[i:] if m["start"] <= s + max_span]
        if len(w) >= min_size:
            out.append(w)
        i += len(w)
    return out


def best_seed_key(db, members, kinds=SEED_KINDS, exclude=()):
    """The bank-visible key held by the most members (ties: the most exclusive), among keys no coarser than
    MAX_KEY_SIZE. Returns (kind, key, holders among members, key size) or None."""
    ids = [m["cust"] for m in members]
    rows = db.execute(f"""SELECT c.kind, c.k, count(DISTINCT c.CUST_ID) hits, any_value(s.customers) size
                          FROM custkeys c JOIN keysize s USING (kind, k)
                          WHERE c.CUST_ID IN (SELECT unnest(?)) AND s.customers <= {MAX_KEY_SIZE}
                            AND c.kind IN (SELECT unnest(?))
                          GROUP BY ALL HAVING count(DISTINCT c.CUST_ID) >= 2
                          ORDER BY hits DESC, size ASC, c.kind, c.k""", [ids, list(kinds)]).fetchall()
    for kind, k, hits, size in rows:
        if (kind, k) not in exclude:
            holders = [r[0] for r in db.execute("""SELECT CUST_ID FROM custkeys WHERE kind = ? AND k = ?
                                                   AND CUST_ID IN (SELECT unnest(?)) ORDER BY first_ts""",
                                                [kind, k, ids]).fetchall()]
            return kind, k, holders, size
    return None


def reachable(db, seed, max_hops=3, max_size=REACH_KEY_SIZE):
    """Customers reachable from the seed over shared keys no coarser than max_size, with hop counts."""
    hops = {c: 0 for c in seed}
    frontier = set(seed)
    for h in range(1, max_hops + 1):
        if not frontier:
            break
        rows = db.execute(f"""SELECT DISTINCT b.CUST_ID FROM custkeys a JOIN keysize s USING (kind, k)
                              JOIN custkeys b ON b.kind = a.kind AND b.k = a.k
                              WHERE a.CUST_ID IN (SELECT unnest(?)) AND s.customers <= {max_size}
                                AND a.kind IN (SELECT unnest(?))""", [sorted(frontier), REACH_KINDS]).fetchall()
        new = {r[0] for r in rows} - set(hops)
        for c in new:
            hops[c] = h
        frontier = new
    return hops


def pattern_seed(members, alerts, start, T):
    """(seed, member customers that fired the monitor, monitor, all customers that fired it in the window) for the
    monitor that fired on the most members within [start, T], when at least 3 did."""
    ids = {m["cust"] for m in members}
    hits, allc = defaultdict(set), defaultdict(set)
    first = {}
    for a in alerts:
        if start <= a["ts"] <= T:
            allc[a["monitor"]].add(a["cust"])
            if a["cust"] in ids:
                hits[a["monitor"]].add(a["cust"])
                first[(a["monitor"], a["cust"])] = min(first.get((a["monitor"], a["cust"]), a["ts"]), a["ts"])
    if not hits:
        return None
    mon = max(hits, key=lambda m: (len(hits[m]), -len(allc[m])))
    if len(hits[mon]) < 3:
        return None
    order = sorted(hits[mon], key=lambda c: first[(mon, c)])
    seed = order[:max(2, min(4, len(order) // 3))]
    return seed, set(order), mon, len(allc[mon])


def seed_size(n):
    return max(2, min(6, math.ceil(0.35 * n)))


def make_incident(out, group_key, members, truth, rng, alerts=None, group_all=()):
    """One incident case from a wave of a truth group, or None when it can't be seeded or expanded answerably."""
    members = sorted(members, key=lambda m: m["first_load"])
    if len(members) < 3:
        return None
    # case time: when most members have become visible, plus two days
    k = max(2, math.ceil(0.8 * len(members)))
    T = members[k - 1]["first_load"] + timedelta(days=2)
    window_start = min(m["start"] for m in members) - timedelta(days=1)
    visible = [m for m in members if m["first_load"] <= T]
    if len(visible) < 3:
        return None
    db = open_asof(out / "bank", T, lock=False)
    build_keys(db)
    best = best_seed_key(db, visible)
    scored, anomaly = [], None
    if best is not None:
        kind, key, holders, size = best
        seed = holders[:seed_size(len(visible))]
        hops = reachable(db, seed)
        scored = [m for m in visible if m["cust"] in hops and m["cust"] not in seed]
        anomaly = {"kind": kind, "key": key, "customers_with_key": size,
                   "text": f"{len(seed)} customers " + KIND_TEXT[kind].format(k=key)
                           + f" (the bank has {size} customers with this {kind.replace('_', ' ')} so far)"}
    if not scored and alerts is not None:
        # no rare shared key (typical of victims of one crew): seed on a monitor spike instead; the members that
        # fired the same monitor in the window are the answerable ones
        pat = pattern_seed(visible, alerts, window_start, T)
        if pat is None:
            return None
        seed, hits, monitor, n_all = pat
        hops = {c: (0 if c in seed else 1) for c in hits}
        scored = [m for m in visible if m["cust"] in hits and m["cust"] not in seed]
        anomaly = {"kind": "monitor", "key": monitor, "customers_with_key": n_all,
                   "text": f"{len(seed)} customers of a recent cluster fired the monitor '{monitor}' "
                           f"({MONITOR_DESCRIPTIONS[monitor]}); {n_all} customers fired it in the window"}
        kind, key = "monitor", monitor
    if not scored:
        return None
    roles = {m["cust"]: m["truth"]["role"] for m in visible}
    return {"case_time": T, "window_start": window_start, "seed": seed, "anomaly": anomaly,
            "key": {"is_incident": True, "group": list(group_key), "ring": group_key[1] if group_key[0] == "ring" else None,
                    "typology": _group_typology(members),
                    "members": [{"cust_id": m["cust"], "role": roles[m["cust"]], "hops": hops.get(m["cust"]),
                                 "scored": m["cust"] in hops,
                                 "seed": m["cust"] in seed} for m in visible],
                    "not_yet_visible": [m["cust"] for m in members if m["cust"] not in roles],
                    # the same crew's customers outside this window: neither required nor counted as wrong
                    "same_group_other": sorted({m["cust"] for m in group_all} - set(roles)),
                    "seed_key": [kind, key]}}


def _group_typology(members):
    return Counter(m["truth"]["typology"] for m in members).most_common(1)[0][0]


FP_KINDS = ["ip", "ach_originator", "merchant", "address", "payee", "device"]


def fp_incidents(out, truth, times, rng, n):
    """Benign clusters: keys whose holders (4-30 customers) the world never touched, at the given case times."""
    touched = truth.ever_touched()
    found, used_kinds = [], Counter()
    for T in times:
        db = open_asof(out / "bank", T, lock=False)
        build_keys(db)
        db.execute("CREATE TEMP TABLE touched AS SELECT unnest(?) c", [sorted(touched)])
        rows = db.execute(f"""
            SELECT c.kind, c.k, count(DISTINCT c.CUST_ID) holders,
                   count(DISTINCT c.CUST_ID) FILTER (WHERE c.last_ts >= CAST(? AS TIMESTAMPTZ) - INTERVAL 7 DAY) recent
            FROM custkeys c JOIN keysize s USING (kind, k)
            WHERE s.customers BETWEEN 4 AND 30 AND c.kind IN (SELECT unnest(?))
              AND NOT EXISTS (SELECT 1 FROM custkeys c2 JOIN touched t ON t.c = c2.CUST_ID WHERE c2.kind = c.kind AND c2.k = c.k)
            GROUP BY ALL HAVING recent >= 3 ORDER BY recent DESC""", [str(T), FP_KINDS]).fetchall()
        rows = [r for r in rows if not any(f["anomaly"]["key"] == r[1] for f in found)]
        if not rows:
            continue
        rows.sort(key=lambda r: (used_kinds[r[0]], -r[3]))
        kind, key, size, recent = rows[0]
        holders = [r[0] for r in db.execute("""SELECT CUST_ID FROM custkeys WHERE kind = ? AND k = ?
                                               AND last_ts >= CAST(? AS TIMESTAMPTZ) - INTERVAL 7 DAY
                                               ORDER BY last_ts DESC""", [kind, key, str(T)]).fetchall()]
        seed = holders[:min(6, max(3, len(holders) // 2))]
        used_kinds[kind] += 1
        found.append({"case_time": T, "window_start": T - timedelta(days=30), "seed": seed,
                      "anomaly": {"kind": kind, "key": key, "customers_with_key": size,
                                  "text": f"{len(seed)} customers " + KIND_TEXT[kind].format(k=key)
                                          + f" (the bank has {size} customers with this {kind.replace('_', ' ')} so far)"},
                      "key": {"is_incident": False, "group": ["benign", kind, key], "ring": None, "typology": "none",
                              "members": [], "seed_key": [kind, key],
                              "benign_holders": holders}})
        if len(found) >= n:
            break
    return found


# --- workspaces ---------------------------------------------------------------------------------------------------------

def write_workspace(root, case_id, case, brief):
    ws = root / case_id
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "drafts").mkdir(parents=True)
    (ws / "case.json").write_text(json.dumps(case, indent=1, default=str))
    (ws / "brief.md").write_text(brief)
    (ws / "notes.md").write_text("")
    return ws


def build(out, *, n_alerts=40, n_incidents=12, n_fp=3, fraud_share=0.45, seed=0, log=print):
    from . import briefs
    out = Path(out)
    rng = random.Random(seed)
    truth = Truth(out)
    keys_dir = out / "truth" / "keys"
    if keys_dir.exists():
        shutil.rmtree(keys_dir)
    keys_dir.mkdir(parents=True)
    for d in ("alert", "incident"):
        if (out / d).exists():
            shutil.rmtree(out / d)
    index = {"alert": [], "incident": []}

    fraud, benign = alert_candidates(out, truth)
    log(f"  alert candidates: {len(fraud)} fraud, {len(benign)} benign")
    cases = sample_alerts(fraud, benign, n_alerts, fraud_share, rng)
    share = sum(1 for a in cases if a["truth"]) / max(len(cases), 1)
    for i, a in enumerate(cases, 1):
        cid = f"A{i:03d}"
        case = {"task": "alert", "case_id": cid, "case_time": iso(a["case_time"]), "bank": "../../bank",
                "customer": a["cust"],
                "alert": {"type": a["monitor"], "monitor": MONITOR_DESCRIPTIONS[a["monitor"]], "reason": a["reason"],
                          "time": iso(a["ts"]), "ref": a["ref"]},
                "fraud_share": round(share, 2)}
        write_workspace(out / "alert", cid, case, briefs.alert_brief(case))
        key = {"case_id": cid, "customer": a["cust"], "monitor": a["monitor"], **alert_key(a)}
        (keys_dir / f"{cid}.json").write_text(json.dumps(key, indent=1, default=str))
        index["alert"].append({"case_id": cid, "monitor": a["monitor"]})

    groups = truth_groups(truth)
    alerts = run_monitors(out)
    incidents = []
    for gkey, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        ws = waves(members)
        rng.shuffle(ws)
        made = 0
        for w in sorted(ws, key=lambda w: -len(w)):
            if made >= 3:
                break
            inc = make_incident(out, gkey, w, truth, rng, alerts, members)
            if inc:
                incidents.append(inc)
                made += 1
    log(f"  true incidents available: {len(incidents)} from {len(groups)} groups")
    incidents.sort(key=lambda x: (-len(x['key']['members'])))
    # keep a spread: at most 3 per group
    true_n = n_incidents - n_fp
    per_group, picked = Counter(), []
    for inc in incidents:
        g = tuple(inc["key"]["group"][:2])
        if per_group[g] < 3 and len(picked) < true_n:
            picked.append(inc)
            per_group[g] += 1
    times = [inc["case_time"] for inc in picked] or []
    rng.shuffle(times)
    fps = fp_incidents(out, truth, times, rng, n_fp)
    allinc = picked + fps
    rng.shuffle(allinc)
    for i, inc in enumerate(allinc, 1):
        cid = f"B{i:03d}"
        case = {"task": "incident", "case_id": cid, "case_time": iso(inc["case_time"]), "bank": "../../bank",
                "window_start": iso(inc["window_start"]), "seed_customers": inc["seed"], "anomaly": inc["anomaly"]}
        write_workspace(out / "incident", cid, case, briefs.incident_brief(case))
        key = {"case_id": cid, **inc["key"], "seed": inc["seed"]}
        (keys_dir / f"{cid}.json").write_text(json.dumps(key, indent=1, default=str))
        index["incident"].append({"case_id": cid, "anomaly_kind": inc["anomaly"]["kind"]})

    summary = summarize(out)
    index["summary"] = summary
    (out / "index.json").write_text(json.dumps(index, indent=1, default=str))
    return summary


def summarize(out):
    out = Path(out)
    keys = [json.loads(p.read_text()) for p in sorted((out / "truth" / "keys").glob("*.json"))]
    a = [k for k in keys if k["case_id"].startswith("A")]
    b = [k for k in keys if k["case_id"].startswith("B")]
    return {
        "alert_cases": len(a),
        "alert_fraud_share": round(sum(k["disposition"] == "fraud" for k in a) / max(len(a), 1), 3),
        "alert_by_typology": dict(Counter(k["typology"] for k in a)),
        "alert_by_role": dict(Counter(k["customer_role"] for k in a)),
        "alert_by_monitor": dict(Counter(f"{k['monitor']}:{k['disposition']}" for k in a)),
        "incident_cases": len(b),
        "incident_true": sum(k["is_incident"] for k in b),
        "incident_by_group": [f"{k['case_id']} {'/'.join(str(x) for x in k['group'][:3])} seed={len(k['seed'])} "
                              f"members={len(k['members'])} scored={sum(m['scored'] and not m['seed'] for m in k['members'])} "
                              f"key={k['seed_key'][0]}" for k in b],
    }
