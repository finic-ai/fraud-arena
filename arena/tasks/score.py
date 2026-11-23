"""Grade answers against the answer keys (cases/NAME/truth/keys)."""

import json
from pathlib import Path

FN_COST, FP_COST = 5.0, 1.0


def load_keys(cases):
    return {p.stem: json.loads(p.read_text()) for p in sorted((Path(cases) / "truth" / "keys").glob("*.json"))}


def answers_flat(trials):
    """{case: [answers]} -> {case#k: answer} for the pooled metrics (each trial counts as one row)."""
    return {f"{c}#{i}": a for c, ts in trials.items() for i, a in enumerate(ts, 1)}


def score_alerts(keys, answers):
    """answers: {case_id: answer dict or None}. Missing answers count as not_fraud with confidence 0.5."""
    rows = []
    for cid, ans in answers.items():
        k = keys[cid.split("#")[0]]
        a = ans or {}
        pred = a.get("disposition", "not_fraud")
        conf = a.get("confidence", 0.5)
        conf = float(conf) if isinstance(conf, (int, float)) else 0.5
        y = k["disposition"] == "fraud"
        rows.append({"case_id": cid, "y": y, "pred": pred == "fraud", "conf": conf, "missing": ans is None,
                     "typ_ok": a.get("typology") == k["typology"], "role_ok": a.get("customer_role") == k["customer_role"],
                     "true_typ": k["typology"], "pred_typ": a.get("typology"), "true_role": k["customer_role"],
                     "pred_role": a.get("customer_role"), "monitor": k["monitor"]})
    n = len(rows)
    if not n:
        return {"n": 0}
    tp = sum(r["y"] and r["pred"] for r in rows)
    fp = sum(not r["y"] and r["pred"] for r in rows)
    fn = sum(r["y"] and not r["pred"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fraud = [r for r in rows if r["y"]]
    cost = (FN_COST * fn + FP_COST * fp) / n
    # the cheaper of the two trivial policies (flag everything / flag nothing), for reference
    trivial = min(FP_COST * sum(not r["y"] for r in rows), FN_COST * sum(r["y"] for r in rows)) / n
    return {"n": n, "missing": sum(r["missing"] for r in rows),
            "accuracy": round(sum(r["y"] == r["pred"] for r in rows) / n, 3),
            "fraud_precision": round(prec, 3), "fraud_recall": round(rec, 3),
            "fraud_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "typology_acc_on_fraud": round(sum(r["typ_ok"] for r in fraud) / len(fraud), 3) if fraud else None,
            "role_acc_on_fraud": round(sum(r["role_ok"] for r in fraud) / len(fraud), 3) if fraud else None,
            "brier": round(sum((r["conf"] - r["y"]) ** 2 for r in rows) / n, 4),
            "cost_per_case": round(cost, 3), "trivial_cost_per_case": round(trivial, 3),
            "cost_saving_vs_trivial": round(1 - cost / trivial, 3) if trivial else None,
            "rows": rows}


def score_incidents(keys, answers):
    rows = []
    for cid, ans in answers.items():
        k = keys[cid.split("#")[0]]
        a = ans or {}
        seed = set(k["seed"])
        pred_members = {m["cust_id"]: m.get("role") for m in a.get("members", []) if isinstance(m, dict)}
        added = set(pred_members) - seed
        verdict = a.get("is_incident")
        r = {"case_id": cid, "is_incident": k["is_incident"], "pred_incident": verdict, "missing": ans is None,
             "verdict_ok": verdict == k["is_incident"], "group": "/".join(str(x) for x in k["group"][:3]),
             "anomaly": k["seed_key"][0]}
        if k["is_incident"]:
            truth = {m["cust_id"]: m for m in k["members"]}
            scored = {c for c, m in truth.items() if m["scored"] and not m["seed"]}
            neutral = ({c for c, m in truth.items() if not m["scored"] and not m["seed"]}
                       | set(k.get("same_group_other", [])) | set(k.get("not_yet_visible", [])))
            hit = added & scored
            judged = added - neutral
            prec = len(hit) / len(judged) if judged else (1.0 if not scored else 0.0)
            rec = len(hit) / len(scored) if scored else 1.0
            role_pool = [c for c in pred_members if c in truth and (c in hit or truth[c]["seed"])]
            if k.get("expansion") == "none":
                scored = set()
            r.update({"typology_ok": a.get("typology") == k["typology"], "true_typology": k["typology"],
                      "expansion": k.get("expansion", "members"),
                      "pred_typology": a.get("typology"), "to_find": len(scored), "added": len(added),
                      "hit": len(hit), "wrong_added": len(judged - scored), "precision": round(prec, 3),
                      "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                      "role_acc": round(sum(pred_members[c] == truth[c]["role"] for c in role_pool) / len(role_pool), 3)
                      if role_pool else None,
                      "seed_kept": len(seed & set(pred_members)), "seed": len(seed)})
            if not verdict:
                r.update({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        else:
            r.update({"named_members": len(pred_members), "named_beyond_seed": len(added)})
        rows.append(r)
    true_rows = [r for r in rows if r["is_incident"]]
    fp_rows = [r for r in rows if not r["is_incident"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {"n": len(rows), "missing": sum(r["missing"] for r in rows),
            "verdict_acc": mean([r["verdict_ok"] for r in rows]),
            "true_incidents": len(true_rows), "fp_incidents": len(fp_rows),
            "typology_acc_on_true": mean([r["typology_ok"] for r in true_rows]),
            "member_precision": mean([r["precision"] for r in true_rows if r["expansion"] != "none"]),
            "member_recall": mean([r["recall"] for r in true_rows if r["expansion"] != "none"]),
            "member_f1": mean([r["f1"] for r in true_rows if r["expansion"] != "none"]),
            "role_acc": mean([r["role_acc"] for r in true_rows if r.get("role_acc") is not None]),
            "fp_verdict_acc": mean([r["verdict_ok"] for r in fp_rows]),
            "fp_mean_named": mean([r["named_members"] for r in fp_rows]),
            "rows": rows}


def collect_answers(run_dir, task):
    """{case_id: answer or None} from runs/NAME/<task>/<case_id>/answer.json."""
    out = {}
    root = Path(run_dir) / task
    for d in sorted(p for p in root.glob("*") if p.is_dir() and (p / "case.json").exists()):
        f = d / "answer.json"
        out[d.name] = json.loads(f.read_text()) if f.exists() else None
    return out


# --- per-case scores and k-trial aggregates (FrontierSWE-style) -----------------------------------------------------

def alert_case_score(key, ans):
    """One trial of one alert case: 0.6 disposition + 0.2 typology + 0.2 role (a benign case answered not_fraud /
    none / legitimate scores 1). A missing answer scores 0."""
    if not ans:
        return {"score": 0.0, "verdict_ok": False}
    ok = ans.get("disposition") == key["disposition"]
    typ = ans.get("typology") == key["typology"]
    role = ans.get("customer_role") == key["customer_role"]
    conf = ans.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else 0.5
    return {"score": round(0.6 * ok + 0.2 * typ + 0.2 * role, 4), "verdict_ok": ok,
            "brier": round((conf - (key["disposition"] == "fraud")) ** 2, 4)}


def incident_case_score(key, ans):
    """One trial of one incident case. True incident with members to find: 0.3 verdict + 0.1 typology + 0.6 member F1
    (F1 0 when the verdict is wrong); verdict-only incident (expansion none): 0.75 verdict + 0.25 typology; benign
    cluster: 0 if called an incident, else 1 minus 0.06 per customer named beyond the seed (at most 0.3 off)."""
    if not ans:
        return {"score": 0.0, "verdict_ok": False}
    r = score_incidents({key["case_id"]: key}, {key["case_id"]: ans})["rows"][0]
    if not key["is_incident"]:
        s = (1.0 - min(0.3, 0.06 * r["named_beyond_seed"])) if r["verdict_ok"] else 0.0
        return {"score": round(s, 4), "verdict_ok": r["verdict_ok"]}
    if key.get("expansion") == "none":
        s = 0.75 * r["verdict_ok"] + 0.25 * r["typology_ok"]
    else:
        s = 0.3 * r["verdict_ok"] + 0.1 * r["typology_ok"] + 0.6 * (r["f1"] if r["verdict_ok"] else 0.0)
    return {"score": round(s, 4), "verdict_ok": r["verdict_ok"], "f1": r["f1"]}


def k_trial_summary(per_case, n_boot=2000, seed=0):
    """per_case: {case: [trial dicts with score, verdict_ok]}. mean@K = mean over cases of the per-case mean over
    trials; worst@K / best@K = mean over cases of the per-case min / max; pass^K = share of cases whose verdict was
    right in every trial; ci95 = percentile bootstrap over cases of mean@K."""
    import random
    cases = sorted(c for c, ts in per_case.items() if ts)
    if not cases:
        return {"n_cases": 0}
    means = [sum(t["score"] for t in per_case[c]) / len(per_case[c]) for c in cases]
    mins = [min(t["score"] for t in per_case[c]) for c in cases]
    maxs = [max(t["score"] for t in per_case[c]) for c in cases]
    passk = [all(t["verdict_ok"] for t in per_case[c]) for c in cases]
    ks = sorted({len(per_case[c]) for c in cases})
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(n_boot))
    m = lambda xs: round(sum(xs) / len(xs), 4)  # noqa: E731
    out = {"n_cases": len(cases), "k": ks[0] if len(ks) == 1 else ks, "mean_at_k": m(means), "worst_at_k": m(mins),
           "best_at_k": m(maxs), "pass_hat_k_verdict": m(passk),
           "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)],
           "per_case": {c: round(v, 4) for c, v in zip(cases, means)}}
    briers = [t["brier"] for c in cases for t in per_case[c] if "brier" in t]
    if briers:
        out["brier"] = m(briers)
    return out


def collect_trials(run_dir, task):
    """{case_id: [answer or None per trial]} from runs/NAME/<task>/<CASE>/trial-<k>/answer.json (or the older
    single-trial layout runs/NAME/<task>/<CASE>/answer.json)."""
    out = {}
    root = Path(run_dir) / task
    for d in sorted(p for p in root.glob("*") if p.is_dir() and not p.name.startswith("bin")):
        trials = sorted(t for t in d.glob("trial-*") if (t / "case.json").exists())
        if not trials and (d / "case.json").exists():
            trials = [d]
        if trials:
            out[d.name] = [json.loads((t / "answer.json").read_text()) if (t / "answer.json").exists() else None
                           for t in trials]
    return out
