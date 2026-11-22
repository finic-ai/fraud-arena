"""Grade answers against the answer keys (cases/NAME/truth/keys)."""

import json
from pathlib import Path

FN_COST, FP_COST = 5.0, 1.0


def load_keys(cases):
    return {p.stem: json.loads(p.read_text()) for p in sorted((Path(cases) / "truth" / "keys").glob("*.json"))}


def score_alerts(keys, answers):
    """answers: {case_id: answer dict or None}. Missing answers count as not_fraud with confidence 0.5."""
    rows = []
    for cid, ans in answers.items():
        k = keys[cid]
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
        k = keys[cid]
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
            r.update({"typology_ok": a.get("typology") == k["typology"], "true_typology": k["typology"],
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
            "member_precision": mean([r["precision"] for r in true_rows]),
            "member_recall": mean([r["recall"] for r in true_rows]),
            "member_f1": mean([r["f1"] for r in true_rows]),
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
