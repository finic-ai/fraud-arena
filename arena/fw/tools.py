"""The defender's analysis tools behind `arena fit`. They run in the harness's process, on the
player's own data, and return what the player may use: model blocks ready to deploy."""

import json
import math
import os
from pathlib import Path

import duckdb


def connect(work):
    """An in-memory DuckDB with a view per Parquet file of the player's data (the bank's tables and the case
    tables) """
    db = duckdb.connect()
    for folder in (Path(work) / "data",):
        for f in sorted(folder.glob("*.parquet")) if folder.exists() else []:
            db.execute(f"CREATE VIEW \"{f.stem}\" AS SELECT * FROM read_parquet('{f}')")
    return db


# --- arena fit ------------------------------------------------------------------------------------------------------

NUMERIC = ("money", "int", "float", "bool")


# What a caught fraud is worth when the decision has no amount (a stated assumption; --fraud-value overrides it).
FRAUD_VALUE = {"brind_login_check": 300.0, "brind_account_open_review": 500.0, "brind_business_open_review": 500.0}
# The action a model cutoff would take, per decision (the cost-aware report prices it).
DEFAULT_ACTION = {"brind_card_auth": "decline", "brind_login_check": "step_up", "brind_payment_hold": "hold",
                  "brind_account_open_review": "review", "brind_business_open_review": "review",
                  "brind_ach_debit_review": "return_debit", "brind_ach_push": "hold", "brind_ach_hold": "hold",
                  "brind_wire_release": "hold", "brind_consumer_wire": "hold", "brind_txn_monitoring": "raise_alert"}


def fit(work, kind, decision, features, *, label="fraud_known", where=None, depth=3, trees=60, balanced=False,
        action=None, fraud_value=None, return_best=False):
    """Fit a model on the case table of `decision` and return (report text, model block). The split is by time:
    the last quarter of cases is the test set."""
    import numpy as np
    decisions = {d["name"]: d for d in json.loads((Path(work) / "decisions.json").read_text())}
    full = decision if decision.startswith("brind_") else "brind_" + decision
    d = decisions.get(full)
    if d is None:
        raise ValueError(f"unknown decision {decision!r}")
    kinds = {f[0]: str(f[1]) for f in d["features"]}
    bad = [f for f in features if f not in kinds or not kinds[f].startswith(NUMERIC)]
    if bad:
        raise ValueError(f"features must be numeric or bool features of {full}; not usable: {bad}")
    table = f"cases_{full.replace('brind_', '')}"
    db = connect(work)
    cond = f"WHERE {where}" if where else ""
    cols = ", ".join(f'"{f}"' for f in features)
    amount_col = '"amount"' if "amount" in kinds else "NULL"
    rows = db.execute(f'SELECT {cols}, {amount_col} AS _amt, {label}::int AS y, ts FROM "{table}" {cond} ORDER BY ts').fetchall()
    if len(rows) < 50:
        raise ValueError(f"only {len(rows)} labeled cases: too few to fit")
    X = np.array([[np.nan if v is None else float(v) for v in r[:-3]] for r in rows], dtype=float)
    amt = np.array([np.nan if r[-3] is None else float(r[-3]) for r in rows], dtype=float)
    y = np.array([r[-2] for r in rows], dtype=int)
    cut = int(len(rows) * 0.75)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    if ytr.sum() < 3:
        raise ValueError(f"the training part has only {int(ytr.sum())} known fraud cases: too few to fit")
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd[sd == 0] = 1.0
        Z = lambda A: (np.nan_to_num(A, nan=0.0) - mu) / sd  # noqa: E731  (missing counts 0, as the engine does)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None).fit(Z(Xtr), ytr)
        w = clf.coef_[0] / sd
        b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
        model = {"kind": "logistic", "features": features, "weights": [round(float(x), 6) for x in w],
                 "intercept": round(b, 6)}
        score = lambda A: 1 / (1 + np.exp(-(np.nan_to_num(A, nan=0.0) @ w + b)))  # noqa: E731
    elif kind == "trees":
        import xgboost as xgb
        pos = max(int(ytr.sum()), 1)
        booster = xgb.train({"objective": "binary:logistic", "max_depth": depth, "eta": 0.1,
                             "scale_pos_weight": (len(ytr) - pos) / pos if balanced else 1.0, "verbosity": 0},
                            xgb.DMatrix(Xtr, label=ytr, feature_names=features), num_boost_round=trees)
        cfg = json.loads(booster.save_config())
        p0 = float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
        dump = [json.loads(t) for t in booster.get_dump(dump_format="json")]
        model = {"kind": "trees", "features": features, "base_score": round(math.log(p0 / (1 - p0)), 8),
                 "sigmoid": True, "trees": dump}
        score = lambda A: booster.predict(xgb.DMatrix(A, feature_names=features))  # noqa: E731
    else:
        raise ValueError("kind is logistic or trees")
    report = [f"{kind} model for {full} on {len(rows)} cases ({int(y.sum())} known fraud); trained on the first "
              f"{cut} ({int(ytr.sum())} fraud), tested on the last {len(rows) - cut} ({int(yte.sum())} fraud)."]
    if yte.sum() > 0 and yte.sum() < len(yte):
        from sklearn.metrics import average_precision_score, roc_auc_score
        p = score(Xte)
        report.append(f"test AUC {roc_auc_score(yte, p):.3f}, average precision {average_precision_score(yte, p):.3f} "
                      f"(fraud base rate {yte.mean():.4f})")
        import numpy as np
        for share in (0.005, 0.01, 0.02, 0.05):
            t = float(np.quantile(p, 1 - share))
            flag = p >= t
            tp = int((flag & (yte == 1)).sum())
            report.append(f"  top {share:.1%} (score >= {t:.4g}): flags {int(flag.sum())} of {len(yte)} test cases, "
                          f"catching {tp} of {int(yte.sum())} fraud ({int(flag.sum()) - tp} good cases flagged)")
        report.append("Scores are calibrated to the base rate, so useful cutoffs are often small: pick one from the "
                      "lines above, e.g. score.NAME > <cutoff>.")
        # Cost-aware: the net dollars each cutoff would have made on the test cases, priced like the match's score.
        from arena.fw.config import ScoreWeights
        wts = ScoreWeights()
        act = action or DEFAULT_ACTION.get(full, "decline")
        fv = float(fraud_value) if fraud_value is not None else FRAUD_VALUE.get(full)
        ate = amt[cut:]
        value = np.where(np.isnan(ate), fv if fv is not None else 0.0, ate)
        cost = np.array([wts.action_cost(act, 0.0 if np.isnan(a) else a) for a in ate])
        report.append(f"Net dollars on the test cases if your rule took action {act!r} above each cutoff (caught fraud "
                      f"counted at {'its amount' if fv is None else f'its amount, or ${fv:.0f} without one'}; each good "
                      f"case flagged costs the match's friction for {act!r}):")
        best = None
        for share in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            t = float(np.quantile(p, 1 - share))
            flag = p >= t
            net = float((value * (flag & (yte == 1))).sum() - (cost * (flag & (yte == 0))).sum())
            report.append(f"  score >= {t:.4g} (top {share:.1%}): net ${net:,.0f}")
            if best is None or net > best[1]:
                best = (t, net, share)
        if best and best[1] > 0:
            report.append(f"Best on the test cases: score >= {best[0]:.4g} (net ${best[1]:,.0f}). Test cases are the "
                          f"last quarter of history; the live week can differ.")
        else:
            report.append("No cutoff made money on the test cases: friction outweighs the fraud caught here.")
        model["_best"] = best
    else:
        report.append("the test part has no known fraud (or only fraud): no test metrics")
    best = model.pop("_best", None)
    if return_best:
        return "\n".join(report), model, best
    return "\n".join(report), model
