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


def fit(work, kind, decision, features, *, label="fraud_known", where=None, depth=3, trees=60, balanced=False):
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
    rows = db.execute(f'SELECT {cols}, {label}::int AS y, ts FROM "{table}" {cond} ORDER BY ts').fetchall()
    if len(rows) < 50:
        raise ValueError(f"only {len(rows)} labeled cases: too few to fit")
    X = np.array([[np.nan if v is None else float(v) for v in r[:-2]] for r in rows], dtype=float)
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
    else:
        report.append("the test part has no known fraud (or only fraud): no test metrics")
    return "\n".join(report), model
