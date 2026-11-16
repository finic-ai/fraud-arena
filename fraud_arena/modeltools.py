"""Turn a trained model into a Runmere deployed-policy model (SPEC §8.8), for `bundle/models/*.json`.

    from fraud_arena.modeltools import from_xgboost, from_sklearn_linear, save
    save(from_xgboost(booster, features=["amount", "txns_1h"]), "bundle/models/fraud.json")

The engine scores trees in f64 in tree order; XGBoost sums in f32, so scores agree to about 1e-6.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def from_xgboost(model, features: list[str]) -> dict:
    """An `xgboost.Booster` or `XGBClassifier`/`XGBRegressor` trained on columns in `features`' order."""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    cfg = json.loads(booster.save_config())
    lp = cfg["learner"]["learner_model_param"]
    objective = cfg["learner"]["objective"]["name"]
    base = float(str(lp["base_score"]).strip("[]"))
    logistic = objective in ("binary:logistic", "reg:logistic")
    margin = math.log(base / (1 - base)) if logistic else base
    names = booster.feature_names
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    if names and list(names) != list(features):
        rename = dict(zip(names, features))
        _rename(trees, rename)
    return {"kind": "trees", "features": list(features), "base_score": margin, "sigmoid": logistic, "trees": trees}


def _rename(nodes, rename):
    for n in nodes:
        if "split" in n:
            n["split"] = rename.get(n["split"], n["split"])
        _rename(n.get("children", []), rename)


def from_sklearn_linear(model, features: list[str], *, logistic: bool | None = None) -> dict:
    """A fitted `LogisticRegression` (binary) or `LinearRegression`/`Ridge`, on raw (unscaled) feature columns. Fold
    any scaler into the weights before exporting: the engine reads raw observation values."""
    coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) == 2 else model.coef_
    icpt = model.intercept_[0] if hasattr(model.intercept_, "__len__") else model.intercept_
    if logistic is None:
        logistic = hasattr(model, "predict_proba")
    return {"kind": "logistic" if logistic else "linear", "features": list(features),
            "weights": [float(w) for w in coef], "intercept": float(icpt)}


def save(model: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model))
    return p
