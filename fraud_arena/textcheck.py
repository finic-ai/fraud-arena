"""The style check, grader side: can a classifier that reads only a text cell tell fraud rows from ordinary ones?

For each rich text column (`textgen.RICH`), rows are labelled from the run's truth: `row_links` maps a row of an
observer table to its event, and the row is fraud when the event's cause is a scenario. Two classifiers, each scored
by stratified cross-validated AUC:

- `ngram`: character 2-4-gram TF-IDF and logistic regression (reads wording, so it also reads content);
- `style`: length, word length, punctuation, case, digits and type-token ratio only;
- `facts`, the baseline: the same row's other columns that the generator sees, without the text. Text that scores
  above it tells the defender more than the row's fields do.

Run it on the world's template text and on the rewritten text of the same rows to see what generation changes.
Truth and tables must come from the same write (a render, or `write_truth` and `write_observer` at the same day),
since `row_links` points at row numbers.

    python -m fraud_arena.textcheck --truth RENDER/truth --tables RENDER [--tables REWRITTEN]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import zlib
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .textgen import RICH

DIM = 1 << 18


def row_truth(truth: Path, observer: str, table: str) -> dict[int, bool] | None:
    """Row number -> fraud, or None when the run has no row-level truth for the table."""
    links = Path(truth) / "row_links.parquet"
    if not links.exists():
        return None
    c = duckdb.connect()
    rows = c.execute(f"""
        select l.row, coalesce(e.cause_kind = 'scenario', false)
        from read_parquet(?) l join '{Path(truth) / 'events.parquet'}' e on e.key = l.event
        where l.observer = ? and l."table" = ?""", [str(links), observer, table]).fetchall()
    return {int(r): bool(f) for r, f in rows} or None


def _grams(s: str) -> Counter:
    s = f" {s.lower()} "
    return Counter(zlib.crc32(s[i:i + n].encode()) % DIM for n in (2, 3, 4) for i in range(len(s) - n + 1))


def ngram_features(texts: list[str]) -> list[dict[int, float]]:
    counts = [_grams(t) for t in texts]
    df = Counter(g for c in counts for g in c)
    n = len(texts)
    out = []
    for c in counts:
        v = {g: (1 + math.log(k)) * (math.log((1 + n) / (1 + df[g])) + 1) for g, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({g: x / norm for g, x in v.items()})
    return out


def style_features(texts: list[str]) -> list[dict[int, float]]:
    raw = []
    for t in texts:
        words = re.findall(r"\w+", t)
        n = max(len(t), 1)
        raw.append([math.log1p(len(t)), math.log1p(len(words)),
                    sum(map(len, words)) / max(len(words), 1),
                    sum(ch in ".,;:!?'\"-()" for ch in t) / n, sum(ch.isupper() for ch in t) / n,
                    sum(ch.isdigit() for ch in t) / n, len({w.lower() for w in words}) / max(len(words), 1),
                    t.count(".") + t.count("!") + t.count("?"), float("!" in t), float("?" in t)])
    k = len(raw[0]) if raw else 0
    mean = [sum(r[j] for r in raw) / len(raw) for j in range(k)]
    sd = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in raw) / len(raw)) or 1.0 for j in range(k)]
    return [{j: (r[j] - mean[j]) / sd[j] for j in range(k)} for r in raw]


def facts_features(rows: list[dict]) -> list[dict[int, float]]:
    def tok(k, v):
        try:
            v = f"~{round(math.log2(1 + abs(float(v))))}"
        except (TypeError, ValueError):
            pass
        return zlib.crc32(f"{k}={v}".encode()) % DIM
    return [{tok(k, v): 1.0 for k, v in r.items()} for r in rows]


def _fit(X, y, epochs=30, rate=0.5, l2=1e-4, seed=0):
    w: dict[int, float] = {}
    b = 0.0
    order = list(range(len(X)))
    rng = random.Random(seed)
    pos = sum(y) or 1
    neg = (len(y) - sum(y)) or 1
    weight = {1: len(y) / (2 * pos), 0: len(y) / (2 * neg)}
    for ep in range(epochs):
        rng.shuffle(order)
        step = rate / (1 + ep)
        for i in order:
            z = b + sum(w.get(j, 0.0) * x for j, x in X[i].items())
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            g = (p - y[i]) * weight[y[i]]
            for j, x in X[i].items():
                w[j] = w.get(j, 0.0) * (1 - step * l2) - step * g * x
            b -= step * g
    return w, b


def _score(model, x):
    w, b = model
    return b + sum(w.get(j, 0.0) * v for j, v in x.items())


def auc(y: list[int], s: list[float]) -> float:
    pairs = sorted(zip(s, y))
    rank, i, ranks = 1, 0, [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (rank + rank + j - i - 1) / 2
        rank += j - i
        i = j
    pos = sum(y)
    neg = len(y) - pos
    return (sum(r for r, (_, lab) in zip(ranks, pairs) if lab) - pos * (pos + 1) / 2) / (pos * neg)


def cv_auc(X, y, folds=5, seed=0) -> float | None:
    pos = [i for i, v in enumerate(y) if v]
    neg = [i for i, v in enumerate(y) if not v]
    k = min(folds, len(pos), len(neg))
    if k < 2:
        return None
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold = {i: n % k for n, i in enumerate(pos)} | {i: n % k for n, i in enumerate(neg)}
    scores = [0.0] * len(y)
    for f in range(k):
        train = [i for i in range(len(y)) if fold[i] != f]
        model = _fit([X[i] for i in train], [y[i] for i in train], seed=seed)
        for i in range(len(y)):
            if fold[i] == f:
                scores[i] = _score(model, X[i])
    return round(auc(y, scores), 3)


def check(truth: Path, tables: Path) -> dict:
    """AUCs per rich text column of the observer folders under `tables` (a render, or a copy of one)."""
    out = {}
    for col in RICH:
        path = Path(tables) / col.institution / f"{col.table}.parquet"
        name = f"{col.institution}.{col.table}.{col.column}"
        if not path.exists():
            continue
        labels = row_truth(truth, col.institution, col.table)
        if labels is None:
            out[name] = {"error": "no row truth yet (row_links is empty)"}
            continue
        table = pq.read_table(path)
        texts = table.column(col.column).to_pylist()
        facts = [{c: table.column(c)[i].as_py() for c in col.facts if c in table.column_names}
                 for i in range(table.num_rows)]
        idx = [i for i, t in enumerate(texts) if t and i in labels]
        y = [int(labels[i]) for i in idx]
        tx = [texts[i] for i in idx]
        rec = {"rows": len(y), "fraud": sum(y)}
        if 0 < sum(y) < len(y):
            rec["auc_ngram"] = cv_auc(ngram_features(tx), y)
            rec["auc_style"] = cv_auc(style_features(tx), y)
            rec["auc_facts"] = cv_auc(facts_features([facts[i] for i in idx]), y)
        out[name] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraud-arena-textcheck")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--tables", type=Path, action="append", required=True,
                    help="a folder of observer folders (repeat to compare template and rewritten text)")
    a = ap.parse_args(argv)
    print(json.dumps({str(t): check(a.truth, t) for t in a.tables}, indent=2))


if __name__ == "__main__":
    main()
