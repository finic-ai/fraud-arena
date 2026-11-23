"""Workflow-mode alert decisions.
    ask      CASES... --player P --out FILE           answer every case's family questions
    fit      CALIB_ANSWER_FILES... --cases CASES... --out DIR   fit and freeze each player's tuned combiner
    compare  CASES... --runs ANSWER_FILES... --combiners DIR [--direct DIR]   score hand, tuned (and direct)
Scoring: every player is told the real-world rate (10%); benign cases are weighted back to it."""
import argparse, json, math, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.tasks.workflow import ask, family, hand_score, fit_tuned, tuned_score
from arena.tasks.schema import REAL_WORLD_ALERT_FRAUD_RATE as RATE

def cases(case_sets):
    for cs in case_sets:
        cs = Path(cs); seed = json.loads((cs / "index.json").read_text())["seed"]
        for d in sorted((cs / "alert").iterdir()):
            if (d / "case.json").exists():
                key = json.loads((cs / "truth" / "keys" / f"{d.name}.json").read_text())
                yield f"s{seed}-{d.name}", d, 1 if key["disposition"] == "fraud" else 0

def slug(p): return p.replace(":", "_").replace("/", "_")

def cmd_ask(a):
    js = list(cases(a.cases))
    def one(j):
        name, d, _ = j; err = ""
        for _ in range(3):
            try:
                ans, cost = ask(a.player, d); return name, {"family": family(d), "answers": ans, "cost": cost}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        return name, {"error": err}
    with ThreadPoolExecutor(a.parallel) as pool:
        res = dict(pool.map(one, js))
    Path(a.out).write_text(json.dumps({"player": a.player, "results": res}, indent=1))
    ok = [v for v in res.values() if "answers" in v]
    print(a.player, f"answered {len(ok)}/{len(res)}", f"cost=${sum(v['cost'] for v in ok):.3f}")

def cmd_fit(a):
    labels = {n: y for n, _, y in cases(a.cases)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in a.runs:
        r = json.loads(Path(f).read_text())
        rows = [(v["family"], v["answers"], labels[k]) for k, v in r["results"].items() if "answers" in v and k in labels]
        comb = fit_tuned(rows, None)
        (Path(a.out) / f"{slug(r['player'])}.json").write_text(json.dumps({"player": r["player"], **comb}, indent=1))
        print("froze", r["player"], f"on {comb['n']} calibration alerts ({comb['fraud']} fraud)")

def metrics(rows):
    P = sum(y for _, y in rows); N = len(rows) - P; w = (P / RATE - P) / N
    W = lambda y: 1 if y else w
    cl = lambda p: min(max(p, 1e-4), 1 - 1e-4)
    ll = sum(W(y) * -math.log(cl(p) if y else 1 - cl(p)) for p, y in rows) / sum(W(y) for _, y in rows)
    br = sum(W(y) * (p - y) ** 2 for p, y in rows) / sum(W(y) for _, y in rows)
    rs = sorted(rows, key=lambda x: -x[0]); tp = fp = 0; ap = 0; i = 0
    while i < len(rs):
        j = i
        while j < len(rs) and rs[j][0] == rs[i][0]: j += 1
        g = rs[i:j]; dd = sum(y for _, y in g); tp += dd; fp += sum((1 - y) * w for _, y in g)
        if dd: ap += dd / P * tp / (tp + fp)
        i = j
    pos = [p for p, y in rows if y]; neg = [p for p, y in rows if not y]
    auc = sum((x > z) + 0.5 * (x == z) for x in pos for z in neg) / (len(pos) * len(neg))
    return {"auc": auc, "pr_auc": ap, "log_loss": ll, "brier": br}

def boot(scores, labels, ids, n=1000):
    pos = [i for i in ids if labels[i]]; neg = [i for i in ids if not labels[i]]; rnd = random.Random(0); out = []
    for _ in range(n):
        s = [rnd.choice(pos) for _ in pos] + [rnd.choice(neg) for _ in neg]
        out.append(metrics([(scores[i], labels[i]) for i in s])["pr_auc"])
    out.sort(); return out[25], out[974]

def cmd_compare(a):
    labels = {n: y for n, _, y in cases(a.cases)}
    ids = sorted(labels); P = sum(labels.values())
    print(f"{len(ids)} alerts, {P} fraud; scored at a {RATE:.0%} fraud rate\n")
    print(f"{'player':38s} {'column':8s} {'AUC':>6s} {'PR-AUC':>7s} {'95% CI':>15s} {'logloss':>8s} {'Brier':>7s} {'cost':>8s}")
    for f in a.runs:
        r = json.loads(Path(f).read_text()); pl = r["player"]; res = r["results"]
        cols = {"hand": {k: hand_score(v["family"], v["answers"]) for k, v in res.items() if "answers" in v}}
        cf = Path(a.combiners) / f"{slug(pl)}.json"
        if cf.exists():
            comb = json.loads(cf.read_text())
            cols["tuned"] = {k: tuned_score(comb, v["family"], v["answers"]) for k, v in res.items() if "answers" in v}
        if a.direct:
            df = Path(a.direct) / f"{slug(pl)}.json"
            if df.exists():
                d = json.loads(df.read_text())["results"]
                cols = {"direct": {k: v["answer"]["disposition"]["fraud"] for k, v in d.items() if "answer" in v}, **cols}
        cost = sum(v.get("cost", 0) for v in res.values())
        for col, sc in cols.items():
            common = [i for i in ids if i in sc]
            m = metrics([(sc[i], labels[i]) for i in common]); lo, hi = boot(sc, labels, common)
            print(f"{pl[:38]:38s} {col:8s} {m['auc']:6.3f} {m['pr_auc']:7.3f} [{lo:5.3f},{hi:5.3f}] {m['log_loss']:8.4f} {m['brier']:7.4f} ${cost:7.3f}")

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("ask"); s.add_argument("cases", nargs="+"); s.add_argument("--player", required=True); s.add_argument("--out", required=True); s.add_argument("--parallel", type=int, default=6)
s = sub.add_parser("fit"); s.add_argument("runs", nargs="+"); s.add_argument("--cases", nargs="+", required=True); s.add_argument("--out", required=True)
s = sub.add_parser("compare"); s.add_argument("cases", nargs="+"); s.add_argument("--runs", nargs="+", required=True); s.add_argument("--combiners", required=True); s.add_argument("--direct")
a = ap.parse_args(); {"ask": cmd_ask, "fit": cmd_fit, "compare": cmd_compare}[a.cmd](a)
