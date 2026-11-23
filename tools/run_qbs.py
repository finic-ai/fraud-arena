"""QBS on the alert cases: ask every question (one call per alert), fit by forward selection on the fit seeds
(cross-validated across fit seeds), freeze, score held-out seeds.
    python tools/run_qbs.py ask CASES... --player P --out FILE
    python tools/run_qbs.py evaluate --fit CASES... --test CASES... --answers FILE... --direct DIR"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.tasks.qbs import ask, QBS

def cases(sets):
    for cs in sets:
        cs = Path(cs); seed = json.loads((cs / "index.json").read_text())["seed"]
        for d in sorted((cs / "alert").iterdir()):
            if (d / "case.json").exists():
                key = json.loads((cs / "truth" / "keys" / f"{d.name}.json").read_text())
                mon = json.loads((d / "case.json").read_text())["alert"]["type"]
                yield f"s{seed}-{d.name}", f"s{seed}", d, mon, key

def cmd_ask(a):
    js = list(cases(a.cases))
    def one(j):
        name, seed, d, mon, key = j; err = ""
        for _ in range(3):
            try:
                ans, cost = ask(a.player, d); return name, {"answers": ans, "cost": cost}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        return name, {"error": err}
    with ThreadPoolExecutor(a.parallel) as pool:
        res = dict(pool.map(one, js))
    Path(a.out).write_text(json.dumps({"player": a.player, "results": res}))
    ok = [v for v in res.values() if "answers" in v]
    print(a.player, f"answered {len(ok)}/{len(res)}", f"cost=${sum(v['cost'] for v in ok):.3f}")

def cmd_eval(a):
    sys.path.insert(0, ".")
    from arena.tasks.decide import leaderboard_metrics
    import random
    for f in a.answers:
        r = json.loads(Path(f).read_text()); pl = r["player"]; ans = dict(r["results"])
        for extra in a.answers_test or []:  # held-out answers kept in their own files
            t = json.loads(Path(extra).read_text())
            if t["player"] == pl:
                ans.update(t["results"])
        def rows(sets):
            out = []
            for name, seed, d, mon, key in cases(sets):
                if name in ans and "answers" in ans[name]:
                    out.append({"name": name, "seed": seed, "monitor": mon, "answers": ans[name]["answers"],
                                "y": 1 if key["disposition"] == "fraud" else 0, "key": key})
            return out
        fit, test = rows(a.fit), rows(a.test)
        m = QBS().fit(fit, [x["seed"] for x in fit])
        p = m.predict(test)
        print(f"\n== {pl}: fit on {len(fit)} alerts ({sum(x['y'] for x in fit)} fraud), held-out test {len(test)} ({sum(x['y'] for x in test)} fraud)")
        print("   forward selection (cross-validated AUC across fit seeds):")
        for q, auc in m.path: print(f"     {auc:.3f}  {q}")
        keys = {x["name"]: x["key"] for x in test}
        def as_results(scores):  # shape leaderboard_metrics expects
            return {n: {"answer": {"disposition": {"fraud": float(s)}, "typology": {"other": 1.0}, "customer_role": {"legitimate": 1.0}}} for n, s in scores.items()}
        qbs_scores = dict(zip([x["name"] for x in test], p))
        if a.dump:
            Path(a.dump).mkdir(parents=True, exist_ok=True)
            Path(a.dump, f"qbs-{pl.split(':')[0]}.json").write_text(json.dumps({"player": "qbs-" + pl, "results": as_results(qbs_scores)}))
        direct = json.loads((Path(a.direct) / f"{pl.split(':')[0]}.json").read_text())["results"]
        direct_scores = {n: v["answer"]["disposition"]["fraud"] for n, v in direct.items() if n in keys}
        from sklearn.metrics import roc_auc_score
        names = [x["name"] for x in test]; y = [keys[n]["disposition"] == "fraud" for n in names]
        for label, sc in (("QBS", qbs_scores), ("direct", direct_scores)):
            lm = leaderboard_metrics(keys, as_results(sc))
            order = sorted(names, key=lambda n: sc[n])  # safest first
            bad = lambda share: sum(keys[n]["disposition"] == "fraud" for n in order[:int(len(order) * share)]) / int(len(order) * share)
            print(f"   {label:6s} AUC {roc_auc_score(y, [sc[n] for n in names]):.3f}  PR-AUC@10% {lm['pr_auc']:.3f}  prec@80%rec {lm['precision_at_80_recall']:.3f}  logloss@10% {lm['log_loss']:.3f}"
                  f"  | fraud share among safest 10/25/50% (sample mix): {bad(.10):.1%} / {bad(.25):.1%} / {bad(.50):.1%}")

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("ask"); s.add_argument("cases", nargs="+"); s.add_argument("--player", required=True); s.add_argument("--out", required=True); s.add_argument("--parallel", type=int, default=6)
s = sub.add_parser("evaluate"); s.add_argument("--fit", nargs="+", required=True); s.add_argument("--test", nargs="+", required=True); s.add_argument("--answers", nargs="+", required=True); s.add_argument("--direct", required=True); s.add_argument("--dump"); s.add_argument("--answers-test", nargs="+")
a = ap.parse_args(); {"ask": cmd_ask, "evaluate": cmd_eval}[a.cmd](a)
