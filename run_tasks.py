"""Fraud Arena historical investigation tasks: alert review (A) and incident investigation (B).

    python run_tasks.py build --split dev --seed 11 --out cases/dev-s11
    python run_tasks.py run --task alert --cases cases/dev-s11 --model openrouter:<model id> --n 12 --trials 3 --out runs/dev-s11-m
    python run_tasks.py run --task incident --cases cases/dev-s11 --model baseline --out runs/dev-s11-baseline
    python run_tasks.py score --runs runs/dev-s11-m runs/dev-s11-baseline
    python run_tasks.py leaderboard --runs runs/* --out site-data/leaderboard.json
    python run_tasks.py publish --split dev --cases cases/dev-s11 cases/dev-s12 --runs runs/* --examples dev-s11/A001 --out site-data
    python run_tasks.py export-harbor --cases cases/dev-s11 --out harbor/dev-s11 [--ids A001,B002]

Seed sets (splits) are in task-sets.json. A model player (`openrouter:<model id>`) is mini-SWE-agent over OpenRouter
with its own `casefile` command; `claude:<model>` runs a headless Claude Code session instead (for smoke tests),
restricted to Bash(casefile:*) and Write/Edit/Read under drafts/. `--model baseline` runs the non-LLM heuristic.
See docs/tasks.md.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
TASK_SETS = REPO / "task-sets.json"

PROMPT = """{brief}
## How to work
Only `casefile` commands work here (one command per call; no pipes, redirects or other programs).
Start with `casefile tables` and `casefile customer ...`, then investigate with `casefile query`, `casefile links` and
`casefile customer`. {depth} You have about {max_turns} commands in total: keep a few for the answer. {write}
`casefile submit drafts/answer.json`; fix and resubmit if it reports problems. {finish}"""

# The answer-writing and finishing lines differ only in mechanics between the two harnesses.
WRITE = {"claude": "Write your answer with the Write tool to drafts/answer.json (only files under drafts/ can be "
                   "written) and run",
         "msa": "Write your answer to drafts/answer.json with a heredoc (only files under drafts/ can be written) and "
                "run"}
FINISH = {"claude": "When it is submitted, stop and reply with one line: your verdict.",
          "msa": "When it is submitted, end your turn."}


# How deep to dig (prompt v2; v1 had no such line and players stopped after 2-6 commands). The same words for fraud
# and benign cases.
DEPTH = {
    "alert": "Check the alert against the customer's whole history (logins and devices, profile changes, money in "
             "and out, disputes and claims, calls) and the customer's rarer links before you decide; a careful review "
             "takes 5-15 commands.",
    "incident": "Look at every seed customer, follow their rarer links (and the links of the customers you add), and "
                "check what the linked customers did, before you decide; a careful investigation takes 15-40 commands.",
}

TASK_TEXT = {
    "alert": "Alert review: one customer flagged by a bank monitor. Decide fraud or not, the typology, the customer's "
             "role and a calibrated confidence, from the bank's own tables as of the case time.",
    "incident": "Incident investigation: a detector flags a few customers sharing a key or a monitor spike. Decide "
                "whether it is one coordinated fraud or a benign cluster, and find every customer in it with their "
                "roles.",
}


def task_sets():
    return json.loads(TASK_SETS.read_text())


def write_wrapper(bin_dir, work):
    bin_dir.mkdir(parents=True, exist_ok=True)
    w = bin_dir / "casefile"
    w.write_text(f'#!/bin/sh\nPYTHONPATH="{REPO}" exec "{sys.executable}" -m arena.tasks.cli "{work}" "$@"\n')
    w.chmod(0o700)


from arena.player import is_player, play_turn  # noqa: E402


def claude_turn(model, work, bin_dir, prompt, log_path, max_turns, timeout=1200):
    model = model.split(":", 1)[1] if model.startswith("claude:") else model
    empty_mcp = work.parent / f"{work.name}-mcp.json"
    empty_mcp.write_text('{"mcpServers": {}}')
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", "Bash,Write,Read",
           "--allowedTools", "Bash(casefile:*)", "Edit(drafts/**)", "Read(drafts/**)",
           "--strict-mcp-config", "--mcp-config", str(empty_mcp), "--setting-sources", "project",
           "--no-session-persistence", "--max-turns", str(max_turns), "--output-format", "json"]
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    token_file = Path(os.environ.get("CLAUDE_TOKEN_FILE", Path.home() / ".claude-token"))
    if not env.get("CLAUDE_CODE_OAUTH_TOKEN") and not env.get("ANTHROPIC_API_KEY") and token_file.exists():
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token_file.read_text().strip()  # the launcher's login, when not exported
    started = time.time()
    try:
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
        out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"result": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        out = {"result": "timeout", "is_error": True}
    Path(log_path).write_text(json.dumps(out, indent=1))
    return {"seconds": round(time.time() - started), "turns": out.get("num_turns"),
            "cost_usd": out.get("total_cost_usd"), "error": bool(out.get("is_error")),
            "summary": str(out.get("result", ""))[:300]}


def cmd_build(a):
    from arena.tasks import export, world
    from arena.tasks.build import build
    sets = task_sets()
    if a.seed not in sets["sets"].get(a.split, []) and not a.allow_any_seed:
        sys.exit(f"seed {a.seed} is not in set '{a.split}' ({TASK_SETS.name}); pass --allow-any-seed for a scratch build")
    roots, days = a.roots or sets["world"]["roots"], a.days or sets["world"]["days"]
    c = sets["cases"]
    out = Path(a.out)
    if (out / "truth" / "world" / "events.parquet").exists() and not a.resimulate:
        print(f"reusing the simulation in {out}/truth (pass --resimulate to run it again)")
        meta = json.loads((out / "world.json").read_text())
        if (meta["seed"], meta["roots"], meta["days"]) != (a.seed, roots, days):
            sys.exit(f"{out} holds seed/roots/days {meta['seed']}/{meta['roots']}/{meta['days']}: use --resimulate")
    else:
        print(f"simulating seed {a.seed}, {roots} roots, {days} days")
        world.simulate(out, seed=a.seed, roots=roots, days=days)
    world.materialize(out)
    export.rewrite_bank(out, a.seed)
    summary = build(out, n_alerts=a.alerts or c["alerts"], n_incidents=a.incidents or c["incidents"],
                    n_fp=a.fp if a.fp is not None else c["fp"], fraud_share=c["fraud_share"], seed=a.seed,
                    split=a.split, set_seeds=sets["sets"].get(a.split, []),
                    builds_per_group=sets["cases"].get("builds_per_group"))
    print(json.dumps(summary, indent=1))


def prepare(cases, task, case_id, work):
    """Copy a case workspace to `work` (fresh for every trial); point it at the case set's bank tables."""
    src = cases / task / case_id
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work)
    case = json.loads((work / "case.json").read_text())
    case["bank"] = str((cases / "bank").resolve())
    (work / "case.json").write_text(json.dumps(case, indent=1))
    return work


def cmd_run(a):
    cases, out = Path(a.cases).resolve(), Path(a.out).resolve()
    index = json.loads((cases / "index.json").read_text())
    ids = sorted(p.name for p in (cases / a.task).iterdir() if p.is_dir())
    if a.ids:
        ids = [i for i in ids if i in a.ids.split(",")]
    if a.n:
        ids = ids[:a.n]
    (out / a.task).mkdir(parents=True, exist_ok=True)
    max_turns = a.max_turns or (30 if a.task == "alert" else 60)
    jobs = [(cid, t) for cid in ids for t in range(1, a.trials + 1)]
    results = {}

    def one(job):
        cid, t = job
        work = prepare(cases, a.task, cid, out / a.task / cid / f"trial-{t}")
        if a.model == "baseline":
            from arena.tasks import baseline
            t0 = time.time()
            ans = baseline.alert_answer(work) if a.task == "alert" else baseline.incident_answer(work)
            (work / "answer.json").write_text(json.dumps(ans, indent=1, default=str))
            return job, {"seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
        bin_dir = out / a.task / cid / f"bin-{t}"
        write_wrapper(bin_dir, work)
        h = "msa" if is_player(a.model) else "claude"
        prompt = PROMPT.format(brief=(work / "brief.md").read_text(), max_turns=max_turns, depth=DEPTH[a.task],
                               write=WRITE[h], finish=FINISH[h])
        if h == "msa":  # the benchmark harness: mini-SWE-agent over OpenRouter
            r = play_turn(a.model, work, bin_dir, "casefile", prompt, out / a.task / cid / f"trial-{t}.traj.json",
                          step_limit=max_turns, require="casefile submit")
        else:  # Claude Code, for smoke tests
            r = claude_turn(a.model, work, bin_dir, prompt, out / a.task / cid / f"trial-{t}.claude.json", max_turns)
        r["submitted"] = (work / "answer.json").exists()
        r["commands"] = sum(1 for _ in open(work / "log.jsonl")) if (work / "log.jsonl").exists() else 0
        print(f"  {cid} trial {t}: {r['seconds']}s turns={r['turns']} cost=${r['cost_usd']} "
              f"submitted={r['submitted']} commands={r['commands']}", flush=True)
        return job, r

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for (cid, t), r in ex.map(one, jobs):
            results.setdefault(cid, {})[str(t)] = r
    meta_path = out / a.task / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model": a.model, "task": a.task, "cases": str(cases), "case_set": cases.name,
                 "split": index.get("split"), "seed": index.get("seed"), "max_turns": max_turns,
                 "trials": a.trials})
    meta.setdefault("results", {}).update(results)
    meta_path.write_text(json.dumps(meta, indent=1))
    cost = sum((r.get("cost_usd") or 0) for rs in results.values() for r in rs.values())
    print(f"ran {len(ids)} {a.task} cases x {a.trials} trials with {a.model}; cost ${cost:.2f}")


def score_run(run, task, keys=None):
    """Pooled metrics (every trial one row) plus the k-trial summary for one run folder and task."""
    from arena.tasks.score import (alert_case_score, answers_flat, collect_trials, incident_case_score,
                                   k_trial_summary, load_keys, score_alerts, score_incidents)
    meta = json.loads((Path(run) / task / "run.json").read_text())
    keys = keys or load_keys(meta["cases"])
    trials = collect_trials(run, task)
    flat = answers_flat(trials)
    sc = score_alerts(keys, flat) if task == "alert" else score_incidents(keys, flat)
    fn = alert_case_score if task == "alert" else incident_case_score
    per_case = {c: [fn(keys[c], a) for a in ts] for c, ts in trials.items()}
    sc["k_trials"] = k_trial_summary(per_case)
    sc.update({"model": meta.get("model"), "split": meta.get("split"), "case_set": meta.get("case_set"),
               "cost_usd": round(sum((r.get("cost_usd") or 0) for rs in meta["results"].values()
                                     for r in (rs.values() if isinstance(rs, dict) and "seconds" not in rs else [rs])), 2)})
    return sc, per_case


def cmd_score(a):
    for run in a.runs:
        for task in ("alert", "incident"):
            if not (Path(run) / task / "run.json").exists():
                continue
            sc, _ = score_run(run, task)
            (Path(run) / task / "score.json").write_text(json.dumps(sc, indent=1, default=str))
            head = {k: v for k, v in sc.items() if k != "rows"}
            head["k_trials"] = {k: v for k, v in sc["k_trials"].items() if k != "per_case"}
            print(f"== {run} [{task}]\n{json.dumps(head, indent=1, default=str)}")


def cmd_leaderboard(a):
    """Per model, split and task family: mean@K, worst@K, best@K, pass^K (verdict), 95% bootstrap CI over cases, plus
    pooled metrics; cases from several case sets of one split are pooled (case key: set/case)."""
    from arena.tasks.score import k_trial_summary
    groups = {}
    for run in a.runs:
        for task in ("alert", "incident"):
            if not (Path(run) / task / "run.json").exists():
                continue
            sc, per_case = score_run(run, task)
            g = groups.setdefault((sc["model"], sc["split"] or "unknown", task), {"per_case": {}, "cost": 0.0,
                                                                                 "runs": [], "pooled": []})
            g["per_case"].update({f"{sc['case_set']}/{c}": v for c, v in per_case.items()})
            g["cost"] += sc["cost_usd"]
            g["runs"].append(str(run))
            g["pooled"].append({k: v for k, v in sc.items() if k not in ("rows", "k_trials")})
    board = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": TASK_TEXT, "models": {}}
    for (model, split, task), g in sorted(groups.items()):
        s = k_trial_summary(g["per_case"])
        s.pop("per_case", None)
        s.update({"cost_usd": round(g["cost"], 2), "case_sets": sorted({r["case_set"] for r in g["pooled"]}),
                  "pooled": g["pooled"]})
        board["models"].setdefault(model, {}).setdefault(split, {})[task] = s
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(board, indent=1, default=str))
    for model, splits in board["models"].items():
        for split, tasks in splits.items():
            for task, s in tasks.items():
                print(f"{model:10} {split:16} {task:8} n={s['n_cases']:3} k={s['k']} mean@k={s['mean_at_k']:.3f} "
                      f"worst={s['worst_at_k']:.3f} best={s['best_at_k']:.3f} pass^k={s['pass_hat_k_verdict']:.2f} "
                      f"ci95={s['ci95']} cost=${s['cost_usd']}")
    print(f"wrote {a.out}")


def public_key(key):
    """An answer key as published: the grading fields, without the simulator's internal involvement records."""
    drop = {"involvements", "same_group_other", "not_yet_visible", "benign_holders"}
    return {k: v for k, v in key.items() if k not in drop}


def trace(work, claude_json):
    """A trial's trace for the site: every casefile command (args, ok, seconds), the final answer, turns and cost."""
    log = [json.loads(line) for line in open(work / "log.jsonl")] if (work / "log.jsonl").exists() else []
    cj = json.loads(claude_json.read_text()) if claude_json.exists() else {}
    return {"commands": [{"args": e["args"], "ok": e.get("ok"), "error": e.get("error"), "seconds": e.get("seconds")}
                         for e in log],
            "answer": json.loads((work / "answer.json").read_text()) if (work / "answer.json").exists() else None,
            "final_message": cj.get("result"), "turns": cj.get("num_turns"), "cost_usd": cj.get("total_cost_usd")}


def cmd_publish(a):
    """Per task family, site-data/<task>.json: description, every case's inputs (case.json + brief), answer keys
    (dev: always; other splits only with --with-keys, i.e. after their season), and example traces from runs."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    examples = set(filter(None, a.examples.split(",")))
    runs = []
    for run in a.runs or []:
        for task in ("alert", "incident"):
            p = Path(run) / task / "run.json"
            if p.exists():
                runs.append((Path(run), task, json.loads(p.read_text())))
    for task in ("alert", "incident"):
        page = {"task": task, "description": TASK_TEXT[task], "split": a.split, "case_sets": [], "cases": []}
        for cs in a.cases:
            cs = Path(cs)
            index = json.loads((cs / "index.json").read_text())
            if index.get("split") != a.split:
                sys.exit(f"refusing: {cs} is split '{index.get('split')}', not '{a.split}'")
            page["case_sets"].append({"name": cs.name, "seed": index["seed"]})
            for entry in index[task]:
                cid = entry["case_id"]
                ws = cs / task / cid
                case = json.loads((ws / "case.json").read_text())
                case.pop("bank", None)
                item = {"id": f"{cs.name}/{cid}", "case": case, "brief": (ws / "brief.md").read_text()}
                if a.split == "dev" or a.with_keys or f"{cs.name}/{cid}" in examples:
                    item["answer_key"] = public_key(json.loads((cs / "truth" / "keys" / f"{cid}.json").read_text()))
                if f"{cs.name}/{cid}" in examples:
                    item["traces"] = []
                    for run, rtask, meta in runs:
                        if rtask != task or meta.get("case_set") != cs.name or not (run / task / cid).exists():
                            continue
                        for t in sorted((run / task / cid).glob("trial-*")):
                            if t.is_dir():
                                item["traces"].append({"model": meta["model"], "trial": t.name,
                                                       **trace(t, run / task / cid / f"{t.name}.claude.json")})
                page["cases"].append(item)
        (out / f"{task}.json").write_text(json.dumps(page, indent=1, default=str))
        print(f"wrote {out / (task + '.json')}: {len(page['cases'])} cases, "
              f"{sum('traces' in c for c in page['cases'])} examples with traces")
    check = os.environ.get("FA_PUBLISH_CHECK")  # optional pre-publish check command, given the output folder
    if check and subprocess.run([*check.split(), str(out)]).returncode:
        sys.exit("publish output failed the pre-publish check (see above)")


def cmd_export_harbor(a):
    from arena.tasks.harbor import export_case
    cases = Path(a.cases).resolve()
    index = json.loads((cases / "index.json").read_text())
    out = Path(a.out)
    want = set(filter(None, a.ids.split(",")))
    n = 0
    for task in ("alert", "incident"):
        for entry in index[task]:
            if want and entry["case_id"] not in want:
                continue
            export_case(cases, task, entry["case_id"], out, index)
            n += 1
    print(f"wrote {n} Harbor task folders under {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True, help="a seed set in task-sets.json (dev, official-...)")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--roots", type=int, default=0)
    b.add_argument("--days", type=int, default=0)
    b.add_argument("--alerts", type=int, default=0)
    b.add_argument("--incidents", type=int, default=0)
    b.add_argument("--fp", type=int, default=None)
    b.add_argument("--resimulate", action="store_true")
    b.add_argument("--allow-any-seed", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--task", choices=["alert", "incident"], required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--model", required=True, help="openrouter:<model id> (the benchmark harness, mini-SWE-agent), claude:<model> or a bare Claude model name (Claude Code, for smoke tests), or baseline")
    r.add_argument("--n", type=int, default=0, help="run the first N cases (default all)")
    r.add_argument("--ids", default="", help="comma-separated case ids")
    r.add_argument("--trials", type=int, default=1, help="K: independent trials per case (fresh workspace each)")
    r.add_argument("--out", required=True)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--max-turns", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--runs", nargs="+", required=True)
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--runs", nargs="+", required=True)
    lb.add_argument("--out", default="site-data/leaderboard.json")
    p = sub.add_parser("publish")
    p.add_argument("--split", required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--runs", nargs="*")
    p.add_argument("--examples", default="", help="comma-separated SET/CASE ids to publish with traces")
    p.add_argument("--with-keys", action="store_true", help="publish answer keys of a non-dev split (after its season)")
    p.add_argument("--out", default="site-data")
    h = sub.add_parser("export-harbor")
    h.add_argument("--cases", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--ids", default="")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "score": cmd_score, "leaderboard": cmd_leaderboard, "publish": cmd_publish,
     "export-harbor": cmd_export_harbor}[a.cmd](a)


if __name__ == "__main__":
    main()
