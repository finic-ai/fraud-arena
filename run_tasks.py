"""Fraud Arena historical investigation tasks: alert review (A) and incident investigation (B).

    python run_tasks.py build --seed 11 --roots 8000 --days 120 --out cases/pilot [--alerts 40 --incidents 12 --fp 3]
    python run_tasks.py run --task alert --cases cases/pilot --model sonnet --n 12 --out runs/pilot-sonnet
    python run_tasks.py run --task incident --cases cases/pilot --model baseline --out runs/pilot-baseline
    python run_tasks.py score --cases cases/pilot --runs runs/pilot-sonnet runs/pilot-baseline

A Claude player is a headless Claude Code session per case, restricted to its own `casefile` command (Bash(casefile:*)),
Write/Edit/Read under drafts/, no MCP servers, no user settings or hooks, and no session saved. `--model baseline`
runs the non-LLM heuristic instead. See docs/tasks.md.
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

PROMPT = """{brief}
## How to work
Only `casefile` commands work here (run them with Bash, one command per call; no pipes, redirects or other programs).
Start with `casefile tables` and `casefile customer ...`, then investigate with `casefile query`, `casefile links` and
`casefile customer`. {depth} You have about {max_turns} tool calls in total: keep a few for the answer. Write your answer with
the Write tool to drafts/answer.json (only files under drafts/ can be written) and run
`casefile submit drafts/answer.json`; fix and resubmit if it reports problems. When it is submitted, stop and reply with
one line: your verdict."""


# How deep to dig (prompt v2; v1 had no such line and players stopped after 2-6 commands). The same words for fraud
# and benign cases.
DEPTH = {
    "alert": "Check the alert against the customer's whole history (logins and devices, profile changes, money in "
             "and out, disputes and claims, calls) and the customer's rarer links before you decide; a careful review "
             "takes 5-15 commands.",
    "incident": "Look at every seed customer, follow their rarer links (and the links of the customers you add), and "
                "check what the linked customers did, before you decide; a careful investigation takes 15-40 commands.",
}


def write_wrapper(bin_dir, work):
    bin_dir.mkdir(parents=True, exist_ok=True)
    w = bin_dir / "casefile"
    w.write_text(f'#!/bin/sh\nPYTHONPATH="{REPO}" exec "{sys.executable}" -m arena.tasks.cli "{work}" "$@"\n')
    w.chmod(0o700)


def claude_turn(model, work, bin_dir, prompt, log_path, max_turns, timeout=1200):
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
    from arena.tasks import world
    from arena.tasks.build import build
    out = Path(a.out)
    if (out / "truth" / "world" / "events.parquet").exists() and not a.resimulate:
        print(f"reusing the simulation in {out}/truth (pass --resimulate to run it again)")
        meta = json.loads((out / "world.json").read_text())
        if (meta["seed"], meta["roots"], meta["days"]) != (a.seed, a.roots, a.days):
            sys.exit(f"{out} holds seed/roots/days {meta['seed']}/{meta['roots']}/{meta['days']}: use --resimulate")
    else:
        print(f"simulating seed {a.seed}, {a.roots} roots, {a.days} days")
        world.simulate(out, seed=a.seed, roots=a.roots, days=a.days)
    world.materialize(out)
    summary = build(out, n_alerts=a.alerts, n_incidents=a.incidents, n_fp=a.fp, fraud_share=a.fraud_share,
                    seed=a.seed)
    print(json.dumps(summary, indent=1))


def prepare(cases, task, case_id, out):
    """Copy a case workspace into the run folder; point it at the case set's bank tables."""
    src = cases / task / case_id
    work = out / task / case_id
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work)
    case = json.loads((work / "case.json").read_text())
    case["bank"] = str((cases / "bank").resolve())
    (work / "case.json").write_text(json.dumps(case, indent=1))
    return work


def cmd_run(a):
    cases, out = Path(a.cases).resolve(), Path(a.out).resolve()
    ids = sorted(p.name for p in (cases / a.task).iterdir() if p.is_dir())
    if a.ids:
        ids = [i for i in ids if i in a.ids.split(",")]
    if a.n:
        ids = ids[:a.n]
    (out / a.task).mkdir(parents=True, exist_ok=True)
    max_turns = a.max_turns or (30 if a.task == "alert" else 60)
    results = {}

    def one(cid):
        work = prepare(cases, a.task, cid, out)
        if a.model == "baseline":
            from arena.tasks import baseline
            t0 = time.time()
            ans = baseline.alert_answer(work) if a.task == "alert" else baseline.incident_answer(work)
            (work / "answer.json").write_text(json.dumps(ans, indent=1, default=str))
            return cid, {"seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
        bin_dir = out / a.task / f"bin-{cid}"
        write_wrapper(bin_dir, work)
        prompt = PROMPT.format(brief=(work / "brief.md").read_text(), max_turns=max_turns, depth=DEPTH[a.task])
        r = claude_turn(a.model, work, bin_dir, prompt, out / a.task / f"{cid}.claude.json", max_turns)
        r["submitted"] = (work / "answer.json").exists()
        r["commands"] = sum(1 for _ in open(work / "log.jsonl")) if (work / "log.jsonl").exists() else 0
        print(f"  {cid}: {r['seconds']}s turns={r['turns']} cost=${r['cost_usd']} submitted={r['submitted']} "
              f"commands={r['commands']}", flush=True)
        return cid, r

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for cid, r in ex.map(one, ids):
            results[cid] = r
    meta_path = out / a.task / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model": a.model, "cases": str(cases), "max_turns": max_turns})
    meta.setdefault("results", {}).update(results)
    meta_path.write_text(json.dumps(meta, indent=1))
    cost = sum((r.get("cost_usd") or 0) for r in results.values())
    print(f"ran {len(results)} {a.task} cases with {a.model}; cost ${cost:.2f}")


def cmd_score(a):
    from arena.tasks.score import collect_answers, load_keys, score_alerts, score_incidents
    keys = load_keys(a.cases)
    report = {}
    for run in a.runs:
        report[run] = {}
        for task in ("alert", "incident"):
            if not (Path(run) / task).exists():
                continue
            answers = collect_answers(run, task)
            if a.only_common:
                answers = {k: v for k, v in answers.items() if k in a.only_common.split(",")}
            sc = score_alerts(keys, answers) if task == "alert" else score_incidents(keys, answers)
            meta_path = Path(run) / task / "run.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                sc["model"] = meta.get("model")
                sc["cost_usd"] = round(sum((r.get("cost_usd") or 0) for r in meta["results"].values()), 2)
            report[run][task] = sc
            (Path(run) / task / "score.json").write_text(json.dumps(sc, indent=1, default=str))
    for run, tasks in report.items():
        for task, sc in tasks.items():
            head = {k: v for k, v in sc.items() if k != "rows"}
            print(f"== {run} [{task}]\n{json.dumps(head, indent=1)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--roots", type=int, default=8000)
    b.add_argument("--days", type=int, default=120)
    b.add_argument("--out", required=True)
    b.add_argument("--alerts", type=int, default=40)
    b.add_argument("--incidents", type=int, default=12)
    b.add_argument("--fp", type=int, default=3)
    b.add_argument("--fraud-share", type=float, default=0.45)
    b.add_argument("--resimulate", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--task", choices=["alert", "incident"], required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--model", default="sonnet", help="a Claude model (sonnet, opus, ...) or 'baseline'")
    r.add_argument("--n", type=int, default=0, help="run the first N cases (default all)")
    r.add_argument("--ids", default="", help="comma-separated case ids")
    r.add_argument("--out", required=True)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--max-turns", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--cases", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--only-common", default="", help="comma-separated case ids to restrict to")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
