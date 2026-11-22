"""Play one Fraud Arena 1v1 match on Fraud World.

    python run_fw_match.py --defender claude:sonnet --attacker claude:opus --weeks 4 --out runs/NAME
    (a player is claude:<model> for a Claude Code session, or builtin for the bank's rules / the rings' scripts)

Each Claude Code player runs on this machine with your Claude Code login, restricted to one tool: its own `arena`
command (Bash(arena:*)), with no MCP servers, no user settings or hooks, and no session saved.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arena.fw.match import Match

REPO = Path(__file__).resolve().parent

BRIEFS = {
    "defender": """# You are the DEFENDER at Brindlemark Bank (Fraud Arena)

Brindlemark is a digital bank in a simulated economy (Runmere's Fraud World) with thousands of customers, merchants,
other banks, a crypto exchange and ten fraud rings. Another AI runs the fraud syndicate (all ten rings) against you and
adapts each week.

You control the bank's automated decisions: card authorizations, logins, account openings, payments and more (`arena
policy help`). Each week you may deploy a new policy: rules (and optional scoring models) that the bank applies to
every decision of that kind. Decisions you write no rules for keep the bank's built-in rules.

Scoring, per week, against a reference run of the same world with the bank's built-in rules and the rings' normal
behaviour: fraud losses you prevent at Brindlemark, minus the extra friction you cause good customers (declines cost
$3 + 10% of the amount, step-ups $0.50, holds $2, reviews $15, alerts $8, blocks $25, account closures $200).

Your data: the bank's own tables (`arena tables`, `arena query`): customers, accounts, card authorizations, logins,
transfers, applications, disputes, alerts and more, as the bank sees them. Fraud labels arrive the way they do in a
bank: through disputes, chargebacks and reports, late and incomplete. You never see the syndicate's plans.
""",
    "attacker": """# You are the ATTACKER: the fraud syndicate (Fraud Arena)

This is a security benchmark in a simulated economy (Runmere's Fraud World). You choose from a fixed menu of strategic
settings for ten fraud rings; you write no messages, scripts or real-world instructions, and nothing real is involved.

The rings run 22 schemes between them (account takeovers by phishing and SIM swap, card testing and card-not-present
fraud, synthetic identities, promo farming, refund services, romance and pig-butchering scams, business email
compromise, payroll diversion, mule flows and more) against two banks, a credit union, a crypto exchange, a brokerage
and merchants. Each week you set the syndicate's plan (`arena policy help`): how much each ring works, which schemes it
favours, where it attacks, timing, devices, proxy exits, identities and mule recruiting.

Another AI defends Brindlemark Bank and adapts its rules each week. Scoring, per week, against a reference run of the
same world where the rings follow their normal habits: what the syndicate takes from BRINDLEMARK (mule pass-through
counts at 10%), minus what the rings would have taken there anyway. Money taken elsewhere doesn't score. The syndicate
works within an effort budget (see `arena policy help`): pushing some rings or schemes harder means easing off others.

Your data: your own results only (`arena results`): what each ring took from each institution each week, and how many
of each scheme you have run. You never see the bank's rules, alerts or data.
""",
}

TASK = """{brief}
## This turn
It is your turn before week {week} of {weeks}. You have about {max_turns} commands. Start with `arena status`, then
`arena policy help`. Study your data, then write your policy as JSON with the Write tool to a NEW file drafts/week{week}.json
(only files under drafts/ can be written) and run `arena policy set drafts/week{week}.json`; fix and retry if it
reports problems. `arena policy show` prints the policy now in force, to start from.
Keep short notes for next week with `arena note "..."`. Only `arena` commands work here.
When your policy is saved, stop and reply with one line summarizing what you changed and why."""


def write_wrapper(bin_dir, work):
    bin_dir.mkdir(parents=True, exist_ok=True)
    w = bin_dir / "arena"
    w.write_text(f'#!/bin/sh\nPYTHONPATH="{REPO}" exec "{sys.executable}" -m arena.fw.cli "{work}" "$@"\n')
    w.chmod(0o755)


def claude_turn(model, work, bin_dir, prompt, log_path, max_turns):
    empty_mcp = work.parent / f"{work.name}-mcp.json"
    empty_mcp.write_text('{"mcpServers": {}}')
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", "Bash,Write,Read",
           "--allowedTools", "Bash(arena:*)", "Edit(drafts/**)", "Read(drafts/**)",
           "--strict-mcp-config", "--mcp-config", str(empty_mcp), "--setting-sources", "project",
           "--no-session-persistence", "--max-turns", str(max_turns), "--output-format", "json"]
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    started = time.time()
    try:
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=1200)
        out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"result": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        out = {"result": "timeout", "is_error": True}
    Path(log_path).write_text(json.dumps(out, indent=1))
    return {"seconds": round(time.time() - started), "turns": out.get("num_turns"),
            "cost_usd": out.get("total_cost_usd"), "error": bool(out.get("is_error")),
            "summary": str(out.get("result", ""))[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defender", default="claude:sonnet")
    ap.add_argument("--attacker", default="claude:opus")
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--roots", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    m = Match(out, seed=args.seed, roots=args.roots, warmup_days=args.warmup)
    players = {"defender": args.defender, "attacker": args.attacker}
    works = {}
    for role, who in players.items():
        work = out / role
        work.mkdir()
        (work / "role").write_text(role)
        (work / "brief.md").write_text(BRIEFS[role])
        (work / "notes.md").write_text("")
        (work / "data").mkdir()
        (work / "drafts").mkdir()
        if role == "defender":
            (work / "decisions.json").write_text(json.dumps(m.decisions, default=str))
            (work / "policy.json").write_text(json.dumps({"format": 1, "rules": []}, indent=1))
        else:
            (work / "menu.json").write_text(json.dumps(m.menu))
            (work / "policy.json").write_text(json.dumps({"all": {}, "rings": {}}, indent=1))
        write_wrapper(out / f"bin-{role}", work)
        works[role] = work
    (out / "logs").mkdir()

    weeks, scores = [], {"defender": [], "attacker": []}
    for wk in range(args.weeks):
        # Views as of the start of this week.
        m.defender_view(works["defender"] / "data")
        att_view = m.attacker_view(weeks, weeks[-1]["_truth"]) if weeks else {"results": [], "schemes_started_to_date": {}}
        (works["attacker"] / "data" / "results.json").write_text(json.dumps(att_view, indent=1))
        for role, work in works.items():
            lines = [f"# Week {wk + 1} of {args.weeks} is next.", ""]
            for i, s in enumerate(scores[role], 1):
                lines.append(f"week {i}: score {s:+.0f}")
            if not scores[role]:
                lines.append("No weeks played yet: the data covers the bank's history before the match." if role == "defender"
                             else "No weeks played yet: you have no results. Set an opening plan.")
            err = weeks[-1]["deploy_error"] if (weeks and role == "defender") else None
            if err:
                lines += ["", f"YOUR LAST POLICY FAILED TO DEPLOY (the previous one stayed): {err}"]
            lines += ["", "Current policy:", (work / "policy.json").read_text()]
            (work / "status.md").write_text("\n".join(lines) + "\n")

        def turn(role):
            who = players[role]
            if not who.startswith("claude:"):
                return {"player": who, "skipped": True}
            prompt = TASK.format(brief=BRIEFS[role], week=wk + 1, weeks=args.weeks, max_turns=args.max_turns)
            return claude_turn(who.split(":", 1)[1], works[role], out / f"bin-{role}", prompt,
                               out / "logs" / f"{role}-week{wk + 1}.json", args.max_turns)

        with ThreadPoolExecutor(max_workers=2) as pool:
            turns = dict(zip(players, pool.map(turn, players)))

        deploy_error = None
        if players["defender"].startswith("claude:"):
            deploy_error = m.deploy(json.loads((works["defender"] / "policy.json").read_text()))
        plan_problems = []
        if players["attacker"].startswith("claude:"):
            from arena.fw.cli import budget_problems
            plan = json.loads((works["attacker"] / "policy.json").read_text())
            plan_problems = budget_problems(plan, m.menu)
            if plan_problems:  # written around the checker: keep last week's plan
                plan = weeks[-1]["policies"]["attacker"] if weeks else {}
                (works["attacker"] / "policy.json").write_text(json.dumps(plan, indent=1))
            plan_problems += m.set_plan(plan)
        res, truth, _ = m.play_week()
        scores["defender"].append(res["defender"]["score"])
        scores["attacker"].append(res["attacker"]["score"])
        deploy_error = deploy_error or (f"your policy stopped the bank's systems mid-week and was rolled back: {m.runtime_error}"
                                        if m.runtime_error else None)
        weeks.append({**res, "_truth": str(truth), "turns": turns, "deploy_error": deploy_error,
                      "plan_problems": plan_problems,
                      "policies": {r: json.loads((w / "policy.json").read_text()) for r, w in works.items()}})
        print(f"week {wk + 1}: defender {res['defender']['score']:+.0f}  attacker {res['attacker']['score']:+.0f}  "
              + "  ".join(f"{r}: {t.get('turns')} turns {t.get('seconds')}s" for r, t in turns.items() if not t.get("skipped"))
              + (f"  DEPLOY ERROR: {deploy_error[:120]}" if deploy_error else ""), flush=True)

    summary = {"players": players, "weeks": args.weeks, "total": {r: round(sum(v), 1) for r, v in scores.items()},
               "per_week": scores}
    (out / "result.json").write_text(json.dumps({**summary, "detail": [{k: v for k, v in w.items() if k != "_truth"}
                                                                        for w in weeks]}, indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
