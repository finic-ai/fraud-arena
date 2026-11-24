"""Play one Fraud Arena 1v1 match on Fraud World.

    python run_fw_match.py --defender claude:sonnet --attacker claude:opus --weeks 4 --out runs/NAME
    (a player is openrouter:<model id> for the benchmark harness: mini-SWE-agent over OpenRouter, any model;
    claude:<model> for a Claude Code session, for smoke tests; builtin for the bank's rules / the rings' scripts.
    Fraud Strategy (the defender task) is rules-only: policies are rules on the decisions' features, no trained models.)

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

from arena.fw.match import NATIVE_CASES, Match
from arena.player import is_player, play_turn

REPO = Path(__file__).resolve().parent

# Lists the bank checks at decision time need outside code; the defender task has them only on an engine without
# native case recording (FA_NATIVE_CASES=0).
LISTS_TOOL = "" if NATIVE_CASES else (
    " `arena list set` for lists the bank checks at decision time (e.g. devices, accounts or customers tied to"
    " confirmed fraud), which rules read as outside.list_NAME;")

FRAUD_1V1 = ("One fraud ring operates in this world; another AI runs it and adapts each week. All\n"
             "fraud in the world is the ring's.")
FRAUD_DEFENDER_ONLY = ("The fraud is the world's own: independent fraudsters and organized rings with\n"
                       "their own crews, who switch tactics when one stops working. No AI plays against you.")

BRIEFS = {
    "defender": """# Fraud Strategy: you run Brindlemark Bank's fraud defenses (Fraud Arena)

Brindlemark is a digital bank in a simulated economy (Runmere's Fraud World): thousands of customers, merchants and
other financial institutions. {fraud_world}

You control the bank's automated decisions: card authorizations, logins, account openings, payments and more (`arena
policy help`). Each week you may deploy a new policy: rules that the bank applies to
every decision of that kind. Decisions you write no rules for keep the bank's built-in rules.

Scoring, per week: the same week is replayed with the bank's built-in rules against the very same fraud. Your score
is the fraud losses your policy prevented at Brindlemark compared with that replay, plus 10% of the fraud money it kept
from passing through Brindlemark accounts on its way elsewhere (mule and laundering flow), minus the extra friction you cause
good customers (declines cost $3 + 10% of the amount, step-ups $0.50, holds $2, reviews $15, alerts $8, blocks $25,
account closures $200). The bank's analysts can work 10 reviews a day; reviews beyond that cost $75 each (overtime).
A worse week of fraud doesn't lower your score by itself; only what your policy does counts.

Your data: the bank's own tables (`arena tables`, `arena query`): customers, accounts, card authorizations, logins,
transfers, applications, disputes, alerts and more, as the bank sees them; and cases_<decision> tables with every
decision the bank made (its observed features, including payment memos and notes, the action taken, and fraud_known:
fraud the bank has discovered or its victim has reported so far: most victims report within days, some never do, so
recent fraud is often still unlabeled). The bank has {history} of history before the match. You
never see the ring's plans.

Your tools: `arena rule add/remove/list` for quick rule changes;{lists_tool} `arena query` to test a rule's hits on the
bank's history before deploying it; and `arena policy set` for whole policies (JSON or YAML). Policies are rules only.

What the actions do in this world (they differ a lot in how much fraud they stop):
- block, reject, decline and return_debit stop the event: the payment, login, authorization or debit doesn't go
  through (and an application is refused). They stop all of the fraud they hit and cost the most friction.
- hold and callback mean the bank calls the account holder before the money goes. When the real holder is making the
  payment, including scam victims who were talked into it, they almost always confirm and it goes out (about 95% of
  P2P payments). When someone else controls the account (a takeover), the call reaches the real owner and most such
  payments are stopped (about 92%). Wires and ACH transfers work the same way through their callback.
- step_up asks for a one-time code. It stops an attacker who has only a password, but not a phishing kit that
  captured the code along with the password, nor someone who has taken over the customer's phone number.
- warn shows a scam warning: few scam victims abandon (about 1 in 8). cooling_off delays the payment a day and tells
  the contact details on file: about half of takeovers are stopped (few when the attacker changed the contact
  details) and about a quarter of scam payments.
- review sends an application to the bank's analysts (10 a day, then $75 each).
""",
    "attacker": """# You are the ATTACKER: a fraud ring (Fraud Arena)

This is a security benchmark in a simulated economy (Runmere's Fraud World). You choose from a fixed menu of strategic
settings for your fraud ring; you write no messages, scripts or real-world instructions, and nothing real is involved.

Your ring has ten crews (numbered 1-10, each with its own specialties) running 22 schemes between them: account
takeovers by phishing and SIM swap, card testing and card-not-present fraud, synthetic identities, promo farming,
refund services, romance and pig-butchering scams, business email compromise, payroll diversion, mule flows and more.
Each week you set the ring's plan (`arena policy help`): how much each crew works, which schemes it favours, timing,
devices, proxy exits, identities and mule recruiting. In the plan, "rings" means your crews. Your target is fixed:
every crew attacks Brindlemark wherever its schemes allow, for the whole match.

Another AI defends Brindlemark Bank and adapts its rules each week. Scoring, per week: the same week is replayed with
your crews on their usual habits against the very same defense. Your score is what the ring takes from BRINDLEMARK
(mule pass-through counts at 10%) beyond that replay. A tougher defense doesn't lower your score by itself; only what
your plan does counts. You work within an effort budget
(see `arena policy help`): pushing some crews or schemes harder means easing off others.

Your data: your own results only (`arena results`): what each crew took from each institution each week, and how many
of each scheme you have run. You never see the bank's rules, alerts or data.
""",
}
BRIEFS["defender"] = BRIEFS["defender"].replace("{lists_tool}", LISTS_TOOL)

TASK = """{brief}
## This turn
It is your turn before week {week} of {weeks}. You have about {max_turns} commands. Start with `arena status`, then
`arena policy help`. Study your data, then write your policy as JSON {write} to a NEW file drafts/week{week}.json
(only files under drafts/ can be written) and run `arena policy set drafts/week{week}.json`; fix and retry if it
reports problems. `arena policy show` prints the policy now in force, to start from.
Keep short notes for next week with `arena note "..."`. Only `arena` commands work here.
{finish}"""

# Only the mechanics of writing a file and ending the turn differ between the two harnesses.
WRITE = {"claude": "with the Write tool", "msa": "with a heredoc"}
FINISH = {"claude": "When your policy is saved, stop and reply with one line summarizing what you changed and why.",
          "msa": "When your policy is saved, end your turn."}


def is_ai(who):
    return who.startswith("claude:") or is_player(who)



WORLD_FOR_CHECK = None  # the match's world file, set in main()


def write_wrapper(bin_dir, work):
    bin_dir.mkdir(parents=True, exist_ok=True)
    w = bin_dir / "arena"
    no_lists = "ARENA_NO_LISTS=1 " if NATIVE_CASES else ""
    # ARENA_WORLD lets the tool compile a policy in the engine when it is saved (errors come back in the same turn).
    world = f'ARENA_WORLD="{WORLD_FOR_CHECK}" ' if WORLD_FOR_CHECK else ""
    cache = f'FA_WORLD_CACHE="{os.environ["FA_WORLD_CACHE"]}" ' if os.environ.get("FA_WORLD_CACHE") else ""
    w.write_text(f'#!/bin/sh\n{no_lists}{world}{cache}PYTHONPATH="{REPO}" exec "{sys.executable}" -m arena.fw.cli "{work}" "$@"\n')
    w.chmod(0o700)


def claude_turn(model, work, bin_dir, prompt, log_path, max_turns):
    empty_mcp = work.parent / f"{work.name}-mcp.json"
    empty_mcp.write_text('{"mcpServers": {}}')
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", "Bash,Write,Read",
           "--allowedTools", "Bash(arena:*)", "Edit(drafts/**)", "Read(drafts/**)",
           "--strict-mcp-config", "--mcp-config", str(empty_mcp), "--setting-sources", "project",
           "--no-session-persistence", "--max-turns", str(max_turns), "--output-format", "json"]
    env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY",)}
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
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
    ap.add_argument("--world", help="a world file (default: the 1v1 tier of the hub version in fraud-world.json)")
    ap.add_argument("--roots", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--intensity", type=float, default=3.0, help="ring volume during the match, x their usual")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--defender-only", action="store_true",
                    help="no attacking agent: advance only the candidate and no-defense runs (requires --attacker none)")
    ap.add_argument("--history-cache", help="a folder holding this seed's warm-up (built once, shared by defenders)")
    ap.add_argument("--warm-only", action="store_true",
                    help="build this seed's warm-up into --history-cache and exit (no weeks are played)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.warm_only and not args.history_cache:
        ap.error("--warm-only needs --history-cache")
    if args.defender_only and is_ai(args.attacker):
        ap.error("--defender-only needs a non-agent attacker (e.g. --attacker none)")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    global WORLD_FOR_CHECK
    m = Match(out, world=args.world and Path(args.world), seed=args.seed, roots=args.roots, warmup_days=args.warmup,
              intensity=args.intensity, history_cache=args.history_cache, defender_only=args.defender_only)
    if args.warm_only:
        print(f"warm-up ready: {m.cache_dir}", flush=True)
        return
    WORLD_FOR_CHECK = str(m.world_path)
    players = {"defender": args.defender, "attacker": args.attacker}
    works = {}
    for role, who in players.items():
        work = out / role
        work.mkdir()
        (work / "role").write_text(role)
        if role == "defender":
            BRIEFS["defender"] = BRIEFS["defender"].replace("{history}", f"{args.warmup} days").replace(
                "{fraud_world}", FRAUD_DEFENDER_ONLY if args.defender_only else FRAUD_1V1)
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
        att_view = (m.attacker_view(weeks, weeks[-1]["_truth"]) if weeks and not args.defender_only
                    else {"results": [], "schemes_started_to_date": {}})
        (works["attacker"] / "data" / "results.json").write_text(json.dumps(att_view, indent=1))
        for role, work in works.items():
            lines = [f"# Week {wk + 1} of {args.weeks} is next.", ""]
            for i, (s, w) in enumerate(zip(scores[role], weeks), 1):
                if role == "defender":
                    d = w["defender"]
                    lines.append(f"week {i}: score {s:+.0f} (losses prevented {d['prevented']:+.0f}, added friction "
                                 f"{d['added_friction']:+.0f}; fraud losses at the bank {d['candidate']['losses']:.0f} vs "
                                 f"{d['no_defense']['losses']:.0f} with the built-in rules)")
                elif w["attacker"] is None:  # defender-only: no attacker results
                    continue
                else:
                    a = w["attacker"]
                    lines.append(f"week {i}: score {s:+.0f} (took {a['candidate']:.0f} at Brindlemark; your crews on their "
                                 f"usual habits would have taken {a['no_attack']:.0f} against the same defense)")
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
            if not is_ai(who):
                return {"player": who, "skipped": True}
            h = "msa" if is_player(who) else "claude"
            prompt = TASK.format(brief=BRIEFS[role], week=wk + 1, weeks=args.weeks, max_turns=args.max_turns,
                                 write=WRITE[h], finish=FINISH[h])
            if h == "msa":  # the benchmark harness: mini-SWE-agent over OpenRouter
                return play_turn(who, works[role], out / f"bin-{role}", "arena", prompt,
                                 out / "logs" / f"{role}-week{wk + 1}.traj.json", step_limit=args.max_turns)
            return claude_turn(who.split(":", 1)[1], works[role], out / f"bin-{role}", prompt,  # smoke tests
                               out / "logs" / f"{role}-week{wk + 1}.json", args.max_turns)

        with ThreadPoolExecutor(max_workers=2) as pool:
            turns = dict(zip(players, pool.map(turn, players)))

        deploy_error = None
        if is_ai(players["defender"]):
            lf = works["defender"] / "lists.json"
            deploy_error = m.deploy(json.loads((works["defender"] / "policy.json").read_text()),
                                    json.loads(lf.read_text()) if lf.exists() else None)
        plan_problems = []
        if is_ai(players["attacker"]):
            from arena.fw.cli import budget_problems
            plan = json.loads((works["attacker"] / "policy.json").read_text())
            plan_problems = budget_problems(plan, m.menu)
            if plan_problems:  # written around the checker: keep last week's plan
                plan = weeks[-1]["policies"]["attacker"] if weeks else {}
                (works["attacker"] / "policy.json").write_text(json.dumps(plan, indent=1))
            plan_problems += m.set_plan(plan)
        res, truth, _ = m.play_week()
        scores["defender"].append(res["defender"]["score"])
        scores["attacker"].append(res["attacker"]["score"] if res["attacker"] else 0.0)
        deploy_error = deploy_error or (f"your policy stopped the bank's systems mid-week and was rolled back: {m.runtime_error}"
                                        if m.runtime_error else None)
        weeks.append({**res, "_truth": str(truth), "turns": turns, "deploy_error": deploy_error,
                      "plan_problems": plan_problems,
                      "policies": {r: json.loads((w / "policy.json").read_text()) for r, w in works.items()}})
        lose = lambda k: f"{res['defender'][k]['losses']:.0f}" if res["defender"][k] else "-"  # noqa: E731
        print(f"week {wk + 1}: defender {res['defender']['score']:+.0f}  attacker {scores['attacker'][-1]:+.0f}  "
              f"[bank losses cand {lose('candidate')} no-def {lose('no_defense')} "
              f"no-att {lose('no_attack')} ref {lose('reference')}]  "
              + "  ".join(f"{r}: {t.get('turns')} turns {t.get('seconds')}s" for r, t in turns.items() if not t.get("skipped"))
              + (f"  DEPLOY ERROR: {deploy_error[:120]}" if deploy_error else ""), flush=True)

    # Headline: the share of the attack's losses at the bank that the defender's policy prevented, net of the friction
    # it added (each week's no-defense replay is the denominator). Dollars stay alongside.
    exposed = sum(w["defender"]["no_defense"]["losses"] for w in weeks)
    net = sum(w["defender"]["prevented"] - w["defender"]["added_friction"] for w in weeks)
    headline = {"defender_share_prevented": round(net / exposed, 4) if exposed else 0.0,
                "losses_exposed": round(exposed, 2), "net_saved": round(net, 2),
                "attacker_extra_take": None if args.defender_only else round(sum(scores["attacker"]), 2)}
    summary = {"players": players, "seed": args.seed, "weeks": args.weeks,
               "headline": headline, "total": {r: round(sum(v), 1) for r, v in scores.items()},
               "per_week": scores}
    (out / "result.json").write_text(json.dumps({**summary, "detail": [{k: v for k, v in w.items() if k != "_truth"}
                                                                        for w in weeks]}, indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
