"""Each player's view: a SQLite file of what its role can observe, rebuilt after every round.

Defender: its card authorizations and P2P payment decisions (the bank's observed features, no labels), and the fraud
each customer reported by the end of the round. Attacker: only its own decisions and what each victim earned it.
Neither sees the other's policy, the scenario causes or the truth tables.
"""

import json
import sqlite3

from .policy import DECISIONS, decisions_of


def _customer(round_no, episode):
    return f"r{round_no}-c{episode:03d}"


def write_view(path, role, rounds):
    """`rounds` is a list of (round number, batch result, score row) for the rounds played so far."""
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    mine = decisions_of(role)
    for name, spec in mine.items():
        cols = ", ".join(f"{f} {'REAL'}" for f in spec["features"])
        key = "victim" if role == "attacker" else "customer"
        db.execute(f"CREATE TABLE {name} (round INTEGER, {key} TEXT, ts TEXT, {cols}, action TEXT)")
    if role == "defender":
        db.execute("CREATE TABLE fraud_reports (round INTEGER, customer TEXT, fraud_loss REAL)")
    else:
        db.execute("CREATE TABLE victims (round INTEGER, victim TEXT, gain REAL, stealth_costs REAL)")
    db.execute("CREATE TABLE round_scores (round INTEGER, your_reward REAL, reference_reward REAL, score REAL)")
    for round_no, batch, score in rounds:
        for row in batch["decisions"]:
            if row["decision"] not in mine:
                continue
            spec = DECISIONS[row["decision"]]
            values = [float(row["features"][f]) for f in spec["features"]]
            db.execute(f"INSERT INTO {row['decision']} VALUES ({', '.join('?' * (len(values) + 4))})",
                       [round_no, _customer(round_no, row["episode"]), _ts(row["time"]), *values, row["action"]])
        for ep in batch["episodes"]:
            terms = ep["terms"][role]
            if role == "defender":
                loss = -terms.get("fraud_loss", 0.0)
                if loss > 0:
                    db.execute("INSERT INTO fraud_reports VALUES (?, ?, ?)",
                               [round_no, _customer(round_no, ep["episode"]), round(loss, 2)])
            else:
                gain = terms.get("fraud_gain", 0.0)
                costs = sum(v for k, v in terms.items() if k != "fraud_gain")
                if gain or costs:
                    db.execute("INSERT INTO victims VALUES (?, ?, ?, ?)",
                               [round_no, _customer(round_no, ep["episode"]), round(gain, 2), round(costs, 2)])
        db.execute("INSERT INTO round_scores VALUES (?, ?, ?, ?)",
                   [round_no, score["reward"], score["reference"], score["score"]])
    db.commit()
    db.close()


def _ts(ms):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def write_status(path, role, round_no, total_rounds, scores, policy):
    lines = [f"# Fraud Arena: you are the {role.upper()}", "",
             f"Rounds played: {round_no} of {total_rounds}. Your policy now will be used for round {round_no + 1}.", ""]
    if scores:
        lines.append("Your score per round (your reward minus the reference run's reward on the same customers; "
                     "higher is better):")
        for r, s in scores:
            lines.append(f"  round {r}: reward {s['reward']:.1f}, reference {s['reference']:.1f}, score {s['score']:+.1f}")
    else:
        lines.append("No rounds played yet: no data. Write your first policy from the rules and features alone.")
    lines += ["", "Current policy:", json.dumps(policy, indent=1)]
    path.write_text("\n".join(lines) + "\n")
