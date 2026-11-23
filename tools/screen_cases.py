"""Verified subset: screen alert cases for answerability with two judges from different model families.

Each judge sees the case packet and the answer key (simulation truth) and says whether a skilled analyst could
determine the true verdict from the packet alone. A judge never changes a label. A case leaves the verified set only
when BOTH judges call it unanswerable. Every verdict is written to <cases>/verified.json for publishing.

    python tools/screen_cases.py CASES... [--parallel 4] [--limit N]

Judges: Claude Opus through Claude Code (`claude -p`) and OpenAI's Codex CLI (`codex exec`), each on the user's own
subscription login, no tools.
"""
import argparse
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """You are auditing a case in a fraud-detection benchmark. Below are the exact case packet a model sees and
the answer key (ground truth from a simulation, which models do NOT see). Decide whether a skilled fraud analyst could
reasonably determine the ground-truth verdict (fraud or not fraud) from the packet alone. Do not judge whether the
label is "fair"; judge whether the packet contains the evidence.

Reply with JSON only:
{"verdict": "answerable" | "ambiguous" | "unanswerable", "evidence": "the packet facts that show the truth, or what is missing"}

## Answer key
KEY

## Case packet
PACKET"""


def key_summary(key):
    keep = {k: key.get(k) for k in ("disposition", "typology", "customer_role", "scenario", "identity")}
    keep["involvements"] = [{k: i.get(k) for k in ("scenario", "role", "start", "end", "first_load", "tables", "events", "n_rows")}
                            for i in key.get("involvements") or []]
    return json.dumps(keep, indent=1)


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    v = d.get("verdict")
    return {"verdict": v if v in ("answerable", "ambiguous", "unanswerable") else "error",
            "evidence": str(d.get("evidence", ""))[:600]}


def claude_judge(prompt):
    p = subprocess.run(["claude", "-p", "--model", "opus", "--tools", "", "--strict-mcp-config", "--mcp-config",
                        '{"mcpServers": {}}', "--setting-sources", "project", "--no-session-persistence",
                        "--max-turns", "1", "--output-format", "json"], input=prompt, capture_output=True, text=True,
                       timeout=900)
    try:
        return parse(json.loads(p.stdout).get("result"))
    except ValueError:
        return {"verdict": "error", "evidence": (p.stdout + p.stderr)[-300:]}


def codex_judge(prompt):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "last.txt"
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral", "-C", d,
                        "-o", str(out), "-"], input=prompt, capture_output=True, text=True, timeout=900)
        return parse(out.read_text() if out.exists() else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for cs in a.cases:
        cs = Path(cs)
        out = cs / "verified.json"
        done = json.loads(out.read_text())["cases"] if out.exists() else {}
        ids = sorted(d.name for d in (cs / "alert").iterdir() if (d / "packet.txt").exists())
        todo = [c for c in ids if c not in done][: a.limit or None]

        def one(cid):
            key = json.loads((cs / "truth" / "keys" / f"{cid}.json").read_text())
            prompt = PROMPT.replace("KEY", key_summary(key)).replace("PACKET", (cs / "alert" / cid / "packet.txt").read_text())
            return cid, {"disposition": key["disposition"], "claude_opus": claude_judge(prompt), "codex": codex_judge(prompt)}

        with ThreadPoolExecutor(a.parallel) as pool:
            for cid, r in pool.map(one, todo):
                r["excluded"] = r["claude_opus"]["verdict"] == "unanswerable" and r["codex"]["verdict"] == "unanswerable"
                done[cid] = r
                out.write_text(json.dumps({"rule": "excluded when both judges say unanswerable", "cases": done}, indent=1))
        ex = [c for c, r in done.items() if r["excluded"]]
        agree = sum(r["claude_opus"]["verdict"] == r["codex"]["verdict"] for r in done.values())
        print(f"{cs.name}: screened {len(done)}, excluded {len(ex)} ({sum(done[c]['disposition'] == 'fraud' for c in ex)} fraud), "
              f"judges agree on {agree}/{len(done)}")


if __name__ == "__main__":
    main()
