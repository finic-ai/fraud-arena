"""Export a case as a Harbor task folder (the layout of FrontierSWE-style task repos):

    <out>/<set>-<case>/
      instruction.md            the brief plus how to work and answer
      task.toml                 task metadata, agent / verifier timeouts, environment
      environment/Dockerfile    python + duckdb image with the bank tables and the `casefile` tool
      environment/bank/         the bank's tables as of the case time (rows with LOAD_TS <= case time only)
      environment/app/          the agent's workspace: case.json, brief.md, notes.md, drafts/
      environment/casefile/     the tool (arena.tasks cli, casedb, links, schema) and its wrapper
      tests/test.sh, verify.py  the verifier: scores /app/answer.json against the answer key, writes the reward
      tests/answer_key.json     the answer key (the verifier's copy; never in the agent's image)
      tests/score.py, schema.py the scoring code
      solution/solve.sh         writes the reference answer (from the key) and submits it
      solution/answer.json

Images are not built here. The reward is the per-case score of docs/tasks.md (0-1) in /logs/verifier/reward.txt,
with the details in /logs/verifier/reward.json."""

import json
import shutil
from pathlib import Path

from .casedb import open_asof

PKG = Path(__file__).resolve().parent
TOOL_MODULES = ["__init__.py", "cli.py", "casedb.py", "links.py", "schema.py"]

DOCKERFILE = """FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PAGER=cat
RUN pip install --no-cache-dir duckdb==2.0.1
RUN useradd -m agent && install -d -o agent /app /logs/agent && install -d /logs/verifier
COPY bank/ /data/bank/
COPY casefile/arena/ /opt/casefile/arena/
COPY casefile/casefile /usr/local/bin/casefile
RUN chmod 0755 /usr/local/bin/casefile && chmod -R a+rX /data /opt/casefile
COPY --chown=agent:agent app/ /app/
WORKDIR /app
USER agent
"""

WRAPPER = """#!/bin/sh
PYTHONPATH=/opt/casefile exec python3 -m arena.tasks.cli /app "$@"
"""

TEST_SH = """#!/bin/bash
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/verify.py"
"""

VERIFY = '''"""Score /app/answer.json against answer_key.json; write /logs/verifier/reward.txt and reward.json."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import alert_case_score, incident_case_score  # noqa: E402

key = json.loads((HERE / "answer_key.json").read_text())
ans_path = Path("/app/answer.json")
ans = json.loads(ans_path.read_text()) if ans_path.exists() else None
fn = alert_case_score if key["task"] == "alert" else incident_case_score
res = fn(key, ans)
out = Path("/logs/verifier")
out.mkdir(parents=True, exist_ok=True)
(out / "reward.txt").write_text(f"{res['score']}\\n")
(out / "reward.json").write_text(json.dumps({"reward": res["score"], **res, "submitted": ans is not None}, indent=1))
print(json.dumps(res))
'''

SOLVE = """#!/bin/bash
set -e
cd /app
mkdir -p drafts
cp "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/answer.json" drafts/answer.json
casefile submit drafts/answer.json
"""

HOWTO = """
## How to work
Run `casefile` commands in the shell from /app (`casefile tables`, `casefile customer ...`, `casefile query "SQL"`,
`casefile links ... --by ...`). Write your answer as JSON to /app/drafts/answer.json and run
`casefile submit drafts/answer.json`; the last valid submission (/app/answer.json) is graded.
"""


def reference_answer(key):
    if key["task"] == "alert":
        fraud = key["disposition"] == "fraud"
        return {"disposition": key["disposition"], "typology": key["typology"], "customer_role": key["customer_role"],
                "confidence": 0.95 if fraud else 0.05,
                "recommended_action": "restrict_account" if fraud else "close_no_action",
                "rationale": "Reference answer from the simulator's truth."}
    if not key["is_incident"]:
        return {"is_incident": False, "typology": "none", "members": [], "link_keys": [], "summary":
                "Reference answer: a benign cluster.", "recommended_actions": ["close_no_action"]}
    members = [{"cust_id": m["cust_id"], "role": m["role"]} for m in key["members"]
               if m["seed"] or m["scored"] or key.get("expansion") == "none"]
    return {"is_incident": True, "typology": key["typology"], "members": members,
            "link_keys": [f"{key['seed_key'][0]}:{key['seed_key'][1]}"],
            "summary": "Reference answer from the simulator's truth.", "recommended_actions": ["file_sar"]}


def export_case(cases, task, case_id, out, index):
    cases, out = Path(cases), Path(out)
    name = f"{cases.name}-{case_id}".lower()
    root = out / name
    if root.exists():
        shutil.rmtree(root)
    env = root / "environment"
    ws_src = cases / task / case_id
    case = json.loads((ws_src / "case.json").read_text())
    key = json.loads((cases / "truth" / "keys" / f"{case_id}.json").read_text())
    key["task"] = task
    # the agent's workspace
    app = env / "app"
    (app / "drafts").mkdir(parents=True)
    case["bank"] = "/data/bank"
    (app / "case.json").write_text(json.dumps(case, indent=1))
    for f in ("brief.md", "notes.md"):
        shutil.copy(ws_src / f, app / f)
    # the bank as of the case time: nothing after it is in the image
    (env / "bank").mkdir()
    db = open_asof(cases / "bank", case["case_time"], lock=False)
    for (t,) in db.execute("SELECT table_name FROM duckdb_tables() WHERE NOT internal ORDER BY 1").fetchall():
        db.execute(f'COPY "{t}" TO \'{env / "bank" / (t + ".parquet")}\' (FORMAT parquet, COMPRESSION zstd)')
    # the tool
    pkg = env / "casefile" / "arena" / "tasks"
    pkg.mkdir(parents=True)
    (env / "casefile" / "arena" / "__init__.py").write_text("")
    for m in TOOL_MODULES:
        shutil.copy(PKG / m, pkg / m)
    (env / "casefile" / "casefile").write_text(WRAPPER)
    (env / "Dockerfile").write_text(DOCKERFILE)
    # verifier
    tests = root / "tests"
    tests.mkdir()
    (tests / "answer_key.json").write_text(json.dumps(
        {k: v for k, v in key.items() if k not in ("involvements",)}, indent=1, default=str))
    (tests / "test.sh").write_text(TEST_SH)
    (tests / "verify.py").write_text(VERIFY)
    shutil.copy(PKG / "score.py", tests / "score.py")
    shutil.copy(PKG / "schema.py", tests / "schema.py")
    (tests / "score.py").write_text((tests / "score.py").read_text().replace("from .schema", "from schema"))
    # reference solution
    sol = root / "solution"
    sol.mkdir()
    (sol / "answer.json").write_text(json.dumps(reference_answer(key), indent=1))
    (sol / "solve.sh").write_text(SOLVE)
    for f in (tests / "test.sh", sol / "solve.sh", env / "casefile" / "casefile"):
        f.chmod(0o755)
    (root / "instruction.md").write_text((ws_src / "brief.md").read_text() + HOWTO)
    kind = "alert review" if task == "alert" else "incident investigation"
    (root / "task.toml").write_text(f'''schema_version = "1.3"

[task]
name = "fraud-arena/{name}"
description = "Fraud Arena {kind} case {case_id} (seed set {index.get('split')}, seed {index.get('seed')}): investigate a simulated bank's tables as of the case time and submit a graded JSON answer."
authors = []
keywords = ["fraud", "investigation", "sql", "{task}"]

[metadata]
difficulty = "{'medium' if task == 'alert' else 'hard'}"
category = "fraud-investigation"
split = "{index.get('split')}"
seed = {index.get('seed')}
case_id = "{case_id}"
oracle_reward_threshold = 0.99

[agent]
timeout_sec = 1200.0
user = "agent"

[verifier]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
network_mode = "none"
cpus = 2
memory_mb = 4096
storage_mb = 4096
''')
    return root
