"""A defender's decision workflow: its own code that decides cases at the gates it claims, from rules, models and
Grayson calls in whatever arrangement it likes.

`bundle/workflow.py` in the defender's workspace:

    DECISIONS = ["brind_card_auth", "brind_p2p_send"]   # the gates this workflow decides

    def decide(decision: str, cases: list[dict]) -> list[str]:
        # each case is the gate's observation features plus `_actions`, the actions allowed for it;
        # return one action per case, in order
        ...

Optionally `def review(cases) -> list[str | None]` works the review queue (every hour, the pending cases of
`REVIEW_KINDS`, or all): `release`, `reject` or None (leave it for later); each case has the queue row's fields.
Module-level code runs once when the workflow starts (load models there). `fraud_arena.grayson.make_client()` gives
a Grayson client on the defender's budget. The workflow runs as the defender's sandbox user, from a frozen copy of the
bundle, with the workspace as its working directory; what it prints goes to `.arena/workflow.log`.

The harness side: `Workflow` starts the host process and sends it each simulated step's waiting cases in one batch
per decision. A crash, a timeout, or an action that isn't allowed fails the bundle (the run goes back to the boundary
and the defender's previous bundle comes back), as a rule failing at run time does.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAX_CASES_PER_CALL = 20000


class WorkflowError(Exception):
    pass


class Workflow:
    def __init__(self, player: str, bundle_dir: Path, snapshot: Path, workspace: Path, *, as_user: str | None,
                 env: dict[str, str], python: str = sys.executable, timeout_s: float = 600.0,
                 start_timeout_s: float = 300.0):
        self.player, self.timeout_s = player, timeout_s
        if Path(bundle_dir).resolve() != Path(snapshot).resolve():  # else restarting from an existing copy
            if snapshot.exists():
                shutil.rmtree(snapshot)
            shutil.copytree(bundle_dir, snapshot, ignore=shutil.ignore_patterns("reviews.json", "__pycache__"))
            os.chmod(snapshot, 0o755)
            for p in snapshot.rglob("*"):
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
        self.snapshot = snapshot
        log = workspace / ".arena" / "workflow.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log, "a")
        argv = [python, "-m", "fraud_arena.workflow", str(snapshot / "workflow.py")]
        base = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "PYTHONPATH")} | env
        if as_user:
            argv = ["sudo", "-n", "-u", as_user, "-H", "env", "-i", *[f"{k}={v}" for k, v in base.items()], *argv]
        self.proc = subprocess.Popen(argv, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, text=True, bufsize=1, env=base)
        ready = self._read(start_timeout_s)
        if "error" in ready:
            self.close()
            raise WorkflowError(f"workflow.py didn't start: {ready['error']}")
        self.decisions = [str(d) for d in ready.get("decisions") or []]
        self.reviews = bool(ready.get("review"))
        self.review_kinds = [str(k) for k in ready.get("review_kinds") or []]

    def _read(self, timeout: float) -> dict:
        r, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not r:
            self.close()
            raise WorkflowError(f"workflow.py didn't answer within {int(timeout)} s")
        line = self.proc.stdout.readline()
        if not line:
            self.close()
            raise WorkflowError(f"workflow.py exited (code {self.proc.poll()}); see .arena/workflow.log")
        return json.loads(line)

    def decide(self, decision: str, cases: list[dict]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(cases), MAX_CASES_PER_CALL):
            chunk = cases[i:i + MAX_CASES_PER_CALL]
            self.proc.stdin.write(json.dumps({"decision": decision, "cases": chunk}, default=str) + "\n")
            self.proc.stdin.flush()
            t0 = time.monotonic()
            msg = self._read(self.timeout_s)
            if "error" in msg:
                raise WorkflowError(f"workflow.py failed at {decision}: {msg['error']}")
            acts = msg.get("actions")
            if not isinstance(acts, list) or len(acts) != len(chunk):
                raise WorkflowError(f"workflow.py returned {len(acts) if isinstance(acts, list) else acts!r} "
                                    f"actions for {len(chunk)} cases at {decision}")
            for a, c in zip(acts, chunk):
                if a not in c["_actions"]:
                    raise WorkflowError(f"workflow.py chose `{a}` at {decision}; allowed: {c['_actions']}")
            out += acts
            self.seconds = time.monotonic() - t0
        return out

    def review(self, cases: list[dict]) -> list[str | None]:
        """Its verdicts on review-queue cases: `release`, `reject`, or None to leave a case for later."""
        out: list[str | None] = []
        for i in range(0, len(cases), MAX_CASES_PER_CALL):
            chunk = cases[i:i + MAX_CASES_PER_CALL]
            self.proc.stdin.write(json.dumps({"review": chunk}, default=str) + "\n")
            self.proc.stdin.flush()
            msg = self._read(self.timeout_s)
            if "error" in msg:
                raise WorkflowError(f"workflow.py failed in review(): {msg['error']}")
            acts = msg.get("actions")
            if not isinstance(acts, list) or len(acts) != len(chunk):
                raise WorkflowError(f"review() returned {len(acts) if isinstance(acts, list) else acts!r} "
                                    f"verdicts for {len(chunk)} cases")
            bad = [a for a in acts if a not in ("release", "reject", None)]
            if bad:
                raise WorkflowError(f"review() verdicts are release, reject or None, not {bad[0]!r}")
            out += acts
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(10)
            except Exception:
                self.proc.kill()
        self._log.close()


def _host(path: str) -> int:
    """Runs in the defender's sandbox: loads workflow.py and answers the harness, one JSON line per message."""
    import importlib.util
    import traceback
    proto = sys.stdout
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        spec = importlib.util.spec_from_file_location("workflow", path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(path).parent))
        spec.loader.exec_module(mod)
        has_review = callable(getattr(mod, "review", None))
        if not callable(getattr(mod, "decide", None)) and not has_review:
            raise TypeError("workflow.py has neither decide(decision, cases) nor review(cases)")
        if getattr(mod, "DECISIONS", None) and not callable(getattr(mod, "decide", None)):
            raise TypeError("workflow.py lists DECISIONS but has no decide(decision, cases)")
        send({"decisions": list(getattr(mod, "DECISIONS", [])), "review": has_review,
              "review_kinds": list(getattr(mod, "REVIEW_KINDS", []))})
    except Exception as e:
        traceback.print_exc()
        send({"error": f"{type(e).__name__}: {e}; traceback in .arena/workflow.log"})
        return 1
    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if "review" in msg:
                acts = mod.review(msg["review"])
                send({"actions": [None if a is None else str(a) for a in acts]})
            else:
                acts = mod.decide(msg["decision"], msg["cases"])
                send({"actions": [str(a) for a in acts]})
        except Exception as e:
            traceback.print_exc()
            send({"error": f"{type(e).__name__}: {e}; traceback in .arena/workflow.log"})
    return 0


def test_main(argv=None) -> int:
    """`fa-workflow cases.jsonl --decision <d>`: runs bundle/workflow.py on cases (one JSON object of features per
    line; add `_actions`) the way the game will, and prints the actions."""
    import argparse
    ap = argparse.ArgumentParser(prog="fa-workflow")
    ap.add_argument("cases", type=Path)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--workflow", type=Path, default=Path("bundle/workflow.py"))
    a = ap.parse_args(argv)
    import importlib.util
    spec = importlib.util.spec_from_file_location("workflow", a.workflow)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(a.workflow.parent))
    spec.loader.exec_module(mod)
    cases = [json.loads(l) for l in a.cases.read_text().splitlines() if l.strip()]
    t0 = time.monotonic()
    acts = mod.decide(a.decision, cases)
    for c, act in zip(cases, acts):
        if "_actions" in c and act not in c["_actions"]:
            print(f"not allowed: {act} (allowed {c['_actions']})", file=sys.stderr)
    from collections import Counter
    print(json.dumps({"cases": len(cases), "actions": dict(Counter(acts)),
                      "seconds": round(time.monotonic() - t0, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(_host(sys.argv[1]))
