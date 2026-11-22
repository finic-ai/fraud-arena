"""The benchmark's player harness: one model-neutral agent loop for every model and every Fraud Arena task.

It is mini-SWE-agent's DefaultAgent (one shell command per step, written as text, so it doesn't depend on any
provider's tool-calling format) over OpenRouter, in a restricted environment: the player can run only its own
command (`arena` in the 1v1, `casefile` in the investigation tasks), write files under drafts/ with a heredoc, read
them back with `cat drafts/...`, and end its turn. Nothing else executes. Model credentials stay in the harness:
the player's commands run with an environment that has no API keys.

    turn = play_turn("openrouter:thinkingmachines/inkling-2-mini", work, bin_dir, "arena", prompt, log_path,
                     step_limit=25)
"""

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

DONE = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM = """You are a player in Fraud Arena, a benchmark. You act through a restricted shell.

Every response has a short THOUGHT, then exactly ONE command in a bash code block:

THOUGHT: your reasoning

```mswea_bash_command
your_command_here
```

Only these commands work:
- `{cli} ...` (your tool; start with `{cli} --help` or the commands named in your task)
- writing a file under drafts/ with a heredoc:
  cat <<'EOF' > drafts/NAME
  ...file content...
  EOF
- reading one back: `cat drafts/NAME`
- ending your turn: `echo {done}` (alone, as its own command)
Anything else is refused. Each command runs fresh in your workspace."""

INSTANCE = """{{task}}

You have at most {{step_limit}} commands this turn. When you have finished, end with:

```mswea_bash_command
echo """ + DONE + """
```"""

HEREDOC = re.compile(r"^cat\s+<<\s*'?(\w+)'?\s*>\s*(drafts/[\w.\-]+)\s*\n(.*)\n\1\s*$", re.S)


class RestrictedEnvironment:
    """Runs only the player's own command, drafts/ heredocs, `cat drafts/...` and the end-of-turn echo."""

    def __init__(self, work, bin_dir, cli, timeout=600, step_limit=0, warn_at=5):
        self.work, self.bin_dir, self.cli, self.timeout = Path(work), Path(bin_dir), cli, timeout
        self.step_limit, self.warn_at = step_limit, warn_at
        self.commands = []
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM")
        self.env = {k: v for k, v in os.environ.items() if k in keep}
        self.env["PATH"] = f"{self.bin_dir}:{os.environ.get('PATH', '')}"

    def execute(self, action, cwd="", *, timeout=None):
        out = self._execute(action, timeout)
        left = self.step_limit - len(self.commands) if self.step_limit else None
        if left is not None and left <= self.warn_at:
            # The same reminder for every model: small models otherwise spend the whole budget investigating.
            out["output"] += (f"\n[{left} command(s) left this turn: finish now — write your answer/policy to drafts/ "
                              f"and submit it, then end the turn]")
        return out

    def _execute(self, action, timeout):
        from minisweagent.exceptions import Submitted
        command = action.get("command", "").strip()
        self.commands.append(command)
        if command == f"echo {DONE}":
            raise Submitted({"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "submission": ""}})
        m = HEREDOC.match(command)
        if m:
            path = (self.work / m.group(2)).resolve()
            if (self.work / "drafts").resolve() not in path.parents:
                return self._out("refused: files can only be written under drafts/", 1)
            path.write_text(m.group(3) + "\n")
            return self._out(f"wrote {m.group(2)} ({len(m.group(3))} characters)", 0)
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return self._out(f"could not parse the command: {e}", 2)
        if len(parts) == 2 and parts[0] == "cat" and parts[1].startswith("drafts/"):
            path = (self.work / parts[1]).resolve()
            if (self.work / "drafts").resolve() not in path.parents or not path.exists():
                return self._out(f"no such draft: {parts[1]}", 1)
            return self._out(path.read_text(), 0)
        if not parts or parts[0] != self.cli or any(p in ("&&", "||", ";", "|", ">", "<") for p in parts):
            return self._out(f"refused: only `{self.cli} ...`, writing drafts/ with a heredoc, `cat drafts/NAME` "
                             f"and `echo {DONE}` work here (one command at a time)", 1)
        try:
            p = subprocess.run([str(self.bin_dir / self.cli), *parts[1:]], cwd=self.work, env=self.env,
                               capture_output=True, text=True, timeout=timeout or self.timeout)
            return self._out((p.stdout + p.stderr)[-20000:], p.returncode)
        except subprocess.TimeoutExpired:
            return self._out("the command timed out", -1)

    @staticmethod
    def _out(text, code):
        return {"output": text, "returncode": code, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return dict(kwargs)

    def serialize(self):
        return {"info": {"config": {"environment_type": "arena.player.RestrictedEnvironment", "cli": self.cli}}}


def make_model(spec):
    """`openrouter:<model id>` (any model, including Claude, through OpenRouter)."""
    from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel
    if not spec.startswith("openrouter:"):
        raise ValueError(f"unknown model spec {spec!r}: use openrouter:<model id>")
    return OpenRouterTextbasedModel(model_name=spec.split(":", 1)[1], model_kwargs={"temperature": 0.0},
                                    cost_tracking="ignore_errors")


def play_turn(spec, work, bin_dir, cli, prompt, log_path, *, step_limit=25, cost_limit=5.0, wall_seconds=1800):
    """One player turn. Returns a summary like the Claude Code runner's: seconds, turns, cost, error, summary."""
    from minisweagent.agents.default import DefaultAgent
    env = RestrictedEnvironment(work, bin_dir, cli, step_limit=step_limit)
    agent = DefaultAgent(make_model(spec), env, system_template=SYSTEM.format(cli=cli, done=DONE),
                         instance_template=INSTANCE, step_limit=step_limit, cost_limit=cost_limit,
                         wall_time_limit_seconds=wall_seconds, output_path=Path(log_path))
    started = time.time()
    try:
        result = agent.run(prompt)
        status = result.get("exit_status", "")
    except Exception as e:  # a provider error ends the turn; the previous policy/answer stays
        status = f"error: {type(e).__name__}: {str(e)[:300]}"
    last = next((m.get("content", "") for m in reversed(agent.messages) if m.get("role") == "assistant"), "")
    return {"seconds": round(time.time() - started), "turns": agent.n_calls, "cost_usd": round(agent.cost, 4),
            "error": status not in ("Submitted",), "exit_status": status, "commands": len(env.commands),
            "summary": str(last)[-300:]}


def is_player(spec):
    return spec.startswith("openrouter:")
