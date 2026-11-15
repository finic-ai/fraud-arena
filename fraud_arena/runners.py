"""Agent runners: one headless agent session per period boundary, in a workspace that persists between periods.

Each runner starts the agent product in its native harness (Claude Code, Codex CLI), points it at the task's MCP
tool server and nothing else that reads data, and returns the session's result and token usage. A session that hits
a subscription usage limit raises `UsageLimited`, so the queue can back off and resume it later.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class McpServer:
    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if (self.url is None) == (self.command is None):
            raise ValueError("an MCP server has a url or a command, not both")


@dataclass
class SessionResult:
    runner: str
    model: str
    ok: bool
    text: str
    seconds: float
    usage: dict[str, int]
    session_id: str | None = None
    error: str | None = None
    raw_log: Path | None = None


class UsageLimited(Exception):
    def __init__(self, message: str, retry_at: float | None):
        super().__init__(message)
        self.retry_at = retry_at


_LIMIT = re.compile(r"usage limit|rate limit|limit reached|try again (at|in)", re.I)


def _retry_at(text: str) -> float | None:
    m = re.search(r"limit reached\|(\d{9,})", text)
    if m:
        return float(m.group(1))
    m = re.search(r"try again in (\d+)\s*(second|minute|hour)", text, re.I)
    if m:
        unit = {"second": 1, "minute": 60, "hour": 3600}[m.group(2).lower()]
        return time.time() + int(m.group(1)) * unit
    return None


class AgentRunner:
    name = "agent"

    def __init__(self, model: str, *, timeout_s: int = 3 * 3600, extra_env: dict[str, str] | None = None,
                 as_user: str | None = None, proxy: str | None = None):
        self.model = model
        self.timeout_s = timeout_s
        self.extra_env = dict(extra_env or {})
        # The agent runs as its own Unix user (it can read only its workspace and its tables) and reaches the
        # network only through an allowlisting proxy; see sandbox/setup-box.sh.
        self.as_user = as_user
        self.proxy = proxy

    def command(self, workspace: Path, servers: list[McpServer], prompt: str) -> tuple[list[str], str | None]:
        raise NotImplementedError

    def parse(self, stdout: str) -> tuple[bool, str, dict[str, int], str | None]:
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TMPDIR", "USER", "SHELL")}
        if self.proxy:
            keep |= {"HTTPS_PROXY": self.proxy, "HTTP_PROXY": self.proxy, "https_proxy": self.proxy,
                     "http_proxy": self.proxy, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        return keep | self.extra_env

    def run(self, workspace: Path, prompt: str, servers: list[McpServer], log: Path) -> SessionResult:
        workspace.mkdir(parents=True, exist_ok=True)
        argv, stdin = self.command(workspace, servers, prompt)
        env = self.env()
        if self.as_user:
            argv = ["sudo", "-n", "-u", self.as_user, "-H", "env", "-i",
                    *[f"{k}={v}" for k, v in env.items() if k not in ("HOME", "USER")], *argv]
        start = time.monotonic()
        proc = subprocess.run(argv, cwd=workspace, input=stdin, capture_output=True, text=True, env=env,
                              timeout=self.timeout_s)
        seconds = time.monotonic() - start
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
        ok, text, usage, session = self.parse(proc.stdout)
        blob = text + "\n" + proc.stderr
        if not ok and _LIMIT.search(blob):
            raise UsageLimited(text.strip()[:300] or proc.stderr.strip()[:300], _retry_at(blob))
        error = None if ok and proc.returncode == 0 else (text.strip()[:500] or proc.stderr.strip()[:500]
                                                          or f"exit {proc.returncode}")
        return SessionResult(self.name, self.model, ok and proc.returncode == 0, text, seconds, usage, session,
                             error, log)


class ClaudeCodeRunner(AgentRunner):
    """`claude -p` with a subscription token (`CLAUDE_CODE_OAUTH_TOKEN` in `extra_env`)."""

    name = "claude-code"

    def command(self, workspace, servers, prompt):
        config = {"mcpServers": {}}
        for s in servers:
            if s.url:
                config["mcpServers"][s.name] = {"type": "http", "url": s.url}
            else:
                config["mcpServers"][s.name] = {"command": s.command, "args": list(s.args), "env": s.env}
        path = workspace / ".arena" / "mcp.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
        argv = ["claude", "-p", "--model", self.model, "--output-format", "json", "--mcp-config", str(path),
                "--strict-mcp-config", "--dangerously-skip-permissions"]
        return argv, prompt

    def parse(self, stdout):
        try:
            out = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
        except json.JSONDecodeError:
            return False, stdout[-2000:], {}, None
        u = out.get("usage") or {}
        usage = {k: int(u.get(k, 0) or 0) for k in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        ok = out.get("subtype") == "success" and not out.get("is_error", False)
        return ok, str(out.get("result", "")), usage, out.get("session_id")


class CodexRunner(AgentRunner):
    """`codex exec --json`, signed in with a ChatGPT subscription (its `CODEX_HOME` holds the login)."""

    name = "codex"

    def command(self, workspace, servers, prompt):
        argv = ["codex", "exec", "--json", "-m", self.model, "-C", str(workspace), "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox", "--ignore-user-config"]
        for s in servers:
            key = f"mcp_servers.{s.name}"
            if s.url:
                argv += ["-c", f'{key}.url="{s.url}"']
            else:
                argv += ["-c", f'{key}.command="{s.command}"', "-c", f"{key}.args={json.dumps(list(s.args))}"]
        argv.append("-")
        return argv, prompt

    def parse(self, stdout):
        usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        text, session, ok, failed = "", None, False, None
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type", "")
            if kind == "thread.started":
                session = ev.get("thread_id")
            elif kind == "turn.completed":
                ok = True
                for k in usage:
                    usage[k] += int((ev.get("usage") or {}).get(k, 0) or 0)
            elif kind in ("turn.failed", "error"):
                failed = json.dumps(ev.get("error", ev))[:500]
            elif kind == "item.completed" and (ev.get("item") or {}).get("type") == "agent_message":
                text = ev["item"].get("text", text)
        if failed:
            return False, failed, usage, session
        return ok, text, usage, session


RUNNERS = {"claude-code": ClaudeCodeRunner, "codex": CodexRunner}
