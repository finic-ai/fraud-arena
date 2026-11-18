"""The model gateway: the only route from a player's sandbox to a model, and the only holder of the API keys.

It runs grader side, as the harness user, on loopback. Each player (and the grader's text generator) gets its own
gateway token, which the gateway swaps for the provider's API key on each call it forwards. A call must name one of
the run's models. Every call's token usage goes to `usage.jsonl`, and once the run has used its approved budget the
gateway refuses further calls.
"""
from __future__ import annotations

import http.client
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROVIDERS = {
    "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com", "OPENAI_API_KEY"),
}
RUNNER_PROVIDER = {"claude-code": "anthropic", "codex": "openai"}
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
_CLIENT_AUTH = {"authorization", "x-api-key"}


class BudgetExceeded(Exception):
    pass


class Gateway:
    def __init__(self, usage_log: Path, *, models: set[str], budget_tokens: int, keys: dict[str, str] | None = None,
                 upstreams: dict[str, str] | None = None, host: str = "127.0.0.1"):
        if budget_tokens <= 0:
            raise ValueError("the gateway needs the run's approved token budget")
        self.models, self.budget = set(models), budget_tokens
        self.keys = keys if keys is not None else {p: os.environ.get(env, "") for p, (_, env) in PROVIDERS.items()}
        self.upstreams = {p: u for p, (u, _) in PROVIDERS.items()} | (upstreams or {})
        self.usage_log = Path(usage_log)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        self.tokens: dict[str, str] = {}
        self.used = 0
        self._lock = threading.Lock()
        gw = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                self._forward(b"")

            def do_POST(self):
                self._forward(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

            def _forward(self, body: bytes):
                provider, _, rest = self.path.lstrip("/").partition("/")
                if provider not in gw.upstreams:
                    return self._error(404, "no such provider")
                caller = gw.caller(self.headers)
                if caller is None:
                    return self._error(403, "unknown gateway token")
                if not gw.keys.get(provider):
                    return self._error(503, f"the gateway has no {provider} key")
                model = None
                if body:
                    try:
                        model = json.loads(body).get("model")
                    except (json.JSONDecodeError, AttributeError):
                        return self._error(400, "the request body isn't a JSON object")
                    if model not in gw.models:
                        return self._error(403, f"model {model!r} isn't one of this run's models")
                if gw.used >= gw.budget:
                    return self._error(429, "the run has used its approved token budget")
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP | _CLIENT_AUTH}
                if provider == "anthropic":
                    headers["x-api-key"] = gw.keys[provider]
                else:
                    headers["Authorization"] = f"Bearer {gw.keys[provider]}"
                up = urlsplit(gw.upstreams[provider])
                conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
                conn = conn_cls(up.netloc, timeout=600)
                started, sent = time.time(), False
                try:
                    conn.request(self.command, f"{up.path.rstrip('/')}/{rest}", body=body or None, headers=headers)
                    resp = conn.getresponse()
                    sent = True
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in _HOP:
                            self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    meter = _Meter()
                    while chunk := resp.read1(65536):
                        meter.feed(chunk)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    gw.record(caller, provider, model, resp.status, meter.usage(), time.time() - started)
                    self.wfile.write(b"0\r\n\r\n")
                except OSError as e:
                    if not sent:
                        self._error(502, f"the provider didn't answer: {type(e).__name__}")
                finally:
                    conn.close()

            def _error(self, code: int, message: str):
                data = json.dumps({"type": "error", "error": {"type": "gateway_error", "message": message}}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def token(self, caller: str) -> str:
        return self.tokens.setdefault(caller, secrets.token_urlsafe(24))

    def caller(self, headers) -> str | None:
        given = headers.get("x-api-key") or headers.get("Authorization", "").removeprefix("Bearer ")
        return next((c for c, t in self.tokens.items() if given and secrets.compare_digest(given, t)), None)

    def record(self, caller: str, provider: str, model: str | None, status: int, usage: dict[str, int],
               seconds: float) -> None:
        with self._lock:
            self.used += sum(usage.values())
            with self.usage_log.open("a") as f:
                f.write(json.dumps({"at": time.time(), "caller": caller, "provider": provider, "model": model,
                                    "status": status, "seconds": round(seconds, 3), **usage,
                                    "run_tokens": self.used}) + "\n")

    def env(self, caller: str, runner: str) -> dict[str, str]:
        """What a session needs to reach its model through the gateway: an address and its own gateway token."""
        provider = RUNNER_PROVIDER.get(runner)
        if provider == "anthropic":
            return {"ANTHROPIC_BASE_URL": f"{self.url}/anthropic", "ANTHROPIC_API_KEY": self.token(caller),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        if provider == "openai":
            return {"FA_GATEWAY_OPENAI_URL": f"{self.url}/openai/v1", "FA_GATEWAY_KEY": self.token(caller)}
        return {}

    def close(self) -> None:
        self.server.shutdown()


class _Meter:
    """Token usage from a response, plain JSON or server-sent events, in Anthropic's or OpenAI's shape."""

    def __init__(self):
        self.buf = b""
        self.found: dict[str, int] = {}

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            self._line(line)

    def usage(self) -> dict[str, int]:
        if self.buf:
            self._line(self.buf)
            self.buf = b""
        return {k: self.found.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                   "cache_creation_input_tokens")}

    def _line(self, line: bytes) -> None:
        line = line.strip().removeprefix(b"data:").strip()
        if not line.startswith(b"{"):
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        for u in (ev.get("usage"), (ev.get("message") or {}).get("usage"), (ev.get("response") or {}).get("usage")):
            if isinstance(u, dict):
                for k, v in u.items():
                    if isinstance(v, int) and k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                    "cache_creation_input_tokens"):
                        self.found[k] = max(self.found.get(k, 0), v)
