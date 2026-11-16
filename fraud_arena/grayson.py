"""Grayson, as a defender agent uses it while it builds and tunes its detection: a budgeted, cached, logged client.

The agent calls it from its sandbox (`fa-grayson ask request.json`). Every call is logged with its usage; identical
requests are answered from the cache, so a replayed period gets the same answers; a period has a call budget.
"""
from __future__ import annotations

import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://api.finic.ai/v1/ask"


class BudgetExhausted(Exception):
    pass


class GraysonClient:
    def __init__(self, state_dir: Path, *, budget: int, api_key: str | None = None, endpoint: str = ENDPOINT,
                 model: str = "grayson-1", timeout_s: float = 60.0):
        self.state = Path(state_dir)
        self.budget = budget
        self.api_key = api_key if api_key is not None else os.environ.get("GRAYSON_API_KEY", "")
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        (self.state / "cache").mkdir(parents=True, exist_ok=True)

    def _count_path(self) -> Path:
        return self.state / "calls.json"

    def calls_used(self) -> int:
        p = self._count_path()
        return json.loads(p.read_text())["used"] if p.exists() else 0

    def reset_period(self) -> None:
        self._count_path().write_text(json.dumps({"used": 0}))

    def ask(self, data: dict, questions: dict) -> dict:
        body = {"model": self.model, "data": data, "questions": questions}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        cached = self.state / "cache" / f"{key}.json"
        if cached.exists():
            out = json.loads(cached.read_text())
            self._log(key, body, out, cached=True)
            return out
        with self._lock:
            used = self.calls_used()
            if used >= self.budget:
                raise BudgetExhausted(f"Grayson budget of {self.budget} calls this period is used up")
            self._count_path().write_text(json.dumps({"used": used + 1}))
        if not self.api_key:
            raise RuntimeError("GRAYSON_API_KEY is not set")
        req = Request(self.endpoint, data=json.dumps(body, allow_nan=False).encode(), headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
            "User-Agent": "fraud-arena/0.1"})
        start = time.monotonic()
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                out = json.load(resp)
        except HTTPError as e:
            e.close()
            raise RuntimeError(f"Grayson HTTP {e.code}") from None
        except (URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"Grayson connection failure: {type(e).__name__}") from None
        tmp = cached.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(out))
        tmp.rename(cached)
        self._log(key, body, out, cached=False, seconds=time.monotonic() - start)
        return out

    def ask_many(self, requests: list[dict], workers: int = 16) -> list[dict]:
        """Many requests at once (Grayson answers in about 0.15 s, so labelling thousands of cases is a batch job):
        each `{"data", "questions"}`, answered in order; a failed one gives `{"error": ...}`."""
        def one(r):
            try:
                return self.ask(r["data"], r["questions"])
            except Exception as e:  # reported per request
                return {"error": f"{type(e).__name__}: {e}"}
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(one, requests))

    def _log(self, key, body, out, *, cached, seconds=0.0):
        rec = {"time": time.time(), "key": key, "cached": cached, "seconds": round(seconds, 3),
               "questions": sorted(body["questions"]), "usage": out.get("usage", {})}
        with self._lock, open(self.state / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


class BrokerClient:
    """The client in a player's sandbox: calls go to the game's broker (`FA_GRAYSON_URL`, `FA_GRAYSON_TOKEN`), which
    holds the API key, the cache and the player's budget, so a player can't spend past its budget or keep the key."""

    def __init__(self, url: str, token: str, timeout_s: float = 900.0):
        self.url, self.token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def _post(self, path: str, body: dict) -> dict:
        req = Request(self.url + path, data=json.dumps(body, allow_nan=False, default=str).encode(),
                      headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            e.close()
            raise RuntimeError(f"Grayson broker HTTP {e.code}: {msg}") from None

    def ask(self, data: dict, questions: dict) -> dict:
        out = self.ask_many([{"data": data, "questions": questions}])[0]
        if "error" in out:
            raise (BudgetExhausted if "budget" in out["error"].lower() else RuntimeError)(out["error"])
        return out

    def ask_many(self, requests: list[dict], workers: int = 16) -> list[dict]:
        return self._post("/ask", {"requests": requests})["answers"]

    def calls_used(self) -> int:
        return self._post("/budget", {})["used"]

    @property
    def budget(self) -> int:
        return self._post("/budget", {})["budget"]


def make_client():
    """The Grayson client for this environment: the game's broker in a player's sandbox, else a direct client."""
    if os.environ.get("FA_GRAYSON_URL"):
        return BrokerClient(os.environ["FA_GRAYSON_URL"], os.environ.get("FA_GRAYSON_TOKEN", ""))
    return GraysonClient(Path(os.environ.get("FA_GRAYSON_STATE", ".arena/grayson")),
                         budget=int(os.environ.get("FA_GRAYSON_BUDGET", "0")))


class Broker:
    """Grader side: a local HTTP service answering players' Grayson calls, each player with its own budget per period
    (calls from its sessions and from its deployed workflow share it). `enabled=False` is the no-Grayson game."""

    def __init__(self, state_dir: Path, *, budget: int, enabled: bool = True, host: str = "127.0.0.1"):
        import secrets
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.state, self.budget, self.enabled = Path(state_dir), budget, enabled
        self.tokens: dict[str, str] = {}
        self.clients: dict[str, GraysonClient] = {}
        self._secrets = secrets
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                auth = self.headers.get("Authorization", "")
                player = next((p for p, t in broker.tokens.items()
                               if secrets.compare_digest(auth, f"Bearer {t}")), None)
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": "not JSON"})
                if player is None:
                    return self._send(403, {"error": "unknown token"})
                if self.path == "/budget":
                    c = broker.client(player)
                    return self._send(200, {"used": c.calls_used() if c else 0, "budget": broker.budget if c else 0})
                if self.path == "/ask":
                    reqs = body.get("requests")
                    if not isinstance(reqs, list) or len(reqs) > 50000:
                        return self._send(400, {"error": "requests must be a list of at most 50000"})
                    return self._send(200, {"answers": broker.ask_many(player, reqs)})
                self._send(404, {"error": "no such endpoint"})

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def token(self, player: str) -> str:
        return self.tokens.setdefault(player, self._secrets.token_urlsafe(24))

    def client(self, player: str) -> GraysonClient | None:
        if not self.enabled:
            return None
        if player not in self.clients:
            self.clients[player] = GraysonClient(self.state / player, budget=self.budget)
        return self.clients[player]

    def ask_many(self, player: str, requests: list[dict]) -> list[dict]:
        c = self.client(player)
        if c is None:
            return [{"error": "Grayson isn't available in this game"} for _ in requests]
        bad = [i for i, r in enumerate(requests) if not (isinstance(r, dict) and "data" in r and "questions" in r)]
        if bad:
            return [{"error": "each request needs data and questions"} for _ in requests]
        return c.ask_many(requests)

    def reset_period(self) -> None:
        for p in self.tokens:
            c = self.client(p)
            if c is not None:
                c.reset_period()

    def env(self, player: str) -> dict[str, str]:
        return {"FA_GRAYSON_URL": self.url, "FA_GRAYSON_TOKEN": self.token(player)}

    def close(self) -> None:
        self.server.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fa-grayson", description="Ask Grayson about a case (budgeted, cached).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="request JSON with 'data' and 'questions'; prints the answer JSON")
    a.add_argument("request", type=Path)
    m = sub.add_parser("batch", help="a JSONL file of requests ({data, questions} per line) -> a JSONL of answers")
    m.add_argument("requests", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--workers", type=int, default=16)
    sub.add_parser("budget", help="calls used and left this period")
    args = ap.parse_args(argv)
    client = make_client()
    if args.cmd == "budget":
        print(json.dumps({"used": client.calls_used(), "budget": client.budget}))
        return 0
    if args.cmd == "batch":
        reqs = [json.loads(l) for l in args.requests.read_text().splitlines() if l.strip()]
        answers = client.ask_many(reqs, workers=args.workers)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(a) + "\n" for a in answers))
        tmp.rename(args.out)
        errors = sum(1 for a in answers if "error" in a)
        print(json.dumps({"answered": len(answers) - errors, "errors": errors, "out": str(args.out),
                          "used": client.calls_used(), "budget": client.budget}))
        return 0
    req = json.loads(args.request.read_text())
    try:
        print(json.dumps(client.ask(req["data"], req["questions"]), indent=2))
    except (BudgetExhausted, RuntimeError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
