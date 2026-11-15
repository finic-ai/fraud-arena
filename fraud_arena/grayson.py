"""Grayson, as a defender agent uses it while it builds and tunes its detection: a budgeted, cached, logged client.

The agent calls it from its sandbox (`fa-grayson ask request.json`). Every call is logged with its usage; identical
requests are answered from the cache, so a replayed period gets the same answers; a period has a call budget.
"""
from __future__ import annotations

import argparse
import hashlib
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
        used = self.calls_used()
        if used >= self.budget:
            raise BudgetExhausted(f"Grayson budget of {self.budget} calls this period is used up")
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
        self._count_path().write_text(json.dumps({"used": used + 1}))
        tmp = cached.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.rename(cached)
        self._log(key, body, out, cached=False, seconds=time.monotonic() - start)
        return out

    def _log(self, key, body, out, *, cached, seconds=0.0):
        rec = {"time": time.time(), "key": key, "cached": cached, "seconds": round(seconds, 3),
               "questions": sorted(body["questions"]), "usage": out.get("usage", {})}
        with open(self.state / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fa-grayson", description="Ask Grayson about a case (budgeted, cached).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="request JSON with 'data' and 'questions'; prints the answer JSON")
    a.add_argument("request", type=Path)
    sub.add_parser("budget", help="calls used and left this period")
    args = ap.parse_args(argv)
    state = Path(os.environ.get("FA_GRAYSON_STATE", ".arena/grayson"))
    client = GraysonClient(state, budget=int(os.environ.get("FA_GRAYSON_BUDGET", "0")))
    if args.cmd == "budget":
        print(json.dumps({"used": client.calls_used(), "budget": client.budget}))
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
