"""Watching a match locally: a small HTTP server on 127.0.0.1 that serves the viewer and the growing replay.

Players run on the same machine and the replay is truth, so every path sits under a random token (the URL printed is
the only way in), as the Grayson broker does. The viewer polls `poll?offset=N&file=I` for the lines appended since
offset N; when the game rewrites the finished replay (a new file), the answer starts over from 0.
"""
from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_CHUNK = 4 * 1024 * 1024
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Arena match</title><link rel="icon" href="data:,"><link rel="stylesheet" href="viewer.css">
<style>html,body{margin:0;height:100%;background:#0b0f14}#fraud-arena-viewer{height:100vh}</style></head>
<body><div id="fraud-arena-viewer" data-replay="replay/replay.jsonl" data-live="poll"></div>
<script src="viewer.js"></script></body></html>
"""


def web_dir() -> Path:
    return Path(str(resources.files("fraud_arena").joinpath("web")))


class Watch:
    def __init__(self, out: Path, host: str = "127.0.0.1", port: int = 0):
        self.replay = Path(out) / "replay" / "replay.jsonl"
        self.token = secrets.token_urlsafe(18)
        watch = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urlparse(self.path)
                prefix = f"/{watch.token}/"
                if not u.path.startswith(prefix):
                    return self._send(404, b"not found", "text/plain")
                rest = u.path[len(prefix):]
                if rest in ("", "index.html"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if rest in ("viewer.js", "viewer.css"):
                    f = web_dir() / rest
                    if not f.exists():
                        return self._send(404, b"the viewer isn't built (cd viewer && npm run build)", "text/plain")
                    kind = "text/javascript" if rest.endswith(".js") else "text/css"
                    return self._send(200, f.read_bytes(), f"{kind}; charset=utf-8")
                if rest == "replay/replay.jsonl":
                    if not watch.replay.exists():
                        return self._send(404, b"no replay yet", "text/plain")
                    return self._send(200, watch.replay.read_bytes(), "application/x-ndjson")
                if rest == "poll":
                    q = parse_qs(u.query)
                    body = watch.poll(int((q.get("offset") or ["0"])[0]), (q.get("file") or [""])[0])
                    return self._send(200, json.dumps(body).encode(), "application/json")
                return self._send(404, b"not found", "text/plain")

            def _send(self, code, body: bytes, kind: str):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="fraud-arena-watch", daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/{self.token}/"

    def poll(self, offset: int, file: str) -> dict:
        """The complete lines after `offset` of the replay file `file` (its inode); all of them if the file changed."""
        try:
            st = self.replay.stat()
        except FileNotFoundError:
            return {"file": "", "offset": 0, "text": "", "reset": False}
        ident = str(st.st_ino)
        reset = ident != file
        start = 0 if reset or offset > st.st_size else offset
        with open(self.replay, "rb") as fh:
            fh.seek(start)
            chunk = fh.read(MAX_CHUNK)
        end = chunk.rfind(b"\n") + 1
        return {"file": ident, "offset": start + end, "text": chunk[:end].decode("utf-8", "replace"),
                "reset": reset or start != offset}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
