import json
import os
import stat
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fraud_arena import sandbox
from fraud_arena.engine import RunmereEngine
from fraud_arena.gateway import Gateway
from fraud_arena.runners import ClaudeCodeRunner, CodexRunner


class FakeProvider:
    """An upstream that records what reached it and answers like Anthropic, plain or streamed."""

    def __init__(self):
        self.seen = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if body.get("stream"):
                    data = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
                            b'{"input_tokens":40,"output_tokens":1}}}\n\n'
                            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":12}}\n\n')
                    ctype = "text/event-stream"
                else:
                    data = json.dumps({"content": [{"text": "hi"}],
                                       "usage": {"input_tokens": 30, "output_tokens": 5}}).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _post(url, body, key):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"x-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.fixture
def gateway(tmp_path):
    up = FakeProvider()
    gw = Gateway(tmp_path / "usage.jsonl", models={"sonnet"}, budget_tokens=100,
                 keys={"anthropic": "sk-real", "openai": ""}, upstreams={"anthropic": up.url})
    yield gw, up
    gw.close()
    up.server.shutdown()


def test_gateway_swaps_the_player_token_for_the_key(gateway, tmp_path):
    gw, up = gateway
    env = gw.env("defender-brindlemark", "claude-code")
    assert "sk-real" not in json.dumps(env)
    status, body = _post(f"{env['ANTHROPIC_BASE_URL']}/v1/messages", {"model": "sonnet"}, env["ANTHROPIC_API_KEY"])
    assert status == 200 and json.loads(body)["content"][0]["text"] == "hi"
    assert up.seen[0]["path"] == "/v1/messages"
    assert up.seen[0]["headers"]["x-api-key"] == "sk-real"
    assert env["ANTHROPIC_API_KEY"] not in json.dumps(up.seen[0]["headers"])
    rec = json.loads((tmp_path / "usage.jsonl").read_text().splitlines()[0])
    assert rec["caller"] == "defender-brindlemark" and rec["input_tokens"] == 30 and rec["output_tokens"] == 5


def test_gateway_refuses_unknown_tokens_and_other_models(gateway):
    gw, up = gateway
    url = gw.env("p", "claude-code")["ANTHROPIC_BASE_URL"] + "/v1/messages"
    assert _post(url, {"model": "sonnet"}, "not-a-token")[0] == 403
    assert _post(url, {"model": "opus"}, gw.token("p"))[0] == 403
    assert up.seen == []


def test_gateway_meters_streams_and_stops_at_the_budget(gateway):
    gw, up = gateway
    url = gw.env("p", "claude-code")["ANTHROPIC_BASE_URL"] + "/v1/messages"
    status, body = _post(url, {"model": "sonnet", "stream": True}, gw.token("p"))
    assert status == 200 and b"message_delta" in body
    assert gw.used == 52
    assert _post(url, {"model": "sonnet", "stream": True}, gw.token("p"))[0] == 200
    assert _post(url, {"model": "sonnet"}, gw.token("p"))[0] == 429
    assert len(up.seen) == 2


def test_gateway_needs_an_approved_budget(tmp_path):
    with pytest.raises(ValueError):
        Gateway(tmp_path / "u.jsonl", models={"sonnet"}, budget_tokens=0, keys={})


def test_secrets_go_through_a_file_not_the_command_line(tmp_path):
    env = {"ANTHROPIC_API_KEY": "gw-token-123", "FA_GRAYSON_TOKEN": "broker-456", "PATH": "/usr/bin"}
    argv = sandbox.as_user("arena-p3", env, ["claude", "-p"], tmp_path / ".arena" / "session-env.json")
    assert argv[:6] == ["sudo", "-n", "-u", "arena-p3", "-H", sandbox.EXEC]
    assert "gw-token-123" not in " ".join(argv) and "broker-456" not in " ".join(argv)
    f = tmp_path / ".arena" / "session-env.json"
    assert json.loads(f.read_text())["ANTHROPIC_API_KEY"] == "gw-token-123"
    assert stat.S_IMODE(f.stat().st_mode) & 0o007 == 0


def test_a_session_as_a_player_shows_no_secret_to_ps(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["env"] = argv, kw["env"]
        seen["file"] = json.loads((tmp_path / "ws" / ".arena" / "session-env.json").read_text())
        out = json.dumps({"subtype": "success", "result": "done", "usage": {}})
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = ClaudeCodeRunner("sonnet", as_user="arena-p1", extra_env={"ANTHROPIC_API_KEY": "gw-secret"})
    res = r.run(tmp_path / "ws", "go", [], tmp_path / "log")
    assert res.ok
    assert "gw-secret" not in " ".join(seen["argv"]) and "gw-secret" not in json.dumps(seen["env"])
    assert seen["file"]["ANTHROPIC_API_KEY"] == "gw-secret"
    assert not (tmp_path / "ws" / ".arena" / "session-env.json").exists()


def test_codex_reaches_its_model_only_through_the_gateway(tmp_path):
    r = CodexRunner("gpt-6.1-sol", extra_env={"FA_GATEWAY_OPENAI_URL": "http://127.0.0.1:9/openai/v1",
                                              "FA_GATEWAY_KEY": "gw-secret"})
    argv, _ = r.command(tmp_path, [], "go")
    joined = " ".join(argv)
    assert 'model_provider="arena"' in joined and "http://127.0.0.1:9/openai/v1" in joined
    assert 'env_key="FA_GATEWAY_KEY"' in joined and "gw-secret" not in joined


def test_a_players_tables_live_in_its_own_folder(tmp_path):
    from fraud_arena.config import EpisodeConfig

    eng = object.__new__(RunmereEngine)
    eng.sandbox_root = tmp_path
    cfg = EpisodeConfig(world=Path("w.yaml"), institution="brindlemark", seed=7)
    d = eng._agent_dir(cfg, "data-day90", "defender-brindlemark")
    assert d == tmp_path / "defender-brindlemark" / "tables" / "brindlemark-s7" / "data-day90"
    assert eng._agent_dir(cfg, "meta", "defender-dovrex").parts[-4] == "defender-dovrex"


def test_private_dir_for_the_current_user(tmp_path):
    import pwd

    me = pwd.getpwuid(os.getuid()).pw_name
    d = sandbox.private_dir(tmp_path / "p", me)
    assert stat.S_IMODE(d.stat().st_mode) & 0o007 == 0


def _args(**kw):
    import argparse

    base = dict(credentials="subscription", proxy="http://127.0.0.1:3128", player_users=True, session_timeout=60,
                model_budget_tokens=None, out="/tmp/out")
    return argparse.Namespace(**(base | kw))


def test_subscriptions_pass_the_token_by_file_and_codex_as_its_login_user(monkeypatch):
    from fraud_arena import cli

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-x")
    env, user, proxy = cli._credentials(_args(), "claude-code", "defender-dovrex", "arena-p2", None)
    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-x"} and user == "arena-p2" and proxy.endswith(":3128")
    env, user, _ = cli._credentials(_args(), "codex", "defender-kellmont", "arena-p3", None)
    assert user == cli.CODEX_USER and env["CODEX_HOME"] == cli.CODEX_HOME
    assert cli._gateway(_args(), ["claude-code:sonnet"]) is None


def test_subscriptions_need_the_token(monkeypatch):
    from fraud_arena import cli

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli._credentials(_args(), "claude-code", "p", "arena-p1", None)


def test_api_keys_go_through_the_gateway_with_an_approved_budget(tmp_path, monkeypatch):
    from fraud_arena import cli

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-x")
    a = _args(credentials="api-key", out=str(tmp_path))
    with pytest.raises(SystemExit):
        cli._gateway(a, ["claude-code:sonnet"])
    a.model_budget_tokens = 1000
    gw = cli._gateway(a, ["claude-code:sonnet", None, "baseline:block-all"])
    try:
        env, user, proxy = cli._credentials(a, "claude-code", "attacker-R3", "arena-p7", gw)
        assert user == "arena-p7" and proxy is None and "sk-ant-oat-x" not in json.dumps(env)
        assert env["ANTHROPIC_API_KEY"] == gw.token("attacker-R3") and gw.models == {"sonnet"}
    finally:
        gw.close()
