import json
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraud_arena import bundle
from fraud_arena.config import EpisodeConfig, ScoreWeights
from fraud_arena.episode import Episode
from fraud_arena.grayson import BudgetExhausted, GraysonClient
from fraud_arena.queue import Job, Queue
from fraud_arena.runners import ClaudeCodeRunner, CodexRunner, McpServer, SessionResult, UsageLimited, _retry_at
from fraud_arena.scoring import paired, window_score


# --- runners ---

def test_claude_parse_success_and_usage():
    out = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done",
                      "session_id": "s1", "usage": {"input_tokens": 10, "output_tokens": 5,
                                                    "cache_read_input_tokens": 100}})
    ok, text, usage, sid = ClaudeCodeRunner("sonnet").parse(out)
    assert ok and text == "done" and sid == "s1"
    assert usage["input_tokens"] == 10 and usage["cache_read_input_tokens"] == 100


def test_claude_command_points_only_at_task_server(tmp_path):
    argv, stdin = ClaudeCodeRunner("sonnet").command(tmp_path, [McpServer("arena", url="http://x/mcp")], "hi")
    assert "--strict-mcp-config" in argv and stdin == "hi"
    cfg = json.loads((tmp_path / ".arena" / "mcp.json").read_text())
    assert list(cfg["mcpServers"]) == ["arena"]


def test_codex_parse_events():
    lines = [{"type": "thread.started", "thread_id": "t"},
             {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
             {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3, "cached_input_tokens": 2}}]
    ok, text, usage, sid = CodexRunner("gpt").parse("\n".join(map(json.dumps, lines)))
    assert ok and text == "ok" and sid == "t" and usage == {"input_tokens": 7, "output_tokens": 3,
                                                            "cached_input_tokens": 2}


def test_usage_limit_reset_time():
    assert _retry_at("Claude AI usage limit reached|1900000000") == 1900000000.0
    assert _retry_at("nothing") is None


# --- bundle ---

def test_bundle_inlines_models_and_checks_decisions(tmp_path):
    b = tmp_path / "bundle"
    (b / "models").mkdir(parents=True)
    (b / "models" / "m.json").write_text(json.dumps({"kind": "logistic", "weights": [1.0], "intercept": 0}))
    (b / "models" / "t.json").write_text(json.dumps([{"nodeid": 0, "leaf": 0.5}]))
    (b / "policy.yaml").write_text("models: {risk: {file: models/m.json, features: [amount]},"
                                   " t: {file: models/t.json, kind: trees, features: [amount]}}\n"
                                   "rules: [{decision: brind_card_auth, when: 'score.risk > 0.9', action: decline}]\n")
    got = bundle.load(b, role="brind_defender", decisions={"brind_card_auth"})
    assert got.policy["models"]["risk"]["kind"] == "logistic" and got.policy["format"] == 1
    assert got.policy["models"]["t"]["trees"] == [{"nodeid": 0, "leaf": 0.5}]
    (b / "policy.yaml").write_text("rules: [{decision: dovrex_login_check, action: block}]\n")
    with pytest.raises(bundle.BundleError, match="isn't one of this role's"):
        bundle.load(b, role="brind_defender", decisions={"brind_card_auth"})


def test_bundle_refuses_files_outside(tmp_path):
    b = tmp_path / "bundle"
    b.mkdir()
    (tmp_path / "secret.json").write_text("{}")
    (b / "policy.yaml").write_text("models: {x: {file: ../secret.json}}\nrules: []\n")
    with pytest.raises(bundle.BundleError, match="isn't in the bundle"):
        bundle.load(b, role="r", decisions=set())


# --- grayson ---

def test_grayson_cache_and_budget(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, *a): return json.dumps({"answers": {"q": {"choice": "fraud"}}}).encode()

    def fake_urlopen(req, timeout):
        calls.append(req)
        return Resp()

    monkeypatch.setattr("fraud_arena.grayson.urlopen", fake_urlopen)
    g = GraysonClient(tmp_path, budget=1, api_key="k")
    q = {"q": {"type": "choice", "instructions": "", "criteria": {"fraud": "", "ok": ""}}}
    assert g.ask({"a": 1}, q)["answers"]["q"]["choice"] == "fraud"
    assert g.ask({"a": 1}, q)["answers"]["q"]["choice"] == "fraud"  # cached, no new call
    assert len(calls) == 1
    with pytest.raises(BudgetExhausted):
        g.ask({"a": 2}, q)


# --- scoring ---

def _write(path, cols):
    pq.write_table(pa.table(cols), path)


def _truth(tmp, victim_loss, decline_good):
    t = tmp / "truth"
    t.mkdir(parents=True)
    ts = pa.array(["2026-07-01T00:00:00Z"] * 2, pa.string())
    from datetime import datetime, timezone
    when = [datetime(2026, 7, 1, tzinfo=timezone.utc)] * 2
    _write(t / "postings.parquet", {"key": ["e1", "e2"], "time": when, "payer": ["acc_v", "acc_mule"],
                                    "payee": ["ext", "ext"], "amount": [victim_loss, victim_loss],
                                    "dirty": [0.0, victim_loss], "cause_kind": ["scenario", "scenario"],
                                    "cause_scenario": ["ato", "ato"]})
    _write(t / "labels.parquet", {"label": ["money_mule"], "subject": ["acc_mule"]})
    _write(t / "events.parquet", {"key": ["e1", "g1"], "cause_kind": ["scenario", "process"]})
    _write(t / "decisions.parquet", {"key": ["g1"] if decline_good else ["e1"], "time": when[:1],
                                     "role": ["brind_defender"], "action": ["decline"], "decision": ["x"]})
    _write(tmp / "owners.parquet", {"entity": ["acc_v", "acc_mule", "brindlemark"],
                                    "institution": ["brindlemark", "brindlemark", "brindlemark"]})
    return t, tmp / "owners.parquet"


def test_scoring_counts_victim_losses_not_laundering_and_prices_friction(tmp_path):
    w = ScoreWeights()
    t, o = _truth(tmp_path / "ref", 100.0, decline_good=False)
    ref = window_score(t, o, institution="brindlemark", role="brind_defender", start="2026-06-01T00:00:00Z",
                       end="2026-08-01T00:00:00Z", weights=w)
    assert ref.losses == 100.0 and ref.friction == 0 and ref.adverse_on_fraud == 1
    t, o = _truth(tmp_path / "cand", 40.0, decline_good=True)
    cand = window_score(t, o, institution="brindlemark", role="brind_defender", start="2026-06-01T00:00:00Z",
                        end="2026-08-01T00:00:00Z", weights=w)
    assert cand.adverse_on_good == 1 and cand.friction == w.decline_fixed
    p = paired(cand, ref, w)
    assert p.losses_prevented == 60.0 and p.composite == 60.0 - w.decline_fixed


# --- queue ---

def test_queue_backs_off_on_usage_limit():
    t = [0.0]
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise UsageLimited("limit", retry_at=100.0)
        return {"ok": True}

    def sleep(s):
        t[0] += s

    q = Queue(Path("/dev/null"), parallel=1, clock=lambda: t[0], sleep=sleep)
    q._note = lambda **k: None
    out = q.run([Job("a", "sub1", flaky)])
    assert out["a"] == {"ok": True} and t[0] >= 100.0


# --- episode with a fake engine ---

class FakeRun:
    def __init__(self, day=0, deployed=None):
        self.day, self.deployed = day, dict(deployed or {})

    def run_until_day(self, d):
        assert d >= self.day
        self.day = d

    def deploy(self, role, policy):
        if "boom" in json.dumps(policy):
            raise ValueError("unknown feature obs.boom")
        self.deployed[role] = policy

    def checkpoint(self):
        return json.dumps({"day": self.day, "deployed": self.deployed}).encode()


class FakeEngine:
    def __init__(self, tmp):
        self.tmp = tmp

    def start(self, cfg): return FakeRun()
    def restore(self, cfg, ck):
        d = json.loads(ck)
        return FakeRun(d["day"], d["deployed"])
    def decisions_of(self, cfg): return {"brind_card_auth"}
    @contextmanager
    def tool_server(self, run, cfg): yield McpServer("arena", url="http://127.0.0.1:1/mcp")
    def own_outcomes(self, run, cfg, a, b): return {"decisions": {"brind_card_auth": {"approve": 10}}}
    def export_scoring(self, run, cfg, out):
        loss = 50.0 if run.deployed else 100.0
        return _truth(out, loss, decline_good=bool(run.deployed))
    def day_iso(self, cfg, day): return "2026-01-01T00:00:00Z" if day < 200 else "2027-01-01T00:00:00Z"


class ScriptedRunner:
    name, model = "scripted", "none"

    def __init__(self, policies):
        self.policies, self.i = policies, 0

    def run(self, ws, prompt, servers, log):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text(self.policies[min(self.i, len(self.policies) - 1)])
        self.i += 1
        return SessionResult("scripted", "none", True, "ok", 0.1, {"output_tokens": 1})


def test_episode_deploys_into_candidate_only_and_reports_refused_deploys(tmp_path):
    cfg = EpisodeConfig(world=Path("w.yaml"), institution="brindlemark", seed=1, start_day=180, period_days=10,
                        horizon_days=30)
    good = "rules: [{decision: brind_card_auth, when: 'obs.amount > 900', action: decline}]\n"
    bad = "rules: [{decision: brind_card_auth, when: 'obs.boom > 1', action: decline}]\n"
    runner = ScriptedRunner([good, bad, good])
    res = Episode(cfg, FakeEngine(tmp_path), runner, tmp_path / "ep").run()
    periods = res["periods"]
    assert [p["boundary"] for p in periods] == [180, 190, 200]
    assert periods[0]["deployed"] and periods[1]["deploy_error"].startswith("ValueError")
    assert res["score"]["losses_prevented"] == 50.0
    assert res["tokens"]["output_tokens"] == 3


def test_episode_resumes_after_usage_limit(tmp_path):
    cfg = EpisodeConfig(world=Path("w.yaml"), institution="brindlemark", seed=1, start_day=180, period_days=10,
                        horizon_days=30)
    good = "rules: [{decision: brind_card_auth, when: 'obs.amount > 900', action: decline}]\n"

    class LimitedOnce(ScriptedRunner):
        def run(self, ws, prompt, servers, log):
            if self.i == 1 and not getattr(self, "hit", False):
                self.hit = True
                raise UsageLimited("limit", None)
            return super().run(ws, prompt, servers, log)

    r = LimitedOnce([good])
    ep = Episode(cfg, FakeEngine(tmp_path), r, tmp_path / "ep")
    with pytest.raises(UsageLimited):
        ep.run()
    state = json.loads((tmp_path / "ep" / "state.json").read_text())
    assert state["next_period"] == 1
    res = ep.run()
    assert [p["boundary"] for p in res["periods"]] == [180, 190, 200]
