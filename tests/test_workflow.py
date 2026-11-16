import json
from pathlib import Path

import pytest

from fraud_arena.game import Game, GameConfig, Player
from fraud_arena.grayson import Broker, BrokerClient
from fraud_arena.workflow import Workflow, WorkflowError

from test_game import Attackers, Engine, Run, Scripted

WF = '''
DECISIONS = ["brind_card_auth"]
THRESHOLD = 500
def decide(decision, cases):
    print("deciding", len(cases))
    return ["decline" if c["amount"] > THRESHOLD else "approve" for c in cases]
'''


def _bundle(tmp_path, text):
    b = tmp_path / "ws" / "bundle"
    b.mkdir(parents=True)
    (b / "workflow.py").write_text(text)
    return b


def test_a_workflow_decides_a_batch_from_a_frozen_copy(tmp_path):
    b = _bundle(tmp_path, WF)
    wf = Workflow("d", b, tmp_path / "snap", tmp_path / "ws", as_user=None, env={})
    (b / "workflow.py").write_text("raise SystemExit")  # edits after deploy don't reach the running copy
    assert wf.decisions == ["brind_card_auth"]
    cases = [{"amount": 900, "_actions": ["approve", "decline"]}, {"amount": 5, "_actions": ["approve", "decline"]}]
    assert wf.decide("brind_card_auth", cases) == ["decline", "approve"]
    wf.close()
    assert "deciding 2" in (tmp_path / "ws" / ".arena" / "workflow.log").read_text()


def test_a_workflow_that_fails_or_answers_badly_is_refused(tmp_path):
    b = _bundle(tmp_path, "DECISIONS=['x']\ndef decide(d, cases):\n    return ['steal' for c in cases]\n")
    wf = Workflow("d", b, tmp_path / "snap", tmp_path / "ws", as_user=None, env={})
    with pytest.raises(WorkflowError, match="not allowed|allowed"):
        wf.decide("x", [{"_actions": ["approve"]}])
    wf.close()
    b2 = _bundle(tmp_path / "2", "DECISIONS=['x']\ndef decide(d, cases):\n    raise KeyError('amount')\n")
    wf = Workflow("d", b2, tmp_path / "snap2", tmp_path / "ws", as_user=None, env={})
    with pytest.raises(WorkflowError, match="KeyError"):
        wf.decide("x", [{"_actions": ["approve"]}])
    wf.close()
    b3 = _bundle(tmp_path / "3", "import nope\n")
    with pytest.raises(WorkflowError, match="didn't start"):
        Workflow("d", b3, tmp_path / "snap3", tmp_path / "ws", as_user=None, env={})


def test_the_broker_holds_the_budget_and_refuses_strangers(tmp_path, monkeypatch):
    br = Broker(tmp_path / "g", budget=2)
    calls = []
    from fraud_arena import grayson

    def fake(self, data, questions):
        with self._lock:
            used = self.calls_used()
            if used >= self.budget:
                raise grayson.BudgetExhausted("budget used up")
            self._count_path().write_text(json.dumps({"used": used + 1}))
        calls.append(data)
        return {"answers": {"q": {"choice": "yes"}}}
    monkeypatch.setattr(grayson.GraysonClient, "ask", fake)
    c = BrokerClient(br.url, br.token("defender-a"))
    req = {"data": {"x": 1}, "questions": {"q": {"type": "choice"}}}
    out = c.ask_many([req] * 3)
    assert [("error" in o) for o in out].count(True) == 1 and c.calls_used() == 2 and c.budget == 2
    with pytest.raises(RuntimeError, match="403"):
        BrokerClient(br.url, "guess").ask_many([req])
    br.reset_period()
    assert c.calls_used() == 0
    br.close()
    off = Broker(tmp_path / "off", budget=50000, enabled=False)
    o = BrokerClient(off.url, off.token("defender-a")).ask_many([req])
    assert "isn't available" in o[0]["error"] and not calls[3:]
    off.close()


class Gated(Run):
    """Event gate fake: each day one card auth per amount in AMOUNTS waits when the deployed rule sends it out."""
    AMOUNTS = [50, 900]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.waiting, self.answers = [], []

    def run_until_day(self, d):
        while self.day < d:
            if self.waiting:
                return
            self.day += 1
            if "external" in json.dumps(self.deployed.get("brind_defender", {})):
                self.waiting = [{"key": f"k{self.day}-{i}", "decision": "brind_card_auth",
                                 "actions": ["approve", "decline"], "features": {"amount": a}}
                                for i, a in enumerate(self.AMOUNTS)]

    def pending_external(self): return list(self.waiting)

    def answer_external(self, answers):
        self.answers += answers
        self.waiting = []

    def fork(self):
        r = Gated(self.day, self.deployed, self.plans)
        r.answers = list(self.answers)
        return r

    def undeploy(self, role): self.deployed.pop(role, None)


def test_the_game_runs_a_defenders_workflow_at_its_gates(tmp_path):
    class E(Engine):
        def start(self, cfg): return Gated()

    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text("rules: []\n")
        (ws / "bundle" / "workflow.py").write_text(WF)

    events = []
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=3, horizon_days=6, settle_days=0)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    g = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=events.append)
    res = g.run()
    assert g.deployed["brind_defender"]["rules"] == [{"decision": "brind_card_auth", "action": "external"}]
    wf = res["decisioning"]["defender-brindlemark"]["workflow"]
    assert wf["cases"] == 12 and wf["actions"] == {"brind_card_auth:approve": 6, "brind_card_auth:decline": 6}


def test_a_workflow_that_crashes_mid_run_is_taken_back(tmp_path):
    class E(Engine):
        def start(self, cfg): return Gated()

    n = {"i": 0}

    def bundle(ws):
        n["i"] += 1
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text("rules: []\n")
        (ws / "bundle" / "workflow.py").write_text(
            "DECISIONS=['brind_card_auth']\ndef decide(d, cases):\n    return [c['nope'] for c in cases]\n"
            if n["i"] == 1 else WF)

    events = []
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=3, horizon_days=6, settle_days=0)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    g = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=events.append)
    res = g.run()
    failed = [e for e in events if e.get("event") == "bundle_failed"]
    assert failed and "KeyError" in failed[0]["error"]
    assert res["decisioning"]["defender-brindlemark"]["workflow"]["cases"] == 6  # the second week's workflow


def test_no_grayson_game_refuses_grayson_gates_and_says_so(tmp_path):
    from fraud_arena.config import EpisodeConfig
    from fraud_arena.episode import EpisodeState, brief_text
    cfg = GameConfig(world=Path("w.yaml"), seed=1, grayson=False)
    text = brief_text(cfg.for_institution("brindlemark"), 180, {}, EpisodeState(), first=True)
    assert "Grayson isn't available" in text and "fa-grayson ask" not in text
    on = brief_text(GameConfig(world=Path("w.yaml"), seed=1).for_institution("brindlemark"), 180, {},
                    EpisodeState(), first=True)
    assert "50000 calls this period" in on and "workflow.py" in on
