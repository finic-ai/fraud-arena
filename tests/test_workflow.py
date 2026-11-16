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
        self.waiting, self.answers, self.ran = [], [], {}

    def run_until_day(self, d):
        """As the engine does: stops before `d` with event-gate cases waiting (returns how many), refuses to go on
        while any is unanswered, and returns 0 at `d`."""
        if self.waiting:
            raise ValueError(f"{len(self.waiting)} event gate case(s) wait for an outside answer")
        while self.day < d:
            if self.ran.get(self.day + 1) is None and "external" in json.dumps(self.deployed.get("brind_defender", {})):
                self.ran[self.day + 1] = True
                self.waiting = [{"key": f"k{self.day + 1}-{i}", "decision": "brind_card_auth", "gate": "event",
                                 "actions": ["approve", "decline"], "default_action": "approve",
                                 "features": {"amount": a}} for i, a in enumerate(self.AMOUNTS)]
                return len(self.waiting)
            self.day += 1
        return 0

    def pending_external(self): return list(self.waiting)

    def answer_external(self, answers):
        answers = list(answers.items()) if isinstance(answers, dict) else list(answers)  # as the engine takes both
        keys = {k for k, _ in answers}
        assert all(w["key"] in keys for w in self.waiting), "every event-gate case needs an answer"
        self.answers += answers
        self.waiting = []

    def fork(self):
        r = Gated(self.day, self.deployed, self.plans)
        r.answers, r.ran, r.waiting = list(self.answers), dict(self.ran), list(self.waiting)
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
    assert g.deployed["brind_defender"]["external"] == {"brind_card_auth": {}}
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


def test_external_entries_fit_their_gate_kind():
    from fraud_arena.bundle import deploy_external

    class R:
        def deploy(self, role, dep):
            ext = dep.get("external", {})
            if "timeout" in ext.get("card", {}):
                raise ValueError("external.card.timeout: `card` is an event gate: its cases stop the run until they "
                                 "are answered, so they have no timeout")
            if "timeout" not in ext.get("alert", {"timeout": 1}):
                raise ValueError("external.alert: `alert` waits for an outside answer, so it needs a `timeout`")
            self.dep = dep
    r = R()
    out = deploy_external(r, "x", {"rules": [], "external": {"card": {"timeout": "2d"}, "alert": {}}})
    assert out["external"] == {"card": {}, "alert": {"timeout": "1d"}} and r.dep == out


def test_rules_sending_cases_nowhere_are_refused(tmp_path):
    class E(Engine):
        def start(self, cfg): return Gated()

    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text("rules: [{decision: brind_card_auth, action: external}]\n")

    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=3, horizon_days=3, settle_days=0)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    g = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=lambda r: None)
    res = g.run()
    assert "neither workflow.py" in res["turns"][0]["error"]


class Queued(Gated):
    """A review queue with two pending cases, worked through interventions; day is fractional (hourly steps)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cases = {"c1": {"id": "c1", "kind": "wire", "amount": 9000, "account": "a1", "case_no": "RV1",
                             "institution": "i-brind", "manual_verdict": "pending"},
                      "c2": {"id": "c2", "kind": "p2p", "amount": 40, "account": "a2", "case_no": "RV2",
                             "institution": "i-brind", "manual_verdict": "pending"}}
        self.marks = []

    def run_until_day(self, d):
        self.marks.append(d)
        self.day = d
        return 0

    def rows(self, etype, attrs, where_attr=None, equals=None):
        if etype == "institution":
            return [{"id": "i-brind", "name": "brindlemark"}]
        if etype == "account":
            return [{"id": "a1", "number": "100001"}, {"id": "a2", "number": "100002"}]
        if etype == "review_desk":
            return [{"id": "desk", "name": "BRINDLEMARK"}]
        rows = [dict(r) for r in self.cases.values()]
        return [r for r in rows if where_attr is None or r.get(where_attr) == equals]

    def intervene(self, etype, eid, attr, value):
        if etype == "review_case":
            self.cases[eid][attr] = value

    def fork(self):
        r = Queued(self.day, self.deployed, self.plans)
        r.cases = {k: dict(v) for k, v in self.cases.items()}
        return r


class QEngine(Engine):
    def start(self, cfg): return Queued()
    def queue_columns(self, cfg):
        return [("CASE_NO", ["case_no"], False), ("CASE_TYPE", ["kind"], True), ("AMT", ["amount"], False),
                ("ACCT_NO", ["account", "number"], False)]
    def spec(self, cfg): return {"entities": {"review_case": {"attributes": {"account": {"type": "ref", "entity": "account"}}}}}
    def write_live(self, run, cfg, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "day.txt").write_text(str(run.day))


def test_a_workflow_works_the_review_queue_hourly_with_queue_rows(tmp_path):
    seen = tmp_path / "seen.jsonl"

    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text("rules: []\n")
        (ws / "bundle" / "workflow.py").write_text(f'''
import json
REVIEW_KINDS = ["wire"]
def review(cases):
    open({str(seen)!r}, "a").write(json.dumps(cases) + "\\n")
    return ["reject" if c["AMT"] > 5000 else None for c in cases]
''')

    events = []
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=1, horizon_days=1, settle_days=0)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    g = Game(cfg, QEngine(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=events.append)
    g.run()
    first = json.loads(seen.read_text().splitlines()[0])
    assert first == [{"CASE_NO": "RV1", "CASE_TYPE": "WIRE", "AMT": 9000, "ACCT_NO": "100001"}]  # wire only, no ids
    assert len(seen.read_text().splitlines()) == 1  # decided at the first hourly look, never asked again
    assert (tmp_path / "sb" / "defender-brindlemark" / "workspace" / "live" / "day.txt").exists()
    assert g.queue_stats["defender-brindlemark"]["decided"] == 1
    assert not any(e.get("event") == "bundle_failed" for e in events)


def test_queue_columns_show_what_the_observer_shows(tmp_path):
    import yaml
    from fraud_arena.engine import RunmereEngine
    from fraud_arena.config import EpisodeConfig
    w = tmp_path / "w.yaml"
    w.write_text(yaml.safe_dump({
        "entities": {"review_case": {"attributes": {
            "case_no": {"type": "string", "visibility": "observable"},
            "kind": {"type": "code", "visibility": "observable"},
            "account": {"type": "ref", "visibility": "observable", "entity": "account"},
            "party": {"type": "ref", "visibility": "hidden", "entity": "person"}}},
            "account": {"attributes": {"number": {"type": "string", "visibility": "observable"},
                                       "owner_secret": {"type": "string", "visibility": "hidden"}}},
            "person": {"attributes": {"name": {"type": "string", "visibility": "observable"}}}},
        "observers": {"brindlemark": {"tables": {"REVIEW_QUEUE": {"columns": {
            "CASE_ID": {"source": {"kind": "key", "of": "review_case"}},
            "CASE_NO": {"source": {"kind": "attribute", "of": "review_case.case_no"}},
            "CASE_TYPE": {"source": {"kind": "attribute", "of": "review_case.kind", "transforms": ["upper"]}},
            "ACCT_ID": {"source": {"kind": "attribute", "of": "review_case.account"}},
            "ACCT_NO": {"source": {"kind": "attribute", "of": "review_case.account.number"}},
            "SECRET": {"source": {"kind": "attribute", "of": "review_case.account.owner_secret"}},
            "PARTY_NAME": {"source": {"kind": "attribute", "of": "review_case.party.name"}}}}}}}}))
    eng = RunmereEngine.__new__(RunmereEngine)  # no runmere build needed to read the world's declarations
    eng._specs = {}
    cfg = EpisodeConfig(world=w, institution="brindlemark", seed=1)
    cols = eng.queue_columns(cfg)
    assert {c for c, _, _ in cols} == {"CASE_NO", "CASE_TYPE", "ACCT_NO"}  # no display ids, nothing hidden
    assert dict((c, u) for c, _, u in cols)["CASE_TYPE"] is True


def test_live_tables_get_the_text_rewrite_and_public_data(tmp_path):
    class Run_:
        day = 181.0
        def write_observer(self, obs, d):
            (Path(d) / f"{obs}.parquet").write_text("x")

    class TG:
        calls = []
        def rewrite(self, folder, inst, causes=None, briefs=None):
            TG.calls.append((Path(folder).name, inst, sorted(p.name for p in Path(folder).rglob("*.parquet"))))
            return {"institution": inst, "tables": {}}

    from fraud_arena.engine import RunmereEngine
    eng = RunmereEngine.__new__(RunmereEngine)
    eng.PUBLIC_OBSERVER = "public"
    from fraud_arena.config import EpisodeConfig
    cfg = EpisodeConfig(world=Path("w"), institution="kellmont", seed=1)
    dest = tmp_path / "ws" / "live"
    eng.write_live(Run_(), cfg, dest, rewrite=lambda f: TG().rewrite(f, "kellmont"))
    assert TG.calls == [("live.tmp", "kellmont", ["kellmont.parquet", "public.parquet"])]  # rewritten before swap
    assert (dest / "public" / "public.parquet").exists() and not (tmp_path / "ws" / "live.tmp").exists()
