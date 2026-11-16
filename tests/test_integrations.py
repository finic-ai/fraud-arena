import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraud_arena import integrations as integ
from fraud_arena.game import Game, GameConfig, Player

from test_game import Attackers, Engine, Run, Scripted

POLICY = {
    "rules": [
        {"decision": "brind_card_auth", "when": "obs.amount < 50", "action": "approve"},
        {"decision": "brind_card_auth", "when": "outside.scam.yes > 0.7 and outside.claims >= 2", "action": "decline"},
        {"decision": "brind_card_auth", "action": "approve"},
    ],
    "integrations": {
        "claims": {"source": "own_data", "sql": "select count(*) from CLAIMS where CUST_ID = $cust",
                   "params": {"cust": "CARD_AUTHS.CUST_ID"}, "default": 0},
        "unused": {"source": "own_data", "sql": "select 1"},
        "scam": {"source": "grayson", "instructions": "scam?", "criteria": {"yes": "a scam", "no": "not"},
                 "data": ["case", "claims"]},
    },
}


def test_only_integrations_the_rules_read_become_engine_features():
    pol = json.loads(json.dumps(POLICY))
    ints = integ.parse(pol)
    assert "integrations" not in pol and set(ints) == {"claims", "unused", "scam"}
    used = integ.declare(pol, ints)
    assert used == {"claims", "scam"} and pol["format"] == 2
    assert pol["outside"] == {"claims": {"type": "number"}, "scam": {"options": ["yes", "no"]}}
    assert pol["external"] == {"brind_card_auth": {}}


def test_a_model_reading_an_integration_makes_its_decision_external():
    pol = {"rules": [{"decision": "d1", "when": "score.m > 0.5", "action": "decline"},
                     {"decision": "d2", "action": "approve"}],
           "models": {"m": {"kind": "logistic", "features": ["amount", "outside.scam.yes"], "weights": [1, 2],
                            "intercept": 0}},
           "integrations": {"scam": {"source": "grayson", "criteria": {"yes": "a", "no": "b"}}}}
    integ.declare(pol, integ.parse(pol))
    assert pol["external"] == {"d1": {}}


@pytest.mark.parametrize("bad, msg", [
    ({"x": {"source": "web"}}, "source is one of"),
    ({"x": {"source": "own_data", "sql": "drop table CLAIMS"}}, "one SELECT"),
    ({"x": {"source": "own_data", "sql": "select $a"}}, "has no entry in params"),
    ({"x": {"source": "grayson", "criteria": {"yes": "y"}}}, "at least two options"),
    ({"x": {"source": "grayson", "criteria": {"a": 1, "b": 2}, "data": ["nope"]}}, "isn't an integration"),
])
def test_bad_integrations_are_refused(bad, msg):
    with pytest.raises(integ.IntegrationError, match=msg):
        integ.parse({"integrations": bad})


def _tables(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"CUST_ID": ["c1", "c1", "c2"], "STATEMENT": ["a", "b", "c"]}), folder / "CLAIMS.parquet")
    (folder / "public").mkdir(exist_ok=True)
    pq.write_table(pa.table({"DOMAIN": ["x.example.com"], "AGE_DAYS": [3]}), folder / "public" / "DOMAINS.parquet")


def test_lookups_see_only_the_institutions_tables(tmp_path):
    _tables(tmp_path / "t")
    secret = tmp_path / "truth.parquet"
    pq.write_table(pa.table({"label": ["fraud"]}), secret)
    d = integ.Data()
    d.load(tmp_path / "t")
    assert d.query("select count(*) from CLAIMS where CUST_ID = $c", {"c": "c1"})[1] == [(2,)]
    assert d.query("select AGE_DAYS from PUBLIC_DOMAINS", {})[1] == [(3,)]
    for sql in (f"select * from read_parquet('{secret}')", "select * from read_csv('/etc/hosts')",
                "set enable_external_access = true"):
        with pytest.raises(Exception):
            d.query(sql, {})


class FakeGrayson:
    def __init__(self, fail=False):
        self.reqs, self.fail = [], fail

    def ask_many(self, reqs):
        self.reqs += reqs
        if self.fail:
            return [{"error": "budget"} for _ in reqs]
        return [{"answers": {"scam": {"choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}}} for _ in reqs]


def test_the_resolver_answers_lookups_and_grayson_with_context(tmp_path):
    _tables(tmp_path / "t")
    d = integ.Data()
    d.load(tmp_path / "t")
    pol = json.loads(json.dumps(POLICY))
    g = FakeGrayson()
    r = integ.Resolver(integ.parse(pol), d, g, case_view=lambda c: {"amount": c["features"]["amount"]})
    cases = [{"key": "k1", "decision": "brind_card_auth", "needs": ["scam", "claims"], "time": 0,
              "features": {"amount": 900, "memo": "raw template text"}, "identifiers": {"CARD_AUTHS.CUST_ID": "c1"}},
             {"key": "k2", "decision": "brind_card_auth", "needs": ["claims"], "time": 0,
              "features": {"amount": 70}, "identifiers": {"CARD_AUTHS.CUST_ID": "nobody"}}]
    out = r.answer(cases)
    assert out == {"k1": {"scam": {"yes": 0.9, "no": 0.1}, "claims": 2.0}, "k2": {"claims": 0.0}}
    assert len(g.reqs) == 1 and g.reqs[0]["data"] == {"case": {"amount": 900}, "claims": [{"count_star()": 2}]}
    bad = integ.Resolver(integ.parse(json.loads(json.dumps(POLICY))), d, FakeGrayson(fail=True))
    assert bad.answer(cases[:1])["k1"]["scam"] == {"yes": 0.5, "no": 0.5}  # no answer: no information


class Asking(Run):
    """An event gate whose policy reads outside features: cases over $100 stop and need `scam`."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.waiting, self.given, self.ran = [], {}, {}

    def run_until_day(self, d):
        if self.waiting:
            raise ValueError("cases wait")
        while self.day < d:
            dep = self.deployed.get("brind_defender") or {}
            if self.ran.get(self.day + 1) is None and "outside" in dep:
                self.ran[self.day + 1] = True
                self.waiting = [{"key": f"k{self.day + 1}-{a}", "decision": "brind_card_auth", "gate": "event",
                                 "needs": ["scam"], "time": 0, "features": {"amount": a, "note": "rent", "statement": "raw template"},
                                 "identifiers": {"CARD_AUTHS.CUST_ID": "c1"}, "actions": ["approve", "decline"],
                                 "default_action": "approve"} for a in (500, 900)]
                return len(self.waiting)
            self.day += 1
        return 0

    def pending_external(self): return list(self.waiting)

    def answer_external(self, answers):
        assert isinstance(answers, dict) and set(answers) == {w["key"] for w in self.waiting}
        self.given.update(answers)
        self.waiting = []

    def fork(self):
        r = Asking(self.day, self.deployed, self.plans)
        r.given, r.ran, r.waiting = dict(self.given), dict(self.ran), list(self.waiting)
        return r

    def undeploy(self, role): self.deployed.pop(role, None)


def test_the_game_answers_integration_features_with_grayson(tmp_path, monkeypatch):
    class E(Engine):
        def start(self, cfg): return Asking()
        def write_live(self, run, cfg, dest, rewrite=None): _tables(dest)
        def decision_points(self, cfg):
            return [{"decision": "brind_card_auth", "role": "brind_defender", "actions": ["approve", "decline"],
                     "default_action": "approve", "features": [{"name": "amount", "kind": "num"},
                                                               {"name": "note", "kind": "str"}]}]

    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text(json.dumps({
            "rules": [{"decision": "brind_card_auth", "when": "obs.amount > 100 and outside.scam.yes > 0.5",
                       "action": "decline"}, {"decision": "brind_card_auth", "action": "approve"}],
            "integrations": {"scam": {"source": "grayson", "instructions": "scam?",
                                      "criteria": {"yes": "a scam", "no": "not"}}}}))

    g = FakeGrayson()
    from fraud_arena import grayson
    monkeypatch.setattr(grayson.Broker, "client", lambda self, player: g)
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=2, horizon_days=2, settle_days=0,
                     manual_review=False)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    game = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=lambda r: None)
    res = game.run()
    dep = game.deployed["brind_defender"]
    assert dep["outside"] == {"scam": {"options": ["yes", "no"]}} and dep["external"] == {"brind_card_auth": {}}
    assert len(g.reqs) == 4 and all(r["data"]["case"]["CARD_AUTHS.CUST_ID"] == "c1" for r in g.reqs)
    assert all("statement" not in r["data"]["case"] for r in g.reqs)  # rich text only once model-written
    assert all(r["data"]["case"]["note"] == "rent" for r in g.reqs)  # routine text, as the tables show it
    assert res["decisioning"]["defender-brindlemark"]["integrations"]["grayson"] == 4
    log = (tmp_path / "out" / "grayson-live" / "defender-brindlemark.jsonl").read_text().splitlines()
    assert len(log) == 4 and json.loads(log[0])["integration"] == "scam"


def test_live_lookups_filter_the_observer_rows_at_the_cases_time(tmp_path):
    from datetime import datetime

    class LiveRun:
        calls = []

        def observer_rows(self, observer, table, where=None, columns=None, limit=None, newest_first=False):
            LiveRun.calls.append((observer, table, where, columns, limit, newest_first))
            return pa.table({"CLAIM_TS": [datetime(2025, 7, 1), datetime(2025, 7, 20)], "AMT": [40.0, 900.0]})

    ints = integ.parse({"integrations": {
        "recent": {"source": "own_data", "table": "CLAIMS", "params": {"cust": "CLAIMS.CUST_ID", "floor": "feature.floor"},
                   "where": {"CUST_ID": "$cust", "CLAIM_TS": {">=": "$now - 90d"}}, "newest_first": True,
                   "sql": "select sum(AMT) from rows where AMT > $floor", "default": 0},
        "site": {"source": "public", "table": "WEBSITES", "where": {"DOMAIN": "x.example.com"}}}})
    with pytest.raises(integ.IntegrationError, match="has no entry in params"):
        integ.parse({"integrations": {"x": {"source": "own_data", "table": "T", "where": {"A": "$b"}}}})
    r = integ.Resolver(ints, integ.Data(), None, observer="brindlemark")
    case = {"key": "k", "decision": "d", "needs": ["recent"], "time": 1754006400000,  # 2025-08-01
            "features": {"floor": 100}, "identifiers": {"CLAIMS.CUST_ID": "BM1"}}
    assert r.answer([case], run=LiveRun()) == {"k": {"recent": 900.0}}
    obs, table, where, cols, limit, newest = LiveRun.calls[0]
    assert (obs, table, newest) == ("brindlemark", "CLAIMS", True) and where["CUST_ID"] == "BM1"
    assert where["CLAIM_TS"][">="] == datetime(2025, 5, 3)  # 90 days before the case, naive UTC
    rows = integ._sql_over(pa.table({"a": [1]}), "select count(*) from rows", {})
    assert rows[1] == [(1,)]
    with pytest.raises(Exception):
        integ._sql_over(pa.table({"a": [1]}), "select * from read_csv('/etc/hosts')", {})
