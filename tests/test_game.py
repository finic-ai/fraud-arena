import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fraud_arena.game import Game, GameConfig, Player
from fraud_arena.runners import McpServer, SessionResult


class Run:
    def __init__(self, day=0, deployed=None, plans=None):
        self.day, self.deployed, self.plans = day, dict(deployed or {}), dict(plans or {})

    def run_until_day(self, d): self.day = d
    def fork(self): return Run(self.day, self.deployed, self.plans)
    def deploy(self, role, policy): self.deployed[role] = policy


class Engine:
    def start(self, cfg): return Run()
    def world(self, cfg): return None
    def decisions_of(self, cfg): return {"brind_card_auth"}
    def role_of(self, cfg): return {"brind_card_auth": "brind_defender"}
    def write_tables(self, run, cfg, tag=None): return None
    def fraud_signals(self, data, prev, cfg, b): return {}
    def fraud_families(self, cfg): return {"ato"}
    @contextmanager
    def tool_server(self, run, cfg, workspace=None, tag=None): yield McpServer("arena", url="http://x/mcp")
    def own_outcomes(self, run, cfg, a, b): return {}
    def day_iso(self, cfg, day): return "2025-01-01T00:00:00Z" if day < 200 else "2026-01-01T00:00:00Z"
    def export_scoring(self, run, cfg, out):
        t = out / "truth"
        t.mkdir(parents=True, exist_ok=True)
        when = [datetime(2025, 8, 1, tzinfo=timezone.utc)] * 2
        loss = 40.0 if run.deployed else 100.0
        stolen = 80.0 if run.plans else 50.0
        pq.write_table(pa.table({"key": ["e1", "e2"], "time": when, "payer": ["acc_v", "acc_w"], "payee": ["m", "m"],
                                 "amount": [loss, stolen], "dirty": [loss, stolen], "cause_kind": ["scenario"] * 2,
                                 "cause_scenario": ["ato", "ring_op"], "cause_instance": ["i0", "i1"]}),
                       t / "postings.parquet")
        pq.write_table(pa.table({"instance": ["i0", "i1"], "anchor": ["x", "ring-7"],
                                 "scenario": ["ato", "r7_ach_return"]}), t / "instances.parquet")
        pq.write_table(pa.table({"type": ["fraudster"], "entity": ["f9"], "attribute": ["ring"], "value": ["ring-7"]}),
                       t / "hidden.parquet")
        pq.write_table(pa.table({"label": ["money_mule"], "subject": ["m"]}), t / "labels.parquet")
        pq.write_table(pa.table({"key": ["e1"], "cause_kind": ["scenario"]}), t / "events.parquet")
        pq.write_table(pa.table({"key": ["z"], "time": when[:1], "role": ["brind_defender"], "action": ["approve"],
                                 "decision": ["brind_card_auth"]}), t / "decisions.parquet")
        pq.write_table(pa.table({"entity": ["acc_v", "acc_w", "brindlemark"],
                                 "institution": ["brindlemark", "dovrex", "brindlemark"]}), out / "owners.parquet")
        return t, out / "owners.parquet"


class Attackers:
    def active_rings(self, run, limit): return [7]
    def entity(self, run, ring): return "ring-7"
    def ring_view(self, run, ring, a, b): return {"ring": ring}
    def menu(self, ring): return {"type": "object"}
    def brief(self, **k): return "plan"
    def apply_plan(self, run, ring, plan): run.plans[ring] = plan
    def ring_loss(self, run, ring):
        o = 80.0 if run.plans else (50.0 if run.day > 180 else 0.0)
        return {"observed": o, "background": 1000.0 if run.day > 180 else 0.0}


class Scripted:
    def __init__(self, name, write):
        self.name, self.model, self.write = name, "m", write

    def run(self, ws, prompt, servers, log):
        self.write(ws)
        return SessionResult(self.name, "m", True, "ok", 0.1, {"output_tokens": 2})


def test_game_runs_defenders_and_attackers_together(tmp_path):
    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text("rules: [{decision: brind_card_auth, action: decline}]\n")

    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=10, horizon_days=20)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    g = Game(cfg, Engine(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(),
             attacker_runners=lambda i, ring: Scripted("codex", lambda ws: (ws / "plan.json").write_text('{"volume": 2}')),
             log=lambda r: None)
    res = g.run()
    assert [t["player"] for t in res["turns"]] == ["defender-brindlemark", "attacker-r7"] * 2
    assert res["defenders"]["defender-brindlemark"]["losses_prevented"] == 60.0
    a = res["attackers"]["attacker-r7"]
    # The fake truth: ring-7's instance i1 pays 80 (game) or 50 (reference) from a Dovrex account.
    assert a["score"] == 80.0 and a["score_of_scripted_ring"] == 50.0 and a["vs_scripted"] == 30.0
    assert a["incident_losses_background"] == 1000.0
    assert res["tokens"]["attacker-r7"]["output_tokens"] == 4


def test_a_bundle_that_fails_at_run_time_is_taken_back(tmp_path):
    class Flaky(Run):
        def run_until_day(self, d):
            if "brind_defender" in self.deployed and "bad" in str(self.deployed["brind_defender"]):
                raise ValueError("brind_card_auth: rule: can't order Null and Float(0.6)")
            self.day = d
        def fork(self): return Flaky(self.day, self.deployed, self.plans)
        def undeploy(self, role): self.deployed.pop(role, None)

    class E(Engine):
        def start(self, cfg): return Flaky()

    n = {"i": 0}
    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        n["i"] += 1
        when = "obs.bad > 0.6" if n["i"] == 1 else "obs.amount > 900"
        (ws / "bundle" / "policy.yaml").write_text(f"rules: [{{decision: brind_card_auth, when: '{when}', action: decline}}]\n")

    events = []
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=10, horizon_days=20, settle_days=0)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    g = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=events.append)
    res = g.run()
    assert any(e.get("event") == "bundle_failed" for e in events)
    assert "taken back" in (g.state["defender-brindlemark"].last_error or "") or res["turns"][1]["applied"]


def test_defender_tables_are_rewritten_before_the_tool_server_starts(tmp_path):
    seen = []

    class E(Engine):
        def write_tables(self, run, cfg, tag=None):
            d = tmp_path / f"{tag}-day{run.day}"
            d.mkdir(parents=True, exist_ok=True)
            return d

        @contextmanager
        def tool_server(self, run, cfg, workspace=None, tag=None):
            seen.append(("server", tag))
            yield McpServer("arena", url="http://x/mcp")

    class TG:
        def rewrite(self, data, institution, **kw):
            seen.append(("rewrite", institution, data.name))
            return {"tables": {"CALLS": {"rows": 1}}, "seconds": 0.0, "tokens": {}}

    logs = []
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=10, horizon_days=10,
                     text_generator="fake:1")
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", lambda ws: None))
    Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=logs.append,
         textgen=TG()).run()
    assert seen[:2] == [("rewrite", "brindlemark", "defender-brindlemark-day180"), ("server", "defender-brindlemark")]
    rec = next(r for r in logs if r.get("event") == "textgen")
    assert rec["boundary"] == 180 and rec["tables"]["CALLS"]["rows"] == 1


def test_attacker_briefs_reach_the_next_boundarys_rewrite(tmp_path):
    import yaml
    world = tmp_path / "w.yaml"
    world.write_text(yaml.safe_dump({
        "observers": {"brindlemark": {"tables": {"CALLS": {"grain": {"event": "bank_call"}}}}},
        "world": {"scenarios": {"r7_ach_return": {"emit": "bank_call"}}}}))
    calls, briefs_seen = [], []

    class E(Engine):
        def write_tables(self, run, cfg, tag=None):
            d = tmp_path / f"{tag}-day{run.day}"
            d.mkdir(parents=True, exist_ok=True)
            return d

        def write_truth(self, run, out):
            out.mkdir(parents=True, exist_ok=True)
            calls.append(run.day)

    class TG:
        def rewrite(self, data, institution, causes=None, briefs=None):
            briefs_seen.append(json.loads(json.dumps(briefs)))
            return {"generator": "fake:1", "tables": {"CALLS": {"rows": 2, "generated": 2, "fallbacks": 1,
                                                                "refusals": {"link": 1}}}, "tokens": {}, "seconds": 0}

    class A(Attackers):
        def brief(self, **k):
            briefs_seen.append(("brief", k.get("attacker_text")))
            return "plan"

    def plan_and_brief(ws):
        (ws / "plan.json").write_text("{}")
        (ws / "briefs").mkdir(exist_ok=True)
        (ws / "briefs" / "ach_return.md").write_text("calm callers")
        (ws / "briefs" / "bank_hopping.md").write_text("not a text scheme")

    cfg = GameConfig(world=world, seed=1, start_day=180, period_days=10, horizon_days=20, text_generator="fake:1",
                     attacker_text=True)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", lambda ws: None))
    g = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=A(),
             attacker_runners=lambda i, ring: Scripted("codex", plan_and_brief), log=lambda r: None, textgen=TG())
    res = g.run()
    assert calls == [180, 190]
    assert briefs_seen[0] == {} and ("brief", True) in briefs_seen and briefs_seen[2] == {"7": {"ach_return":
                                                                                               "calm callers"}}
    assert (tmp_path / "out" / "briefs" / "attacker-r7-180" / "ach_return.md").exists()
    assert any("bank_hopping" in n for n in g.brief_notes["attacker-r7"])
    m = res["text"]["by_model"]["fake:1"]
    assert m["rows"] == 4 and m["fallbacks"] == 2 and m["refusals"] == {"link": 2}
    assert "textcheck" in res["text"]
