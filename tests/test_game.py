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
    def write_tables(self, run, cfg, tag=None): pass
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
        pq.write_table(pa.table({"instance": ["i0", "i1"], "anchor": ["x", "ring-7"]}), t / "instances.parquet")
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
    assert a["stolen"] == 80.0 and a["stolen_by_scripted_ring"] == 50.0 and a["vs_scripted"] == 30.0
    assert res["tokens"]["attacker-r7"]["output_tokens"] == 4
