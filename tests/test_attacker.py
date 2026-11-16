"""The attacker side: the plan menu from the world's declarations, validation, the ring view's allow-list, and one
boundary with a fake runner."""
import json
import os
from pathlib import Path

import pytest

from fraud_arena.attacker import menu, operator, view
from fraud_arena.runners import AgentRunner

WORLD = Path(os.environ.get("FRAUD_WORLD", "/var/folders/ry/jh93lvp15pn2dnnynp0t2j8r0000gn/T/m6-rings/worlds/"
                                           "fraud-world/world.yaml"))
needs_world = pytest.mark.skipif(
    not WORLD.exists() or "interventions:" not in WORLD.read_text(), reason="a Fraud World with ring plans")


@needs_world
def test_every_ring_has_a_schema_from_the_declarations():
    assert menu.check_menu(menu.declarations(WORLD)) == []
    for ring in menu.RINGS:
        s = menu.plan_schema(WORLD, ring)
        assert set(s["properties"]["tactics"]["properties"]) == set(menu.RINGS[ring]["tactics"])
        assert ("where" in s["properties"]) == bool(menu.RINGS[ring]["where"])


@needs_world
def test_validation_refuses_what_the_menu_lacks():
    s = menu.plan_schema(WORLD, 1)
    assert menu.validate_plan({"volume": 2, "where": {"brindlemark": 1}, "identities": "synthetic"}, s) == []
    bad = menu.validate_plan({"volume": 5, "where": {"dovrex": 1}, "tactics": {"romance": 1}, "text": "hi"}, s)
    assert any("volume" in e for e in bad) and any("dovrex" in e for e in bad)
    assert any("romance" in e for e in bad) and any("text" in e for e in bad)
    assert menu.validate_plan({"where": {"brindlemark": 0, "arbelton": 0}}, s) != []


def test_a_plan_sets_every_attribute_and_unset_keys_go_back_to_scripted():
    sets = dict(menu.interventions({"quiet": True}, 4))
    assert sets["plan_quiet"] is True and sets["plan_volume"] is None and sets["plan_where"] == "scripted"
    assert sets["plan_tactic_romance"] is None and sets["plan_devices"] == "scripted"
    full = dict(menu.interventions({"where": {"brindlemark": 0.8}, "tactics": {"bank_hopping": 2}}, 1))
    assert full["plan_where"] == "planned" and full["plan_share_brindlemark"] == 0.8
    assert full["plan_share_arbelton"] == 0.0 and full["plan_tactic_bank_hopping"] == 2
    assert len(menu.interventions({}, 1)) == len(menu.interventions({}, 9))


def snap(incidents, accounts=()):
    ring = {"id": "r1", "number": 1, "mules": 3, "mule_target": 15, "launched_at": "2025-01-01T00:00:00Z",
            "plan_mule_target": None, **{f"{k}_{i}": 0.0 for i in view.INSTITUTIONS for k in ("ok", "fail")},
            **{f"seen_{i}": "2025-01-01T00:00:00Z" for i in view.INSTITUTIONS}, **{k: None for k in view.PLAN_ATTRS}}
    ring["ok_brindlemark"], ring["seen_brindlemark"] = 2.5, "2025-03-01T00:00:00Z"
    return {"ring": ring, "incidents": incidents, "accounts": list(accounts)}


def test_the_view_has_only_the_rings_own_knowledge():
    i = lambda f, inst, res, loss: {"family": f, "institution": inst, "result": res, "loss": loss}
    before = snap({"a": i("bank_hopping", "brindlemark", "succeeded", 100.0)})
    now = snap({"a": i("bank_hopping", "brindlemark", "succeeded", 100.0),
                "b": i("bank_hopping", "brindlemark", "blocked", 0.0),
                "c": i("bank_hopping", "arbelton", "attempted", 0.0)},
               [{"institution": "brindlemark", "status": "closed", "restricted": False},
                {"institution": "brindlemark", "status": "open", "restricted": True}])
    v = view.ring_view(now, before, ring=1, day=70, period_start=63, epoch_ms=1735689600000)
    assert set(v) == view.VIEW_KEYS
    assert [(a["institution"], a["attempts"], a["blocked"], a["pending"]) for a in v["attempts"]] == [
        ("arbelton", 1, 0, 1), ("brindlemark", 1, 1, 0)]
    assert v["attempts_total"][1]["taken"] == 100.0
    assert v["memory"]["brindlemark"] == {"recent_successes": 2.5, "recent_failures": 0.0, "last_heard_day": 59}
    assert v["memory"]["arbelton"]["last_heard_day"] is None
    assert v["accounts"] == {"brindlemark": {"open": 0, "restricted": 1, "closed": 1}}
    with pytest.raises(ValueError):
        view.check_view({**v, "alerts": []})


class FakeRunner(AgentRunner):
    name = "fake"

    def __init__(self, plan):
        super().__init__("fake-model")
        self.plan = plan

    def run(self, workspace, prompt, servers, log):
        assert servers == [] and "plan.json" in prompt and (workspace / "view.json").exists()
        if self.plan is not None:
            (workspace / "plan.json").write_text(json.dumps(self.plan))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"modelUsage": {"claude-opus-test": {}}}))
        from fraud_arena.runners import SessionResult
        return SessionResult("fake", self.model, True, "done", 1.0, {"input_tokens": 10, "output_tokens": 5}, "s1",
                             None, log)


@needs_world
@pytest.mark.parametrize("plan,scripted", [({"volume": 0.5, "devices": "spread"}, False), ({"volume": 9}, True),
                                           (None, True)])
def test_one_boundary(tmp_path, plan, scripted):
    cfg = operator.OperatorConfig(ring=1)
    v = view.ring_view(snap({}), None, ring=1, day=7, period_start=0, epoch_ms=1735689600000)
    r = operator.run_boundary(cfg, FakeRunner(plan), workspace=tmp_path / "ws", logs=tmp_path / "logs", world=WORLD,
                              view=v, day=7, period_days=7, boundary=1, boundaries=4)
    assert r.used_scripted is scripted and r.plan == ({} if scripted else plan)
    assert r.model_version == "claude-opus-test" and r.usage["input_tokens"] == 10
    rec = json.loads((tmp_path / "logs" / "attacker.jsonl").read_text().splitlines()[-1])
    assert rec["ring"] == 1 and rec["model"] == "fake-model"


class FakeRun:
    """What the panel reads and writes of a `runmere.Run`."""

    def __init__(self):
        self.set = []

    def rows(self, entity_type, attrs, where_attr=None, equals=None):
        if entity_type == "ring":
            r = snap({})["ring"]
            return [{"id": "r1", **{a: r.get(a) for a in attrs}}] if equals == 1 else []
        if entity_type == "institution":
            return [{"id": "i1", "name": "brindlemark"}]
        return []

    def intervene(self, entity_type, entity, attr, value):
        self.set.append((entity_type, entity, attr, value))


@needs_world
def test_the_panel_runs_operated_rings_and_resumes(tmp_path, monkeypatch):
    from fraud_arena.attacker import panel as panel_mod
    monkeypatch.setattr(panel_mod, "make_runner", lambda cfg: FakeRunner({"quiet": True}))
    p = panel_mod.AttackerPanel([operator.OperatorConfig(ring=1)], world=WORLD, root=tmp_path, epoch_ms=1735689600000)
    run = FakeRun()
    out = p.boundary(run, day=7, period_days=7, boundary=1, boundaries=2)
    assert [r.plan for r in out] == [{"quiet": True}]
    assert ("ring", "r1", "plan_quiet", True) in run.set
    again = panel_mod.AttackerPanel([operator.OperatorConfig(ring=1)], world=WORLD, root=tmp_path,
                                    epoch_ms=1735689600000)
    run2 = FakeRun()
    assert again.boundary(run2, day=7, period_days=7, boundary=1, boundaries=2) == []
    assert run2.set == run.set
    ref = FakeRun()
    again.replay(ref, 7)
    assert ref.set == run.set
