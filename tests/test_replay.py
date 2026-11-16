import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fraud_arena import replay as rp
from fraud_arena.demo import play
from fraud_arena.game import Game, GameConfig, Player

from test_game import Attackers, Engine, Run, Scripted

ROOT = Path(__file__).resolve().parents[1]


def _frames(out):
    return rp.read_frames(Path(out) / "replay" / "replay.jsonl")


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    base = tmp_path_factory.mktemp("demo")
    logs = []
    t0 = time.perf_counter()
    result = play(base / "out", base / "sb", log=logs.append)
    return {"out": base / "out", "sb": base / "sb", "result": result, "logs": logs,
            "seconds": time.perf_counter() - t0}


def test_live_frames_follow_the_match(tmp_path, monkeypatch):
    monkeypatch.setattr(rp.Recorder, "finish", lambda self, **k: None)
    play(tmp_path / "out", tmp_path / "sb")
    frames = _frames(tmp_path / "out")
    meta = frames[0]
    assert meta["t"] == "meta" and meta["status"] == "running" and meta["boundaries"] == list(range(28, 85, 7))
    assert {n["id"] for n in meta["nodes"]} >= {"brindlemark", "dovrex", "hopcrate", "kellmont", "R2", "R7"}
    assert next(n for n in meta["nodes"] if n["id"] == "R2")["player"] == "attacker-r2"
    days = [f for f in frames if f["t"] == "day"]
    assert sorted({f["day"] for f in days if f["run"] == "game"}) == list(range(28, 98))
    assert sorted({f["day"] for f in days if f["run"] == "reference"}) == list(range(28, 98))
    assert all(f["src"] == "live" for f in days)
    r2 = [f["rings"]["R2"] for f in days if f["run"] == "game" and "R2" in f["rings"]]
    assert sum(x["attempts"] for x in r2) > 50 and sum(x["blocked"] for x in r2) > 0
    assert any(k == "c" for f in days for _, _, k, *_ in f["flows"])
    bounds = [f for f in frames if f["t"] == "boundary"]
    assert [b["day"] for b in bounds] == list(range(28, 84, 7)) and len(bounds[0]["turns"]) == 6
    # Frames are in the order the game made them: the boundary before the days it starts.
    assert frames.index(bounds[1]) > frames.index(next(f for f in days if f["day"] == 34 and f["run"] == "game"))


def test_finished_replay_is_exact_and_matches_the_result(demo):
    frames = _frames(demo["out"])
    meta = frames[0]
    assert meta["status"] == "finished" and meta["source"] == "truth" and meta["overhead"]["live_days"] == 140
    days = [f for f in frames if f["t"] == "day"]
    assert all(f["src"] == "truth" for f in days)
    final = next(f for f in frames if f["t"] == "score" and f.get("final"))
    assert final["day"] == 98
    for name, s in demo["result"]["defenders"].items():
        assert final["defenders"][name]["composite"] == round(s["composite"], 2)
    for name, s in demo["result"]["attackers"].items():
        assert final["attackers"][name]["vs_scripted"] == round(s["vs_scripted"], 2)
    mid = [f for f in frames if f["t"] == "score" and not f.get("final")]
    assert [f["day"] for f in mid] == list(range(35, 85, 7))
    assert set(mid[0]["defenders"]) == set(demo["result"]["defenders"]) and set(mid[0]["attackers"]) == {
        "attacker-r2", "attacker-r7"}
    kinds = {k for f in days for _, _, k, *_ in f["flows"]}
    assert kinds == {"g", "f", "c", "x"}
    g = [f for f in days if f["run"] == "game"]
    caught = sum(n for f in g for s, t, k, n, a in f["flows"] if k == "c" and s == "R2" and t == "brindlemark")
    assert caught > 0
    dec = [f["decisions"]["kellmont"]["kell_checkout"] for f in g if "kellmont" in f["decisions"]]
    assert sum(c.get("decline", [0, 0])[0] for c in dec) > 0  # the strict defender's friction
    assert frames[-1]["t"] == "end"
    assert (demo["out"] / "replay" / "replay.jsonl").stat().st_size < 2_000_000
    assert json.loads((demo["out"] / "replay" / "meta.json").read_text())["revision"] == meta["revision"]


def test_turns_say_what_changed(demo):
    bounds = [f for f in _frames(demo["out"]) if f["t"] == "boundary"]
    brind = [next(t for t in b["turns"] if t["player"] == "defender-brindlemark") for b in bounds]
    assert brind[0]["changes"]["changed"] and brind[0]["changes"]["rules"] == 1
    assert brind[1]["changes"]["notes"] == ["Ring R2 moved under 2k; follow it down"]
    hop = [next(t for t in b["turns"] if t["player"] == "defender-hopcrate") for b in bounds]
    assert hop[0]["changes"]["changed"] and not hop[1]["changes"]["changed"]
    r2 = [next(t for t in b["turns"] if t["player"] == "attacker-r2") for b in bounds]
    assert r2[0]["changes"]["changed"] and r2[1]["changes"]["keys"] == ["max_amount", "volume"]
    assert r2[1]["changes"]["plan"]["max_amount"] == 1900


def test_the_replay_never_reaches_a_workspace(tmp_path, demo):
    play(tmp_path / "out", tmp_path / "sb", replay=False)
    assert not (tmp_path / "out" / "replay").exists()
    assert _files(tmp_path / "sb") == _files(demo["sb"])
    assert (demo["out"] / "replay").stat().st_mode & 0o077 == 0


def test_the_replay_command_rebuilds_the_finished_replay(demo, tmp_path):
    from fraud_arena.cli import main

    out = tmp_path / "out"
    shutil.copytree(demo["out"], out)
    before = _frames(out)
    (out / "replay" / "replay.jsonl").unlink()
    (out / "replay" / "meta.json").unlink()
    assert main(["replay", str(out)]) == 0
    after = _frames(out)
    strip = lambda fs: [{k: v for k, v in f.items() if k not in ("revision", "overhead")} for f in fs]
    assert [f for f in strip(after) if f["t"] != "score"] == [f for f in strip(before) if f["t"] != "score"]
    # DuckDB's parallel float sums can differ in the last cent from one run to the next.
    sa, sb = [f for f in after if f["t"] == "score"], [f for f in before if f["t"] == "score"]
    assert [f["day"] for f in sa] == [f["day"] for f in sb]
    for x, y in zip(sa, sb):
        for side in ("defenders", "attackers"):
            assert x[side].keys() == y[side].keys()
            for name in x[side]:
                assert x[side][name] == pytest.approx(y[side][name], abs=0.05)
    assert main(["replay", str(out), "--no-scores"]) == 0
    assert not [f for f in _frames(out) if f["t"] == "score" and not f.get("final")]


def test_a_rebuild_without_truth_keeps_the_live_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(rp.Recorder, "finish", lambda self, **k: None)
    play(tmp_path / "out", tmp_path / "sb")
    shutil.rmtree(tmp_path / "out" / "score")
    stats = rp.rebuild(tmp_path / "out")
    frames = _frames(tmp_path / "out")
    assert stats["source"] == "live" and frames[0]["status"] == "finished"
    assert {f["src"] for f in frames if f["t"] == "day"} == {"live"}
    assert next(f for f in frames if f["t"] == "score")["final"]


def test_a_rolled_back_period_leaves_no_frames(tmp_path):
    class Flaky(Run):
        def run_until_day(self, d):
            if "brind_defender" in self.deployed and "bad" in str(self.deployed["brind_defender"]) and d > 182:
                raise ValueError("brind_card_auth: rule: can't order Null and Float(0.6)")
            self.day = d
        def fork(self): return Flaky(self.day, self.deployed, self.plans)
        def undeploy(self, role): self.deployed.pop(role, None)

    class E(Engine):
        def start(self, cfg): return Flaky()

    def bundle(ws):
        (ws / "bundle").mkdir(parents=True, exist_ok=True)
        (ws / "bundle" / "policy.yaml").write_text(
            "rules: [{decision: brind_card_auth, when: 'obs.bad > 0.6', action: decline}]\n")

    events = []
    cfg = GameConfig(world=Path("w.yaml"), seed=1, start_day=180, period_days=5, horizon_days=10, settle_days=0)
    d = Player("defender-brindlemark", "defender", "brindlemark", Scripted("claude-code", bundle))
    res = Game(cfg, E(), [d], tmp_path / "out", tmp_path / "sb", attacker_api=Attackers(), log=events.append).run()
    assert res["defenders"]
    raw = (tmp_path / "out" / "replay" / "replay.jsonl").read_text()
    assert '"t":"rollback"' not in raw  # the finished replay was rebuilt without it
    assert any(e.get("event") == "bundle_failed" for e in events)
    rp_events = [e for e in events if e.get("event") == "replay"]
    assert rp_events and any("unreadable" in n for n in rp_events[0]["notes"])  # the fake truth has no history
    game_days = sorted(f["day"] for f in _frames(tmp_path / "out") if f["t"] == "day" and f["run"] == "game")
    assert game_days == sorted(set(game_days)) == list(range(180, 190))


def test_read_frames_drops_rolled_back_days(tmp_path):
    f = tmp_path / "r.jsonl"
    lines = [{"t": "meta"}, {"t": "day", "day": 1, "run": "game"}, {"t": "day", "day": 2, "run": "game"},
             {"t": "day", "day": 2, "run": "reference"}, {"t": "rollback", "day": 2},
             {"t": "day", "day": 2, "run": "game", "x": 1}]
    f.write_text("\n".join(json.dumps(x) for x in lines) + '\n{"t": "day", "da')
    got = rp.read_frames(f)
    assert [(x["t"], x.get("day"), x.get("run"), x.get("x")) for x in got] == [
        ("meta", None, None, None), ("day", 1, "game", None), ("day", 2, "reference", None), ("day", 2, "game", 1)]


def test_ring_changes_count_each_transition_once():
    prev = {"a": ("ring-2", "i-b", "attempted", 0.0)}
    now = {"a": ("ring-2", "i-b", "succeeded", 500.0), "b": ("ring-2", "i-x", "blocked", 0.0),
           "c": (None, None, "attempted", 0.0)}
    rings, flows = rp.ring_changes(now, prev, {"ring-2": "R2"},
                                   lambda i: {"i-b": "brindlemark", "i-x": "background"}.get(i, "web"))
    assert rings["R2"]["attempts"] == 1 and rings["R2"]["successes"] == 1 and rings["R2"]["blocked"] == 1
    assert rings["R2"]["loss_observed"] == 500.0 and rings["fraudsters"]["attempts"] == 1
    assert flows == [["R2", "background", "c", 1, 0.0], ["R2", "brindlemark", "f", 1, 500.0]]
    assert rp.ring_changes(now, now, {}, lambda i: "web") == ({}, [])


def test_watch_serves_the_viewer_and_follows_the_replay(tmp_path):
    from fraud_arena.watch import Watch

    rdir = tmp_path / "replay"
    rdir.mkdir()
    (rdir / "replay.jsonl").write_text('{"t":"meta"}\n{"t":"day","day":1}\n{"t":"da')
    w = Watch(tmp_path)
    try:
        get = lambda path: urllib.request.urlopen(w.url + path, timeout=5).read()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(w.url.rsplit("/", 2)[0] + "/replay/replay.jsonl", timeout=5)
        assert b'data-replay="replay/replay.jsonl"' in get("")
        p = json.loads(get("poll?offset=0&file="))
        assert p["text"].count("\n") == 2 and p["reset"]
        with open(rdir / "replay.jsonl", "a") as fh:
            fh.write('y","day":2}\n')
        q = json.loads(get(f"poll?offset={p['offset']}&file={p['file']}"))
        assert q["text"] == '{"t":"day","day":2}\n' and not q["reset"]
        tmp = rdir / "new.jsonl"
        tmp.write_text('{"t":"meta","status":"finished"}\n')
        tmp.rename(rdir / "replay.jsonl")
        r = json.loads(get(f"poll?offset={q['offset']}&file={q['file']}"))
        assert r["reset"] and r["text"].startswith('{"t":"meta","status":"finished"}')
        if (ROOT / "fraud_arena" / "web" / "viewer.js").exists():
            assert len(get("viewer.js")) > 1000
    finally:
        w.close()


def test_the_sample_replay_is_marked_and_readable():
    sample = ROOT / "viewer" / "sample" / "replay.jsonl"
    frames = rp.read_frames(sample)
    assert frames[0]["format"] == rp.FORMAT and frames[0]["synthetic"] is True
    assert "Synthetic" in frames[0]["synthetic_note"]
    assert {"day", "boundary", "score", "end"} <= {f["t"] for f in frames}


def test_the_viewer_bundle_is_built_and_builds():
    js, css = ROOT / "fraud_arena" / "web" / "viewer.js", ROOT / "fraud_arena" / "web" / "viewer.css"
    assert js.exists() and css.exists()
    assert b"fraud-arena-viewer" in js.read_bytes()
    viewer = ROOT / "viewer"
    if not (viewer / "node_modules" / ".bin" / "esbuild").exists() or shutil.which("node") is None:
        pytest.skip("viewer/node_modules isn't installed (cd viewer && npm ci)")
    subprocess.run(["npm", "run", "--silent", "check"], cwd=viewer, check=True, timeout=300)
    out = subprocess.run(["npm", "run", "--silent", "build:stdout"], cwd=viewer, check=True, timeout=300,
                         capture_output=True).stdout
    assert out == js.read_bytes(), "fraud_arena/web/viewer.js is stale: cd viewer && npm run build"
