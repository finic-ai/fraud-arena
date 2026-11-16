"""Binds `Episode` to Runmere's Python run (`runmere.Run`, SPEC §8.8, runmere PR feat/deployable-policies)."""
from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .config import EpisodeConfig
from .runners import McpServer


class RunmereEngine:
    def __init__(self, sandbox_root: Path, *, agent_user: str | None = None, python: str = sys.executable):
        import runmere

        self.rm = runmere
        self.sandbox_root = Path(sandbox_root)
        self.agent_user = agent_user
        self.python = python
        self._worlds = {}
        self._specs = {}

    def world(self, cfg: EpisodeConfig):
        key = str(cfg.world)
        if key not in self._worlds:
            self._worlds[key] = self.rm.World.load(key)
        return self._worlds[key]

    def spec(self, cfg: EpisodeConfig) -> dict:
        key = str(cfg.world)
        if key not in self._specs:
            self._specs[key] = yaml.safe_load(Path(key).read_text())
        return self._specs[key]

    def start(self, cfg):
        return self.world(cfg).start(cfg.seed, cfg.partitions, cfg.roots)

    def restore(self, cfg, checkpoint: bytes):
        return self.rm.Run.restore(self.world(cfg), checkpoint, cfg.roots)

    def decision_points(self, cfg) -> list[dict]:
        w = self.world(cfg)
        if hasattr(w, "decisions"):
            out = []
            for d in w.decisions(cfg.role):
                feats = [{"name": f[0], "kind": f[1]} if isinstance(f, (list, tuple)) else f for f in d["features"]]
                out.append({"decision": d["name"], "actions": d["actions"], "default_action": d["default_action"],
                            "observation": d.get("observation"), "params": d.get("params"), "features": feats})
            return out
        env = self.spec(cfg)["env"]
        out = []
        for name, d in env["decisions"].items():
            if d.get("role") != cfg.role:
                continue
            obs = env["observations"].get(d.get("observation"), {})
            out.append({"decision": name, "at": d.get("at"), "actions": d.get("actions"),
                        "default_action": d.get("default_action"), "features": sorted((obs.get("features") or {})),
                        "tables": obs.get("tables", [])})
        return out

    def decisions_of(self, cfg) -> set[str]:
        return {d["decision"] for d in self.decision_points(cfg)}

    def day_iso(self, cfg, day: int) -> str:
        epoch = datetime.fromisoformat(self.spec(cfg)["world"]["epoch"].replace("Z", "+00:00"))
        return (epoch + timedelta(days=day)).astimezone(timezone.utc).isoformat()

    def _agent_dir(self, cfg, name: str) -> Path:
        d = self.sandbox_root / f"{cfg.institution}-s{cfg.seed}" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @contextmanager
    def tool_server(self, run, cfg):
        """The institution's tables as of now, in a folder the agent's user can read, served over stdio."""
        data = self._agent_dir(cfg, f"data-day{int(run.day)}")
        if not any(data.iterdir()):
            run.write_observer(cfg.observer, str(data))
        meta = self._agent_dir(cfg, "meta")
        (meta / "decisions.json").write_text(json.dumps(self.decision_points(cfg), indent=2, default=str))
        fmt = meta / "deploy-format.md"
        fmt.write_text(resources.files("fraud_arena").joinpath("briefs/deploy-format.md").read_text())
        ws = self.sandbox_root / f"{cfg.institution}-s{cfg.seed}" / "workspace"
        yield McpServer("arena", command=self.python, args=(
            "-m", "fraud_arena.toolserver", "--data", str(data), "--decisions", str(meta / "decisions.json"),
            "--workspace", str(ws), "--format", str(fmt)))

    def own_outcomes(self, run, cfg, start_day: int, end_day: int) -> dict:
        """The institution's own decisions in the period, by decision and action: what it knows it did."""
        dec = run.truth_table("decisions")
        c = duckdb.connect()
        c.register("d", dec)
        rows = c.execute("""select decision, action, count(*) from d
                            where role = ? and time >= ?::timestamptz and time < ?::timestamptz
                            group by all order by 1, 2""",
                         [cfg.role, self.day_iso(cfg, start_day), self.day_iso(cfg, end_day)]).fetchall()
        out: dict = {}
        for d, a, n in rows:
            out.setdefault(d, {})[a] = n
        return {"days": [start_day, end_day], "decisions": out}

    def export_scoring(self, run, cfg, out: Path) -> tuple[Path, Path]:
        out.mkdir(parents=True, exist_ok=True)
        truth = out / "truth"
        if truth.exists():
            shutil.rmtree(truth)
        run.write_truth(str(truth))
        acc = run.entities("account", ["institution"])
        inst = run.entities("institution")
        name_col = "name" if "name" in inst.column_names else inst.column_names[1]
        by_id = dict(zip(inst.column("entity").to_pylist(), inst.column(name_col).to_pylist()))
        owners = pa.table({
            "entity": acc.column("entity").to_pylist() + list(by_id),
            "institution": [by_id.get(i, i) for i in acc.column("institution").to_pylist()] + list(by_id.values()),
        })
        pq.write_table(owners, out / "owners.parquet")
        return truth, out / "owners.parquet"
