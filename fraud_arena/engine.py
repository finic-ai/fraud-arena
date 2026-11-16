"""Binds `Episode` to Runmere's Python run (`runmere.Run`, SPEC §8.8, runmere PR feat/deployable-policies)."""
from __future__ import annotations

import json
import re
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

from .config import INSTITUTIONS, EpisodeConfig
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
            roles = [r for r in cfg.roles if any(x["role"] == r for x in w.decisions())]
            for d in [x for r in roles for x in w.decisions(r)]:
                feats = [{"name": f[0], "kind": f[1]} if isinstance(f, (list, tuple)) else f for f in d["features"]]
                out.append({"decision": d["name"], "role": d["role"], "actions": d["actions"],
                            "default_action": d["default_action"],
                            "observation": d.get("observation"), "params": d.get("params"), "features": feats})
            return out
        env = self.spec(cfg)["env"]
        out = []
        for name, d in env["decisions"].items():
            if d.get("role") not in cfg.roles:
                continue
            obs = env["observations"].get(d.get("observation"), {})
            out.append({"decision": name, "role": d.get("role"), "at": d.get("at"), "actions": d.get("actions"),
                        "default_action": d.get("default_action"), "features": sorted((obs.get("features") or {})),
                        "tables": obs.get("tables", [])})
        return out

    META_LABELS = {"incident", "ring_member", "ring_account", "credentials_phished", "phishing_campaign"}

    def fraud_families(self, cfg) -> set[str]:
        """The world's fraud families, as its labels name them (for typology labels)."""
        text = Path(str(cfg.world)).read_text()
        names = set(re.findall(r"label:\s*\{\s*name:\s*([a-z_0-9]+)", text))
        names |= set(re.findall(r"^\s*label:\s*([a-z_0-9]+)\s*$", text, re.M))
        return names - self.META_LABELS

    def decisions_of(self, cfg) -> set[str]:
        return {d["decision"] for d in self.decision_points(cfg)}

    def role_of(self, cfg) -> dict[str, str]:
        return {d["decision"]: d["role"] for d in self.decision_points(cfg)}

    def day_iso(self, cfg, day: int) -> str:
        epoch = datetime.fromisoformat(self.spec(cfg)["clock"]["epoch"].replace("Z", "+00:00"))
        return (epoch + timedelta(days=day)).astimezone(timezone.utc).isoformat()

    def _agent_dir(self, cfg, name: str) -> Path:
        d = self.sandbox_root / f"{cfg.institution}-s{cfg.seed}" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _tables_dir(self, run, cfg, tag: str | None) -> Path:
        return self._agent_dir(cfg, f"{tag + '-' if tag else ''}data-day{int(run.day)}")

    PUBLIC_OBSERVER = "public"

    def write_tables(self, run, cfg, tag: str | None = None) -> Path:
        """The institution's tables as of now, once per boundary, where the player's user can read them, and the
        public data anyone can look up (the ad library, public profiles and listings, domains and websites, the block
        explorer) under `public/`, served as `PUBLIC_<TABLE>`: a fraud team investigates with what's public too."""
        data = self._tables_dir(run, cfg, tag)
        if not any(data.iterdir()):
            run.write_observer(cfg.observer, str(data))
            try:
                (data / "public").mkdir(exist_ok=True)
                run.write_observer(self.PUBLIC_OBSERVER, str(data / "public"))
            except Exception:  # a world without a public observer
                pass
        return data

    @contextmanager
    def tool_server(self, run, cfg, *, workspace: Path | None = None, tag: str | None = None):
        data = self.write_tables(run, cfg, tag)
        meta = self._agent_dir(cfg, f"{tag + '-' if tag else ''}meta")
        (meta / "decisions.json").write_text(json.dumps(self.decision_points(cfg), indent=2, default=str))
        fmt = meta / "deploy-format.md"
        fmt.write_text(resources.files("fraud_arena").joinpath("briefs/deploy-format.md").read_text())
        ws = workspace or self.sandbox_root / f"{cfg.institution}-s{cfg.seed}" / "workspace"
        yield McpServer("arena", command=self.python, args=(
            "-m", "fraud_arena.toolserver", "--data", str(data), "--decisions", str(meta / "decisions.json"),
            "--workspace", str(ws), "--format", str(fmt)))

    SIGNALS = re.compile(r"DISPUTE|CHARGEBACK|CLAIM|RETURN|COMPLAINT|ALERT|FRAUD|TICKET|REFUND|CASE|REVIEW|SAR|REPORT")

    def fraud_signals(self, data: Path, previous: Path | None, cfg, end_day: int) -> dict:
        """What the institution newly learned about fraud since its last turn, from its own tables: for each table
        that records fraud reports, disputes, returns, claims, complaints, alerts or reviews, the rows (and amounts)
        that weren't there at the previous boundary. At the first boundary, the rows of the last 30 days. Only the
        institution's own records, as of now."""
        c = duckdb.connect()
        out = {}
        for f in sorted(Path(data).glob("*.parquet")):
            name = f.stem.upper()
            if not self.SIGNALS.search(name):
                continue
            cols = c.execute("describe select * from read_parquet(?)", [str(f)]).fetchall()
            amt = next((n for n, t, *_ in cols if (n.endswith("_AMT") or n.endswith("AMOUNT"))
                        and t.split("(")[0] in ("DOUBLE", "DECIMAL", "FLOAT", "BIGINT", "INTEGER")), None)
            agg = f'count(*), coalesce(sum("{amt}"), 0)' if amt else "count(*), 0"
            now = c.execute(f"select {agg} from read_parquet(?)", [str(f)]).fetchone()
            old = (0, 0)
            prev = previous / f.relative_to(data) if previous else None
            if prev is not None and prev.exists():
                old = c.execute(f"select {agg} from read_parquet(?)", [str(prev)]).fetchone()
            elif previous is None:
                ts = next((n for n, t, *_ in cols if t.startswith("TIMESTAMP")), None)
                if ts is None:
                    continue
                old = c.execute(f'select {agg} from read_parquet(?) where "{ts}" < ?::timestamptz',
                                [str(f), self.day_iso(cfg, end_day - 30)]).fetchone()
            rec = {"new_rows": int(now[0] - old[0]), "total_rows": int(now[0])}
            if amt:
                rec[f"new_{amt.lower()}"] = round(float(now[1] - old[1]), 2)
            if rec["total_rows"]:
                out[name] = rec
        return out

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
        # Who bears a posting: a bank or exchange account is its institution's, except that an account a merchant
        # of the game (Hopcrate, Kellmont) holds is that merchant's (its refunds and payouts are its losses); the
        # merchants' businesses and their customers' shop accounts are theirs too.
        inst = run.entities("institution")
        name_col = "name" if "name" in inst.column_names else inst.column_names[1]
        inst_name = dict(zip(inst.column("entity").to_pylist(), inst.column(name_col).to_pylist()))
        bus = run.entities("business", ["brand_key"])
        merchant = {e: k for e, k in zip(bus.column("entity").to_pylist(), bus.column("brand_key").to_pylist())
                    if k in INSTITUTIONS}
        ents, owner = [], []
        acc = run.entities("account", ["institution", "holder"])
        for e, i, h in zip(acc.column("entity").to_pylist(), acc.column("institution").to_pylist(),
                           acc.column("holder").to_pylist()):
            ents.append(e)
            owner.append(merchant.get(h) or inst_name.get(i, i))
        for e, k in merchant.items():
            ents.append(e)
            owner.append(k)
        try:
            shop = run.entities("shop_account", ["merchant"])
            for e, m in zip(shop.column("entity").to_pylist(), shop.column("merchant").to_pylist()):
                if m in merchant:
                    ents.append(e)
                    owner.append(merchant[m])
        except Exception:  # a world without shop accounts
            pass
        for e, n in inst_name.items():
            ents.append(e)
            owner.append(n)
        owners = pa.table({"entity": ents, "institution": owner})
        pq.write_table(owners, out / "owners.parquet")
        return truth, out / "owners.parquet"
