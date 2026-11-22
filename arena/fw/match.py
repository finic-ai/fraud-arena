"""A Fraud Arena 1v1 match on Fraud World: one defender at Brindlemark against the fraud syndicate (all ten rings).

The world warms up, checkpoints, and two runs continue from the checkpoint: the candidate (the players' policies) and
the reference (the bank's built-in rules and the rings' scripts). Each week the defender's deployment (native rules and
optional models, SPEC §8.8) and the attacker's ring plan (interventions, SPEC §8.9) apply from the week's start; both
runs advance a week and each side is scored against the reference on that week.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import runmere

from . import scoring
from .config import INSTITUTIONS, ScoreWeights

WORLD = Path("/private/var/folders/ry/jh93lvp15pn2dnnynp0t2j8r0000gn/T/rm-main/worlds/fraud-world/world.yaml")
INSTITUTION, ROLE, OBSERVER = "brindlemark", "brind_defender", "brindlemark"
WEIGHTS = ScoreWeights()


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def ring_menu(world_path=WORLD):
    """The ring plan menu from the world's `env.interventions`: attribute -> {min, max} or {values}."""
    import yaml
    spec = yaml.safe_load(Path(world_path).read_text())
    menu = {}
    for name, d in (spec.get("env", {}).get("interventions") or {}).items():
        if d.get("entity") != "ring" or not d["attribute"].startswith("plan_"):
            continue
        if "min" in d:
            menu[d["attribute"]] = {"min": d["min"], "max": d["max"]}
        elif "values" in d:
            menu[d["attribute"]] = {"values": d["values"]}
        else:  # bool or code without a values list on the intervention: read the attribute's declaration
            menu[d["attribute"]] = {"type": "bool_or_code"}
    import re
    text = Path(world_path).read_text()
    for attr, kind, rest in re.findall(r"(plan_[a-z_]+)\??: \{type: (code|bool)([^}]*)\}", text):
        if attr in menu and "min" not in menu[attr]:
            if kind == "bool":
                menu[attr] = {"values": [True, False]}
            elif "values" not in menu[attr]:
                m = re.search(r"values: \[([^\]]*)\]", rest)
                if m:
                    menu[attr] = {"values": [v.strip() for v in m.group(1).split(",")]}
    return menu


def code_lists(world_path=WORLD):
    import yaml
    spec = yaml.safe_load(Path(world_path).read_text())
    return {k: v.get("values", []) for k, v in ((spec.get("vocab") or {}).get("code_lists") or {}).items()}


class Match:
    def __init__(self, out, *, world=WORLD, seed=7, roots=1500, warmup_days=60, week_days=7):
        self.out = Path(out)
        self.world_path, self.seed, self.roots = Path(world), seed, roots
        self.week_days, self.warmup_days = week_days, warmup_days
        self.world = runmere.World.load(str(self.world_path))
        base = self.world.start(seed, 1, roots)
        base.run_until_day(warmup_days)
        ckpt = base.checkpoint()
        self.cand = runmere.Run.restore(self.world, ckpt, roots)
        self.ref = runmere.Run.restore(self.world, ckpt, roots)
        rings = self.cand.entities("ring", ["number"]).to_pylist()
        self.rings = {r["number"]: r["entity"] for r in rings}
        self.week = 0
        self.decisions = self.world.decisions(ROLE)
        lists = code_lists(self.world_path)
        for d in self.decisions:  # attach the allowed values of code features (matched by name, else by size)
            codes = {}
            for name, kind in d["features"]:
                if str(kind).startswith("code"):
                    if name in lists:
                        codes[name] = lists[name]
                    else:
                        n = str(kind)[5:-1]
                        same = [k for k, v in lists.items() if n.isdigit() and len(v) == int(n)]
                        if len(same) == 1:
                            codes[name] = lists[same[0]]
            d["code_values"] = codes
        self.menu = ring_menu(self.world_path)
        self.deployed, self.previous, self.plan, self.runtime_error = None, None, {}, None

    # --- applying the players' choices -------------------------------------------------------------------------

    def deploy(self, policy):
        """Deploy the defender's policy into the candidate run; None restores the built-in rules. Returns an error
        message, or None. A failed deploy leaves the previous policy in place."""
        try:
            self.previous = self.deployed
            if policy is None:
                self.cand.undeploy(ROLE)
            else:
                self.cand.deploy(ROLE, policy)
            self.deployed = policy
            return None
        except Exception as error:
            return f"{type(error).__name__}: {error}"

    def set_plan(self, plan):
        """Apply the attacker's plan: {"all": {attr: value}, "rings": {"<number>": {attr: value}}}. Attributes not
        mentioned keep their current value; null unsets one (the script). Returns a list of problems."""
        problems = []
        for number, ring in self.rings.items():
            values = dict((plan or {}).get("all") or {})
            values.update(((plan or {}).get("rings") or {}).get(str(number)) or {})
            for attr, value in values.items():
                try:
                    self.cand.intervene("ring", ring, attr, value)
                except Exception as error:
                    problems.append(f"ring {number} {attr}={value!r}: {error}")
        self.plan = plan
        return problems

    # --- a week --------------------------------------------------------------------------------------------------

    def play_week(self):
        start_day = self.warmup_days + self.week * self.week_days
        end_day = start_day + self.week_days
        start = iso(self.cand.time)
        self.runtime_error = None
        trial = self.cand.fork()
        try:
            trial.run_until_day(end_day)
            self.cand = trial
        except Exception as error:
            # The defender's rules stopped the run (e.g. ordering a `none` feature): keep the previous policy and
            # play the week from the untouched candidate state.
            self.runtime_error = f"{type(error).__name__}: {error}"
            self.cand.deploy(ROLE, self.previous) if self.previous else self.cand.undeploy(ROLE)
            self.deployed = self.previous
            self.cand.run_until_day(end_day)
        self.ref.run_until_day(end_day)
        end = iso(self.cand.time)
        self.week += 1
        wdir = self.out / "truth" / f"week{self.week}"
        cand_truth, owners = self._export(self.cand, wdir / "candidate")
        ref_truth, _ = self._export(self.ref, wdir / "reference")
        d_c = scoring.window_score(cand_truth, owners, institution=INSTITUTION, role=ROLE, start=start, end=end,
                                   weights=WEIGHTS)
        d_r = scoring.window_score(ref_truth, owners, institution=INSTITUTION, role=ROLE, start=start, end=end,
                                   weights=WEIGHTS)
        defender = scoring.paired(d_c, d_r, WEIGHTS)
        observed = list(INSTITUTIONS)
        att = {"candidate": 0.0, "reference": 0.0, "by_ring": {}}
        for number, ring in self.rings.items():
            c = scoring.ring_score(cand_truth, owners, ring=ring, number=number, institutions=observed, start=start,
                                   end=end, weights=WEIGHTS)
            r = scoring.ring_score(ref_truth, owners, ring=ring, number=number, institutions=observed, start=start,
                                   end=end, weights=WEIGHTS)
            # Ranked: what the syndicate takes from the defended bank only (losses + mule flow at its weight).
            def at_bank(x):
                b = x["by_institution"].get(INSTITUTION, {"losses": 0.0, "mule_flow": 0.0})
                return b["losses"] + WEIGHTS.mule_flow * b["mule_flow"]
            att["candidate"] += at_bank(c)
            att["reference"] += at_bank(r)
            att["all_institutions_candidate"] = att.get("all_institutions_candidate", 0.0) + c["score"]
            att["all_institutions_reference"] = att.get("all_institutions_reference", 0.0) + r["score"]
            att["by_ring"][number] = {"candidate": c, "reference": r}
        result = {
            "week": self.week, "start": start, "end": end,
            "defender": {"score": defender.composite, "prevented": defender.losses_prevented,
                         "added_friction": defender.added_friction, "candidate": d_c.as_dict(),
                         "reference": d_r.as_dict()},
            "attacker": {"score": att["candidate"] - att["reference"], "candidate": att["candidate"],
                         "reference": att["reference"], "by_ring": att["by_ring"],
                         "all_institutions_score": att["all_institutions_candidate"] - att["all_institutions_reference"]},
        }
        return result, cand_truth, owners

    def _export(self, run, out):
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        truth = out / "truth"
        run.write_truth(str(truth))
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
        shop = run.entities("shop_account", ["merchant"])
        for e, m in zip(shop.column("entity").to_pylist(), shop.column("merchant").to_pylist()):
            if m in merchant:
                ents.append(e)
                owner.append(merchant[m])
        for e, n in inst_name.items():
            ents.append(e)
            owner.append(n)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"entity": ents, "institution": owner}), out / "owners.parquet")
        return truth, out / "owners.parquet"

    # --- views ---------------------------------------------------------------------------------------------------

    def defender_view(self, dest):
        """The bank's own tables as of now, as Parquet and one DuckDB file of views."""
        data = Path(dest)
        if data.exists():
            shutil.rmtree(data)
        data.mkdir(parents=True)
        self.cand.write_observer(OBSERVER, str(data))
        db = duckdb.connect(str(data / "bank.duckdb"))
        for f in sorted(data.glob("*.parquet")):
            db.execute(f"CREATE VIEW {f.stem} AS SELECT * FROM read_parquet('{f}')")
        db.close()
        return data

    def attacker_view(self, weekly_results, cand_truth):
        """What the syndicate knows: per week, ring and scheme, the schemes it started and what each ring took from
        each observed institution. Never the bank's rules, alerts or tables."""
        rows = []
        for res in weekly_results:
            for number, rr in res["attacker"]["by_ring"].items():
                for inst, v in rr["candidate"]["by_institution"].items():
                    rows.append({"week": res["week"], "ring": number, "institution": inst,
                                 "taken": round(v["losses"], 2), "mule_flow": round(v["mule_flow"], 2)})
        starts = duckdb.sql(f"""SELECT scenario, count(*) AS started FROM '{cand_truth}/instances.parquet'
                                WHERE scenario LIKE 'r%\\_%' ESCAPE '\\' OR scenario LIKE 'ring\\_%' ESCAPE '\\'
                                GROUP BY 1 ORDER BY 1""").fetchall()
        return {"results": rows, "schemes_started_to_date": dict(starts)}
