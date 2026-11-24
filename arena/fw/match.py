"""A Fraud Arena 1v1 match on Fraud World: one defender at Brindlemark against the fraud syndicate (all ten rings).

The world warms up, checkpoints, and two runs continue from the checkpoint: the candidate (the players' policies) and
the reference (the bank's built-in rules and the rings' scripts). Each week the defender's deployment (native rules and
optional models, SPEC §8.8) and the attacker's ring plan (interventions, SPEC §8.9) apply from the week's start; both
runs advance a week and each side is scored against the reference on that week.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import runmere

from . import scoring
from .deployment import LOG, compose, world_bits
from .config import INSTITUTIONS, ScoreWeights
from ..world_source import tier

INSTITUTION, ROLE, OBSERVER = "brindlemark", "brind_defender", "brindlemark"
WEIGHTS = ScoreWeights()
# Victim reports: the share of fraud whose victim reports it, and the delay range in days after it became knowable.
REPORT_SHARE, REPORT_DAYS = 0.8, (2, 10)
# Percent of aggregator sign-ins kept as training cases (enough to learn that the channel is benign).
AGGREGATOR_SAMPLE = 5


# An engine that records a deployed policy's cases itself (`record_cases`, `Run.take_cases`) needs no outside decider
# to log the defender's training data; only a defender's lists still use one.
NATIVE_CASES = hasattr(runmere.Run, "take_cases") and __import__("os").environ.get("FA_NATIVE_CASES", "1") != "0"

def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def ring_menu(world_path):
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


def code_lists(world_path):
    import yaml
    spec = yaml.safe_load(Path(world_path).read_text())
    return {k: v.get("values", []) for k, v in ((spec.get("vocab") or {}).get("code_lists") or {}).items()}


def _run_all(runs, day):
    """Advance runs to `day` in parallel threads; returns each run's exception or None."""
    import threading
    errors = [None] * len(runs)

    def go(i):
        try:
            runs[i].run_until_day(day)
        except Exception as error:  # noqa: BLE001
            errors[i] = error

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(runs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


# Engines with Runmere's file checkpoints (no size cap, streamed to disk) can cache any warm-up.
FILE_CHECKPOINTS = hasattr(runmere.Run, "checkpoint_to") and hasattr(runmere.Run, "restore_from")


def cache_dir(root, world_path, seed, roots, warmup_days):
    """The cache folder of one warm-up: keyed by the world file's content, seed, households and warm-up days."""
    import hashlib
    world = hashlib.sha256(Path(world_path).read_bytes()).hexdigest()[:16]
    return Path(root) / f"w{world}-s{seed}-r{roots}-d{warmup_days}"


class Match:
    def __init__(self, out, *, world=None, seed=7, roots=1500, warmup_days=60, week_days=7, intensity=1.0,
                 history_cache=None, defender_only=False):
        # Defender-only: no attacking agent, so only the two runs the defender's score needs (candidate and
        # no-defense) advance each week; the no-attack and reference runs are skipped.
        self.defender_only = defender_only
        self.out = Path(out)
        self.lists = {}  # the defender's lists: name -> {key, set of values}
        # The 1v1 tier of the Fraud World version pinned in fraud-world.json, unless a world file is given.
        self.world_path, self.seed, self.roots = Path(world or tier("arena")), seed, roots
        self.week_days, self.warmup_days = week_days, warmup_days
        self.world = runmere.World.load(str(self.world_path))
        # Log the bank's cases during the warm-up too (its history, for training): the logging deployment is
        # behaviour-identical to the built-in rules, and it is taken away before the checkpoint.
        self.builtin, self.gates = world_bits(self.world_path, ROLE, "brindlemark_rules")
        self.week_cases = []
        # The warm-up depends only on (world, seed, households, days), so it is built once and shared: every
        # defender playing this seed starts from the identical history. The cache folder is keyed by those settings.
        self.cache_dir = cache_dir(history_cache, self.world_path, seed, roots, warmup_days) if history_cache else None
        restore = self._warm_up(self.cache_dir)
        history = self.week_cases
        self.history_truth = self.out / "truth" / "warmup"
        self.cand, self.ref = restore(), restore()
        self._warm_restore = restore  # one more copy, untouched by the match's settings, for exporting the history
        rings = self.cand.entities("ring", ["number"]).to_pylist()
        self.rings = {r["number"]: r["entity"] for r in rings}
        # Match intensity: from the checkpoint on, every ring works at `intensity` times its usual volume in both
        # runs, so the match has more fraud than the bank's history. The attacker's plan_volume multiplies on top.
        self.intensity = intensity
        # The match's usual pace: every crew at `intensity` x volume and, wherever its schemes allow, aiming at the
        # defended bank. Both runs start the match this way; the attacker's plan changes the candidate from here.
        self.match_baseline = {"plan_where": "planned", "plan_share_brindlemark": 1.0, "plan_share_arbelton": 0.0,
                               "plan_share_cedarmoor": 0.0, "plan_share_dovrex": 0.0, "plan_share_everhollow": 0.0}
        for ring in self.rings.values():
            for run in (self.cand, self.ref):
                run.intervene("ring", ring, "plan_volume", float(intensity))
                for attr, value in self.match_baseline.items():
                    run.intervene("ring", ring, attr, value)
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
        # Targeting is fixed: every crew aims at the defended bank for the whole match, so the attacker can't win by
        # moving to an undefended institution. The attacker's menu has no plan_where or plan_share_* settings.
        self.menu = {a: d for a, d in ring_menu(self.world_path).items() if a not in self.match_baseline}
        # The scripted crews' settings at the match start (after the baseline), restored in the no-attack replay.
        cols = [a for a in self.menu if a in self.cand.entities("ring").column_names] or list(self.menu)
        table = self.cand.entities("ring", ["number"] + cols).to_pylist()
        self.script_values = {r["entity"]: {a: r.get(a) for a in cols} for r in table}
        self.deployed, self.previous, self.plan, self.runtime_error = None, None, {}, None
        self.touched = set()  # every ring attribute the attacker has set so far
        # The case log the logging rule fills (training data), starting with the warm-up's history.
        self.week_cases, self.all_cases = [], history
        warm = self._warm_restore()
        self.last_truth, _ = self._export(warm, self.history_truth)
        del warm, self._warm_restore
        self.cand.deploy(ROLE, compose({}, self.builtin, self.gates, native=NATIVE_CASES))
        if not NATIVE_CASES:
            self.cand.set_outside(self._outside)

    def _warm_up(self, cache):
        """Warm the world up (or restore a cached warm-up) and return a function giving a fresh run at the end of it."""
        import pickle
        roots = self.roots
        ckpt_file = cache / "checkpoint.ckpt" if cache else None
        if cache and (cache / "history_cases.pkl").exists():
            self.week_cases = pickle.loads((cache / "history_cases.pkl").read_bytes())
            print(f"  warm-up restored from cache {cache.name}", flush=True)
            if ckpt_file.exists():
                return lambda: runmere.Run.restore_from(self.world, str(ckpt_file), roots)
            data = (cache / "checkpoint.bin").read_bytes()
            return lambda: runmere.Run.restore(self.world, data, roots)
        base = self.world.start(self.seed, 1, roots)
        base.deploy(ROLE, compose({}, self.builtin, self.gates, native=NATIVE_CASES))
        if not NATIVE_CASES:
            base.set_outside(self._outside)
        base.run_until_day(self.warmup_days)
        self._take(base)
        base.undeploy(ROLE)
        base.set_outside(None)
        if cache and FILE_CHECKPOINTS:
            tmp = cache.parent / f".{cache.name}.{os.getpid()}.tmp"
            tmp.mkdir(parents=True)
            base.checkpoint_to(str(tmp / "checkpoint.ckpt"))
            (tmp / "history_cases.pkl").write_bytes(pickle.dumps(self.week_cases))
            try:
                tmp.rename(cache)  # atomic: other processes see the whole cache or none of it
                print(f"  warm-up built and cached as {cache.name}", flush=True)
            except OSError:  # another process cached the same warm-up first: theirs is identical
                shutil.rmtree(tmp, ignore_errors=True)
            return lambda: runmere.Run.restore_from(self.world, str(ckpt_file), roots)
        try:
            data = base.checkpoint()
        except ValueError as error:  # an engine without file checkpoints caps them at 512 MiB
            print(f"  no shareable checkpoint ({error}); forking in memory, history not cached", flush=True)
            forks = iter([base.fork(), base.fork(), base])
            return lambda: next(forks)
        if cache:
            tmp = cache.parent / f".{cache.name}.{os.getpid()}.tmp"
            tmp.mkdir(parents=True)
            (tmp / "checkpoint.bin").write_bytes(data)
            (tmp / "history_cases.pkl").write_bytes(pickle.dumps(self.week_cases))
            try:
                tmp.rename(cache)
                print(f"  warm-up built and cached as {cache.name}", flush=True)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
        return lambda: runmere.Run.restore(self.world, data, roots)

    def _outside(self, case, view):
        """Answers the candidate run's outside requests: logs the case (the defender's training data)."""
        return self._answer(case, view, log=True, lists=self.lists)

    def _needs_outside(self):
        """Whether the candidate's deployment reads outside features: only on an engine without `take_cases` (the
        logging rule and lists). With native case recording the defender task uses no outside code, and no lists."""
        return not NATIVE_CASES

    def _take(self, run):
        """Logs the cases that `run` recorded (`record_cases`), as `_answer` logs them: event gates only, aggregator
        sign-ins sampled."""
        if not NATIVE_CASES:
            return
        for case in run.take_cases():
            if case.get("gate") != "event":
                continue
            feats = case.get("features") or {}
            if feats.get("aggregator") and \
                    int(__import__("hashlib").sha1(str(case.get("key")).encode()).hexdigest(), 16) % 100 >= AGGREGATOR_SAMPLE:
                continue
            self.week_cases.append({"key": case.get("key"), "decision": case.get("decision"),
                                    "time": case.get("time"), "features": feats,
                                    "identifiers": case.get("identifiers") or {}})

    def _answer(self, case, view, *, log, lists=None):
        needs = case.get("needs", [])
        answers = {}
        if LOG in needs:
            feats = case.get("features") or {}
            if log and feats.get("aggregator") and \
                    int(__import__("hashlib").sha1(str(case.get("key")).encode()).hexdigest(), 16) % 100 >= AGGREGATOR_SAMPLE:
                log = False  # aggregator polls are high-volume and benign: keep a small, deterministic sample
            if log:
                self.week_cases.append({"key": case.get("key"), "decision": case.get("decision"),
                                        "time": case.get("time"), "features": case.get("features") or {},
                                        "identifiers": case.get("identifiers") or {}})
            answers[LOG] = 0.0
        rest = [n for n in needs if n != LOG]
        ids = {k.split(".")[-1].lower(): str(v) for k, v in (case.get("identifiers") or {}).items() if v is not None}
        for n in rest:
            spec = (lists or {}).get(n[5:]) if n.startswith("list_") else None
            if spec is None:
                raise ValueError(f"no outside source for {n}")
            answers[n] = 1.0 if ids.get(spec["key"]) in spec["set"] else 0.0
        return answers

    def _quiet_outside(self, policy):
        """An outside dispatcher for a counterfactual run of `policy`: no case logging, the same lists."""
        lists = self.lists
        return lambda case, view: self._answer(case, view, log=False, lists=lists)

    def _script_plan(self, run):
        """Put every ring of `run` back on the match's usual pace: the scripted crews, no attacker plan."""
        touched = set(self.match_baseline) | {"plan_volume"} | self.touched
        for ring in self.rings.values():
            start = self.script_values.get(ring, {})
            for attr in touched:
                if attr == "plan_volume":
                    value = float(self.intensity)
                elif attr in self.match_baseline:
                    value = self.match_baseline[attr]
                else:
                    value = start.get(attr)
                if value is None and attr not in start:
                    continue
                run.intervene("ring", ring, attr, value)

    def _apply(self, run, policy, lists=None):
        if NATIVE_CASES:  # lists need outside code at decision time, which this task doesn't use
            lists = None
        self.lists = {n: {"key": l["key"], "set": set(l["values"])} for n, l in (lists or {}).items()}
        run.deploy(ROLE, compose(policy or {}, self.builtin, self.gates, lists=self.lists, native=NATIVE_CASES))
        run.set_outside(self._outside if self._needs_outside() else None)

    # --- applying the players' choices -------------------------------------------------------------------------

    def deploy(self, policy, lists=None):
        """Deploy the defender's policy into the candidate run; None restores the built-in rules. Returns an error
        message, or None. A failed deploy leaves the previous policy in place."""
        import copy
        import re
        self.previous, self.previous_lists = self.deployed, getattr(self, "deployed_lists", None)
        policy, dropped = copy.deepcopy(policy), []
        for _ in range(50):
            try:
                self._apply(self.cand, policy, lists)
                self.deployed, self.deployed_lists = policy, lists
                return f"deployed without {len(dropped)} rule(s) the engine refused: " + "; ".join(dropped) if dropped else None
            except Exception as error:
                m = re.match(r"rules\.(\d+)\.", str(error))
                rules = (policy or {}).get("rules") or []
                if not m or int(m.group(1)) >= len(rules):
                    return f"{type(error).__name__}: {error}"
                rules.pop(int(m.group(1)))
                dropped.append(str(error)[:200])
        return "too many refused rules: " + "; ".join(dropped[:3])

    def set_plan(self, plan):
        """Apply the attacker's plan: {"all": {attr: value}, "rings": {"<number>": {attr: value}}}. Attributes not
        mentioned keep their current value; null unsets one (the script). Returns a list of problems."""
        problems = []
        for number, ring in self.rings.items():
            values = dict((plan or {}).get("all") or {})
            values.update(((plan or {}).get("rings") or {}).get(str(number)) or {})
            for attr, value in values.items():
                if attr in self.match_baseline:
                    problems.append(f"ring {number} {attr}: targeting is fixed at Brindlemark")
                    continue
                if attr == "plan_volume":  # relative to the match intensity; unset means the intensity itself
                    from .cli import MAX_RING_VOLUME
                    value = float(self.intensity) * (1.0 if value is None else min(float(value), MAX_RING_VOLUME))
                elif attr.startswith("plan_tactic_") and value is not None:
                    from .cli import MAX_TACTIC
                    value = min(float(value), MAX_TACTIC)
                try:
                    self.cand.intervene("ring", ring, attr, value)
                except Exception as error:
                    problems.append(f"ring {number} {attr}={value!r}: {error}")
        self.plan = plan
        for values in [(plan or {}).get("all") or {}] + list(((plan or {}).get("rings") or {}).values()):
            self.touched |= set(values)
        return problems

    # --- a week --------------------------------------------------------------------------------------------------

    def play_week(self):
        """Play a week. Four runs advance from the week's start:

        - candidate: the defender's policy against the attacker's plan (what happens);
        - no-defense counterfactual: the bank's built-in rules against the same attacker plan;
        - no-attack counterfactual: the defender's policy against the scripted crews (no attacker plan);
        - reference: built-in rules against scripted crews, for context only.

        The defender is scored on what its policy saved against this week's attack (candidate vs no-defense), the
        attacker on what its plan took beyond the scripted crews against this week's defense (candidate vs
        no-attack). Each side is then judged on its own choices, not on the other side's.
        """
        start_day = self.warmup_days + self.week * self.week_days
        end_day = start_day + self.week_days
        start = iso(self.cand.time)
        self.runtime_error = None
        week_start = self.cand
        self.week_cases = []

        def forks():
            trial = week_start.fork()
            if self._needs_outside():
                trial.set_outside(self._outside)
            if self.defender_only:
                return trial, None
            no_att = week_start.fork()  # the defender's policy against the scripted crews
            if self._needs_outside():
                no_att.set_outside(self._quiet_outside(self.deployed))
            self._script_plan(no_att)
            return trial, no_att

        trial, no_att = forks()
        no_def = week_start.fork()  # the bank's built-in rules against the same attacker plan
        no_def.undeploy(ROLE)
        no_def.set_outside(None)
        # The four runs advance concurrently: the engine releases the GIL while it simulates.
        runs = [trial, no_def] if self.defender_only else [trial, no_att, no_def, self.ref]
        errors = _run_all(runs, end_day)
        if self.defender_only:  # same positions as the four-run case: trial, no_att, no_def, ref
            errors = [errors[0], None, errors[1], None]
        if errors[0] or errors[1]:
            # The defender's rules stopped the run (e.g. ordering a `none` feature): keep the previous policy and
            # play the week again from the untouched week-start state.
            self.runtime_error = f"{type(errors[0] or errors[1]).__name__}: {errors[0] or errors[1]}"
            self.week_cases = []
            self._apply(week_start, self.previous, getattr(self, "previous_lists", None))
            self.deployed, self.deployed_lists = self.previous, getattr(self, "previous_lists", None)
            trial, no_att = forks()
            again = _run_all([r for r in (trial, no_att) if r is not None], end_day)
            if any(again):
                raise next(e for e in again if e)
        if errors[2] or errors[3]:
            raise errors[2] or errors[3]
        self._take(trial)
        self.cand = trial
        end = iso(self.cand.time)
        self.week += 1
        wdir = self.out / "truth" / f"week{self.week}"
        cand_truth, owners = self._export(self.cand, wdir / "candidate")
        self.all_cases += self.week_cases
        self.last_truth = cand_truth
        truths = {"no_defense": self._export(no_def, wdir / "no_defense")[0]}
        if not self.defender_only:
            truths["no_attack"] = self._export(no_att, wdir / "no_attack")[0]
            truths["reference"] = self._export(self.ref, wdir / "reference")[0]
        bank = lambda t: scoring.window_score(t, owners, institution=INSTITUTION, role=ROLE, start=start, end=end,  # noqa: E731
                                              weights=WEIGHTS)
        d_c, d_nd = bank(cand_truth), bank(truths["no_defense"])
        defender = scoring.paired(d_c, d_nd, WEIGHTS)
        if self.defender_only:
            result = {
                "week": self.week, "start": start, "end": end,
                "defender": {"score": defender.composite, "prevented": defender.losses_prevented,
                             "added_friction": defender.added_friction, "candidate": d_c.as_dict(),
                             "no_defense": d_nd.as_dict(), "no_attack": None, "reference": None,
                             "vs_reference": None},
                "attacker": None,
            }
            return result, cand_truth, owners
        d_na, d_r = bank(truths["no_attack"]), bank(truths["reference"])
        vs_reference = scoring.paired(d_c, d_r, WEIGHTS)

        def at_bank(x):  # what a ring takes from the defended bank: losses + mule flow at its weight
            b = x["by_institution"].get(INSTITUTION, {"losses": 0.0, "mule_flow": 0.0})
            return b["losses"] + WEIGHTS.mule_flow * b["mule_flow"]

        takes = {"candidate": 0.0, "no_attack": 0.0, "no_defense": 0.0, "reference": 0.0}
        by_ring, all_inst = {}, {"candidate": 0.0, "no_attack": 0.0}
        for number, ring in self.rings.items():
            per = {}
            for name, t in (("candidate", cand_truth), *truths.items()):
                per[name] = scoring.ring_score(t, owners, ring=ring, number=number, institutions=list(INSTITUTIONS),
                                               start=start, end=end, weights=WEIGHTS)
                takes[name] += at_bank(per[name])
            all_inst["candidate"] += per["candidate"]["score"]
            all_inst["no_attack"] += per["no_attack"]["score"]
            by_ring[number] = {"candidate": per["candidate"], "no_attack": per["no_attack"]}
        result = {
            "week": self.week, "start": start, "end": end,
            "defender": {"score": defender.composite, "prevented": defender.losses_prevented,
                         "added_friction": defender.added_friction, "candidate": d_c.as_dict(),
                         "no_defense": d_nd.as_dict(), "no_attack": d_na.as_dict(), "reference": d_r.as_dict(),
                         "vs_reference": vs_reference.composite},
            "attacker": {"score": takes["candidate"] - takes["no_attack"], "candidate": takes["candidate"],
                         "no_attack": takes["no_attack"], "no_defense": takes["no_defense"],
                         "reference": takes["reference"], "by_ring": by_ring,
                         "all_institutions_score": all_inst["candidate"] - all_inst["no_attack"]},
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
        self._write_cases(data)
        db = duckdb.connect(str(data / "bank.duckdb"))
        for f in sorted(data.glob("*.parquet")):
            db.execute(f"CREATE VIEW {f.stem} AS SELECT * FROM read_parquet('{f}')")
        db.close()
        return data

    def _write_cases(self, data):
        """One table per decision of the cases the bank decided during the match: observed features, identifiers,
        the action taken, and `fraud_known`: true only for fraud the bank has discovered by now (truth's discovery
        time), as a fraud team's labels are, or reported by its victim: REPORT_SHARE of fraud cases are reported
        REPORT_DAYS after the fraud became knowable (a dispute, a chargeback, a customer's call). Undiscovered,
        unreported fraud reads false, as it would at a bank."""
        if not self.all_cases or self.last_truth is None:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        T = self.last_truth
        now = iso(self.cand.time)
        # A case's key is its own id, not the decision's key: cases join the decision log on (decision, time).
        known = {(d, t) for d, t in duckdb.sql(f"""
            SELECT DISTINCT d.decision, epoch_ms(d.time) FROM '{T}/decisions.parquet' d
            JOIN '{T}/events.parquet' e ON e.key = d.key
            JOIN '{T}/labels.parquet' l ON l.cause_instance = e.cause_instance
            WHERE d.role = '{ROLE}' AND e.cause_label = 'fraud' AND (
              (l.discovered_at IS NOT NULL AND l.discovered_at <= TIMESTAMPTZ '{now}')
              OR (l.knowable_at IS NOT NULL AND hash(e.cause_instance || ':reported') % 1000 < {int(REPORT_SHARE * 1000)}
                  AND l.knowable_at + to_days(CAST({REPORT_DAYS[0]} + hash(e.cause_instance || ':delay')
                      % {REPORT_DAYS[1] - REPORT_DAYS[0] + 1} AS INTEGER)) <= TIMESTAMPTZ '{now}'))""").fetchall()}
        actions = {(d, t): a for d, t, a in duckdb.sql(
            f"SELECT decision, epoch_ms(time), action FROM '{T}/decisions.parquet' WHERE role = '{ROLE}'").fetchall()}
        by_decision = {}
        for c in self.all_cases:
            dt = (c["decision"], int(c["time"]))
            row = {"key": c["key"], "ts": iso(c["time"]), "action": actions.get(dt), "fraud_known": dt in known}
            for k, v in c["identifiers"].items():
                row["id_" + k.split(".")[-1].lower()] = v
            for k, v in c["features"].items():
                row[k] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
            by_decision.setdefault(c["decision"], []).append(row)
        for decision, rows in by_decision.items():
            cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("key", "ts", "action", "fraud_known"), k))
            table = pa.Table.from_pylist([{k: r.get(k) for k in cols} for r in rows])
            pq.write_table(table, data / f"cases_{decision.replace('brind_', '')}.parquet")

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
