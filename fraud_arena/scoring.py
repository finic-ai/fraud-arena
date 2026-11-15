"""Scores of a defender over an episode window, from the truth tables and entity attributes of a run (grader side).

- Losses: money that fraud took out of the institution and its customers: postings with a tainted (`dirty`) amount
  whose payer is the institution or one of its accounts, plus the institution's own payouts that a scenario caused
  (refunds, bonuses, credits it can't recover).
- Friction: the role's adverse actions on good activity (events not caused by a scenario), priced by `ScoreWeights`.
- Displacement: losses at every other institution, observed or background.
A candidate is scored against the reference run of the same seed: differences, never raw levels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from .config import ScoreWeights

ADVERSE = ("decline", "reject", "step_up", "hold", "standard", "review", "raise_alert", "block", "restrict",
           "close_account")


@dataclass
class WindowScore:
    losses: float
    friction: float
    adverse_on_good: int
    adverse_on_fraud: int
    displacement: float
    losses_by_scenario: dict[str, float]

    def as_dict(self):
        return asdict(self)


def window_score(truth: Path, accounts_institution: Path, *, institution: str, role: str, start: str, end: str,
                 weights: ScoreWeights) -> WindowScore:
    """`truth`: a folder of `truth/*.parquet` (render layout). `accounts_institution`: Parquet with columns `entity`
    (stable id) and `institution` (the institution entity's id or name) for every account, plus a row for each
    institution entity itself (`entity` = its id, `institution` = its name). `start`/`end`: ISO timestamps."""
    c = duckdb.connect()
    T = str(truth)
    c.execute(f"create view owners as select entity, institution from '{accounts_institution}'")
    c.execute(f"""create view post as select * from '{T}/postings.parquet'
                  where time >= timestamptz '{start}' and time < timestamptz '{end}'""")
    # Tainted money leaving a mule's or a ring's own account is laundering, not a loss; a taken-over victim's
    # account still counts.
    c.execute(f"""create view launder as select distinct subject from '{T}/labels.parquet'
                  where label in ('ring_account', 'money_mule')""")
    rows = c.execute("""
        select o.institution, coalesce(p.cause_scenario, ''),
               sum(case when p.dirty > 0 then p.dirty else p.amount end)::double
        from post p join owners o on o.entity = p.payer
        where (p.dirty > 0 and p.payer not in (select subject from launder))
           or (p.cause_kind = 'scenario' and o.entity = o.institution)
        group by all""").fetchall()
    own, other, by_scenario = 0.0, 0.0, {}
    for inst, scen, amt in rows:
        if inst == institution:
            own += amt
            by_scenario[scen or "untraced"] = by_scenario.get(scen or "untraced", 0.0) + amt
        else:
            other += amt
    dec = c.execute(f"""
        select d.action, e.cause_kind, coalesce(a.amt, 0)::double
        from '{T}/decisions.parquet' d
        left join '{T}/events.parquet' e on e.key = d.key
        left join (select key, max(amount) amt from '{T}/postings.parquet' group by key) a on a.key = d.key
        where d.role = ? and d.time >= timestamptz '{start}' and d.time < timestamptz '{end}'
          and d.action in {ADVERSE}""", [role]).fetchall()
    friction, good, fraud = 0.0, 0, 0
    for action, cause, amount in dec:
        if cause == "scenario":
            fraud += 1
        else:
            good += 1
            friction += weights.action_cost(action, amount)
    return WindowScore(own, friction, good, fraud, other, by_scenario)


@dataclass
class PairedScore:
    losses_prevented: float
    added_friction: float
    displacement: float
    composite: float
    candidate: WindowScore
    reference: WindowScore

    def as_dict(self):
        d = asdict(self)
        return d


def paired(candidate: WindowScore, reference: WindowScore, weights: ScoreWeights) -> PairedScore:
    prevented = reference.losses - candidate.losses
    friction = candidate.friction - reference.friction
    displaced = candidate.displacement - reference.displacement
    composite = prevented - friction - (displaced if weights.displacement_in_composite else 0.0)
    return PairedScore(prevented, friction, displaced, composite, candidate, reference)
