"""Scores of a defender over an episode window, from the truth tables and entity attributes of a run (grader side).

- Losses: money that fraud took out of the institution and its customers: scenario-caused postings paid by the
  institution or one of its accounts to anything outside it (refunds, bonuses and credits it pays included), except
  laundering out of mule and ring accounts. Gross: recoveries such as chargebacks aren't netted yet.
- Mule activity: scenario money passing through the institution's accounts (received and paid on), not a loss but
  scored as its own component at `ScoreWeights.mule_flow` per dollar (owner: count it, with a lower weight).
- Friction: the role's adverse actions on good activity (events not caused by a scenario), priced by `ScoreWeights`.
- Displacement: losses at every other institution, observed or background.
A candidate is scored against the reference run of the same seed: differences, never raw levels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from .config import ScoreWeights

# Observed institutions that are merchants: they hand over goods, so a refund or chargeback of a fraudulent sale is
# a loss whatever the sale brought in.
MERCHANTS = ("hopcrate", "kellmont")

ADVERSE = ("decline", "reject", "deny", "refuse", "step_up", "hold", "standard", "review", "raise_alert", "block", "restrict",
           "close_account")


@dataclass
class WindowScore:
    losses: float
    friction: float
    adverse_on_good: int
    adverse_on_fraud: int
    displacement: float
    losses_by_scenario: dict[str, float]
    mule_flow: float = 0.0

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
    # A loss is scenario-caused money leaving an institution: paid by the institution itself or by one of its
    # accounts to anything outside it. Fraud World taints only some flows (`dirty`), so the cause decides, not the
    # taint. Pass-through is not a loss: an account's loss is its scenario-caused outflow beyond the scenario-caused
    # money it received (a mule forwarding what a scam sent it nets to zero, labelled or not), counted as the change
    # over the window of that cumulative net, clipped at zero. Ring and mule accounts never count.
    c.execute(f"""create view launder as select distinct subject from '{T}/labels.parquet'
                  where label in ('ring_account', 'money_mule')""")
    c.execute(f"""create view flows as
        select p.time, po.institution, coalesce(p.cause_scenario, '') scen, p.payer acct, p.amount amt
        from '{T}/postings.parquet' p join owners po on po.entity = p.payer
        left join owners pe on pe.entity = p.payee
        where p.cause_kind = 'scenario' and p.amount > 0
          and (pe.institution is null or pe.institution <> po.institution)
        union all
        select p.time, pe.institution, coalesce(p.cause_scenario, ''), p.payee, -p.amount
        from '{T}/postings.parquet' p join owners pe on pe.entity = p.payee
        where p.cause_kind = 'scenario' and p.amount > 0""")
    # A merchant's fraud inflow is the fraudulent sale itself, which a refund or chargeback pays back while the goods
    # are gone: netting would cancel the merchant's real loss, so merchants' accounts count gross outflow.
    merch = "(" + ", ".join(f"'{m}'" for m in MERCHANTS) + ")"
    rows = c.execute(f"""
        with acct as (
          select acct, any_value(institution) institution,
                 case when any_value(institution) in {merch}
                   then sum(case when amt > 0 and time >= timestamptz '{start}' and time < timestamptz '{end}'
                                 then amt else 0 end)
                   else greatest(sum(case when time < timestamptz '{end}' then amt else 0 end), 0)
                      - greatest(sum(case when time < timestamptz '{start}' then amt else 0 end), 0) end loss
          from flows where acct not in (select subject from launder) group by acct),
        share as (
          select acct, scen, sum(amt) out from flows
          where amt > 0 and time >= timestamptz '{start}' and time < timestamptz '{end}' group by all)
        select a.institution, s.scen, sum(a.loss * s.out / t.total)::double
        from acct a join share s using (acct)
        join (select acct, sum(out) total from share group by acct) t using (acct)
        where a.loss > 0 group by all""").fetchall()
    # Mule activity: scenario money that passed through an institution's accounts (received and paid on), and
    # everything ring and mule accounts paid out. Not a loss, but an institution that moves it carries the risk of
    # it, so it is scored as its own component with its own weight.
    mule_rows = c.execute(f"""
        with a as (
          select acct, any_value(institution) institution, (acct in (select subject from launder)) labelled,
                 sum(case when amt > 0 and time < timestamptz '{end}' then amt else 0 end) out_e,
                 sum(case when amt < 0 and time < timestamptz '{end}' then -amt else 0 end) in_e,
                 sum(case when amt > 0 and time < timestamptz '{start}' then amt else 0 end) out_s,
                 sum(case when amt < 0 and time < timestamptz '{start}' then -amt else 0 end) in_s
          from flows group by acct)
        select institution,
               sum(case when labelled then out_e - out_s else least(in_e, out_e) - least(in_s, out_s) end)::double
        from a group by 1""").fetchall()
    mule = dict(mule_rows)
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
    return WindowScore(own, friction, good, fraud, other, by_scenario, float(mule.get(institution) or 0.0))


@dataclass
class PairedScore:
    losses_prevented: float
    added_friction: float
    displacement: float
    composite: float
    mule_flow_prevented: float
    candidate: WindowScore
    reference: WindowScore

    def as_dict(self):
        d = asdict(self)
        return d


def paired(candidate: WindowScore, reference: WindowScore, weights: ScoreWeights) -> PairedScore:
    prevented = reference.losses - candidate.losses
    friction = candidate.friction - reference.friction
    displaced = candidate.displacement - reference.displacement
    mule = reference.mule_flow - candidate.mule_flow
    composite = (prevented + weights.mule_flow * mule - friction
                 - (displaced if weights.displacement_in_composite else 0.0))
    return PairedScore(prevented, friction, displaced, composite, mule, candidate, reference)
