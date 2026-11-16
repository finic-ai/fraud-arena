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
    merch = "(" + ", ".join(f"'{m}'" for m in MERCHANTS) + ")"
    rows = c.execute(f"""
        with acct as (
          select acct, any_value(institution) institution,
                 greatest(sum(case when time < timestamptz '{end}' then amt else 0 end), 0)
               - greatest(sum(case when time < timestamptz '{start}' then amt else 0 end), 0) loss
          from flows where acct not in (select subject from launder) and institution not in {merch}
          group by acct),
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
        from a where institution not in {"(" + ", ".join(f"'{m}'" for m in MERCHANTS) + ")"} group by 1""").fetchall()
    mule = dict(mule_rows)
    rows = list(rows) + merchant_losses(c, T, merch, start, end) + charge_offs(c, T)
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


def charge_offs(c, T: str) -> list[tuple[str, str, float]]:
    """An institution's charge-offs: what ring and mule accounts owe it when the run ends (an ACH return or a
    chargeback reversing money they already moved on leaves them overdrawn). Their own flows are left out of the
    netting, so this is the only place that loss shows. Earlier charge-offs are the same in the reference run."""
    if not (Path(T) / "balances.parquet").exists():
        return []
    return c.execute(f"""
        select o.institution, 'charge_off', sum(-b.closing)::double
        from '{T}/balances.parquet' b join owners o on o.entity = b.entity
        where b.closing < 0 and b.entity in (select subject from launder)
        group by 1""").fetchall()


def merchant_losses(c, T: str, merch: str, start: str, end: str) -> list[tuple[str, str, float]]:
    """A merchant hands over goods, so the fraudulent sale it was paid doesn't offset a refund, chargeback or credit
    it pays out (netting would cancel every merchant loss). Per fraud instance, its scenario-caused payments out of the
    merchant in the window, less:
    - cancellations: a payment to a counterparty that paid the merchant the same amount, in the same instance, within
      the hour before it (an order declined or cancelled is reversed; no goods left);
    - wins: an inflow, after a payment, from the counterparty it paid, of the same amount, matched one to one (a
      chargeback or dispute the merchant wins back). A later order from the same card is a new sale, not a win."""
    cols = {r[0] for r in c.execute(f"describe select * from '{T}/postings.parquet'").fetchall()}
    if "cause_instance" not in cols:
        return []
    rows = c.execute(f"""
        select epoch(p.time), p.payer, p.payee, p.amount::double, p.cause_instance, coalesce(p.cause_scenario, ''),
               po.institution, pe.institution
        from '{T}/postings.parquet' p left join owners po on po.entity = p.payer
        left join owners pe on pe.entity = p.payee
        where p.cause_kind = 'scenario' and p.amount > 0 and p.time < timestamptz '{end}'
          and (po.institution in {merch} or pe.institution in {merch})
        order by p.cause_instance, p.time""").fetchall()
    start_t = c.execute(f"select epoch(timestamptz '{start}')").fetchone()[0]
    by_inst: dict = {}
    for r in rows:
        by_inst.setdefault(r[4], []).append(r)
    out: dict[tuple[str, str], float] = {}
    hour = 3600
    for inst, ps in by_inst.items():
        sales = [r for r in ps if r[7] in MERCHANTS and r[6] != r[7]]
        pays = [r for r in ps if r[6] in MERCHANTS and r[6] != r[7]]
        used_sales, open_pays, loss = set(), [], {}
        for o in pays:
            cancel = next((i for i, s_ in enumerate(sales) if i not in used_sales and s_[1] == o[2]
                           and s_[7] == o[6] and abs(s_[3] - o[3]) < 0.005
                           and 0 <= o[0] - s_[0] <= hour), None)
            if cancel is not None:
                used_sales.add(cancel)
                continue
            open_pays.append(o)
        for w in sales:
            match = next((o for o in open_pays if o[2] == w[1] and o[6] == w[7] and abs(o[3] - w[3]) < 0.005
                          and o[0] < w[0]), None)
            if match is not None:
                open_pays.remove(match)
        for o in open_pays:
            if o[0] >= start_t:
                k = (o[6], o[5])
                out[k] = out.get(k, 0.0) + o[3]
    return [(inst, scen, amt) for (inst, scen), amt in out.items()]


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


def ring_score(truth: Path, owners: Path, *, ring: str, institutions, start: str, end: str,
               weights: ScoreWeights, number: int | None = None) -> dict:
    """What a ring inflicted on the observed institutions, by the defenders' own rules: the losses and the mule flow
    (at its weight) of the postings its operations caused (instances anchored at the ring, its fraudsters or the
    entities it controls), so the attacker's score mirrors what the defenders are charged. Background banks don't
    count."""
    import shutil
    import tempfile
    c = duckdb.connect()
    T = str(truth)
    from .attribution import ring_instances
    insts = ring_instances(c, truth, ring=ring, number=number)
    fraudsters = {e for (e,) in c.execute(f"""
        select distinct entity from '{T}/hidden.parquet' where type = 'fraudster'
          and attribute in ('ring', 'second_ring') and value = ?""", [ring]).fetchall()}
    members = {ring} | fraudsters | {e for (e,) in c.execute(f"""
        select distinct entity from '{T}/hidden.parquet'
        where (attribute in ('ring', 'second_ring') and value = ?)
           or (attribute in ('operator', 'controlled_by', 'recruited_by', 'fraudster', 'herder')
               and value in (select unnest(?::varchar[])))""", [ring, sorted(fraudsters)]).fetchall()}
    tmp = Path(tempfile.mkdtemp(prefix="ring-"))
    try:
        for f in Path(T).glob("*.parquet"):
            (tmp / f.name).symlink_to(f.resolve())
        (tmp / "postings.parquet").unlink()
        if (tmp / "balances.parquet").exists():
            (tmp / "balances.parquet").unlink()
            c.execute(f"""copy (select * from '{T}/balances.parquet' where entity in (select unnest(?::varchar[])))
                          to '{tmp}/balances.parquet' (format parquet)""", [sorted(members)])
        c.execute(f"""copy (select * from '{T}/postings.parquet' where cause_instance in (select unnest(?::varchar[])))
                      to '{tmp}/postings.parquet' (format parquet)""", [sorted(insts)])
        per = {}
        for inst in institutions:
            w = window_score(tmp, owners, institution=inst, role="-", start=start, end=end, weights=weights)
            per[inst] = {"losses": w.losses, "mule_flow": w.mule_flow}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    losses = sum(v["losses"] for v in per.values())
    mule = sum(v["mule_flow"] for v in per.values())
    return {"losses": losses, "mule_flow": mule, "score": losses + weights.mule_flow * mule,
            "by_institution": per, "instances": len(insts)}
