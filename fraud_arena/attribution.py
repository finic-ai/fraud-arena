"""Which fraud instances a ring's operations caused, from a run's truth (grader side): its own schemes (`r<N>_*`)
and the shared schemes anchored at the ring, its fraudsters, or what they operate, control or recruited. One rule for
the attacker's score (`scoring.ring_score`) and for which texts the attacker writes (`textgen.RowCauses`)."""
from __future__ import annotations

from pathlib import Path


def ring_instances(c, truth: Path, *, ring: str | None = None, number: int | None = None) -> set[str]:
    T = str(truth)
    if ring is None:
        ids = [e for (e,) in c.execute(f"""select entity from '{T}/hidden.parquet'
               where type = 'ring' and attribute = 'number' and try_cast(value as bigint) = ?""",
                                       [number]).fetchall()]
    else:
        ids = [ring]
    fraudsters = {e for (e,) in c.execute(f"""
        select distinct entity from '{T}/hidden.parquet' where type = 'fraudster'
          and attribute in ('ring', 'second_ring') and value in (select unnest(?::varchar[]))""", [ids]).fetchall()}
    members = set(ids) | fraudsters | {e for (e,) in c.execute(f"""
        select distinct entity from '{T}/hidden.parquet'
        where (attribute in ('ring', 'second_ring') and value in (select unnest(?::varchar[])))
           or (attribute in ('operator', 'controlled_by', 'recruited_by', 'fraudster', 'herder')
               and value in (select unnest(?::varchar[])))""", [ids, sorted(fraudsters)]).fetchall()}
    prefix = f"r{number}_" if number is not None else None
    return {i for (i,) in c.execute(f"""
        select instance from '{T}/instances.parquet'
        where anchor in (select unnest(?::varchar[])) or (? is not null and starts_with(scenario, ?))""",
                                    [sorted(members), prefix, prefix or ""]).fetchall()}
