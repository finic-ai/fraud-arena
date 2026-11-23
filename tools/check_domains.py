"""Check that the invented company domains in a build's bank tables are not registered.

    python tools/check_domains.py cases/dev-s11 [cases/dev-s12 ...]

Collects every domain of an exported email address or website that is not one of the mapped mail providers
(arena/tasks/export.PROVIDERS), then asks DNS for its NS/SOA records and whois for a registration. A domain with
either is reported; add an override for it in export.COMPANY_OVERRIDES (or COMPANY_DOMAINS) and rebuild."""

import glob
import subprocess
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.tasks.export import EMAIL_COLUMNS, PROVIDERS, URL_COLUMNS  # noqa: E402

MAIL = {d for opts in PROVIDERS.values() for d, _ in opts}


def domains(case_dir):
    db = duckdb.connect()
    out = set()
    for f in glob.glob(str(Path(case_dir) / "bank" / "*.parquet")):
        t = Path(f).stem
        for c in EMAIL_COLUMNS.get(t, []):
            out |= {r[0] for r in db.execute(f"""SELECT DISTINCT lower(split_part("{c}", '@', 2)) FROM read_parquet('{f}')
                                                 WHERE "{c}" LIKE '%@%'""").fetchall()}
        for c in URL_COLUMNS.get(t, []):
            out |= {r[0] for r in db.execute(f"""SELECT DISTINCT lower("{c}") FROM read_parquet('{f}')
                                                 WHERE "{c}" IS NOT NULL""").fetchall()}
    return sorted(out - MAIL)


def registered(dom):
    dns = any(subprocess.run(["dig", "+short", kind, dom], capture_output=True, text=True, timeout=20).stdout.strip()
              for kind in ("NS", "SOA"))
    try:
        w = subprocess.run(["whois", dom], capture_output=True, text=True, timeout=30).stdout.lower()
        who = "domain name:" in w and "no match" not in w and "not found" not in w
    except (subprocess.TimeoutExpired, FileNotFoundError):
        who = False
    return dns or who


def main(dirs):
    doms = sorted({d for c in dirs for d in domains(c)})
    bad = [d for d in doms if registered(d)]
    for d in doms:
        print(f"{'REGISTERED' if d in bad else 'free':10} {d}")
    print(f"{len(doms)} company domains, {len(bad)} registered")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
