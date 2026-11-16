"""Model-written text for the defenders' tables (v1 item 5, phase 1: per-row text, no attacker briefs).

Fraud World writes each free-text cell from a few templates, so a defender could spot fraud by its template. At each
boundary, after the harness writes an institution's tables and before its tool server starts, `TextGen.rewrite`
replaces every rich text cell (call notes, claim statements, support tickets) with a text a generator writes from
that row alone: a few of the row's own columns and the template text as the gist of what happened. The generator
never sees truth: it reads only the exported folder, and only the columns the column's spec names. Every row, fraud
or ordinary, gets the same prompt and the same generator.

Each text is cached by (generator, prompt version, institution, table, column, row key, facts, template, seed), so a
row keeps its text across boundaries and replays reuse it. Rows are batched (a JSON list in, a JSON list out), every
output is checked (same ids, length, the gist's numbers kept, no real brands, links, emails or phone numbers), a
failed item is retried once, and then falls back to its template.

    python -m fraud_arena.textgen --data DIR --institution brindlemark --generator claude-code:sonnet \
        --cache ~/.cache/fraud-arena/textgen --seed 3 [--out DIR] [--world world.yaml] [--limit 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .runners import ClaudeCodeRunner, CodexRunner, UsageLimited

PROMPT_VERSION = "textgen-v1"
MARKER = ".textgen.json"


@dataclass(frozen=True)
class TextColumn:
    institution: str
    table: str
    column: str
    record: str  # what the cell is, in the institution's records
    writer: str  # who writes it
    facts: tuple[str, ...]  # the row's columns the generator sees
    key: tuple[str, ...]  # the row's columns that identify it across boundaries (never shown to the generator)
    max_chars: int = 900


INSTITUTION_OF = {
    "brindlemark": "Brindlemark Bank, a retail and business bank",
    "dovrex": "Dovrex, a cryptocurrency exchange",
    "hopcrate": "Hopcrate, a food and grocery delivery app (customers, couriers and restaurants)",
    "kellmont": "Kellmont, an online and in-store retailer",
}

# The rich text columns the defenders see. Kellmont has none in Fraud World today.
RICH = [
    TextColumn("brindlemark", "CALLS", "NOTES", "the agent's note on a customer call or chat",
               "a Brindlemark call-centre agent, typing the note right after the contact", ("CHANNEL", "REASON",
               "DURATION_S"), ("CUST_ID", "CALL_TS")),
    TextColumn("brindlemark", "CLAIMS", "STATEMENT", "the customer's own statement on a claim form",
               "the customer filing the claim, in their own words", ("KIND", "AMT"), ("CUST_ID", "CLAIM_TS")),
    TextColumn("dovrex", "SUPPORT_TICKETS", "BODY", "a support ticket's message", "the customer writing to Dovrex "
               "support", ("CATEGORY", "CHANNEL"), ("CUST_ID", "OPENED_TS")),
    TextColumn("hopcrate", "SUPPORT_TICKETS", "BODY", "a support ticket's message", "the customer, courier or "
               "restaurant writing to Hopcrate support", ("CATEGORY", "CHANNEL"), ("CONTACT_EMAIL", "OPENED_TS")),
]

# Short routine text: stays world-written in phase 1 (v1 item 4's library will cover it).
ROUTINE = {("brindlemark", "P2P", "NOTE"), ("brindlemark", "P2P", "MEMO"), ("brindlemark", "CONSUMER_WIRES", "NOTE"),
           ("brindlemark", "TREASURY_APPROVALS", "MEMO"), ("brindlemark", "ACH", "ENTRY_DESC"),
           ("brindlemark", "ACCOUNT_PARTIES", "NOTE"), ("dovrex", "WITHDRAWAL_ADDRESSES", "LABEL"),
           ("hopcrate", "COURIER_PROFILE_CHANGES", "OLD_VALUE"), ("hopcrate", "COURIER_PROFILE_CHANGES", "NEW_VALUE")}


def columns_of(institution: str) -> list[TextColumn]:
    return [c for c in RICH if c.institution == institution]


def world_text_columns(world: Path) -> list[tuple[str, str, str]]:
    """Every (observer, table, column) the world declares as free text (`logical: text`)."""
    import yaml

    spec = yaml.safe_load(Path(world).read_text())
    out = []
    for obs, o in (spec.get("observers") or {}).items():
        for table, t in ((o or {}).get("tables") or {}).items():
            for col, c in ((t or {}).get("columns") or {}).items():
                if isinstance(c, dict) and c.get("logical") == "text":
                    out.append((obs, table, col))
    return out


BRANDS = [
    "chase", "wells fargo", "bank of america", "citibank", "citi", "capital one", "us bank", "u.s. bank", "pnc",
    "truist", "td bank", "hsbc", "barclays", "santander", "ally", "discover", "american express", "amex", "visa",
    "mastercard", "venmo", "zelle", "paypal", "cash app", "cashapp", "square", "stripe", "western union",
    "moneygram", "wise", "revolut", "chime", "robinhood", "coinbase", "binance", "kraken", "gemini",
    "crypto.com", "metamask", "tether", "amazon", "walmart", "target", "costco", "best buy", "home depot",
    "ebay", "etsy", "shopify", "doordash", "uber", "uber eats", "grubhub", "instacart", "postmates", "lyft",
    "fedex", "ups", "usps", "dhl", "apple", "google", "gmail", "microsoft", "outlook", "yahoo", "facebook",
    "instagram", "whatsapp", "telegram", "signal", "tiktok", "twitter", "netflix", "spotify", "irs", "fbi", "ftc",
    "social security administration", "geek squad", "norton", "mcafee", "att", "at&t", "verizon", "t-mobile",
]
_URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|io|co|app|gov|edu|us|info|biz|xyz|ly)\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b|\b1-?8[0-9]{2}\b")
_INST_NAME = re.compile(r"\b((?:[A-Z][\w&'-]*\s+){1,3})(Bank|Credit Union|Exchange|Wallet|Pay|Financial)\b")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z'&-]+")
_LEAD = {"the", "my", "your", "our", "this", "that", "a", "an", "their", "his", "her", "other", "another", "same",
         "local", "online", "new", "old"}


def world_vocab(world: Path | None) -> set[str]:
    """Proper names the world uses: every capitalised word of every string in the world YAML."""
    if world is None:
        return set()
    import yaml

    words: set[str] = set()

    def walk(o):
        if isinstance(o, str):
            words.update(w.lower() for w in _WORD.findall(o) if w[0].isupper())
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(yaml.safe_load(Path(world).read_text()))
    return words


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(s) if len(re.sub(r"\D", "", n)) >= 3}


def check_text(text, gist: str, facts: dict, vocab: set[str], max_chars: int) -> str | None:
    """Why a generated text is refused, or None."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "empty"
    if len(text) > max_chars:
        return "too long"
    if _norm(text) == _norm(gist):
        return "copies the gist"
    known = (gist + " " + " ".join(map(str, facts.values()))).lower()
    for m in _EMAIL.findall(text):
        if m.lower() not in known:
            return "email"
    if _URL.search(_EMAIL.sub("", text)) and not _URL.search(gist):
        return "link"
    for m in _PHONE.findall(text):
        if m not in gist:
            return "phone number"
    low = text.lower()
    for b in BRANDS:
        if re.search(rf"(?<![\w-]){re.escape(b)}(?![\w-])", low) and b not in known and \
                not all(w in vocab for w in b.split()):
            return f"names {b!r}"
    for m in _INST_NAME.finditer(text):
        words = [w.lower() for w in _WORD.findall(m.group(1))]
        if any(w not in vocab and w not in _LEAD and w not in known for w in words):
            return f"names {m.group(0)!r}"
    missing = _numbers(gist) - _numbers(text)
    if missing:
        return f"drops {sorted(missing)[0]}"
    for n in _NUM.findall(text):
        if len(re.sub(r"\D", "", n)) >= 7 and n.replace(",", "") not in _numbers(known):
            return "new long number"
    return None


class GenerationError(Exception):
    pass


class _ClaudeText(ClaudeCodeRunner):
    """`claude -p` with no tools, no MCP and a short system prompt: a plain completion on the subscription."""

    def command(self, workspace, servers, prompt):
        return (["claude", "-p", "--model", self.model, "--output-format", "json", "--tools", "",
                 "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence",
                 "--system-prompt", "You write realistic records for a fictional world. Reply with JSON only."],
                prompt)


class _CodexText(CodexRunner):
    def command(self, workspace, servers, prompt):
        return (["codex", "exec", "--json", "-m", self.model, "-C", str(workspace), "--skip-git-repo-check",
                 "--sandbox", "read-only", "--ignore-user-config", "--ephemeral", "-"], prompt)


class CliGenerator:
    """A generator on a subscription CLI. `generate(prompt)` returns (text, usage); raises `UsageLimited` or
    `GenerationError`."""

    def __init__(self, spec: str, *, as_user=None, proxy=None, extra_env=None, workdir: Path | None = None,
                 timeout_s: int = 900):
        kind, model = spec.split(":", 1)
        cls = {"claude-code": _ClaudeText, "codex": _CodexText}.get(kind)
        if cls is None:
            raise ValueError(f"unknown text generator {spec!r}: claude-code:<model>, codex:<model> or none")
        self.id = spec
        self.runner = cls(model, timeout_s=timeout_s, extra_env=extra_env, as_user=as_user, proxy=proxy)
        self.workdir = Path(workdir or tempfile.gettempdir()) / "fa-textgen"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str) -> tuple[str, dict]:
        ws = Path(tempfile.mkdtemp(dir=self.workdir))
        try:
            res = self.runner.run(ws, prompt, [], ws / "session.log")
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        if not res.ok:
            raise GenerationError(res.error or "generation failed")
        return res.text, res.usage


def make_generator(spec: str | None, **kw):
    if not spec or spec == "none":
        return None
    return CliGenerator(spec, **kw)


def prompt_for(col: TextColumn, items: list[dict]) -> str:
    return f"""You write the free text of records at {INSTITUTION_OF[col.institution]}, in a fictional world.

Each item below is one row of its {col.table} table. Write the row's text: {col.record}, written by {col.writer}.
`gist` is what the record says happened; `facts` are other fields of the same row.

- The gist is a summary, not the text: write the text fresh, in that writer's own words. Never copy the gist.
- Keep the gist's meaning and every number in it (amounts, reference numbers, order and courier ids) exactly.
  Don't add events, outcomes or details that change what happened.
- Write it the way real ones read, and vary it from item to item: length, tone, wording, abbreviations, the odd
  typo or missing punctuation. Items are independent; don't make them sound alike.
- Name only people, companies and places that appear in the gist or facts. No real company or brand names, no
  websites, emails or phone numbers.

Reply with only a JSON array, one object per item, in any order: [{{"id": "<item id>", "text": "<the text>"}}, ...]

Items:
{json.dumps(items, ensure_ascii=False, indent=1)}"""


def parse_reply(text: str) -> dict[str, str]:
    s = text.strip()
    if "[" not in s:
        raise ValueError("no JSON array in the reply")
    s = s[s.index("["): s.rindex("]") + 1]
    out = {}
    for x in json.loads(s):
        if isinstance(x, dict) and "id" in x:
            out[str(x["id"])] = x.get("text")
    return out


@dataclass
class _Row:
    index: int
    key: str
    facts: dict
    gist: str
    text: str | None = None
    fallback: bool = False
    reason: str | None = None


@dataclass
class Stats:
    rows: int = 0
    cache_hits: int = 0
    generated: int = 0
    fallbacks: int = 0
    not_cached: int = 0
    calls: int = 0
    refusals: dict = field(default_factory=dict)


def _atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_table_atomic(table: pa.Table, path: Path):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.chmod(tmp, path.stat().st_mode & 0o777 if path.exists() else 0o664)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _str(v) -> str:
    return "" if v is None else str(v)


class TextGen:
    def __init__(self, generator, cache: Path, *, seed: int, world: Path | None = None,
                 vocab: set[str] | None = None, batch_size: int = 40, workers: int = 4, max_waits: int = 20,
                 sleep=time.sleep, log=None):
        self.gen, self.cache, self.seed = generator, Path(cache), seed
        self.vocab = vocab if vocab is not None else world_vocab(world)
        self.batch_size, self.workers, self.max_waits = batch_size, workers, max_waits
        self.sleep, self.log = sleep, log or (lambda rec: None)
        self._lock = threading.Lock()

    def key(self, col: TextColumn, rowkey: dict, facts: dict, gist: str) -> str:
        blob = json.dumps([self.gen.id, PROMPT_VERSION, col.institution, col.table, col.column, rowkey, facts, gist,
                           self.seed], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache / key[:2] / f"{key}.json"

    def _cached(self, key: str):
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def rewrite(self, data: Path, institution: str, *, out: Path | None = None, limit: int | None = None) -> dict:
        """Rewrites the rich text columns of an institution's exported tables in `data` (or in a copy, `out`)."""
        data = Path(data)
        if out is not None:
            out = Path(out)
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(data, out)
            data = out
        if (data / MARKER).exists():
            return {"skipped": "already rewritten"}
        t0 = time.monotonic()
        usage: dict[str, int] = {}
        tables = {}
        budget = [limit if limit is not None else float("inf")]
        for col in columns_of(institution):
            path = data / f"{col.table}.parquet"
            if not path.exists():
                continue
            table = pq.read_table(path)
            if col.column not in table.column_names:
                continue
            st = Stats()
            rows = self._rows(col, table)
            st.rows = len(rows)
            todo = []
            for r in rows:
                hit = self._cached(r.key)
                if hit is not None:
                    r.text, r.fallback = hit["text"], hit.get("fallback", False)
                    st.cache_hits += 1
                    st.fallbacks += int(r.fallback)
                elif budget[0] > 0:
                    todo.append(r)
                    budget[0] -= 1
            self._generate(col, todo, st, usage)
            texts = table.column(col.column).to_pylist()
            for r in rows:
                if r.text is not None:
                    texts[r.index] = r.text
            i = table.column_names.index(col.column)
            table = table.set_column(i, table.schema.field(i), pa.array(texts, type=table.schema.field(i).type))
            write_table_atomic(table, path)
            tables[col.table] = {"rows": st.rows, "rewritten": sum(r.text is not None and not r.fallback
                                                                    for r in rows),
                                 "cache_hits": st.cache_hits, "generated": st.generated, "fallbacks": st.fallbacks,
                                 "not_generated": sum(r.text is None for r in rows), "calls": st.calls,
                                 "refusals": st.refusals}
        stats = {"institution": institution, "generator": self.gen.id, "tables": tables,
                 "seconds": round(time.monotonic() - t0, 1), "tokens": usage}
        _atomic_write_bytes(data / MARKER, json.dumps({"prompt_version": PROMPT_VERSION,
                                                       "generator": self.gen.id}).encode())
        return stats

    def _rows(self, col: TextColumn, table: pa.Table) -> list[_Row]:
        cols = {c: table.column(c).to_pylist() for c in set(col.facts + col.key + (col.column,))
                if c in table.column_names}
        out = []
        for i in range(table.num_rows):
            gist = cols[col.column][i]
            if not gist:
                continue
            facts = {c: _str(cols[c][i]) for c in col.facts if c in cols}
            rowkey = {c: _str(cols[c][i]) for c in col.key if c in cols}
            out.append(_Row(i, self.key(col, rowkey, facts, gist), facts, gist))
        return out

    def _generate(self, col: TextColumn, rows: list[_Row], st: Stats, usage: dict):
        if not rows:
            return
        batches = [rows[i:i + self.batch_size] for i in range(0, len(rows), self.batch_size)]
        with ThreadPoolExecutor(max(1, min(self.workers, len(batches)))) as pool:
            list(pool.map(lambda b: self._batch(col, b, st, usage), batches))

    def _call(self, col: TextColumn, rows: list[_Row], st: Stats, usage: dict) -> dict[str, str] | None:
        items = [{"id": str(n + 1), "facts": r.facts, "gist": r.gist} for n, r in enumerate(rows)]
        prompt = prompt_for(col, items)
        waits = 0
        while True:
            try:
                text, u = self.gen.generate(prompt)
                break
            except UsageLimited as e:
                waits += 1
                if waits > self.max_waits:
                    raise
                until = e.retry_at or (time.time() + 30 * 60)
                self.log({"event": "textgen_usage_limited", "until": until})
                self.sleep(max(60.0, until - time.time()))
            except GenerationError as e:
                with self._lock:
                    st.calls += 1
                self.log({"event": "textgen_call_failed", "table": col.table, "error": str(e)[:300]})
                return None
        with self._lock:
            st.calls += 1
            for k, v in (u or {}).items():
                usage[k] = usage.get(k, 0) + int(v or 0)
        try:
            return parse_reply(text)
        except (ValueError, json.JSONDecodeError):
            return {}

    def _batch(self, col: TextColumn, rows: list[_Row], st: Stats, usage: dict):
        pending, called_ok = rows, True
        for attempt in range(2):
            reply = self._call(col, pending, st, usage)
            if reply is None:
                called_ok = False
                reply = {}
            else:
                called_ok = True
            failed = []
            for n, r in enumerate(pending):
                t = reply.get(str(n + 1))
                why = check_text(t, r.gist, r.facts, self.vocab, col.max_chars) if t is not None else "missing"
                if why is None:
                    r.text, r.fallback, r.reason = t.strip(), False, None
                else:
                    r.reason = why
                    failed.append(r)
            pending = failed
            if not pending:
                break
        for r in rows:
            if r.text is None:
                r.text, r.fallback = r.gist, True
            with self._lock:
                if r.fallback:
                    st.fallbacks += 1
                    why = (r.reason or "?").split(" ", 1)[0]
                    st.refusals[why] = st.refusals.get(why, 0) + 1
                else:
                    st.generated += 1
            # A fallback after a failed call isn't cached: the next boundary tries again.
            if r.fallback and not called_ok:
                with self._lock:
                    st.not_cached += 1
                continue
            _atomic_write_bytes(self._cache_path(r.key), json.dumps(
                {"text": r.text, "fallback": r.fallback, "reason": r.reason if r.fallback else None},
                ensure_ascii=False).encode())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraud-arena-textgen")
    ap.add_argument("--data", type=Path, required=True, help="an institution's exported tables")
    ap.add_argument("--institution", required=True)
    ap.add_argument("--generator", required=True, help="claude-code:<model> or codex:<model>")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, help="write a rewritten copy here instead of rewriting in place")
    ap.add_argument("--world", type=Path)
    ap.add_argument("--limit", type=int, help="generate at most this many rows")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    gen = make_generator(a.generator)
    tg = TextGen(gen, a.cache, seed=a.seed, world=a.world, batch_size=a.batch_size, workers=a.workers,
                 log=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps(tg.rewrite(a.data, a.institution, out=a.out, limit=a.limit), indent=2))


if __name__ == "__main__":
    main()
