import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraud_arena import textcheck, textgen
from fraud_arena.runners import UsageLimited
from fraud_arena.textgen import MARKER, TextGen, check_text, parse_reply, write_table_atomic

VOCAB = {"brindlemark", "hopcrate", "dovrex", "arbelton", "national"}


def items_of(prompt):
    return json.loads(prompt[prompt.index("Items:") + len("Items:"):])


def brief_of(prompt):
    return prompt[prompt.index("Brief:\n<<<\n") + 11:prompt.index("\n>>>")]


class FakeGen:
    """Answers every item with a rewrite of its gist; `bad(gist, attempt)` can make an item's answer fail the filter."""

    id = "fake:1"

    def __init__(self, bad=lambda gist, attempt: None, fail_calls=0, limited=0):
        self.prompts, self.bad, self.fail_calls, self.limited = [], bad, fail_calls, limited
        self.attempts = {}

    def generate(self, prompt):
        from fraud_arena.textgen import GenerationError

        self.prompts.append(prompt)
        if self.limited:
            self.limited -= 1
            raise UsageLimited("usage limit reached", None)
        if self.fail_calls:
            self.fail_calls -= 1
            raise GenerationError("boom")
        tag = "B" if "Ordinary records" not in brief_of(prompt) else "W"
        out = []
        for it in items_of(prompt):
            rec = {"id": it["id"]}
            for col, g in it["gist"].items():
                n = self.attempts[g] = self.attempts.get(g, 0) + 1
                rec[col] = self.bad(g, n) or f"{tag}: {g} ({it['facts'].get('CHANNEL', '')})"
            out.append(rec)
        return "```json\n" + json.dumps(list(reversed(out))) + "\n```", {"input_tokens": 100, "output_tokens": 50}


def calls(n=5, secret="TRUTHSECRET"):
    return pa.table({
        "CALL_ID": [str(i + 1) for i in range(n)],
        "CUST_ID": [f"BM{i:04d}" for i in range(n)],
        "CALL_TS": pa.array([1_700_000_000_000 + i for i in range(n)], pa.timestamp("ms", tz="UTC")),
        "CHANNEL": ["PHONE"] * n,
        "REASON": ["FRAUD_REPORT"] * n,
        "DURATION_S": list(range(100, 100 + n)),
        "SECRET": [secret] * n,
        "NOTES": [f"Caller reports a charge they did not make. Ref {324900 + i}." for i in range(n)],
    })


def data(tmp_path, n=5, **kw):
    d = tmp_path / "brindlemark"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(calls(n, **kw), d / "CALLS.parquet")
    return d


def tg(gen, tmp_path, **kw):
    return TextGen(gen, tmp_path / "cache", seed=3, vocab=VOCAB, sleep=lambda s: None, **kw)


def test_rewrites_rich_text_and_keeps_everything_else(tmp_path):
    d = data(tmp_path)
    before = pq.read_table(d / "CALLS.parquet")
    stats = tg(FakeGen(), tmp_path).rewrite(d, "brindlemark")
    after = pq.read_table(d / "CALLS.parquet")
    assert after.schema == before.schema
    assert after.drop(["NOTES"]).equals(before.drop(["NOTES"]))
    assert all(t.startswith("W: ") for t in after.column("NOTES").to_pylist())
    assert stats["tables"]["CALLS"] == {"rows": 5, "rewritten": 5, "cache_hits": 0, "generated": 5, "fallbacks": 0,
                                        "not_generated": 0, "calls": 1, "from_attacker_briefs": 0, "refusals": {}}
    assert stats["tokens"] == {"input_tokens": 100, "output_tokens": 50}
    assert (d / MARKER).exists()
    assert not [p for p in d.iterdir() if p.name.endswith(".tmp")]


def test_rewriting_twice_is_a_no_op(tmp_path):
    d = data(tmp_path)
    gen = FakeGen()
    tg(gen, tmp_path).rewrite(d, "brindlemark")
    assert tg(gen, tmp_path).rewrite(d, "brindlemark") == {"skipped": "already rewritten"}
    assert len(gen.prompts) == 1


def test_cache_makes_later_boundaries_and_replays_identical(tmp_path):
    d1, d2 = data(tmp_path / "b1"), data(tmp_path / "b2", n=8)
    gen = FakeGen()
    tg(gen, tmp_path).rewrite(d1, "brindlemark")
    stats = tg(gen, tmp_path).rewrite(d2, "brindlemark")
    assert stats["tables"]["CALLS"]["cache_hits"] == 5 and stats["tables"]["CALLS"]["generated"] == 3
    a = pq.read_table(d1 / "CALLS.parquet").column("NOTES").to_pylist()
    b = pq.read_table(d2 / "CALLS.parquet").column("NOTES").to_pylist()
    assert b[:5] == a
    replay = data(tmp_path / "replay")
    other = FakeGen(bad=lambda g, n: "something else entirely, Ref " + re.findall(r"\d+", g)[0])
    tg(other, tmp_path).rewrite(replay, "brindlemark")
    assert other.prompts == [] and pq.read_table(replay / "CALLS.parquet").column("NOTES").to_pylist() == a


def test_cache_key_depends_on_seed_generator_and_prompt_version(tmp_path, monkeypatch):
    d = data(tmp_path / "a")
    gen = FakeGen()
    tg(gen, tmp_path).rewrite(d, "brindlemark")
    TextGen(gen, tmp_path / "cache", seed=4, vocab=VOCAB).rewrite(data(tmp_path / "b"), "brindlemark")
    assert len(gen.prompts) == 2
    import fraud_arena.textgen as t
    monkeypatch.setattr(t, "PROMPT_VERSION", "textgen-test")
    tg(gen, tmp_path).rewrite(data(tmp_path / "c"), "brindlemark")
    assert len(gen.prompts) == 3


def test_batches_use_local_ids_and_cover_every_row(tmp_path):
    d = data(tmp_path, n=23)
    gen = FakeGen()
    stats = tg(gen, tmp_path, batch_size=10, workers=3).rewrite(d, "brindlemark")
    assert stats["tables"]["CALLS"]["calls"] == 3
    sizes = sorted(len(items_of(p)) for p in gen.prompts)
    assert sizes == [3, 10, 10]
    for p in gen.prompts:
        ids = [it["id"] for it in items_of(p)]
        assert ids == [str(i + 1) for i in range(len(ids))]
    notes = pq.read_table(d / "CALLS.parquet").column("NOTES").to_pylist()
    for i, t in enumerate(notes):
        assert f"Ref {324900 + i}" in t


def test_no_column_outside_the_specs_facts_reaches_the_prompt(tmp_path):
    d = data(tmp_path, secret="TRUTHSECRET")
    gen = FakeGen()
    tg(gen, tmp_path).rewrite(d, "brindlemark")
    for p in gen.prompts:
        assert "TRUTHSECRET" not in p and "BM000" not in p and "CALL_TS" not in p
        for it in items_of(p):
            assert set(it) == {"id", "facts", "gist"}
            assert set(it["facts"]) == {"CHANNEL", "REASON", "DURATION_S"} and set(it["gist"]) == {"NOTES"}


def test_every_row_gets_the_same_prompt_text(tmp_path):
    d = data(tmp_path, n=4)
    gen = FakeGen()
    tg(gen, tmp_path, batch_size=1).rewrite(d, "brindlemark")
    heads = {p[:p.index("Items:")] for p in gen.prompts}
    assert len(heads) == 1


@pytest.mark.parametrize("text,why", [
    ("Customer says call 555-123-4567 about Ref 324900.", "phone number"),
    ("See www.example.com for Ref 324900.", "link"),
    ("Customer emailed bob@mail.net about Ref 324900.", "email"),
    ("Customer moved the money to Chase last week. Ref 324900.", "names"),
    ("Customer sent it with Venmo. Ref 324900.", "names"),
    ("Funds went to Pinecrest Bank. Ref 324900.", "names"),
    ("Customer reports a charge.", "drops"),
    ("", "empty"),
    ("caller reports a charge - ref 324900", "copies"),
    ("x" * 2000 + " Ref 324900", "too long"),
])
def test_filter_refuses(text, why):
    got = check_text(text, "Caller reports a charge. Ref 324900.", {"CHANNEL": "PHONE"}, VOCAB, 900)
    assert got is not None and got.startswith(why)


def test_filter_accepts_world_names_and_the_gists_own_details():
    gist = "Wire of $1,774.80 to Arbelton National Bank. Ref 324900."
    assert check_text("Cust confirmed the $1,774.80 wire to Arbelton National Bank, ref 324900.", gist, {}, VOCAB,
                      900) is None
    assert check_text("Talked to the customer at Brindlemark Bank about it. Ref 324900", "Ref 324900", {}, VOCAB,
                      900) is None


def test_failed_items_are_retried_once_then_fall_back_to_the_template(tmp_path):
    d = data(tmp_path, n=3)
    first_bad = lambda g, n: "Call 555-123-4567" if "324900" in g and n == 1 else None
    always_bad = lambda g, n: "Ask at www.real-bank.com" if "324901" in g else None
    gen = FakeGen(bad=lambda g, n: first_bad(g, n) or always_bad(g, n))
    stats = tg(gen, tmp_path).rewrite(d, "brindlemark")
    notes = pq.read_table(d / "CALLS.parquet").column("NOTES").to_pylist()
    assert notes[0].startswith("W: ") and notes[2].startswith("W: ")
    assert notes[1] == "Caller reports a charge they did not make. Ref 324901."
    assert stats["tables"]["CALLS"]["fallbacks"] == 1 and stats["tables"]["CALLS"]["refusals"] == {"link": 1}
    assert len(gen.prompts) == 2 and len(items_of(gen.prompts[1])) == 2


def test_a_failed_call_falls_back_without_caching(tmp_path):
    d = data(tmp_path, n=2)
    stats = tg(FakeGen(fail_calls=2), tmp_path).rewrite(d, "brindlemark")
    assert stats["tables"]["CALLS"]["fallbacks"] == 2
    gen = FakeGen()
    stats = tg(gen, tmp_path).rewrite(data(tmp_path / "next", n=2), "brindlemark")
    assert stats["tables"]["CALLS"]["generated"] == 2 and stats["tables"]["CALLS"]["cache_hits"] == 0


def test_usage_limits_wait_and_retry(tmp_path):
    slept = []
    gen = FakeGen(limited=2)
    TextGen(gen, tmp_path / "cache", seed=3, vocab=VOCAB, sleep=slept.append).rewrite(data(tmp_path), "brindlemark")
    assert len(slept) == 2 and len(gen.prompts) == 3


def test_limit_caps_generated_rows(tmp_path):
    d = data(tmp_path, n=6)
    stats = tg(FakeGen(), tmp_path).rewrite(d, "brindlemark", limit=4)
    assert stats["tables"]["CALLS"]["generated"] == 4 and stats["tables"]["CALLS"]["not_generated"] == 2


def test_rewrite_into_a_copy_leaves_the_source(tmp_path):
    d = data(tmp_path)
    before = pq.read_table(d / "CALLS.parquet")
    tg(FakeGen(), tmp_path).rewrite(d, "brindlemark", out=tmp_path / "copy")
    assert pq.read_table(d / "CALLS.parquet").equals(before)
    assert not (d / MARKER).exists()


def test_atomic_write_keeps_the_old_file_when_writing_fails(tmp_path, monkeypatch):
    d = data(tmp_path)
    before = pq.read_table(d / "CALLS.parquet")

    def broken(table, where, **kw):
        open(where, "wb").write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(pq, "write_table", broken)
    with pytest.raises(OSError):
        write_table_atomic(before, d / "CALLS.parquet")
    monkeypatch.undo()
    assert pq.read_table(d / "CALLS.parquet").equals(before)
    assert [p.name for p in d.iterdir()] == ["CALLS.parquet"]


def test_parse_reply_takes_the_json_array_out_of_prose():
    assert parse_reply('Sure:\n```json\n[{"id": 1, "NOTES": "a"}]\n```') == {"1": {"id": 1, "NOTES": "a"}}


def test_tables_of_other_institutions_and_unknown_tables_are_untouched(tmp_path):
    d = data(tmp_path)
    pq.write_table(pa.table({"NOTE": ["groceries"]}), d / "P2P.parquet")
    before = pq.read_table(d / "P2P.parquet")
    assert tg(FakeGen(), tmp_path).rewrite(d, "kellmont")["tables"] == {}
    assert pq.read_table(d / "P2P.parquet").equals(before)


def test_textcheck_separates_template_fraud_and_reports_missing_row_truth(tmp_path):
    n = 40
    texts = [("Unauthorized debit, statement taken. Ref %d." if i % 2 else "Travel note added. Ref %d.") % i
             for i in range(n)]
    obs = tmp_path / "run" / "brindlemark"
    obs.mkdir(parents=True)
    pq.write_table(pa.table({"NOTES": texts}), obs / "CALLS.parquet")
    truth = tmp_path / "run" / "truth"
    truth.mkdir()
    pq.write_table(pa.table({"key": [f"e{i}" for i in range(n)],
                             "cause_kind": ["scenario" if i % 2 else "process" for i in range(n)]}),
                   truth / "events.parquet")
    assert textcheck.check(truth, tmp_path / "run")["brindlemark.CALLS"]["error"].startswith("no row truth")
    pq.write_table(pa.table({"observer": ["brindlemark"] * n, "table": ["CALLS"] * n, "row": list(range(n)),
                             "event": [f"e{i}" for i in range(n)]}), truth / "row_links.parquet")
    rec = textcheck.check(truth, tmp_path / "run")["brindlemark.CALLS"]
    assert rec["rows"] == n and rec["fraud"] == n // 2 and rec["auc_ngram"] > 0.95


def test_auc():
    assert textcheck.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert textcheck.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


class Causes:
    """Rows 0 and 2 of CALLS are ring 7's `ach_return`; the rest are ordinary."""

    def __init__(self, rows=((0, 7, "ach_return"), (2, 7, "ach_return"))):
        self.rows = {("brindlemark", "CALLS", r): (ring, scheme) for r, ring, scheme in rows}

    def of(self, observer, table, row):
        return self.rows.get((observer, table, row))

    def ring_rows(self):
        return set(self.rows)


R7 = {7: {"ach_return": "Callers are calm and say they never authorised the debit."}}


def notes(d):
    return pq.read_table(d / "CALLS.parquet").column("NOTES").to_pylist()


def test_ring_rows_are_written_from_their_rings_brief_and_the_rest_from_the_worlds(tmp_path):
    d = data(tmp_path, n=4)
    gen = FakeGen()
    stats = tg(gen, tmp_path, match="m1").rewrite(d, "brindlemark", causes=Causes(), briefs=R7)
    got = notes(d)
    assert [t[0] for t in got] == ["B", "W", "B", "W"]
    assert stats["tables"]["CALLS"]["from_attacker_briefs"] == 2
    assert sorted(brief_of(p) for p in gen.prompts) == sorted([R7[7]["ach_return"], textgen.WORLD_BRIEF])
    heads = {p[:p.index("Brief:")] for p in gen.prompts}
    assert len(heads) == 1
    for p in gen.prompts:
        assert "ach_return" not in p and "7" not in json.dumps([it["facts"] for it in items_of(p)])


def test_a_ring_without_a_brief_for_the_scheme_gets_the_worlds(tmp_path):
    d = data(tmp_path, n=3)
    tg(FakeGen(), tmp_path, match="m1").rewrite(d, "brindlemark", causes=Causes(), briefs={7: {"other": "x"}})
    assert [t[0] for t in notes(d)] == ["W", "W", "W"]


def test_a_new_brief_never_rewrites_rows_already_written(tmp_path):
    gen = FakeGen()
    tg(gen, tmp_path, match="m1").rewrite(data(tmp_path / "b1", n=2), "brindlemark", causes=Causes(), briefs={})
    later = data(tmp_path / "b2", n=4)
    stats = tg(gen, tmp_path, match="m1").rewrite(later, "brindlemark", causes=Causes(), briefs=R7)
    got = notes(later)
    assert got[0].startswith("W") and got[2].startswith("B")
    assert stats["tables"]["CALLS"]["cache_hits"] == 2


def test_cache_key_includes_the_brief():
    assert TextGen.key("base", "brief one") != TextGen.key("base", "brief two")


def test_ordinary_texts_are_reused_across_matches_and_ring_texts_are_not(tmp_path):
    gen = FakeGen()
    tg(gen, tmp_path, match="m1").rewrite(data(tmp_path / "a", n=4), "brindlemark", causes=Causes(), briefs=R7)
    n = len(gen.prompts)
    other = {7: {"ach_return": "Callers are angry."}}
    stats = tg(gen, tmp_path, match="m2").rewrite(data(tmp_path / "b", n=4), "brindlemark", causes=Causes(),
                                                  briefs=other)
    assert stats["tables"]["CALLS"]["cache_hits"] == 2 and stats["tables"]["CALLS"]["generated"] == 2
    assert [brief_of(p) for p in gen.prompts[n:]] == ["Callers are angry."]


def test_public_ads_are_rewritten_column_by_column_and_may_name_their_own_domain(tmp_path):
    d = tmp_path / "brindlemark"
    (d / "public").mkdir(parents=True)
    pq.write_table(pa.table({"AD_ID": ["AD1", "AD1", "AD2"], "ADVERTISER_NAME": ["VANTAGE ONLINE INC"] * 3,
                             "PLATFORM": ["SOCIAL"] * 3, "KEYWORDS": ["remote job"] * 3,
                             "LANDING_DOMAIN": ["workpath.example.co"] * 3,
                             "HEADLINE": ["Work from home", "Work from home", "Weekly pay"],
                             "BODY": ["Apply in 5 minutes.", "Apply in 5 minutes.", "Flexible hours."]}),
                   d / "public" / "ADS.parquet")
    gen = FakeGen(bad=lambda g, n: "Apply at workpath.example.co today" if g.startswith("Apply") else None)
    stats = tg(gen, tmp_path).rewrite(d, "brindlemark")
    t = pq.read_table(d / "public" / "ADS.parquet").to_pylist()
    assert t[0]["HEADLINE"] == "W: Work from home ()" and t[0]["BODY"] == "Apply at workpath.example.co today"
    assert t[1] == t[0]
    assert stats["tables"]["PUBLIC_ADS"]["rows"] == 3 and stats["tables"]["PUBLIC_ADS"]["cache_hits"] == 1
    assert set(items_of(gen.prompts[0])[0]["gist"]) == {"HEADLINE", "BODY"}


def test_row_causes_follow_the_attacker_scoring_rule(tmp_path):
    T = tmp_path / "truth"
    T.mkdir()
    pq.write_table(pa.table({"type": ["ring", "fraudster", "fraudster"], "entity": ["ring7", "f1", "f2"],
                             "attribute": ["number", "ring", "ring"], "value": ["7", "ring7", "ring3"]}),
                   T / "hidden.parquet")
    pq.write_table(pa.table({"scenario": ["r7_ach_return", "romance", "romance", "family"],
                             "instance": ["i1", "i2", "i3", "i4"], "anchor": ["x", "f1", "f2", "y"]}),
                   T / "instances.parquet")
    pq.write_table(pa.table({"key": ["e1", "e2", "e3", "e4"],
                             "cause_kind": ["scenario", "scenario", "scenario", "process"],
                             "cause_instance": ["i1", "i2", "i3", None],
                             "cause_scenario": ["r7_ach_return", "romance", "romance", None]}), T / "events.parquet")
    pq.write_table(pa.table({"observer": ["brindlemark"] * 4, "table": ["CALLS"] * 4, "row": [0, 1, 2, 3],
                             "event": ["e1", "e2", "e3", "e4"]}), T / "row_links.parquet")
    c = textgen.RowCauses(T, [7])
    assert c.of("brindlemark", "CALLS", 0) == (7, "ach_return")
    assert c.of("brindlemark", "CALLS", 1) == (7, "romance")
    assert c.of("brindlemark", "CALLS", 2) is None and c.of("brindlemark", "CALLS", 3) is None
    assert c.fraud == {("brindlemark", "CALLS", r) for r in (0, 1, 2)}


def test_read_briefs_keeps_the_rings_text_schemes_and_caps_length(tmp_path):
    b = tmp_path / "briefs"
    b.mkdir()
    (b / "ach_return.md").write_text("x" * 5000)
    (b / "bank_hopping.md").write_text("no texts")
    (b / "empty.md").write_text("")
    got, refused = textgen.read_briefs(b, {"ach_return", "empty"})
    assert set(got) == {"ach_return"} and len(got["ach_return"]) == textgen.MAX_BRIEF_CHARS
    assert any("bank_hopping" in r for r in refused) and any("cut" in r for r in refused)


def test_scheme_texts_names_what_each_of_the_rings_schemes_writes(tmp_path):
    import yaml
    w = tmp_path / "w.yaml"
    w.write_text(yaml.safe_dump({
        "observers": {"brindlemark": {"tables": {"CALLS": {"grain": {"event": "bank_call"}}}},
                      "public": {"tables": {"ADS": {"grain": {"event": "ad_update"}}}}},
        "world": {"scenarios": {"r7_ach_return": {"steps": [{"emit": "bank_call"}]},
                                "r8_job_ads": {"steps": [{"emit": "ad_update"}]},
                                "r7_setup": {"steps": [{"emit": "p2p_send"}]},
                                "romance": {"steps": [{"emit": "bank_call"}]}}}}))
    got = textgen.scheme_texts(w, 7)
    assert list(got) == ["ach_return"] and got["ach_return"][0].startswith("Brindlemark CALLS")
    assert list(textgen.scheme_texts(w, 8)) == ["job_ads"]


def test_textcheck_reports_ring_rows_against_ordinary_ones(tmp_path):
    n = 40
    obs = tmp_path / "brindlemark"
    obs.mkdir()
    texts = ["URGENT!!! never authorised. Ref %d" % i if i % 4 == 0 else "Travel note added. Ref %d." % i
             for i in range(n)]
    pq.write_table(pa.table({"NOTES": texts}), obs / "CALLS.parquet")
    T = tmp_path / "truth"
    T.mkdir()
    pq.write_table(pa.table({"key": [f"e{i}" for i in range(n)],
                             "cause_kind": ["scenario" if i % 2 == 0 else "process" for i in range(n)]}),
                   T / "events.parquet")
    pq.write_table(pa.table({"observer": ["brindlemark"] * n, "table": ["CALLS"] * n, "row": list(range(n)),
                             "event": [f"e{i}" for i in range(n)]}), T / "row_links.parquet")
    ring = {("brindlemark", "CALLS", i) for i in range(0, n, 4)}
    rec = textcheck.check(T, {"brindlemark": obs}, ring)["brindlemark.CALLS"]
    assert rec["ring_rows"] == 10 and rec["ring_auc_style"] > 0.9


def test_attacker_brief_explains_scheme_briefs_only_when_attacker_text_is_on():
    from fraud_arena.attacker import operator
    off = operator.brief(ring=7, schema={"properties": {}}, day=7, period_days=7, boundary=1, boundaries=4)
    on = operator.brief(ring=7, schema={"properties": {}}, day=7, period_days=7, boundary=1, boundaries=4,
                        text_schemes={"ach_return": ["Brindlemark CALLS: notes"]}, text_notes=["x.md: refused"])
    assert "You don't write any messages" in off and "briefs/" not in off
    assert "briefs/<scheme>.md" in on and "`ach_return`: Brindlemark CALLS" in on and "x.md: refused" in on
    assert "No real companies or brands" in on and "published" in on
