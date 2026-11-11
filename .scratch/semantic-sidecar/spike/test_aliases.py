"""Short forms: what a name can be abbreviated to, what a text defines, and which route wins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aliases import AliasTable, is_short_form, mine_definitions, short_forms, strongest
from generate_text import generate_threads
from run_b import run


def test_short_forms_of_a_name() -> None:
    assert "IBM" in short_forms("International Business Machines Corp")
    assert "IBMC" in short_forms("International Business Machines Corp")
    assert "MB" in short_forms("Mercedes-Benz AG")
    assert "PC" in short_forms("PricewaterhouseCoopers")
    assert "BOA" in short_forms("Bank of America") and "BA" in short_forms("Bank of America")
    assert short_forms("Volkswagen") == set()  # one lower-case compound: letters alone cannot split it


@pytest.mark.parametrize("s,expected", [("IBM", True), ("I.B.M.", True), ("H&M", True), ("PwC", True), ("VW", True),
                                        ("Harborwave", False), ("Ibm", False), ("NORTHWIND", False), ("the", False), ("A B", False)])
def test_what_counts_as_a_short_form(s: str, expected: bool) -> None:
    assert is_short_form(s) is expected


def test_definitions_are_read_in_both_orders() -> None:
    d = mine_definitions("Dear colleagues, Lumenstone Foods Corp (LF) has asked for terms.")
    assert [(x["short"], x["long"]) for x in d] == [("LF", "Lumenstone Foods Corp")]
    d = mine_definitions("We met IBM (International Business Machines) on Monday.")
    assert [(x["short"], x["long"]) for x in d] == [("IBM", "International Business Machines")]
    assert mine_definitions("Invoice issued by Orbitbrook Kestrel (Cardiff) for work.") == []
    assert mine_definitions("Lumenstone Foods Corp (XQ) has asked.") == []  # letters do not fit the name


def test_every_route_proposes_and_scope_stays_in_its_thread() -> None:
    names = ["Lumenstone Foods Corp", "Lakefield Finance Ltd", "Larchmoor Freight Inc", "Orbitbrook Kestrel Engineering Inc"]
    t = AliasTable(names)
    assert t.candidates("LF") == {0: {"initials": 1}, 1: {"initials": 1}, 2: {"initials": 1}}
    assert set(t.candidates("OE")) == {3} and set(t.candidates("OKE")) == {3}  # a word may be skipped
    t.add_alias("LF", 2)
    t.add_definition("L.F.", 1)
    t.add_in_scope("thread-7", 0)
    t.add_in_scope("thread-7", 3)  # another organisation of the thread, whose initials do not fit
    got = t.candidates("LF", scope="thread-7")
    assert got == {0: {"scope": 1, "initials": 1}, 1: {"defined": 1, "initials": 1}, 2: {"alias": 1, "initials": 1}}
    assert [strongest(got[j]) for j in (0, 1, 2)] == ["scope", "defined", "alias"]
    assert "scope" not in t.candidates("LF", scope="thread-8")[0]
    assert t.candidates("ZZ") == {}


@pytest.fixture(scope="module")
def threads(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("threads")
    for k, rows in generate_threads(500, 31, 400).items():
        with open(out / f"{k}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:  # type: ignore[attr-defined]
                fh.write(json.dumps(row) + "\n")
    return out


def test_generated_offsets_and_signals_are_true(threads: Path) -> None:
    docs = {d["doc_id"]: d for d in map(json.loads, open(threads / "docs.jsonl", encoding="utf-8"))}
    mentions = [json.loads(x) for x in open(threads / "mentions.jsonl", encoding="utf-8")]
    assert all(docs[m["doc_id"]]["text"][m["start"] : m["end"]] == m["surface"] for m in mentions)
    full = {(docs[m["doc_id"]]["thread_id"], m["entity_id"]) for m in mentions if m["variant"] != "acronym"}
    acr = [m for m in mentions if m["variant"] == "acronym"]
    assert acr and all(("scope" in m["signals"]) == ((docs[m["doc_id"]]["thread_id"], m["entity_id"]) in full) for m in acr)
    assert {s for m in acr for s in m["signals"]} == {"scope", "defined", "alias"}
    assert any(not m["signals"] for m in acr)


def test_alias_routes_resolve_short_forms_that_characters_cannot(threads: Path) -> None:
    # No model at all: the lexical scorer. Without aliases an acronym shares no n-grams with
    # its anchor; with them, a full name in the same thread settles it.
    plain = run(threads, "lexical", 5, 300, 500, 7)
    routed = run(threads, "lexical", 5, 300, 500, 7, use_aliases=True)
    assert plain["by_variant"]["acronym"]["top1_accuracy"] < 0.1
    assert routed["by_variant"]["acronym"]["top1_accuracy"] > 0.5
    by_signal = routed["short_forms"]["by_signal"]
    assert by_signal["scope"]["top1_accuracy"] > 0.8
    assert routed["short_forms"]["ranker_weights"]["in_thread"] > 0
    assert by_signal["scope"]["top1_accuracy"] > by_signal["none"]["top1_accuracy"]
    for v in ("canonical", "typo", "abbreviated"):  # full names are linked exactly as before
        assert routed["by_variant"][v]["top1_accuracy"] == plain["by_variant"][v]["top1_accuracy"]
    assert set(routed["short_forms"]["by_route"]) <= {"scope", "defined", "alias", "initials", "none"}
    assert np.isfinite(list(routed["features"].values())).all()
