"""The proposal file for a host's semantic layer: written from approved decisions only, never
from anything seeded, rejected, undecided or decided on another statement; the same bytes for
the same inputs; and a kept value that rests only on what the file holds.

Moved here from the estate prototype's own tests (ticket 43); what those tests did through a
build, these do on the export a build writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml
from _proposal_estate import DEALERS, claim, element, export, in_force, line, pair
from latence_core.errors import ConfigError
from latence_proposal_writer import proposal, schema, to_yaml


def _written(document: dict[str, Any], estate: str = "yard") -> tuple[str, dict[str, Any]]:
    """The file's text and what a YAML reader takes it for, held against the schema."""
    text = to_yaml(document, estate)
    doc: dict[str, Any] = yaml.safe_load(text)
    jsonschema.validate(doc, schema())
    return text, doc


def _origin(el: dict[str, Any], by: str = "reviewer-1", at: str = "2026-11-12") -> dict[str, Any]:
    return {"element": el["id"], "claim": el["claim"],
            "decision": {"verdict": "approved", "by": by, "at": at}}


def test_an_approved_pair_and_an_approved_master_list_are_written_with_the_decision_they_rest_on() -> None:
    doc = export()
    alfa, dealers = pair(doc, "Alfa"), element(doc, "concept:shop.dealers")
    decided = in_force(line(alfa), line(dealers, by="owner-of-the-list", at="2026-11-13T09:30:00+01:00"))
    document, report = proposal(doc, decided, names_of=("shop.dealers",))
    text, read = _written(document)
    assert read["source"] == "ontology" and len(read["entries"]) == 2
    # The value kept is the one more rows carry; the object is the one that holds most of the
    # value set, not the first by name.
    assert read["entries"][0] == {"spellings": {"object": "shop.recalls", "column": "maker", "value": "Alfa Works, Inc.",
                                                "forms": ["Alfa Works North"]}, "origin": _origin(alfa)}
    assert read["entries"][1] == {"names": {"object": "shop.dealers", "column": "dealer_id", "values": DEALERS},
                                  "origin": _origin(dealers, by="owner-of-the-list", at="2026-11-13")}  # the day of a timestamp
    # "0123" and "Yes" are names, not a number and a truth value.
    assert all(isinstance(v, str) for v in read["entries"][1]["names"]["values"].values())
    assert report["entries"] == [
        {"entry": 1, "kind": "spellings", "element": alfa["id"], "claim": alfa["claim"], "decided_by": "reviewer-1",
         "decided_on": "2026-11-12", "day_stated_by": "decision", "object": "shop.recalls", "column": "maker",
         "value": "Alfa Works, Inc.", "forms": ["Alfa Works North"]},
        {"entry": 2, "kind": "names", "element": "concept:shop.dealers", "claim": dealers["claim"],
         "decided_by": "owner-of-the-list", "decided_on": "2026-11-13", "day_stated_by": "decision",
         "object": "shop.dealers", "column": "dealer_id", "records": 30}]
    assert report["stale"] == [] and report["not_written"] == [] and report["not_carried"] == {}
    # The estate may name the object a value set's spellings are stated on, if it holds the set.
    document, _ = proposal(doc, decided, names_of=("shop.dealers",), spellings_on={"maker": "shop.claims"})
    assert document["entries"][0]["spellings"]["object"] == "shop.claims"
    with pytest.raises(ConfigError, match="shop.dealers does not hold the value set maker"):
        proposal(doc, decided, spellings_on={"maker": "shop.dealers"})


def test_a_rejected_or_undecided_element_is_in_no_proposal() -> None:
    doc = export()
    alfa, beta, gama, dealers = pair(doc, "Alfa"), pair(doc, "Beta"), pair(doc, "Gama"), element(doc, "concept:shop.dealers")
    assert gama["claim"]  # the third pair is listed and nobody decides it
    decided = in_force(line(beta), line(alfa), line(dealers, verdict="rejected"), line(beta, verdict="rejected"))
    document, report = proposal(doc, decided, names_of=("shop.dealers",))
    text, read = _written(document)
    assert [e["origin"]["element"] for e in read["entries"]] == [alfa["id"]]
    assert "Beta" not in text and "Gama" not in text and "Dealer" not in text and "names:" not in text
    assert [e["element"] for e in report["entries"]] == [alfa["id"]]
    assert report["not_written"] == [] and report["stale"] == []


def test_a_seeded_record_and_a_link_between_records_are_in_no_proposal() -> None:
    doc = export()
    assert doc["meta"]["resolution"]["certified_on"] == "simulated"  # the machine's links in the export
    accounts, dealers = element(doc, "concept:partner.accounts"), element(doc, "concept:shop.dealers")
    with_seeded = pair(doc, "D3~partner.accounts:PA4")  # a dealer and a seeded account, left for a person
    two_dealers = pair(doc, "D1~shop.dealers:D2")  # two dealers a person called the same
    north = pair(doc, "same:market:")  # two names of a market, one of them only in the seeded source
    # The estate even names the seeded list among those whose names may go, and a person
    # approves all five.
    decided = in_force(line(accounts), line(dealers), line(with_seeded), line(two_dealers), line(north))
    document, report = proposal(doc, decided, names_of=("shop.dealers", "partner.accounts"))
    text, read = _written(document)
    assert [e["origin"]["element"] for e in read["entries"]] == ["concept:shop.dealers"]
    assert read["entries"][0]["names"]["values"] == DEALERS  # each dealer by its own name, no linked account's
    assert "Portal account" not in text and "partner.accounts" not in text and "PA1" not in text
    assert "North Area" not in text
    said = json.dumps(read, ensure_ascii=False)  # the entries, without the file's opening comment
    assert "seeded" not in said and "certified_on" not in said and "simulated" not in said
    assert report["not_written"] == [{"subject": "concept:partner.accounts", "why": "seeded"},
                                     {"subject": north["id"], "why": "seeded"},
                                     {"subject": two_dealers["id"], "why": "one_object"},  # two records of one object: no link
                                     {"subject": with_seeded["id"], "why": "seeded"}]
    assert report["not_carried"] == {}


def test_an_entry_whose_decisions_claim_no_longer_matches_the_build_is_listed_as_stale_and_not_written() -> None:
    doc = export()
    alfa, beta, dealers = pair(doc, "Alfa"), pair(doc, "Beta"), element(doc, "concept:shop.dealers")
    then = claim(999)  # the list was approved when it stated something else
    decided = in_force(line(dealers, stated=then), line(alfa, stated=beta["claim"]), line(beta))
    document, report = proposal(doc, decided, names_of=("shop.dealers",))
    text, read = _written(document)
    assert [e["origin"]["element"] for e in read["entries"]] == [beta["id"]]
    assert "Alfa" not in text and "Dealer" not in text
    assert report["stale"] == [{"subject": "concept:shop.dealers", "decided_on": then, "now": dealers["claim"]},
                               {"subject": alfa["id"], "decided_on": beta["claim"], "now": alfa["claim"]}]
    assert report["not_written"] == []


def test_two_runs_on_the_same_inputs_give_the_same_bytes_whatever_order_the_decisions_come_in() -> None:
    doc = export()
    lines = [line(element(doc, "concept:shop.dealers")), line(pair(doc, "Gama")), line(pair(doc, "Alfa")),
             line(pair(doc, "Beta"))]
    first = to_yaml(proposal(doc, in_force(*lines), names_of=("shop.dealers",))[0], "yard")
    assert [next(iter(e)) for e in yaml.safe_load(first)["entries"]] == ["spellings", "spellings", "spellings", "names"]
    assert to_yaml(proposal(doc, in_force(*lines), names_of=("shop.dealers",))[0], "yard") == first
    assert to_yaml(proposal(doc, in_force(*reversed(lines)), names_of=("shop.dealers",))[0], "yard") == first
    assert first.endswith("\n") and "\r" not in first
    assert first.splitlines()[0].startswith("# ") and '# Estate: "yard"' in first


def test_a_kept_value_never_rests_on_a_decision_that_is_not_written() -> None:
    def a_pair(a: str, b: str, rows_a: int, rows_b: int, became: str, n: int) -> dict[str, Any]:
        return {"id": f"same:maker:{a}~{b}", "kind": "possibly_same", "concept": "concept:value:maker",
                "a": f"value:maker:{a}", "b": f"value:maker:{b}", "rows_a": rows_a, "rows_b": rows_b,
                "merged_into": f"value:maker:{became}", "claim": claim(n)}

    # Three spellings, two approved pairs. A build that applied both made all three one value:
    # the third name, which most rows carry.
    first = a_pair("Ab Cars", "Ab Cars Inc", 5, 4, "Ab Cars International", 1)
    second = a_pair("Ab Cars Inc", "Ab Cars International", 4, 20, "Ab Cars International", 2)
    doc = {"concepts": [{"id": "concept:value:maker", "kind": "reference", "column": "maker",
                         "objects": ["s.a", "s.b"], "claim": "0" * 16}],
           "relations": [], "measures": [], "nodes": [], "review": [first, second]}
    decided = {first["id"]: line(first), second["id"]: line(second, at=None)}  # the second states no day
    document, report = proposal(doc, decided)
    # Only the first pair is written, so its value is one of its own two names: nothing in the
    # file says the third is the same.
    assert [e["spellings"] for e in document["entries"]] == [
        {"object": "s.a", "column": "maker", "value": "Ab Cars", "forms": ["Ab Cars Inc"]}]
    assert "International" not in json.dumps(document)
    assert report["not_written"] == [{"subject": second["id"], "why": "no_day"}]
    # With both written, both say what the build says.
    document, _ = proposal(doc, decided, decided_on="2026-11-12")
    assert [(e["spellings"]["value"], e["spellings"]["forms"]) for e in document["entries"]] == [
        ("Ab Cars International", ["Ab Cars Inc"]), ("Ab Cars International", ["Ab Cars", "Ab Cars Inc"])]


def test_a_decision_that_states_no_day_or_no_claim_is_not_written_and_a_day_is_never_made_up() -> None:
    doc = export()
    alfa, dealers = pair(doc, "Alfa"), element(doc, "concept:shop.dealers")
    decided = in_force(line(alfa, at=None), line(dealers, stated=None))
    document, report = proposal(doc, decided, names_of=("shop.dealers",))
    assert _written(document)[1] == {"source": "ontology", "entries": []}
    assert report["entries"] == []
    assert report["not_written"] == [{"subject": "concept:shop.dealers", "why": "no_claim"},
                                     {"subject": alfa["id"], "why": "no_day"}]
    # Whoever writes the file may state the day for the decisions that state none; the report
    # says where the day came from.
    document, report = proposal(doc, decided, names_of=("shop.dealers",), decided_on="2026-11-14")
    assert document["entries"][0]["origin"] == _origin(alfa, at="2026-11-14")
    assert report["entries"][0]["day_stated_by"] == "command"
    assert report["not_written"] == [{"subject": "concept:shop.dealers", "why": "no_claim"}]
    with pytest.raises(ConfigError, match="no ISO date"):
        proposal(doc, decided, decided_on="14 Nov")
    with pytest.raises(ConfigError, match="no ISO date"):
        proposal(doc, in_force(line(alfa, at="yesterday")))


def test_what_the_estate_does_not_name_and_what_the_file_has_no_entry_for_is_counted_and_not_written() -> None:
    doc = export()
    dealers = element(doc, "concept:shop.dealers")
    relation = next(r for r in doc["relations"] if r["status"] == "proposed")
    subclass = element(doc, "subclass:dealer~organisation")
    gone = {"subject": "concept:shop.gone", "verdict": "approved", "by": "reviewer-1", "claim": "0" * 16,
            "at": "2026-11-12"}
    document, report = proposal(doc, in_force(line(dealers), line(relation), line(subclass), gone))
    assert document["entries"] == []  # an approval of a concept does not by itself send its rows' names anywhere
    assert report["not_carried"] == {"concept": 1, "relation": 1, "subclass": 1}
    assert report["not_written"] == [{"subject": "concept:shop.gone", "why": "not_in_the_build"}]


def test_the_schema_ships_with_the_package_and_states_what_the_writer_writes() -> None:
    held = schema()
    assert held["properties"]["source"] == {"description": held["properties"]["source"]["description"],
                                            "const": "ontology"}
    # A file with a mark nothing written carries is not of this shape.
    doc = export()
    document, _ = proposal(doc, in_force(line(pair(doc, "Alfa"))))
    read = yaml.safe_load(to_yaml(document, "yard"))
    read["entries"][0]["origin"]["seeded"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(read, held)


def test_a_download_of_the_page_alone_writes_the_file_with_no_day_given_and_no_list_named() -> None:
    """The host's page (its ticket 282) downloads every decision with its time in UTC, and
    offers a decision on a stated list the estate marked for the proposal. From such a download
    and the export, the file is written with no day given and no list named by the caller."""
    doc = export()
    marked = {**doc, "concepts": [{**c, "proposal": "names"} if c["id"] in ("concept:shop.dealers", "concept:partner.accounts") else c
                                  for c in doc["concepts"]]}
    download = (Path(__file__).parent / "fixtures" / "page-decisions.jsonl").read_text(encoding="utf-8")
    decided = in_force(*(json.loads(row) for row in download.splitlines()))  # the last line on an element counts
    document, report = proposal(marked, decided)
    _, read = _written(document)
    assert [(e["kind"], e["element"], e["decided_by"], e["decided_on"], e["day_stated_by"]) for e in report["entries"]] == [
        ("spellings", "same:maker:Alfa Works, Inc.~Alfa Works North", "reviewer-7", "2026-11-14", "decision"),
        ("names", "concept:shop.dealers", "reviewer-7", "2026-11-14", "decision")]
    assert read["entries"][1]["names"]["values"] == DEALERS and report["not_written"] == []
    # A day given is for lines that state none; it never replaces the day a line states.
    assert proposal(marked, decided, decided_on="2026-11-01")[1]["entries"] == report["entries"]
    # Without the mark, an approval of the list still sends no names anywhere.
    assert [e["kind"] for e in proposal(doc, decided)[1]["entries"]] == ["spellings"]
    # A seeded list stays out, marked or not.
    seeded = in_force(line(element(marked, "concept:partner.accounts")))
    assert proposal(marked, seeded)[1]["not_written"] == [{"subject": "concept:partner.accounts", "why": "seeded"}]
