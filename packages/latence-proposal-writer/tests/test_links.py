"""A link between two records that a person decided reaches the host's layer (ticket 45), in
version 2 of the file only, as an entry of its own; a link the machine made, one certified on
simulated labels, and one that holds a seeded record never do."""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest
import yaml
from _proposal_estate import DEALERS, claim, export, in_force, line, pair
from latence_core.errors import ConfigError
from latence_proposal_writer import SCHEMA_VERSION, proposal, schema, to_yaml

TWO_DEALERS = "same:shop.dealers:D1~shop.dealers:D2"
ANSWERED = "same:shop.dealers:D4~shop.dealers:D5"  # answered by a person, so no longer listed


def _with_an_answered_link(doc: dict[str, Any], stated: bool = True) -> dict[str, Any]:
    """The export once a build applied a person's link of D5 to D4: an edge that says who
    decided it and, where the build states it, the pair's claim."""
    edge = {"kind": "same_as", "from": "shop.dealers:D5", "to": "shop.dealers:D4", "decided_by": "reviewer-2",
            **({"claim": claim(33)} if stated else {})}
    return {**doc, "edges": [*doc["edges"], edge]}


def _record(key: str) -> dict[str, str]:
    return {"object": "shop.dealers", "column": "dealer_id", "key": key, "name": DEALERS[key]}


def test_a_link_a_person_approved_is_written_in_version_2_and_in_no_file_of_version_1() -> None:
    doc = _with_an_answered_link(export())
    two = pair(doc, "D1~shop.dealers:D2")
    answered = {"subject": ANSWERED, "verdict": "approved", "by": "reviewer-2", "claim": claim(33), "at": "2026-11-14T10:00:00Z"}
    decided = in_force(line(two), answered, line(pair(doc, "Alfa")))
    first, first_report = proposal(doc, decided)
    assert "version" not in first and [next(iter(e)) for e in first["entries"]] == ["spellings"]
    assert first_report["not_carried"] == {"pair_of_records": 2}
    assert to_yaml(first, "yard") == to_yaml(proposal(doc, decided, version=1)[0], "yard")  # version 1 is today's file
    second, report = proposal(doc, decided, version=2)
    text = to_yaml(second, "yard")
    read = yaml.safe_load(text)
    jsonschema.validate(read, schema(2))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(read, schema(1))  # a reader of version 1 cannot take it for its own
    assert read["version"] == "2" and text.index('version: "2"') < text.index("entries:")
    assert read["entries"][1:] == [
        {"links": {"records": [_record("D1"), _record("D2")]},
         "origin": {"element": TWO_DEALERS, "claim": two["claim"], "decision": {"verdict": "approved", "by": "reviewer-1", "at": "2026-11-12"}}},
        {"links": {"records": [_record("D4"), _record("D5")]},
         "origin": {"element": ANSWERED, "claim": claim(33), "decision": {"verdict": "approved", "by": "reviewer-2", "at": "2026-11-14"}}}]
    assert [(e["kind"], e["element"], e.get("records")) for e in report["entries"]] == [
        ("spellings", pair(doc, "Alfa")["id"], None), ("links", TWO_DEALERS, 2), ("links", ANSWERED, 2)]
    assert report["not_carried"] == {} and report["not_written"] == []
    assert SCHEMA_VERSION == 2


def test_no_link_the_machine_made_or_that_holds_a_seeded_record_or_was_decided_on_another_statement_is_written() -> None:
    doc = export()
    assert sum(e["kind"] == "same_as" for e in doc["edges"]) == 3  # the machine's, certified on simulated labels
    with_seeded, two = pair(doc, "D3~partner.accounts:PA4"), pair(doc, "D1~shop.dealers:D2")
    machine = {"subject": "same:shop.dealers:D1~partner.accounts:PA1", "verdict": "approved", "by": "reviewer-1",
               "claim": claim(50), "at": "2026-11-12"}  # a line on a link nobody listed and no person decided
    decided = in_force(line(with_seeded), line(two, stated=claim(999)), machine)
    document, report = proposal(doc, decided, version=2)
    assert document == {"version": "2", "source": "ontology", "entries": []}
    assert "PA1" not in to_yaml(document, "yard") and "Portal" not in to_yaml(document, "yard")
    assert report["stale"] == [{"subject": TWO_DEALERS, "decided_on": claim(999), "now": two["claim"]}]
    assert report["not_written"] == [{"subject": machine["subject"], "why": "seeded"}, {"subject": with_seeded["id"], "why": "seeded"}]
    # A link a build applied but whose edge states no claim cannot be checked against what the
    # person decided, and is not written; nor is one without a day.
    bare = _with_an_answered_link(doc, stated=False)
    answered = {"subject": ANSWERED, "verdict": "approved", "by": "reviewer-2", "claim": claim(33), "at": "2026-11-14"}
    assert proposal(bare, in_force(answered), version=2)[1]["not_written"] == [{"subject": ANSWERED, "why": "not_in_the_build"}]
    assert proposal(doc, in_force(line(two, at=None)), version=2)[1]["not_written"] == [{"subject": TWO_DEALERS, "why": "no_day"}]


def test_a_version_the_writer_does_not_know_is_refused() -> None:
    with pytest.raises(ConfigError, match="version 3"):
        proposal(export(), {}, version=3)
    with pytest.raises(ConfigError, match="version 3"):
        schema(3)
