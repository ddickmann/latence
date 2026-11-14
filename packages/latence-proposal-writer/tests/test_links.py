"""A link between two records that a person decided reaches the host's layer (ticket 45) in the
shape the host's tool reads (its ticket 294): `format: layer-proposal/2`, a `link` entry with
the master list's record, the other source's record, the statements the person saw and the
decision. The writer never writes what the host refuses; each refusal has its test here."""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest
import yaml
from _proposal_estate import claim, export, in_force, line, pair
from latence_core.errors import ConfigError
from latence_proposal_writer import SCHEMA_VERSION, proposal, schema, to_yaml

NAME_SAME = {"kind": "name", "code": "name.same", "params": {}, "ref": ""}
TOWN_SAME = {"kind": "town", "code": "town.same", "params": {}, "ref": ""}
SOURCE = {"kind": "source", "code": "source.two", "params": {"a": "dealer master", "b": "crm"}, "ref": ""}
LINK_C1 = "same:shop.dealers:D1~crm.accounts:C1"


def _estate(certified_on: str = "reviewer") -> dict[str, Any]:
    """The synthetic export with a second source that is not seeded (a CRM's accounts), pairs
    of a dealer and an account left for a person, and the build's resolution."""
    doc = export()
    accounts = {"id": "concept:crm.accounts", "kind": "entity", "object_name": "crm.accounts", "key": ["account_no"],
                "claim": claim(60), "status": "proposed"}
    nodes = [{"id": f"crm.accounts:C{i}", "concept": "concept:crm.accounts", "key": f"C{i}", "label": f"Account {i}"} for i in range(1, 6)]

    def item(a: str, b: str, n: int, p: float | None = 0.5651, reasons: tuple[dict[str, Any], ...] = (NAME_SAME, TOWN_SAME)) -> dict[str, Any]:
        return {"id": f"same:{a}~{b}", "kind": "possibly_same", "concept": "concept:shop.dealers", "a": a, "b": b, "claim": claim(n),
                "reasons": list(reasons), "evidence": [*reasons, SOURCE], **({"p": p} if p is not None else {})}

    review = [item("shop.dealers:D1", "crm.accounts:C1", 70), item("shop.dealers:D2", "crm.accounts:C2", 71, p=None),
              item("shop.dealers:D3", "crm.accounts:C3", 72, reasons=()), item("shop.dealers:D4", "crm.accounts:C2", 73),
              item("shop.dealers:D5", "crm.accounts:C4", 74)]
    machine = {"kind": "same_as", "from": "crm.accounts:C4", "to": "shop.dealers:D6", "actor": "engine", "p": 0.97}
    return {**doc, "concepts": [*doc["concepts"], accounts], "nodes": [*doc["nodes"], *nodes], "review": [*doc["review"], *review],
            "edges": [*doc["edges"], machine], "meta": {"resolution": {"certified_on": certified_on, "seeded": False}}}


def _item(doc: dict[str, Any], subject: str) -> dict[str, Any]:
    found: dict[str, Any] = next(r for r in doc["review"] if r["id"] == subject)
    return found


DEALERS_APPROVED = "concept:shop.dealers"


def _decided(doc: dict[str, Any], *subjects: str, dealers: bool = True) -> dict[str, dict[str, Any]]:
    lines = [line(_item(doc, s)) for s in subjects]
    if dealers:
        lines.append(line(next(c for c in doc["concepts"] if c["id"] == DEALERS_APPROVED)))
    return in_force(*lines)


def test_an_approved_link_is_written_in_the_hosts_shape_and_the_file_says_its_format() -> None:
    doc = _estate()
    document, report = proposal(doc, _decided(doc, LINK_C1), names_of=("shop.dealers",))
    text = to_yaml(document, "yard")
    read = yaml.safe_load(text)
    jsonschema.validate(read, schema(2))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(read, schema(1))  # a reader of version 1 cannot take it for its own
    assert read["format"] == "layer-proposal/2" and text.index("format:") < text.index("source:") < text.index("entries:")
    assert read["entries"][-1] == {
        "link": {"record": {"object": "shop.dealers", "column": "dealer_id", "key": "D1"},
                 "linked": {"object": "crm.accounts", "column": "account_no", "key": "C1"},
                 "evidence": [{"code": "name.same", "params": {}}, {"code": "town.same", "params": {}}], "p": 0.5651},
        "origin": {"element": LINK_C1, "claim": claim(70), "decision": {"verdict": "approved", "by": "reviewer-1", "at": "2026-11-12"},
                   "certified_on": "reviewer"}}
    assert [e["kind"] for e in report["entries"]] == ["names", "link"] and report["not_written"] == []
    assert SCHEMA_VERSION == 2


def test_a_file_without_a_link_is_version_1_byte_for_byte_and_a_host_on_version_1_gets_no_link() -> None:
    doc = _estate()
    without = to_yaml(proposal(doc, _decided(doc), names_of=("shop.dealers",))[0], "yard")
    assert "format" not in yaml.safe_load(without)
    assert without == to_yaml(proposal(export(), _decided(export()), names_of=("shop.dealers",))[0], "yard")
    document, report = proposal(doc, _decided(doc, LINK_C1), names_of=("shop.dealers",), links=False)
    assert "format" not in document and report["not_carried"] == {"pair_of_records": 1}


def test_a_link_with_a_seeded_record_or_one_no_person_approved_is_never_written() -> None:
    doc = _estate()
    seeded = "same:shop.dealers:D3~partner.accounts:PA4"  # the portal's accounts are seeded
    decided = {**_decided(doc), seeded: line(pair(doc, "D3~partner.accounts:PA4")),
               LINK_C1: line(_item(doc, LINK_C1), verdict="rejected")}
    document, report = proposal(doc, decided, names_of=("shop.dealers",))
    assert "format" not in document and [e["kind"] for e in report["entries"]] == ["names"]
    assert report["not_written"] == [{"subject": seeded, "why": "seeded"}]
    # The machine's links (one certified on simulated labels among them) are in no file.
    assert all(e["kind"] == "same_as" and "decided_by" not in e for e in doc["edges"])
    assert "C4" not in to_yaml(document, "yard") and "PA1" not in to_yaml(document, "yard")


def test_certified_on_is_written_only_as_reviewer() -> None:
    for said in ("simulated", "mixed"):
        doc = _estate(certified_on=said)
        document, _ = proposal(doc, _decided(doc, LINK_C1), names_of=("shop.dealers",))
        assert "certified_on" not in document["entries"][-1]["origin"]
        assert said not in to_yaml(document, "yard")


def test_a_link_not_stated_in_full_is_not_written() -> None:
    doc = _estate()
    no_claim = {**_decided(doc), LINK_C1: line(_item(doc, LINK_C1), stated=None)}
    assert proposal(doc, no_claim, names_of=("shop.dealers",))[1]["not_written"] == [{"subject": LINK_C1, "why": "no_claim"}]
    keyless = {**doc, "nodes": [{k: v for k, v in n.items() if k != "key"} if n["id"] == "crm.accounts:C1" else n for n in doc["nodes"]]}
    assert proposal(keyless, _decided(keyless, LINK_C1), names_of=("shop.dealers",))[1]["not_written"] == [
        {"subject": LINK_C1, "why": "no_records"}]
    no_reasons = "same:shop.dealers:D3~crm.accounts:C3"  # nothing the person saw is stated
    assert proposal(doc, _decided(doc, no_reasons), names_of=("shop.dealers",))[1]["not_written"] == [
        {"subject": no_reasons, "why": "no_evidence"}]
    # Without the pair's probability the entry carries none; nothing is made up.
    two = "same:shop.dealers:D2~crm.accounts:C2"
    assert "p" not in proposal(doc, _decided(doc, two), names_of=("shop.dealers",))[0]["entries"][-1]["link"]


def test_no_self_link_and_no_two_records_of_one_object() -> None:
    doc = _estate()
    two_dealers = "same:shop.dealers:D1~shop.dealers:D2"
    decided = {**_decided(doc), two_dealers: line(pair(doc, "D1~shop.dealers:D2"))}
    assert proposal(doc, decided, names_of=("shop.dealers",))[1]["not_written"] == [{"subject": two_dealers, "why": "one_object"}]


def test_no_link_whose_master_list_is_not_in_the_file() -> None:
    doc = _estate()
    for decided, names in ((_decided(doc, LINK_C1, dealers=False), ("shop.dealers",)), (_decided(doc, LINK_C1), ())):
        document, report = proposal(doc, decided, names_of=names)
        assert "format" not in document and {"subject": LINK_C1, "why": "master_not_in_the_file"} in report["not_written"]


def test_no_record_linked_to_another_record_already() -> None:
    doc = _estate()
    twice = ("same:shop.dealers:D2~crm.accounts:C2", "same:shop.dealers:D4~crm.accounts:C2")  # one account, two dealers
    elsewhere = "same:shop.dealers:D5~crm.accounts:C4"  # the build links the account to D6
    document, report = proposal(doc, _decided(doc, *twice, elsewhere, LINK_C1), names_of=("shop.dealers",))
    assert [e["link"]["linked"]["key"] for e in document["entries"] if "link" in e] == ["C1"]
    assert report["not_written"] == [{"subject": twice[0], "why": "linked_elsewhere"}, {"subject": twice[1], "why": "linked_elsewhere"},
                                     {"subject": elsewhere, "why": "linked_elsewhere"}]


def test_a_schema_the_writer_does_not_know_is_refused() -> None:
    with pytest.raises(ConfigError, match="version 3"):
        schema(3)
