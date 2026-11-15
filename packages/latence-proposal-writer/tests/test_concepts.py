"""The entries over concepts (ticket 58): ``match``, ``broader`` and ``labels`` in a file of
format ``layer-proposal/3``, in the shapes the host's tool reads (its ticket 304, as the owner
decided it on 15 Nov: a reviewer approves a match and every named domain's data owner certifies
it; a close match is never placed). The estate here is synthetic: two domains, ``sales``
(schema ``shop``) and ``service`` (schema ``svc``), share a value set of makers, which a step
from models rolls up to."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from _proposal_estate import claim, in_force, line
from latence_proposal_writer import in_layer, proposal, read_receipts, schema, to_yaml

DOMAINS = {"sales": {"schemas": ["shop"], "data_owner": "owner-sales"},
           "service": {"schemas": ["svc"], "data_owner": "owner-service"}}
KEY = "dimension:conformed.maker"
LABELS = "labels:concept:value:maker@en"


def _estate(key: str | None = KEY) -> dict[str, Any]:
    def holds(owner: str, column: str, values: int, n: int) -> dict[str, Any]:
        return {"id": f"relation:{owner}~{column}", "from": f"concept:{owner}",
                "to": f"concept:value:{column}", "columns": [column], "claim": claim(n),
                "signals": [{"kind": "data", "code": "shared_column.holds",
                             "params": {"values": values}, "ref": ""}]}

    maker = {"id": "concept:value:maker", "kind": "reference", "column": "maker",
             "objects": ["shop.claims", "svc.repairs", "svc.visits"], "claim": claim(1),
             "label": "Maker",
             "labels": {"pref": [{"text": "Maker", "lang": "en", "origin": "engine"}],
                        "alt": [{"text": "brand", "lang": "en", "origin": "catalog"},
                                {"text": "make", "lang": "en", "origin": "catalog"}],
                        "claims": {"en": claim(9)}},
             **({"match": {"key": key, "namespace": "conformed", "read_from": "0" * 16}}
                if key else {})}
    model = {"id": "concept:value:model", "kind": "reference", "column": "model",
             "objects": ["shop.claims", "svc.repairs"], "claim": claim(2), "label": "Model"}
    nodes = [{"id": f"value:maker:{m}", "concept": "concept:value:maker", "label": m, "rows": r}
             for m, r in (("ALFA", 500), ("BETA", 300))]
    nodes += [{"id": f"value:model:{m}", "concept": "concept:value:model", "label": m, "rows": r}
              for m, r in (("A1", 200), ("A2", 300), ("B1", 300))]
    edges = [{"kind": "broader", "from": f"value:model:{a}", "to": f"value:maker:{b}",
              "relation": "roll_up:value:model->value:maker", "rows": r}
             for a, b, r in (("A1", "ALFA", 200), ("A2", "ALFA", 297), ("B1", "BETA", 300))]
    return {
        "format": "graph-export/3", "concepts": [maker, model],
        "relations": [holds("shop.claims", "maker", 2, 10), holds("svc.repairs", "maker", 2, 11),
                      holds("svc.visits", "maker", 1, 12)],
        "measures": [],
        "structure": {"roll_ups": [{"id": "roll_up:value:model->value:maker",
                                    "from": "concept:value:model", "to": "concept:value:maker",
                                    "kind": "roll_up", "claim": claim(20)}],
                      "equivalent": []},
        "classes": {"classes": [], "subclass_of": []},
        "nodes": nodes, "edges": edges, "review": [], "label_requests": {}, "meta": {},
    }


def _certified(*owners: str, on: str = claim(1)) -> list[dict[str, Any]]:
    return [{"subject": "concept:value:maker", "by": o, "at": "2026-11-14", "claim": on}
            for o in owners]


def _maker(doc: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = doc["concepts"][0]
    return found


def test_a_match_both_data_owners_certified_is_written_in_the_hosts_shape() -> None:
    doc = _estate()
    document, report = proposal(doc, in_force(line(_maker(doc))), domains=DOMAINS,
                                certifications=_certified("owner-sales", "owner-service"))
    text = to_yaml(document, "two-domains")
    read = yaml.safe_load(text)
    jsonschema.validate(read, schema(3))
    assert read["format"] == "layer-proposal/3"
    assert "close: false" in text  # a truth value, not a text
    assert read["entries"] == [{
        "match": {"key": KEY, "concepts": [
            {"domain": "sales", "object": "shop.claims", "column": "maker"},
            {"domain": "service", "object": "svc.repairs", "column": "maker"}], "close": False},
        "origin": {"element": "concept:value:maker", "claim": claim(1),
                   "decision": {"verdict": "approved", "by": "reviewer-1", "at": "2026-11-12"},
                   "certified": [{"domain": "sales", "by": "owner-sales", "at": "2026-11-14"},
                                 {"domain": "service", "by": "owner-service",
                                  "at": "2026-11-14"}]}}]
    assert [e["kind"] for e in report["entries"]] == ["match"]


def test_a_match_is_held_back_without_every_certification_or_without_a_key() -> None:
    doc = _estate()
    decided = in_force(line(_maker(doc)))
    for certs in (_certified("owner-sales"), _certified(),
                  _certified("owner-sales", "owner-service", on=claim(99)),
                  _certified("owner-sales", "reviewer-1")):
        document, report = proposal(doc, decided, domains=DOMAINS, certifications=certs)
        assert document["entries"] == [] and "format" not in document
        assert report["not_written"] == [{"subject": "concept:value:maker", "why": "not_certified"}]
    keyless = _estate(key=None)
    document, report = proposal(keyless, in_force(line(_maker(keyless))), domains=DOMAINS,
                                certifications=_certified("owner-sales", "owner-service"))
    assert document["entries"] == []
    assert report["not_written"] == [{"subject": "concept:value:maker", "why": "no_key"}]
    # Without the estate's domains no match is written, as before this version.
    document, report = proposal(doc, decided,
                                certifications=_certified("owner-sales", "owner-service"))
    assert document["entries"] == [] and report["not_carried"] == {"concept": 1}


def test_an_approved_step_rolls_each_member_up_with_what_the_data_showed() -> None:
    doc = _estate()
    step = doc["structure"]["roll_ups"][0]
    document, report = proposal(doc, in_force(line(step)))
    read = yaml.safe_load(to_yaml(document, "two-domains"))
    jsonschema.validate(read, schema(3))
    origin = {"element": step["id"], "claim": claim(20),
              "decision": {"verdict": "approved", "by": "reviewer-1", "at": "2026-11-12"}}
    assert read["entries"] == [
        {"broader": {"object": "shop.claims", "column": "model", "value": "A1",
                     "broader": ["ALFA"], "support": {"rows": 200, "share": 1.0}},
         "origin": origin},
        {"broader": {"object": "shop.claims", "column": "model", "value": "A2",
                     "broader": ["ALFA"], "support": {"rows": 297, "share": 0.99}},
         "origin": origin},
        {"broader": {"object": "shop.claims", "column": "model", "value": "B1",
                     "broader": ["BETA"], "support": {"rows": 300, "share": 1.0}},
         "origin": origin},
    ]
    assert proposal(doc, in_force(line(step, verdict="rejected")))[0]["entries"] == []
    stale = proposal(doc, in_force(line(step, stated=claim(77))))[1]["stale"]
    assert stale == [{"subject": step["id"], "decided_on": claim(77), "now": claim(20)}]


def test_labels_are_written_only_where_a_person_approved_the_words() -> None:
    doc = _estate()
    words = {"id": LABELS, "claim": claim(9)}
    document, report = proposal(doc, in_force(line(words)))
    read = yaml.safe_load(to_yaml(document, "two-domains"))
    jsonschema.validate(read, schema(3))
    assert read["entries"] == [{
        "labels": {"object": "shop.claims", "column": "maker", "language": "en",
                   "names": "Maker", "synonyms": ["brand", "make"]},
        "origin": {"element": LABELS, "claim": claim(9),
                   "decision": {"verdict": "approved", "by": "reviewer-1", "at": "2026-11-12"}}}]
    # The concept's own approval is on what it states, not on its words: no labels entry.
    assert proposal(doc, in_force(line(_maker(doc))))[0]["entries"] == []
    # Words decided on other words are stale.
    assert proposal(doc, in_force(line(words, stated=claim(8))))[1]["stale"] == [
        {"subject": LABELS, "decided_on": claim(8), "now": claim(9)}]


def test_a_host_on_version_2_gets_no_entry_over_concepts() -> None:
    doc = _estate()
    step = doc["structure"]["roll_ups"][0]
    decided = in_force(line(step), line({"id": LABELS, "claim": claim(9)}), line(_maker(doc)))
    document, report = proposal(doc, decided, concepts_too=False, domains=DOMAINS,
                                certifications=_certified("owner-sales", "owner-service"))
    assert document == {"source": "ontology", "entries": []}
    assert report["not_carried"] == {"concept": 1, "labels": 1, "roll_up": 1}


def test_the_receipt_says_which_release_placed_a_match_a_step_and_labels(tmp_path: Path) -> None:
    doc = _estate()
    step = doc["structure"]["roll_ups"][0]
    approved = {"verdict": "approved", "by": "reviewer-1"}
    doc["concepts"][0] = {**_maker(doc), "decision": approved,
                          "labels": {**_maker(doc)["labels"], "decisions": {"en": approved}}}
    doc["structure"]["roll_ups"][0] = {**step, "decision": approved}
    receipt = {"format": "layer-receipt/1", "library": "sales", "version": "2026.11.15-1",
               "placed": [
                   {"entry": "p.yaml#1", "kind": "match", "element": "concept:value:maker",
                    "claim": claim(1)},
                   {"entry": "p.yaml#2", "kind": "broader", "element": step["id"],
                    "claim": claim(20)},
                   {"entry": "p.yaml#5", "kind": "labels", "element": LABELS, "claim": claim(9)}],
               "already_there": [], "refused": []}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    marked, count = in_layer(doc, read_receipts((path,)))
    release = {"library": "sales", "version": "2026.11.15-1"}
    assert count == 3
    assert marked["concepts"][0]["in_layer"] == release
    assert marked["concepts"][0]["labels"]["in_layer"] == {"en": release}
    assert marked["structure"]["roll_ups"][0]["in_layer"] == release
