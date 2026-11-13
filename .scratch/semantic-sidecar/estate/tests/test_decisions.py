"""A reviewer's decision on a hierarchy step, a part-of or a class changes the layer, survives
a rebuild, and is never applied to a proposal that now says something else (ticket 28)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import rdflib
from rdflib.namespace import OWL, RDFS, SKOS

from estate.classes import propose_classes
from estate.cli import build
from estate.ledger import Decisions, claim_of
from estate.reader_duckdb import DuckDBReader
from estate.structure import broader_edges, build_structure
from estate.turtle import to_turtle

from .test_classes import MODEL, STRUCTURE, FakeModel
from .test_ledger_export import project  # noqa: F401
from .test_structure import _model, shop  # noqa: F401


def _decide(kind: str, el: dict[str, Any], verdict: str, with_claim: bool = True) -> dict[str, Any]:
    return {"subject": el["id"], "verdict": verdict, "by": "reviewer-1", **({"claim": claim_of(kind, el)} if with_claim else {})}


def _decisions(*lines: dict[str, Any]) -> Decisions:
    return Decisions({d["subject"]: d for d in lines})


def test_a_rejected_step_leaves_and_the_step_it_hid_comes_back(shop: DuckDBReader) -> None:  # noqa: F811
    before = build_structure(shop, _model())
    ids = {r["id"] for r in before["roll_ups"]}
    assert "roll_up:value:town->value:region" in ids and "roll_up:value:town->value:country" not in ids
    town_region = next(r for r in before["roll_ups"] if r["id"] == "roll_up:value:town->value:region")
    decisions = _decisions(_decide("roll_up", town_region, "rejected"))
    after = build_structure(shop, _model(), decisions=decisions)
    ids = {r["id"] for r in after["roll_ups"]}
    assert "roll_up:value:town->value:region" not in ids and "roll_up:value:town->value:country" in ids  # town under country directly now
    known = {f"value:town:T{k}" for k in range(60)} | {f"value:region:R{k}" for k in range(6)} | {f"value:country:C{k}" for k in range(2)}
    edges = broader_edges(shop, after, known)
    assert not any(e["relation"] == "roll_up:value:town->value:region" for e in edges)
    assert sum(e["relation"] == "roll_up:value:town->value:country" for e in edges) == 60
    assert decisions.summary()["rejected"] == 1 and decisions.summary()["stale"] == []


def test_an_approved_step_says_so_and_a_rejected_equivalence_gives_two_levels(shop: DuckDBReader) -> None:  # noqa: F811
    before = build_structure(shop, _model())
    same = before["equivalent"][0]
    assert same["concepts"] == ["concept:value:region", "concept:value:region_code"]
    region_country = next(r for r in before["roll_ups"] if r["id"] == "roll_up:value:region->value:country")
    after = build_structure(shop, _model(), decisions=_decisions(_decide("equivalent", same, "rejected"), _decide("roll_up", region_country, "approved")))
    assert after["equivalent"] == []
    steps = {r["id"]: r for r in after["roll_ups"]}
    assert steps["roll_up:value:region->value:country"]["decision"] == {"verdict": "approved", "by": "reviewer-1"}
    assert "roll_up:value:region_code->value:country" in steps  # the second name is a level of its own again


def _voice(parts_parent: str) -> FakeModel:
    voice = FakeModel({"classes": [], "concepts": [{"id": "concept:s.parts", "parent": parts_parent, "definition": "A spare part.", "reason": "as the names read"}]})
    voice.propose = lambda pack, top: voice.answer  # type: ignore[method-assign]
    return voice


def test_a_decision_is_not_applied_to_a_proposal_that_now_says_something_else() -> None:
    first = propose_classes(MODEL, STRUCTURE, voice=_voice("document"))
    part = next(s for s in first["subclass_of"] if s["concept"] == "concept:s.parts")
    assert part["parent"] == "class:document" and part["other_parent"] == "class:product"  # the model says document, the columns say product
    approval = _decide("subclass", part, "approved")
    kept = propose_classes(MODEL, STRUCTURE, voice=_voice("document"), decisions=_decisions(approval))
    settled = next(s for s in kept["subclass_of"] if s["concept"] == "concept:s.parts")
    assert settled["decision"]["verdict"] == "approved" and not settled["contested"]  # a person settled it
    # The model now says a part is an event. The id of the placement is the same; no voice says document any more.
    decisions = _decisions(approval)
    moved = propose_classes(MODEL, STRUCTURE, voice=_voice("event"), decisions=decisions)
    now = next(s for s in moved["subclass_of"] if s["concept"] == "concept:s.parts")
    assert now["parent"] == "class:event" and "decision" not in now
    assert decisions.summary()["stale"] == [{"subject": "subclass:s.parts", "decided_on": approval["claim"], "now": claim_of("subclass", now)}]
    assert decisions.summary()["approved"] == 0


def test_an_approval_of_what_the_other_voice_said_settles_the_placement() -> None:
    first = propose_classes(MODEL, STRUCTURE, voice=_voice("document"))
    part = next(s for s in first["subclass_of"] if s["concept"] == "concept:s.parts")
    for_the_columns = _decide("subclass", {**part, "parent": part["other_parent"]}, "approved")
    out = propose_classes(MODEL, STRUCTURE, voice=_voice("document"), decisions=_decisions(for_the_columns))
    now = next(s for s in out["subclass_of"] if s["concept"] == "concept:s.parts")
    assert now["parent"] == "class:product" and now["decision"]["verdict"] == "approved" and not now["contested"]


def test_a_decision_on_a_placement_needs_the_claim_it_was_made_on() -> None:
    first = propose_classes(MODEL, STRUCTURE)
    dealer = next(s for s in first["subclass_of"] if s["concept"] == "concept:s.dealers")
    decisions = _decisions(_decide("subclass", dealer, "rejected", with_claim=False))
    out = propose_classes(MODEL, STRUCTURE, decisions=decisions)
    assert any(s["concept"] == "concept:s.dealers" for s in out["subclass_of"])  # still there: the line does not say what it rejects
    assert decisions.summary()["without_claim"] == ["subclass:s.dealers"]


def test_a_rejected_placement_falls_to_the_other_voice_and_a_rejected_class_hands_on_its_members() -> None:
    voice = FakeModel({
        "classes": [{"name": "Service event", "parent": "event", "definition": "Something a workshop did."}],
        "concepts": [{"id": "concept:s.claims", "parent": "Service event", "definition": "A claim.", "reason": "dated"},
                     {"id": "concept:s.parts", "parent": "document", "definition": "A part.", "reason": "it has a description"},
                     {"id": "concept:s.dealers", "parent": "organisation", "definition": "A workshop.", "reason": "a name"}]})
    voice.propose = lambda pack, top: voice.answer  # type: ignore[method-assign]
    first = propose_classes(MODEL, STRUCTURE, voice=voice)
    by = {s["concept"]: s for s in first["subclass_of"]}
    assert by["concept:s.parts"]["parent"] == "class:document" and by["concept:s.parts"]["other_parent"] == "class:product"
    own = first["classes"][0]
    decisions = _decisions(_decide("subclass", by["concept:s.parts"], "rejected"), _decide("class", own, "rejected"),
                           _decide("subclass", by["concept:s.dealers"], "rejected"))
    after = propose_classes(MODEL, STRUCTURE, voice=voice, decisions=decisions)
    by = {s["concept"]: s for s in after["subclass_of"]}
    assert by["concept:s.parts"]["parent"] == "class:product" and not by["concept:s.parts"]["contested"]  # what the columns said
    assert after["classes"] == [] and by["concept:s.claims"]["parent"] == "class:event"  # the class is gone, its member sits under its parent
    assert "concept:s.dealers" not in by and "concept:s.dealers" in after["unplaced"]  # both voices said organisation: nothing else to fall to
    assert decisions.summary()["rejected"] == 3


def _export(project: dict[str, Path]) -> dict[str, Any]:  # noqa: F811
    return json.loads((project["out"] / "ontology.json").read_text())


def test_decisions_on_structure_and_classes_survive_a_rebuild_and_every_element_is_an_assertion(project: dict[str, Path]) -> None:  # noqa: F811
    build(**project)
    first = _export(project)
    step = next(r for r in first["structure"]["roll_ups"] if r["kind"] == "roll_up")
    placed = first["classes"]["subclass_of"][0]
    with open(project["out"] / "decisions.jsonl", "a") as fh:
        fh.write(json.dumps({"subject": step["id"], "verdict": "rejected", "by": "reviewer-1", "claim": step["claim"]}) + "\n")
        fh.write(json.dumps({"subject": placed["id"], "verdict": "approved", "by": "reviewer-1", "claim": placed["claim"]}) + "\n")
        fh.write(json.dumps({"subject": "roll_up:value:nothing->value:here", "verdict": "approved", "by": "reviewer-1", "claim": "0" * 16}) + "\n")
    for _ in range(2):
        summary = build(**project)
        export = _export(project)
        assert step["id"] not in {r["id"] for r in export["structure"]["roll_ups"]}
        assert not any(e.get("relation") == step["id"] for e in export["edges"])
        assert next(s for s in export["classes"]["subclass_of"] if s["id"] == placed["id"])["decision"] == {"verdict": "approved", "by": "reviewer-1"}
        assert summary["decisions"] == {"approved": 1, "rejected": 1, "unknown_subjects": ["roll_up:value:nothing->value:here"], "stale": [], "without_claim": []}
    rows = [json.loads(x) for x in (project["out"] / "assertions.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in rows}
    assert {"concept", "relation", "measure", "roll_up", "subclass"} <= kinds and all(r["actor"] for r in rows)
    assert step["id"] in {r["subject"] for r in rows}  # rejected: out of the layer, still in the ledger
    every = [el["id"] for group in (export["concepts"], export["relations"], export["measures"], export["classes"]["classes"], export["classes"]["subclass_of"],
                                    *(export["structure"][k] for k in ("roll_ups", "equivalent", "part_of", "describes"))) for el in group]
    assert set(every) <= {r["subject"] for r in rows}


def test_the_turtle_file_states_a_step_plainly_only_once_a_person_approved_it(shop: DuckDBReader) -> None:  # noqa: F811
    from estate.export import build_export

    def turtle(decisions: Decisions | None) -> rdflib.Graph:
        model = {**_model(), "measures": [], "meta": {"objects": 3, "metric_views": 0, "support_floor": 0.9, "dropped_below_floor": []}}
        for c in model["concepts"]:
            c.setdefault("status", "proposed"); c.setdefault("signals", []); c.setdefault("definition", ""); c.setdefault("instances", "available")
        for r in model["relations"]:
            r.update({"label": r["id"], "status": "proposed", "signals": [], "cardinality": "many_to_one", "support": None})
        st = build_structure(shop, model, decisions=decisions)
        nodes = [{"id": f"value:town:T{k}", "concept": "concept:value:town", "label": f"T{k}"} for k in range(60)] + \
                [{"id": f"value:region:R{k}", "concept": "concept:value:region", "label": f"R{k}"} for k in range(6)]
        edges = broader_edges(shop, st, {n["id"] for n in nodes})
        inst = {"nodes": nodes, "edges": edges, "review": [], "meta": {"nodes": len(nodes), "edges": len(edges), "review_items": 0, "certified": False, "nodes_per_concept": {}}}
        export = build_export(model, source="shop", title="Shop", instances=inst, structure=st)
        return rdflib.Graph().parse(data=to_turtle(export, "https://example.invalid/shop/"), format="turtle")

    g = turtle(None)
    assert list(g.subject_objects(SKOS.broader)) == []  # nobody approved the step: no value is stated to sit under another
    marked = [a for a in g.subjects(OWL.annotatedProperty, SKOS.broader)]
    assert len(marked) == 60 and all(str(g.value(a, rdflib.URIRef("https://example.invalid/shop/vocabulary/reviewStatus"))) == "proposed" for a in marked)
    assert not any((o, OWL.onProperty, None) in g for _s, o in g.subject_objects(RDFS.subClassOf))  # the part-of is a proposal too
    st = build_structure(shop, _model())
    step = next(r for r in st["roll_ups"] if r["id"] == "roll_up:value:town->value:region")
    part = st["part_of"][0]
    g2 = turtle(_decisions(_decide("roll_up", step, "approved"), _decide("part_of", part, "approved")))
    assert len(list(g2.subject_objects(SKOS.broader))) == 60 and list(g2.subjects(OWL.annotatedProperty, SKOS.broader)) == []
    assert sum((o, OWL.onProperty, None) in g2 for _s, o in g2.subject_objects(RDFS.subClassOf)) == 1
