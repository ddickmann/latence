"""The Turtle file: it parses, says what kind of thing each concept is, marks what nobody has
approved, and holds no personal value."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rdflib
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from estate.classes import propose_classes
from estate.export import build_export
from estate.induce import induce
from estate.instances import build_instances
from estate.reader_duckdb import DuckDBReader
from estate.structure import broader_edges, build_structure
from estate.turtle import to_turtle

from .conftest import OPTIONS

BASE = "https://example.invalid/estate/toy/"
V = rdflib.Namespace(BASE + "vocabulary/")


def _export(reader: DuckDBReader) -> dict:
    model = induce(reader, **OPTIONS, without_personal_columns=("shop.dealers",))
    inst = build_instances(reader, model, personal_objects=("shop.owners", "shop.vehicles"), without_personal_columns=("shop.dealers",))
    structure = build_structure(reader, model, ("shop.owners", "shop.vehicles"))
    inst = {**inst, "edges": inst["edges"] + broader_edges(reader, structure, {n["id"] for n in inst["nodes"]})}
    classes = propose_classes(model, structure, ("shop.owners", "shop.vehicles"))
    return build_export(model, source="toy", title="Toy estate", instances=inst, structure=structure, classes=classes)


def test_the_file_parses_and_carries_classes_properties_and_value_hierarchies(reader: DuckDBReader) -> None:
    export = _export(reader)
    text = to_turtle(export, BASE)
    g = rdflib.Graph().parse(data=text, format="turtle")
    labels = {str(o): s for s, o in g.subject_objects(RDFS.label)}
    assert (labels["Dealer"], RDF.type, OWL.Class) in g and (labels["Organisation"], RDF.type, OWL.Class) in g
    assert (labels["Make"], RDF.type, SKOS.ConceptScheme) in g
    assert len(set(g.subjects(RDF.type, OWL.ObjectProperty))) >= len([r for r in export["relations"] if not r["from"].startswith("concept:value:")])
    assert len(set(g.subjects(RDF.type, OWL.DatatypeProperty))) == sum(len(c.get("attributes") or []) for c in export["concepts"])
    values = set(g.subjects(RDF.type, SKOS.Concept))
    assert len(values) == sum(n["concept"].startswith("concept:value:") for n in export["nodes"])
    # Every model under its make, as a proposal: the step is one, and nobody has approved it.
    assert len(list(g.subjects(OWL.annotatedProperty, SKOS.broader))) == sum(e["kind"] == "broader" for e in export["edges"]) > 0
    assert list(g.subject_objects(SKOS.broader)) == []
    assert to_turtle(export, BASE) == text  # the same export gives the same file


def test_what_nobody_approved_is_marked_on_the_element_or_on_the_axiom(reader: DuckDBReader) -> None:
    export = _export(reader)
    g = rdflib.Graph().parse(data=to_turtle(export, BASE), format="turtle")
    labels = {str(o): s for s, o in g.subject_objects(RDFS.label)}
    dealer, organisation = labels["Dealer"], labels["Organisation"]
    assert (dealer, RDFS.subClassOf, organisation) not in g  # proposed: not asserted as a plain triple
    axiom = next(a for a in g.subjects(OWL.annotatedSource, dealer) if (a, OWL.annotatedTarget, organisation) in g)
    assert str(g.value(axiom, V.reviewStatus)) == "proposed" and (axiom, OWL.annotatedProperty, RDFS.subClassOf) in g
    assert str(g.value(organisation, V.reviewStatus)) == "stated"
    approved = json.loads(json.dumps(export))
    next(s for s in approved["classes"]["subclass_of"] if s["concept"] == "concept:shop.dealers")["decision"] = {"verdict": "approved", "by": "a reviewer"}
    g2 = rdflib.Graph().parse(data=to_turtle(approved, BASE), format="turtle")
    labels2 = {str(o): s for s, o in g2.subject_objects(RDFS.label)}
    assert (labels2["Dealer"], RDFS.subClassOf, labels2["Organisation"]) in g2  # once a person approves, it is a plain statement


def test_a_personal_column_is_named_and_flagged_and_none_of_its_values_is_there(reader: DuckDBReader) -> None:
    text = to_turtle(_export(reader), BASE)
    g = rdflib.Graph().parse(data=text, format="turtle")
    flagged = {str(g.value(s, RDFS.label)) for s in g.subjects(V.personal, rdflib.Literal(True))}
    assert "Contact email" in flagged
    assert "@example.org" not in text and "Person 7" not in text
    assert "shop.dealers" not in "".join(str(s) for s in g.subjects())  # identifiers are opaque


def test_the_base_address_has_to_be_an_absolute_iri(reader: DuckDBReader) -> None:
    with pytest.raises(ValueError, match="absolute IRI"):
        to_turtle(_export(reader), "estate/toy")
    assert Path(__file__).exists()


def test_a_concept_that_links_two_things_is_written_as_a_part_of_neither(reader: DuckDBReader) -> None:
    export = _export(reader)
    step = {"signals": [], "status": "proposed", "claim": "0" * 16}
    export["structure"]["part_of"] = [
        {**step, "id": "part_of:a", "from": "concept:shop.claims", "to": "concept:shop.dealers", "kind": "part_of"},
        {**step, "id": "part_of:b", "from": "concept:shop.visits", "to": "concept:shop.dealers", "kind": "links"},
        {**step, "id": "part_of:c", "from": "concept:shop.visits", "to": "concept:shop.parts", "kind": "links"},
    ]
    g = rdflib.Graph().parse(data=to_turtle(export, BASE), format="turtle")
    labels = {str(o): s for s, o in g.subject_objects(RDFS.label)}
    restricted = {g.value(a, OWL.annotatedSource) for a, o in g.subject_objects(OWL.annotatedTarget) if (o, OWL.onProperty, V.partOf) in g}
    assert restricted == {labels["Claim"]}
