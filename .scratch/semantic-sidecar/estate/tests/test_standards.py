"""SHACL and PROV-O beside the Turtle file (ticket 42): both parse, the shapes are valid SHACL
and hold data to what is approved, and every element says where it came from and who decided."""

from __future__ import annotations

import json
from pathlib import Path

import rdflib
from pyshacl import validate
from rdflib.namespace import OWL, RDF

from estate.cli import build

from .test_ledger_export import _export, project  # noqa: F401  (the fixture)

SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
PROV = rdflib.Namespace("http://www.w3.org/ns/prov#")
RELATION = "relation:shop.claims.dealer_id->shop.dealers"


def _base(project: dict[str, Path]) -> str:
    import yaml

    cfg = yaml.safe_load(project["config"].read_text(encoding="utf-8"))
    return str(cfg.get("export", {}).get("base_iri", f"urn:estate:{cfg['name']}:"))


def _built(project: dict[str, Path]) -> tuple[rdflib.Graph, rdflib.Graph, rdflib.Graph]:
    build(**project)
    export = _export(project)
    relation = next(r for r in export["relations"] if r["id"] == RELATION)
    with open(project["out"] / "decisions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"subject": RELATION, "verdict": "approved", "by": "reviewer-1", "claim": relation["claim"]}) + "\n")
    build(**project)
    graphs = []
    for name in ("ontology.shacl.ttl", "ontology.prov.ttl", "ontology.ttl"):
        g = rdflib.Graph()
        g.parse(project["out"] / name, format="turtle")
        graphs.append(g)
    return graphs[0], graphs[1], graphs[2]


def test_the_shapes_are_valid_shacl_and_hold_data_only_to_what_is_approved(project: dict[str, Path]) -> None:
    shapes, _prov, owl = _built(project)
    conforms, _report, text = validate(rdflib.Graph(), shacl_graph=shapes, meta_shacl=True)
    assert conforms, text  # the shapes graph itself is valid SHACL
    targets = set(shapes.objects(None, SH.targetClass))
    assert targets and targets <= set(owl.subjects(RDF.type, OWL.Class))  # the same IRIs as the Turtle file
    export = _export(project)
    claims_label = next(c["label"] for c in export["concepts"] if c["id"] == "concept:shop.claims")
    shape = next(s for s in shapes.subjects(RDF.type, SH.NodeShape) if str(next(shapes.objects(s, rdflib.RDFS.label))) == claims_label)
    claims_cls = next(shapes.objects(shape, SH.targetClass))
    props = list(shapes.objects(shape, SH.property))
    deactivated = [p for p in props if (p, SH.deactivated, rdflib.Literal(True)) in shapes]
    relations = [p for p in props if (p, SH["class"], None) in shapes]
    approved = [p for p in relations if p not in deactivated and (p, SH.maxCount, None) in shapes]
    assert deactivated and len(approved) >= 1  # what is approved or stated holds; the proposed relations wait for review
    approved = [p for p in approved if str(next(shapes.objects(p, rdflib.URIRef("{base}vocabulary/reviewStatus".format(base=_base(project)))))) == "approved"]
    assert len(approved) == 1
    approved_shape = approved[0]
    path, target = next(shapes.objects(approved_shape, SH.path)), next(shapes.objects(approved_shape, SH["class"]))
    # A claim that names two dealers breaks the approved many-to-one relation; one dealer does not.
    data = rdflib.Graph()
    claim, d1, d2 = rdflib.URIRef("urn:test:c1"), rdflib.URIRef("urn:test:d1"), rdflib.URIRef("urn:test:d2")
    data.add((claim, RDF.type, claims_cls))
    for d in (d1, d2):
        data.add((d, RDF.type, target))
    data.add((claim, path, d1))
    assert validate(data, shacl_graph=shapes)[0]
    data.add((claim, path, d2))
    assert not validate(data, shacl_graph=shapes)[0]


def test_a_personal_column_may_hold_no_value(project: dict[str, Path]) -> None:
    shapes, _prov, _owl = _built(project)
    zero = list(shapes.subjects(SH.maxCount, rdflib.Literal(0)))
    export = _export(project)
    personal = [a for c in export["concepts"] if c["kind"] != "reference" for a in c.get("attributes") or [] if a["sensitive"]]
    assert personal and len(zero) == len(personal)


def test_every_element_says_it_came_from_the_build_and_a_decision_names_its_person(project: dict[str, Path]) -> None:
    _shapes, prov, owl = _built(project)
    export = _export(project)
    entities = set(prov.subjects(RDF.type, PROV.Entity))
    built = set(prov.subjects(PROV.wasGeneratedBy, None))
    assert len(built) == len(export["concepts"]) + len(export["relations"]) + len(export["measures"])
    assert built <= entities and all((e, PROV.wasAttributedTo, None) in prov for e in built)
    influenced = list(prov.subject_objects(PROV.wasInfluencedBy))
    assert len(influenced) == 1
    _element, decision = influenced[0]
    person = next(prov.objects(decision, PROV.wasAssociatedWith))
    assert (person, RDF.type, PROV.Person) in prov and str(next(prov.objects(person, rdflib.RDFS.label))) == "reviewer-1"
    derived = set(prov.subjects(PROV.wasDerivedFrom, None))
    assert derived and derived <= built and derived & set(owl.subjects(RDF.type, OWL.Class))
