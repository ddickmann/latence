"""The ontology in the standard forms other tools read: OWL 2 QL and SKOS, as one Turtle file.

- every class of the class tree and every concept is an ``owl:Class`` with its label,
  definition and ``rdfs:subClassOf``;
- a relation is an ``owl:ObjectProperty`` and an attribute an ``owl:DatatypeProperty``, each
  with domain and range;
- part-of is an existential restriction on ``partOf``, which OWL 2 QL allows on that side;
- a value set is a ``skos:ConceptScheme``, its values are ``skos:Concept`` with ``skos:broader``
  along every roll-up.

Identifiers are opaque (a hash of the element's id under the estate's base address); the
readable name is the label. A personal column is named and flagged, and no value of it is in
the file, because no such value is in the export this is written from.

What a person has not yet approved is written too, and marked: ``reviewStatus "proposed"`` on
the element, or on the axiom where the element itself is settled. A consumer that wants only
settled statements filters on that. Rows of master data, the links between records and the
measures are not in this file yet.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

PREFIXES = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""
XSD = (("INT", "integer"), ("DECIMAL", "decimal"), ("NUMERIC", "decimal"), ("DOUBLE", "double"), ("FLOAT", "double"), ("BOOL", "boolean"),
       ("TIMESTAMP", "dateTime"), ("DATE", "date"))


def _lit(text: object) -> str:
    s = str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{s}"'


def _defined(texts: list[dict[str, str]] | None) -> list[tuple[str, str]]:
    """One ``skos:definition`` per text, with its language tag."""
    return [("skos:definition", f"{_lit(t['text'])}@{t['lang']}") for t in texts or []]


def _xsd(sql_type: str) -> str:
    t = sql_type.upper()
    return "xsd:" + next((x for key, x in XSD if key in t), "string")


class _Writer:
    def __init__(self, base: str) -> None:
        self.base, self.lines = base, [PREFIXES + f"@prefix e: <{base}> .\n@prefix v: <{base}vocabulary/> .\n"]

    def iri(self, element_id: str) -> str:
        return "e:x" + hashlib.sha1(element_id.encode("utf-8")).hexdigest()[:14]

    def add(self, subject: str, pairs: list[tuple[str, str]]) -> None:
        body = " ;\n    ".join(f"{p} {o}" for p, o in pairs)
        self.lines.append(f"{subject} {body} .\n")

    def axiom(self, s: str, p: str, o: str, status: str) -> None:
        """A statement about two settled things that is itself not settled."""
        self.lines.append(f"[] a owl:Axiom ;\n    owl:annotatedSource {s} ;\n    owl:annotatedProperty {p} ;\n    owl:annotatedTarget {o} ;\n    v:reviewStatus {_lit(status)} .\n")


def _status(el: dict[str, Any]) -> str:
    verdict = (el.get("decision") or {}).get("verdict")
    return "approved" if verdict == "approved" else "stated" if el.get("status") == "declared" else "proposed"


def to_turtle(export: dict[str, Any], base: str) -> str:
    """The Turtle text for an export (``graph-export/3``). The same export gives the same text."""
    if not re.match(r"^[a-z][a-z0-9+.-]*:", base) or not base.endswith(("/", "#", ":")):
        raise ValueError("the base address must be an absolute IRI that ends in '/', '#' or ':'")
    w = _Writer(base)
    w.add("v:reviewStatus", [("a", "owl:AnnotationProperty"), ("rdfs:label", _lit("review status")),
                             ("rdfs:comment", _lit("stated by the catalog, approved by a person, or proposed and not yet reviewed"))])
    w.add("v:personal", [("a", "owl:AnnotationProperty"), ("rdfs:label", _lit("personal")), ("rdfs:comment", _lit("the column holds personal data; no value of it is exported"))])
    w.add("v:rollsUpTo", [("a", "owl:AnnotationProperty"), ("rdfs:label", _lit("rolls up to"))])
    w.add("v:partOf", [("a", "owl:ObjectProperty"), ("rdfs:label", _lit("part of"))])
    w.add("e:ontology", [("a", "owl:Ontology"), ("rdfs:label", _lit(export["meta"]["title"])),
                         ("rdfs:comment", _lit(f"Induced from the estate {export['meta']['source']}. Elements marked proposed have not been reviewed by a person."))])

    classes = export.get("classes") or {}
    for c in classes.get("top", []):
        w.add(w.iri(c["id"]), [("a", "owl:Class"), ("rdfs:label", _lit(c["label"])), *_defined(c["definition"]), ("v:reviewStatus", _lit("stated"))])
    for c in classes.get("classes", []):
        w.add(w.iri(c["id"]), [("a", "owl:Class"), ("rdfs:label", _lit(c["label"])), *_defined(c["definition"]),
                               ("rdfs:subClassOf", w.iri(c["parent"])), ("v:reviewStatus", _lit(_status(c)))])
    kind_of = {s["concept"]: s for s in classes.get("subclass_of", [])}

    schemes = {c["id"] for c in export["concepts"] if c["kind"] == "reference"}
    for c in export["concepts"]:
        iri, status = w.iri(c["id"]), _status(c)
        if c["id"] in schemes:
            pairs = [("a", "skos:ConceptScheme"), ("rdfs:label", _lit(c["label"])), ("v:reviewStatus", _lit(status))]
        else:
            pairs = [("a", "owl:Class"), ("rdfs:label", _lit(c["label"])), ("v:reviewStatus", _lit(status))]
        pairs.extend(_defined(c.get("definition") or (kind_of.get(c["id"]) or {}).get("definition")))
        w.add(iri, pairs)
        placed = kind_of.get(c["id"])
        if placed and c["id"] not in schemes:
            if _status(placed) == "proposed":
                w.axiom(iri, "rdfs:subClassOf", w.iri(placed["parent"]), "proposed")
            else:
                w.add(iri, [("rdfs:subClassOf", w.iri(placed["parent"]))])
        for a in c.get("attributes") or []:
            pairs = [("a", "owl:DatatypeProperty"), ("rdfs:label", _lit(a["label"])), ("rdfs:domain", iri), ("rdfs:range", _xsd(a["type"])),
                     ("v:reviewStatus", _lit(status))]
            if a.get("sensitive"):
                pairs.append(("v:personal", "true"))
            w.add(w.iri(f"{c['id']}#{a['column']}"), pairs)

    for r in export["relations"]:
        if r["from"] in schemes:
            continue  # a roll-up between two value sets is written with the schemes below
        prop, target = w.iri(r["id"]), w.iri(r["to"])
        pairs = [("a", "owl:ObjectProperty"), ("rdfs:label", _lit(r["label"])), ("rdfs:domain", w.iri(r["from"])), ("v:reviewStatus", _lit(_status(r)))]
        if r["to"] not in schemes:
            pairs.append(("rdfs:range", target))
        else:
            pairs.append(("rdfs:range", "skos:Concept"))
            pairs.append(("rdfs:comment", _lit(f"its values are concepts of the scheme {next(c['label'] for c in export['concepts'] if c['id'] == r['to'])}")))
        w.add(prop, pairs)

    structure = export.get("structure") or {}
    for p in structure.get("part_of", []):
        if p["kind"] != "part_of":
            continue  # a concept that links two things is a part of neither
        w.lines.append(f"{w.iri(p['from'])} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty v:partOf ; owl:someValuesFrom {w.iri(p['to'])} ] .\n")
        w.add(w.iri(p["id"]), [("a", "owl:NamedIndividual"), ("rdfs:label", _lit(f"{p['from']} is part of {p['to']}")), ("v:reviewStatus", _lit(_status(p)))])
    for r in structure.get("roll_ups", []):
        if r["from"] in schemes and r["to"] in schemes:
            w.add(w.iri(r["from"]), [("v:rollsUpTo", w.iri(r["to"]))])
            w.add(w.iri(r["id"]), [("a", "owl:NamedIndividual"), ("rdfs:label", _lit(r["label"])), ("v:reviewStatus", _lit(_status(r)))])

    of_scheme = {n["id"]: n for n in export.get("nodes", []) if n["concept"] in schemes}
    for n in of_scheme.values():
        w.add(w.iri(n["id"]), [("a", "skos:Concept"), ("skos:inScheme", w.iri(n["concept"])), ("skos:prefLabel", _lit(n["label"]))])
    for e in export.get("edges", []):
        if e["kind"] == "broader" and e["from"] in of_scheme and e["to"] in of_scheme:
            w.add(w.iri(e["from"]), [("skos:broader", w.iri(e["to"]))])
    return "\n".join(w.lines)
