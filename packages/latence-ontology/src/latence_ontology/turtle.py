"""The ontology in the standard forms other tools read: OWL 2 QL and SKOS, as one Turtle file.

- every class of the class tree and every concept is an ``owl:Class`` with its label,
  definition and ``rdfs:subClassOf``;
- a relation is an ``owl:ObjectProperty`` and an attribute an ``owl:DatatypeProperty``, each
  with domain and range;
- part-of is an existential restriction on ``partOf``, which OWL 2 QL allows on that side;
- a value set is a ``skos:ConceptScheme``, its values are ``skos:Concept`` with ``skos:broader``
  along every roll-up.

Identifiers are opaque and persistent: the IRI the ledger holds for the element (``iris.py``:
minted once from its first id, carried over a rename a person confirmed), under the estate's
base address; the readable name is the label, ``skos:prefLabel`` per language with its
``skos:altLabel``s. Where the host's alias table names an element's key, the element is
``skos:exactMatch`` the key (``crosswalk.py``), and a value of a matched value set names its
member key: the set's key and its code. A personal column is named and flagged, and no value
of it is in the file, because no such value is in the export this is written from.

What a person has not yet approved is written too, and marked: ``reviewStatus "proposed"`` on
the element, or on the axiom where the element itself is settled. A consumer that wants only
settled statements filters on that. Rows of master data, the links between records and the
measures are not in this file yet.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .iris import minted

PREFIXES = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""
XSD = (
    ("INT", "integer"),
    ("DECIMAL", "decimal"),
    ("NUMERIC", "decimal"),
    ("DOUBLE", "double"),
    ("FLOAT", "double"),
    ("BOOL", "boolean"),
    ("TIMESTAMP", "dateTime"),
    ("DATE", "date"),
)


def _lit(text: object) -> str:
    s = (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{s}"'


def _defined(texts: list[dict[str, str]] | None) -> list[tuple[str, str]]:
    """One ``skos:definition`` per text, with its language tag."""
    return [("skos:definition", f"{_lit(t['text'])}@{t['lang']}") for t in texts or []]


def _xsd(sql_type: str) -> str:
    t = sql_type.upper()
    return "xsd:" + next((x for key, x in XSD if key in t), "string")


KEY_BASE = "urn:concept-key:"
"""Where a host's concept key is written as an IRI (``<urn:concept-key:dimension:ns.make>``),
unless the estate file names another (``export.key_iri``)."""


def _labelled(el: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``skos:prefLabel`` and ``skos:altLabel`` per language, from the element's ``labels``."""
    said = el.get("labels") or {}
    return [
        (prop, f"{_lit(t['text'])}@{t['lang']}")
        for prop, part in (("skos:prefLabel", "pref"), ("skos:altLabel", "alt"))
        for t in said.get(part) or []
    ]


class _Writer:
    def __init__(
        self, base: str, iris: Mapping[str, str] | None = None, key_base: str = KEY_BASE
    ) -> None:
        self.base, self.lines = (
            base,
            [PREFIXES + f"@prefix e: <{base}> .\n@prefix v: <{base}vocabulary/> .\n"],
        )
        self.iris, self.key_base = dict(iris or {}), key_base

    def iri(self, element_id: str) -> str:
        """The element's IRI as the ledger holds it; an id the ledger does not give one (a row
        of master data, a value) is named as the ledger would mint it."""
        return "e:" + self.iris.get(element_id, minted(element_id))

    def key(self, key: str) -> str:
        return f"<{self.key_base}{key}>"

    def add(self, subject: str, pairs: list[tuple[str, str]]) -> None:
        body = " ;\n    ".join(f"{p} {o}" for p, o in pairs)
        self.lines.append(f"{subject} {body} .\n")

    def axiom(self, s: str, p: str, o: str, status: str) -> None:
        """A statement about two settled things that is itself not settled."""
        self.lines.append(
            f"[] a owl:Axiom ;\n    owl:annotatedSource {s} ;\n    owl:annotatedProperty {p} ;\n  "
            f"  owl:annotatedTarget {o} ;\n    v:reviewStatus {_lit(status)} .\n"
        )


def _status(el: dict[str, Any]) -> str:
    verdict = (el.get("decision") or {}).get("verdict")
    return (
        "approved"
        if verdict == "approved"
        else "stated"
        if el.get("status") == "declared"
        else "proposed"
    )


def to_turtle(
    export: dict[str, Any],
    base: str,
    iris: Mapping[str, str] | None = None,
    key_base: str = KEY_BASE,
) -> str:
    """The Turtle text for an export (``graph-export/3``). The same export gives the same text.
    ``iris`` are the local names the ledger holds, by element id."""
    if not re.match(r"^[a-z][a-z0-9+.-]*:", base) or not base.endswith(("/", "#", ":")):
        raise ValueError("the base address must be an absolute IRI that ends in '/', '#' or ':'")
    w = _Writer(base, iris, key_base)
    w.add(
        "v:reviewStatus",
        [
            ("a", "owl:AnnotationProperty"),
            ("rdfs:label", _lit("review status")),
            (
                "rdfs:comment",
                _lit(
                    "stated by the catalog, approved by a person, or proposed and not yet reviewed"
                ),
            ),
        ],
    )
    w.add(
        "v:personal",
        [
            ("a", "owl:AnnotationProperty"),
            ("rdfs:label", _lit("personal")),
            ("rdfs:comment", _lit("the column holds personal data; no value of it is exported")),
        ],
    )
    if any(a.get("masked") for c in export["concepts"] for a in c.get("attributes") or []):
        w.add(
            "v:masked",
            [
                ("a", "owl:AnnotationProperty"),
                ("rdfs:label", _lit("masked")),
                (
                    "rdfs:comment",
                    _lit("the catalog masks the column; the engine reads no value of it"),
                ),
            ],
        )
    w.add("v:rollsUpTo", [("a", "owl:AnnotationProperty"), ("rdfs:label", _lit("rolls up to"))])
    w.add("v:partOf", [("a", "owl:ObjectProperty"), ("rdfs:label", _lit("part of"))])
    for name, label in (
        ("v:member", "member key: a value's dimension key and its code, as the host keys it"),
        ("v:dimension", "the dimension key of a member"),
        ("v:code", "the stored code of a member"),
    ):
        w.add(name, [("a", "owl:AnnotationProperty"), ("rdfs:label", _lit(label))])
    w.add(
        "e:ontology",
        [
            ("a", "owl:Ontology"),
            ("rdfs:label", _lit(export["meta"]["title"])),
            (
                "rdfs:comment",
                _lit(
                    f"Induced from the estate {export['meta']['source']}. Elements marked "
                    "proposed have not been reviewed by a person."
                ),
            ),
        ],
    )

    classes = export.get("classes") or {}
    for c in classes.get("top", []):
        w.add(
            w.iri(c["id"]),
            [
                ("a", "owl:Class"),
                ("rdfs:label", _lit(c["label"])),
                *_defined(c["definition"]),
                ("v:reviewStatus", _lit("stated")),
            ],
        )
    for c in classes.get("classes", []):
        w.add(
            w.iri(c["id"]),
            [
                ("a", "owl:Class"),
                ("rdfs:label", _lit(c["label"])),
                *_defined(c["definition"]),
                ("rdfs:subClassOf", w.iri(c["parent"])),
                ("v:reviewStatus", _lit(_status(c))),
            ],
        )
    kind_of = {s["concept"]: s for s in classes.get("subclass_of", [])}

    schemes = {c["id"] for c in export["concepts"] if c["kind"] == "reference"}
    for c in export["concepts"]:
        iri, status = w.iri(c["id"]), _status(c)
        if c["id"] in schemes:
            pairs = [
                ("a", "skos:ConceptScheme"),
                ("rdfs:label", _lit(c["label"])),
                ("v:reviewStatus", _lit(status)),
            ]
        else:
            pairs = [
                ("a", "owl:Class"),
                ("rdfs:label", _lit(c["label"])),
                ("v:reviewStatus", _lit(status)),
            ]
        pairs.extend(_labelled(c))
        if c.get("match"):
            pairs.append(("skos:exactMatch", w.key(str(c["match"]["key"]))))
        pairs.extend(
            _defined(c.get("definition") or (kind_of.get(c["id"]) or {}).get("definition"))
        )
        w.add(iri, pairs)
        placed = kind_of.get(c["id"])
        if placed and c["id"] not in schemes:
            if _status(placed) == "proposed":
                w.axiom(iri, "rdfs:subClassOf", w.iri(placed["parent"]), "proposed")
            else:
                w.add(iri, [("rdfs:subClassOf", w.iri(placed["parent"]))])
        for a in c.get("attributes") or []:
            pairs = [
                ("a", "owl:DatatypeProperty"),
                ("rdfs:label", _lit(a["label"])),
                ("rdfs:domain", iri),
                ("rdfs:range", _xsd(a["type"])),
                ("v:reviewStatus", _lit(status)),
            ]
            if a.get("masked"):
                pairs.append(
                    ("v:masked", "true")
                )  # withheld by the catalog's mask, not known to be personal
            elif a.get("sensitive"):
                pairs.append(("v:personal", "true"))
            w.add(w.iri(f"{c['id']}#{a['column']}"), pairs)

    for r in export["relations"]:
        if r["from"] in schemes:
            continue  # a roll-up between two value sets is written with the schemes below
        prop, target = w.iri(r["id"]), w.iri(r["to"])
        pairs = [
            ("a", "owl:ObjectProperty"),
            ("rdfs:label", _lit(r["label"])),
            ("rdfs:domain", w.iri(r["from"])),
            ("v:reviewStatus", _lit(_status(r))),
        ]
        if r["to"] not in schemes:
            pairs.append(("rdfs:range", target))
        else:
            pairs.append(("rdfs:range", "skos:Concept"))
            pairs.append(
                (
                    "rdfs:comment",
                    _lit(
                        "its values are concepts of the scheme "
                        f"{next(c['label'] for c in export['concepts'] if c['id'] == r['to'])}"
                    ),
                )
            )
        w.add(prop, pairs)

    structure = export.get("structure") or {}
    for p in structure.get("part_of", []):
        if p["kind"] != "part_of":
            continue  # a concept that links two things is a part of neither
        restriction = (
            f"[ a owl:Restriction ; owl:onProperty v:partOf ; owl:someValuesFrom {w.iri(p['to'])} ]"
        )
        if _status(p) == "proposed":
            w.axiom(w.iri(p["from"]), "rdfs:subClassOf", restriction, "proposed")
        else:
            w.lines.append(f"{w.iri(p['from'])} rdfs:subClassOf {restriction} .\n")
        w.add(
            w.iri(p["id"]),
            [
                ("a", "owl:NamedIndividual"),
                ("rdfs:label", _lit(f"{p['from']} is part of {p['to']}")),
                ("v:reviewStatus", _lit(_status(p))),
            ],
        )
    step_status = {r["id"]: _status(r) for r in structure.get("roll_ups", [])}
    for r in structure.get("roll_ups", []):
        if r["from"] in schemes and r["to"] in schemes:
            if step_status[r["id"]] == "proposed":
                w.axiom(w.iri(r["from"]), "v:rollsUpTo", w.iri(r["to"]), "proposed")
            else:
                w.add(w.iri(r["from"]), [("v:rollsUpTo", w.iri(r["to"]))])
            w.add(
                w.iri(r["id"]),
                [
                    ("a", "owl:NamedIndividual"),
                    ("rdfs:label", _lit(r["label"])),
                    ("v:reviewStatus", _lit(_status(r))),
                ],
            )

    of_scheme = {n["id"]: n for n in export.get("nodes", []) if n["concept"] in schemes}
    matched = {c["id"]: c for c in export["concepts"] if c.get("match")}
    for n in of_scheme.values():
        pairs = [
            ("a", "skos:Concept"),
            ("skos:inScheme", w.iri(n["concept"])),
            ("skos:prefLabel", _lit(n["label"])),
        ]
        scheme = matched.get(n["concept"])
        if scheme is not None:  # a member: its dimension's key and its code, never one string
            code = str(n["id"]).removeprefix(f"value:{scheme['column']}:")
            pairs.append(
                (
                    "v:member",
                    f"[ v:dimension {w.key(str(scheme['match']['key']))} ; v:code {_lit(code)} ]",
                )
            )
        w.add(w.iri(n["id"]), pairs)
    for e in export.get("edges", []):
        if e["kind"] == "broader" and e["from"] in of_scheme and e["to"] in of_scheme:
            # A value sits under another because of a step; until a person approves the step,
            # that is a proposal like the step itself.
            if step_status.get(str(e.get("relation")), "proposed") == "proposed":
                w.axiom(w.iri(e["from"]), "skos:broader", w.iri(e["to"]), "proposed")
            else:
                w.add(w.iri(e["from"]), [("skos:broader", w.iri(e["to"]))])
    return "\n".join(w.lines)
