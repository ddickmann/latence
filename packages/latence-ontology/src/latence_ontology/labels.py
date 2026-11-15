"""An element's identity in the export (ticket 57): its persistent IRI, its labels per language
and the host's key where the host's alias table names one.

- ``iri``: the base address and the local name the ledger holds (``iris.py``), on concepts,
  relations, measures and classes.
- ``labels``: the preferred labels (one per language) and the other labels, each a text with
  its language and who wrote it, as every free text of the export is
  (``{"pref": [{"text": "Make", "lang": "en", "origin": "engine"}], "alt": [...]}``): the label
  the build gives, in the estate's language (``language`` in the estate file, English where it
  names none), and the synonyms a metric view's dimension states for a value set (``catalog``).
  The Turtle file writes them as ``skos:prefLabel`` and ``skos:altLabel``. Nothing here
  hard-codes a language: a second one is another text in each list.
- ``match``: the host's key (``crosswalk.py``); on a value set also ``members: "code"``, which
  says that a member is the pair of that key and the value's code.

A concept's words in one language are an element a person decides on its own (ticket 58): its
id is ``labels:<concept id>@<language>`` and its claim covers the preferred label and the other
labels (``labels.claims``, per language); the decision in force on it is ``labels.decisions``.
An approved one may go to a host's glossary in a proposal file (``labels`` entry).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ledger import Decisions, claim_of


def label_element(concept_id: str, lang: str, said: Mapping[str, Any]) -> dict[str, Any]:
    """A concept's words in one language as an element a person decides."""
    return {
        "id": f"labels:{concept_id}@{lang}",
        "concept": concept_id,
        "lang": lang,
        "pref": [t["text"] for t in said.get("pref") or [] if t["lang"] == lang],
        "alt": sorted(t["text"] for t in said.get("alt") or [] if t["lang"] == lang),
    }


def _labels(el: Mapping[str, Any], language: str) -> dict[str, list[dict[str, str]]]:
    pref = str(el.get("label") or "")
    alt = [str(w) for w in dict.fromkeys(el.get("synonyms") or []) if str(w) and str(w) != pref]
    return {
        "pref": [{"text": pref, "lang": language, "origin": "engine"}] if pref else [],
        "alt": [{"text": w, "lang": language, "origin": "catalog"} for w in alt],
    }


def with_identity(
    export: dict[str, Any],
    base: str,
    iris: Mapping[str, str],
    matches: Mapping[str, Mapping[str, Any]],
    language: str = "en",
    decisions: Decisions | None = None,
) -> dict[str, Any]:
    """The export with ``iri``, ``labels`` and ``match`` on its elements; a concept's labels
    with their claim per language and the decision in force on them, asked of ``decisions``."""

    def decided(c: Mapping[str, Any]) -> dict[str, Any]:
        said = _labels(c, language)
        langs = sorted({t["lang"] for part in ("pref", "alt") for t in said[part]})
        claims, verdicts = {}, {}
        for lang in langs:
            el = label_element(str(c["id"]), lang, said)
            claims[lang] = claim_of("labels", el)
            v = decisions.verdict("labels", el) if decisions is not None else None
            if v is not None:
                verdicts[lang] = v
        return {**said, "claims": claims, **({"decisions": verdicts} if verdicts else {})}

    def iri(el_id: str) -> dict[str, str]:
        return {"iri": base + iris[el_id]} if el_id in iris else {}

    def match(el: Mapping[str, Any]) -> dict[str, Any]:
        found = matches.get(str(el["id"]))
        if found is None:
            return {}
        members = {"members": "code"} if el.get("kind") == "reference" else {}
        return {"match": {**found, **members}}

    concepts = [
        {**c, **iri(c["id"]), "labels": decided(c), **match(c)}
        for c in export["concepts"]
    ]
    relations = [{**r, **iri(r["id"])} for r in export["relations"]]
    measures = [
        {**m, **iri(m["id"]), "labels": _labels(m, language), **match(m)}
        for m in export["measures"]
    ]
    out = {**export, "concepts": concepts, "relations": relations, "measures": measures}
    classes = export.get("classes")
    if isinstance(classes, dict):
        out["classes"] = {
            **classes,
            **{
                part: [{**c, **iri(c["id"])} for c in classes.get(part, [])]
                for part in ("top", "classes")
                if part in classes
            },
        }
    return out
