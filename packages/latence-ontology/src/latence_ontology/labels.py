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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
) -> dict[str, Any]:
    """The export with ``iri``, ``labels`` and ``match`` on its elements."""

    def iri(el_id: str) -> dict[str, str]:
        return {"iri": base + iris[el_id]} if el_id in iris else {}

    def match(el: Mapping[str, Any]) -> dict[str, Any]:
        found = matches.get(str(el["id"]))
        if found is None:
            return {}
        members = {"members": "code"} if el.get("kind") == "reference" else {}
        return {"match": {**found, **members}}

    concepts = [
        {**c, **iri(c["id"]), "labels": _labels(c, language), **match(c)}
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
