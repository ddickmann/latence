"""RED^FM fixture builders, shared by the conftest and the suite tests.

They emit the REAL on-disk shape (``boundaries``, not ``start``/``end``; an exact
``text[start:end] == surfaceform`` equality), which the loader enforces — so a fixture that
drifted from the real corpus fails loudly rather than manufacturing a green run against a shape
the real data never produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def redfm_entity(text: str, surface: str, entity_type: str, uri: str) -> dict[str, Any]:
    """A RED^FM entity annotation in the REAL on-disk shape.

    ``boundaries`` (not ``start``/``end``) and an exact ``text[start:end] == surfaceform``
    equality are what the real JSONL carries at the pinned revision, and the loader enforces
    both — so a fixture that drifted from the real shape would fail here rather than quietly
    scoring something the real corpus could never produce.
    """
    start = text.index(surface)
    return {
        "uri": uri,
        "boundaries": [start, start + len(surface)],
        "surfaceform": surface,
        "annotator": "Me",
        "type": entity_type,
    }


def redfm_row(
    docid: str,
    text: str,
    entities: Sequence[dict[str, Any]],
    relations: Sequence[tuple[dict[str, Any], str, dict[str, Any]]] = (),
    language: str = "en",
) -> dict[str, Any]:
    """One RED^FM JSONL row in the real shape."""
    return {
        "docid": docid,
        "title": docid,
        "uri": "Q1",
        "text": text,
        "lan": language,
        "entities": list(entities),
        "relations": [
            {
                "subject": subject,
                "predicate": {
                    "uri": "P0",
                    "boundaries": None,
                    "surfaceform": predicate,
                    "annotator": "NoSubject-Triple-aligner",
                },
                "object": object_,
                "sentence_id": 0,
                "dependency_path": None,
                "confidence": 0.9,
                "annotator": "NoSubject-Triple-aligner",
                "answer_true": 4,
                "answer_false": 0,
            }
            for subject, predicate, object_ in relations
        ],
    }
