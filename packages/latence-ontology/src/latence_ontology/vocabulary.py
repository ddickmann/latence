"""An outside vocabulary as the third voice on the class tree (ticket 55).

A published class hierarchy, given as a file the estate file names (``classes.vocabulary``),
proposes a top class for a concept whose label or name matches one of its classes' labels. It
votes like the other two voices: it proposes, it never settles a placement alone, and where it
disagrees with another voice the placement is contested.

The file states the vocabulary's name and version, its licence (an SPDX id) and where the
licence was read (``licence_source``), and its classes, each with its labels, its own id in the
vocabulary and the top class of the tree it falls under (``top``: one of the six). **The licence
is checked before the vocabulary is used**: a file that states none, or one the estate file does
not allow (``licences_allowed``, by default the permissive and public-domain dedications below),
is refused with a typed error; the licence and its source are recorded in the export
(``classes.voices.vocabulary``) with the file's digest.

Matching is on words, never on values: a concept's label and its object's name, lower-cased,
with ``_`` read as a space and a final plural ``s`` dropped, against each class's labels treated
the same way. A label two classes of different top classes carry matches nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from latence_core.errors import ConfigError

from .codes import TOP, say

ALLOWED = ("CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT", "BSD-3-Clause", "ODC-By-1.0", "PDDL-1.0")
"""The licences a vocabulary may carry where the estate file names none: public-domain
dedications and permissive licences whose duty is attribution at most. A share-alike licence
(CC-BY-SA) is allowed only where the estate file names it."""


def _word(text: str) -> str:
    w = re.sub(r"\s+", " ", str(text).replace("_", " ").strip().lower())
    return w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w


def read_vocabulary(path: Path, allowed: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The vocabulary file, its licence checked against ``allowed`` (``ALLOWED`` by default)."""
    raw = path.read_bytes()
    doc = json.loads(raw)
    for field in ("name", "version", "licence", "licence_source", "classes"):
        if not doc.get(field):
            raise ConfigError(f"{path}: a vocabulary states its {field}; this one does not")
    licence = str(doc["licence"])
    permitted = allowed if allowed is not None else ALLOWED
    if licence not in permitted:
        raise ConfigError(
            f"{path}: the vocabulary {doc['name']} is licensed {licence}, which the estate file "
            f"does not allow (allowed: {', '.join(permitted)}; classes.vocabulary.licences_allowed)"
        )
    for c in doc["classes"]:
        if c.get("top") not in TOP:
            raise ConfigError(
                f"{path}: the class {c.get('id')} falls under {c.get('top')!r}, which is no top "
                f"class ({', '.join(TOP)})"
            )
        if not c.get("id") or not c.get("labels"):
            raise ConfigError(f"{path}: every class states its id and its labels")
    return {**doc, "digest": hashlib.sha256(raw).hexdigest()[:16]}


def vocabulary_voice(
    pack: list[dict[str, Any]], vocabulary: dict[str, Any]
) -> dict[str, tuple[str, dict[str, Any]]]:
    """For each concept a class of the vocabulary names, its top class and the statement."""
    by_word: dict[str, list[dict[str, Any]]] = {}
    for c in vocabulary["classes"]:
        for label in c["labels"]:
            by_word.setdefault(_word(label), []).append(c)
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for concept in pack:
        names = [str(concept.get("label") or "")]
        obj = str(concept.get("object") or "")
        if obj:
            names.append(obj.rpartition(".")[2])
        found = [c for n in names if n for c in by_word.get(_word(n), [])]
        tops = {c["top"] for c in found}
        if len(tops) != 1:
            continue  # nothing, or classes of two top classes: no proposal
        first = sorted(found, key=lambda c: str(c["id"]))[0]
        out[concept["id"]] = (
            first["top"],
            say("vocabulary.matched", str(first["id"]), vocabulary=str(vocabulary["name"]),
                label=str(first["labels"][0])),
        )
    return out


def recorded(vocabulary: dict[str, Any]) -> dict[str, str]:
    """What the export records of the vocabulary a build used."""
    return {k: str(vocabulary[k]) for k in ("name", "version", "licence", "licence_source",
                                            "digest")}
