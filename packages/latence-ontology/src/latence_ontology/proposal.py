"""The proposal file for a host's semantic layer: a thin call into the optional adapter package
``latence-proposal-writer`` (ADR-0070, ticket 43), which writes the file from a build's export
and the decisions in force, and reads the host's receipt back. Nothing of the writer lives here;
this module names this package's commands in the file's opening comment, and refuses to write a
document that breaks the shipped schema of the version it states. The check lives here rather
than in the writer because the writer imports nothing but the standard library and the core
(ADR-0070), and this package already carries jsonschema for the export's own guard."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from typing import Any

import jsonschema
from latence_core.errors import ContractError
from latence_proposal_writer import in_layer, proposal, read_receipts, schema
from latence_proposal_writer import to_yaml as _to_yaml

COMMAND = "latence-ontology propose"
_VERSION_OF = {None: 1, "layer-proposal/2": 2, "layer-proposal/3": 3}

__all__ = ["COMMAND", "checked", "in_layer", "proposal", "read_receipts", "to_yaml"]


@cache
def _validator(version: int) -> jsonschema.protocols.Validator:
    return jsonschema.Draft202012Validator(schema(version))


def checked(document: Mapping[str, Any]) -> None:
    """Refuse a document that breaks the shipped schema of the version it states (no ``format``
    is version 1): the file is the one seam a host reads (ADR-0070)."""
    stated = document.get("format")
    version = _VERSION_OF.get(stated) if stated is None or isinstance(stated, str) else None
    if version is None:
        raise ContractError(f"the proposal file states format {stated!r}, which no shipped "
                            f"schema describes ({', '.join(k for k in _VERSION_OF if k)} or none)")
    broken = sorted(
        "/".join(str(p) for p in e.absolute_path) or "(the file)"
        for e in _validator(version).iter_errors(json.loads(json.dumps(document)))
    )
    if broken:
        raise ContractError(f"the proposal file breaks its schema (version {version}) at "
                            f"{len(broken)} place(s), first {', '.join(broken[:5])}; not written")


def to_yaml(document: Mapping[str, Any], estate: str, command: str = COMMAND) -> str:
    """The proposal file's text, written by ``command``, once it is checked against the schema
    of the version it states."""
    checked(document)
    return _to_yaml(document, estate, command=command)
