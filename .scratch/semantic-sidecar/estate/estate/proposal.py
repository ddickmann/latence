"""The proposal file for a host's semantic layer: a thin call into the optional adapter package
``latence-proposal-writer`` (ADR-0070, ticket 43), which writes the file from a build's export
and the decisions in force, and reads the host's receipt back. Nothing of the writer lives here
any more; this module only names this prototype's command in the file's opening comment."""

from __future__ import annotations

from typing import Any

from latence_proposal_writer import in_layer, proposal, read_receipts
from latence_proposal_writer import to_yaml as _to_yaml

COMMAND = "python -m estate propose"

__all__ = ["COMMAND", "in_layer", "proposal", "read_receipts", "to_yaml"]


def to_yaml(document: dict[str, Any], estate: str) -> str:
    """The proposal file's text, written by this prototype's command."""
    return _to_yaml(document, estate, command=COMMAND)
