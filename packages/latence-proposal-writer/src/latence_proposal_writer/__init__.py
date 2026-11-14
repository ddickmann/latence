"""The optional adapter for a host's semantic layer (ADR-0070): the proposal file written from
the approved decisions of an ontology build, and the host's receipt read back."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .receipt import RECEIPT, in_layer, read_receipts
from .writer import CLASS_KINDS, COMMAND, SOURCE, STRUCTURE_KINDS, kept_values, proposal, to_yaml

__all__ = ["CLASS_KINDS", "COMMAND", "RECEIPT", "SOURCE", "STRUCTURE_KINDS", "in_layer",
           "kept_values", "proposal", "read_receipts", "schema", "to_yaml"]


def schema() -> dict[str, Any]:
    """The JSON Schema of the file as a YAML reader reads it back (shipped with the package)."""
    held_at = files(__package__).joinpath("schema", "proposal-file.schema.json")
    text = held_at.read_text(encoding="utf-8")
    held: dict[str, Any] = json.loads(text)
    return held
