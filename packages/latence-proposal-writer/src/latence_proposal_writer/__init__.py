"""The optional adapter for a host's semantic layer (ADR-0070): the proposal file written from
the approved decisions of an ontology build, and the host's receipt read back."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from latence_core.errors import ConfigError

from .receipt import RECEIPT, in_layer, read_receipts
from .writer import (
    CLASS_KINDS,
    COMMAND,
    SCHEMA_VERSION,
    SOURCE,
    STRUCTURE_KINDS,
    VERSIONS,
    kept_values,
    proposal,
    to_yaml,
)

__all__ = ["CLASS_KINDS", "COMMAND", "RECEIPT", "SCHEMA_VERSION", "SOURCE", "STRUCTURE_KINDS",
           "VERSIONS", "in_layer", "kept_values", "proposal", "read_receipts", "schema", "to_yaml"]

_SCHEMAS = {1: "proposal-file.schema.json", 2: "proposal-file-2.schema.json"}


def schema(version: int = 1) -> dict[str, Any]:
    """The JSON Schema of the file of a version, as a YAML reader reads it back (shipped with
    the package)."""
    if version not in _SCHEMAS:
        raise ConfigError(f"no schema of the proposal file in version {version}; "
                          f"there are {sorted(_SCHEMAS)}")
    held_at = files(__package__).joinpath("schema", _SCHEMAS[version])
    text = held_at.read_text(encoding="utf-8")
    held: dict[str, Any] = json.loads(text)
    return held
