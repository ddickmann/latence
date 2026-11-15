"""The ontology build over a data estate: it reads a catalog (a DuckDB mirror or Unity Catalog),
induces the schema level of the estate's ontology with the evidence for every element, keeps
assertions and a person's decisions in an append-only ledger, and writes the file a screen reads
(``graph-export/4``) with its Turtle, SHACL and PROV-O companions.

The command line is ``latence-ontology`` (``python -m latence_ontology``)."""

from __future__ import annotations

from pathlib import Path

SCHEMAS = Path(__file__).resolve().parent / "schemas"
"""The JSON Schema of the export (``graph-export-4.schema.json``) and the table of codes
(``codes.json``), shipped as package data."""

__all__ = ["SCHEMAS"]
