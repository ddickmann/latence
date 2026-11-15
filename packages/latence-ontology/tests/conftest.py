"""The fixtures shared by the package's test modules: the toy estate (a DuckDB file and its
reader), the toy estate as a build folder (``project``), the yard estate with a second source
(``yard``) and the shop estate of the structure tests (``shop``)."""

from __future__ import annotations

from _ontology_toy import reader, toy_path
from test_ontology_ledger_export import project
from test_ontology_proposal import yard
from test_ontology_structure import shop

__all__ = ["project", "reader", "shop", "toy_path", "yard"]
