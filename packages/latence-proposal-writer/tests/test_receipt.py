"""The host's receipt read back: an element says since which release of the host's library it
is in the layer only where a receipt says it was placed, on the statement it still makes, and
while a person's approval of it is in force."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _proposal_estate import export, pair
from latence_core.errors import ConfigError
from latence_proposal_writer import in_layer, read_receipts

SRC = Path(__file__).resolve().parents[1] / "src" / "latence_proposal_writer"


def _receipt(path: Path, version: str, placed: list[tuple[dict[str, Any], str | None]],
             already: tuple[dict[str, Any], ...] = (), refused: tuple[dict[str, Any], ...] = (),
             fmt: str = "layer-receipt/1", library: str | None = "yard-semantics") -> Path:
    def taken(el: dict[str, Any], stated: str | None = None) -> dict[str, Any]:
        return {"entry": "proposal.yaml#1", "kind": "spellings", "element": el["id"],
                "claim": stated or el["claim"], "dimension": "maker",
                "node": "semantics.dimensions.maker.values"}

    path.write_text(json.dumps({
        "format": fmt, **({"library": library} if library else {}), "version": version,
        "placed": [taken(el, c) for el, c in placed], "already_there": [taken(el) for el in already],
        "refused": [{**taken(el), "reason": "no dimension reads it"} for el in refused]}), encoding="utf-8")
    return path


def _approved(doc: dict[str, Any], *ids: str) -> dict[str, Any]:
    """The export as a build writes it once a person approved these elements."""
    def mark(el: dict[str, Any]) -> dict[str, Any]:
        return {**el, "decision": {"verdict": "approved", "by": "reviewer-1"}} if el["id"] in ids else el

    return {**doc, "concepts": [mark(c) for c in doc["concepts"]], "review": [mark(r) for r in doc["review"]]}


def _taken(doc: dict[str, Any]) -> dict[str, Any]:
    return {el["id"]: el["in_layer"] for el in doc["concepts"] + doc["review"] if "in_layer" in el}


def test_the_export_says_since_which_release_an_element_is_in_the_layer_only_where_a_receipt_says_it_was_placed(tmp_path: Path) -> None:
    base = export()
    alfa, beta, gama = pair(base, "Alfa"), pair(base, "Beta"), pair(base, "Gama")
    dealers = next(c for c in base["concepts"] if c["id"] == "concept:shop.dealers")
    doc = _approved(base, alfa["id"], beta["id"], gama["id"], dealers["id"])
    assert in_layer(doc, read_receipts(())) == (doc, 0)  # no receipt: nothing is said, never guessed
    # Placed: the Alfa pair, and the dealers on a statement the list no longer makes. Already
    # there: the Gama pair. Refused: the Beta pair.
    first = _receipt(tmp_path / "r1.json", "1.1.0", [(alfa, None), (dealers, "0" * 16)], already=(gama,), refused=(beta,))
    marked, count = in_layer(doc, read_receipts((first,)))
    assert _taken(marked) == {alfa["id"]: {"library": "yard-semantics", "version": "1.1.0"}} and count == 1
    # A later release placed the dealers and holds the Alfa pair still: the pair is in the
    # layer since the first.
    second = _receipt(tmp_path / "r2.json", "1.2.0", [(dealers, None), (alfa, None)])
    marked, count = in_layer(doc, read_receipts((first, second)))
    assert _taken(marked) == {alfa["id"]: {"library": "yard-semantics", "version": "1.1.0"},
                              "concept:shop.dealers": {"library": "yard-semantics", "version": "1.2.0"}}
    # The approval is withdrawn: the element no longer says it is in the layer.
    withdrawn = _approved(base, beta["id"], gama["id"], dealers["id"])
    marked, count = in_layer(withdrawn, read_receipts((first, second)))
    assert _taken(marked) == {"concept:shop.dealers": {"library": "yard-semantics", "version": "1.2.0"}} and count == 1


def test_a_receipt_of_another_format_or_without_its_library_is_refused(tmp_path: Path) -> None:
    dealers = export()["concepts"][0]
    with pytest.raises(ConfigError, match="layer-receipt/1"):
        read_receipts((_receipt(tmp_path / "r3.json", "1.3.0", [(dealers, None)], fmt="layer-receipt/2"),))
    with pytest.raises(ConfigError, match="names the library and the release"):
        read_receipts((_receipt(tmp_path / "r4.json", "1.3.0", [(dealers, None)], library=None),))


def test_the_writer_imports_nothing_but_the_standard_library_and_the_core() -> None:
    """The seam to a host is a file (ADR-0070): no code of a host, and none of the estate
    prototype, is imported by the package that writes it."""
    allowed = set(sys.stdlib_module_names) | {"latence_core", "latence_proposal_writer"}
    seen: set[str] = set()
    for path in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                seen |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                seen.add(node.module.split(".")[0])
    assert seen and seen <= allowed, sorted(seen - allowed)
