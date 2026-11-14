"""Mentions from the estate's documents, linked to its rows (ticket 42)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

from estate.cli import build

from .test_ledger_export import SCHEMA, _export, project  # noqa: F401  (the fixture)

LETTER = ("Dear Dealer 7,\n\nthe claim on the ALFA M3 was paid. Dealer 10 asked about it too; "
          "dealer 1 did not.\nPerson 7 called twice.\n")


def test_a_document_names_rows_by_their_names_and_each_mention_is_a_span_linked_to_one_row(project: dict[str, Path], tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    (docs / "letters").mkdir(parents=True)
    (docs / "letters" / "to-the-dealers.md").write_text(LETTER, encoding="utf-8")
    (docs / "ignored.pdf").write_bytes(b"%PDF-1.4 Dealer 3")
    cfg = yaml.safe_load(project["config"].read_text(encoding="utf-8"))  # the dealers' rows are shown, their e-mail column is not
    project["config"].write_text(yaml.safe_dump({**cfg, "without_personal_columns": ["shop.dealers"]}), encoding="utf-8")
    summary = build(**project, documents=docs)
    export = _export(project)
    jsonschema.validate(export, SCHEMA)
    found = {(m["text"], m.get("node")) for m in export["mentions"]}
    assert ("Dealer 7", "shop.dealers:D7") in found and ("Dealer 10", "shop.dealers:D10") in found
    assert ("dealer 1", "shop.dealers:D1") in found  # case folded, and never inside "Dealer 10"
    assert ("ALFA", "value:make:ALFA") in found
    assert not any("Person" in m["text"] for m in export["mentions"])  # rows of an object about persons are never held, never found
    for m in export["mentions"]:
        assert LETTER[m["start"]:m["end"]] == m["text"] and m["document"] == "letters/to-the-dealers.md"
        assert m["digest"].startswith("sha256:")
    assert summary["mentions"] == export["meta"]["mentions"] == {"documents": 1, "mentions": len(export["mentions"]),
                                                                 "linked": len(export["mentions"]), "ambiguous": 0}
    assert LETTER not in json.dumps(export)  # spans, not the documents


def test_a_name_two_rows_carry_is_left_for_a_person(project: dict[str, Path], tmp_path: Path) -> None:
    from estate.mentions import find_mentions

    export = {"nodes": [{"id": "s.a:1", "concept": "concept:s.a", "label": "Acme"}, {"id": "s.b:9", "concept": "concept:s.b", "label": "ACME"},
                        {"id": "s.a:2", "concept": "concept:s.a", "label": "Acme North", "aliases": ["Acme N."]}]}
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "note.txt").write_text("Acme North and Acme N. and acme.", encoding="utf-8")
    mentions, counts = find_mentions(export, docs)
    assert [(m["text"], m.get("node"), m.get("candidates")) for m in mentions] == [
        ("Acme North", "s.a:2", None), ("Acme N.", "s.a:2", None), ("acme", None, ["s.a:1", "s.b:9"])]
    assert counts == {"documents": 1, "mentions": 3, "linked": 2, "ambiguous": 1}


def test_without_documents_the_export_says_nothing_of_them(project: dict[str, Path]) -> None:
    build(**project)
    export = _export(project)
    assert "mentions" not in export and "mentions" not in export["meta"]
