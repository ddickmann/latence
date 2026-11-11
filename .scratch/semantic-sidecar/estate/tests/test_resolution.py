"""The records of two sources handed to the resolution engine, and its answer in the export."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import jsonschema
import pytest

from estate.export import build_export
from estate.induce import induce
from estate.instances import build_instances
from estate.reader_duckdb import DuckDBReader
from estate.resolution import merge, write_records

from .conftest import OPTIONS

LINK = {"duplicate_free": ["dealer master"], "labels": {"fit": 10, "test": 20}, "records": [
    {"object": "shop.dealers", "id": "dealer_id", "name": "name", "source": "dealer master"},
    {"object": "shop.parts", "id": "part_id", "name": "model", "source": "second source"},
]}
SHOWN = ("shop.dealers",)


def test_the_records_file_names_each_record_by_its_node_id_and_reads_no_personal_column(reader: DuckDBReader, tmp_path: Path) -> None:
    truth = {f"P{k}": ("D1" if k <= 3 else f"other:{k}") for k in range(1, 101)}
    asked: list[str] = []
    real = reader.query
    reader.query = lambda sql: (asked.append(sql), real(sql))[1]  # type: ignore[method-assign]
    counts = write_records(reader, LINK["records"], truth, tmp_path / "records.parquet")
    assert counts == {"dealer master": 40, "second source": 100}
    rows = duckdb.connect().execute(f"SELECT unique_id, source_system, name, cluster FROM read_parquet('{tmp_path / 'records.parquet'}') ORDER BY 1").fetchall()
    by_id = {r[0]: r for r in rows}
    assert by_id["shop.dealers:D1"][1:] == ("dealer master", "Dealer 1", "D1")  # the master's truth is its own key
    assert by_id["shop.parts:P1"][3] == "D1" and by_id["shop.parts:P6"][3] == "other:6"
    assert not any("contact_email" in sql for sql in asked)
    with pytest.raises(ValueError, match="personal"):
        write_records(reader, [{**LINK["records"][0], "name": "contact_email"}], truth, tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="no truth"):
        write_records(reader, LINK["records"], {}, tmp_path / "y.parquet")


def _answer() -> dict:
    ev = [{"kind": "name", "detail": "the same name apart from the legal form"}, {"kind": "town", "detail": "the same town"},
          {"kind": "source", "detail": "dealer master and second source"}]
    return {
        "format": "resolution/1", "records": 140, "sources": {"dealer master": 40, "second source": 100}, "candidate_pairs": 900,
        "labels": {"training": 10, "accept_test": 14, "reject_test": 6, "total": 30, "from": "the truth column of the input"},
        "lines": {"accept_at": 0.6, "reject_below": 0.2, "target": 0.95, "delta": 0.05}, "certified": True,
        "duplicate_free": {"sources": ["dealer master"], "pairs_ruled_out": 3}, "counts": {"accept": 3, "review": 2, "reject": 895},
        "clusters": {"shop.dealers:D1": "shop.dealers:D1", "shop.parts:P1": "shop.dealers:D1", "shop.parts:P2": "shop.dealers:D1",
                     "shop.dealers:D2": "shop.dealers:D2", "shop.parts:P3": "shop.dealers:D2", "shop.parts:P4": "shop.parts:P4",
                     "shop.dealers:D3": "shop.dealers:D3", "shop.dealers:D4": "shop.dealers:D3", "shop.parts:P5": "shop.dealers:D3"},
        "accepted": [{"a": "shop.dealers:D1", "b": "shop.parts:P1", "p": 0.97, "evidence": ev},
                     {"a": "shop.dealers:D1", "b": "shop.parts:P2", "p": 0.91, "evidence": ev},
                     {"a": "shop.dealers:D2", "b": "shop.parts:P3", "p": 0.88, "evidence": ev}],
        "review": [{"a": "shop.parts:P4", "b": "shop.dealers:D2", "p": 0.55, "evidence": ev},
                   {"a": "shop.dealers:D9", "b": "shop.parts:P4", "p": 0.31, "evidence": ev}],
        "review_total": 2,
    }


def _instances(reader: DuckDBReader) -> tuple[dict, dict]:
    model = induce(reader, **OPTIONS, without_personal_columns=SHOWN)
    inst = build_instances(reader, model, personal_objects=("shop.owners", "shop.vehicles"), without_personal_columns=SHOWN,
                           also={"shop.parts": "model"})
    return model, inst


def test_the_engines_answer_becomes_links_review_items_and_a_summary(reader: DuckDBReader) -> None:
    model, inst = _instances(reader)
    before = json.dumps(inst, sort_keys=True)
    truth = {"P1": "D1", "P2": "D7", "P3": "D2", "P4": "D2", "P5": "D3"}
    concept_of = {c["object_name"]: c["id"] for c in model["concepts"] if c.get("object_name")}
    out = merge(inst, _answer(), LINK, truth, concept_of, seeded=True)
    assert json.dumps(inst, sort_keys=True) == before  # what was passed in is not changed
    links = [e for e in out["edges"] if e["kind"] == "same_as"]
    assert [(e["from"], e["to"], e["p"]) for e in links] == [("shop.parts:P1", "shop.dealers:D1", 0.97), ("shop.parts:P2", "shop.dealers:D1", 0.91),
                                                           ("shop.parts:P3", "shop.dealers:D2", 0.88)]
    assert links[0]["evidence"][0] == {"kind": "name", "detail": "the same name apart from the legal form"}
    items = [r for r in out["review"] if r["id"].startswith("same:shop.")]
    assert [(r["a"], r["b"], r["strength"]) for r in items] == [("shop.dealers:D2", "shop.parts:P4", "strong"), ("shop.dealers:D9", "shop.parts:P4", "weak")]
    assert items[0]["reason"] == "the same name apart from the legal form; the same town" and items[0]["concept"] == "concept:shop.dealers"
    r = out["meta"]["resolution"]
    assert r["seeded"] is True and r["certified"] is True and r["linked"] == 3 and r["clusters_with_two_master_records"] == 1  # D3 and D4 in one cluster: no link is drawn
    assert r["checked_against_seed"] == {"linked": 3, "linked_right": 2, "belong_to_a_master_record": 5, "share_right": 0.6667, "share_found": 0.4}
    assert r["labels"]["total"] == 30 and "standing in for a reviewer" in r["labels_from"]
    assert out["meta"]["edges"] == len(out["edges"]) and out["meta"]["review_items"] == len(out["review"])


def test_the_export_with_links_matches_its_schema(reader: DuckDBReader) -> None:
    model, inst = _instances(reader)
    concept_of = {c["object_name"]: c["id"] for c in model["concepts"] if c.get("object_name")}
    next(c for c in model["concepts"] if c["id"] == "concept:shop.parts")["seeded"] = True
    export = build_export(model, source="toy", title="Toy", instances=merge(inst, _answer(), LINK, {"P1": "D1"}, concept_of, seeded=True))
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "graph-export-2.schema.json").read_text())
    jsonschema.validate(export, schema)
    assert export["meta"]["resolution"]["linked"] == 3
    known = {n["id"] for n in export["nodes"]} | {c["id"] for c in export["concepts"]}
    assert all(e["from"] in known and e["to"] in known for e in export["edges"])
    assert all(r["a"] in known and r["b"] in known for r in export["review"])
    assert build_export(model, source="toy", title="Toy", instances=inst)["meta"]["resolution"] is None
