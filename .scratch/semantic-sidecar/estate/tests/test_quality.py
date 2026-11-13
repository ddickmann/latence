"""Defects the vehicle estate showed in what the engine proposes (ticket 27), each on a small
planted estate."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from estate.induce import induce, label_of, singular
from estate.instances import build_instances
from estate.reader_duckdb import DuckDBReader
from estate.structure import build_structure, entity_names


@pytest.mark.parametrize(("plural", "one"), [
    ("cases", "case"), ("addresses", "address"), ("statuses", "status"), ("categories", "category"), ("boxes", "box"),
    ("batches", "batch"), ("dishes", "dish"), ("quizzes", "quiz"), ("dealers", "dealer"), ("houses", "house"),
    ("phases", "phase"), ("analyses", "analysis"), ("series", "series"), ("class", "class"), ("status", "status"), ("data", "data"),
])
def test_one_of_a_plural(plural: str, one: str) -> None:
    assert singular(plural) == one


def test_a_label_the_estate_states_wins_over_the_rule() -> None:
    assert label_of("investigation_cases") == "Investigation case"
    assert label_of("mfr_communications", {"mfr_communications": "Manufacturer communication"}) == "Manufacturer communication"


@pytest.fixture()
def recalls(tmp_path: Path) -> DuckDBReader:
    """20 campaigns that each cover two models (so a campaign row is keyed on both), 30 dealers,
    and what each dealer completed per campaign. Cases with a title of their own, reported on
    over ten periods and noted on five times; a maker per case."""
    path = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE SCHEMA old")
    con.execute("CREATE TABLE s.campaigns AS SELECT 'K' || (i % 20) AS campaign_id, 'M' || (i // 20) AS model, 'fix ' || (i % 20) AS remedy FROM range(40) t(i)")
    con.execute("CREATE TABLE s.dealers AS SELECT 'D' || i AS dealer_id, 'Dealer ' || i AS name FROM range(30) t(i)")
    con.execute("CREATE TABLE s.completions AS SELECT 'K' || (i % 20) AS campaign_id, 'D' || (i // 20) AS dealer_id, i % 7 AS done FROM range(600) t(i)")
    case = "'A' || c AS case_id, 'Title of case ' || c AS title, 'maker' || (c % 5) AS maker"
    con.execute(f"CREATE TABLE s.cases AS SELECT {case}, DATE '2024-01-01' + CAST(c AS INTEGER) AS opened FROM range(30) t(c)")
    con.execute(f"CREATE TABLE s.case_reports AS SELECT {case}, p AS period FROM range(30) t(c), range(10) u(p)")
    con.execute(f"CREATE TABLE s.case_notes AS SELECT {case}, n AS note_no FROM range(30) t(c), range(5) u(n)")
    con.execute("CREATE TABLE s.makers AS SELECT 'maker' || i AS maker, i AS founded FROM range(5) t(i)")
    con.execute("CREATE TABLE old.dealers AS SELECT 'D' || i AS dealer_id, 'Dealer ' || i AS name FROM range(30) t(i)")
    con.close()
    return DuckDBReader(path, schemas=["s", "old"], name="recalls")


def test_a_reference_to_one_column_of_a_composite_key_is_a_relation_and_no_part_of(recalls: DuckDBReader) -> None:
    model = induce(recalls)
    campaigns = next(c for c in model["concepts"] if c["id"] == "concept:s.campaigns")
    assert campaigns["key"] == ["campaign_id", "model"]
    partial = next(r for r in model["relations"] if r["id"] == "relation:s.completions.campaign_id->s.campaigns")
    assert partial["to_columns"] == ["campaign_id"] and partial["partial_key"] is True and partial["cardinality"] == "many_to_many"
    assert partial["support"]["share"] == 1.0 and partial["signals"][0]["code"] == "name_and_inclusion.part_of_key"
    st = build_structure(recalls, model)
    of_completion = {p["to"]: p["kind"] for p in st["part_of"] if p["from"] == "concept:s.completions"}
    # It links a campaign and a dealer (the dealer table of either layer); it is a part of neither.
    assert of_completion == {"concept:s.campaigns": "links", "concept:s.dealers": "links", "concept:old.dealers": "links"}
    assert next(p for p in st["part_of"] if p["to"] == "concept:s.campaigns")["signals"][0]["code"] == "key_contains_part_of_key"


def test_a_title_that_goes_one_to_one_with_an_entitys_key_is_its_name_and_not_a_level(recalls: DuckDBReader) -> None:
    model = induce(recalls)
    assert "concept:value:title" in {c["id"] for c in model["concepts"]}  # the column is shared by three objects and repeats: a value set to the induction
    names = entity_names(recalls, model)
    assert names == {"title": [{"entity": "concept:s.cases", "object": "s.cases", "keys": 30, "values": 30}]}
    st = build_structure(recalls, model)
    assert not any("title" in r["id"] for r in st["roll_ups"])  # neither title -> maker nor case -> title
    said = next(n for n in st["not_proposed"] if n["id"] == "not_proposed:value:title")
    assert said["signals"][0]["code"] == "names_an_entity" and said["signals"][0]["params"] == {"value_set": "Title", "entity": "Case", "keys": 30, "values": 30}


def test_the_estate_can_say_which_columns_label_a_row(recalls: DuckDBReader) -> None:
    model = induce(recalls)
    inst = build_instances(recalls, model, label_columns={"s.cases": ["case_id", "title"]})
    assert next(n for n in inst["nodes"] if n["id"] == "s.cases:A7")["label"] == "A7 · Title of case 7"
    with pytest.raises(ValueError, match="s.cases.nope"):
        build_instances(recalls, model, label_columns={"s.cases": ["nope"]})


def test_two_concepts_never_share_a_label(recalls: DuckDBReader) -> None:
    model = induce(recalls)
    labels = [c["label"] for c in model["concepts"]]
    assert len(labels) == len(set(labels))
    assert {c["id"]: c["label"] for c in model["concepts"] if c["id"].endswith(".dealers")} == {"concept:old.dealers": "Dealer (old)", "concept:s.dealers": "Dealer (s)"}
    # A column named after a table is that table's key in every layer that holds one of the name.
    assert {c["id"]: c["key"] for c in model["concepts"] if c["id"].endswith(".dealers")} == {"concept:old.dealers": ["dealer_id"], "concept:s.dealers": ["dealer_id"]}
