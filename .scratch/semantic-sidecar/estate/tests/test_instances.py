from __future__ import annotations

from estate.induce import induce
from estate.instances import build_instances, possibly_same
from estate.reader_duckdb import DuckDBReader

from .conftest import OPTIONS

PERSONAL = ("shop.owners", "shop.vehicles")


def _build(reader: DuckDBReader) -> tuple[dict, dict]:
    model = induce(reader, **OPTIONS)
    return model, build_instances(reader, model, personal_objects=PERSONAL, resolve=("make",))


def test_values_and_master_rows_become_nodes_and_fact_rows_never_do(reader: DuckDBReader) -> None:
    _, inst = _build(reader)
    ids = {n["id"] for n in inst["nodes"]}
    assert {"value:make:ALFA", "value:market:north", "shop.dealers:D1", "shop.parts:P100"} <= ids
    assert not any(i.startswith(("shop.claims:", "shop.visits:")) for i in ids)  # a fact row is never a node
    assert inst["meta"]["nodes_per_concept"]["concept:shop.dealers"] == 40
    assert next(n for n in inst["nodes"] if n["id"] == "shop.dealers:D7")["label"] == "Dealer 7"


def test_personal_objects_give_no_nodes_and_no_personal_value_is_read(reader: DuckDBReader) -> None:
    model = induce(reader, **OPTIONS)  # induction may count a personal object's other columns; it reads no values
    asked: list[str] = []
    real = reader.query
    reader.query = lambda sql: (asked.append(sql), real(sql))[1]  # type: ignore[method-assign]
    inst = build_instances(reader, model, personal_objects=PERSONAL, resolve=("make",))
    assert not any(n["id"].startswith(("shop.owners:", "shop.vehicles:")) for n in inst["nodes"])
    assert asked and not any("contact_email" in sql or '"vin"' in sql or '"owners"' in sql or '"vehicles"' in sql for sql in asked)
    text = repr(inst)
    assert "@example.org" not in text and "Person 7" not in text


def test_a_fact_is_a_count_on_an_edge_with_what_was_counted(reader: DuckDBReader) -> None:
    _, inst = _build(reader)
    counts = [e for e in inst["edges"] if e["kind"] == "fact_count" and e["from"] == "shop.dealers:D1" and e["to"] == "concept:shop.claims"]
    assert [(e["rows"], e["counted"]) for e in counts] == [(50, {"object": "shop.claims", "group_by": ["dealer_id"]})]
    meets = [e for e in inst["edges"] if e["kind"] == "meet_in"]
    assert meets and all(e["via"] == "concept:shop.claims" and e["rows"] > 0 for e in meets)
    assert all(e["from"].startswith("shop.") and e["to"].startswith("shop.") for e in meets)
    refers = [e for e in inst["edges"] if e["kind"] == "refers_to" and e["from"] == "shop.dealers:D2"]
    assert {e["to"] for e in refers} == {"value:market:north"}


def test_every_edge_ends_at_something_that_exists(reader: DuckDBReader) -> None:
    model, inst = _build(reader)
    known = {n["id"] for n in inst["nodes"]} | {c["id"] for c in model["concepts"]}
    assert all(e["from"] in known and e["to"] in known for e in inst["edges"])
    assert inst["meta"]["certified"] is False


def test_names_that_may_be_the_same_are_listed_with_the_reason() -> None:
    names = [("Kia America, Inc.", 144), ("Kia Motors America", 12), ("Chrysler (FCA US, LLC)", 74), ("Chrysler (FCA US LLC)", 10),
             ("Toyota Motor Corporation", 99), ("Southeast Toyota Distributors, LLC", 6), ("Ford Motor Company", 129),
             ("Honda (American Honda Motor Co.)", 337)]
    found = {(p["a"], p["b"]): p for p in possibly_same(names)}
    assert found[("Kia America, Inc.", "Kia Motors America")]["strength"] == "strong"
    assert found[("Chrysler (FCA US LLC)", "Chrysler (FCA US, LLC)")]["strength"] == "strong"
    weak = found[("Southeast Toyota Distributors, LLC", "Toyota Motor Corporation")]
    assert weak["strength"] == "weak" and "toyota" in weak["reason"]
    assert not any("Ford Motor Company" in k for k in found)  # "motor" is in too many names to be a reason
