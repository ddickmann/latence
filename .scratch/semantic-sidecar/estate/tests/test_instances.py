from __future__ import annotations

from estate.induce import induce
from estate.instances import build_instances, possibly_same
from estate.reader_duckdb import DuckDBReader

from .conftest import OPTIONS

PERSONAL = ("shop.owners", "shop.vehicles")


SHOWN = ("shop.dealers",)  # an organisation that also carries a contact person's columns


def _build(reader: DuckDBReader) -> tuple[dict, dict]:
    model = induce(reader, **OPTIONS, without_personal_columns=SHOWN)
    return model, build_instances(reader, model, personal_objects=PERSONAL, resolve=("make",), without_personal_columns=SHOWN)


def test_an_object_with_a_personal_column_gives_no_nodes_unless_the_estate_says_so(reader: DuckDBReader) -> None:
    model = induce(reader, **OPTIONS)
    inst = build_instances(reader, model, personal_objects=PERSONAL, resolve=("make",))
    assert not any(n["id"].startswith("shop.dealers:") for n in inst["nodes"])
    assert not any(e["from"].startswith("shop.dealers:") or e["to"].startswith("shop.dealers:") for e in inst["edges"])
    assert next(c for c in model["concepts"] if c["id"] == "concept:shop.dealers")["instances"] == "withheld_personal"
    assert any(n["id"].startswith("shop.parts:") for n in inst["nodes"])  # an object without personal columns is unaffected

    model, inst = _build(reader)
    assert sum(n["id"].startswith("shop.dealers:") for n in inst["nodes"]) == 40
    assert next(c for c in model["concepts"] if c["id"] == "concept:shop.dealers")["instances"] == "without_personal_columns"
    assert "@example.org" not in repr(inst)


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
    inst = build_instances(reader, model, personal_objects=PERSONAL, resolve=("make",), without_personal_columns=SHOWN)
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
    assert weak["strength"] == "weak" and weak["reasons"] == [{"kind": "shared_rare_word", "code": "shared_rare_word", "params": {"word": "toyota"}, "ref": ""}]
    assert not any("Ford Motor Company" in k for k in found)  # "motor" is in too many names to be a reason


def test_two_names_a_reviewer_calls_the_same_become_one_value_with_the_other_as_its_alias() -> None:
    from estate.instances import merge_values

    inst = {
        "nodes": [{"id": "value:maker:Kia America, Inc.", "concept": "concept:value:maker", "label": "Kia America, Inc.", "rows": 144},
                  {"id": "value:maker:Kia Motors America", "concept": "concept:value:maker", "label": "Kia Motors America", "rows": 12},
                  {"id": "value:make:KIA", "concept": "concept:value:make", "label": "KIA", "rows": 300},
                  {"id": "s.campaigns:C1", "concept": "concept:s.campaigns", "label": "C1", "key": "C1"},
                  {"id": "s.campaigns:C2", "concept": "concept:s.campaigns", "label": "C2", "key": "C2"}],
        "edges": [{"from": "value:maker:Kia America, Inc.", "to": "value:make:KIA", "kind": "broader", "relation": "roll_up:value:maker->value:make", "rows": 140},
                  {"from": "value:maker:Kia Motors America", "to": "value:make:KIA", "kind": "broader", "relation": "roll_up:value:maker->value:make", "rows": 12},
                  {"from": "s.campaigns:C1", "to": "value:maker:Kia America, Inc.", "kind": "refers_to", "relation": "r"},
                  {"from": "s.campaigns:C2", "to": "value:maker:Kia Motors America", "kind": "refers_to", "relation": "r"}],
        "review": [{"id": "same:maker:Kia America, Inc.~Kia Motors America", "kind": "possibly_same", "a": "value:maker:Kia America, Inc.", "b": "value:maker:Kia Motors America"}],
        "meta": {"nodes": 5, "edges": 4, "nodes_per_concept": {"concept:value:maker": 2, "concept:value:make": 1, "concept:s.campaigns": 2}},
    }
    before = repr(inst)
    out = merge_values(inst, [("value:maker:Kia Motors America", "value:maker:Kia America, Inc.")])
    assert repr(inst) == before  # what was passed in is not changed
    kept = next(n for n in out["nodes"] if n["id"] == "value:maker:Kia America, Inc.")
    assert kept["rows"] == 156 and kept["aliases"] == ["Kia Motors America"]  # the name with more rows stays
    assert not any(n["id"] == "value:maker:Kia Motors America" for n in out["nodes"])
    assert [(e["from"], e["to"], e.get("rows")) for e in out["edges"] if e["kind"] == "broader"] == [("value:maker:Kia America, Inc.", "value:make:KIA", 152)]
    assert {(e["from"], e["to"]) for e in out["edges"] if e["kind"] == "refers_to"} == {("s.campaigns:C1", "value:maker:Kia America, Inc."), ("s.campaigns:C2", "value:maker:Kia America, Inc.")}
    assert out["meta"]["nodes"] == 4 and out["meta"]["edges"] == 3 and out["meta"]["nodes_per_concept"]["concept:value:maker"] == 1
    assert out["meta"]["merged_values"] == [{"kept": "value:maker:Kia America, Inc.", "alias": "value:maker:Kia Motors America"}]
    assert out["review"][0]["merged_into"] == "value:maker:Kia America, Inc."  # the item stays, and says what became of it
