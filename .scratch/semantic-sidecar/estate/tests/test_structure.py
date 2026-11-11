"""Structure the data proves: roll-ups between value sets, equivalent value sets, part-of from
keys, derived objects that describe an entity, and which value sits under which."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from estate.reader_duckdb import DuckDBReader
from estate.structure import broader_edges, build_structure, explain_review


@pytest.fixture()
def shop(tmp_path: Path) -> DuckDBReader:
    """60 towns in 6 regions in 2 countries; a region code that says the same as the region;
    11 brand names of which one is used for two brands; a colour that goes with any region.
    Brand names and colours cycle with 11 and 7, which share nothing with 60."""
    path = tmp_path / "shop.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA s")
    cols = ("'T' || (i % 60) AS town, 'R' || (i % 60 % 6) AS region, 'code-' || (i % 60 % 6) AS region_code, 'C' || (i % 60 % 6 % 2) AS country, "
            "'N' || (i % 11) AS brand_name, CASE WHEN i % 22 = 7 THEN 'B9' ELSE 'B' || (i % 11 % 4) END AS brand, 'colour' || (i % 7) AS colour")
    con.execute(f"CREATE TABLE s.orders AS SELECT 'O' || i AS order_id, {cols} FROM range(600) t(i)")
    con.execute(f"CREATE TABLE s.visits AS SELECT i AS visit_no, {cols} FROM range(900) t(i)")
    con.execute("CREATE TABLE s.lines AS SELECT 'O' || (i % 600) AS order_id, i % 3 AS line_no FROM range(1800) t(i)")
    con.close()
    return DuckDBReader(path, schemas=["s"], name="shop")


def _model() -> dict:
    def ref(col: str, values: int, objects: tuple[str, ...] = ("s.orders", "s.visits")) -> dict:
        return {"id": f"concept:value:{col}", "kind": "reference", "column": col, "label": col.replace("_", " ").title(), "values": values, "objects": list(objects)}

    concepts = [ref("town", 60), ref("region", 6), ref("region_code", 6, ("s.orders", "s.visits")), ref("country", 2), ref("brand_name", 11), ref("brand", 5), ref("colour", 7),
                {"id": "concept:s.orders", "kind": "entity", "label": "Order", "key": ["order_id"], "object_name": "s.orders"},
                {"id": "concept:s.lines", "kind": "fact", "label": "Line", "key": ["order_id", "line_no"], "object_name": "s.lines"},
                {"id": "concept:m.order_profile", "kind": "derived", "label": "Order profile", "key": ["order_id"], "object_name": "m.order_profile"}]
    relations = [{"id": "relation:s.lines.order_id->s.orders", "from": "concept:s.lines", "to": "concept:s.orders", "columns": ["order_id"], "to_columns": ["order_id"]}]
    relations += [{"id": f"relation:s.orders.{c}->value:{c}", "from": "concept:s.orders", "to": f"concept:value:{c}", "columns": [c], "to_columns": [c]}
                  for c in ("town", "region", "country", "colour")]
    return {"concepts": concepts, "relations": relations}


def test_roll_ups_are_proposed_with_their_support_and_what_follows_from_two_is_left_out(shop: DuckDBReader) -> None:
    st = build_structure(shop, _model())
    steps = {r["id"]: r for r in st["roll_ups"]}
    assert {"roll_up:value:town->value:region", "roll_up:value:region->value:country", "roll_up:value:brand_name->value:brand"} <= set(steps)
    assert "roll_up:value:town->value:country" not in steps  # it follows from town -> region -> country
    assert not any("colour" in k for k in steps)  # a colour goes with any region: no hierarchy
    clean, near = steps["roll_up:value:town->value:region"], steps["roll_up:value:brand_name->value:brand"]
    assert clean["support"]["values_breaking"] == 0 and "each town goes with exactly one region" in clean["signals"][0]["detail"]
    assert near["support"] == {"objects": ["s.orders", "s.visits"], "values_checked": 22, "values_breaking": 2, "share": 0.9091}
    assert "20 of 22 values of brand_name" in near["signals"][0]["detail"]
    assert all(r["status"] == "proposed" for r in st["roll_ups"])  # nothing here is accepted by the machine


def test_two_columns_that_cut_the_rows_the_same_way_are_one_level(shop: DuckDBReader) -> None:
    st = build_structure(shop, _model())
    assert [e["concepts"] for e in st["equivalent"]] == [["concept:value:region", "concept:value:region_code"]]
    assert not any("region_code" in r["id"] for r in st["roll_ups"])  # town -> region, not town -> region code as well


def test_master_data_hangs_below_the_finest_level_it_carries(shop: DuckDBReader) -> None:
    st = build_structure(shop, _model())
    below = [r for r in st["roll_ups"] if r["kind"] == "belongs_to"]
    assert [(r["from"], r["to"]) for r in below] == [("concept:s.orders", "concept:value:town")]  # not region or country: they follow from the town


def test_part_of_comes_from_keys_and_a_derived_object_describes_its_entity(shop: DuckDBReader) -> None:
    st = build_structure(shop, _model())
    assert [(p["from"], p["to"], p["kind"]) for p in st["part_of"]] == [("concept:s.lines", "concept:s.orders", "part_of")]
    assert "contains the key of Order (order_id)" in st["part_of"][0]["signals"][0]["detail"]
    assert [(d["from"], d["to"]) for d in st["describes"]] == [("concept:m.order_profile", "concept:s.orders")]


def test_every_value_gets_the_value_above_it_and_a_contested_one_says_so(shop: DuckDBReader) -> None:
    st = build_structure(shop, _model())
    known = {f"value:town:T{k}" for k in range(60)} | {f"value:region:R{k}" for k in range(6)} | {f"value:brand_name:N{k}" for k in range(11)} | {f"value:brand:B{k}" for k in (0, 1, 2, 3, 9)}
    edges = broader_edges(shop, st, known)
    up = {e["from"]: e for e in edges}
    assert up["value:town:T7"]["to"] == "value:region:R1" and "contested" not in up["value:town:T7"]
    assert up["value:brand_name:N7"]["to"] == "value:brand:B3" and up["value:brand_name:N7"]["contested"] is True  # mostly B3, sometimes B9
    assert sum(e["from"].startswith("value:town:") for e in edges) == 60
    assert all(e["from"] in known and e["to"] in known and e["kind"] == "broader" for e in edges)


def test_the_hierarchy_is_a_reason_for_or_against_two_names_being_the_same() -> None:
    nodes = [{"id": f"value:maker:{n}", "concept": "concept:value:maker", "label": n} for n in ("Kia America", "Kia Motors", "Toyota Corp", "Lexus Corp")]
    nodes += [{"id": f"value:make:{n}", "concept": "concept:value:make", "label": n} for n in ("KIA", "TOYOTA", "LEXUS")]
    edges = [{"from": "value:maker:Kia America", "to": "value:make:KIA", "kind": "broader"}, {"from": "value:maker:Kia Motors", "to": "value:make:KIA", "kind": "broader"},
             {"from": "value:maker:Toyota Corp", "to": "value:make:TOYOTA", "kind": "broader"}, {"from": "value:maker:Lexus Corp", "to": "value:make:LEXUS", "kind": "broader"}]
    review = [{"id": "1", "a": "value:maker:Kia America", "b": "value:maker:Kia Motors", "reason": "both names begin with “kia”", "strength": "weak"},
              {"id": "2", "a": "value:maker:Toyota Corp", "b": "value:maker:Lexus Corp", "reason": "both names contain “corp”", "strength": "strong"},
              {"id": "3", "a": "value:other:x", "b": "value:other:y", "reason": "r", "strength": "weak"}]
    out = explain_review(review, edges, nodes, [{"id": "concept:value:make", "label": "Make"}])
    assert out[0]["reason"].endswith("; both belong to the make KIA") and out[0]["strength"] == "strong"
    assert out[1]["reason"].endswith("; they belong to different makes (TOYOTA, LEXUS)") and out[1]["strength"] == "weak"
    assert out[2] == review[2] and review[0]["strength"] == "weak"  # no hierarchy above them: unchanged; the input is not altered
