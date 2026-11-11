from __future__ import annotations

import json

from estate.induce import induce, label_of
from estate.metric_views import join_columns, parse_listing
from estate.reader_duckdb import DuckDBReader

from .conftest import OPTIONS, SENSITIVE, VIEWS


def _rel(model: dict, frm: str, to: str) -> dict | None:
    return next((r for r in model["relations"] if r["from"] == f"concept:{frm}" and r["to"] == f"concept:{to}"), None)


def test_metric_views_are_read_with_joins_dimensions_and_measures() -> None:
    (v,) = parse_listing(VIEWS, schema_prefix="acme_")
    assert v.name == "shop_claims" and v.source == "shop.claims"
    assert [(j.source, join_columns(j.on)) for j in v.joins] == [("shop.dealers", ("dealer_id", "dealer_id"))]
    assert v.dimensions[0].display_name == "Make" and v.dimensions[0].synonyms == ("brand",)
    assert [m.name for m in v.measures] == ["claim_count", "paid_amount"]
    assert join_columns("source.a = x.b AND source.c = x.d") is None  # only a single equality is read


def test_a_join_a_metric_view_states_is_declared_and_counted_on_the_data(reader: DuckDBReader) -> None:
    model = induce(reader)
    r = _rel(model, "shop.claims", "shop.dealers")
    assert r is not None and r["status"] == "declared" and r["columns"] == ["dealer_id"]
    assert r["signals"][0]["kind"] == "metric_view_join"
    assert r["support"] == {"holds": 2000, "breaks": 0, "empty": 0, "share": 1.0}


def test_a_mined_reference_is_proposed_with_its_violations_counted(reader: DuckDBReader) -> None:
    r = _rel(induce(reader), "shop.claims", "shop.parts")
    assert r is not None and r["status"] == "proposed"
    assert r["support"]["breaks"] == 60 and r["support"]["holds"] == 1940 and r["support"]["share"] == 0.97


def test_a_reference_that_holds_for_half_the_rows_is_dropped_and_listed(reader: DuckDBReader) -> None:
    model = induce(reader)
    assert _rel(model, "shop.visits", "shop.dealers") is None
    (d,) = [x for x in model["meta"]["dropped_below_floor"] if x["from"] == "shop.visits"]
    assert d["breaks"] == 250 and d["holds"] == 250


def test_concepts_carry_their_key_their_kind_and_their_signals(reader: DuckDBReader) -> None:
    by = {c["id"]: c for c in induce(reader)["concepts"]}
    dealers, claims = by["concept:shop.dealers"], by["concept:shop.claims"]
    assert dealers["key"] == ["dealer_id"] and dealers["kind"] == "entity" and dealers["definition"] == "One row per dealer."
    assert claims["kind"] == "fact" and claims["status"] == "declared"  # a metric view's source, with measures
    assert {s["kind"] for s in dealers["signals"]} >= {"unique_key", "referenced"}
    assert label_of("recall_campaigns") == "Recall campaign"


def test_a_sensitive_column_is_never_queried_and_its_concept_withholds_instances(toy_path, reader: DuckDBReader) -> None:
    asked: list[str] = []
    real = reader.query
    reader.query = lambda sql: (asked.append(sql), real(sql))[1]  # type: ignore[method-assign]
    model = induce(reader)
    assert asked and not any("contact_email" in sql for sql in asked)
    dealers = next(c for c in model["concepts"] if c["id"] == "concept:shop.dealers")
    assert dealers["instances"] == "withheld_personal"
    assert next(a for a in dealers["attributes"] if a["column"] == "contact_email")["sensitive"] is True


def test_shared_columns_become_value_sets_and_a_hierarchy_is_found(reader: DuckDBReader) -> None:
    model = induce(reader)
    by = {c["id"]: c for c in model["concepts"]}
    make = by["concept:value:make"]
    assert make["kind"] == "reference" and make["label"] == "Make" and make["synonyms"] == ["brand"] and make["values"] == 4
    assert "concept:value:market" in by
    h = next(r for r in model["relations"] if r["id"] == "relation:value:model->value:make")
    assert h["signals"][0]["kind"] == "functional_dependency" and h["support"]["objects_checked"] == h["support"]["objects_holding"] >= 2
    assert not any(r["id"] == "relation:value:make->value:model" for r in model["relations"])


def test_measures_are_bound_to_the_concept_of_their_view(reader: DuckDBReader) -> None:
    ms = induce(reader)["measures"]
    assert [(m["name"], m["concept"], m["status"]) for m in ms] == [
        ("claim_count", "concept:shop.claims", "declared"), ("paid_amount", "concept:shop.claims", "declared")]
    assert ms[0]["label"] == "Claims" and ms[1]["label"] == "Paid amount"


def test_a_verified_join_from_the_host_raises_a_mined_relation_to_declared(reader: DuckDBReader) -> None:
    model = induce(reader, verified_joins=[("shop.claims", "shop.parts"), ("shop.claims", "shop.nowhere")])
    r = _rel(model, "shop.claims", "shop.parts")
    assert r is not None and r["status"] == "declared" and r["signals"][-1]["kind"] == "verified_join"
    assert model["meta"]["verified_joins_without_columns"] == [{"from": "shop.claims", "to": "shop.nowhere"}]


def test_the_same_estate_gives_the_same_bytes(reader: DuckDBReader, toy_path) -> None:
    again = DuckDBReader(toy_path, schemas=["shop", "marts"], sensitive=SENSITIVE, metric_views=VIEWS,
                         schema_prefix="acme_", name="toy")
    a, b = induce(reader), induce(again)
    a["meta"].pop("queries"); b["meta"].pop("queries")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_a_key_is_not_a_line_number_and_not_a_column_named_after_another_object(reader: DuckDBReader) -> None:
    # Seen on the real mirror (2026-11-11): a source line number was taken as a key, and the
    # vehicle, whose own key is personal, was keyed by its owner.
    model = induce(reader, **OPTIONS)
    by = {c["id"]: c for c in model["concepts"]}
    assert by["concept:shop.vehicles"]["key"] == []  # its key (the VIN) is personal and is not looked at
    assert by["concept:shop.owners"]["key"] == ["owner_id"]
    assert _rel(model, "shop.vehicles", "shop.owners") is not None and _rel(model, "shop.owners", "shop.vehicles") is None
    unhelped = {c["id"]: c for c in induce(reader)["concepts"]}
    assert unhelped["concept:shop.vehicles"]["key"] == ["source_line"]  # what the ignore list prevents


def test_derived_objects_are_marked_and_nothing_is_mined_from_or_into_them(reader: DuckDBReader) -> None:
    model = induce(reader, **OPTIONS)
    agg = next(c for c in model["concepts"] if c["id"] == "concept:marts.claims_by_dealer")
    assert agg["kind"] == "derived"
    assert not any("marts.claims_by_dealer" in r["id"] for r in model["relations"])
    assert any("marts.claims_by_dealer" in r["id"] for r in induce(reader)["relations"])


def test_a_hierarchy_that_follows_from_two_others_is_left_out(reader: DuckDBReader) -> None:
    ids = {r["id"] for r in induce(reader, **OPTIONS)["relations"]}
    assert "relation:value:model->value:make" in ids and "relation:value:make->value:market" in ids
    assert "relation:value:model->value:market" not in ids
